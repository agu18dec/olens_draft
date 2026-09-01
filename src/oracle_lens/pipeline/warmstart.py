"""WV warm start: continuation-text SFT with embedding-injected activations.

Round-0 established (2026-07-20) that the raw base model cannot read L44 activations at any
injection scale (Paris control 8/8 = plumbing proven; real-vs-shuffled gap null at alpha up to
200k). This is NLA's own finding — their recipe warm-starts the AV with SFT before RL. Ours is
the fully-unsupervised analogue of their summary-SFT stage:

    input : the WV prompt with ``alpha * h/||h||`` REPLACING the <concept> slot embedding
    target: ``<explanation>\\n`` + the TRUE next tokens after h + ``\\n</explanation>``

CE is masked to the target tokens; the only way down is through the injected vector — this
forges the reading pathway. Span ids come straight from the shards (never re-tokenized); the
tag wrapper is id-concatenated around them (the joint decodes to the same string; the reward
path strips tags and bare-tokenizes anyway).

Alpha is selected by NLA's SFT proxy folded into the warm start itself: mini runs at a few
alphas, each against a shuffled-pairing control; pick max CE(shuffled) - CE(real).
mytorch-lightning harness (same rationale as train_recon: reuse the debugged loop).
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import train as mtl_train
from mytorch_lightning.mydule import Mydule
from torch import Tensor
from torch.utils.data import Dataset

from oracle_lens.pipeline.shards import LongSpanPairs
from oracle_lens.pipeline.verbalizer import WVPrompt


@dataclass
class WarmstartConfig:
    """One warm-start run. ``run_name`` is the descriptive wandb name (never coded)."""

    run_name: str
    layer: int = 44
    alpha: float = 8000.0
    n_examples: int = 50_000
    min_len: int = 5
    max_len: int = 64
    shuffled_pairing: bool = False  # control arm: activations permuted against continuations
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    lr_sched: str = "constant"
    micro_batch: int = 8
    grad_accum: int = 4
    epochs: int = 1
    eval_every_steps: int = 100
    max_eval_rows: int = 1024
    warmup_steps: int = 20
    seed: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_warmstart_examples(
    pairs: LongSpanPairs,
    tokenizer: Any,
    cfg: WarmstartConfig,
) -> list[dict[str, Tensor]]:
    """(activation, target_ids) examples; target = tag-wrapped ORIGINAL span ids + EOS."""
    open_ids = tokenizer("<explanation>\n", add_special_tokens=False)["input_ids"]
    close_ids = tokenizer("\n</explanation>", add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    lens = pairs.lengths
    keep = torch.nonzero((lens >= cfg.min_len) & (lens <= cfg.max_len)).squeeze(-1)
    gen = torch.Generator().manual_seed(cfg.seed)
    order = keep[torch.randperm(len(keep), generator=gen)][: cfg.n_examples]
    examples = []
    for i in order.tolist():
        span = pairs.row_ids(i).tolist()
        target = [*open_ids, *span, *close_ids, eos]
        examples.append(
            {
                "activation": pairs.targets[i].float(),
                "target_ids": torch.tensor(target, dtype=torch.long),
            }
        )
    if cfg.shuffled_pairing:
        # control: same marginals, pairing destroyed — CE(shuffled) - CE(real) is the
        # information-transfer signal (NLA's alpha-selection proxy)
        acts = [e["activation"] for e in examples]
        perm = torch.randperm(len(acts), generator=torch.Generator().manual_seed(cfg.seed + 1))
        for e, j in zip(examples, perm.tolist(), strict=True):
            e["activation"] = acts[j]
    return examples


class WarmstartDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, examples: list[dict[str, Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return self.examples[i]


def warmstart_collate(
    rows: list[dict[str, Tensor]], *, prompt_len: int, pad_id: int
) -> dict[str, Tensor]:
    """Right-pad targets; labels = -100 on prompt+pad (CE only on target tokens)."""
    tmax = max(int(r["target_ids"].shape[0]) for r in rows)
    b = len(rows)
    target_ids = torch.full((b, tmax), pad_id, dtype=torch.long)
    labels = torch.full((b, prompt_len + tmax), -100, dtype=torch.long)
    attn = torch.zeros(b, prompt_len + tmax, dtype=torch.long)
    attn[:, :prompt_len] = 1
    for row, r in enumerate(rows):
        n = int(r["target_ids"].shape[0])
        target_ids[row, :n] = r["target_ids"]
        labels[row, prompt_len : prompt_len + n] = r["target_ids"]
        attn[row, prompt_len : prompt_len + n] = 1
    return {
        "activation": torch.stack([r["activation"] for r in rows]),
        "target_ids": target_ids,
        "labels": labels,
        "attention_mask": attn,
    }


class WarmstartMydule(Mydule):  # type: ignore[misc]
    """Masked-CE SFT with the activation replacing the <concept> slot embedding."""

    def __init__(
        self,
        model: torch.nn.Module,
        prompt: WVPrompt,
        cfg: WarmstartConfig,
        train_data: WarmstartDataset,
        val_data: WarmstartDataset,
        pad_id: int,
    ) -> None:
        super().__init__()
        self._model = model
        self._prompt = prompt
        self._cfg = cfg
        self._train_data = train_data
        self._val_data = val_data
        self._pad_id = pad_id
        self._prompt_ids = torch.tensor(prompt.input_ids, dtype=torch.long)
        self._val_losses: list[Tensor] = []
        self._n_val_batches = max(1, len(val_data) // cfg.micro_batch)
        self.last_val_metrics: dict[str, float] = {}

    def create_model(self) -> torch.nn.Module:
        return self._model

    def configure_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self._model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self._cfg.lr)

    def train_data(self) -> WarmstartDataset:
        return self._train_data

    def val_data(self) -> WarmstartDataset:
        return self._val_data

    def configure_training_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt_len = len(self._prompt.input_ids)
        pad_id = self._pad_id
        args["collate_fn"] = lambda rows: warmstart_collate(
            rows, prompt_len=prompt_len, pad_id=pad_id
        )
        args["num_workers"] = 0
        args["persistent_workers"] = False
        return args

    def configure_validation_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.configure_training_dl(args)

    def _ce(self, batch: dict[str, Tensor]) -> Tensor:
        device = batch["activation"].device
        b = batch["activation"].shape[0]
        embed: Any = self._model.get_input_embeddings()  # type: ignore[operator]
        prompt_ids = self._prompt_ids.to(device).unsqueeze(0).expand(b, -1)
        # clone: input-require-grads marks embed output a leaf; in-place writes forbidden
        prompt_embeds = embed(prompt_ids).clone()
        unit = batch["activation"] / batch["activation"].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        prompt_embeds[:, self._prompt.slot, :] = (self._cfg.alpha * unit).to(prompt_embeds.dtype)
        target_embeds = embed(batch["target_ids"])
        inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
        # DDP-wrapped forward when the harness set one (silent no-sync otherwise)
        wrapped = getattr(self, "model", None)
        fwd = wrapped if wrapped is not None else self._model
        out = fwd(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = out.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        loss: Tensor = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_info: Any) -> Tensor:
        return self._ce(batch)

    def validation_step(self, batch: dict[str, Tensor], batch_info: Any) -> None:
        self._val_losses.append(self._ce(batch).detach().cpu())
        if len(self._val_losses) >= self._n_val_batches:
            val_ce = float(torch.stack(self._val_losses).mean())
            self.log("val_ce", torch.tensor(val_ce), across_devices=False)
            print(f"[warmstart] val_ce = {val_ce:.4f}", flush=True)
            self.last_val_metrics = {"val_ce": val_ce}
            self._val_losses = []

    def hyperparams_to_log(self) -> dict[str, object]:
        return self._cfg.to_dict()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def train_warmstart(
    model: torch.nn.Module,
    prompt: WVPrompt,
    train_data: WarmstartDataset,
    val_data: WarmstartDataset,
    cfg: WarmstartConfig,
    *,
    pad_id: int,
    wandb_project: str | None = None,
) -> dict[str, float]:
    mydule = WarmstartMydule(model, prompt, cfg, train_data, val_data, pad_id)
    tc = TrainingConfig()
    tc.name = cfg.run_name
    tc.max_epochs = cfg.epochs
    steps = max(1, cfg.epochs * (len(train_data) // cfg.micro_batch) // cfg.grad_accum)
    tc.max_steps = steps
    tc.train_batch_size = cfg.micro_batch
    tc.val_batch_size = cfg.micro_batch
    tc.gradient_accumulation_steps = cfg.grad_accum
    tc.gradient_clip_val = 1.0
    tc.lr = cfg.lr
    tc.lr_sched_type = cfg.lr_sched
    tc.final_lr = 0.0
    tc.warmup_steps = min(cfg.warmup_steps, max(1, steps // 10))
    tc.amp_dtype = None
    tc.validation_step_interval = cfg.eval_every_steps
    tc.validation_epoch_interval = 1
    tc.train_drop_last = True
    tc.val_drop_last = True
    tc.num_workers = 0
    tc.seed = cfg.seed
    tc.base_save_dir = "/tmp/mtl-runs"
    if wandb_project is not None:
        tc.logger = "wandb"
        tc.wandb_config.project = wandb_project
    tc.finalize()
    mtl_train(tc, mydule)
    return dict(mydule.last_val_metrics)
