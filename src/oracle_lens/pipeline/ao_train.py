"""Stage AO-1: supervised Activation Oracle that inverts the frozen AR.

The autoencoder is ``activation -> AO -> text -> AR -> activation'``; quality is the
reconstruction FVE ``cos^2(AR(AO(h)), h)`` in whitened space. Because AR is frozen and
deterministic, ``(AR(out), out)`` is *free* supervised data for any assistant text ``out``: train
AO so ``AO(AR(out)) ~= out`` and its emissions reconstruct by construction — the autoencoder
objective, obtained without a reward signal (RL is deferred, gated on the measured gap below AR's
ceiling).

This is the WV warm start's mechanics (``warmstart.py``: inject a vector into the ``<concept>``
slot embedding, masked-CE on a tag-wrapped target) with two changes:

- **Target is the span itself**, not its continuation. AO-1's objective *is* AR-inversion:
  produce the very text whose reconstruction was injected.
- **The injected representation is a swept knob** (``inject_rep``), the one place the
  whitened/raw/normed grid lives. Four arms (``Rep``): the vector's *source* is either ``AR(out)``
  (on the AR manifold, arms A/B/C) or the true preceding activation ``h`` (control arm D); its
  *transform* is unit-normalized, whitened-then-unit, or globally scaled.

The decision metric ``reconstruction_cos2`` scores held-out REAL activations against two anchors
(AR-of-true-text upper bound, mismatched ~0 floor) and picks the grid winner.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from jaxtyping import Float
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import do_train
from mytorch_lightning.entry import train as mtl_train
from mytorch_lightning.mydule import Mydule
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler

from oracle_lens.core.whitening import Whitener

# The injection primitives live in the harness-free ``ola.inject`` (so readout/serving import
# without pulling in mytorch_lightning); re-exported here for backward compatibility.
from oracle_lens.pipeline.inject import (  # noqa: F401  (re-exported for importers of ao_train)
    Rep,
    fit_scale,
    inject_rep,
    rep_source,
)
from oracle_lens.pipeline.scorer import AggMode, FrozenReconstructor
from oracle_lens.pipeline.shards import LongSpanPairs
from oracle_lens.pipeline.verbalizer import WVPrompt


@dataclass
class AOConfig:
    """One AO-1 grid arm. ``run_name`` is the descriptive wandb name (never coded)."""

    run_name: str
    rep: Rep
    layer: int = 44
    alpha: float = 8000.0
    scale: float = 1.0  # set by fit_scale for the raw_scaled arm; unused by the others
    n_examples: int = 500_000
    min_len: int = 5
    max_len: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    lr_sched: str = "constant"
    micro_batch: int = 4
    grad_accum: int = 8
    epochs: int = 1
    eval_every_steps: int = 200
    max_eval_rows: int = 1024
    warmup_steps: int = 20
    seed: int = 0
    init_from: str = ""  # "run/stepN" to resume the LoRA from a saved checkpoint (else fresh)
    skip_examples: int = 0  # drop the first N of the seeded permutation (resume onto UNSEEN rows)
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_ao1_examples(
    pairs: LongSpanPairs,
    tokenizer: Any,
    cfg: AOConfig,
    *,
    ar_out: Float[Tensor, "n d"] | None = None,
) -> list[dict[str, Tensor]]:
    """``(inject_vec, target_ids)`` examples; target = tag-wrapped ORIGINAL span ids + EOS.

    The injected vector's SOURCE follows the arm: ``real_h`` uses the stored preceding activation
    ``pairs.targets[i]``; the AR-manifold arms use ``ar_out[i]`` (a precomputed ``AR(span_text)``,
    aligned row-for-row with ``pairs``). The transform is applied later, at inject time, so all
    three AR-manifold arms share ONE precompute.
    """
    source = rep_source(cfg.rep)
    if source == "ar_out" and ar_out is None:
        raise ValueError(f"rep {cfg.rep!r} needs a precomputed ar_out tensor")
    open_ids = tokenizer("<explanation>\n", add_special_tokens=False)["input_ids"]
    close_ids = tokenizer("\n</explanation>", add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    lens = pairs.lengths
    keep = torch.nonzero((lens >= cfg.min_len) & (lens <= cfg.max_len)).squeeze(-1)
    gen = torch.Generator().manual_seed(cfg.seed)
    perm = keep[torch.randperm(len(keep), generator=gen)]
    # skip_examples lets a resume continue onto rows the earlier run never SELECTED: with the same
    # seed/min_len/max_len the permutation is identical, so perm[skip:] is provably disjoint from
    # the earlier run's perm[:skip] slice (no data is retrained).
    order = perm[cfg.skip_examples : cfg.skip_examples + cfg.n_examples]
    examples: list[dict[str, Tensor]] = []
    for i in order.tolist():
        span = pairs.row_ids(i).tolist()
        target = [*open_ids, *span, *close_ids, eos]
        vec = pairs.targets[i].float() if source == "real_h" else ar_out[i].float()  # type: ignore[index]
        examples.append(
            {
                "inject_vec": vec,
                "target_ids": torch.tensor(target, dtype=torch.long),
            }
        )
    return examples


class AODataset(Dataset[dict[str, Tensor]]):
    def __init__(self, examples: list[dict[str, Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return self.examples[i]


def ao_collate(rows: list[dict[str, Tensor]], *, prompt_len: int, pad_id: int) -> dict[str, Tensor]:
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
        "inject_vec": torch.stack([r["inject_vec"] for r in rows]),
        "target_ids": target_ids,
        "labels": labels,
        "attention_mask": attn,
    }


class AOMydule(Mydule):  # type: ignore[misc]
    """Masked-CE SFT with ``inject_rep(source_vec)`` replacing the ``<concept>`` slot embedding."""

    def __init__(
        self,
        model: torch.nn.Module,
        prompt: WVPrompt,
        cfg: AOConfig,
        train_data: AODataset,
        val_data: AODataset,
        pad_id: int,
        whitener: Whitener | None = None,
        ckpt_dir: Any = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._prompt = prompt
        self._cfg = cfg
        self._train_data = train_data
        self._val_data = val_data
        self._pad_id = pad_id
        self._whitener = whitener
        self._ckpt_dir = ckpt_dir  # if set, save LoRA at every validation (rank 0) -> step{N}/lora
        self._prompt_ids = torch.tensor(prompt.input_ids, dtype=torch.long)
        self._val_losses: list[Tensor] = []
        self._n_val_batches = max(1, len(val_data) // cfg.micro_batch)
        self.last_val_metrics: dict[str, float] = {}

    def create_model(self) -> torch.nn.Module:
        return self._model

    def configure_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self._model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self._cfg.lr)

    def train_data(self) -> AODataset:
        return self._train_data

    def val_data(self) -> AODataset:
        return self._val_data

    def configure_training_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt_len = len(self._prompt.input_ids)
        pad_id = self._pad_id
        args["collate_fn"] = lambda rows: ao_collate(rows, prompt_len=prompt_len, pad_id=pad_id)
        args["num_workers"] = 0
        args["persistent_workers"] = False
        # Shard the TRAIN set across ranks so 8-GPU DDP is one epoch of disjoint data (not the
        # whole set x world_size). Val is left unsharded: every rank validates the full set, so
        # val_ce is the same whole-set metric on each rank (<=1024 rows, negligible duplicate).
        data: Any = args.get("dataset")
        is_train = data is self._train_data
        world = self.trainer.config.world_size() if (is_train and self.trainer) else 1
        rank = self.trainer.global_rank if (is_train and self.trainer) else 0
        if world > 1:
            args["sampler"] = DistributedSampler(
                data, num_replicas=world, rank=rank, shuffle=True, seed=self._cfg.seed
            )
            args.pop("shuffle", None)  # exclusive with an explicit sampler
        return args

    def configure_validation_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.configure_training_dl(args)

    def _ce(self, batch: dict[str, Tensor]) -> Tensor:
        device = batch["inject_vec"].device
        b = batch["inject_vec"].shape[0]
        embed: Any = self._model.get_input_embeddings()  # type: ignore[operator]
        prompt_ids = self._prompt_ids.to(device).unsqueeze(0).expand(b, -1)
        # clone: input-require-grads marks embed output a leaf; in-place writes forbidden
        prompt_embeds = embed(prompt_ids).clone()
        v = inject_rep(
            batch["inject_vec"],
            self._cfg.rep,
            alpha=self._cfg.alpha,
            scale=self._cfg.scale,
            whitener=self._whitener,
        )
        prompt_embeds[:, self._prompt.slot, :] = v.to(prompt_embeds.dtype)
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
        # under DDP mytorch shards the val set -> each rank sees len(val)//world batches, so the
        # trigger count must be per-rank (else the val_ce + checkpoint block never fires)
        world = self.trainer.config.world_size() if self.trainer is not None else 1
        n_expected = max(1, (len(self._val_data) // max(1, world)) // self._cfg.micro_batch)
        if len(self._val_losses) >= n_expected:
            val_ce = float(torch.stack(self._val_losses).mean())
            self.log("val_ce", torch.tensor(val_ce), across_devices=False)
            print(f"[ao] val_ce = {val_ce:.4f}", flush=True)
            self.last_val_metrics = {"val_ce": val_ce}
            self._val_losses = []
            self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Rank-0 LoRA checkpoint at each validation -> ckpt_dir/step{N}/lora (mid-run evals)."""
        if self._ckpt_dir is None:
            return
        rank = self.trainer.global_rank if self.trainer is not None else 0
        if rank != 0:
            return
        step = int(getattr(self.trainer, "global_step", 0)) if self.trainer is not None else 0
        out = self._ckpt_dir / f"step{step}"
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out / "lora"))  # type: ignore[operator]
        print(f"[ao] checkpoint saved: step {step}", flush=True)

    def hyperparams_to_log(self) -> dict[str, object]:
        return self._cfg.to_dict()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def train_ao(
    model: torch.nn.Module,
    prompt: WVPrompt,
    train_data: AODataset,
    val_data: AODataset,
    cfg: AOConfig,
    *,
    pad_id: int,
    whitener: Whitener | None = None,
    wandb_project: str | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    ckpt_dir: Any = None,
) -> dict[str, float]:
    """One AO-1 run; ``world_size > 1`` runs ONE DDP rank in this process (caller spawns them).

    grad_accum is divided by world_size (when it divides) so the GLOBAL batch is unchanged as GPUs
    scale — the LR stays valid and 8-GPU is a ~k x speedup at the same optimization, not a bigger
    effective batch. Steps are counted from the PER-RANK shard (DistributedSampler splits the set).
    ``ckpt_dir`` set -> a LoRA checkpoint is saved at every validation for mid-run evals.
    """
    mydule = AOMydule(
        model, prompt, cfg, train_data, val_data, pad_id, whitener=whitener, ckpt_dir=ckpt_dir
    )
    world_size = max(1, world_size)
    grad_accum = max(1, cfg.grad_accum // world_size) if world_size > 1 else cfg.grad_accum
    per_rank = max(1, len(train_data) // world_size)
    tc = TrainingConfig()
    tc.name = cfg.run_name
    tc.max_epochs = cfg.epochs
    steps = max(1, cfg.epochs * (per_rank // cfg.micro_batch) // grad_accum)
    tc.max_steps = steps
    tc.ngpu = world_size
    tc.train_batch_size = cfg.micro_batch
    tc.val_batch_size = cfg.micro_batch
    tc.gradient_accumulation_steps = grad_accum
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
    if world_size > 1:
        do_train(tc, local_rank, mydule)
    else:
        mtl_train(tc, mydule)
    return dict(mydule.last_val_metrics)


def reconstruction_cos2(
    frozen_ar: FrozenReconstructor,
    texts_per_act: list[list[str]],
    targets: Float[Tensor, "m d"],
    *,
    mode: AggMode = "single",
) -> Tensor:
    """Decision metric core: FVE_fitted = ``cos^2(AR(text), h)`` per activation (whitened space).

    Sign is folded by squaring (the official fitted-coefficient FVE — a negative whitened cosine
    still explains variance under a fitted negative coefficient). Invalid (all-empty) rows stay
    NaN so the caller can drop them before averaging.
    """
    out = frozen_ar.score(texts_per_act, targets, mode=mode)
    return out["whitened_cos"] ** 2
