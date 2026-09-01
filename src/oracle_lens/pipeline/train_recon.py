"""Long-span reconstructor training (mytorch-lightning harness, whitened-MSE loss).

Adapts ``oracle_lens.train_recon`` to ragged 1..1024-token spans: bucketed static shapes
(``buckets.BUCKETS``) instead of one pad_width, a DDP-aware ``BucketBatchSampler`` instead of
plain shuffling, and the plain **whitened MSE** loss (user decision — one error metric across
the whole system; the whitened-cosine variant stays available as a logged metric only).

mytorch-lightning stays ONLY here (reusing the debugged loop — rewriting it is the hand-rolled
-loop risk); the WV/RL side has no mytorch. Harness caveats inherited: LR schedulers write the
same lr to every param group (LoRA and head share ``cfg.lr``), and a batch_sampler discards the
harness's DistributedSampler (sharding lives in ``BucketBatchSampler``).

Default recipe = the frozen oracle-lens scaling recipe (LN head, ridge_c=0.1, lr 3e-4); epochs
default to 1 (standing user decision: scale data, not epochs).
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from jaxtyping import Float
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import do_train
from mytorch_lightning.entry import train as mtl_train
from mytorch_lightning.mydule import Mydule
from torch import Tensor
from torch.utils.data import Dataset

from oracle_lens.core.evals import predict_mean_r2, retrieval_top1
from oracle_lens.core.reconstructor import (
    Reconstructor,
    cosine_mean,
    r2,
    whitened_cosine_loss,
)
from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.buckets import (
    BUCKETS,
    Bucket,
    BucketBatchSampler,
    RaggedPairsDataset,
    batches_per_rank,
    bucket_collate,
)
from oracle_lens.pipeline.shards import LENGTH_BUCKETS, LongSpanPairs


def whitened_mse_loss(
    pred: Float[Tensor, "b d"],
    target: Float[Tensor, "b d"],
    whitener: Whitener,
) -> Float[Tensor, ""]:
    """Mean squared error between whitened prediction and whitened target (mean over b*d)."""
    loss: Tensor = torch.nn.functional.mse_loss(whitener.whiten(pred), whitener.whiten(target))
    return loss


@dataclass
class LongSpanReconConfig:
    """One training run. ``run_name`` is the descriptive wandb name (never coded)."""

    run_name: str
    layer: int = 44
    n_pairs: int = 30_000  # 0 = everything after base_skip
    # warm start: continue from a saved run's LoRA+head (compile-first load, `_orig_mod.` keys)
    init_from: str = ""
    # skip the first K rows of the FIXED base shuffle (seed 1234) — lets a continuation run
    # train on exactly the pairs a previous prefix run never saw (1 epoch per pair, ever)
    base_skip: int = 0
    # diagnostics (2026-07-19 flat-curve investigation):
    # C/E1: train only on spans within [min_train_len, max_train_len] (0 = off) — composition
    # test vs the old N<=32 curve (old-like mix = min 5, max 32: drops the 35% ultra-short rows
    # and the 28% long rows of the log-uniform draw)
    max_train_len: int = 0
    min_train_len: int = 0
    # drop NAME_N anonymization-placeholder spans from TRAINING data (evals untouched);
    # applied before prefix selection so n_pairs still yields full quantity (user 2026-07-19)
    drop_placeholders: bool = True
    # A: multiply batch_rows for buckets with pad_width >= 256 — every optimizer step then
    # averages >=8-32 rows instead of 4-16 (old recipe: uniform 128 rows/step)
    long_rows_boost: int = 1
    # D: truncate the backbone at this block instead of cfg.layer (0 = at cfg.layer, the
    # default L44->L44 design; 63 = full depth: final-layer state -> head -> L44 activation)
    truncate_at: int = 0
    ridge_c: float = 0.1
    head_layer_norm: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 3e-4
    lr_sched: str = "constant"  # mtl lr_sched_type: cosine | linear | constant
    grad_accum: int = 1
    grad_checkpointing: bool = True
    compile_blocks: bool = True
    epochs: int = 1  # standing decision: scale data, not epochs
    # "whitened_cosine" (default — user decision 2026-07-19: follow the oracle-lens recipe,
    # MSE between UNIT-NORMALIZED whitened vectors = 2(1-cos); also exactly the NLA repo's
    # -mse_nrm reward metric) or "whitened_mse" (magnitude-sensitive alternative, kept as flag)
    loss: str = "whitened_cosine"
    eval_every_steps: int = 200
    max_eval_rows: int = 4096
    retrieval_pool: int = 100
    warmup_steps: int = 20
    seed: int = 0
    pad_id: int = 0
    # DDP effective-batch policy (same semantics as oracle_lens): keep_global_batch divides
    # grad_accum by world_size so the global batch is unchanged when world_size | grad_accum;
    # when it can't divide, lr_linear_scale applies the linear LR-scaling rule instead.
    keep_global_batch: bool = True
    lr_linear_scale: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LongSpanReconMydule(Mydule):  # type: ignore[misc]
    """Whitened-MSE training step + whole-set validation metrics over bucketed batches."""

    def __init__(
        self,
        recon: Reconstructor,
        whitener: Whitener,
        cfg: LongSpanReconConfig,
        train_pairs: LongSpanPairs,
        eval_pairs: LongSpanPairs,
        *,
        buckets: tuple[Bucket, ...] = BUCKETS,
    ) -> None:
        super().__init__()
        self._recon = recon
        self._whitener = whitener
        self._cfg = cfg
        self._buckets = buckets
        self._train_data = RaggedPairsDataset(train_pairs)
        self._eval_data = RaggedPairsDataset(eval_pairs)
        self._tokens_consumed = 0
        self._t_start: float | None = None
        self._tokens_at_start = 0
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []
        self._val_lens: list[Tensor] = []
        # every rank validates the FULL eval set (num_replicas=1 below): whole-set metrics with
        # no cross-rank gather; duplicate compute over <=4096 rows is negligible
        self._n_val_batches = len(
            BucketBatchSampler(self._eval_data.lengths, buckets=buckets, num_replicas=1, rank=0)
        )
        self._train_mean_raw = train_pairs.targets.float().mean(dim=0)
        self.last_val_metrics: dict[str, float] = {}

    # --- model / optimizer -------------------------------------------------------------
    def create_model(self) -> torch.nn.Module:
        return self._recon  # built (truncated + LoRA + compiled) by the caller

    def configure_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self._recon.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self._cfg.lr)

    # --- data --------------------------------------------------------------------------
    def train_data(self) -> Dataset[dict[str, Tensor]]:
        return self._train_data

    def val_data(self) -> Dataset[dict[str, Tensor]]:
        return self._eval_data

    def _collate(self) -> Callable[[list[dict[str, Tensor]]], dict[str, Tensor]]:
        pad_id, buckets = self._cfg.pad_id, self._buckets

        def collate(rows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
            return bucket_collate(rows, pad_id=pad_id, buckets=buckets)

        return collate

    def configure_training_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        args["collate_fn"] = self._collate()
        args["num_workers"] = 0  # tensors already in RAM; fork workers would copy ~GBs
        args["persistent_workers"] = False
        data: Any = args["dataset"]
        is_train = data is self._train_data
        world = self.trainer.config.world_size() if (is_train and self.trainer) else 1
        rank = self.trainer.global_rank if (is_train and self.trainer) else 0
        args["batch_sampler"] = BucketBatchSampler(
            data.lengths,
            buckets=self._buckets,
            seed=self._cfg.seed,
            num_replicas=world,
            rank=rank,
        )
        # batch_sampler is exclusive with these DataLoader kwargs
        for k in ("batch_size", "shuffle", "sampler", "drop_last"):
            args.pop(k, None)
        return args

    def configure_validation_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.configure_training_dl(args)

    # --- steps -------------------------------------------------------------------------
    def training_step(self, batch: dict[str, Tensor], batch_info: Any) -> Tensor:
        pred = self.model(batch["input_ids"], batch["attention_mask"])
        if self._cfg.loss == "whitened_cosine":
            loss = whitened_cosine_loss(pred, batch["target"].float(), self._whitener)
        else:
            loss = whitened_mse_loss(pred, batch["target"].float(), self._whitener)
        self._tokens_consumed += int(batch["span_len"].sum())
        now = time.monotonic()
        if self._t_start is None:
            self._t_start = now  # first step ends warmup; rate excludes load/compile
            self._tokens_at_start = self._tokens_consumed
        elif now > self._t_start:
            rate = (self._tokens_consumed - self._tokens_at_start) / (now - self._t_start)
            self.log(
                "tokens_per_sec", torch.tensor(float(rate)), reduction="max", across_devices=False
            )
        self.log(
            "tokens_consumed",
            torch.tensor(float(self._tokens_consumed)),
            reduction="max",
            across_devices=False,
        )
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_info: Any) -> None:
        pred = self.model(batch["input_ids"], batch["attention_mask"])
        self._val_preds.append(pred.float().detach().cpu())
        self._val_targets.append(batch["target"].float().cpu())
        self._val_lens.append(batch["span_len"].cpu())
        if len(self._val_preds) >= self._n_val_batches:
            metrics = self._whole_set_metrics()
            for key, value in metrics.items():
                self.log(key, torch.tensor(value), across_devices=False)
            core = {k: round(v, 4) for k, v in metrics.items() if "/" not in k}
            print(f"[val] {core}", flush=True)
            self.last_val_metrics = metrics
            self._val_preds, self._val_targets, self._val_lens = [], [], []

    def _whole_set_metrics(self) -> dict[str, float]:
        pred = torch.cat(self._val_preds)
        target = torch.cat(self._val_targets)
        lens = torch.cat(self._val_lens)
        whitener = self._whitener.to("cpu")
        pred_w = whitener.whiten(pred)
        target_w = whitener.whiten(target)
        gen = torch.Generator().manual_seed(self._cfg.seed)
        cos = torch.nn.functional.cosine_similarity(pred_w, target_w, dim=-1)
        fve = r2(pred_w, target_w)
        metrics = {
            # validation counterparts of both loss variants
            "whitened_mse": float(torch.nn.functional.mse_loss(pred_w, target_w)),
            "whitened_cosine_loss": float(2.0 * (1.0 - cos.mean())),
            # FVE = whitened R^2 (NLA convention): 1 - residual / centered variance
            "fve": fve,
            "whitened_r2": fve,
            "raw_r2": r2(pred, target),
            "whitened_cos": cosine_mean(pred_w, target_w),
            "whitened_cos2": float((cos**2).mean()),
            "retrieval_top1": retrieval_top1(
                pred_w,
                target_w,
                pool=min(self._cfg.retrieval_pool, pred_w.shape[0]),
                generator=gen,
            ),
            "predict_mean_r2": predict_mean_r2(
                target_w, whitener.whiten(self._train_mean_raw.unsqueeze(0)).squeeze(0)
            ),
        }
        for lo, hi in LENGTH_BUCKETS:
            sel = (lens >= lo) & (lens <= hi)
            if int(sel.sum()) >= 2:
                metrics[f"whitened_r2/N{lo}-{hi}"] = r2(pred_w[sel], target_w[sel])
        # per-rank counter x world = GLOBAL tokens (ranks train disjoint shards; the 100k run
        # logged 485k for an actual 3.88M before this fix)
        world = self.trainer.config.world_size() if self.trainer is not None else 1
        metrics["tokens_consumed_at_eval"] = float(self._tokens_consumed * max(1, world))
        return metrics

    def hyperparams_to_log(self) -> dict[str, object]:
        return self._cfg.to_dict()

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        out: Tensor = self._recon(*args, **kwargs)
        return out


def build_longspan_training_config(
    cfg: LongSpanReconConfig,
    train_lengths: list[int],
    *,
    buckets: tuple[Bucket, ...] = BUCKETS,
    world_size: int = 1,
    wandb_project: str | None = None,
    base_save_dir: str = "/tmp/mtl-runs",
) -> TrainingConfig:
    """Map the experiment config onto the harness TrainingConfig.

    Batch accounting is in MICRO-BATCHES from the bucketed sampler (row counts differ per
    bucket but every micro-batch holds the same padded token count), so ``max_steps`` comes
    from ``batches_per_rank`` — not from n_pairs // batch_size. DDP effective-batch policy is
    the oracle-lens one (see ``LongSpanReconConfig.keep_global_batch``).
    """
    world_size = max(1, world_size)
    grad_accum = cfg.grad_accum
    if world_size > 1 and cfg.keep_global_batch:
        grad_accum = max(1, cfg.grad_accum // world_size)
    growth = (world_size * grad_accum) / cfg.grad_accum
    per_rank_batches = batches_per_rank(train_lengths, buckets=buckets, num_replicas=world_size)
    steps = max(1, cfg.epochs * per_rank_batches // grad_accum)

    tc = TrainingConfig()
    tc.name = cfg.run_name
    tc.ngpu = world_size
    tc.max_epochs = cfg.epochs
    tc.max_steps = steps
    tc.train_batch_size = max(b.batch_rows for b in buckets)  # superseded by batch_sampler
    tc.val_batch_size = max(b.batch_rows for b in buckets)
    tc.gradient_accumulation_steps = grad_accum
    tc.gradient_clip_val = 1.0
    tc.lr = cfg.lr * growth if cfg.lr_linear_scale else cfg.lr
    if world_size > 1:
        print(
            f"[ddp] world_size={world_size} grad_accum {cfg.grad_accum}->{grad_accum} "
            f"token-batch growth x{growth:.2f} lr={tc.lr:.2e} max_steps={steps}",
            flush=True,
        )
        if growth != 1.0 and not cfg.lr_linear_scale:
            print(
                "[ddp] WARNING: global token batch changed but lr_linear_scale=False — set it "
                "(or raise grad_accum to a multiple of world_size) to keep the run "
                "mathematically comparable to 1-GPU.",
                flush=True,
            )
    tc.lr_sched_type = cfg.lr_sched
    tc.final_lr = 0.0
    tc.warmup_steps = min(cfg.warmup_steps, max(1, steps // 10))
    tc.amp_dtype = None  # dtypes are explicit (bf16 base, fp32 LoRA+head)
    tc.validation_step_interval = cfg.eval_every_steps
    tc.validation_epoch_interval = 1
    tc.train_drop_last = True
    tc.val_drop_last = True
    tc.num_workers = 0
    tc.seed = cfg.seed
    tc.base_save_dir = base_save_dir
    if wandb_project is not None:
        tc.logger = "wandb"
        tc.wandb_config.project = wandb_project
    tc.finalize()
    return tc


def train_longspan_reconstructor(
    recon: Reconstructor,
    train_pairs: LongSpanPairs,
    eval_pairs: LongSpanPairs,
    whitener: Whitener,
    cfg: LongSpanReconConfig,
    *,
    wandb_project: str | None = None,
    local_rank: int = 0,
    world_size: int = 1,
) -> dict[str, float]:
    """Run one training; returns the last validation metrics.

    ``world_size > 1`` runs ONE DDP rank in this process (the caller spawns the workers, each
    building its model on ``cuda:local_rank``): dispatch goes through the harness's per-rank
    ``do_train`` — ``mtl_train`` would fork the pre-built GPU model and re-spawn.
    """
    if cfg.max_eval_rows and len(eval_pairs) > cfg.max_eval_rows:
        eval_pairs = eval_pairs.select(torch.arange(cfg.max_eval_rows))
    if cfg.max_train_len or cfg.min_train_len:
        lens = train_pairs.lengths
        mask = torch.ones(len(train_pairs), dtype=torch.bool)
        if cfg.max_train_len:
            mask &= lens <= cfg.max_train_len
        if cfg.min_train_len:
            mask &= lens >= cfg.min_train_len
        keep = torch.nonzero(mask).squeeze(-1)
        print(
            f"[diag] len filter [{cfg.min_train_len},{cfg.max_train_len}]: "
            f"{len(keep)}/{len(train_pairs)} rows"
        )
        train_pairs = train_pairs.select(keep)
    buckets = BUCKETS
    if cfg.long_rows_boost > 1:
        buckets = tuple(
            Bucket(b.pad_width, b.batch_rows * (cfg.long_rows_boost if b.pad_width >= 256 else 1))
            for b in BUCKETS
        )
        print(f"[diag A] long_rows_boost={cfg.long_rows_boost}: buckets={buckets}")
    mydule = LongSpanReconMydule(
        recon, whitener.to("cuda"), cfg, train_pairs, eval_pairs, buckets=buckets
    )
    tc = build_longspan_training_config(
        cfg,
        mydule._train_data.lengths,
        buckets=buckets,
        world_size=world_size,
        wandb_project=wandb_project,
    )
    if world_size > 1:
        do_train(tc, local_rank, mydule)
    else:
        mtl_train(tc, mydule)
    return dict(mydule.last_val_metrics)
