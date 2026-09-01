"""Soft-token SFT: inject ONE vector at a prompt slot, train the LM to emit a text span.

The shared training core for both lenses that do this — the **GT-continuation lens**
(`gt_train.py`, injects the true stored residual) and the **AO ladder** (`ao_ladder.py`, injects
an AR reconstruction). Everything here is lens-agnostic:

- ``SoftTokenConfig`` — the run knobs (batch, lr, layers, resume offsets),
- ``LayerBucketBatchSampler`` — layer-pure micro-batches; ``deal_key`` additionally synchronizes a
  property (the AO uses crop length) across DDP ranks so a step is not priced by its slowest rank,
- ``soft_token_collate`` — right-pad, mask loss to the target, count span/supervised tokens EXACTLY,
- ``SoftTokenMydule`` — masked CE with the injected vector written into the prompt's slot embedding;
  exact token accounting (all-reduced), per-layer and per-length val CE, checkpoint + resume state,
- ``train_soft_token`` — the harness config, including the multi-node topology and the val-batch
  sanity check.

A lens supplies: rendered prompts (one per layer), a Dataset yielding
``{inject_vec, layer_idx, target_ids, span_len}``, and an injection transform. It should NOT need
to touch this file.

Target construction is byte-identical to AO-1: ``<explanation>\n`` + span ids +
``\n</explanation>`` + EOS, masked-CE on target tokens only.
"""

import json
import os
import signal
import time
from collections.abc import Iterator, Sized
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import torch
import torch.distributed as dist
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import do_train
from mytorch_lightning.entry import train as mtl_train
from mytorch_lightning.mydule import Mydule
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from oracle_lens.pipeline.inject import SCALE_FREE_GT, GtTransform, inject_gt
from oracle_lens.pipeline.verbalizer import WVPrompt

# Set by the SIGUSR1 handler (gpu.sbatch sends USR1 to the process group with a ~20 s grace
# before teardown on preemption/timeout). Checked once per training step; all DDP ranks agree
# via an all_reduce so the checkpoint-and-exit is collective, not racy.
_PREEMPT_FLAG = {"set": False}


def install_preempt_handler() -> None:
    """Install in every spawned rank BEFORE training starts (see gpu.sbatch on_preempt)."""

    def _handler(signum: int, frame: FrameType | None) -> None:
        _PREEMPT_FLAG["set"] = True
        print("[sft] SIGUSR1 — will save resume state and exit at the next step", flush=True)

    signal.signal(signal.SIGUSR1, _handler)


@dataclass
class SoftTokenConfig:
    """One soft-token SFT run. ``scales`` is keyed by ``str(layer)`` (JSON round-trip safe)."""

    run_name: str
    transform: GtTransform = "unit"
    layers: tuple[int, ...] = (44,)
    alpha: float = 8000.0
    clip_mult: float = 4.0
    scales: dict[str, float] = field(default_factory=dict)  # per-layer, fit_scale (scaled arms)
    n_examples: int = 500_000
    min_len: int = 5
    max_len: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    lr_sched: str = "constant"
    micro_batch: int = 4
    # Validation batch size. MUST be small enough that every (layer[, length]) group can form a
    # full batch: the sampler drops partials, so a val batch size above the smallest group size
    # yields ZERO batches -> validation never fires -> no checkpoints (silent, cost a 29-min run
    # on 2026-07-29). ``train_soft_token`` asserts this at startup. 0 = same as micro_batch.
    val_micro_batch: int = 0
    grad_accum: int = 8
    epochs: int = 1
    eval_every_steps: int = 200
    max_eval_rows: int = 1024
    warmup_steps: int = 20
    seed: int = 0
    init_from: str = ""  # "run/stepN" to resume the LoRA from a saved checkpoint (else fresh)
    skip_examples: int = 0  # drop the first N of the seeded permutation (resume onto UNSEEN rows)
    # Mid-epoch resume: per-rank micro-batches already consumed. The batch sampler regenerates
    # the identical batch list from (seed, epoch), so skipping the first K batches is an EXACT
    # continuation (skip_examples cannot be — the sampler shuffles, so consumed rows are not a
    # permutation prefix). Set from resume/state.json by the entry script; epochs must be 1.
    skip_batches: int = 0
    # Exact-remainder continuation: .npy of int64 example indices (OLA_ROOT-relative path ok)
    # EXCLUDED from the train sampler before grouping, so the run trains only the remainder in a
    # fresh seeded order. Unlike skip_batches this survives a (micro_batch, world) change: the
    # consumed set is computed offline by replaying the PARENT's sampler geometry
    # (scripts/iolens/u64_consumed_set.py). Mutually exclusive with skip_batches.
    exclude_idx_file: str = ""
    # DataLoader worker processes for the TRAIN loader (0 = read in the training loop). The AO
    # ladder's per-item LazyTargets mmap read is NFS-latency-bound — workers hide it behind
    # compute. LazyTargets reopens per PID, so forking is safe.
    num_workers: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SoftTokenConfig":
        d = dict(d)
        d["layers"] = tuple(d.get("layers", (44,)))
        return cls(**d)

    def scale_for(self, layer: int) -> float:
        # Scale-free arms fix the injected norm at alpha by unit-normalising, so they carry no
        # per-layer fit_scale entry; indexing `scales` for them would KeyError.
        if self.transform in SCALE_FREE_GT:
            return 1.0
        return self.scales[str(layer)]


def layer_indices(data: Any) -> list[int]:
    """Per-example layer index for the batch sampler.

    Datasets with materialized example dicts are walked; datasets whose layer
    assignment is arithmetic (the AO ladder's (crop x layer) cross product) expose
    ``layer_idx_per_example()`` instead so millions of dicts are never built.
    """
    fn = getattr(data, "layer_idx_per_example", None)
    if fn is not None:
        return list(fn())
    return [int(ex["layer_idx"]) for ex in data.examples]


def deal_key_for(data: Any) -> dict[int, int] | None:
    """group -> DDP deal bucket, if the dataset defines one (see LayerBucketBatchSampler).

    The AO ladder groups by ``layer_idx * n_lengths + length_idx``, so the bucket is the length
    index: ranks then share a sequence length each step (no straggler) while still covering
    different layers.
    """
    fn = getattr(data, "deal_key", None)
    return dict(fn()) if fn is not None else None


def group_indices(data: Any) -> list[int]:
    """Batch-group key per example. Defaults to the layer index (layer-pure batches).

    Datasets may expose ``group_idx_per_example()`` to refine grouping — the AO ladder groups by
    (layer, crop length): with 6 discrete lengths every batch is length-pure too, so the collate
    pads nothing (measured ~1.4x padded-token waste under length-mixed batches) and shapes are
    static per group. Any refinement MUST keep batches layer-pure (the collate's contract).
    """
    fn = getattr(data, "group_idx_per_example", None)
    if fn is not None:
        return list(fn())
    return layer_indices(data)


def load_exclude_mask(cfg: SoftTokenConfig, n_examples: int) -> Tensor | None:
    """Bool mask (True = excluded) from ``cfg.exclude_idx_file`` (int64 .npy of example indices).

    Relative paths resolve under ``$OLA_ROOT``. Indices must be unique and < ``n_examples`` —
    a mismatch means the consumed set was computed against a DIFFERENT dataset (wrong pool or
    arout), which would silently exclude the wrong examples, so it is a hard error.
    """
    if not cfg.exclude_idx_file:
        return None
    import numpy as np

    p = Path(cfg.exclude_idx_file)
    if not p.is_absolute():
        p = Path(os.environ["OLA_ROOT"]) / p
    idx = torch.from_numpy(np.load(p)).long()
    if int(idx.max()) >= n_examples or int(idx.min()) < 0:
        raise SystemExit(
            f"[sft] exclude_idx_file {p}: index range [{int(idx.min())}, {int(idx.max())}] "
            f"does not fit the {n_examples}-example dataset — consumed set was computed "
            "against a different pool/arout"
        )
    mask = torch.zeros(n_examples, dtype=torch.bool)
    mask[idx] = True
    if int(mask.sum()) != len(idx):
        raise SystemExit(f"[sft] exclude_idx_file {p}: duplicate indices")
    return mask


class LayerBucketBatchSampler(Sampler[list[int]]):
    """Layer-pure micro-batches, sharded across DDP ranks by whole batches.

    Within each layer group the order is a seeded per-epoch permutation; partial batches are
    dropped (layer purity over completeness); the batch list is then globally shuffled and rank
    ``r`` takes a contiguous ``r::world`` slice trimmed so every rank steps the same count.
    """

    def __init__(
        self,
        layer_idx_per_example: list[int],
        *,
        micro_batch: int,
        world: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        skip_batches: int = 0,
        deal_key: dict[int, int] | None = None,
        exclude_mask: Tensor | None = None,
    ) -> None:
        # ``deal_key`` maps a group -> the property that must MATCH ACROSS RANKS in a step.
        # Without it every rank draws an independent group, so a DDP step costs the slowest rank:
        # with 6 crop lengths, 84% of steps have some rank holding the 129-token N=64 batch while
        # others hold 67-token ones. Measured 1.38x slowdown = exactly E[max of 8]/E[mean] over
        # the length mix. Dealing same-length batches across ranks removes it and keeps layers
        # diverse within the step (layer only shifts the prompt by 1 token).
        self.deal_key = deal_key or {}
        self.groups: dict[int, list[int]] = {}
        if exclude_mask is None:
            for i, ly in enumerate(layer_idx_per_example):
                self.groups.setdefault(ly, []).append(i)
        else:
            # Exact-remainder mode (cfg.exclude_idx_file): consumed examples are dropped BEFORE
            # grouping, so batches are rebuilt over the remainder only, in a fresh seeded order.
            # Cannot be combined with skip_batches (which indexes the FULL batch list).
            if skip_batches:
                raise ValueError("exclude_mask and skip_batches are mutually exclusive")
            if len(exclude_mask) != len(layer_idx_per_example):
                raise ValueError(
                    f"exclude_mask has {len(exclude_mask)} entries for "
                    f"{len(layer_idx_per_example)} examples"
                )
            excl = exclude_mask.tolist()
            for i, ly in enumerate(layer_idx_per_example):
                if not excl[i]:
                    self.groups.setdefault(ly, []).append(i)
        self.micro_batch = micro_batch
        self.world = max(1, world)
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        # Mid-epoch resume: the batch list is a pure function of (seed, epoch), so skipping the
        # first ``skip_batches`` per-rank slots continues the interrupted epoch exactly.
        self.skip_batches = skip_batches
        n_batches = sum(len(g) // micro_batch for g in self.groups.values())
        if self.deal_key and self.world > 1:
            # only whole rank-chunks WITHIN a deal bucket are usable
            per_bucket: dict[int, int] = {}
            for g, idx in self.groups.items():
                per_bucket[self.deal_key.get(g, 0)] = per_bucket.get(self.deal_key.get(g, 0), 0) + (
                    len(idx) // micro_batch
                )
            self._per_rank = sum(nb // self.world for nb in per_bucket.values())
        else:
            self._per_rank = n_batches // self.world

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return max(0, self._per_rank - self.skip_batches)

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        tagged: list[tuple[int, list[int]]] = []  # (deal bucket, batch)
        for ly in sorted(self.groups):
            idx = torch.tensor(self.groups[ly], dtype=torch.long)
            if self.shuffle:
                idx = idx[torch.randperm(len(idx), generator=gen)]
            n_full = len(idx) // self.micro_batch
            bucket = self.deal_key.get(ly, 0)
            for b in range(n_full):
                tagged.append(
                    (bucket, idx[b * self.micro_batch : (b + 1) * self.micro_batch].tolist())
                )
        if not self.deal_key or self.world == 1:
            batches = [t[1] for t in tagged]
            if self.shuffle:
                batches = [batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()]
            for b in range(self.skip_batches, self._per_rank):
                yield batches[b * self.world + self.rank]
            return
        # deal rank-chunks WITHIN each bucket: every rank gets the same bucket in a given step
        by_bucket: dict[int, list[list[int]]] = {}
        for bucket, batch in tagged:
            by_bucket.setdefault(bucket, []).append(batch)
        chunks: list[list[list[int]]] = []
        for bucket in sorted(by_bucket):
            bl = by_bucket[bucket]
            if self.shuffle:
                bl = [bl[i] for i in torch.randperm(len(bl), generator=gen).tolist()]
            for c in range(len(bl) // self.world):
                chunks.append(bl[c * self.world : (c + 1) * self.world])
        if self.shuffle:
            chunks = [chunks[i] for i in torch.randperm(len(chunks), generator=gen).tolist()]
        for c in range(self.skip_batches, min(self._per_rank, len(chunks))):
            yield chunks[c][self.rank]


def soft_token_collate(
    rows: list[dict[str, Tensor]], *, prompt_lens: list[int], pad_id: int
) -> dict[str, Tensor]:
    """Right-pad targets; labels = -100 on prompt+pad. Batches are layer-pure by construction."""
    layer_idx = int(rows[0]["layer_idx"])
    if any(int(r["layer_idx"]) != layer_idx for r in rows):
        raise ValueError("mixed-layer batch — LayerBucketBatchSampler contract violated")
    prompt_len = prompt_lens[layer_idx]
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
    # Exact token accounting (runbook invariant: count, never estimate). Computed on CPU here so
    # the trainer only accumulates ints. Supervised = every label position that contributes to CE;
    # span = the raw continuation tokens (the AR-comparable x-axis unit).
    n_span = int(sum(int(r["span_len"]) for r in rows))
    n_sup = int((labels != -100).sum())
    span_lens = torch.stack([r["span_len"] for r in rows])
    return {
        "inject_vec": torch.stack([r["inject_vec"] for r in rows]),
        "layer_idx": torch.tensor(layer_idx, dtype=torch.long),
        "target_ids": target_ids,
        "labels": labels,
        "attention_mask": attn,
        "span_lens": span_lens,
        "n_span_tokens": torch.tensor(n_span, dtype=torch.long),
        "n_sup_tokens": torch.tensor(n_sup, dtype=torch.long),
    }


class SoftTokenMydule(Mydule):  # type: ignore[misc]
    """Masked-CE SFT: the injected vector replaces the per-layer prompt's ``<concept>`` slot."""

    def __init__(
        self,
        model: torch.nn.Module,
        prompts: list[WVPrompt],
        cfg: SoftTokenConfig,
        train_data: Dataset[dict[str, Tensor]],
        val_data: Dataset[dict[str, Tensor]],
        pad_id: int,
        ckpt_dir: Any = None,
        resume_state: dict[str, Any] | None = None,
        spaces: list[Any] | None = None,
    ) -> None:
        super().__init__()
        if len(prompts) != len(cfg.layers):
            raise ValueError(f"{len(prompts)} prompts for {len(cfg.layers)} layers")
        self._model = model
        self._prompts = prompts
        self._cfg = cfg
        self._train_data = train_data
        self._val_data = val_data
        self._pad_id = pad_id
        self._ckpt_dir = ckpt_dir
        self._prompt_ids = [torch.tensor(p.input_ids, dtype=torch.long) for p in prompts]
        self._prompt_lens = [len(p.input_ids) for p in prompts]
        self._scales = [cfg.scale_for(ly) for ly in cfg.layers]
        # Per-layer ruler for the metric-space arms (wht_unit / j_unit), aligned with cfg.layers.
        self._spaces: list[Any] = list(spaces) if spaces is not None else [None] * len(cfg.layers)
        if cfg.transform in SCALE_FREE_GT - {"unit"}:
            missing = [ly for ly, sp in zip(cfg.layers, self._spaces, strict=True) if sp is None]
            if missing:
                raise ValueError(
                    f"transform={cfg.transform} needs a metric space per layer; missing {missing}"
                    " (drop those layers from cfg.layers — e.g. L63 has no Jacobian)"
                )
        self._val_losses: list[Tensor] = []
        self._val_layer_losses: dict[int, list[Tensor]] = {}
        # Per crop LENGTH too: the target wraps the span in ~9 fixed tokens, so a single averaged
        # CE mixes short crops (mostly free wrapper tokens) with long ones. Batches are
        # length-pure under the AO ladder's grouping, so one key per batch is exact.
        self._val_len_losses: dict[int, list[Tensor]] = {}
        self.last_val_metrics: dict[str, float] = {}
        # Exact token accounting + resume offsets. Per-rank counters are all_reduce(SUM)ed at
        # validation so recorded totals are exact (rank0-count x world is only approximate — each
        # rank sees different batches). ``resume_state`` comes from resume/state.json.
        rs = resume_state or {}
        self._step_offset = int(rs.get("micro_steps", 0))
        self._tokens_span_prev = int(rs.get("tokens_span", 0))  # global, from prior segments
        self._tokens_sup_prev = int(rs.get("tokens_sup", 0))
        self._resume_opt_path = str(rs.get("optimizer_path", ""))
        self._tokens_span_rank = 0  # this rank, this segment
        self._tokens_sup_rank = 0
        # running train loss since the last validation -> recorded next to val_ce in meta.json, so
        # every checkpoint carries a train/val PAIR (the only way to tell overfitting from
        # under-training from the artifacts alone).
        self._train_loss_sum = 0.0
        self._train_loss_n = 0
        self._tokens_span_global = self._tokens_span_prev  # refreshed at each validation
        self._tokens_sup_global = self._tokens_sup_prev
        self._t0: float | None = None
        self._exclude_mask_cache: Tensor | None = None

    def _val_micro_batch(self) -> int:
        return self._cfg.val_micro_batch or self._cfg.micro_batch

    def create_model(self) -> torch.nn.Module:
        return self._model

    def configure_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self._model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self._cfg.lr)
        if self._resume_opt_path:
            # Param order is deterministic (same module traversal), so state maps 1:1;
            # load_state_dict casts state tensors to each param's device.
            opt.load_state_dict(torch.load(self._resume_opt_path, map_location="cpu"))
            print(f"[sft] optimizer state restored from {self._resume_opt_path}", flush=True)
        return opt

    def train_data(self) -> Dataset[dict[str, Tensor]]:
        return self._train_data

    def val_data(self) -> Dataset[dict[str, Tensor]]:
        return self._val_data

    def _configure_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt_lens = self._prompt_lens
        pad_id = self._pad_id
        args["collate_fn"] = lambda rows: soft_token_collate(
            rows, prompt_lens=prompt_lens, pad_id=pad_id
        )
        data: Any = args.get("dataset")
        is_train = data is self._train_data
        args["num_workers"] = self._cfg.num_workers if is_train else 0
        args["persistent_workers"] = bool(self._cfg.num_workers) and is_train
        # Train batches are sharded across ranks (each rank one epoch of disjoint batches, same
        # contract as train_ao); val is unsharded so val_ce is the same whole-set metric per rank.
        world = self.trainer.config.world_size() if (is_train and self.trainer) else 1
        rank = self.trainer.global_rank if (is_train and self.trainer) else 0
        exclude = None
        if is_train and self._cfg.exclude_idx_file:
            if self._exclude_mask_cache is None:
                n_train = len(data) if isinstance(data, Sized) else 0
                self._exclude_mask_cache = load_exclude_mask(self._cfg, n_train)
            exclude = self._exclude_mask_cache
        sampler = LayerBucketBatchSampler(
            group_indices(data),
            micro_batch=self._cfg.micro_batch if is_train else self._val_micro_batch(),
            world=world,
            rank=rank,
            seed=self._cfg.seed,
            shuffle=is_train,
            skip_batches=self._cfg.skip_batches if is_train else 0,
            deal_key=deal_key_for(data) if is_train else None,
            exclude_mask=exclude,
        )
        args["batch_sampler"] = sampler
        # batch_sampler is exclusive with these DataLoader args — the harness sets them by default
        for k in ("batch_size", "shuffle", "sampler", "drop_last"):
            args.pop(k, None)
        return args

    def configure_training_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._configure_dl(args)

    def configure_validation_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._configure_dl(args)

    def _ce(self, batch: dict[str, Tensor]) -> Tensor:
        device = batch["inject_vec"].device
        b = batch["inject_vec"].shape[0]
        layer_idx = int(batch["layer_idx"])
        prompt = self._prompts[layer_idx]
        embed: Any = self._model.get_input_embeddings()  # type: ignore[operator]
        prompt_ids = self._prompt_ids[layer_idx].to(device).unsqueeze(0).expand(b, -1)
        # clone: input-require-grads marks embed output a leaf; in-place writes forbidden
        prompt_embeds = embed(prompt_ids).clone()
        v = inject_gt(
            batch["inject_vec"],
            self._cfg.transform,
            alpha=self._cfg.alpha,
            scale=self._scales[layer_idx],
            clip_mult=self._cfg.clip_mult,
            space=self._spaces[layer_idx],
        )
        prompt_embeds[:, prompt.slot, :] = v.to(prompt_embeds.dtype)
        target_embeds = embed(batch["target_ids"])
        inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
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
        if self._t0 is None:
            self._t0 = time.monotonic()  # from the first step: excludes load/compile time
        loss = self._ce(batch)
        self._train_loss_sum += float(loss.detach())
        self._train_loss_n += 1
        self._tokens_span_rank += int(batch["n_span_tokens"])
        self._tokens_sup_rank += int(batch["n_sup_tokens"])
        dt = max(time.monotonic() - self._t0, 1e-9)
        self.log(
            "span_tokens_per_sec_rank",
            torch.tensor(self._tokens_span_rank / dt),
            across_devices=False,
        )
        self._maybe_preempt_exit(batch["inject_vec"].device)
        return loss

    def _world(self) -> int:
        return int(self.trainer.config.world_size()) if self.trainer is not None else 1

    def _sync_token_totals(self, device: torch.device) -> None:
        """Exact global token totals: all ranks participate (call from a collective context)."""
        t = torch.tensor(
            [self._tokens_span_rank, self._tokens_sup_rank], dtype=torch.long, device=device
        )
        if self._world() > 1 and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self._tokens_span_global = self._tokens_span_prev + int(t[0])
        self._tokens_sup_global = self._tokens_sup_prev + int(t[1])

    def _maybe_preempt_exit(self, device: torch.device) -> None:
        """Collective SIGUSR1 check; on agreement: save resume state on rank 0 and exit cleanly."""
        flag = torch.tensor(1.0 if _PREEMPT_FLAG["set"] else 0.0, device=device)
        if self._world() > 1 and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if float(flag) == 0.0:
            return
        self._sync_token_totals(device)
        self._save_resume_state()
        if self._world() > 1 and dist.is_initialized():
            dist.barrier()
        print("[sft] preempt save complete — exiting", flush=True)
        raise SystemExit(0)

    def validation_step(self, batch: dict[str, Tensor], batch_info: Any) -> None:
        loss = self._ce(batch).detach().cpu()
        self._val_losses.append(loss)
        self._val_layer_losses.setdefault(int(batch["layer_idx"]), []).append(loss)
        lens = batch.get("span_lens")
        if lens is not None and int(lens.min()) == int(lens.max()):
            self._val_len_losses.setdefault(int(lens[0]), []).append(loss)
        # trigger count = the val sampler's batch count (val is unsharded: world=1 on every rank,
        # layer-pure FULL batches only — partial batches are dropped by the sampler)
        sampler = LayerBucketBatchSampler(
            group_indices(self._val_data),
            micro_batch=self._val_micro_batch(),
            shuffle=False,
        )
        if len(self._val_losses) >= max(1, len(sampler)):
            self._sync_token_totals(batch["inject_vec"].device)
            val_ce = float(torch.stack(self._val_losses).mean())
            self.log("val_ce", torch.tensor(val_ce), across_devices=False)
            metrics = {"val_ce": val_ce}
            for ly_idx, losses in sorted(self._val_layer_losses.items()):
                layer = self._cfg.layers[ly_idx]
                ce = float(torch.stack(losses).mean())
                metrics[f"val_ce_L{layer}"] = ce
                self.log(f"val_ce_L{layer}", torch.tensor(ce), across_devices=False)
            for n, losses in sorted(self._val_len_losses.items()):
                metrics[f"val_ce_N{n}"] = float(torch.stack(losses).mean())
            if self._train_loss_n:
                metrics["train_ce"] = self._train_loss_sum / self._train_loss_n
                metrics["train_val_gap"] = metrics["val_ce"] - metrics["train_ce"]
                # Log the MEAN train loss beside each val point. Per-batch train loss is
                # unusable for comparison: batches are length-pure, so a single reading swings
                # 0.33-3.06 purely with the crop length it drew. NOTE the harness also emits a
                # stray `val_epoch/loss` — that is a train-step metric its accumulator had
                # pending when validation began, NOT validation loss. Read val_ce / train_ce.
                for name in ("train_ce", "train_val_gap"):
                    self.log(name, torch.tensor(metrics[name]), across_devices=False)
            metrics["tokens_span"] = float(self._tokens_span_global)
            metrics["tokens_sup"] = float(self._tokens_sup_global)
            tspan = torch.tensor(float(self._tokens_span_global))
            self.log("tokens_span", tspan, across_devices=False)
            print(f"[sft] {metrics}", flush=True)
            self.last_val_metrics = metrics
            self._val_losses = []
            self._val_layer_losses = {}
            self._val_len_losses = {}
            self._train_loss_sum, self._train_loss_n = 0.0, 0
            self._save_checkpoint()
            self._save_resume_state()

    def _micro_step(self) -> int:
        """Micro-steps consumed across ALL segments (offset survives resume)."""
        cur = int(getattr(self.trainer, "global_step", 0)) if self.trainer is not None else 0
        return self._step_offset + cur

    def _save_checkpoint(self) -> None:
        """Rank-0 LoRA checkpoint at each validation -> ckpt_dir/step{N}/{lora, meta.json}.

        ``meta.json`` carries the EXACT recorded token totals — the scaling-plot x-axis reads
        this, never a per-step estimate.
        """
        if self._ckpt_dir is None:
            return
        rank = self.trainer.global_rank if self.trainer is not None else 0
        if rank != 0:
            return
        step = self._micro_step()
        out = self._ckpt_dir / f"step{step}"
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out / "lora"))  # type: ignore[operator]
        meta = {
            "micro_steps": step,
            "tokens_span": self._tokens_span_global,
            "tokens_sup": self._tokens_sup_global,
            **self.last_val_metrics,
            "config": self._cfg.to_dict(),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[sft] checkpoint saved: step {step}", flush=True)

    def _save_resume_state(self) -> None:
        """Atomic rank-0 resume bundle: {lora/, optimizer.pt, state.json} (~2-3 GB).

        Deliberately NOT the harness ``save_every_n_steps`` (that serializes the whole ~54 GB
        PEFT-wrapped model). ``.tmp`` -> ``os.replace`` so a preemption mid-write never corrupts
        the previous good state.
        """
        if self._ckpt_dir is None or self.trainer is None:
            return
        if self.trainer.global_rank != 0:
            return
        base = Path(str(self._ckpt_dir))
        base.mkdir(parents=True, exist_ok=True)
        tmp = base / "resume.tmp"
        if tmp.exists():
            import shutil

            shutil.rmtree(tmp)
        tmp.mkdir()
        self._model.save_pretrained(str(tmp / "lora"))  # type: ignore[operator]
        opt = getattr(self.trainer, "optimizer", None)
        if opt is not None:
            torch.save(opt.state_dict(), tmp / "optimizer.pt")
        state = {
            "micro_steps": self._micro_step(),
            "tokens_span": self._tokens_span_global,
            "tokens_sup": self._tokens_sup_global,
            "world_size": self._world(),
            "config": self._cfg.to_dict(),
        }
        (tmp / "state.json").write_text(json.dumps(state, indent=2))
        final = base / "resume"
        if final.exists():
            import shutil

            shutil.rmtree(final)
        os.replace(tmp, final)
        print(f"[sft] resume state saved at micro-step {state['micro_steps']}", flush=True)

    def hyperparams_to_log(self) -> dict[str, object]:
        return self._cfg.to_dict()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def train_soft_token(
    model: torch.nn.Module,
    prompts: list[WVPrompt],
    train_data: Dataset[dict[str, Tensor]],
    val_data: Dataset[dict[str, Tensor]],
    cfg: SoftTokenConfig,
    *,
    pad_id: int,
    wandb_project: str | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    local_gpus: int = 0,
    node_rank: int = 0,
    master_addr: str = "localhost",
    ckpt_dir: Any = None,
    resume_state: dict[str, Any] | None = None,
    spaces: list[Any] | None = None,
) -> dict[str, float]:
    """One GT-continuation run; ``world_size > 1`` runs ONE DDP rank in this process.

    Same global-batch contract as ``train_ao``: grad_accum is divided by world_size (when it
    divides) so LR stays valid as GPUs scale; steps are counted from the per-rank BATCH shard
    (the batch sampler splits whole layer-pure batches, not rows).

    ``world_size`` is the GLOBAL rank count. For multi-node, also pass ``local_gpus`` (GPUs per
    node), ``node_rank`` and ``master_addr``: the harness derives
    ``global_rank = node_rank * ngpu + local_rank`` and ``world_size() = ngpu * nodes``, so the
    batch sampler's rank sharding and the token all_reduce stay globally correct.
    """
    mydule = SoftTokenMydule(
        model,
        prompts,
        cfg,
        train_data,
        val_data,
        pad_id,
        spaces=spaces,
        ckpt_dir=ckpt_dir,
        resume_state=resume_state,
    )
    world_size = max(1, world_size)
    grad_accum = max(1, cfg.grad_accum // world_size) if world_size > 1 else cfg.grad_accum
    n_train_ex = len(train_data) if isinstance(train_data, Sized) else 0
    per_rank_batches = LayerBucketBatchSampler(
        group_indices(train_data),
        micro_batch=cfg.micro_batch,
        world=world_size,
        skip_batches=cfg.skip_batches,
        deal_key=deal_key_for(train_data),
        exclude_mask=load_exclude_mask(cfg, n_train_ex),
    )
    val_mb = cfg.val_micro_batch or cfg.micro_batch
    n_val_batches = len(
        LayerBucketBatchSampler(group_indices(val_data), micro_batch=val_mb, shuffle=False)
    )
    if n_val_batches == 0:
        raise SystemExit(
            f"[sft] val set yields ZERO batches at val_micro_batch={val_mb}: the sampler drops "
            f"partial (layer[,length])-pure batches, so no group reaches {val_mb} examples. "
            f"Lower val_micro_batch or enlarge the val set — otherwise validation (and every "
            f"checkpoint) silently never fires."
        )
    n_val_ex = len(val_data) if isinstance(val_data, Sized) else -1
    print(f"[sft] val: {n_val_batches} batches of {val_mb} ({n_val_ex} examples)", flush=True)

    tc = TrainingConfig()
    tc.name = cfg.run_name
    tc.max_epochs = cfg.epochs
    steps = max(1, cfg.epochs * len(per_rank_batches) // grad_accum)
    tc.max_steps = steps
    tc.ngpu = local_gpus or world_size
    tc.nodes = max(1, world_size // (local_gpus or world_size))
    tc.node_rank = node_rank
    tc.master_addr = master_addr
    tc.train_batch_size = cfg.micro_batch
    tc.val_batch_size = val_mb
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
    # entry.py overwrites $MASTER_PORT from this config field (default 10210) — read the per-job
    # port env.sh derived from SLURM_JOB_ID, else two jobs on one node collide.
    tc.master_port = int(os.environ.get("MASTER_PORT", "10210"))
    # user-scoped: a bare /tmp/mtl-runs collides with other users' dirs on shared nodes
    tc.base_save_dir = f"/tmp/mtl-runs-{os.environ.get('USER', 'unknown')}"
    if wandb_project is not None:
        tc.logger = "wandb"
        tc.wandb_config.project = wandb_project
    tc.finalize()
    if world_size > 1:
        do_train(tc, local_rank, mydule)
    else:
        mtl_train(tc, mydule)
    return dict(mydule.last_val_metrics)
