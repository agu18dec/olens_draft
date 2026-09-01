"""GT-continuation lens: supervised (true activation -> the text that followed it).

The AO grid's ``real_h`` control arm promoted to a first-class lens
(``docs/project/experiments/gt_continuation_lens/design.md``). Differences from ``ao_train``:

- **Source is always the true stored residual** (``MultiLayerPairs.targets[row, layer]``); the
  swept knob is the *transform* (``inject.GtTransform``), with ``scaled`` scales fit PER LAYER.
- **All 17 dumped layers, one model.** The prompt names the layer, so there is one rendered
  prompt per layer and every micro-batch must be layer-pure: ``LayerBucketBatchSampler`` shards
  whole single-layer batches across DDP ranks (rank sharding over batches keeps the global batch
  unchanged as GPUs scale, same contract as ``train_ao``).
- **Targets are fetched lazily** (row index kept, vector read in ``__getitem__``) so the > RAM
  pool trains through ``LazyTargets`` mmaps without a materialization pass.

Target construction is byte-identical to AO-1: ``<explanation>\\n`` + span ids +
``\\n</explanation>`` + EOS, masked-CE on target tokens only — the span IS the continuation
under the span convention (activation at ``prev_pos = start - 1``).
"""

import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import torch
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import do_train
from mytorch_lightning.entry import train as mtl_train
from mytorch_lightning.mydule import Mydule
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from oracle_lens.pipeline.inject import SCALE_FREE_GT, GtTransform, fit_scale, inject_gt
from oracle_lens.pipeline.multilayer import MultiLayerPairs
from oracle_lens.pipeline.shards import LENGTH_BUCKETS
from oracle_lens.pipeline.verbalizer import WVPrompt


@dataclass
class GTConfig:
    """One GT-continuation run. ``scales`` is keyed by ``str(layer)`` (JSON round-trip safe)."""

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
    final_lr: float = 0.0  # floor for decaying schedules (linear/cosine)
    micro_batch: int = 4
    grad_accum: int = 8
    epochs: int = 1
    eval_every_steps: int = 200
    max_eval_rows: int = 1024
    warmup_steps: int = 20
    seed: int = 0
    init_from: str = ""  # "run/stepN" to resume the LoRA from a saved checkpoint (else fresh)
    skip_examples: int = 0  # drop the first N of the seeded permutation (resume onto UNSEEN rows)
    # Long-sequence path (token_budget > 0): micro-batch ROWS vary per (layer, octave) group so
    # every micro-batch is ~token_budget padded tokens; 0 keeps the legacy fixed micro_batch.
    token_budget: int = 0
    mb_max: int = 64  # cap on rows per micro-batch under the token-budget path
    # Val-only token budget (0 = token_budget). The val sampler drops partial batches like the
    # train one, so a small val set spread over (layer, octave) groups can starve to ZERO
    # batches at the train budget (distill r1: 41 rows / ~30 groups / 22-row batches -> "0it").
    # A small val budget makes 1-2-row batches and keeps every group scored.
    val_token_budget: int = 0
    skip_batches: int = 0  # exact resume: skip the first K per-rank batches of the seeded order
    # In-loop FVE probe (off when fve_ar_ckpt == ""): every fve_every_evals-th validation flush,
    # rank 0 generates from fve_n stratified val rows at fve_layers, reconstructs the generations
    # through the FROZEN AR (mounted as a second, frozen PEFT adapter on the same base model),
    # and logs whitened cos^2 vs the injected activation as val_fve / val_fve_L{n}.
    fve_ar_ckpt: str = ""  # dir with lora/ + heads.pt (the converted AR layout)
    fve_n: int = 64  # probe rows per layer, drawn stratified over octaves from the val set
    fve_layers: tuple[int, ...] = (44,)
    fve_every_evals: int = 4
    fve_batch: int = 8
    fve_ridge_c: float = 0.1
    # DataLoader workers for the batch-sampler path (0 = legacy in-thread loading; profiling
    # 2026-07-28 showed workers hide the LazyTargets NFS latency for a free ~2%)
    num_workers: int = 0
    # length-sorted greedy batch packing (see LayerLengthBucketBatchSampler): ~25% less padding
    # waste in wide octaves; off by default so pre-2026-07-28 runs replay identically
    length_packed: bool = False
    # One row per (conv_index, prev_pos): overlapping-span assembly leaves ~5% of rows sharing
    # their source activation with a sibling row at another continuation length, and a single
    # epoch must not visit the same activation twice (user rule 2026-07-29). First occurrence
    # in the seeded permutation wins (an unbiased draw among the competing lengths); off by
    # default so pre-2026-07-29 runs replay identically.
    dedup_activations: bool = False
    # per-block torch.compile(dynamic=True) — the AR line's compile_blocks adapted to variable
    # shapes; validate speed+parity at mini scale before any main-run adoption
    compile_blocks: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GTConfig":
        d = dict(d)
        d["layers"] = tuple(d.get("layers", (44,)))
        d["fve_layers"] = tuple(d.get("fve_layers", (44,)))
        return cls(**d)

    def scale_for(self, layer: int) -> float:
        # Scale-free arms fix the injected norm at alpha by unit-normalising, so they carry no
        # per-layer fit_scale entry; indexing `scales` for them would KeyError.
        if self.transform in SCALE_FREE_GT:
            return 1.0
        return self.scales[str(layer)]


def fit_layer_scales(
    pairs: MultiLayerPairs,
    layers: tuple[int, ...],
    *,
    alpha: float,
    sample_n: int = 4096,
    seed: int = 0,
) -> dict[str, float]:
    """Per-layer ``fit_scale`` on a seeded row sample (LazyTargets-friendly: sample_n row reads)."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pairs), generator=gen)[: min(sample_n, len(pairs))]
    rows = torch.stack([torch.as_tensor(pairs.targets[int(i)]) for i in idx.tolist()]).float()
    out: dict[str, float] = {}
    for layer in layers:
        pos = pairs.layers.index(layer)
        out[str(layer)] = fit_scale(rows[:, pos, :], target_norm=alpha)
    return out


def build_gt_examples(
    pairs: MultiLayerPairs, tokenizer: Any, cfg: GTConfig
) -> list[dict[str, Tensor]]:
    """``(row, layer_idx, target_ids)`` examples; the activation is fetched lazily at load time.

    One example per selected row, layer drawn uniformly from ``cfg.layers`` (seeded) so
    ``n_examples`` semantics match AO-1. The same seed/min_len/max_len makes the permutation
    identical across runs, so ``skip_examples`` resumes onto provably unseen rows (AO contract).
    """
    for layer in cfg.layers:
        if layer not in pairs.layers:
            raise ValueError(f"layer {layer} not in shard layers {pairs.layers}")
    open_ids = tokenizer("<explanation>\n", add_special_tokens=False)["input_ids"]
    close_ids = tokenizer("\n</explanation>", add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    lens = pairs.lengths
    keep = torch.nonzero((lens >= cfg.min_len) & (lens <= cfg.max_len)).squeeze(-1)
    gen = torch.Generator().manual_seed(cfg.seed)
    perm = keep[torch.randperm(len(keep), generator=gen)]
    if cfg.dedup_activations:
        stride = int(pairs.prev_pos.max()) + 1
        key = pairs.conv_index[perm].to(torch.int64) * stride + pairs.prev_pos[perm].to(torch.int64)
        _, inv = torch.unique(key, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), len(perm), dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(len(perm)), reduce="amin")
        mask = torch.arange(len(perm)) == first[inv]
        print(
            f"[gt] dedup_activations: dropped {int((~mask).sum())}/{len(perm)} rows sharing "
            "(conv, prev_pos) with an earlier-permuted sibling",
            flush=True,
        )
        perm = perm[mask]
    order = perm[cfg.skip_examples : cfg.skip_examples + cfg.n_examples]
    layer_draw = torch.randint(len(cfg.layers), (len(order),), generator=gen)
    examples: list[dict[str, Tensor]] = []
    for j, i in enumerate(order.tolist()):
        span = pairs.row_ids(i).tolist()
        target = [*open_ids, *span, *close_ids, eos]
        examples.append(
            {
                "row": torch.tensor(i, dtype=torch.long),
                "layer_idx": torch.tensor(int(layer_draw[j]), dtype=torch.long),
                "target_ids": torch.tensor(target, dtype=torch.long),
                "n_span": torch.tensor(len(span), dtype=torch.long),
            }
        )
    return examples


class GTDataset(Dataset[dict[str, Tensor]]):
    """Fetches the injected vector lazily: one ``targets[row]`` mmap read per item."""

    def __init__(
        self, pairs: MultiLayerPairs, examples: list[dict[str, Tensor]], layers: tuple[int, ...]
    ) -> None:
        self.pairs = pairs
        self.examples = examples
        self.layer_pos = [pairs.layers.index(ly) for ly in layers]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        ex = self.examples[i]
        row = int(ex["row"])
        layer_idx = int(ex["layer_idx"])
        vec = torch.as_tensor(self.pairs.targets[row])[self.layer_pos[layer_idx]].float()
        return {
            "inject_vec": vec,
            "layer_idx": ex["layer_idx"],
            "target_ids": ex["target_ids"],
            "n_span": ex["n_span"],
        }


def build_distill_examples(dpairs: Any, cfg: GTConfig) -> list[dict[str, Tensor]]:
    """Round-2 examples from ``distill_v1`` rows: VERBATIM stored targets, per-row layer.

    Differences from :func:`build_gt_examples`: the target ids come fully formatted from the
    shard (k ``<explanation>`` blocks + EOS — nothing is re-wrapped), the layer is the row's
    stored assignment (mapped into ``cfg.layers`` index space; rows at layers outside
    ``cfg.layers`` are dropped, so an L44-only smoke just works), and ``n_span`` is the TOTAL
    target length (wrapper overhead is 0 by construction — the token-budget sampler then
    reads ``overhead = prompt_len``). Same seeded permutation + ``skip_examples``/``n_examples``
    window as AO/GT so resume semantics carry over.
    """
    from oracle_lens.pipeline.multilayer import LAYERS

    cfg_pos_of: dict[int, int] = {ly: i for i, ly in enumerate(cfg.layers)}
    lens = dpairs.lengths
    keep = [
        i
        for i in range(len(dpairs))
        if LAYERS[int(dpairs.layer_idx[i])] in cfg_pos_of
        and cfg.min_len <= int(lens[i]) <= cfg.max_len
    ]
    gen = torch.Generator().manual_seed(cfg.seed)
    keep_t = torch.tensor(keep, dtype=torch.long)
    perm = keep_t[torch.randperm(len(keep_t), generator=gen)]
    order = perm[cfg.skip_examples : cfg.skip_examples + cfg.n_examples]
    examples: list[dict[str, Tensor]] = []
    for i in order.tolist():
        target = dpairs.row_target(i)
        examples.append(
            {
                "row": torch.tensor(i, dtype=torch.long),
                "layer_idx": torch.tensor(
                    cfg_pos_of[LAYERS[int(dpairs.layer_idx[i])]], dtype=torch.long
                ),
                "target_ids": target,
                "n_span": torch.tensor(int(target.shape[0]), dtype=torch.long),
            }
        )
    return examples


class DistillDataset(Dataset[dict[str, Tensor]]):
    """Round-2 dataset: the injection vector is the row's single stored ``vec`` (no layer stack)."""

    def __init__(self, dpairs: Any, examples: list[dict[str, Tensor]]) -> None:
        self.pairs = dpairs
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        ex = self.examples[i]
        return {
            "inject_vec": self.pairs.vec[int(ex["row"])].float(),
            "layer_idx": ex["layer_idx"],
            "target_ids": ex["target_ids"],
            "n_span": ex["n_span"],
        }


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
    ) -> None:
        self.groups: dict[int, list[int]] = {}
        for i, ly in enumerate(layer_idx_per_example):
            self.groups.setdefault(ly, []).append(i)
        self.micro_batch = micro_batch
        self.world = max(1, world)
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        n_batches = sum(len(g) // micro_batch for g in self.groups.values())
        self._per_rank = n_batches // self.world

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._per_rank

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        batches: list[list[int]] = []
        for ly in sorted(self.groups):
            idx = torch.tensor(self.groups[ly], dtype=torch.long)
            if self.shuffle:
                idx = idx[torch.randperm(len(idx), generator=gen)]
            n_full = len(idx) // self.micro_batch
            for b in range(n_full):
                batches.append(idx[b * self.micro_batch : (b + 1) * self.micro_batch].tolist())
        if self.shuffle:
            order = torch.randperm(len(batches), generator=gen).tolist()
            batches = [batches[i] for i in order]
        for b in range(self._per_rank):
            yield batches[b * self.world + self.rank]


def bucket_index_of(n: int, buckets: tuple[tuple[int, int], ...]) -> int:
    """Index of the length bucket containing ``n`` (raises if outside every bucket)."""
    for bi, (lo, hi) in enumerate(buckets):
        if lo <= n <= hi:
            return bi
    raise ValueError(f"length {n} outside all buckets {buckets}")


def buckets_for(cfg: "GTConfig") -> tuple[tuple[int, int], ...]:
    """The length buckets for ``cfg`` — octaves, extended past 1024 when ``max_len`` needs it.

    Round-2 distill targets are whole formatted LISTS (k blocks + wrapper), so a row can exceed
    the 1..1024 octave ladder (k4n256 ≈ 1100 tokens); such runs set ``max_len`` accordingly and
    get one extra bucket per doubling. Legacy configs (``max_len <= 1024``) are bit-identical to
    ``LENGTH_BUCKETS``.
    """
    buckets = LENGTH_BUCKETS
    hi = buckets[-1][1]
    while hi < cfg.max_len:
        buckets = (*buckets, (hi + 1, hi * 2))
        hi *= 2
    return buckets


class LayerLengthBucketBatchSampler(Sampler[list[int]]):
    """Micro-batches pure in BOTH layer and length octave, sized to a token budget.

    The long path's replacement for ``LayerBucketBatchSampler``: at max_len 1024 a fixed row
    count either OOMs on the long octaves or wastes the short ones, so rows per micro-batch are
    ``clamp(token_budget // (overhead_per_layer[layer] + bucket_hi), 1, mb_max)`` — every batch
    lands near ``token_budget`` padded tokens. Same DDP contract as the layer sampler (whole
    batches sharded ``batches[b * world + rank]``, partial batches dropped, seeded per-epoch
    shuffle); ``skip_batches`` drops the first K per-rank batches of the seeded order for exact
    preemption resume (the order is deterministic under (seed, epoch, world), so the suffix is
    provably the unseen remainder).
    """

    def __init__(
        self,
        layer_idx_per_example: list[int],
        span_len_per_example: list[int],
        *,
        overhead_per_layer: list[int],  # prompt tokens + <explanation> wrapper + eos, per layer
        token_budget: int,
        mb_max: int = 64,
        buckets: tuple[tuple[int, int], ...] = LENGTH_BUCKETS,
        world: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        skip_batches: int = 0,
        length_packed: bool = False,
    ) -> None:
        if token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {token_budget}")
        if len(layer_idx_per_example) != len(span_len_per_example):
            raise ValueError("layer_idx and span_len must align")
        self.groups: dict[tuple[int, int], list[int]] = {}
        for i, (ly, n) in enumerate(zip(layer_idx_per_example, span_len_per_example, strict=True)):
            self.groups.setdefault((ly, bucket_index_of(n, buckets)), []).append(i)
        self.buckets = buckets
        self.overhead_per_layer = overhead_per_layer
        self.token_budget = token_budget
        self.mb_max = mb_max
        self.world = max(1, world)
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.skip_batches = skip_batches
        self.length_packed = length_packed
        self._span_lens = span_len_per_example
        self.epoch = 0
        if length_packed:
            # exact count via a dry pack (deterministic; shuffle only permutes batch ORDER)
            n_batches = sum(1 for _ in self._packed_batches())
        else:
            n_batches = sum(
                len(g) // self.rows_for_group(ly, bi) for (ly, bi), g in self.groups.items()
            )
        self._per_rank = n_batches // self.world
        if skip_batches >= max(1, self._per_rank):
            raise ValueError(f"skip_batches {skip_batches} >= per-rank batches {self._per_rank}")

    def _packed_batches(self) -> Iterator[list[int]]:
        """Length-sorted greedy packing per (layer, octave) group (full batches only)."""
        for ly, bi in sorted(self.groups):
            order_in_group = sorted(self.groups[(ly, bi)], key=lambda i: (self._span_lens[i], i))
            overhead = self.overhead_per_layer[ly]
            cur: list[int] = []
            cur_max = 0
            for i in order_in_group:
                seq = overhead + self._span_lens[i]
                new_max = max(cur_max, seq)
                full = (len(cur) + 1) * new_max > self.token_budget or len(cur) >= self.mb_max
                if cur and full:
                    yield cur
                    cur, cur_max = [], 0
                    new_max = seq
                cur.append(i)
                cur_max = new_max
            # trailing partial dropped, mirroring the fixed-mb path's contract

    def rows_for_group(self, layer_idx: int, bucket_idx: int) -> int:
        """Rows per micro-batch for a (layer, octave) group under the token budget."""
        seq = self.overhead_per_layer[layer_idx] + self.buckets[bucket_idx][1]
        return max(1, min(self.mb_max, self.token_budget // seq))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._per_rank - self.skip_batches

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        batches: list[list[int]] = []
        if self.length_packed:
            # Length-sorted greedy packing: batches pad to their OWN max, not bucket_hi, and row
            # counts use actual lengths — kills the ~25% padding waste the fixed-mb path pays in
            # wide octaves (measured on the 2026-07-28 main run). Deterministic under (seed,
            # epoch): the sort is stable and only batch ORDER is shuffled.
            batches = list(self._packed_batches())
        else:
            for ly, bi in sorted(self.groups):
                idx = torch.tensor(self.groups[(ly, bi)], dtype=torch.long)
                if self.shuffle:
                    idx = idx[torch.randperm(len(idx), generator=gen)]
                mb = self.rows_for_group(ly, bi)
                for b in range(len(idx) // mb):
                    batches.append(idx[b * mb : (b + 1) * mb].tolist())
        if self.shuffle:
            order = torch.randperm(len(batches), generator=gen).tolist()
            batches = [batches[i] for i in order]
        n_usable = (len(batches) // self.world) * self.world
        per_rank = n_usable // self.world
        for b in range(self.skip_batches, per_rank):
            yield batches[b * self.world + self.rank]


def gt_collate(
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
    out = {
        "inject_vec": torch.stack([r["inject_vec"] for r in rows]),
        "layer_idx": torch.tensor(layer_idx, dtype=torch.long),
        "target_ids": target_ids,
        "labels": labels,
        "attention_mask": attn,
    }
    if "n_span" in rows[0]:  # absent only for legacy callers; required on the token-budget path
        out["n_span"] = torch.stack([r["n_span"] for r in rows])
    return out


def _val_sampler_cfg(cfg: GTConfig) -> GTConfig:
    """The sampler config for VALIDATION: ``val_token_budget`` swaps in when set (see field)."""
    if cfg.val_token_budget <= 0 or cfg.token_budget <= 0:
        return cfg
    return replace(cfg, token_budget=cfg.val_token_budget)


def make_batch_sampler(
    cfg: GTConfig,
    examples: list[dict[str, Tensor]],
    prompt_lens: list[int],
    *,
    world: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    skip_batches: int = 0,
) -> "LayerBucketBatchSampler | LayerLengthBucketBatchSampler":
    """The batch sampler for a dataset under ``cfg`` — legacy fixed-rows or token-budgeted.

    Both the train/val dataloaders and the step-count/resume math construct the sampler through
    this ONE function so their batch orders provably agree.
    """
    layer_idx = [int(ex["layer_idx"]) for ex in examples]
    if cfg.token_budget <= 0:
        return LayerBucketBatchSampler(
            layer_idx,
            micro_batch=cfg.micro_batch,
            world=world,
            rank=rank,
            seed=cfg.seed,
            shuffle=shuffle,
        )
    span_lens = [int(ex["n_span"]) for ex in examples]
    # wrapper overhead (<explanation> tags + eos) is constant per run: recover it from row 0
    wrapper = (int(examples[0]["target_ids"].shape[0]) - span_lens[0]) if examples else 0
    overhead = [pl + wrapper for pl in prompt_lens]
    return LayerLengthBucketBatchSampler(
        layer_idx,
        span_lens,
        overhead_per_layer=overhead,
        token_budget=cfg.token_budget,
        mb_max=cfg.mb_max,
        buckets=buckets_for(cfg),
        world=world,
        rank=rank,
        seed=cfg.seed,
        shuffle=shuffle,
        skip_batches=skip_batches,
        length_packed=cfg.length_packed,
    )


class GTMydule(Mydule):  # type: ignore[misc]
    """Masked-CE SFT with ``inject_gt`` replacing the per-layer prompt's ``<concept>`` slot."""

    def __init__(
        self,
        model: torch.nn.Module,
        prompts: list[WVPrompt],
        cfg: GTConfig,
        train_data: "GTDataset | DistillDataset",
        val_data: "GTDataset | DistillDataset",
        pad_id: int,
        ckpt_dir: Any = None,
        tokenizer: Any = None,
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
        self._tokenizer = tokenizer
        self._val_flushes = 0
        self._fve_ctx: dict[str, Any] | None = None
        self._prompt_ids = [torch.tensor(p.input_ids, dtype=torch.long) for p in prompts]
        self._prompt_lens = [len(p.input_ids) for p in prompts]
        self._scales = [cfg.scale_for(ly) for ly in cfg.layers]
        # Per-layer ruler for the metric-space arms (wht_unit / j_unit), aligned with cfg.layers.
        # None entries mean "no ruler for this layer" (e.g. L63 has no Jacobian) — the arms that
        # need one raise on None rather than silently injecting an un-mapped vector.
        self._spaces: list[Any] = list(spaces) if spaces is not None else [None] * len(cfg.layers)
        if cfg.transform in SCALE_FREE_GT - {"unit"}:
            missing = [ly for ly, sp in zip(cfg.layers, self._spaces, strict=True) if sp is None]
            if missing:
                raise ValueError(
                    f"transform={cfg.transform} needs a metric space per layer; missing {missing}"
                    " (exclude those layers from cfg.layers — e.g. L63 has no Jacobian)"
                )
        self._val_losses: list[Tensor] = []
        self._val_layer_losses: dict[int, list[Tensor]] = {}
        self._val_bucket_losses: dict[int, list[Tensor]] = {}
        self.last_val_metrics: dict[str, float] = {}

    def create_model(self) -> torch.nn.Module:
        return self._model

    def configure_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self._model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self._cfg.lr)
        self._maybe_restore_optimizer(opt)
        return opt

    def _maybe_restore_optimizer(self, opt: torch.optim.Optimizer) -> None:
        """Restore Adam moments + schedule position from ``optim.pt`` beside the init LoRA.

        Checkpoints made before optimizer saving existed have no ``optim.pt``; resume then
        falls back to the old weights-only behaviour (fresh moments — pair with a re-warmup).
        Restoring ``trainer.global_step`` keeps step-dependent LR schedules (warmup/cosine)
        and checkpoint numbering continuous across restarts.
        """
        if not self._cfg.init_from or self._ckpt_dir is None:
            return
        path = self._ckpt_dir.parent / self._cfg.init_from / "optim.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        opt.load_state_dict(state["optimizer"])
        if self.trainer is not None:
            self.trainer.global_step = int(state["global_step"])
            # A restored odd global_step means the first batch is not a zero-grad step, so the
            # trainer never stamps step_start_time before the first log commit (same guard as
            # the trainer's own dcp load path).
            self.trainer.step_start_time = time.time()
        print(
            f"[gt] restored optimizer state + global_step={int(state['global_step'])} from {path}",
            flush=True,
        )

    def train_data(self) -> "GTDataset | DistillDataset":
        return self._train_data

    def val_data(self) -> "GTDataset | DistillDataset":
        return self._val_data

    def _configure_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt_lens = self._prompt_lens
        pad_id = self._pad_id
        args["collate_fn"] = lambda rows: gt_collate(rows, prompt_lens=prompt_lens, pad_id=pad_id)
        args["num_workers"] = self._cfg.num_workers
        args["persistent_workers"] = self._cfg.num_workers > 0
        if self._cfg.num_workers > 0:
            args["prefetch_factor"] = 4
        data: Any = args.get("dataset")
        is_train = data is self._train_data
        # Train batches are sharded across ranks (each rank one epoch of disjoint batches, same
        # contract as train_ao); val is unsharded so val_ce is the same whole-set metric per rank.
        world = self.trainer.config.world_size() if (is_train and self.trainer) else 1
        rank = self.trainer.global_rank if (is_train and self.trainer) else 0
        sampler = make_batch_sampler(
            self._cfg if is_train else _val_sampler_cfg(self._cfg),
            data.examples,
            self._prompt_lens,
            world=world,
            rank=rank,
            shuffle=is_train,
            skip_batches=self._cfg.skip_batches if is_train else 0,
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
        return self._ce(batch)

    def validation_step(self, batch: dict[str, Tensor], batch_info: Any) -> None:
        loss = self._ce(batch).detach().cpu()
        self._val_losses.append(loss)
        self._val_layer_losses.setdefault(int(batch["layer_idx"]), []).append(loss)
        if self._cfg.token_budget > 0:
            # token-budget batches are octave-pure, so any row's bucket is the batch's bucket
            bucket = bucket_index_of(int(batch["n_span"].max()), buckets_for(self._cfg))
            self._val_bucket_losses.setdefault(bucket, []).append(loss)
        # trigger count = the val sampler's batch count (val is unsharded: world=1 on every rank,
        # layer-pure FULL batches only — partial batches are dropped by the sampler)
        sampler = make_batch_sampler(
            _val_sampler_cfg(self._cfg), self._val_data.examples, self._prompt_lens,
            shuffle=False,
        )
        if len(self._val_losses) >= max(1, len(sampler)):
            val_ce = float(torch.stack(self._val_losses).mean())
            self.log("val_ce", torch.tensor(val_ce), across_devices=False)
            metrics = {"val_ce": val_ce}
            for ly_idx, losses in sorted(self._val_layer_losses.items()):
                layer = self._cfg.layers[ly_idx]
                ce = float(torch.stack(losses).mean())
                metrics[f"val_ce_L{layer}"] = ce
                self.log(f"val_ce_L{layer}", torch.tensor(ce), across_devices=False)
            for bucket, losses in sorted(self._val_bucket_losses.items()):
                lo, hi = buckets_for(self._cfg)[bucket]
                ce = float(torch.stack(losses).mean())
                metrics[f"val_ce_N{lo}-{hi}"] = ce
                self.log(f"val_ce_N{lo}-{hi}", torch.tensor(ce), across_devices=False)
            rank = self.trainer.global_rank if self.trainer is not None else 0
            self._val_flushes += 1
            if (
                self._cfg.fve_ar_ckpt
                and rank == 0
                and self._tokenizer is not None
                and self._val_flushes % max(1, self._cfg.fve_every_evals) == 1
            ):
                try:
                    fve_metrics = self._run_fve_probe()
                    metrics.update(fve_metrics)
                    for k, v in fve_metrics.items():
                        self.log(k, torch.tensor(v), across_devices=False)
                except Exception as e:  # a metric probe must never kill training
                    print(f"[gt] WARN fve probe failed: {type(e).__name__}: {e}", flush=True)
            print(f"[gt] {metrics}", flush=True)
            self.last_val_metrics = metrics
            self._val_losses = []
            self._val_layer_losses = {}
            self._val_bucket_losses = {}
            self._save_checkpoint()

    def _fve_setup(self) -> dict[str, Any]:
        """One-time probe setup: AR as a second frozen adapter + heads + whiteners + probe rows.

        The AR LoRA targets the same module paths as the oracle's (both are all-linear over the
        Qwen blocks), so it mounts on the SAME base model as adapter "ar" — no second 27B in
        memory. Its params are frozen (``is_trainable=False``) and were never registered with
        the optimizer (setup runs lazily, long after ``configure_optimizer``).
        """
        from pathlib import Path

        from oracle_lens.core.reconstructor import ReconstructorHead
        from oracle_lens.core.whitening import Whitener, whitening_matrix
        from oracle_lens.pipeline.multilayer import LAYERS

        cfg = self._cfg
        ar_dir = Path(cfg.fve_ar_ckpt)
        self._model.load_adapter(str(ar_dir / "lora"), adapter_name="ar")  # type: ignore[operator]
        state = torch.load(ar_dir / "heads.pt", map_location="cuda", weights_only=True)
        d = state["head"]["linear.weight"].shape[0]
        head = ReconstructorHead(d, layer_norm=True)
        head.load_state_dict(state["head"])
        emb = torch.nn.Embedding(len(LAYERS), d)
        emb.load_state_dict(state["layer_emb"])
        head, emb = head.to("cuda"), emb.to("cuda")
        head.requires_grad_(False)
        emb.requires_grad_(False)

        pairs = self._train_data.pairs
        gen = torch.Generator().manual_seed(cfg.seed + 7)
        sample_idx = torch.randperm(len(pairs), generator=gen)[:4096].tolist()
        whiteners: dict[int, Any] = {}
        for ly in cfg.fve_layers:
            pos = pairs.layers.index(ly)
            sample = torch.stack(
                [torch.as_tensor(pairs.targets[i])[pos] for i in sample_idx]
            ).float()
            mu = sample.mean(dim=0)
            xc = sample - mu
            cov = (xc.T @ xc) / (len(sample) - 1)
            whiteners[ly] = Whitener(
                mu=mu, w=whitening_matrix(cov, ridge_c=cfg.fve_ridge_c), ridge_c=cfg.fve_ridge_c
            )
        # stratified probe rows from the VAL pool, fixed for the whole run (comparable curve)
        vp = self._val_data.pairs
        lengths = vp.lengths
        rows_by_bucket: dict[int, list[int]] = {}
        per_bucket = max(1, cfg.fve_n // len(LENGTH_BUCKETS))
        for bi, (lo, hi) in enumerate(LENGTH_BUCKETS):
            members = torch.nonzero((lengths >= lo) & (lengths <= hi)).squeeze(-1)
            if len(members):
                take = members[torch.randperm(len(members), generator=gen)[:per_bucket]]
                rows_by_bucket[bi] = take.tolist()
        print(
            f"[gt] fve probe ready: {sum(len(v) for v in rows_by_bucket.values())} rows x "
            f"{list(cfg.fve_layers)} layers (ar={ar_dir.name})",
            flush=True,
        )
        return {"head": head, "emb": emb, "whiteners": whiteners, "rows": rows_by_bucket}

    def _run_fve_probe(self) -> dict[str, float]:
        """Generate -> frozen-AR reconstruction -> whitened cos^2, on the fixed probe rows."""
        from oracle_lens.pipeline.multilayer import LAYERS
        from oracle_lens.pipeline.scorer import bare_token_ids, prepare_reward_text

        if self._fve_ctx is None:
            self._fve_ctx = self._fve_setup()
        ctx = self._fve_ctx
        cfg = self._cfg
        model: Any = self._model
        tokenizer = self._tokenizer
        vp = self._val_data.pairs
        was_training = model.training
        model.eval()
        per_layer: dict[int, list[Tensor]] = {}
        try:
            with torch.no_grad():
                for ly in cfg.fve_layers:
                    li_cfg = cfg.layers.index(ly)
                    prompt = self._prompts[li_cfg]
                    prompt_ids = torch.tensor([prompt.input_ids], device="cuda")
                    pos = vp.layers.index(ly)
                    embed = model.get_input_embeddings()
                    for bi, rows in ctx["rows"].items():
                        max_new = LENGTH_BUCKETS[bi][1] + 16
                        for lo_i in range(0, len(rows), cfg.fve_batch):
                            chunk = rows[lo_i : lo_i + cfg.fve_batch]
                            acts = (
                                torch.stack([torch.as_tensor(vp.targets[i])[pos] for i in chunk])
                                .float()
                                .to("cuda")
                            )
                            b = acts.shape[0]
                            inp = embed(prompt_ids.expand(b, -1)).clone()
                            v = inject_gt(
                                acts,
                                cfg.transform,
                                alpha=cfg.alpha,
                                scale=cfg.scale_for(ly),
                                clip_mult=cfg.clip_mult,
                                space=self._spaces[li_cfg],
                            )
                            inp[:, prompt.slot, :] = v.to(inp.dtype)
                            out = model.generate(
                                inputs_embeds=inp,
                                attention_mask=torch.ones(
                                    b, inp.shape[1], dtype=torch.long, device="cuda"
                                ),
                                max_new_tokens=max_new,
                                do_sample=True,
                                temperature=1.0,
                                top_p=0.95,
                                use_cache=True,
                                pad_token_id=self._pad_id,
                            )
                            texts = [
                                prepare_reward_text(
                                    str(tokenizer.decode(r, skip_special_tokens=True))
                                )
                                for r in out
                            ]
                            # reconstruct through the frozen AR adapter on the same base model
                            model.set_adapter("ar")
                            try:
                                id_rows = [
                                    bare_token_ids(tokenizer, t) or [self._pad_id] for t in texts
                                ]
                                width = max(len(r) for r in id_rows)
                                ids = torch.full(
                                    (len(id_rows), width), self._pad_id, dtype=torch.long
                                )
                                mask = torch.zeros(len(id_rows), width, dtype=torch.long)
                                for j, r in enumerate(id_rows):
                                    ids[j, : len(r)] = torch.tensor(r)
                                    mask[j, : len(r)] = 1
                                fwd = model(
                                    input_ids=ids.to("cuda"),
                                    attention_mask=mask.to("cuda"),
                                    use_cache=False,
                                    output_hidden_states=True,
                                )
                                last = mask.to("cuda").sum(dim=1) - 1
                                rows_t = torch.arange(len(id_rows), device="cuda")
                                h = fwd.hidden_states[ly + 1][rows_t, last].float()
                                pred = ctx["head"](h + ctx["emb"].weight[LAYERS.index(ly)])
                            finally:
                                model.set_adapter("default")
                            w = ctx["whiteners"][ly]
                            cos = torch.nn.functional.cosine_similarity(
                                w.whiten(pred.cpu()), w.whiten(acts.cpu()), dim=-1
                            )
                            per_layer.setdefault(ly, []).append(cos**2)
        finally:
            model.set_adapter("default")
            if was_training:
                model.train()
        metrics: dict[str, float] = {}
        for ly, parts in per_layer.items():
            metrics[f"val_fve_L{ly}"] = float(torch.cat(parts).mean())
        metrics["val_fve"] = float(torch.cat([torch.cat(p) for p in per_layer.values()]).mean())
        return metrics

    def _save_checkpoint(self) -> None:
        """Rank-0 LoRA checkpoint at each validation -> ckpt_dir/step{N}/lora (FVE companion)."""
        if self._ckpt_dir is None:
            return
        rank = self.trainer.global_rank if self.trainer is not None else 0
        if rank != 0:
            return
        step = int(getattr(self.trainer, "global_step", 0)) if self.trainer is not None else 0
        out = self._ckpt_dir / f"step{step}"
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out / "lora"))  # type: ignore[operator]
        opt = self.trainer.optimizer if self.trainer is not None else None
        if opt is not None:
            torch.save({"optimizer": opt.state_dict(), "global_step": step}, out / "optim.pt")
        print(f"[gt] checkpoint saved: step {step}", flush=True)

    def hyperparams_to_log(self) -> dict[str, object]:
        return self._cfg.to_dict()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def train_gt(
    model: torch.nn.Module,
    prompts: list[WVPrompt],
    train_data: "GTDataset | DistillDataset",
    val_data: "GTDataset | DistillDataset",
    cfg: GTConfig,
    *,
    pad_id: int,
    wandb_project: str | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    ckpt_dir: Any = None,
    nodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "",
    master_port: int = 0,
    tokenizer: Any = None,
    spaces: list[Any] | None = None,
) -> dict[str, float]:
    """One GT-continuation run; ``world_size > 1`` runs ONE DDP rank in this process.

    Same global-batch contract as ``train_ao``: grad_accum is divided by world_size (when it
    divides) so LR stays valid as GPUs scale; steps are counted from the per-rank BATCH shard
    (the batch sampler splits whole layer-pure batches, not rows).

    ``world_size`` is GLOBAL (all nodes); ``nodes > 1`` uses the harness's native multi-node
    path — every node runs this once per local rank with the same ``master_addr``/``master_port``
    and its own ``node_rank`` (global rank = node_rank * ngpu + local_rank).
    """
    mydule = GTMydule(
        model, prompts, cfg, train_data, val_data, pad_id, ckpt_dir=ckpt_dir,
        tokenizer=tokenizer, spaces=spaces,
    )
    world_size = max(1, world_size)
    nodes = max(1, nodes)
    if world_size % nodes != 0:
        raise ValueError(f"world_size {world_size} not divisible by nodes {nodes}")
    grad_accum = max(1, cfg.grad_accum // world_size) if world_size > 1 else cfg.grad_accum
    per_rank_batches = make_batch_sampler(
        cfg,
        train_data.examples,
        [len(p.input_ids) for p in prompts],
        world=world_size,
        skip_batches=cfg.skip_batches,
    )
    tc = TrainingConfig()
    tc.name = cfg.run_name
    tc.max_epochs = cfg.epochs
    steps = max(1, cfg.epochs * len(per_rank_batches) // grad_accum)
    tc.max_steps = steps
    tc.ngpu = world_size // nodes
    tc.nodes = nodes
    tc.node_rank = node_rank
    if master_addr:
        tc.master_addr = master_addr
    if master_port:
        tc.master_port = master_port
    tc.train_batch_size = cfg.micro_batch
    tc.val_batch_size = cfg.micro_batch
    tc.gradient_accumulation_steps = grad_accum
    tc.gradient_clip_val = 1.0
    tc.lr = cfg.lr
    tc.lr_sched_type = cfg.lr_sched
    tc.final_lr = cfg.final_lr
    tc.warmup_steps = min(cfg.warmup_steps, max(1, steps // 10))
    tc.amp_dtype = None
    tc.validation_step_interval = cfg.eval_every_steps
    tc.validation_epoch_interval = 1
    tc.train_drop_last = True
    tc.val_drop_last = True
    tc.num_workers = 0
    tc.seed = cfg.seed
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
