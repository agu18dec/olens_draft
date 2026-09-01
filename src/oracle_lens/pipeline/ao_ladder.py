"""AO-ladder trainer wiring: (crop x layer) examples over precomputed AR reconstructions.

The trainer itself IS ``gt_train`` — same Mydule, sampler, collate, injection mechanics. This
module supplies the two AO-specific pieces:

- ``AOLadderDataset``: examples are the full (kept crop x 17 layers) cross product, indexed
  arithmetically (``idx = crop * n_layers + layer``) so tens of millions of examples never
  materialize as dicts; the injected vector is the crop's **AR reconstruction** read from the
  precompute shards through the stock ``LazyTargets`` mmap reader.
- ``ao_gt_config``: the ``GTConfig`` for an AO run — ``transform="scaled"`` with every layer's
  scale set to the ONE frozen global constant (`raw_scaled` contract: raw AR output x constant,
  soft token at the prompt slot), provenance in ``extra``.

Alignment contract: ar_out shard ``k`` covers crop indices ``[lo, hi)`` of the pool's canonical
crop enumeration (``AOPool.crop_index`` — row-major over ``keep``), and every shard records the
pool fingerprint; ``load_arout`` hard-fails on gaps, overlaps, or a fingerprint mismatch.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from oracle_lens.pipeline.ao_pool import CROP_LENGTHS, AOPool
from oracle_lens.pipeline.multilayer import LAYERS, LazyTargets
from oracle_lens.pipeline.soft_token_sft import SoftTokenConfig


def load_arout(
    arout_dir: Path, *, split: str, expect_fingerprint: str | None = None
) -> tuple[LazyTargets, dict[str, Any]]:
    """Open the ``ao_arout_{split}_*.safetensors`` shards as one lazy ``[n_crops, 17, d]`` view.

    Verifies contiguous coverage ``0..n_total`` in shard order and (when given) that every shard
    was computed from the expected pool fingerprint — a stale precompute after a pool change is
    a hard error, not a silent misalignment.
    """
    from safetensors import safe_open

    paths = sorted(arout_dir.glob(f"ao_arout_{split}_*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no ao_arout_{split}_* shards under {arout_dir}")
    rows: list[int] = []
    metas: list[dict[str, Any]] = []
    for p in paths:
        with safe_open(str(p), framework="pt") as f:
            meta = json.loads((f.metadata() or {}).get("meta", "{}"))
            n = f.get_slice("targets").get_shape()[0]
        rows.append(int(n))
        metas.append(meta)
    order = sorted(range(len(paths)), key=lambda i: int(metas[i]["lo"]))
    paths = [paths[i] for i in order]
    rows = [rows[i] for i in order]
    metas = [metas[i] for i in order]
    cursor = 0
    for p, n, meta in zip(paths, rows, metas, strict=True):
        if int(meta["lo"]) != cursor or int(meta["hi"]) - int(meta["lo"]) != n:
            raise ValueError(f"arout shard misalignment at {p}: {meta} (cursor {cursor})")
        if expect_fingerprint is not None and meta.get("pool_fingerprint") != expect_fingerprint:
            raise ValueError(
                f"arout shard {p} was computed from a DIFFERENT pool "
                f"({meta.get('pool_fingerprint')} != {expect_fingerprint}) — rerun precompute"
            )
        if meta.get("layers_per_crop") != metas[0].get("layers_per_crop"):
            raise ValueError(f"arout shard {p} mixes k-sliced and full-layer storage")
        if meta.get("heads_sha") != metas[0].get("heads_sha"):
            # the pool fingerprint does NOT encode WHICH AR produced a shard — heads_sha does.
            # A rung-B shard dropped into rung A's dir must be an error, not a silent mix.
            raise ValueError(
                f"arout shard {p} was computed from a DIFFERENT AR "
                f"({meta.get('heads_sha')} != {metas[0].get('heads_sha')}) — one dir per AR rung"
            )
        cursor += n
    d = int(metas[0].get("d_model", 5120))
    k_sliced = int(metas[0].get("layers_per_crop") or 0)
    targets = LazyTargets(paths, rows, n_layers=k_sliced or len(LAYERS), d=d)
    top: dict[str, Any] = {
        "n_crops": cursor,
        "ar_run": metas[0].get("ar_run"),
        "shards": len(paths),
    }
    if k_sliced:
        # k-sliced storage: row i holds only the k seeded layers; the pick table (int8, tiny)
        # loads eagerly so the dataset can verify its own seeded picks match EXACTLY.
        from safetensors import safe_open as _so

        picks = []
        for p in paths:
            with _so(str(p), framework="pt") as f:
                picks.append(f.get_tensor("layer_pick"))
        top["layer_pick"] = torch.cat(picks)
        top["slice_meta"] = {
            "layers_per_crop": k_sliced,
            # the pick universe these picks were drawn over — the dataset must match it
            "n_universe": int(metas[0].get("n_universe", 0)) or None,
            "layer_seed": int(metas[0]["layer_seed"]),
            "split_seed": int(metas[0]["split_seed"]),
            "n_val_crops": int(metas[0]["n_val_crops"]),
        }
    # Row->layer semantics, straight from the shards. Callers pass this to AOLadderDataset so the
    # layer universe is never assumed (see the comment there).
    top["ao_layers"] = list(metas[0].get("ao_layers") or [])
    top["ar_layers"] = list(metas[0].get("ar_layers") or [])
    # AR identity, surfaced for launch banners and cross-rung gatechecks (absent on old shards).
    top["head_mode"] = metas[0].get("head_mode")
    top["heads_sha"] = metas[0].get("heads_sha")
    return targets, top


def conv_split(
    pool: AOPool, *, n_val_crops: int, seed: int = 1234, n_avail: int | None = None
) -> tuple[Tensor, Tensor]:
    """Split canonical crop indices into (train, val) at **conversation** granularity.

    A window's 6 crops are nested prefixes of one 64-token text, and two windows sampled from the
    same conversation can overlap in text — so a clean held-out set must reserve whole
    conversations, not crops or windows. Conversations are chosen by a seeded permutation until
    the val crop quota is met; every crop of a reserved conversation goes to val, none to train.
    """
    rows, _lens = pool.crop_index()
    avail = len(rows) if n_avail is None else min(len(rows), n_avail)
    rows = rows[:avail]
    conv_of_crop = pool.conv[rows]
    uniq = torch.unique(conv_of_crop)
    gen = torch.Generator().manual_seed(seed)
    order = uniq[torch.randperm(len(uniq), generator=gen)]
    counts = torch.bincount(
        torch.searchsorted(uniq, conv_of_crop), minlength=len(uniq)
    )  # crops per conversation
    pos_of_conv = torch.searchsorted(uniq, order)
    take = torch.cumsum(counts[pos_of_conv], dim=0) <= max(1, n_val_crops)
    val_convs = order[take] if bool(take.any()) else order[:1]
    is_val = torch.isin(conv_of_crop, val_convs)
    idx = torch.arange(avail)
    return idx[~is_val], idx[is_val]


class AOLadderDataset(Dataset[dict[str, Tensor]]):
    """(crop x layer) examples: ``idx = crop * n_layers + layer_idx``; vec = AR(crop)[layer].

    ``n_crops`` caps consumption to a nested prefix of the pool's canonical crop order (the
    chunked-precompute unit); target ids are assembled on the fly from precomputed tag ids so no
    per-example dict is ever materialized.
    """

    def __init__(
        self,
        pool: AOPool,
        arout: LazyTargets,
        *,
        tokenizer: Any,
        n_crops: int | None = None,
        layers_per_crop: int = 0,
        layer_seed: int = 0,
        crop_sel: Tensor | None = None,
        target_prefix: str = "",
        arout_pick: Tensor | None = None,
        ao_layers: "list[int] | tuple[int, ...] | None" = None,
        layer_sel: tuple[int, ...] | None = None,
        wrap_tags: bool = True,
        group_by_length: bool = True,
        max_len: int | None = None,
        adopt_arout_picks: bool = False,
    ) -> None:
        rows, lens = pool.crop_index()
        avail = min(len(rows), len(arout))
        # ``crop_sel`` = explicit canonical-crop indices (see ``conv_split``). Splits MUST be made
        # at conversation granularity: a window's 6 crops are nested prefixes of one 64-token text,
        # and two windows of the same conversation can overlap — a contiguous crop-index cut would
        # leak text across train/val.
        sel = torch.arange(avail) if crop_sel is None else crop_sel[crop_sel < avail].long()
        if n_crops is not None:
            sel = sel[:n_crops]
        self.crop_sel = sel
        self.n_crops = len(sel)
        self.rows = rows[sel]
        self.lens = lens[sel]
        self.pool = pool
        self.arout = arout
        # The AO's layer universe comes from the AROUT SHARDS (their `ao_layers` metadata), not
        # from the canonical LAYERS. Two reasons it must not be assumed:
        #   * the AR drops layer 0, so its arout has 16 rows for LAYERS[1:];
        #   * --layer-min 20 restricts it further (12 rows) — Agam wants the AO to see 20-63 only.
        # Assuming 17 both mis-maps row->layer (each layer reads its neighbour) and lets a pick
        # index off the end of the array. layer_idx returned below is a position in THIS list, and
        # consumers must resolve it via self.ao_layers rather than LAYERS.
        self.ao_layers = list(ao_layers) if ao_layers else list(LAYERS)
        # ``layer_sel`` further restricts which of those layers may be drawn — the j_unit arm
        # excludes L63 (no Jacobian for the target block), matching the AR's J-loss coverage.
        # ``layer_ids`` are POSITIONS in self.ao_layers, so arout rows and the per-layer prompt
        # keep lining up.
        self.layer_ids = (
            tuple(range(len(self.ao_layers))) if layer_sel is None
            else tuple(self.ao_layers.index(ly) for ly in layer_sel)
        )
        self.n_layers = len(self.layer_ids)
        # Layer subsampling for TEXT diversity (user 2026-07-29): each crop trains only k seeded
        # layers (0 = all 17). The full cross product repeats every text 17x per epoch, which
        # drove val-CE inflection via text memorization; k=4 keeps per-layer volume high while
        # cutting repetition 4.25x. Deterministic in (layer_seed, crop) -> exact resume holds.
        self.k = layers_per_crop or self.n_layers
        if self.k < self.n_layers:
            gen = torch.Generator().manual_seed(layer_seed * 999_983 + 17)
            scores = torch.rand(self.n_crops, self.n_layers, generator=gen)
            order = scores.argsort(dim=1)[:, : self.k]
            ids = torch.tensor(self.layer_ids, dtype=torch.int8)
            self.layer_pick = ids[order]
        else:
            self.layer_pick = (
                torch.tensor(self.layer_ids, dtype=torch.int8)
                .unsqueeze(0)
                .expand(self.n_crops, -1)
            )
        # k-sliced arout (iolens): shards store only the k seeded layers per crop; row i's layout
        # is [k, d] in pick order, so __getitem__ indexes by POSITION j, not layer. The stored
        # pick table must match this dataset's own seeded picks EXACTLY — drift means the
        # precompute ran with different (layer_seed, split_seed, n_val_crops, k) and every vector
        # would silently be the wrong layer.
        self.arout_sliced = arout_pick is not None
        if arout_pick is not None:
            if arout.n_layers != self.k:
                raise ValueError(
                    f"k-sliced arout stores {arout.n_layers} layers but dataset k={self.k}"
                )
            stored = arout_pick[self.crop_sel]
            if adopt_arout_picks:
                # MERGED pools (replay mix): crop indices shift relative to each source pool, so
                # the seeded re-draw above can never equal the stored table. The stored picks ARE
                # the ground truth (the vectors were computed for them) — adopt them wholesale.
                # Only legal for k-sliced storage, where row layout is [k, d] in pick order.
                bad = ~torch.isin(
                    stored.long(), torch.tensor(self.layer_ids, dtype=torch.long)
                )
                if bool(bad.any()):
                    raise ValueError(
                        "stored arout picks fall outside this dataset's layer universe — "
                        "the shards were precomputed against a different ao_layers set"
                    )
                self.layer_pick = stored.to(torch.int8)
            elif not torch.equal(stored.to(torch.int8), self.layer_pick.to(torch.int8)):
                raise ValueError(
                    "k-sliced arout layer_pick does not match the dataset's seeded picks — "
                    "precompute ran with different (layer_seed, split_seed, n_val_crops, k); "
                    "rerun precompute"
                )
        # --max-len: SUBSET, never re-draw. Picks above were drawn over the UNFILTERED
        # selection so they equal the arout's stored per-crop picks; masking rows afterwards
        # keeps every surviving crop's pick (and its arout row index) unchanged. Filtering
        # before the draw would shift the seeded stream and every vector would silently be
        # the wrong layer — exactly the drift the equality check above exists to refuse.
        if max_len is not None:
            keep = torch.tensor(
                [self.pool.lengths[int(li)] <= max_len for li in self.lens], dtype=torch.bool
            )
            self.crop_sel = self.crop_sel[keep]
            self.rows = self.rows[keep]
            self.lens = self.lens[keep]
            self.layer_pick = self.layer_pick[keep]
            self.n_crops = int(keep.sum())

        # wrap_tags=False (the continuation_raw prompt) makes the target the RAW span + EOS —
        # no <explanation> wrapper anywhere in the supervised tokens.
        self.open_ids = (
            tokenizer("<explanation>\n", add_special_tokens=False)["input_ids"]
            if wrap_tags else []
        )
        self.close_ids = (
            tokenizer("\n</explanation>", add_special_tokens=False)["input_ids"]
            if wrap_tags else []
        )
        # Dot-variant knob (user 2026-07-31): a non-empty prefix (".") becomes the FIRST
        # supervised token before the span — "the sentence was interrupted" — to test whether it
        # breaks the pure-continuation habit. Constant per example, so (layer, length)-purity and
        # compiled static shapes are preserved; it counts into tokens_sup like the wrapper, never
        # into span tokens.
        self.prefix_ids = (
            list(tokenizer(target_prefix, add_special_tokens=False)["input_ids"])
            if target_prefix
            else []
        )
        self.eos = int(tokenizer.eos_token_id)
        # group_by_length=False -> batches are layer-pure only (val at 64 lengths: 704
        # (layer,length) groups starve the partial-batch-dropping sampler; padding waste is a
        # perf concern, not correctness — soft_token_collate masks pad with -100).
        self.group_by_length = group_by_length

    def __len__(self) -> int:
        return self.n_crops * self.k

    def layer_idx_per_example(self) -> Iterator[int]:
        """Per-example layer index (row-major over the per-crop layer picks)."""
        return iter(self.layer_pick.reshape(-1).long().tolist())

    def group_idx_per_example(self) -> Iterator[int]:
        """(layer, crop-length) group key -> batches are layer- AND length-pure (zero padding).

        group = layer_idx * n_lengths + length_idx, with n_lengths = len(pool.lengths).
        Lengths come from the POOL (self-describing) — the long-emission pools go beyond 64.
        With group_by_length=False, the key is the layer alone (layer-pure, mixed lengths).
        """
        if not self.group_by_length:
            return self.layer_idx_per_example()
        n_lengths = len(self.pool.lengths)
        grid = self.layer_pick.long() * n_lengths + self.lens.unsqueeze(1)
        return iter(grid.reshape(-1).tolist())

    def deal_key(self) -> dict[int, int]:
        """group (layer*n_lengths+length) -> length idx: ranks share a seq length per DDP step."""
        if not self.group_by_length:
            return dict.fromkeys(range(self.n_layers), 0)
        n_lengths = len(self.pool.lengths)
        return {ly * n_lengths + j: j for ly in range(self.n_layers) for j in range(n_lengths)}

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        crop, j = divmod(idx, self.k)
        layer_idx = int(self.layer_pick[crop, j])
        row = int(self.rows[crop])
        n = self.pool.lengths[int(self.lens[crop])]
        span = self.pool.ids[row, :n].tolist()
        target = [*self.open_ids, *self.prefix_ids, *span, *self.close_ids, self.eos]
        # sliced arout rows are [k, d] in pick order (position j); full rows are [17, d] (layer)
        arow = torch.as_tensor(self.arout[int(self.crop_sel[crop])])
        vec = (arow[j] if self.arout_sliced else arow[layer_idx]).float()
        return {
            "inject_vec": vec,
            "layer_idx": torch.tensor(layer_idx, dtype=torch.long),
            "target_ids": torch.tensor(target, dtype=torch.long),
            "span_len": torch.tensor(n, dtype=torch.long),
        }

    def share_memory_(self) -> "AOLadderDataset":
        """Small index tensors into shared memory for ``mp.spawn`` (arout is mmap-backed)."""
        t: Tensor
        tensors = (self.rows, self.lens, self.layer_pick, self.crop_sel)
        for t in (*tensors, self.pool.ids, self.pool.keep):
            t.share_memory_()  # type: ignore[no-untyped-call]
        return self


def ao_gt_config(
    *,
    run_name: str,
    ao_layers: "list[int] | tuple[int, ...] | None" = None,
    ar_run: str,
    scale: float,
    pool_path: str,
    n_examples: int,
    alpha: float = 8000.0,
    lora_r: int = 16,
    lr: float = 1e-4,
    micro_batch: int = 4,
    grad_accum: int = 8,
    eval_every_steps: int = 200,
    max_eval_rows: int = 1024,
    seed: int = 0,
    init_from: str = "",
    skip_batches: int = 0,
    target_prefix: str = "",
    crop_lengths: "tuple[int, ...] | None" = None,
) -> SoftTokenConfig:
    """The AO run's GTConfig: one frozen global scale wired through the ``scaled`` transform.

    ``scales`` maps every layer to the SAME constant, so ``GTMydule`` injects
    ``scale * AR(crop)[layer]`` — the ``raw_scaled`` contract — with zero trainer changes.
    """
    return SoftTokenConfig(
        run_name=run_name,
        transform="scaled",
        # Labels/prompts must name the layers the AO ACTUALLY sees: soft_token_sft uses
        # cfg.layers[pick_idx] for both `val_ce_L{n}` and the rendered prompt, so hardcoding
        # LAYERS mislabels every metric under --layer-min (pick 0 = layer 20 was logged as
        # 'L0', pick 11 = layer 63 as 'L44').
        layers=tuple(ao_layers) if ao_layers else LAYERS,
        alpha=alpha,
        scales={str(ly): float(scale) for ly in (tuple(ao_layers) if ao_layers else LAYERS)},
        n_examples=n_examples,
        min_len=min(crop_lengths or CROP_LENGTHS),
        max_len=max(crop_lengths or CROP_LENGTHS),
        lora_r=lora_r,
        lora_alpha=2 * lora_r,
        lr=lr,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        eval_every_steps=eval_every_steps,
        max_eval_rows=max_eval_rows,
        seed=seed,
        init_from=init_from,
        skip_batches=skip_batches,
        extra={
            "kind": "ao_ladder",
            "ar_run": ar_run,
            "target_prefix": target_prefix,
            "global_scale": repr(float(scale)),
            "pool": pool_path,
            "scale_mode": "global_frozen",
        },
    )
