"""Ragged pair shards for the long-span reconstructor.

The oracle-lens shard format right-pads ``phrase_ids`` to ``[n, max_n]``; at ``max_n = 1024``
with a log-uniform length law (mean length ≈ 147) that wastes ~7x the storage, so spans are
stored RAGGED: one flat id tensor plus row offsets. A per-row ``layer`` column is carried from
day one so a future prompt-conditioned any-layer reconstructor can consume mixed-layer shards
through this same loader.

    span_ids          [total_tokens] int32   (flat, concatenated)
    offsets           [n+1]          int64   (row i = span_ids[offsets[i]:offsets[i+1]])
    targets           [n, d]         bf16    (layer residual at prev_pos, raw space)
    layer             [n]            int64
    conv_index        [n]            int64
    prev_pos          [n]            int64
    prev_token_id     [n]            int64   (rendered token AT prev_pos — delimiter stats)
    prev_is_assistant [n]            bool

Span *text* is decode-on-demand from ids (never re-tokenized — the column-zero gotcha).
"""

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import torch
from jaxtyping import Bool, Float, Int
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor


class RaggedPairs(Protocol):
    """Structural schema shared by ``LongSpanPairs`` and ``multilayer.MultiLayerPairs``.

    The filters/stats below only touch the flat-ids + offsets side, so they accept any pair
    class carrying it (the "same span-length machinery applies" promise in ``multilayer``).
    """

    @property
    def span_ids(self) -> Int[Tensor, " total"]: ...

    @property
    def offsets(self) -> Int[Tensor, " n_plus_1"]: ...

    @property
    def lengths(self) -> Int[Tensor, " n"]: ...

    def __len__(self) -> int: ...

    def row_ids(self, i: int) -> Int[Tensor, " n_i"]: ...

    def select(self, idx: Int[Tensor, " m"]) -> Self: ...


@dataclass(frozen=True)
class LongSpanShardMeta:
    """Provenance carried on every shard (values stringified for safetensors)."""

    model_id: str
    dataset_id: str
    layer: int
    t_max: int
    n_max: int
    n_min: int
    length_law: str
    delimiter_upsample: float
    region_start: int
    region_end: int
    split: str
    seed: int
    git_commit: str
    n_pairs: int
    schema: str = "ragged-1"

    def to_metadata(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_metadata(cls, meta: dict[str, str]) -> "LongSpanShardMeta":
        return cls(
            model_id=meta["model_id"],
            dataset_id=meta["dataset_id"],
            layer=int(meta["layer"]),
            t_max=int(meta["t_max"]),
            n_max=int(meta["n_max"]),
            n_min=int(meta.get("n_min", "1")),
            length_law=meta["length_law"],
            delimiter_upsample=float(meta["delimiter_upsample"]),
            region_start=int(meta["region_start"]),
            region_end=int(meta["region_end"]),
            split=meta["split"],
            seed=int(meta["seed"]),
            git_commit=meta["git_commit"],
            n_pairs=int(meta["n_pairs"]),
            schema=meta.get("schema", "ragged-1"),
        )


@dataclass(frozen=True)
class LongSpanPairs:
    """Stacked ragged pairs from one or more shards (targets kept in stored dtype)."""

    span_ids: Int[Tensor, "total"]
    offsets: Int[Tensor, "n_plus_1"]
    targets: Float[Tensor, "n d"]
    layer: Int[Tensor, "n"]
    conv_index: Int[Tensor, "n"]
    prev_pos: Int[Tensor, "n"]
    prev_token_id: Int[Tensor, "n"]
    prev_is_assistant: Bool[Tensor, "n"]

    def __len__(self) -> int:
        return int(self.offsets.shape[0]) - 1

    @property
    def lengths(self) -> Int[Tensor, "n"]:
        return self.offsets[1:] - self.offsets[:-1]

    def row_ids(self, i: int) -> Int[Tensor, "n_i"]:
        return self.span_ids[int(self.offsets[i]) : int(self.offsets[i + 1])].long()

    def select(self, idx: Int[Tensor, "m"]) -> "LongSpanPairs":
        """Row subset, vectorized (a Python loop over 1M rows would dominate load time)."""
        idx = idx.long()
        lens = self.lengths[idx]
        new_offsets = torch.zeros(len(idx) + 1, dtype=torch.int64)
        torch.cumsum(lens, dim=0, out=new_offsets[1:])
        # flat gather positions: for row j, offsets[idx[j]] + (0..lens[j]-1)
        starts = self.offsets[idx]
        pos = torch.repeat_interleave(starts - new_offsets[:-1], lens) + torch.arange(
            int(lens.sum())
        )
        return LongSpanPairs(
            span_ids=self.span_ids[pos],
            offsets=new_offsets,
            targets=self.targets[idx],
            layer=self.layer[idx],
            conv_index=self.conv_index[idx],
            prev_pos=self.prev_pos[idx],
            prev_token_id=self.prev_token_id[idx],
            prev_is_assistant=self.prev_is_assistant[idx],
        )

    def share_memory_(self) -> "LongSpanPairs":
        """Move tensors to shared memory (one physical copy across DDP spawn workers)."""
        for t in (
            self.span_ids,
            self.offsets,
            self.targets,
            self.layer,
            self.conv_index,
            self.prev_pos,
            self.prev_token_id,
            self.prev_is_assistant,
        ):
            t.share_memory_()  # type: ignore[no-untyped-call]
        return self


def concat_pairs(a: LongSpanPairs, b: LongSpanPairs) -> LongSpanPairs:
    """Row-wise concatenation (offsets shifted) — e.g. base-suffix + long-span harvest mixture."""
    return LongSpanPairs(
        span_ids=torch.cat([a.span_ids, b.span_ids]),
        offsets=torch.cat([a.offsets, b.offsets[1:] + int(a.offsets[-1])]),
        targets=torch.cat([a.targets, b.targets]),
        layer=torch.cat([a.layer, b.layer]),
        conv_index=torch.cat([a.conv_index, b.conv_index]),
        prev_pos=torch.cat([a.prev_pos, b.prev_pos]),
        prev_token_id=torch.cat([a.prev_token_id, b.prev_token_id]),
        prev_is_assistant=torch.cat([a.prev_is_assistant, b.prev_is_assistant]),
    )


def save_long_span_shard(path: Path, pairs: LongSpanPairs, meta: LongSpanShardMeta) -> None:
    if len(pairs) != meta.n_pairs:
        raise ValueError(f"meta.n_pairs {meta.n_pairs} != rows {len(pairs)}")
    tmp = path.with_suffix(".tmp")
    save_file(
        {
            "span_ids": pairs.span_ids.to(torch.int32).cpu(),
            "offsets": pairs.offsets.cpu(),
            "targets": pairs.targets.to(torch.bfloat16).cpu(),
            "layer": pairs.layer.cpu(),
            "conv_index": pairs.conv_index.cpu(),
            "prev_pos": pairs.prev_pos.cpu(),
            "prev_token_id": pairs.prev_token_id.cpu(),
            "prev_is_assistant": pairs.prev_is_assistant.cpu(),
        },
        str(tmp),
        metadata=meta.to_metadata(),
    )
    tmp.replace(path)  # atomic against retries


def load_long_span_shards(paths: list[Path]) -> tuple[LongSpanPairs, LongSpanShardMeta]:
    """Concatenate shards (row order = path order), shifting offsets. Meta from the first;
    model/layer/t_max/schema consistency asserted across shards."""
    if not paths:
        raise ValueError("no shard paths given")
    parts = [load_file(str(p), device="cpu") for p in paths]
    metas = [_read_meta(p) for p in paths]
    first = metas[0]
    for m in metas[1:]:
        # t_max/n_min/n_max may differ across shard families (e.g. the targeted long-span
        # harvest uses t_max=4096, n_min=257) — provenance, not semantics; model/layer/schema
        # must match.
        same = (m.model_id, m.layer, m.schema) == (first.model_id, first.layer, first.schema)
        if not same:
            raise ValueError(f"shard meta mismatch: {m} vs {first}")
    offsets_parts: list[Tensor] = []
    shift = 0
    for t in parts:
        off = t["offsets"]
        offsets_parts.append(off[1:] + shift if shift or offsets_parts else off)
        shift += int(off[-1])
    pairs = LongSpanPairs(
        span_ids=torch.cat([t["span_ids"] for t in parts]),
        offsets=torch.cat(offsets_parts),
        targets=torch.cat([t["targets"] for t in parts]),
        layer=torch.cat([t["layer"] for t in parts]),
        conv_index=torch.cat([t["conv_index"] for t in parts]),
        prev_pos=torch.cat([t["prev_pos"] for t in parts]),
        prev_token_id=torch.cat([t["prev_token_id"] for t in parts]),
        prev_is_assistant=torch.cat([t["prev_is_assistant"] for t in parts]),
    )
    total = sum(int(m.n_pairs) for m in metas)
    if len(pairs) != total:
        raise ValueError(f"row count {len(pairs)} != sum of shard metas {total}")
    return pairs, first


def _read_meta(path: Path) -> LongSpanShardMeta:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta is None:
        raise ValueError(f"{path}: shard has no metadata")
    return LongSpanShardMeta.from_metadata(meta)


def boundary_leak_row_mask(pairs: RaggedPairs, im_start_id: int) -> "Bool[Tensor, 'n']":
    """True where the span contains ``<|im_start|>`` (turn-boundary mask misalignment).

    Vectorized: flat hit positions -> owning row via searchsorted.
    """
    hits = torch.nonzero(pairs.span_ids == im_start_id).squeeze(-1)
    bad_rows = torch.unique(torch.searchsorted(pairs.offsets, hits, right=True) - 1)
    bad = torch.zeros(len(pairs), dtype=torch.bool)
    bad[bad_rows] = True
    return bad


def drop_boundary_leaks[P: RaggedPairs](
    pairs: P, im_start_id: int, *, label: str = ""
) -> tuple[P, int]:
    """Drop rows whose span contains ``<|im_start|>`` (turn-boundary mask misalignment —
    Qwen strips <think> from history; same filter as the oracle-lens recon, runbook 2026-07-14).
    """
    keep = torch.nonzero(~boundary_leak_row_mask(pairs, im_start_id)).squeeze(-1)
    n_dropped = len(pairs) - int(keep.shape[0])
    if label:
        pct = 100.0 * n_dropped / max(1, len(pairs))
        print(f"[boundary-leak filter] {label}: dropped {n_dropped}/{len(pairs)} ({pct:.2f}%)")
    return pairs.select(keep), n_dropped


# lmsys anonymization placeholders (NAME_1, NAME_2, ...) — 6.6% of base rows, ~32% of 513-1024
# spans. User decision 2026-07-19: drop from all future TRAINING data (evals untouched).
PLACEHOLDER_RE = re.compile(r"NAME_\d+")


def placeholder_row_mask(pairs: RaggedPairs, tokenizer: object) -> "Bool[Tensor, 'n']":
    """True where the decoded span contains an anonymization placeholder."""
    decode: Any = tokenizer.batch_decode  # type: ignore[attr-defined]  # duck-typed
    hits = torch.zeros(len(pairs), dtype=torch.bool)
    batch = 20_000
    for lo in range(0, len(pairs), batch):
        ids: Iterable[list[int]] = [
            pairs.row_ids(i).tolist() for i in range(lo, min(lo + batch, len(pairs)))
        ]
        for j, text in enumerate(decode(ids)):
            if PLACEHOLDER_RE.search(text):
                hits[lo + j] = True
    return hits


def drop_placeholder_rows[P: RaggedPairs](
    pairs: P, tokenizer: object, *, label: str = ""
) -> tuple[P, int]:
    """Drop NAME_N rows (training-data hygiene; run BEFORE prefix selection so any requested
    n_pairs still comes out at full quantity from the clean pool)."""
    hits = placeholder_row_mask(pairs, tokenizer)
    keep = torch.nonzero(~hits).squeeze(-1)
    n_dropped = len(pairs) - int(keep.shape[0])
    if label:
        pct = 100.0 * n_dropped / max(1, len(pairs))
        print(f"[placeholder filter] {label}: dropped {n_dropped}/{len(pairs)} ({pct:.2f}%)")
    return pairs.select(keep), n_dropped


LENGTH_BUCKETS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
)


def length_histogram(
    lengths: Int[Tensor, "n"], *, buckets: tuple[tuple[int, int], ...] = LENGTH_BUCKETS
) -> list[dict[str, float]]:
    """Per-bucket row count/fraction and token count/share — THE length-distribution report."""
    total_rows = int(lengths.shape[0])
    total_tokens = int(lengths.sum())
    out: list[dict[str, float]] = []
    for lo, hi in buckets:
        sel = (lengths >= lo) & (lengths <= hi)
        rows = int(sel.sum())
        tokens = int(lengths[sel].sum())
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "rows": float(rows),
                "row_frac": rows / max(1, total_rows),
                "tokens": float(tokens),
                "token_share": tokens / max(1, total_tokens),
            }
        )
    return out


def duplicate_stats(
    pairs: RaggedPairs, *, top_k: int = 20
) -> tuple[dict[str, float], list[tuple[int, list[int]]], list[dict[str, float]]]:
    """Exact-duplicate span accounting (token-id hash). NO dedup happens anywhere — user
    decision: duplicates stay in the training data, we just count them.

    Returns (summary, top duplicated spans as (count, ids), per-length-bucket breakdown).
    ``dup_row_frac`` = fraction of rows whose exact span occurs more than once.
    """
    counts: Counter[bytes] = Counter()
    rep: dict[bytes, list[int]] = {}
    row_hash: list[bytes] = []
    ids32 = pairs.span_ids.to(torch.int32)
    offsets = pairs.offsets.tolist()
    raw = ids32.numpy()
    for i in range(len(pairs)):
        h = hashlib.blake2b(raw[offsets[i] : offsets[i + 1]].tobytes(), digest_size=16).digest()
        counts[h] += 1
        row_hash.append(h)
        if h not in rep:
            rep[h] = raw[offsets[i] : offsets[i + 1]].tolist()
    n_rows = len(pairs)
    n_unique = len(counts)
    dup_rows = sum(c for c in counts.values() if c > 1)
    summary = {
        "n_rows": float(n_rows),
        "n_unique_spans": float(n_unique),
        "dup_row_frac": dup_rows / max(1, n_rows),
        "max_multiplicity": float(max(counts.values(), default=0)),
    }
    top = [(c, rep[h]) for h, c in counts.most_common(top_k) if c > 1]
    lengths = pairs.lengths
    by_bucket: list[dict[str, float]] = []
    for lo, hi in LENGTH_BUCKETS:
        sel = ((lengths >= lo) & (lengths <= hi)).tolist()
        rows_b = sum(sel)
        dup_b = sum(1 for i, s in enumerate(sel) if s and counts[row_hash[i]] > 1)
        uniq_b = len({row_hash[i] for i, s in enumerate(sel) if s})
        by_bucket.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "rows": float(rows_b),
                "dup_row_frac": dup_b / max(1, rows_b),
                "n_unique": float(uniq_b),
            }
        )
    return summary, top, by_bucket
