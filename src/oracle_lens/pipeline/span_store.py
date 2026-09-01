"""Span-only intermediate shards for the long oracle lens (schema ``span_only_v1``).

``MultiLayerPairs`` minus the target stack: the sampled spans (flat ids + offsets + provenance
columns) with NO activations. The assembly step (``scripts/ola/assemble_long_spans.py``) writes
these once from the rollout harvest; the AR precompute step reads them and emits full
``multilayer_v1`` shards whose ``targets`` are the frozen reconstructor's predictions — so span
selection is decided exactly once and every AR checkpoint sees the identical spans.

Satisfies ``shards.RaggedPairs`` structurally, so ``drop_boundary_leaks`` /
``drop_placeholder_rows`` / ``length_histogram`` apply unchanged.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Bool, Int
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass(frozen=True)
class SpanOnlyShardMeta:
    """Provenance for one span-only shard (values stringified for safetensors)."""

    model_id: str
    dataset_id: str
    n_max: int
    n_min: int
    length_law: str
    split: str
    seed: int
    git_commit: str
    n_pairs: int
    schema: str = "span_only_v1"

    def to_metadata(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_metadata(cls, meta: dict[str, str]) -> "SpanOnlyShardMeta":
        return cls(
            model_id=meta["model_id"],
            dataset_id=meta["dataset_id"],
            n_max=int(meta["n_max"]),
            n_min=int(meta["n_min"]),
            length_law=meta["length_law"],
            split=meta["split"],
            seed=int(meta["seed"]),
            git_commit=meta["git_commit"],
            n_pairs=int(meta["n_pairs"]),
            schema=meta.get("schema", "span_only_v1"),
        )


@dataclass(frozen=True)
class SpanOnlyPairs:
    """Ragged spans + provenance, no targets (``RaggedPairs``-compatible)."""

    span_ids: Int[Tensor, "total"]
    offsets: Int[Tensor, "n_plus_1"]
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

    def select(self, idx: Int[Tensor, "m"]) -> "SpanOnlyPairs":
        """Row subset, vectorized (mirrors ``MultiLayerPairs.select``)."""
        idx = idx.long()
        lens = self.lengths[idx]
        new_offsets = torch.zeros(len(idx) + 1, dtype=torch.int64)
        torch.cumsum(lens, dim=0, out=new_offsets[1:])
        starts = self.offsets[idx]
        pos = torch.repeat_interleave(starts - new_offsets[:-1], lens) + torch.arange(
            int(lens.sum())
        )
        return SpanOnlyPairs(
            span_ids=self.span_ids[pos],
            offsets=new_offsets,
            conv_index=self.conv_index[idx],
            prev_pos=self.prev_pos[idx],
            prev_token_id=self.prev_token_id[idx],
            prev_is_assistant=self.prev_is_assistant[idx],
        )


def concat_span_pairs(parts: list[SpanOnlyPairs]) -> SpanOnlyPairs:
    """Row-wise concatenation with offset shifting (base pass + targeted long pass)."""
    if not parts:
        raise ValueError("no parts to concatenate")
    offsets_parts: list[Tensor] = []
    shift = 0
    for p in parts:
        off = p.offsets
        offsets_parts.append(off[1:] + shift if shift or offsets_parts else off)
        shift += int(off[-1])
    return SpanOnlyPairs(
        span_ids=torch.cat([p.span_ids for p in parts]),
        offsets=torch.cat(offsets_parts),
        conv_index=torch.cat([p.conv_index for p in parts]),
        prev_pos=torch.cat([p.prev_pos for p in parts]),
        prev_token_id=torch.cat([p.prev_token_id for p in parts]),
        prev_is_assistant=torch.cat([p.prev_is_assistant for p in parts]),
    )


def save_span_shard(path: Path, pairs: SpanOnlyPairs, meta: SpanOnlyShardMeta) -> None:
    if len(pairs) != meta.n_pairs:
        raise ValueError(f"meta.n_pairs {meta.n_pairs} != rows {len(pairs)}")
    tmp = path.with_suffix(".tmp")
    save_file(
        {
            "span_ids": pairs.span_ids.to(torch.int32).cpu(),
            "offsets": pairs.offsets.cpu(),
            "conv_index": pairs.conv_index.cpu(),
            "prev_pos": pairs.prev_pos.cpu(),
            "prev_token_id": pairs.prev_token_id.cpu(),
            "prev_is_assistant": pairs.prev_is_assistant.cpu(),
        },
        str(tmp),
        metadata=meta.to_metadata(),
    )
    tmp.replace(path)  # atomic against retries


def _read_meta(path: Path) -> SpanOnlyShardMeta:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta is None:
        raise ValueError(f"{path}: shard has no metadata")
    return SpanOnlyShardMeta.from_metadata(meta)


def load_span_shards(paths: list[Path]) -> tuple[SpanOnlyPairs, SpanOnlyShardMeta]:
    """Concatenate shards (row order = path order), shifting offsets; model/schema must match."""
    if not paths:
        raise ValueError("no shard paths given")
    parts = [load_file(str(p), device="cpu") for p in paths]
    metas = [_read_meta(p) for p in paths]
    first = metas[0]
    for m in metas[1:]:
        if (m.model_id, m.schema, m.split) != (first.model_id, first.schema, first.split):
            raise ValueError(f"shard meta mismatch: {m} vs {first}")
    offsets_parts: list[Tensor] = []
    shift = 0
    for t in parts:
        off = t["offsets"]
        offsets_parts.append(off[1:] + shift if shift or offsets_parts else off)
        shift += int(off[-1])
    pairs = SpanOnlyPairs(
        span_ids=torch.cat([t["span_ids"] for t in parts]),
        offsets=torch.cat(offsets_parts),
        conv_index=torch.cat([t["conv_index"] for t in parts]),
        prev_pos=torch.cat([t["prev_pos"] for t in parts]),
        prev_token_id=torch.cat([t["prev_token_id"] for t in parts]),
        prev_is_assistant=torch.cat([t["prev_is_assistant"] for t in parts]),
    )
    total = sum(int(m.n_pairs) for m in metas)
    if len(pairs) != total:
        raise ValueError(f"row count {len(pairs)} != sum of shard metas {total}")
    return pairs, first
