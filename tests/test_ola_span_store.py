"""Unit tests for the span-only shard format (long oracle lens intermediate)."""

from pathlib import Path

import torch

from oracle_lens.pipeline.shards import (
    boundary_leak_row_mask,
    drop_boundary_leaks,
    drop_placeholder_rows,
)
from oracle_lens.pipeline.span_store import (
    SpanOnlyPairs,
    SpanOnlyShardMeta,
    concat_span_pairs,
    load_span_shards,
    save_span_shard,
)

IM_START = 999


def _pairs() -> SpanOnlyPairs:
    """Three rows: [1,2,3], [4,IM_START], [5,6]."""
    return SpanOnlyPairs(
        span_ids=torch.tensor([1, 2, 3, 4, IM_START, 5, 6], dtype=torch.int32),
        offsets=torch.tensor([0, 3, 5, 7], dtype=torch.int64),
        conv_index=torch.tensor([0, 0, 1]),
        prev_pos=torch.tensor([10, 20, 30]),
        prev_token_id=torch.tensor([7, 8, 9]),
        prev_is_assistant=torch.tensor([True, False, True]),
    )


def _meta(n: int, split: str = "train") -> SpanOnlyShardMeta:
    return SpanOnlyShardMeta(
        model_id="m",
        dataset_id="d",
        n_max=1024,
        n_min=1,
        length_law="loguniform_octave_quota",
        split=split,
        seed=0,
        git_commit="deadbeef",
        n_pairs=n,
    )


def test_roundtrip(tmp_path: Path) -> None:
    pairs = _pairs()
    path = tmp_path / "spans_0000.safetensors"
    save_span_shard(path, pairs, _meta(len(pairs)))
    loaded, meta = load_span_shards([path])
    assert len(loaded) == 3
    assert meta.schema == "span_only_v1"
    assert torch.equal(loaded.span_ids.long(), pairs.span_ids.long())
    assert torch.equal(loaded.offsets, pairs.offsets)
    assert torch.equal(loaded.prev_pos, pairs.prev_pos)


def test_multi_shard_concat(tmp_path: Path) -> None:
    pairs = _pairs()
    for i in range(2):
        save_span_shard(tmp_path / f"spans_{i:04d}.safetensors", pairs, _meta(len(pairs)))
    loaded, _ = load_span_shards(sorted(tmp_path.glob("*.safetensors")))
    assert len(loaded) == 6
    assert loaded.row_ids(3).tolist() == [1, 2, 3]  # second shard's row 0, offsets shifted


def test_select_reindexes_ragged_ids() -> None:
    pairs = _pairs()
    sub = pairs.select(torch.tensor([0, 2]))
    assert len(sub) == 2
    assert sub.row_ids(0).tolist() == [1, 2, 3]
    assert sub.row_ids(1).tolist() == [5, 6]
    assert sub.conv_index.tolist() == [0, 1]


def test_concat_pairs_shifts_offsets() -> None:
    both = concat_span_pairs([_pairs(), _pairs()])
    assert len(both) == 6
    assert both.row_ids(4).tolist() == [4, IM_START]


def test_ragged_protocol_filters_apply() -> None:
    pairs = _pairs()
    assert boundary_leak_row_mask(pairs, IM_START).tolist() == [False, True, False]
    kept, n_dropped = drop_boundary_leaks(pairs, IM_START)
    assert n_dropped == 1 and len(kept) == 2
    assert kept.row_ids(1).tolist() == [5, 6]


class _StubTokenizer:
    """Row [4, IM_START] decodes to an anonymization placeholder; the rest are clean."""

    def batch_decode(self, ids: list[list[int]]) -> list[str]:
        return ["NAME_1 says hi" if 4 in row else "clean text" for row in ids]


def test_placeholder_filter_applies() -> None:
    kept, n_dropped = drop_placeholder_rows(_pairs(), _StubTokenizer())
    assert n_dropped == 1 and len(kept) == 2
    assert kept.row_ids(0).tolist() == [1, 2, 3]
