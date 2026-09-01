"""Ragged shard round-trip, concat, vectorized select, leak filter, duplicate accounting."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from oracle_lens.pipeline.shards import (
    LongSpanPairs,
    LongSpanShardMeta,
    drop_boundary_leaks,
    duplicate_stats,
    length_histogram,
    load_long_span_shards,
    save_long_span_shard,
)

D = 8


def make_pairs(id_rows: list[list[int]], *, layer: int = 44, seed: int = 0) -> LongSpanPairs:
    gen = torch.Generator().manual_seed(seed)
    n = len(id_rows)
    offsets = torch.zeros(n + 1, dtype=torch.int64)
    for i, row in enumerate(id_rows):
        offsets[i + 1] = offsets[i] + len(row)
    return LongSpanPairs(
        span_ids=torch.tensor([t for row in id_rows for t in row], dtype=torch.int32),
        offsets=offsets,
        targets=torch.randn(n, D, generator=gen).to(torch.bfloat16),
        layer=torch.full((n,), layer, dtype=torch.int64),
        conv_index=torch.arange(n, dtype=torch.int64),
        prev_pos=torch.arange(1, n + 1, dtype=torch.int64),
        prev_token_id=torch.arange(100, 100 + n, dtype=torch.int64),
        prev_is_assistant=torch.ones(n, dtype=torch.bool),
    )


def make_meta(n_pairs: int, split: str = "train") -> LongSpanShardMeta:
    return LongSpanShardMeta(
        model_id="test/model",
        dataset_id="test/data",
        layer=44,
        t_max=2048,
        n_max=1024,
        n_min=1,
        length_law="loguniform",
        delimiter_upsample=1.0,
        region_start=0,
        region_end=1000,
        split=split,
        seed=0,
        git_commit="deadbeef",
        n_pairs=n_pairs,
    )


def test_roundtrip(tmp_path: Path) -> None:
    pairs = make_pairs([[1, 2, 3], [4], [5, 6, 7, 8, 9]])
    path = tmp_path / "shard.safetensors"
    save_long_span_shard(path, pairs, make_meta(3))
    loaded, meta = load_long_span_shards([path])
    assert meta.n_pairs == 3 and meta.length_law == "loguniform"
    assert torch.equal(loaded.offsets, pairs.offsets)
    assert torch.equal(loaded.span_ids.long(), pairs.span_ids.long())
    assert torch.equal(loaded.targets, pairs.targets)
    assert loaded.row_ids(2).tolist() == [5, 6, 7, 8, 9]


def test_multi_shard_concat_shifts_offsets(tmp_path: Path) -> None:
    a = make_pairs([[1, 2], [3]], seed=1)
    b = make_pairs([[7, 8, 9], [10, 11]], seed=2)
    pa, pb = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_long_span_shard(pa, a, make_meta(2))
    save_long_span_shard(pb, b, make_meta(2))
    both, _ = load_long_span_shards([pa, pb])
    assert len(both) == 4
    assert both.row_ids(0).tolist() == [1, 2]
    assert both.row_ids(2).tolist() == [7, 8, 9]
    assert both.row_ids(3).tolist() == [10, 11]
    assert torch.equal(both.lengths, torch.tensor([2, 1, 3, 2]))


def test_meta_mismatch_raises(tmp_path: Path) -> None:
    a = make_pairs([[1]])
    pa, pb = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_long_span_shard(pa, a, make_meta(1))
    save_long_span_shard(pb, a, replace(make_meta(1), layer=20))
    with pytest.raises(ValueError, match="meta mismatch"):
        load_long_span_shards([pa, pb])


def test_select_matches_naive() -> None:
    rows = [[i, i + 1, i + 2][: 1 + i % 3] for i in range(50)]
    pairs = make_pairs(rows)
    idx = torch.tensor([7, 0, 49, 13, 13, 30])
    sub = pairs.select(idx)
    assert len(sub) == 6
    for j, i in enumerate(idx.tolist()):
        assert sub.row_ids(j).tolist() == pairs.row_ids(i).tolist()
        assert torch.equal(sub.targets[j], pairs.targets[i])
        assert int(sub.conv_index[j]) == int(pairs.conv_index[i])


def test_drop_boundary_leaks() -> None:
    im_start = 999
    pairs = make_pairs([[1, 2], [3, im_start], [5], [im_start], [6, 7, 8]])
    kept, n_dropped = drop_boundary_leaks(pairs, im_start)
    assert n_dropped == 2
    assert len(kept) == 3
    assert kept.row_ids(0).tolist() == [1, 2]
    assert kept.row_ids(1).tolist() == [5]
    assert kept.row_ids(2).tolist() == [6, 7, 8]


def test_duplicate_stats_counts_planted_dups() -> None:
    pairs = make_pairs([[1, 2, 3], [4, 5], [1, 2, 3], [6], [1, 2, 3], [4, 5]])
    summary, top, by_bucket = duplicate_stats(pairs)
    assert summary["n_rows"] == 6.0
    assert summary["n_unique_spans"] == 3.0
    assert summary["dup_row_frac"] == pytest.approx(5 / 6)
    assert summary["max_multiplicity"] == 3.0
    assert top[0] == (3, [1, 2, 3])
    assert top[1] == (2, [4, 5])
    # bucket (1,1): one row [6], not duplicated; bucket (2,4): five rows, all duplicated
    b1 = next(b for b in by_bucket if b["lo"] == 1.0)
    b24 = next(b for b in by_bucket if b["lo"] == 2.0)
    assert b1["rows"] == 1.0 and b1["dup_row_frac"] == 0.0
    assert b24["rows"] == 5.0 and b24["dup_row_frac"] == pytest.approx(1.0)
    assert b24["n_unique"] == 2.0


def test_length_histogram_fractions_sum_to_one() -> None:
    lens = torch.tensor([1, 2, 8, 33, 100, 600, 1024])
    hist = length_histogram(lens)
    assert sum(h["row_frac"] for h in hist) == pytest.approx(1.0)
    assert sum(h["token_share"] for h in hist) == pytest.approx(1.0)
    assert sum(h["rows"] for h in hist) == 7
