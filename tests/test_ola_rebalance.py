"""Unit tests for the per-octave quota selector (long oracle lens dataset balance)."""

import pytest
import torch

from oracle_lens.pipeline.rebalance import octave_quota_select
from oracle_lens.pipeline.shards import LENGTH_BUCKETS


def _lengths(counts: dict[int, int]) -> torch.Tensor:
    """A lengths tensor with ``counts[bucket_index]`` rows in each bucket (at the bucket's lo)."""
    vals = []
    for bi, c in counts.items():
        vals += [LENGTH_BUCKETS[bi][0]] * c
    return torch.tensor(vals, dtype=torch.long)


def test_exact_quota_per_overfull_bucket() -> None:
    lengths = _lengths({0: 100, 5: 100, 9: 100})
    kept = octave_quota_select(lengths, quota_per_bucket=10)
    kept_lengths = lengths[kept]
    for bi in (0, 5, 9):
        lo, hi = LENGTH_BUCKETS[bi]
        assert int(((kept_lengths >= lo) & (kept_lengths <= hi)).sum()) == 10


def test_underfilled_bucket_passes_through_whole() -> None:
    lengths = _lengths({3: 4, 8: 50})
    kept = octave_quota_select(lengths, quota_per_bucket=10)
    kept_lengths = lengths[kept]
    lo, hi = LENGTH_BUCKETS[3]
    assert int(((kept_lengths >= lo) & (kept_lengths <= hi)).sum()) == 4  # all 4 kept
    lo, hi = LENGTH_BUCKETS[8]
    assert int(((kept_lengths >= lo) & (kept_lengths <= hi)).sum()) == 10


def test_deterministic_and_sorted() -> None:
    lengths = _lengths({2: 40, 7: 40})
    a = octave_quota_select(lengths, quota_per_bucket=8, seed=3)
    b = octave_quota_select(lengths, quota_per_bucket=8, seed=3)
    assert torch.equal(a, b)
    assert torch.equal(a, torch.sort(a).values)
    c = octave_quota_select(lengths, quota_per_bucket=8, seed=4)
    assert not torch.equal(a, c)  # seed matters


def test_out_of_bucket_rows_dropped() -> None:
    lengths = torch.tensor([1, 4, 2048, 5000], dtype=torch.long)  # last two exceed 1024
    kept = octave_quota_select(lengths, quota_per_bucket=10)
    assert kept.tolist() == [0, 1]


def test_bad_quota_raises() -> None:
    with pytest.raises(ValueError):
        octave_quota_select(torch.tensor([1, 2]), quota_per_bucket=0)
