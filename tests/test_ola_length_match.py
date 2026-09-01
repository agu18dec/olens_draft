"""CPU tests for length-matched subsampling: the subsampled pool's octave histogram must match the
target within sampling noise (this is what removes the span-length confound across data cells)."""

import torch

from oracle_lens.pipeline.multilayer import (
    _octave,
    length_hist_fractions,
    length_matched_indices,
)


def test_octave_buckets() -> None:
    assert [_octave(x) for x in (1, 2, 3, 4, 7, 8, 15, 16)] == [0, 1, 1, 2, 2, 3, 3, 4]


def test_hist_fractions_sum_to_one() -> None:
    lengths = torch.tensor([1, 1, 2, 4, 4, 4, 8])
    frac = length_hist_fractions(lengths)
    assert abs(sum(frac.values()) - 1.0) < 1e-9
    assert frac[0] == 2 / 7 and frac[2] == 3 / 7  # two len-1, three len-4..7


def test_matched_pool_follows_target() -> None:
    # a LONG-skewed source (like PT): mostly length 64-127 (octave 6)
    long_src = torch.cat([torch.full((900,), 100), torch.full((100,), 2)])
    # target = SHORT-skewed (like assistant): 70% octave 0, 30% octave 1
    target = {0: 0.7, 1: 0.3}
    idx = length_matched_indices(long_src, target, n_total=200, seed=0)
    got = length_hist_fractions(long_src[idx])
    # only the target's buckets appear, in ~the target proportion (source octave 6 dropped entirely)
    assert set(got) <= {0, 1}
    # source has no octave-0 spans (no length-1); target octave 0 contributes nothing.
    # octave 1 fills entirely from the 100 len-2 spans.
    assert got.get(1, 0) > 0.99  # all picks land in octave 1 (the only available target bucket)


def test_matched_pool_hits_proportions_when_available() -> None:
    # source with plenty in both octave 0 (len 1) and octave 3 (len 8-15)
    src = torch.cat([torch.ones(500, dtype=torch.int64), torch.full((500,), 10)])
    target = {0: 0.25, 3: 0.75}
    idx = length_matched_indices(src, target, n_total=400, seed=1)
    got = length_hist_fractions(src[idx])
    assert abs(got[0] - 0.25) < 0.05 and abs(got[3] - 0.75) < 0.05


def test_uniform_length_indices_equal_counts() -> None:
    from oracle_lens.pipeline.multilayer import uniform_length_indices

    # loguniform-ish pool: many short, few long (within 1..8)
    lengths = torch.cat([torch.full((800 // n,), n) for n in range(1, 9)])
    idx = uniform_length_indices(lengths, 1, 8, n_total=400, seed=0)
    got = torch.bincount(lengths[idx])
    # equal counts per length (50 each), no length over-represented
    assert set(got[1:9].tolist()) == {50}
    assert len(idx) == 400


def test_uniform_length_indices_scarce_takes_all() -> None:
    from oracle_lens.pipeline.multilayer import uniform_length_indices

    lengths = torch.cat([torch.ones(100, dtype=torch.int64), torch.full((3,), 4)])
    idx = uniform_length_indices(lengths, 1, 4, n_total=80, seed=0)  # want 20/length
    got = torch.bincount(lengths[idx], minlength=5)
    assert got[1] == 20 and got[4] == 3  # scarce length capped at available
