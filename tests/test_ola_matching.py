"""Tests for length-histogram matching (matching.py) — the cross-cell fairness machinery."""

import pytest
import torch

from oracle_lens.pipeline.matching import (
    OCTAVES,
    histogram_fingerprint,
    length_histogram,
    match_indices,
    octave_of,
    scale_to,
)


def test_octave_of_edges() -> None:
    assert octave_of(1) == 1
    assert octave_of(2) == 2
    assert octave_of(3) == 2
    assert octave_of(31) == 16
    assert octave_of(32) == 32
    assert octave_of(511) == 256
    assert octave_of(512) == 512
    with pytest.raises(ValueError):
        octave_of(0)


def test_histogram_counts_and_zero_bins() -> None:
    hist = length_histogram(torch.tensor([1, 1, 2, 40, 512]))
    assert hist[1] == 2 and hist[2] == 1 and hist[32] == 1 and hist[512] == 1
    assert hist[64] == 0  # zero bins present, not missing
    assert sum(hist.values()) == 5


def test_scale_to_exact_total_and_shape() -> None:
    hist = dict.fromkeys(OCTAVES, 0)
    hist[1], hist[2], hist[4] = 300, 200, 100
    for n in (7, 100, 601):
        scaled = scale_to(hist, n)
        assert sum(scaled.values()) == n  # largest-remainder rounding is exact
    scaled = scale_to(hist, 600)
    assert (scaled[1], scaled[2], scaled[4]) == (300, 200, 100)


def test_match_indices_realizes_quota_exactly() -> None:
    lengths = torch.tensor([1] * 50 + [40] * 30 + [300] * 20)
    target = dict.fromkeys(OCTAVES, 0)
    target[1], target[32], target[256] = 10, 10, 10
    idx = match_indices(lengths, target, seed=0)
    assert len(idx) == 30
    assert length_histogram(lengths[idx]) == target
    # seeded => reproducible; different seed => different selection
    assert torch.equal(idx, match_indices(lengths, target, seed=0))
    assert not torch.equal(idx, match_indices(lengths, target, seed=1))


def test_match_indices_infeasible_is_an_error_not_truncation() -> None:
    lengths = torch.tensor([1] * 5)
    target = dict.fromkeys(OCTAVES, 0)
    target[1] = 6
    with pytest.raises(ValueError, match="only 5 available"):
        match_indices(lengths, target, seed=0)


def test_fingerprint_equality_is_the_cross_cell_assertion() -> None:
    # two "cells" with different raw lengths but identical octave histograms must match
    cell_a = torch.tensor([1, 3, 40, 300])
    cell_b = torch.tensor([1, 2, 33, 511])
    assert histogram_fingerprint(cell_a) == histogram_fingerprint(cell_b)
    cell_c = torch.tensor([1, 3, 40, 512])  # 512 lands in a different bin than 511
    assert histogram_fingerprint(cell_a) != histogram_fingerprint(cell_c)


def test_end_to_end_two_cells_match() -> None:
    """The pilot flow: H* = cell A's shape at rung n; both cells select; fingerprints agree."""
    gen = torch.Generator().manual_seed(7)
    cell_a = torch.randint(1, 513, (2000,), generator=gen)
    cell_b = torch.randint(1, 513, (3000,), generator=gen)
    target = scale_to(length_histogram(cell_a), 500)
    idx_a = match_indices(cell_a, target, seed=0)
    idx_b = match_indices(cell_b, target, seed=0)
    assert len(idx_a) == len(idx_b) == 500
    assert histogram_fingerprint(cell_a[idx_a]) == histogram_fingerprint(cell_b[idx_b])
