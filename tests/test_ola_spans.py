"""Span-sampling law tests: log-uniform lengths, run clamping, delimiter knob, no cross-turn."""

import itertools
import math
import random
from collections import Counter

from oracle_lens.pipeline.spans import (
    LongSpan,
    carve_disjoint_spans,
    draw_log_uniform,
    longest_valid_run,
    sample_long_spans,
)


def test_log_uniform_equal_mass_per_octave() -> None:
    rng = random.Random(0)
    draws = [draw_log_uniform(rng, 1024) for _ in range(40_000)]
    octaves = Counter(int(math.log2(n)) for n in draws)
    total = sum(octaves.values())
    fracs = [octaves.get(j, 0) / total for j in range(10)]
    # ~1/10 mass per octave [2^j, 2^{j+1}); loose bounds, 40k draws
    assert all(0.05 < f < 0.16 for f in fracs), fracs
    assert min(draws) >= 1 and max(draws) <= 1024


def test_longest_valid_run_respects_start_ge_1() -> None:
    # run covering position 0 loses one usable slot
    assert longest_valid_run([True, True, True, False]) == 2
    assert longest_valid_run([False, True, True, True]) == 3
    assert longest_valid_run([False, False]) == 0
    assert longest_valid_run([True]) == 0  # only position 0 -> no valid start


def test_lengths_clamped_to_longest_run() -> None:
    mask = [False] * 5 + [True] * 37 + [False] * 5
    rng = random.Random(1)
    spans = sample_long_spans(mask, max_per_conv=32, rng=rng, n_max=1024)
    assert spans, "expected spans from a 37-token run"
    assert all(s.n_tokens <= 37 for s in spans)


def test_spans_never_cross_turn_boundaries() -> None:
    rng_mask = random.Random(2)
    mask = [rng_mask.random() < 0.6 for _ in range(400)]
    spans = sample_long_spans(mask, max_per_conv=64, rng=random.Random(3), n_max=1024)
    for s in spans:
        assert s.start >= 1
        assert s.prev_pos == s.start - 1
        assert all(mask[s.start : s.start + s.n_tokens]), "span crossed a False position"
        assert s.prev_is_assistant == mask[s.prev_pos]


def test_delimiter_upsample_one_is_bit_identical_to_default() -> None:
    mask = [False] + [True] * 200
    delim = [i % 7 == 0 for i in range(201)]
    a = sample_long_spans(mask, max_per_conv=8, rng=random.Random(5), n_max=64)
    b = sample_long_spans(
        mask,
        max_per_conv=8,
        rng=random.Random(5),
        n_max=64,
        delimiter_prev=delim,
        delimiter_upsample=1.0,
    )
    assert a == b


def test_delimiter_upsample_raises_delimiter_fraction() -> None:
    mask = [False] + [True] * 400
    delim = [i % 10 == 0 for i in range(401)]

    def frac(upsample: float, seed: int) -> float:
        spans = sample_long_spans(
            mask,
            max_per_conv=6,
            rng=random.Random(seed),
            n_max=8,
            delimiter_prev=delim,
            delimiter_upsample=upsample,
        )
        assert spans
        return sum(delim[s.prev_pos] for s in spans) / len(spans)

    base = sum(frac(1.0, seed) for seed in range(60)) / 60
    boosted = sum(frac(50.0, seed) for seed in range(60)) / 60
    assert boosted > base + 0.3, (base, boosted)


def test_spans_deduplicated() -> None:
    mask = [False] + [True] * 6  # tiny: collisions likely
    spans = sample_long_spans(mask, max_per_conv=64, rng=random.Random(7), n_max=4)
    keys = [(s.start, s.n_tokens) for s in spans]
    assert len(keys) == len(set(keys))


def _assert_disjoint(spans: list[LongSpan]) -> None:
    ivs = sorted((s.start, s.start + s.n_tokens) for s in spans)
    for (_, e0), (s1, _) in itertools.pairwise(ivs):
        assert s1 >= e0, f"overlap: ends {e0} but next starts {s1}"


def test_carve_disjoint_no_overlap_cross_octave() -> None:
    """No two carved spans share a token position — including long vs short octaves."""
    mask = [False] * 7 + [True] * 2000
    spans = carve_disjoint_spans(mask, rng=random.Random(0))
    assert len(spans) > 5
    _assert_disjoint(spans)
    starts = [s.start for s in spans]
    assert len(set(starts)) == len(starts)  # one span per activation, free


def test_carve_disjoint_stays_in_region_and_start_ge_1() -> None:
    mask = [True] * 50 + [False] * 10 + [True] * 800 + [False] * 3
    spans = carve_disjoint_spans(mask, rng=random.Random(1))
    _assert_disjoint(spans)
    for sp in spans:
        assert sp.start >= 1
        assert all(mask[sp.start : sp.start + sp.n_tokens])
        assert sp.prev_is_assistant == mask[sp.prev_pos]


def test_carve_disjoint_deterministic() -> None:
    mask = [False] * 5 + [True] * 1500
    a = carve_disjoint_spans(mask, rng=random.Random(7))
    b = carve_disjoint_spans(mask, rng=random.Random(7))
    assert a == b
    c = carve_disjoint_spans(mask, rng=random.Random(8))
    assert a != c


def test_carve_disjoint_reserves_long_and_mid() -> None:
    """A qualifying conv always yields >=1 long (513+) and >=1 mid (257-512) span."""
    mask = [False] * 4 + [True] * 1400
    for seed in range(20):
        spans = carve_disjoint_spans(mask, rng=random.Random(seed))
        lens = [s.n_tokens for s in spans]
        assert any(n >= 513 for n in lens), f"seed {seed}: no long span"
        assert any(257 <= n <= 512 for n in lens), f"seed {seed}: no mid span"
        _assert_disjoint(spans)


def test_carve_disjoint_short_conv_yields_fill_only() -> None:
    """Below both reserved minima the carve still fills with short spans."""
    mask = [False] * 3 + [True] * 40
    spans = carve_disjoint_spans(mask, rng=random.Random(3))
    assert spans
    assert all(s.n_tokens <= 40 for s in spans)
    _assert_disjoint(spans)


def test_carve_disjoint_gap_zero_tiles_exactly() -> None:
    """gap_max=0 covers the fill region wall-to-wall (100% utilization)."""
    mask = [False] * 2 + [True] * 300
    spans = carve_disjoint_spans(
        mask, rng=random.Random(5), gap_max=0, long_per_conv=0, mid_per_conv=0
    )
    covered = sum(s.n_tokens for s in spans)
    assert covered == 300
    _assert_disjoint(spans)


def test_carve_disjoint_utilization_with_gaps() -> None:
    """Default gaps waste only a modest fraction of the region."""
    mask = [False] * 2 + [True] * 5000
    spans = carve_disjoint_spans(mask, rng=random.Random(11))
    covered = sum(s.n_tokens for s in spans)
    assert 0.75 <= covered / 5000 <= 1.0


def test_carve_disjoint_n_max_32_skips_above_cap_octaves() -> None:
    """n_max=32 (the crop32 regime) must not crash on cascade octaves above the cap."""
    mask = [False] * 4 + [True] * 2000
    for seed in range(10):
        spans = carve_disjoint_spans(
            mask, rng=random.Random(seed), n_max=32, long_per_conv=0, mid_per_conv=0
        )
        assert spans
        assert all(1 <= s.n_tokens <= 32 for s in spans)
        _assert_disjoint(spans)


def test_carve_disjoint_n_max_32_covers_every_octave() -> None:
    """On a long region the ≤32 cascade guarantees supply in all five octaves."""
    mask = [False] * 4 + [True] * 2000
    octaves = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32)]
    for seed in range(10):
        lens = [
            s.n_tokens
            for s in carve_disjoint_spans(
                mask, rng=random.Random(seed), n_max=32, long_per_conv=0, mid_per_conv=0
            )
        ]
        for lo, hi in octaves:
            assert any(lo <= n <= hi for n in lens), f"seed {seed}: octave {lo}-{hi} empty"


def test_carve_disjoint_n_max_32_clamped_reserved_passes() -> None:
    """build_span_pool clamps the reserved bounds to n_max; the clamped call must work."""
    mask = [False] * 4 + [True] * 500
    spans = carve_disjoint_spans(
        mask,
        rng=random.Random(2),
        n_max=32,
        long_per_conv=1,
        long_n_min=32,
        mid_per_conv=1,
        mid_n_min=32,
        mid_n_max=32,
    )
    assert sum(1 for s in spans if s.n_tokens == 32) >= 2  # both reserved placements landed
    assert all(s.n_tokens <= 32 for s in spans)
    _assert_disjoint(spans)
