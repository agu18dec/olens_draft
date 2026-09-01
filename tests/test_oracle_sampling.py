"""Phrase-span sampling: bounds, mask containment, determinism, split stability."""

import random

from oracle_lens.core.sampling import sample_phrase_spans, split_of


def _mask(pattern: str) -> list[bool]:
    return [c == "1" for c in pattern]


def test_spans_inside_mask_and_start_geq_1() -> None:
    mask = _mask("0011111111110000111111")
    spans = sample_phrase_spans(mask, max_per_conv=32, rng=random.Random(0), n_max=6)
    assert spans, "expected at least one span"
    for s in spans:
        assert s.start >= 1
        assert s.prev_pos == s.start - 1
        assert all(mask[s.start : s.start + s.n_tokens])
        assert s.prev_is_assistant == mask[s.prev_pos]


def test_deterministic_under_seed() -> None:
    mask = _mask("01" * 64)
    a = sample_phrase_spans(mask, max_per_conv=8, rng=random.Random(7))
    b = sample_phrase_spans(mask, max_per_conv=8, rng=random.Random(7))
    assert a == b


def test_no_duplicate_spans() -> None:
    mask = _mask("0" + "1" * 40)
    spans = sample_phrase_spans(mask, max_per_conv=64, rng=random.Random(1), n_max=4)
    assert len({(s.start, s.n_tokens) for s in spans}) == len(spans)


def test_short_turns_still_yield_small_n() -> None:
    # Only 3 assistant tokens: N in {17..32} never fits, but retries should land small Ns.
    mask = _mask("0111")
    spans = sample_phrase_spans(mask, max_per_conv=4, rng=random.Random(3))
    assert spans
    assert all(s.n_tokens <= 3 for s in spans)


def test_n_roughly_uniform_on_long_mask() -> None:
    mask = _mask("0" + "1" * 2000)
    counts = [0] * 33
    for seed in range(200):
        for s in sample_phrase_spans(mask, max_per_conv=8, rng=random.Random(seed)):
            counts[s.n_tokens] += 1
    drawn = counts[1:]
    assert min(drawn) > 0
    assert max(drawn) < 4 * (sum(drawn) / len(drawn))  # loose uniformity bound


def test_split_stable_and_fraction() -> None:
    keys = [f"lmsys:{i}" for i in range(20_000)]
    splits = [split_of(k) for k in keys]
    assert splits == [split_of(k) for k in keys]  # process-stable by construction
    eval_frac = splits.count("eval") / len(splits)
    assert 0.03 < eval_frac < 0.07  # ~5%
