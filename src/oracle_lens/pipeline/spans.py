"""Long-span sampling for the long-span reconstructor (pure, seeded, deterministic).

Adapted from ``oracle_lens.sampling.sample_phrase_spans`` (N ≤ 32, uniform) with two changes:

- **Length law is log-uniform over {1 .. min(n_max, longest assistant run)}.** lmsys assistant
  turns are overwhelmingly short, so uniform{1..1024} would burn most draws on unfittable N and
  concentrate nearly all sample count in long spans; log-uniform gives each length octave equal
  sample counts while still putting real mass at 512-1024. Clamping the upper bound to the
  conversation's longest run (instead of retrying) keeps short conversations contributing.
- **Delimiter upsampling knob.** ``delimiter_prev`` marks rendered positions whose token is a
  delimiter (punctuation/newline); ``delimiter_upsample`` multiplies the draw weight of starts
  whose *preceding* token is a delimiter. At exactly 1.0 the unweighted code path runs, so the
  default is bit-identical to no knob at all.

A span is (tokens ``start..start+n_tokens-1``, target = residual at ``prev_pos = start - 1``),
fully inside one assistant turn, ``start >= 1``.
"""

import math
import random
import string
from dataclasses import dataclass
from typing import Protocol

DELIMITER_CHARS = frozenset(
    # fullwidth CJK punctuation is intentional (lmsys is multilingual), hence the noqa
    string.punctuation + string.whitespace + "，。！？；：、…—·「」『』（）《》【】〜～￥"  # noqa: RUF001
)


@dataclass(frozen=True)
class LongSpan:
    """One sampled long span within a rendered conversation."""

    prev_pos: int  # position of the target activation (= start - 1)
    start: int  # first span token
    n_tokens: int  # N ~ log-uniform{1 .. min(n_max, longest run)}
    prev_is_assistant: bool


class DecodeLike(Protocol):
    """The one tokenizer method delimiter detection needs."""

    def decode(self, token_ids: list[int]) -> str: ...


def delimiter_mask(rendered_ids: list[int], tokenizer: DecodeLike) -> list[bool]:
    """True where the rendered token decodes (stripped) to pure delimiter characters.

    Decodes each *unique* id once (cache); a token whose stripped text is empty (pure
    whitespace/newline) also counts as a delimiter.
    """
    cache: dict[int, bool] = {}
    out: list[bool] = []
    for tid in rendered_ids:
        if tid not in cache:
            text = tokenizer.decode([tid]).strip()
            cache[tid] = all(ch in DELIMITER_CHARS for ch in text)  # '' -> True
        out.append(cache[tid])
    return out


def longest_valid_run(assistant_mask: list[bool]) -> int:
    """Longest all-True run usable under the ``start >= 1`` constraint.

    A run covering position 0 loses its first slot (spans must leave room for ``prev_pos``).
    """
    best = 0
    run_start: int | None = None
    for pos, flag in enumerate([*assistant_mask, False]):
        if flag and run_start is None:
            run_start = pos
        elif not flag and run_start is not None:
            best = max(best, pos - max(run_start, 1))
            run_start = None
    return best


def valid_runs(assistant_mask: list[bool]) -> list[tuple[int, int]]:
    """Maximal all-True runs as half-open ``[start, end)``, with ``start >= 1`` applied.

    A run covering position 0 loses its first slot (spans must leave room for ``prev_pos``);
    runs that become empty under that constraint are dropped.
    """
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for pos, flag in enumerate([*assistant_mask, False]):
        if flag and run_start is None:
            run_start = pos
        elif not flag and run_start is not None:
            start = max(run_start, 1)
            if pos > start:
                runs.append((start, pos))
            run_start = None
    return runs


def draw_log_uniform(rng: random.Random, n_hi: int, *, n_lo: int = 1) -> int:
    """N with P(N=k) ∝ ln((k+1)/k) ≈ 1/k over {n_lo..n_hi} (equal mass per octave)."""
    if n_hi < n_lo or n_lo < 1:
        raise ValueError(f"need 1 <= n_lo <= n_hi, got {n_lo}..{n_hi}")
    n = int(math.exp(rng.uniform(math.log(n_lo), math.log(n_hi + 1))))
    return min(max(n, n_lo), n_hi)


def sample_long_spans(
    assistant_mask: list[bool],
    *,
    max_per_conv: int,
    rng: random.Random,
    n_min: int = 1,
    n_max: int = 1024,
    max_attempts: int = 64,
    delimiter_prev: list[bool] | None = None,
    delimiter_upsample: float = 1.0,
    length_law: str = "loguniform",
) -> list[LongSpan]:
    """Up to ``max_per_conv`` deduplicated spans fully inside assistant turns, ``start >= 1``.

    Per draw: N ~ ``length_law``{n_min .. min(n_max, longest run)} — "loguniform" (equal mass
    per octave; the long-span default) or "uniform" (the previous reconstructor's law, N-first
    then a uniform valid start) — then a start among positions where
    ``assistant_mask[start:start+N]`` is all-True — uniform, or weighted by
    ``delimiter_upsample`` on starts whose token at ``start-1`` is a delimiter.

    ``n_min > 1`` turns this into a targeted long-span harvest: conversations whose longest
    assistant run is shorter than ``n_min`` yield nothing (used to enrich the >512 tail from a
    fresh corpus region, where plain sampling gets ~0.1% of rows there).
    """
    if delimiter_upsample <= 0.0:
        raise ValueError(f"delimiter_upsample must be > 0, got {delimiter_upsample}")
    if n_min < 1 or n_min > n_max:
        raise ValueError(f"need 1 <= n_min <= n_max, got {n_min}..{n_max}")
    if length_law not in ("loguniform", "uniform"):
        raise ValueError(f"unknown length_law {length_law!r}")
    length = len(assistant_mask)
    if delimiter_prev is not None and len(delimiter_prev) != length:
        raise ValueError("delimiter_prev must align with assistant_mask")
    n_hi = min(n_max, longest_valid_run(assistant_mask))
    if n_hi < n_min:
        return []
    # prefix[i] = number of True in mask[:i]; span all-True iff prefix[s+n] - prefix[s] == n
    prefix = [0]
    for flag in assistant_mask:
        prefix.append(prefix[-1] + int(flag))
    seen: set[tuple[int, int]] = set()
    spans: list[LongSpan] = []
    attempts = 0
    weighted = delimiter_upsample != 1.0 and delimiter_prev is not None
    while len(spans) < max_per_conv and attempts < max_attempts:
        attempts += 1
        if length_law == "uniform":
            n = rng.randint(n_min, n_hi)
        else:
            n = draw_log_uniform(rng, n_hi, n_lo=n_min)
        valid = [s for s in range(1, length - n + 1) if prefix[s + n] - prefix[s] == n]
        if not valid:
            continue
        if weighted:
            assert delimiter_prev is not None
            weights = [delimiter_upsample if delimiter_prev[s - 1] else 1.0 for s in valid]
            start = rng.choices(valid, weights=weights, k=1)[0]
        else:
            start = valid[rng.randrange(len(valid))]
        if (start, n) in seen:
            continue
        seen.add((start, n))
        spans.append(
            LongSpan(
                prev_pos=start - 1,
                start=start,
                n_tokens=n,
                prev_is_assistant=assistant_mask[start - 1],
            )
        )
    return spans


def _place_reserved(
    segments: list[tuple[int, int]],
    rng: random.Random,
    *,
    n_lo: int,
    n_hi: int,
    per_conv: int,
) -> tuple[list[LongSpan], list[tuple[int, int]]]:
    """Place up to ``per_conv`` spans of ``n ~ loguniform{n_lo..n_hi}`` at random offsets.

    Each placement splits its segment into left/right remainders. Segments are consumed
    largest-first so a qualifying conversation always yields its reserved span; the random
    offset keeps start positions unbiased within the segment.
    """
    placed: list[LongSpan] = []
    free = list(segments)
    for _ in range(per_conv):
        fits = [i for i, (s, e) in enumerate(free) if e - s >= n_lo]
        if not fits:
            break
        seg_i = max(fits, key=lambda i: free[i][1] - free[i][0])
        s, e = free.pop(seg_i)
        n = draw_log_uniform(rng, min(n_hi, e - s), n_lo=n_lo)
        start = rng.randint(s, e - n)
        placed.append(
            LongSpan(prev_pos=start - 1, start=start, n_tokens=n, prev_is_assistant=False)
        )
        if start - s >= 1:
            free.append((s, start))
        if e - (start + n) >= 1:
            free.append((start + n, e))
    return placed, free


def carve_uniform_spans(
    assistant_mask: list[bool],
    *,
    rng: random.Random,
    n_lo: int = 1,
    n_hi: int = 32,
    gap_max: int = 8,
) -> list[LongSpan]:
    """Tile the assistant region with DISJOINT spans of N ~ uniform{n_lo..n_hi} — the iolens AR
    law (Agam 2026-08-01): every span is its own (text, target) pair at its own position; no
    span is a prefix of another, no activation is shared between pairs, and position coverage
    is uniform by construction (the octave carve's U-shaped target clustering disappears).

    Walk each maximal assistant run left to right: random gap ``g ~ uniform{0..gap_max}``, then
    ``N ~ uniform{n_lo..n_hi}`` (clamped to the run tail; a tail shorter than ``n_lo`` is
    dropped). Exact-uniform N is restored downstream by ``uniform_length_indices`` — the clamp
    only skews the tail slightly. A 567-token answer yields ~25-30 spans (vs ~5 sources under
    the octave carve), each with a distinct target.
    """
    if gap_max < 0:
        raise ValueError(f"gap_max must be >= 0, got {gap_max}")
    if not (1 <= n_lo <= n_hi):
        raise ValueError(f"bad span bounds: n_lo={n_lo} n_hi={n_hi}")
    spans: list[LongSpan] = []
    for seg_start, seg_end in valid_runs(assistant_mask):
        cursor = seg_start + rng.randint(0, gap_max)
        while cursor < seg_end:
            n = min(rng.randint(n_lo, n_hi), seg_end - cursor)
            if n < n_lo:
                break
            spans.append(
                LongSpan(
                    prev_pos=cursor - 1,
                    start=cursor,
                    n_tokens=n,
                    prev_is_assistant=assistant_mask[cursor - 1],
                )
            )
            cursor += n + rng.randint(0, gap_max)
    return spans


def carve_disjoint_spans(
    assistant_mask: list[bool],
    *,
    rng: random.Random,
    n_max: int = 1024,
    gap_max: int = 32,
    long_per_conv: int = 1,
    long_n_min: int = 513,
    mid_per_conv: int = 1,
    mid_n_min: int = 257,
    mid_n_max: int = 512,
    cascade_octaves: tuple[tuple[int, int], ...] = (
        (129, 256),
        (65, 128),
        (33, 64),
        (17, 32),
        (9, 16),
        (5, 8),
        (2, 4),
        (1, 1),
    ),
) -> list[LongSpan]:
    """Mutually DISJOINT spans covering most of the assistant region (pure, seeded).

    The redundancy-free replacement for the three ``sample_long_spans`` passes: no two spans
    of a conversation share a token position (cross-octave included), so no activation is
    reused and no target text repeats — the 2026-07-29 val_ce-rise diagnosis (4.26x token
    redundancy at 18.6 spans/conv; rise concentrated in the 2.67x-redundant long octaves).

    Three phases, all within maximal assistant runs (``start >= 1``):

    1. RESERVED LONG: up to ``long_per_conv`` spans, ``n ~ loguniform{long_n_min..n_max}``, at
       a uniform-random offset (largest segment first). Guarantees long-octave supply that a
       left-to-right log-uniform walk would starve (~10% of draws are >= 513).
    2. RESERVED CASCADE: same for ``[mid_n_min, mid_n_max]`` (``mid_per_conv``), then one span
       per octave for ``cascade_octaves`` in the ever-smaller remainders. Without the cascade
       the upper-mid octaves starve — measured on the 2026-07 harvest, reserved long+mid eat
       most of a mean-668-token conv and log-uniform fill over the scraps left 129-256 at 23k
       supply vs 121k for 33-64 (a 231k-pair balanced ceiling).
    3. FILL: walk each remaining segment left to right — random gap ``g ~ uniform{0..gap_max}``
       then ``n ~ loguniform{1..min(n_max, remaining)}`` — until the segment is exhausted.
       Gaps (plus the random reserved offsets) keep start positions from tiling
       deterministically; expected utilization stays ~90% at the default ``gap_max``.

    ``octave_quota_select`` downstream equalizes per-octave counts; over-supplied octaves are
    trimmed there, never re-sampled here.
    """
    if gap_max < 0:
        raise ValueError(f"gap_max must be >= 0, got {gap_max}")
    # bounds only constrain a pass that will actually place spans — a disabled pass (per_conv=0)
    # must not reject small n_max (e.g. the crop32 carve disables long/mid entirely)
    if long_per_conv > 0 and not 1 <= long_n_min <= n_max:
        raise ValueError(f"bad reserved-pass bounds: long_n_min={long_n_min} n_max={n_max}")
    if mid_per_conv > 0 and not 1 <= mid_n_min <= mid_n_max <= n_max:
        raise ValueError(f"bad reserved-pass bounds: mid=[{mid_n_min},{mid_n_max}] n_max={n_max}")
    spans: list[LongSpan] = []
    segments = valid_runs(assistant_mask)
    long_placed, segments = _place_reserved(
        segments, rng, n_lo=long_n_min, n_hi=n_max, per_conv=long_per_conv
    )
    mid_placed, segments = _place_reserved(
        segments, rng, n_lo=mid_n_min, n_hi=mid_n_max, per_conv=mid_per_conv
    )
    spans += long_placed + mid_placed
    for lo, hi in cascade_octaves:
        if lo > n_max:  # octave entirely above the length cap (e.g. n_max=32 crop runs)
            continue
        placed, segments = _place_reserved(segments, rng, n_lo=lo, n_hi=min(hi, n_max), per_conv=1)
        spans += placed
    for s, e in segments:
        cursor = s + rng.randint(0, gap_max)
        while e - cursor >= 1:
            n = draw_log_uniform(rng, min(n_max, e - cursor))
            spans.append(
                LongSpan(prev_pos=cursor - 1, start=cursor, n_tokens=n, prev_is_assistant=False)
            )
            cursor += n + rng.randint(0, gap_max)
    # placement used interval bookkeeping only; stamp the real mask verdict for prev_pos here
    return [
        LongSpan(
            prev_pos=sp.prev_pos,
            start=sp.start,
            n_tokens=sp.n_tokens,
            prev_is_assistant=assistant_mask[sp.prev_pos],
        )
        for sp in spans
    ]
