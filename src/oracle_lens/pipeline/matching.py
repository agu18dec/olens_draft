"""Length-histogram matching across data cells — pure, seeded, CPU-only.

The comparison design (docs/project/experiments/ola/scaling_plan.md §3): every cell trains and
evals on the SAME span-length histogram ``H*``, because span length otherwise sets both the task
difficulty and — through the constant-padded-token bucketing — the statistical batch size. With
matched histograms, examples/tokens/updates all become proportional across cells and the x-axis
choice stops mattering.

No deduplication (user decision 2026-07-24): spans are matched on length only.

Usage::

    hist_a = length_histogram(pairs_a.lengths)
    hist_b = length_histogram(pairs_b.lengths)
    target = scale_to(hist_a, n_pairs)                 # H* = cell A's shape at rung size n
    idx_b  = match_indices(pairs_b.lengths, target, seed=0)
    matched_b = pairs_b.select(idx_b)
    assert histogram_fingerprint(matched_b.lengths) == histogram_fingerprint(matched_a.lengths)
"""

import hashlib

import torch
from jaxtyping import Int
from torch import Tensor

#: Octave bin edges over the span-length range. ``max_len=512`` in training, and neither
#: on-policy cell produces spans past the 512-token generation cap, so 512 is the top edge.
OCTAVES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def octave_of(length: int) -> int:
    """Largest octave edge <= length (the bin label)."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    lo = OCTAVES[0]
    for edge in OCTAVES:
        if length >= edge:
            lo = edge
    return lo


def length_histogram(lengths: Int[Tensor, "n"]) -> dict[int, int]:
    """Span count per octave bin. Bins with zero mass are present with count 0."""
    hist = dict.fromkeys(OCTAVES, 0)
    for v in lengths.tolist():
        hist[octave_of(int(v))] += 1
    return hist


def scale_to(hist: dict[int, int], n: int) -> dict[int, int]:
    """``hist``'s shape scaled to exactly ``n`` total (largest-remainder rounding).

    This is how a reference cell's realized shape becomes the per-bin quota ``H*`` for a rung of
    size ``n``: proportions are preserved, the rounding error is spread over the largest
    remainders, and the total is exact so two cells scaled to the same ``n`` get identical quotas.
    """
    total = sum(hist.values())
    if total == 0:
        raise ValueError("empty histogram")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    raw = {b: n * c / total for b, c in hist.items()}
    out = {b: int(v) for b, v in raw.items()}
    short = n - sum(out.values())
    for b in sorted(raw, key=lambda b: raw[b] - int(raw[b]), reverse=True)[:short]:
        out[b] += 1
    return out


def match_indices(
    lengths: Int[Tensor, "n"], target: dict[int, int], *, seed: int
) -> Int[Tensor, "m"]:
    """Row indices realizing exactly the ``target`` per-bin quotas (seeded, without replacement).

    Raises if any bin cannot meet its quota — a matched comparison with a silently-short bin is
    exactly the failure mode this module exists to prevent, so infeasibility is an error, never a
    truncation. The returned indices are shuffled (same generator), so downstream prefix-taking
    does not correlate with bin order.
    """
    gen = torch.Generator().manual_seed(seed)
    bins: dict[int, list[int]] = {b: [] for b in OCTAVES}
    for i, v in enumerate(lengths.tolist()):
        bins[octave_of(int(v))].append(i)
    picked: list[int] = []
    for b, want in sorted(target.items()):
        have = len(bins.get(b, []))
        if want > have:
            raise ValueError(f"bin {b}: need {want} spans, only {have} available")
        if want == 0:
            continue
        pool = torch.tensor(bins[b], dtype=torch.long)
        picked.extend(pool[torch.randperm(len(pool), generator=gen)[:want]].tolist())
    order = torch.randperm(len(picked), generator=gen)
    return torch.tensor(picked, dtype=torch.long)[order]


def histogram_fingerprint(lengths: Int[Tensor, "n"]) -> str:
    """Deterministic hash of the octave histogram — the cross-cell equality assertion.

    Two cells at the same rung must have IDENTICAL fingerprints before training starts; jobs
    refuse to run otherwise (scaling_plan.md §3). Hashing the histogram rather than comparing
    floats makes the check exact and loggable in ``runs/<name>.json``.
    """
    hist = length_histogram(lengths)
    blob = ",".join(f"{b}:{hist[b]}" for b in OCTAVES)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
