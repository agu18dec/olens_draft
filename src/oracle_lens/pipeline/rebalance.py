"""Per-octave quota selection for the long oracle lens dataset (pure, seeded, deterministic).

The capture-time length law is log-uniform, but ``sample_long_spans`` clamps each draw to the
conversation's longest run, so the REALIZED distribution over-fills short octaves and
under-fills long ones. This selects a fixed per-bucket quota from the candidate pool so the
final dataset is (as close as supply allows to) exactly log-uniform: equal row counts per
``LENGTH_BUCKETS`` octave. Underfilled buckets pass through whole — never silently, the caller
reports the pre/post histogram (``shards.length_histogram``).
"""

import torch
from jaxtyping import Int
from torch import Tensor

from oracle_lens.pipeline.shards import LENGTH_BUCKETS


def octave_quota_select(
    lengths: Int[Tensor, " n"],
    *,
    quota_per_bucket: int,
    buckets: tuple[tuple[int, int], ...] = LENGTH_BUCKETS,
    seed: int = 0,
) -> Int[Tensor, " m"]:
    """Kept row indices: per bucket, a seeded sample of ``min(quota, available)`` rows.

    Deterministic under ``(lengths, quota_per_bucket, buckets, seed)``; returned indices are
    sorted ascending so downstream ``select`` preserves the pool's row order. Rows outside every
    bucket (length 0 or > the last bucket's hi) are dropped.
    """
    if quota_per_bucket < 1:
        raise ValueError(f"quota_per_bucket must be >= 1, got {quota_per_bucket}")
    kept: list[Tensor] = []
    for bi, (lo, hi) in enumerate(buckets):
        members = torch.nonzero((lengths >= lo) & (lengths <= hi)).squeeze(-1)
        if len(members) > quota_per_bucket:
            # independent generator per bucket: one bucket's fill state never perturbs another's
            gen = torch.Generator().manual_seed(seed * 1_000_003 + bi)
            members = members[torch.randperm(len(members), generator=gen)[:quota_per_bucket]]
        kept.append(members)
    return torch.sort(torch.cat(kept)).values
