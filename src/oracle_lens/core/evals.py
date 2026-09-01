"""Reconstructor evaluation: dumb baselines that gate every run.

- ``retrieval_top1``: is a prediction closer (whitened cosine) to its OWN target than to the
  other targets in a random pool? Chance = 1/pool. ≫ chance means the phrase input is actually
  used; ~chance with a passing overfit gate is the "actually broken" trigger for revisiting the
  input presentation.
- ``predict_mean_r2``: R² of predicting the train-set mean — ~0 on the eval set by construction;
  logged so every run carries its own floor.

The multi-tokenness final evals (phrase-beats-parts, first-token confound vs J-lens) build on
these and land with the eval-set builder (plan step 8).
"""

import torch
from jaxtyping import Float
from torch import Tensor

from oracle_lens.core.reconstructor import r2


def retrieval_top1(
    pred_w: Float[Tensor, "n d"],
    target_w: Float[Tensor, "n d"],
    *,
    pool: int = 100,
    generator: torch.Generator | None = None,
) -> float:
    """Fraction of rows whose prediction ranks its own target first among ``pool`` candidates.

    Candidates = own target + (pool-1) targets sampled from other rows. Cosine similarity in
    the whitened space. Deterministic under a seeded ``generator``.
    """
    n = pred_w.shape[0]
    if n < pool:
        raise ValueError(f"need at least pool={pool} rows, got {n}")
    p = pred_w / pred_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    t = target_w / target_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    hits = 0
    for i in range(n):
        perm = torch.randperm(n, generator=generator)
        others = perm[perm != i][: pool - 1]
        own = (p[i] @ t[i]).item()
        rival = (p[i] @ t[others].T).max().item()
        hits += int(own > rival)
    return hits / n


def predict_mean_r2(target_w: Float[Tensor, "n d"], train_mean_w: Float[Tensor, "d"]) -> float:
    """R² of the constant train-mean predictor on the eval targets (the floor, ≈0)."""
    pred = train_mean_w.unsqueeze(0).expand_as(target_w)
    return r2(pred, target_w)
