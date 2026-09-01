"""Pooling per-cell moments into one basis must be EXACT, not approximate.

The 2x2 AR comparison hangs on all four arms sharing one whitened ruler, so an error here silently
mis-scales every FVE in the headline figure.
"""

import pytest
import torch

from oracle_lens.core.stats import MomentAccumulator
from oracle_lens.core.whitening import pool_moments, whitening_matrix


def _groups(d: int = 24, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # Deliberately different means AND scales: pooling must capture the between-group shift.
    a = torch.randn(4000, d, generator=g, dtype=torch.float64) * 2.0 + 5.0
    b = torch.randn(2500, d, generator=g, dtype=torch.float64) * 0.5 - 3.0
    return [a, b]


def _moments(x: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0)
    xc = x - mu
    return x.shape[0], mu, (xc.T @ xc) / x.shape[0]  # population covariance, denominator n


def test_pool_moments_equals_moments_of_concatenation() -> None:
    parts = [_moments(x) for x in _groups()]
    n, mu, cov = pool_moments(parts)
    n_ref, mu_ref, cov_ref = _moments(torch.cat(_groups(), dim=0))
    assert n == n_ref
    torch.testing.assert_close(mu, mu_ref, atol=1e-10, rtol=0)
    torch.testing.assert_close(cov, cov_ref, atol=1e-10, rtol=0)


def test_pool_moments_agrees_with_accumulator_merge() -> None:
    """Independent route: rebuild accumulators from (n, mu, cov) and use the tested merge()."""
    parts = [_moments(x) for x in _groups()]
    accs = [MomentAccumulator.from_moments(n, mu, cov) for n, mu, cov in parts]
    merged = accs[0]
    for other in accs[1:]:
        merged.merge(other)
    n, mu, cov = pool_moments(parts)
    assert merged.count == n
    torch.testing.assert_close(merged.mean(), mu, atol=1e-8, rtol=0)
    torch.testing.assert_close(merged.covariance(), cov, atol=1e-6, rtol=0)


def test_between_group_term_is_not_dropped() -> None:
    """A naive average of the Sigmas understates pooled variance; guard against that regression."""
    parts = [_moments(x) for x in _groups()]
    n, _, cov = pool_moments(parts)
    within = sum(float(p[0]) * p[2].diagonal().mean() for p in parts) / n
    assert float(cov.diagonal().mean()) > float(within) * 1.05


def test_single_source_pooling_is_identity() -> None:
    n0, mu0, cov0 = _moments(_groups()[0])
    n, mu, cov = pool_moments([(n0, mu0, cov0)])
    assert n == n0
    torch.testing.assert_close(mu, mu0, atol=1e-12, rtol=0)
    torch.testing.assert_close(cov, cov0, atol=1e-12, rtol=0)


def test_pooled_basis_whitens_pooled_data() -> None:
    """With a ridge, W Sigma W^T has eigenvalues s/(s+lam) — NOT the identity.

    Pin the real target so nobody "fixes" the gate to assert isotropy and gets a false failure.
    """
    parts = [_moments(x) for x in _groups()]
    _, _mu, cov = pool_moments(parts)
    ridge_c = 0.1
    w = whitening_matrix(cov.float(), ridge_c=ridge_c).to(torch.float64)
    lam = ridge_c * float(cov.diagonal().mean())
    got = torch.linalg.eigvalsh(w @ cov @ w.T).sort().values
    s = torch.linalg.eigvalsh(cov).sort().values
    torch.testing.assert_close(got, s / (s + lam), atol=2e-6, rtol=0)


def test_pool_moments_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        pool_moments([])
    n0, mu0, cov0 = _moments(_groups()[0])
    with pytest.raises(ValueError, match="shape mismatch"):
        pool_moments([(n0, mu0, cov0), (10, mu0[:5], cov0[:5, :5])])
    with pytest.raises(ValueError, match="non-positive"):
        pool_moments([(0, mu0, cov0)])
