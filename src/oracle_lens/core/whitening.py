"""Whitening transform for the reconstructor's loss metric.

The artifact stores μ and Σ (ridge-free) so the ridge coefficient stays a *training-time* knob:
``W_c = (Σ + λI)^{-1/2}`` with ``λ = ridge_c · mean(diag Σ)`` is rebuilt per run from the same
moments. ``mean(diag Σ)`` is the average per-dimension activation variance, so ``ridge_c`` is a
dimensionless fraction of typical variance (transfers across layers/models; same convention as
``template_lens.templates``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from safetensors.torch import load_file, save_file
from torch import Tensor


def whitening_matrix(cov: Float[Tensor, "d d"], *, ridge_c: float) -> Float[Tensor, "d d"]:
    """W = (Σ + λI)^{-1/2} via symmetric eigendecomposition in fp64, returned fp32.

    λ = ridge_c · mean(diag Σ), clamped away from zero. Eigenvalues are clamped at λ's floor so
    a numerically-negative tail (rank-deficient estimate) cannot produce NaNs.
    """
    cov64 = cov.to(torch.float64)
    cov64 = 0.5 * (cov64 + cov64.T)
    lam = torch.clamp_min(ridge_c * cov64.diagonal().mean(), 1e-12)
    eigvals, eigvecs = torch.linalg.eigh(cov64)
    inv_sqrt = (eigvals.clamp_min(0.0) + lam).rsqrt()
    w: Tensor = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T
    return w.to(torch.float32)


def unwhitening_matrix(cov: Float[Tensor, "d d"], *, ridge_c: float) -> Float[Tensor, "d d"]:
    """W⁻¹ = (Σ + λI)^{1/2} — the exact inverse of :func:`whitening_matrix` (same eigenbasis,
    same λ floor and eigenvalue clamp), so ``unwhiten(whiten(x)) == x`` to fp32 precision.
    Needed by the activation-ablation path, which projects in whitened space and must map the
    result back to raw activation space for re-injection.
    """
    cov64 = cov.to(torch.float64)
    cov64 = 0.5 * (cov64 + cov64.T)
    lam = torch.clamp_min(ridge_c * cov64.diagonal().mean(), 1e-12)
    eigvals, eigvecs = torch.linalg.eigh(cov64)
    sqrt = (eigvals.clamp_min(0.0) + lam).sqrt()
    w_inv: Tensor = (eigvecs * sqrt.unsqueeze(0)) @ eigvecs.T
    return w_inv.to(torch.float32)


@dataclass(frozen=True)
class Whitener:
    """Frozen whitening transform: whiten(x) = (x - μ) @ Wᵀ (W symmetric)."""

    mu: Float[Tensor, "d"]
    w: Float[Tensor, "d d"]
    ridge_c: float

    def whiten(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]:
        return (x.float() - self.mu.to(x.device)) @ self.w.to(x.device).T

    def to(self, device: str | torch.device) -> "Whitener":
        return Whitener(mu=self.mu.to(device), w=self.w.to(device), ridge_c=self.ridge_c)


def save_moments(
    path: Path,
    mu: Float[Tensor, "d"],
    cov: Float[Tensor, "d d"],
    *,
    meta: dict[str, str],
) -> None:
    """Persist μ/Σ (fp32) + provenance; the ridge is applied at load, not here."""
    save_file({"mu": mu.float().cpu(), "cov": cov.float().cpu()}, str(path), metadata=meta)


def load_whitener(path: Path, *, ridge_c: float, device: str = "cpu") -> Whitener:
    tensors = load_file(str(path), device="cpu")
    w = whitening_matrix(tensors["cov"], ridge_c=ridge_c)
    return Whitener(mu=tensors["mu"].to(device), w=w.to(device), ridge_c=ridge_c)


def pool_moments(
    parts: Sequence[tuple[int, Float[Tensor, "d"], Float[Tensor, "d d"]]],
) -> tuple[int, Float[Tensor, "d"], Float[Tensor, "d d"]]:
    """Exact moments of the concatenation of k groups, from their ``(n, μ, Σ)`` alone.

    Σ is the *population* covariance (denominator ``n``, as produced by
    :meth:`~oracle_lens.core.stats.MomentAccumulator.covariance`), so the pooled
    covariance is within-group plus between-group scatter::

        n = Σ nᵢ
        μ = Σ nᵢ μᵢ / n
        Σ = Σ nᵢ Σᵢ / n  +  Σ nᵢ (μᵢ - μ)(μᵢ - μ)ᵀ / n

    The between-group term is what makes this a *domain-neutral* basis: averaging the Σᵢ alone
    silently discards the cross-domain mean shift and understates the pooled variance. Computed in
    fp64 and returned fp64; this form avoids the ~4-5 digit cancellation of accumulating
    ``Σᵢ + μᵢμᵢᵀ`` and subtracting ``μμᵀ`` back off when ``‖μ‖² ≫ mean(diag Σ)``.
    """
    if not parts:
        raise ValueError("pool_moments needs at least one (n, mu, cov) part")
    d = parts[0][1].shape[0]
    for i, (n_i, mu_i, cov_i) in enumerate(parts):
        if n_i <= 0:
            raise ValueError(f"part {i}: non-positive sample count {n_i}")
        if mu_i.shape != (d,) or cov_i.shape != (d, d):
            raise ValueError(
                f"part {i}: shape mismatch — mu {tuple(mu_i.shape)}, cov {tuple(cov_i.shape)}, "
                f"expected ({d},) and ({d}, {d})"
            )
    n = sum(int(n_i) for n_i, _, _ in parts)
    mu = torch.zeros(d, dtype=torch.float64)
    for n_i, mu_i, _ in parts:
        mu += float(n_i) * mu_i.to(torch.float64)
    mu /= n
    cov = torch.zeros(d, d, dtype=torch.float64)
    for n_i, mu_i, cov_i in parts:
        delta = mu_i.to(torch.float64) - mu
        cov += float(n_i) * (cov_i.to(torch.float64) + torch.outer(delta, delta))
    cov /= n
    cov = 0.5 * (cov + cov.T)
    return n, mu, cov
