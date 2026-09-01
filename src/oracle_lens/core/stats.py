"""Streaming mean/covariance of one layer's activations (whitening statistics).

Single-layer specialization of the ``template_lens.accumulate`` pattern: additive fp64 sums, so
the Modal shard-merge is a plain add. fp64 (not the template lens's fp32) because the dump feeds
~10M samples whose fp32 running sum would lose ~7 digits of mantissa; at [d] + [d, d] for one
layer the fp64 memory cost is trivial (~210 MB for d=5120).
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass
class MomentAccumulator:
    """Accumulate sum(x) and sum(x xᵀ) over ``[n, d]`` batches; derive μ and Σ."""

    d_model: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.count: int = 0
        self.sum: Tensor = torch.zeros(self.d_model, dtype=torch.float64, device=self.device)
        self.outer_sum: Tensor = torch.zeros(
            self.d_model, self.d_model, dtype=torch.float64, device=self.device
        )

    def update(self, x: Float[Tensor, "n d"]) -> None:
        x64 = x.to(device=self.device, dtype=torch.float64)
        self.sum += x64.sum(dim=0)
        self.outer_sum += x64.T @ x64
        self.count += x64.shape[0]

    def mean(self) -> Float[Tensor, "d"]:
        if self.count == 0:
            raise RuntimeError("no activations accumulated")
        return self.sum / self.count

    def covariance(self) -> Float[Tensor, "d d"]:
        """Σ = E[xxᵀ] - μμᵀ (fp64; symmetrized against accumulation drift)."""
        mu = self.mean()
        cov = self.outer_sum / self.count - torch.outer(mu, mu)
        return 0.5 * (cov + cov.T)

    def merge(self, other: "MomentAccumulator") -> None:
        if other.d_model != self.d_model:
            raise ValueError(f"d_model mismatch: {self.d_model} vs {other.d_model}")
        self.sum += other.sum.to(self.sum.device)
        self.outer_sum += other.outer_sum.to(self.outer_sum.device)
        self.count += other.count

    @classmethod
    def from_moments(
        cls, n: int, mu: Float[Tensor, "d"], cov: Float[Tensor, "d d"], *, device: str = "cpu"
    ) -> "MomentAccumulator":
        """Rebuild an accumulator from published ``(n, μ, Σ)`` so :meth:`merge` can pool artifacts.

        Inverse of :meth:`mean`/:meth:`covariance`. Kept as the *cross-check* path for
        :func:`~oracle_lens.core.whitening.pool_moments`, not the production one:
        re-forming ``outer_sum = n(Σ + μμᵀ)`` and later subtracting ``μμᵀ`` loses precision when
        ``‖μ‖² ≫ mean(diag Σ)``.
        """
        acc = cls(d_model=int(mu.shape[0]), device=device)
        mu64 = mu.to(device=device, dtype=torch.float64)
        cov64 = cov.to(device=device, dtype=torch.float64)
        acc.count = int(n)
        acc.sum = float(n) * mu64
        acc.outer_sum = float(n) * (cov64 + torch.outer(mu64, mu64))
        return acc

    def save(self, path: Path) -> None:
        save_file(
            {
                "sum": self.sum.cpu(),
                "outer_sum": self.outer_sum.cpu(),
                "count": torch.tensor([self.count], dtype=torch.int64),
            },
            str(path),
            metadata={"d_model": str(self.d_model)},
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "MomentAccumulator":
        tensors = load_file(str(path), device="cpu")
        acc = cls(d_model=int(tensors["sum"].shape[0]), device=device)
        acc.sum = tensors["sum"].to(device=device, dtype=torch.float64)
        acc.outer_sum = tensors["outer_sum"].to(device=device, dtype=torch.float64)
        acc.count = int(tensors["count"].item())
        return acc
