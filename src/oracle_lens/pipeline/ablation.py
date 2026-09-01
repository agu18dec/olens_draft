"""Whitened-space direction ablation of activations — the multi-token robustness test.

The hack this tests for: an oracle lens that reads only the FIRST constituent's direction from
the activation ("Lake") and autocompletes the rest ("Michigan") from the LM prior would score
high FVE on multi-token concepts without genuinely reading them. The test: project the
frozen-AR direction of ONE constituent out of ``h`` (in whitened space — the FVE metric
space), re-run the lens on the ablated vector, and check whether the readout / FVE for that
constituent actually collapses (vs a matched random-direction control).

Conventions introduced here (no prior projection utility exists in the repo):

- All geometry happens in whitened space: ``x_w = W(x - μ)`` with the layer's own
  :class:`~oracle_lens.core.whitening.Whitener`; results map back through
  ``W⁻¹ = (Σ+λI)^{1/2}`` (:func:`~oracle_lens.core.whitening.unwhitening_matrix`).
- A concept/constituent direction is the whitened, centered frozen-AR prediction of the bare
  span: ``û ∝ W(AR(span)[layer] - μ)``.
- Ablation is a rank-1 orthogonal projection: ``h_w' = h_w - (h_w·û)û``.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from oracle_lens.core.whitening import (
    Whitener,
    unwhitening_matrix,
    whitening_matrix,
)


@dataclass(frozen=True)
class WhitenedSpace:
    """A layer's whitening transform together with its exact inverse."""

    whitener: Whitener
    w_inv: Float[Tensor, "d d"]

    @classmethod
    def from_moments(
        cls, mu: Float[Tensor, "d"], cov: Float[Tensor, "d d"], *, ridge_c: float
    ) -> "WhitenedSpace":
        whitener = Whitener(mu=mu, w=whitening_matrix(cov, ridge_c=ridge_c), ridge_c=ridge_c)
        return cls(whitener=whitener, w_inv=unwhitening_matrix(cov, ridge_c=ridge_c))

    def whiten(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]:
        return self.whitener.whiten(x)

    def unwhiten(self, xw: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]:
        w_inv = self.w_inv.to(xw.device)
        mu = self.whitener.mu.to(xw.device)
        return xw.float() @ w_inv.T + mu

    def to(self, device: str | torch.device) -> "WhitenedSpace":
        return WhitenedSpace(whitener=self.whitener.to(device), w_inv=self.w_inv.to(device))


def unit(v: Float[Tensor, "d"]) -> Float[Tensor, "d"]:
    """v / ‖v‖ with a clamp against zero vectors."""
    out: Tensor = v / v.norm().clamp_min(1e-9)
    return out


def whitened_direction(space: WhitenedSpace, d_raw: Float[Tensor, "d"]) -> Float[Tensor, "d"]:
    """Unit direction of a raw activation prediction (e.g. ``AR(span)[layer]``) in whitened
    space. Centering by μ is part of the whitening — the direction is relative to the mean
    activation, not the origin."""
    return unit(space.whiten(d_raw.unsqueeze(0))[0])


def random_unit(d_model: int, generator: torch.Generator) -> Float[Tensor, "d"]:
    """Isotropic random unit direction in whitened space (the matched control)."""
    return unit(torch.randn(d_model, generator=generator))


def project_out_rows(
    space: WhitenedSpace, h: Float[Tensor, "n d"], d_raw: Float[Tensor, "n d"]
) -> tuple[Float[Tensor, "n d"], Float[Tensor, "n"]]:
    """Per-row rank-1 whitened ablation: row ``i`` loses ITS OWN direction (the deflation-chain
    step, where every row removes a different phrase's AR image).

    ``d_raw`` rows are raw-space AR predictions; each becomes a unit direction via
    :func:`whitened_direction`'s convention (whiten ⇒ centered by μ, then normalize). Returns
    ``(h_ablated_raw, coef)`` with ``coef[i] = whiten(h[i])·û_i`` (signed);
    ``whiten(h_ablated[i])·û_i == 0`` by construction.
    """
    h_w = space.whiten(h)
    u = space.whiten(d_raw)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    coef = (h_w * u).sum(-1)
    h_w_abl = h_w - coef.unsqueeze(-1) * u
    return space.unwhiten(h_w_abl), coef


def project_out(
    space: WhitenedSpace, h: Float[Tensor, "n d"], u_hat: Float[Tensor, "d"]
) -> tuple[Float[Tensor, "n d"], Float[Tensor, "n"]]:
    """Rank-1 whitened-space ablation of ``û`` from each row of ``h`` (raw activations).

    Returns ``(h_ablated, coef)`` where ``coef[i] = whiten(h[i])·û`` is the removed whitened
    component (signed — its magnitude is the "how much of this direction was present" gate
    quantity). ``whiten(h_ablated)·û == 0`` exactly by construction.
    """
    u_hat = unit(u_hat.float()).to(h.device)
    h_w = space.whiten(h)
    coef = h_w @ u_hat
    h_w_abl = h_w - coef.unsqueeze(-1) * u_hat.unsqueeze(0)
    return space.unwhiten(h_w_abl), coef
