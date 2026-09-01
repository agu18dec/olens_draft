"""Jacobian-lens metric space: z(x) = (x - μ) @ Jᵀ — a drop-in "ruler" beside ``Whitener``.

``J`` is the corpus-averaged Jacobian-lens map for one source layer (``∂h_target/∂h_l``,
averaged over prompts and future positions), so cosines in this space compare activations by
their *downstream-readable* content rather than their statistically-whitened direction. The
method is deliberately named ``whiten`` so ``JSpace`` is structurally interchangeable with
``oracle_lens.whitening.Whitener`` (both are frozen affine maps used only inside cosine
losses/metrics); no covariance enters here.

Layer coverage: a J stack of depth ``n`` covers source layers ``0..n-1`` only — the target
block itself (layer 63 for the official Qwen3.6-27B artifact) has no Jacobian. ``load_jspaces``
returns a dict WITHOUT the uncovered layers; callers must account for them loudly (a ``None``
slot in a loss list), never average as if all layers were present.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class MetricSpace(Protocol):
    """A frozen affine map into the space where cosine losses/metrics are computed."""

    def whiten(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]: ...

    def to(self, device: str | torch.device) -> "MetricSpace": ...


@dataclass(frozen=True)
class JSpace:
    """z(x) = (x - μ) @ Jᵀ. J is NOT symmetric — the transpose convention matches
    ``jlens.lens.JacobianLens.transport`` (``residual @ J.T``)."""

    mu: Float[Tensor, "d"]
    j: Float[Tensor, "d d"]

    def whiten(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]:
        return (x.float() - self.mu.to(x.device)) @ self.j.to(x.device).T

    def to(self, device: str | torch.device) -> "JSpace":
        return JSpace(mu=self.mu.to(device), j=self.j.to(device))


@dataclass(frozen=True)
class MixedSpace:
    """``(1-λ)·L_whitened + λ·L_J`` expressed as a SINGLE ruler, so the mixed objective needs no
    change to the loss function or the trainer.

    ``whiten(x) = concat[√(1-λ)·unit(W(x-μ)), √λ·unit(J(x-μ))]``

    Each block is unit-normalised before weighting, so the concatenation has norm 1 and the
    caller's own unit-normalisation is a no-op. The standard ``2(1-cos)`` loss then evaluates to
    EXACTLY ``(1-λ)·2(1-cos_W) + λ·2(1-cos_J)`` — the λ-weighted sum of the two arms' losses
    (proved by the cross terms vanishing; asserted numerically in tests).

    NOTE this is a *loss* ruler only. Its cosine is a blend, so FVE measured through it is not
    the FVE of either space; the trainer logs whitened and J FVE through their own rulers.
    """

    whitener: MetricSpace
    jspace: MetricSpace
    lam: float

    def whiten(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n d2"]:
        w = self.whitener.whiten(x)
        j = self.jspace.whiten(x)
        uw = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        uj = j / j.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.cat([(1.0 - self.lam) ** 0.5 * uw, self.lam**0.5 * uj], dim=-1)

    def to(self, device: str | torch.device) -> "MixedSpace":
        return MixedSpace(self.whitener.to(device), self.jspace.to(device), self.lam)


def load_mu(path: Path) -> Float[Tensor, "d"]:
    """μ only, from a whitening-moments artifact (key ``mu``) — skips the fp64 eigh entirely."""
    from safetensors.torch import load_file

    return load_file(str(path))["mu"].float()


def load_jspaces(
    repo: str,
    filename: str,
    mu_dir: Path,
    layers: tuple[int, ...],
    *,
    revision: str | None = None,
    mu_prefix: str = "whitening",
) -> dict[int, JSpace]:
    """One ``JSpace`` per COVERED layer; uncovered layers (>= J-stack depth) are absent.

    The J stack is loaded fp32 (fp16 on disk); μ comes from the same per-layer moment artifacts
    the whiteners use (``{mu_prefix}_L{layer}.safetensors``) so centering is shared across arms.
    """
    from jlens.lens import JacobianLens
    from oracle_lens.jlens_readout import stacked_jacobians

    lens = JacobianLens.from_pretrained(repo, filename=filename, revision=revision)
    jac = stacked_jacobians(lens, device="cpu", dtype=torch.float32)  # [n_covered, d, d]
    out: dict[int, JSpace] = {}
    for lyr in layers:
        if lyr >= jac.shape[0]:
            continue
        mu = load_mu(mu_dir / f"{mu_prefix}_L{lyr}.safetensors")
        out[lyr] = JSpace(mu=mu, j=jac[lyr].contiguous())
    return out
