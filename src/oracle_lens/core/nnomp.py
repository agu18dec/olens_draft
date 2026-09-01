"""Batched non-negative orthogonal matching pursuit (the oracle-lens teacher).

Decomposes each whitened activation x as a sparse non-negative combination of dictionary
directions (paper: up to sixteen phrases, one phrase length at a time, random half of the
dictionary per datapoint). Greedy: pick the atom with the largest positive correlation to the
residual, then re-fit ALL selected coefficients by non-negative least squares (projected
gradient on the small k x k Gram), and repeat until max_atoms or the FVE gain stalls.

Fully batched (gather + bmm) so 100k activations against a ~100k-atom dictionary run in
minutes on one GPU; pure torch, so the unit tests exercise the exact production function.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


@dataclass(frozen=True)
class NNOMPResult:
    atoms: Int[Tensor, "b k"]  # selection order; -1 where unused
    coeffs: Float[Tensor, "b k"]  # NNLS coefficients (0 where unused)
    fve: Float[Tensor, "b"]  # fraction of variance explained: 1 - ||r||^2 / ||x||^2


def _nnls_batched(
    dsel: Float[Tensor, "b k d"],
    x: Float[Tensor, "b d"],
    valid: Bool[Tensor, "b k"],
    *,
    iters: int = 80,
) -> Float[Tensor, "b k"]:
    """Non-negative least squares per row via projected gradient on the k x k Gram.

    Step size 1/trace(G) (trace >= lambda_max for PSD, so always stable; k <= 16 keeps the
    conservative rate cheap). Invalid (not-yet-selected) slots are masked to zero throughout.
    """
    gram = torch.bmm(dsel, dsel.transpose(1, 2))  # [b, k, k]
    rhs = torch.bmm(dsel, x.unsqueeze(-1)).squeeze(-1)  # [b, k]
    step = 1.0 / gram.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-12)  # [b]
    w = torch.zeros_like(rhs)
    vf = valid.to(dsel.dtype)  # match the input dtype — .float() silently promoted w on the
    # first update and broke the bf16 scoring path (bmm dtype mismatch, r2s m1 2026-08-01)
    for _ in range(iters):
        grad = torch.bmm(gram, w.unsqueeze(-1)).squeeze(-1) - rhs
        w = ((w - step.unsqueeze(-1) * grad) * vf).clamp_min(0.0)
    return w


def nnomp_batch(
    x: Float[Tensor, "b d"],
    dictionary: Float[Tensor, "n d"],
    *,
    max_atoms: int = 16,
    min_gain: float = 1e-3,
    mask: Bool[Tensor, "b n"] | None = None,
    chunk: int = 8192,
) -> NNOMPResult:
    """Decompose each row of ``x`` against unit-normalized ``dictionary`` rows.

    ``mask`` True = atom usable for that row (the paper's random-half restriction).
    Correlations are computed in ``chunk``-column pieces so a large dictionary never
    materializes [b, n] beyond one chunk at a time... (it does materialize [b, n] for argmax;
    chunking bounds the intermediate matmul workspace, not the scores).

    ``dictionary`` may live on a DIFFERENT device than ``x`` (r2sp: a >VRAM prefix-expanded
    dict held in host RAM): column chunks are streamed to ``x.device`` per matmul and the
    per-step atom gather is done on the dictionary's device. Math is identical to the
    same-device path — only transfer placement differs.
    """
    b, _ = x.shape
    n = dictionary.shape[0]
    device = x.device
    stream = dictionary.device != device
    atoms = torch.full((b, max_atoms), -1, dtype=torch.long, device=device)
    coeffs = torch.zeros(b, max_atoms, device=device)
    x_norm2 = (x**2).sum(-1).clamp_min(1e-12)
    residual = x.clone()
    active = torch.ones(b, dtype=torch.bool, device=device)
    prev_fve = torch.zeros(b, device=device)

    for step_idx in range(max_atoms):
        # correlations of the residual with every atom (chunked over dictionary rows)
        scores = torch.empty(b, n, device=device)
        for lo in range(0, n, chunk):
            dchunk = dictionary[lo : lo + chunk]
            if stream:
                dchunk = dchunk.to(device, non_blocking=True)
            scores[:, lo : lo + chunk] = residual @ dchunk.T
        if mask is not None:
            scores.masked_fill_(~mask, float("-inf"))
        for k in range(step_idx):  # never reselect
            sel = atoms[:, k]
            ok = sel >= 0
            scores[torch.nonzero(ok).squeeze(-1), sel[ok]] = float("-inf")
        best_score, best_idx = scores.max(dim=-1)
        # no selection when the residual is already explained (float-noise correlations
        # would otherwise append ~zero-coefficient atoms before the gain check fires)
        take = active & (best_score > 0) & (prev_fve < 1.0 - min_gain)
        if not bool(take.any()):
            break
        atoms[:, step_idx] = torch.where(take, best_idx, atoms[:, step_idx])

        valid = atoms >= 0  # [b, max_atoms]
        gather_idx = atoms.clamp_min(0)
        if stream:
            dsel = dictionary[gather_idx.to(dictionary.device)].to(device)  # [b, max_atoms, d]
        else:
            dsel = dictionary[gather_idx]  # [b, max_atoms, d]
        w = _nnls_batched(dsel, x, valid)
        # zero-coefficient atoms stay recorded but contribute nothing (paper drops them;
        # keeping the slot preserves selection order — filter w==0 downstream)
        recon = torch.bmm(w.unsqueeze(1), dsel).squeeze(1)
        residual = x - recon
        fve = 1.0 - (residual**2).sum(-1) / x_norm2
        gained = fve - prev_fve
        active = take & (gained > min_gain)
        prev_fve = torch.where(take, fve, prev_fve)
        coeffs = torch.where(valid, w, coeffs)
        # rows that stopped keep their previous residual implicitly via `active` gating:
        # their atoms[:, step_idx] stayed -1, so valid/w for them are unchanged.

    return NNOMPResult(atoms=atoms, coeffs=coeffs, fve=prev_fve)


def nnls_refit(
    vectors: Float[Tensor, "b k d"],
    x: Float[Tensor, "b d"],
    valid: Bool[Tensor, "b k"],
) -> tuple[Float[Tensor, "b k"], Float[Tensor, "b"]]:
    """Non-negative least-squares refit of given vectors against x -> (coeffs, FVE).

    The oracle-eval scorer: the oracle proposes phrases, the reconstructor maps them to
    vectors, and this measures how much of the activation they jointly explain.
    """
    w = _nnls_batched(vectors, x, valid)
    recon = torch.bmm(w.unsqueeze(1), vectors).squeeze(1)
    fve = 1.0 - ((x - recon) ** 2).sum(-1) / (x**2).sum(-1).clamp_min(1e-12)
    return w, fve


def half_mask(
    b: int, n: int, *, seed: int, device: str | torch.device = "cpu"
) -> Bool[Tensor, "b n"]:
    """The paper's per-datapoint random-half dictionary restriction (seeded, reproducible)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(b, n, generator=gen) < 0.5).to(device)
