"""Batched lens readout + lens storage.

Readout: pure functions over tensors, vectorised over every layer at once (the official
``jlens.lens.JacobianLens.apply`` reads one layer of one prompt at a time — fine for
inspection, unusable for corpus-scale scans). Logit lens is the J-lens with Jl := I:

    lens(h_l) = softmax( W_U · norm( h_l ) )

The model-specific ``norm`` is passed in as a callable.

Storage: the on-disk format is the official ``jlens`` ``.pt`` layout
(``JacobianLens.save``/``load``/``from_pretrained``-compatible), with this repo's provenance
embedded under an extra ``"provenance"`` key that the official loader ignores. Legacy
safetensors artifacts (pre-2026-07) are still readable; new saves are always ``.pt``.
External lenses (e.g. the prefitted set on ``neuronpedia/jacobian-lens``) load via
``jlens.lens.JacobianLens.from_pretrained`` and enter the batched math through
:func:`stacked_jacobians`.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import einops
import torch
from jaxtyping import Float, Int
from safetensors import safe_open
from torch import Tensor

from jlens.lens import JacobianLens


def logit_lens_logits(
    resid: Float[Tensor, "layer pos d_model"],
    w_u: Float[Tensor, "d_vocab d_model"],
    norm: Callable[[Tensor], Tensor],
) -> Float[Tensor, "layer pos d_vocab"]:
    """Pre-softmax logit-lens scores: ``W_U · norm(h_l)``, no softmax.

    The signed magnitude survives here — the softmax in :func:`logit_lens` squashes
    tail tokens to ~0 over a 250k vocab, so anything reading individual (non-top-k)
    tokens (e.g. a tracked-token table) wants the logit, not the probability.

    Lens math runs in fp32 regardless of the inputs' dtype: einsum does not promote a bf16
    operand against an fp32 one, so the cast lives here, not in every caller.
    """
    normed = norm(resid.float())
    return einops.einsum(
        normed.float(), w_u.float(), "layer pos d_model, d_vocab d_model -> layer pos d_vocab"
    )


def logit_lens(
    resid: Float[Tensor, "layer pos d_model"],
    w_u: Float[Tensor, "d_vocab d_model"],
    norm: Callable[[Tensor], Tensor],
) -> Float[Tensor, "layer pos d_vocab"]:
    """Per-(layer, position) token distribution implied by reading h_l as a final-layer rep."""
    return logit_lens_logits(resid, w_u, norm).softmax(dim=-1)


def jacobian_lens_logits(
    resid: Float[Tensor, "layer pos d_model"],
    jacobians: Float[Tensor, "layer d_out d_in"],
    w_u: Float[Tensor, "d_vocab d_model"],
    norm: Callable[[Tensor], Tensor],
) -> Float[Tensor, "layer pos d_vocab"]:
    """Pre-softmax J-lens scores ``W_U · norm(Jl · h_l)`` (cf. :func:`jacobian_lens`)."""
    transported = einops.einsum(
        jacobians.float(), resid.float(), "layer d_out d_in, layer pos d_in -> layer pos d_out"
    )
    return logit_lens_logits(transported, w_u, norm)


def jacobian_lens(
    resid: Float[Tensor, "layer pos d_model"],
    jacobians: Float[Tensor, "layer d_out d_in"],
    w_u: Float[Tensor, "d_vocab d_model"],
    norm: Callable[[Tensor], Tensor],
) -> Float[Tensor, "layer pos d_vocab"]:
    """J-lens readout: softmax(W_U · norm(Jl · h_l)).

    ``jacobians[l]`` transports h_l to final-layer geometry (the averaged linear map standing
    in for layers l+1..L); the rest is exactly the logit lens. With ``jacobians`` = identity
    this reduces to :func:`logit_lens`.
    """
    return jacobian_lens_logits(resid, jacobians, w_u, norm).softmax(dim=-1)


def jacobian_lens_with_fallback(
    resid: Float[Tensor, "layer pos d_model"],
    jacobians: Float[Tensor, "covered d_out d_in"],
    w_u: Float[Tensor, "d_vocab d_model"],
    norm: Callable[[Tensor], Tensor],
) -> Float[Tensor, "layer pos d_vocab"]:
    """J-lens on the layers the artifact covers, logit lens beyond the target block.

    A J artifact spans layers ``0..target``; later layers have no Jacobian to the target by
    construction, and there the logit lens (``J = I``) is the natural readout. This wrapper
    lets the full residual stack flow through one readout callable (the visualizer seam).
    """
    covered = jacobians.shape[0]
    j_part = jacobian_lens(resid[:covered], jacobians, w_u, norm)
    if covered == resid.shape[0]:
        return j_part
    return torch.cat([j_part, logit_lens(resid[covered:], w_u, norm)], dim=0)


def jlens_vector(
    jacobians: Float[Tensor, "layer d_out d_in"],
    w_u: Float[Tensor, "d_vocab d_out"],
    token_id: int,
    *,
    layer: int | None = None,
) -> Tensor:
    """The J-lens *direction* for ``token_id`` in residual space: ``W_U[token_id] @ J[l]``.

    ``jacobian_lens`` reads a residual via ``W_U·(J·h)``; the equivalent per-token direction is
    ``v = Jᵀ W_U[t]`` so that ``<h, v>`` is that token's logit. Returns the ``[d_in]`` vector at
    ``layer`` (when given) or the per-layer stack ``[layer, d_in]`` over every covered layer — the
    steering/ablation/swap handle used by the causal scripts.
    """
    row = w_u[token_id].float()
    if layer is not None:
        return row @ jacobians[layer].float()
    return torch.stack([row @ jacobians[ell].float() for ell in range(jacobians.shape[0])])


def jlens_token_norms(
    jacobians: Float[Tensor, "layer d_out d_in"],
    w_u: Float[Tensor, "d_vocab d_out"],
) -> Float[Tensor, "layer d_vocab"]:
    """‖Jᵀ W_U[t]‖ per (layer, token) — the :func:`cosine_readout` denominator.

    Each layer is a full ``[d_vocab, d_model]`` matmul, so callers ranking many prompts against
    the same artifact should compute this once and pass it back via ``cosine_readout(denom=...)``.
    """
    return torch.stack(
        [(w_u @ jacobians[ell]).norm(dim=1) for ell in range(jacobians.shape[0])]
    ).clamp_min(1e-9)


def cosine_readout(
    jacobians: Float[Tensor, "layer d_out d_in"],
    resid: Float[Tensor, "layer pos d_in"],
    w_u: Float[Tensor, "d_vocab d_out"],
    *,
    denom: Float[Tensor, "layer d_vocab"] | None = None,
) -> Float[Tensor, "layer pos d_vocab"]:
    """Magnitude-free projection readout ``(W_U·J·h) / ‖Jᵀ W_U[t]‖`` (no RMSNorm, no softmax).

    Companion to :func:`jacobian_lens`: dividing out each token's transported-unembedding norm
    removes the high-norm "glitch" tokens that dominate the softmax ranking in the upper-mid band,
    giving the report's "projection onto the J-lens vector". Returns raw scores for ranking, not a
    distribution. Inputs are assumed fp32 (callers cast).
    """
    transported = einops.einsum(
        jacobians, resid, "layer d_out d_in, layer pos d_in -> layer pos d_out"
    )
    logits = einops.einsum(transported, w_u, "layer pos d_out, vocab d_out -> layer pos vocab")
    if denom is None:
        denom = jlens_token_norms(jacobians, w_u)
    return logits / denom.unsqueeze(1)


def top_k(probs: Float[Tensor, "d_vocab"], k: int = 5) -> tuple[list[int], list[float]]:
    """Top-k token ids and their probabilities for a single distribution."""
    vals, idx = torch.topk(probs, k)
    return idx.tolist(), vals.tolist()


def grid_top_k(
    readout: Float[Tensor, "layer pos d_vocab"], k: int = 5
) -> tuple[Int[Tensor, "layer pos k"], Float[Tensor, "layer pos k"]]:
    """Top-k token ids and probabilities at every (layer, position).

    Vectorised `top_k` over the whole readout — the bridge from a lens readout to the
    visualizer, which only ever needs the few highest-scoring tokens per cell.
    """
    vals, idx = torch.topk(readout, k, dim=-1)
    return idx, vals


# --------------------------------------------------------------------------- #
# Lens storage (official jlens .pt format; legacy safetensors still readable)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LensMeta:
    """Everything needed to interpret a fitted lens: model, corpus, recipe, provenance.

    Embedded in the ``.pt`` under a ``"provenance"`` key (ignored by the official loader);
    legacy safetensors artifacts carry the same fields in their header (where
    ``target_layer`` was historically stored as ``target_block``).
    """

    model_id: str
    dataset_id: str
    target_layer: int  # which block's output the Jacobians map to (normalized, >= 0)
    t_max: int
    n_prompts: int  # prompts accumulated so far
    docs_consumed: int  # corpus docs consumed (>= n_prompts; short docs are skipped)
    n_positions: float  # Σ of applied source weights — the normalizer (= Σ B·K for uniform)
    git_commit: str
    skip_first: int = 0  # source positions excluded per sequence (0 in pre-2026-06-12 artifacts)
    config_json: str = "{}"  # resolved CLI config, for provenance
    weighting: str = "uniform"  # source-weighting scheme of legacy ablation artifacts (Q1)
    corpus_mode: str = "pretrain"  # "pretrain" | "chat_full" | "chat_assistant" (Q5)

    def to_provenance(self) -> dict[str, str | int | float]:
        return dict(asdict(self))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LensMeta":
        """Parse provenance from a ``.pt`` payload (typed values) or a legacy safetensors
        header (all-string values, ``target_layer`` under its old name ``target_layer``)."""
        target = mapping.get("target_layer", mapping.get("target_block"))
        if target is None:
            raise ValueError("lens provenance has neither target_layer nor target_block")
        return cls(
            model_id=str(mapping["model_id"]),
            dataset_id=str(mapping["dataset_id"]),
            target_layer=int(target),
            t_max=int(mapping["t_max"]),
            n_prompts=int(mapping["n_prompts"]),
            docs_consumed=int(mapping["docs_consumed"]),
            n_positions=float(mapping.get("n_positions", 0)),
            git_commit=str(mapping["git_commit"]),
            skip_first=int(mapping.get("skip_first", 0)),
            config_json=str(mapping.get("config_json", "{}")),
            weighting=str(mapping.get("weighting", "uniform")),
            corpus_mode=str(mapping.get("corpus_mode", "pretrain")),
        )


def save_jacobians(
    path: Path,
    jacobians: Float[Tensor, "layer d_out d_in"],
    meta: LensMeta,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Write the J stack in the official ``jlens`` ``.pt`` layout, provenance embedded.

    The payload is exactly what ``jlens.lens.JacobianLens.save`` writes (fp16 by default:
    entries are O(1), so fp16's extra mantissa bits beat bf16), plus a ``"provenance"`` key
    the official ``load``/``from_pretrained`` ignore. Rows are layers ``0..len-1``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = {layer: jacobians[layer].to(dtype).contiguous() for layer in range(len(jacobians))}
    torch.save(
        {
            "J": stack,
            "n_prompts": meta.n_prompts,
            "source_layers": sorted(stack),
            "d_model": int(jacobians.shape[-1]),
            "provenance": meta.to_provenance(),
        },
        path,
    )


def load_jacobians(
    path: Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Float[Tensor, "layer d_out d_in"], LensMeta]:
    """Read a J stack (cast to ``dtype`` for lens math) and its provenance.

    Accepts both the official ``.pt`` layout (with embedded provenance — every lens this
    repo saves) and the legacy safetensors layout. A ``.pt`` fitted elsewhere carries no
    provenance and is rejected here — load it with
    ``jlens.lens.JacobianLens.from_pretrained`` and bridge via :func:`stacked_jacobians`.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            jacobians = f.get_tensor("jacobians").to(dtype)
            header = f.metadata()
            if header is None:
                raise ValueError(f"{path} has no metadata header")
            return jacobians, LensMeta.from_mapping(header)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in payload:
        raise ValueError(f"{path} is not a lens file (keys: {sorted(payload)!r})")
    if "provenance" not in payload:
        raise ValueError(
            f"{path} carries no provenance (an externally fitted lens?) — load it with "
            "jlens.lens.JacobianLens.from_pretrained and use stacked_jacobians()"
        )
    lens = JacobianLens(
        jacobians=payload["J"], n_prompts=payload["n_prompts"], d_model=payload["d_model"]
    )
    stack = stacked_jacobians(lens, device=device, dtype=dtype)
    return stack, LensMeta.from_mapping(payload["provenance"])


def stacked_jacobians(
    lens: JacobianLens,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Float[Tensor, "layer d_out d_in"]:
    """A contiguous ``[layer, d, d]`` stack for the batched readout functions above.

    Requires ``source_layers`` to be contiguous from 0 (true of every published lens);
    the batched math indexes rows by layer, so gaps would silently misalign readouts.
    """
    layers = lens.source_layers
    if layers != list(range(len(layers))):
        raise ValueError(f"lens source_layers are not contiguous from 0: {layers}")
    return torch.stack([lens.jacobians[layer] for layer in layers]).to(device=device, dtype=dtype)


def to_jacobian_lens(
    jacobians: Float[Tensor, "layer d_out d_in"], *, n_prompts: int
) -> JacobianLens:
    """Wrap a J stack as an official ``JacobianLens`` (rows = layers ``0..len-1``) — the
    bridge to ``apply``/``merge``/``save`` and everything else upstream provides."""
    return JacobianLens(
        jacobians={layer: jacobians[layer] for layer in range(len(jacobians))},
        n_prompts=n_prompts,
        d_model=int(jacobians.shape[-1]),
    )


# Directory names that are mount points / data-flow roots, never part of a run's identity.
_GENERIC_ROOTS = frozenset({"", "outputs", "out", "output", "vol", "data", "results", "tmp"})


def artifact_name(jac_path: Path | str) -> str:
    """A clean run name for a lens artifact, used to key its eval outputs/plots.

    Handles both the nested ``{root}/{name}/{run}/lens.pt`` (→ ``{name}_{run}``) and the
    flat ``{root}/{name}/lens.pt`` (→ ``{name}``). A generic root dir (``outputs``,
    ``vol``, …) or the filesystem root is never folded in, so ``/vol/outputs/foo/lens.pt``
    → ``foo``, not ``outputs_foo``.
    """
    path = Path(jac_path)
    artifact_dir = path.parent if path.suffix in {".safetensors", ".pt"} else path
    parent = artifact_dir.parent
    if parent.name in _GENERIC_ROOTS or parent == parent.parent:
        return artifact_dir.name
    return f"{parent.name}_{artifact_dir.name}"
