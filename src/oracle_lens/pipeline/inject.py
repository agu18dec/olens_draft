"""Pure slot-injection primitives for the AO grid — the one place the representation sweep lives.

Kept free of the training harness (``mytorch_lightning``) on purpose: the readout/playground/serving
paths need only these functions, so they must import without pulling in the trainer. ``ao_train``
re-exports everything here for backward compatibility.
"""

from typing import Literal

from jaxtyping import Float
from torch import Tensor

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.jspace import MetricSpace

__all__ = ["GtTransform", "Rep", "fit_scale", "inject_gt", "inject_rep", "rep_source"]

# The 5 grid arms. Each maps to (source of the injected vector, transform applied to it).
Rep = Literal["raw_unit", "wht_unit", "j_unit", "raw_scaled", "real_h"]

# source: "ar_out" = a representation of AR(out) (on the AR manifold); "real_h" = the true stored
# preceding activation (control — tests AR(out) vs real h as the training signal).
_REP_SOURCE: dict[Rep, Literal["ar_out", "real_h"]] = {
    "raw_unit": "ar_out",
    "wht_unit": "ar_out",
    "j_unit": "ar_out",
    "raw_scaled": "ar_out",
    "real_h": "real_h",
}
# transform: how the raw source vector becomes the embedding-space injection.
_REP_TRANSFORM: dict[Rep, Literal["unit", "wht_unit", "j_unit", "scaled"]] = {
    "raw_unit": "unit",
    "wht_unit": "wht_unit",
    "j_unit": "j_unit",
    "raw_scaled": "scaled",
    "real_h": "unit",
}


def rep_source(rep: Rep) -> Literal["ar_out", "real_h"]:
    """Which vector feeds the slot: a reconstruction of the text, or the real activation."""
    return _REP_SOURCE[rep]


def inject_rep(
    vec: Tensor,
    rep: Rep,
    *,
    alpha: float,
    scale: float,
    whitener: Whitener | None = None,
    jspace: MetricSpace | None = None,
) -> Tensor:
    """Raw source vector ``[b, d]`` -> embedding-space injection ``[b, d]`` for arm ``rep``.

    - ``unit`` (raw_unit / real_h): ``alpha * vec/||vec||`` — magnitude discarded, every slot at
      norm ``alpha`` (the established injection scale).
    - ``wht_unit``: whiten (``W(vec - mu)``) then unit-normalize and scale by ``alpha`` — the same
      metric the AR is trained in.
    - ``j_unit``: the J ruler (``J̄(vec - mu)``) then unit-normalize and scale by ``alpha`` — the
      arm of record for a J-trained AR. The AR objective matches J-space *direction* only, so the
      untrained norm must not leak to the AO; hence unit-normalize, exactly as ``wht_unit`` does.
    - ``scaled``: ``scale * vec`` with a SINGLE global scalar (``fit_scale``) — the *unnormed* arm,
      the only one that preserves per-example magnitude across the batch.

    ``whitener`` and ``jspace`` are separate arguments even though both satisfy ``MetricSpace``:
    passing the wrong ruler would silently train a different experiment.
    """
    v = vec.float()
    transform = _REP_TRANSFORM[rep]
    if transform == "unit":
        unit: Tensor = alpha * (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        return unit
    if transform == "scaled":
        return scale * v
    if transform == "wht_unit":
        if whitener is None:
            raise ValueError("wht_unit rep requires a whitener")
        w = whitener.whiten(v)
        wht: Tensor = alpha * (w / w.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        return wht
    if transform == "j_unit":
        if jspace is None:
            raise ValueError("j_unit rep requires a jspace")
        z = jspace.whiten(v)
        jun: Tensor = alpha * (z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        return jun
    raise ValueError(f"unknown transform {transform!r}")  # pragma: no cover


# GT-continuation lens arms (design.md §6). Unlike ``Rep`` the source is ALWAYS the true stored
# activation, so only the transform varies — and ``scaled`` scales are fit PER LAYER (residual
# norms grow orders of magnitude with depth; one global scalar would whisper early layers and
# clip late ones).
GtTransform = Literal["raw", "unit", "scaled", "scaled_clip", "wht_unit", "j_unit"]

# Arms whose injected norm is fixed at ``alpha`` by unit-normalisation, so no per-layer
# ``fit_scale`` entry exists for them. ``SoftTokenConfig.scale_for`` must return 1.0 here rather
# than indexing its ``scales`` dict (which would KeyError).
SCALE_FREE_GT: frozenset[str] = frozenset({"unit", "wht_unit", "j_unit"})


def inject_gt(
    vec: Float[Tensor, "b d"],
    transform: GtTransform,
    *,
    alpha: float,
    scale: float,
    clip_mult: float = 4.0,
    space: MetricSpace | None = None,
) -> Float[Tensor, "b d"]:
    """True activation ``[b, d]`` -> embedding-space injection for the GT-continuation arms.

    - ``raw``: the identity — the stored activation goes in unmodified. Equal to
      ``scaled`` at scale 1.0 by construction; it exists so "no scaling" is an explicit
      intent rather than a silently-defaulted scale (see olens_sglang/common.Transform).
    - ``unit``: ``alpha * vec/||vec||`` — the AO ``real_h`` arm's transform.
    - ``scaled``: ``scale * vec`` with a per-layer ``fit_scale`` scalar (median norm == alpha).
    - ``scaled_clip``: ``scaled``, then norms clipped at ``clip_mult * alpha`` — tames the
      massive-activation/attention-sink outlier rows that AR(out) reconstructions never had.
    - ``wht_unit`` / ``j_unit``: map into a metric space (``W(x-μ)`` or ``J̄(x-μ)``), then
      unit-normalise and scale by ``alpha``. One code path for both because ``MetricSpace``
      already unifies ``Whitener`` and ``JSpace``; the CALLER picks which ruler, per layer.
      Unit-normalising is load-bearing for ``j_unit``: the AR objective matches J-space
      DIRECTION only, so the untrained norm must not leak to the AO.
    """
    v = vec.float()
    if transform == "raw":
        return v
    if transform == "unit":
        unit: Tensor = alpha * (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        return unit
    if transform in ("wht_unit", "j_unit"):
        if space is None:
            raise ValueError(f"{transform} transform requires a metric space for this layer")
        z = space.whiten(v)
        zu: Tensor = alpha * (z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        return zu
    scaled: Tensor = scale * v
    if transform == "scaled":
        return scaled
    if transform == "scaled_clip":
        norms = scaled.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cap = clip_mult * alpha
        clipped: Tensor = scaled * (cap / norms).clamp(max=1.0)
        return clipped
    raise ValueError(f"unknown transform {transform!r}")  # pragma: no cover


def fit_scale(vecs: Float[Tensor, "n d"], *, target_norm: float) -> float:
    """Global scalar ``s`` for the ``raw_scaled`` arm: median ``||s*vec|| == target_norm``.

    Matches the unit arms' characteristic slot norm (``target_norm = alpha``) while keeping the
    magnitude spread that unit-normalization throws away.
    """
    med = float(vecs.float().norm(dim=-1).median())
    return target_norm / max(med, 1e-9)
