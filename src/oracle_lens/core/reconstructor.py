"""The reconstructor: LoRA-tuned subject model + plain linear head, and its loss/metrics.

Architecture (paper: "a copy of the subject model fine-tuned to map a short phrase of text to
the residual-stream activation that immediately precedes it"):

    phrase ids (left-padded, out of context)
      → subject model truncated to blocks 0..layer, final norm replaced by Identity
      → last_hidden_state[:, -1, :]           (block-`layer` output at the phrase's last token)
      → nn.Linear(d, d)                        (optionally LayerNorm first — config knob)
      → predicted raw-space activation

Truncation makes ``last_hidden_state`` exactly the block-`layer` residual with no
``output_hidden_states`` indexing (whose tuple layout is transformers-version-dependent) and no
forward-hook/gradient-checkpointing interaction, while skipping the unused upper blocks.
"""

from typing import Any

from jaxtyping import Float, Int
from torch import Tensor, nn

from oracle_lens.core.whitening import Whitener


def truncate_backbone(model: Any, *, layer: int) -> Any:
    """Truncate a causal LM to blocks 0..layer and neutralize the final norm, in place.

    Finds the text decoder as the module owning both a ``layers`` ModuleList and a ``norm``
    (the Llama/Qwen layout, same convention ``jlens.hf`` detects) and returns it; forwards on
    the returned module yield ``last_hidden_state`` == block-``layer`` output exactly.
    """
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and hasattr(module, "norm"):
            if layer >= len(layers):
                raise ValueError(f"layer {layer} out of range ({len(layers)} blocks)")
            del layers[layer + 1 :]
            module.norm = nn.Identity()
            return module
    raise ValueError("no decoder module with `layers` + `norm` found")


class ReconstructorHead(nn.Module):
    """Plain Linear(d, d); optional LayerNorm in front (knob, default off). fp32."""

    def __init__(self, d_model: int, *, layer_norm: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False) if layer_norm else None
        self.linear = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        h = x.float()
        if self.norm is not None:
            h = self.norm(h)
        out: Tensor = self.linear(h)
        return out


class Reconstructor(nn.Module):
    """Truncated (optionally LoRA-wrapped) backbone + head. Predicts raw-space activations."""

    def __init__(self, backbone: Any, head: ReconstructorHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self, input_ids: Int[Tensor, "b p"], attention_mask: Int[Tensor, "b p"]
    ) -> Float[Tensor, "b d"]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        final = out.last_hidden_state[:, -1, :]  # left-pad puts the last real token at -1
        pred: Tensor = self.head(final)
        return pred


def whitened_cosine_loss(
    pred: Float[Tensor, "b d"],
    target: Float[Tensor, "b d"],
    whitener: Whitener,
) -> Float[Tensor, ""]:
    """Mean over batch of ‖û - u‖² with û, u the unit-normalized whitened vectors (= 2(1-cos))."""
    pred_w = whitener.whiten(pred)
    target_w = whitener.whiten(target.float())
    u_hat = pred_w / pred_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    u = target_w / target_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    loss: Tensor = ((u_hat - u) ** 2).sum(dim=-1).mean()
    return loss


def r2(pred: Float[Tensor, "n d"], target: Float[Tensor, "n d"]) -> float:
    """1 - ‖pred - target‖²_F / ‖target - target.mean(0)‖²_F (usable in whitened OR raw space)."""
    resid = ((pred - target) ** 2).sum()
    total = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1.0 - resid / total.clamp_min(1e-12))


def cosine_mean(pred: Float[Tensor, "n d"], target: Float[Tensor, "n d"]) -> float:
    cos = nn.functional.cosine_similarity(pred, target, dim=-1)
    return float(cos.mean())


# Length buckets for per-length whitened-R² metrics. Extended past 32 for the long-phrase
# reconstructor (≤1024): short buckets unchanged so old runs stay comparable; long buckets added.
N_BUCKETS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
)


def bucket_metrics(
    pred_w: Float[Tensor, "n d"],
    target_w: Float[Tensor, "n d"],
    phrase_len: Int[Tensor, "n"],
    *,
    buckets: tuple[tuple[int, int], ...] = N_BUCKETS,
) -> dict[str, float]:
    """Whitened R² per phrase-length bucket, keyed ``whitened_r2/N{lo}-{hi}``."""
    out: dict[str, float] = {}
    for lo, hi in buckets:
        sel = (phrase_len >= lo) & (phrase_len <= hi)
        if int(sel.sum()) >= 2:
            out[f"whitened_r2/N{lo}-{hi}"] = r2(pred_w[sel], target_w[sel])
    return out
