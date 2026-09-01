"""Reconstructor: loss known values, R² known values, truncation semantics on a tiny Llama."""

from typing import Any

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from oracle_lens.core.reconstructor import (
    Reconstructor,
    ReconstructorHead,
    r2,
    truncate_backbone,
    whitened_cosine_loss,
)
from oracle_lens.core.whitening import Whitener


def _identity_whitener(d: int) -> Whitener:
    return Whitener(mu=torch.zeros(d), w=torch.eye(d), ridge_c=0.0)


def test_loss_known_values() -> None:
    """Hand-computed: identical → 0; orthogonal → 2; opposite → 4 (loss = 2(1-cos))."""
    wh = _identity_whitener(2)
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    assert float(whitened_cosine_loss(a, a, wh)) < 1e-12
    torch.testing.assert_close(whitened_cosine_loss(a, b, wh), torch.tensor(2.0))
    torch.testing.assert_close(whitened_cosine_loss(a, -a, wh), torch.tensor(4.0))


def test_loss_is_scale_invariant() -> None:
    wh = _identity_whitener(3)
    pred = torch.randn(8, 3)
    target = torch.randn(8, 3)
    torch.testing.assert_close(
        whitened_cosine_loss(pred, target, wh),
        whitened_cosine_loss(7.0 * pred, target, wh),
    )


def test_r2_known_values() -> None:
    """target = [[0],[2]] (mean 1, total SS 2): mean-predictor → 0; pred [[0],[1]] → 0.5."""
    target = torch.tensor([[0.0], [2.0]])
    assert abs(r2(torch.tensor([[1.0], [1.0]]), target)) < 1e-6
    assert abs(r2(torch.tensor([[0.0], [1.0]]), target) - 0.5) < 1e-6
    assert abs(r2(target, target) - 1.0) < 1e-6


def _tiny_llama(n_layers: int = 4, d: int = 32) -> Any:
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=d,
        intermediate_size=2 * d,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)  # type: ignore[no-untyped-call]


def test_truncate_backbone_matches_hidden_states_index() -> None:
    """last_hidden_state of the truncated model == hidden_states[layer+1] of the full model.

    hidden_states[0] is the embeddings; hidden_states[i+1] is block i's output (pre-final-norm
    for i < n_layers-1, which is why we pin a middle layer, not the last).
    """
    model = _tiny_llama()
    model.eval()
    ids = torch.randint(0, 128, (2, 7))
    attn = torch.ones_like(ids)
    with torch.no_grad():
        ref = model.model(
            input_ids=ids, attention_mask=attn, output_hidden_states=True
        ).hidden_states
    layer = 2
    inner = truncate_backbone(model, layer=layer)
    assert len(inner.layers) == layer + 1
    with torch.no_grad():
        out = inner(input_ids=ids, attention_mask=attn).last_hidden_state
    torch.testing.assert_close(out, ref[layer + 1])


def test_truncate_backbone_rejects_out_of_range() -> None:
    model = _tiny_llama(n_layers=3)
    try:
        truncate_backbone(model, layer=3)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_reconstructor_forward_and_grad() -> None:
    model = _tiny_llama()
    inner = truncate_backbone(model, layer=1)
    recon = Reconstructor(inner, ReconstructorHead(32))
    recon.eval()  # no dropout; grads still flow
    ids = torch.randint(0, 128, (3, 5))
    attn = torch.ones_like(ids)
    pred = recon(ids, attn)
    assert pred.shape == (3, 32)
    loss = whitened_cosine_loss(pred, torch.randn(3, 32), _identity_whitener(32))
    loss.backward()  # type: ignore[no-untyped-call]
    grad = recon.head.linear.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_head_layer_norm_knob() -> None:
    head_plain = ReconstructorHead(16, layer_norm=False)
    head_ln = ReconstructorHead(16, layer_norm=True)
    x = torch.randn(4, 16)
    assert head_plain(x).shape == head_ln(x).shape == (4, 16)
    assert head_plain.norm is None and head_ln.norm is not None
