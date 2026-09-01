"""Multi-layer reconstructor: loss/FVE math, right-pad collate + last-token gather, head stack."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.multilayer_reconstructor import (
    MLReconDataset,
    MultiLayerReconstructor,
    build_ml_heads,
    ml_collate,
    multilayer_fve,
    multilayer_whitened_cosine_loss,
)

D = 8
NL = 3


def ident_whiteners(n: int = NL, d: int = D) -> list[Whitener]:
    return [Whitener(mu=torch.zeros(d), w=torch.eye(d), ridge_c=0.0) for _ in range(n)]


def test_loss_zero_when_preds_equal_targets() -> None:
    gen = torch.Generator().manual_seed(0)
    t = torch.randn(4, NL, D, generator=gen)
    loss = multilayer_whitened_cosine_loss(t.clone(), t, ident_whiteners())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_two_when_antiparallel() -> None:
    t = torch.randn(3, NL, D, generator=torch.Generator().manual_seed(1))
    loss = multilayer_whitened_cosine_loss(-t, t, ident_whiteners())  # cos=-1 -> 2(1-cos)=4
    assert loss.item() == pytest.approx(4.0, abs=1e-5)


def test_fve_one_when_aligned_and_per_layer_shape() -> None:
    t = torch.randn(5, NL, D, generator=torch.Generator().manual_seed(2))
    fve = multilayer_fve(t * 3.0, t, ident_whiteners())  # scaled -> cos=1 -> cos^2=1
    assert fve.shape == (NL,)
    assert torch.allclose(fve, torch.ones(NL), atol=1e-5)


def test_ml_collate_right_pads_and_stacks() -> None:
    rows = [
        {"ids": torch.tensor([1, 2, 3]), "target": torch.zeros(NL, D)},
        {"ids": torch.tensor([4, 5]), "target": torch.ones(NL, D)},
    ]
    b = ml_collate(rows, pad_id=0)
    assert b["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]  # right-padded
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert b["targets"].shape == (2, NL, D)


class _StubBackbone(nn.Module):
    """Returns last_hidden_state = a per-position ramp so gather-of-last-token is checkable."""

    def forward(self, input_ids: Tensor, attention_mask: Tensor, use_cache: bool = False) -> Any:
        b, p = input_ids.shape
        # position j -> vector of all (j+1); last real token index differs per row
        hs = (torch.arange(p).float() + 1).view(1, p, 1).expand(b, p, D).clone()
        return SimpleNamespace(last_hidden_state=hs)


def test_reconstructor_gathers_last_real_token() -> None:
    heads = build_ml_heads(D, NL, layer_norm=False)
    # make heads identity-ish: set to identity weight, zero bias
    for h in heads:
        with torch.no_grad():
            h.linear.weight.copy_(torch.eye(D))
            h.linear.bias.zero_()
    model = MultiLayerReconstructor(_StubBackbone(), heads)
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    attn = torch.tensor([[1, 1, 1], [1, 1, 0]])
    out = model(ids, attn)  # [2, NL, D]
    assert out.shape == (2, NL, D)
    # row 0 last real token at pos 2 -> value 3; row 1 last real at pos 1 -> value 2
    assert torch.allclose(out[0], torch.full((NL, D), 3.0))
    assert torch.allclose(out[1], torch.full((NL, D), 2.0))


def test_dataset_returns_ids_and_target() -> None:
    from test_ola_multilayer import make_ml_pairs

    pairs = make_ml_pairs([[7, 8, 9], [10, 11]])
    ds = MLReconDataset(pairs)
    assert len(ds) == 2
    item = ds[0]
    assert item["ids"].tolist() == [7, 8, 9]
    assert item["target"].shape == (NL, D)
