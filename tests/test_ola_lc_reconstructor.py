"""CPU tests for the layer-conditioned reconstructor (architecture b) — shape + shared-head +
right-pad read, using a mock backbone (no real model)."""

import torch
from torch import nn

from oracle_lens.pipeline.multilayer_reconstructor import LayerConditionedReconstructor


class _Out:
    def __init__(self, hs: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hs
        self.last_hidden_state = hs[-1]


class _MockBackbone(nn.Module):
    """Returns hidden_states where hs[i][b, pos, :] encodes (i, pos) so reads are checkable."""

    def __init__(self, n_blocks: int, d: int) -> None:
        super().__init__()
        self.n_blocks, self.d = n_blocks, d

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw: object) -> _Out:
        b, p = input_ids.shape
        hs = []
        for i in range(self.n_blocks + 1):  # hs[0]=emb, hs[i]=block i-1 out
            t = torch.zeros(b, p, self.d)
            for pos in range(p):
                t[:, pos, 0] = i          # channel 0 encodes the hidden-state index
                t[:, pos, 1] = pos        # channel 1 encodes the position
            hs.append(t)
        return _Out(tuple(hs))


def _model(layers: tuple[int, ...], d: int = 16) -> LayerConditionedReconstructor:
    bb = _MockBackbone(n_blocks=max(layers) + 2, d=d)
    return LayerConditionedReconstructor(bb, layers, d, layer_norm=False)


def test_output_shape() -> None:
    m = _model((0, 4, 8))
    mask = torch.ones(3, 5, dtype=torch.long)
    out = m(input_ids=torch.randint(0, 9, (3, 5)), attention_mask=mask)
    assert out.shape == (3, 3, 16)


def test_one_shared_head_and_emb() -> None:
    m = _model((0, 4, 8, 12))
    # exactly ONE head (Linear d->d), and a layer embedding with one row per target layer
    assert isinstance(m.head.linear, nn.Linear)
    assert m.layer_emb.weight.shape == (4, 16)


def test_reads_layer_and_last_real_token() -> None:
    """head input for layer li = hs[layer+1] at the last REAL token (mask.sum-1) + emb[li]."""
    layers = (0, 4)
    d = 16
    m = _model(layers, d)
    # make the head identity so we can read its input back out
    with torch.no_grad():
        m.head.linear.weight.copy_(torch.eye(d))
        m.head.linear.bias.zero_()
        m.layer_emb.weight.zero_()
    mask = torch.tensor([[1, 1, 1, 0, 0]])  # last real token at index 2
    out = m(input_ids=torch.zeros(1, 5, dtype=torch.long), attention_mask=mask)
    # layer 0 -> hs[1] (channel0==1, channel1==2=last real pos); layer 4 -> hs[5] (channel0==5)
    assert out[0, 0, 0].item() == 1 and out[0, 0, 1].item() == 2
    assert out[0, 1, 0].item() == 5 and out[0, 1, 1].item() == 2
