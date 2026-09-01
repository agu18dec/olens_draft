"""CPU tests for the prompt-tag reconstructor: layer identity lives in the PROMPT.

The whole 2x2 result rests on this arch reading the FINAL layer and being told the target layer
only by prepended text. Each test below pins one of those claims with a mock backbone that encodes
(hidden_state_index, position) in channels 0/1, so what the head received is directly readable.
"""

import pytest
import torch
from torch import nn

from oracle_lens.pipeline.multilayer_reconstructor import (
    PromptTagReconstructor,
    head_state,
)


class _Out:
    def __init__(self, hs: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hs
        self.last_hidden_state = hs[-1]


class _MockBackbone(nn.Module):
    """hs[i][b, pos, :] encodes (i, pos); also records every (ids, mask) it was called with."""

    def __init__(self, n_blocks: int, d: int) -> None:
        super().__init__()
        self.n_blocks, self.d = n_blocks, d
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw: object) -> _Out:
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        b, p = input_ids.shape
        hs = []
        for i in range(self.n_blocks + 1):
            t = torch.zeros(b, p, self.d)
            for pos in range(p):
                t[:, pos, 0] = i
                t[:, pos, 1] = pos
            hs.append(t)
        return _Out(tuple(hs))


LAYERS_12 = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63)


def _model(
    layers: tuple[int, ...] = LAYERS_12, d: int = 16, tag_w: int = 5
) -> PromptTagReconstructor:
    bb = _MockBackbone(n_blocks=max(layers) + 1, d=d)
    # distinct, equal-length tag rows: row li starts with a sentinel 900+li
    tag = torch.stack(
        [
            torch.tensor([900 + li] + [7] * (tag_w - 1), dtype=torch.long)
            for li in range(len(layers))
        ]
    )
    return PromptTagReconstructor(bb, layers, d, tag, layer_norm=False)


def _identity_head(m: PromptTagReconstructor) -> None:
    with torch.no_grad():
        m.head.linear.weight.copy_(torch.eye(m.head.linear.weight.shape[0]))
        m.head.linear.bias.zero_()


def test_train_shape_is_single_layer_and_val_shape_is_all_layers() -> None:
    m = _model()
    ids, mask = torch.randint(0, 9, (3, 7)), torch.ones(3, 7, dtype=torch.long)
    assert m(input_ids=ids, attention_mask=mask, layer_idx=4).shape == (3, 1, 16)
    assert m(input_ids=ids, attention_mask=mask).shape == (3, 12, 16)


def test_no_layer_embedding_and_exactly_one_head() -> None:
    """The defining property of this arch: layer conditioning carries NO weights."""
    m = _model()
    assert not hasattr(m, "layer_emb")
    assert isinstance(m.head.linear, nn.Linear)
    assert len([mod for mod in m.modules() if isinstance(mod, nn.Linear)]) == 1
    assert m.tags.shape == (12, 5)


def test_tag_is_prepended_and_padding_stays_at_the_tail() -> None:
    m = _model(tag_w=5)
    ids = torch.tensor([[11, 12, 13, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    m(input_ids=ids, attention_mask=mask, layer_idx=3)
    got_ids, got_mask = m.backbone.calls[-1]
    assert got_ids[0].tolist() == [903, 7, 7, 7, 7, 11, 12, 13, 0, 0]
    # tag positions are always attended; the span's own right-padding is untouched at the tail
    assert got_mask[0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]


def test_reads_final_layer_at_last_real_token_after_the_tag() -> None:
    """Two claims at once: the read is the FINAL hidden state (not matched depth), and the read
    index accounts for the tag — tag_width + n_real - 1, not tag_width + pad_width - 1."""
    m = _model(tag_w=5)
    _identity_head(m)
    mask = torch.tensor([[1, 1, 1, 0, 0]])  # 3 real span tokens
    ids = torch.zeros(1, 5, dtype=torch.long)
    for li in range(len(m.layers)):
        out = m(input_ids=ids, attention_mask=mask, layer_idx=li)
        # channel 0 == n_blocks for EVERY layer: same final read regardless of target layer
        assert out[0, 0, 0].item() == m.backbone.n_blocks
        # channel 1 == 5 (tag) + 3 (real) - 1
        assert out[0, 0, 1].item() == 7


def test_val_path_equals_concatenated_train_paths() -> None:
    """The 16-forward val path must be exactly the per-layer train path, or val and train
    measure different functions and the curve is meaningless."""
    m = _model()
    _identity_head(m)
    ids, mask = torch.randint(0, 9, (2, 6)), torch.ones(2, 6, dtype=torch.long)
    full = m(input_ids=ids, attention_mask=mask)
    per = torch.cat(
        [m(input_ids=ids, attention_mask=mask, layer_idx=li) for li in range(len(m.layers))], dim=1
    )
    torch.testing.assert_close(full, per)


def test_rejects_tag_layer_count_mismatch() -> None:
    bb = _MockBackbone(n_blocks=64, d=16)
    with pytest.raises(ValueError, match="tag_ids has 3 rows for 12 layers"):
        PromptTagReconstructor(bb, LAYERS_12, 16, torch.zeros(3, 5, dtype=torch.long))


def test_head_state_round_trips_both_architectures() -> None:
    """head_state must never raise on an arch without layer_emb — that crash used to land at the
    FIRST milestone, i.e. after the training was already paid for."""
    from oracle_lens.pipeline.multilayer_reconstructor import LayerConditionedReconstructor

    pt = _model()
    st = head_state(pt)
    assert set(st) == {"head", "tag_ids", "layers"}
    assert "layer_emb" not in st
    assert st["layers"] == list(LAYERS_12)

    lc = LayerConditionedReconstructor(
        _MockBackbone(n_blocks=64, d=16), LAYERS_12, 16, layer_norm=False
    )
    st_lc = head_state(lc)
    assert set(st_lc) == {"head", "layer_emb"}
