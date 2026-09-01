"""Reward-path tokenization pin (bare, no template) + aggregation-mode math."""

from typing import Any, cast

import pytest
import torch
from test_ola_train import D, StubRecon, identity_whitener

from oracle_lens.pipeline.scorer import (
    FrozenReconstructor,
    bare_token_ids,
    prepare_reward_text,
)


class TinyTokenizer:
    """Char-level fake tokenizer: id = codepoint; specials are 0/1."""

    all_special_ids = (0, 1)

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False, "reward path must tokenize bare"
        return {"input_ids": [ord(c) % 60 + 2 for c in text]}


def test_prepare_reward_text_strips_template_markers() -> None:
    raw = "<|im_start|>assistant\n<explanation>\nthe cat sat\n</explanation><|im_end|>"
    assert prepare_reward_text(raw) == "the cat sat"
    assert prepare_reward_text("<think>x</think> plain") == "x plain"
    assert prepare_reward_text("no markers") == "no markers"


def test_bare_token_ids_no_specials_and_truncated() -> None:
    tok = TinyTokenizer()
    ids = bare_token_ids(tok, "abc", max_len=2)
    assert len(ids) == 2
    assert all(i not in (0, 1) for i in ids)
    assert bare_token_ids(tok, "<explanation></explanation>") == []


def make_scorer() -> FrozenReconstructor:
    stub = StubRecon(vocab=64)

    class Backbone:
        def __call__(self, input_ids: Any, attention_mask: Any, use_cache: bool) -> Any:
            h = stub.embed(input_ids) * attention_mask.unsqueeze(-1)
            pooled = h.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)

            class Out:
                last_hidden_state = pooled.unsqueeze(1)

            return Out()

        def parameters(self) -> list[Any]:
            return []

    class Head:
        linear = stub.proj

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = stub.proj(x)
            return out

    return FrozenReconstructor(
        backbone=Backbone(),
        head=cast(Any, Head()),
        tokenizer=TinyTokenizer(),
        whitener=identity_whitener(),
        device="cpu",
    )


def test_score_single_vs_sum_and_invalid_rows() -> None:
    scorer = make_scorer()
    targets = torch.randn(3, D)
    texts = [["hello world"], ["abc", "def"], ["<explanation></explanation>"]]
    single = scorer.score(texts, targets, mode="single")
    both = scorer.score(texts, targets, mode="sum")
    assert single["valid"].tolist() == [1.0, 1.0, 0.0]
    assert torch.isnan(single["whitened_mse"][2])
    assert not torch.isnan(single["whitened_mse"][0])
    # sum over a 2-text row differs from scoring only the first text
    assert both["whitened_mse"][1] != pytest.approx(float(single["whitened_mse"][1]))
    # single-text rows agree between modes
    assert both["whitened_mse"][0] == pytest.approx(float(single["whitened_mse"][0]))


def test_score_nnls_beats_or_matches_single() -> None:
    scorer = make_scorer()
    targets = torch.randn(4, D)
    texts = [["alpha beta", "gamma delta", "epsilon"] for _ in range(4)]
    single = scorer.score(texts, targets, mode="single")
    nnls = scorer.score(texts, targets, mode="nnls")
    # nnls refits coefficients over the same vectors -> residual can't be worse in raw space;
    # whitened-with-identity == raw here
    assert (nnls["whitened_mse"] <= single["whitened_mse"] + 1e-5).all()


def test_reconstruct_orders_rows_correctly() -> None:
    scorer = make_scorer()
    out, valid = scorer.reconstruct(["aa", "bbbb", "c"])
    assert out.shape == (3, D)
    assert valid.all()
    # row order must match input order despite length-sorted batching
    solo0, _ = scorer.reconstruct(["aa"])
    solo2, _ = scorer.reconstruct(["c"])
    assert torch.allclose(out[0], solo0[0], atol=1e-5)
    assert torch.allclose(out[2], solo2[0], atol=1e-5)
