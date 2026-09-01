"""CPU tests for the AO GRPO reward core (``oracle_lens.pipeline.rl_reward``).

Dumb by design: every number is checkable by hand or against a five-line numpy reference.
No GPU, no model, no tokenizer downloads (stub tokenizer where one is needed).
"""

import math
from typing import ClassVar

import numpy as np
import torch

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.rl_reward import (
    AR_SPAN_WIDTH,
    FAILED_EXTRACTION_REWARD,
    RewardSpace,
    ar_positions,
    extract_explanation,
    reward_text_ids,
    score_rows,
)


def _identity_whitener(d: int) -> Whitener:
    return Whitener(mu=torch.zeros(d), w=torch.eye(d), ridge_c=0.1)


def _random_whitener(d: int, seed: int) -> Whitener:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    return Whitener(mu=torch.randn(d, generator=g), w=a, ridge_c=0.1)


class TestScoreRows:
    def test_perfect_prediction_is_zero_reward_and_fve_one(self) -> None:
        d = 8
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, d, generator=g)
        res = score_rows(
            x, x.clone(), torch.zeros(4, dtype=torch.long) + 44, {44: _random_whitener(d, 1)}
        )
        assert torch.allclose(res.reward, torch.zeros(4), atol=1e-5)
        assert torch.allclose(res.fve, torch.ones(4), atol=1e-5)
        assert bool(res.valid.all())

    def test_antipodal_prediction_is_floor(self) -> None:
        d = 8
        x = torch.randn(3, d)
        res = score_rows(-x, x, torch.full((3,), 44), {44: _identity_whitener(d)})
        assert torch.allclose(res.reward, torch.full((3,), FAILED_EXTRACTION_REWARD), atol=1e-5)

    def test_matches_numpy_reference(self) -> None:
        """r = -2(1-cos) of whitened vectors — checked against explicit numpy."""
        d, n = 16, 5
        rng = np.random.default_rng(7)
        preds = rng.standard_normal((n, d)).astype(np.float32)
        golds = rng.standard_normal((n, d)).astype(np.float32)
        w = _random_whitener(d, 3)
        res = score_rows(
            torch.from_numpy(preds),
            torch.from_numpy(golds),
            torch.full((n,), 44),
            {44: w},
        )
        wm, mu = w.w.numpy(), w.mu.numpy()
        for i in range(n):
            p = (preds[i] - mu) @ wm.T
            g = (golds[i] - mu) @ wm.T
            cos = float(p @ g / (np.linalg.norm(p) * np.linalg.norm(g)))
            assert math.isclose(float(res.reward[i]), -2.0 * (1.0 - cos), rel_tol=0, abs_tol=1e-4)
            assert math.isclose(float(res.fve[i]), cos**2, abs_tol=1e-4)

    def test_unit_norm_off_is_negative_whitened_mse(self) -> None:
        d = 8
        g = torch.Generator().manual_seed(2)
        preds, golds = torch.randn(3, d, generator=g), torch.randn(3, d, generator=g)
        w = _random_whitener(d, 5)
        res = score_rows(preds, golds, torch.full((3,), 20), {20: w}, RewardSpace(unit_norm=False))
        expect = -((w.whiten(preds) - w.whiten(golds)) ** 2).mean(dim=-1)
        assert torch.allclose(res.reward, expect, atol=1e-5)

    def test_whiten_off_scores_centered_raw(self) -> None:
        """whiten=False must ignore the whitening matrix entirely (centered-raw cosine)."""
        d = 8
        g = torch.Generator().manual_seed(4)
        preds, golds = torch.randn(3, d, generator=g), torch.randn(3, d, generator=g)
        wa, wb = _random_whitener(d, 6), _random_whitener(d, 7)
        wb = Whitener(mu=wa.mu, w=wb.w, ridge_c=0.1)  # same mu, different matrix
        ra = score_rows(preds, golds, torch.full((3,), 44), {44: wa}, RewardSpace(whiten=False))
        rb = score_rows(preds, golds, torch.full((3,), 44), {44: wb}, RewardSpace(whiten=False))
        assert torch.allclose(ra.reward, rb.reward)  # matrix can't matter
        wc = Whitener(mu=torch.zeros(d), w=wa.w, ridge_c=0.1)  # different mu -> different score
        rc = score_rows(preds, golds, torch.full((3,), 44), {44: wc}, RewardSpace(whiten=False))
        assert not torch.allclose(ra.reward, rc.reward)

    def test_config_toggles_change_the_number(self) -> None:
        d = 8
        g = torch.Generator().manual_seed(9)
        preds, golds = torch.randn(2, d, generator=g), torch.randn(2, d, generator=g)
        w = {44: _random_whitener(d, 8)}
        layers = torch.full((2,), 44)
        seen = {
            (sp.whiten, sp.unit_norm): score_rows(preds, golds, layers, w, sp).reward
            for sp in (RewardSpace(), RewardSpace(whiten=False), RewardSpace(unit_norm=False))
        }
        vals = list(seen.values())
        assert not torch.allclose(vals[0], vals[1])
        assert not torch.allclose(vals[0], vals[2])

    def test_per_layer_whitener_selection(self) -> None:
        """Rows are scored in their OWN layer's space — swapping layers changes the score."""
        d = 8
        g = torch.Generator().manual_seed(11)
        preds, golds = torch.randn(2, d, generator=g), torch.randn(2, d, generator=g)
        whiteners = {20: _random_whitener(d, 20), 44: _random_whitener(d, 44)}
        a = score_rows(preds, golds, torch.tensor([20, 44]), whiteners)
        b = score_rows(preds, golds, torch.tensor([44, 20]), whiteners)
        assert not torch.allclose(a.reward, b.reward)
        # and each row of `a` matches a single-layer call
        solo = score_rows(preds[:1], golds[:1], torch.tensor([20]), whiteners)
        assert torch.allclose(a.reward[0], solo.reward[0])

    def test_nan_gold_gets_floor_not_nan(self) -> None:
        d = 8
        preds = torch.randn(2, d)
        golds = torch.randn(2, d)
        golds[1, 0] = float("nan")
        res = score_rows(preds, golds, torch.full((2,), 44), {44: _identity_whitener(d)})
        assert bool(res.valid[0]) and not bool(res.valid[1])
        assert float(res.reward[1]) == FAILED_EXTRACTION_REWARD
        assert torch.isfinite(res.reward).all()


class TestExtraction:
    def test_explanation_body_extracted(self) -> None:
        assert extract_explanation("<explanation>a cat sat</explanation>") == "a cat sat"

    def test_unclosed_tag_extracts_to_end(self) -> None:
        assert extract_explanation("<explanation>a cat sat") == "a cat sat"

    def test_no_tags_falls_back_to_whole_text(self) -> None:
        assert extract_explanation("just text") == "just text"

    def test_scaffolding_stripped(self) -> None:
        out = extract_explanation("<explanation><think>hm</think>the answer</explanation>")
        assert "think" not in out and "answer" in out

    def test_empty_is_empty(self) -> None:
        assert extract_explanation("") == ""
        assert extract_explanation("<explanation></explanation>") == ""


class _StubTokenizer:
    """Whitespace tokenizer: token = position of the word in the call (never special)."""

    all_special_ids: ClassVar[list[int]] = [0]

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [100 + i for i, _ in enumerate(text.split())]}


class TestRewardTextIds:
    def test_truncates_to_ar_span_width(self) -> None:
        ids = reward_text_ids(_StubTokenizer(), "w " * 200)  # type: ignore[arg-type]
        assert len(ids) == AR_SPAN_WIDTH

    def test_empty_text_gives_no_ids(self) -> None:
        assert reward_text_ids(_StubTokenizer(), "") == []  # type: ignore[arg-type]


class TestArPositions:
    def test_full_17_rows(self) -> None:
        pos = ar_positions(17)
        assert pos[0] == 0 and pos[4] == 1 and pos[63] == 16

    def test_16_rows_drops_layer_zero(self) -> None:
        """The iolens FINAL AR has 16 layer-emb rows: layer 0 dropped, 4 is row 0."""
        pos = ar_positions(16)
        assert 0 not in pos
        assert pos[4] == 0 and pos[8] == 1 and pos[63] == 15


class TestCaptureBlockOutputs:
    def test_matches_output_hidden_states(self) -> None:
        """Hook capture must equal hs[layer+1] exactly (the tf-5.3 qwen3_5 modeling has no
        output_hidden_states — the miles env relies on the hooked path)."""
        from transformers import LlamaConfig, LlamaModel

        from oracle_lens.pipeline.rl_reward import capture_block_outputs

        cfg = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=128,
        )
        m = LlamaModel(cfg)
        # mirror truncate_backbone: the recon backbone has norm=Identity, so
        # hs[last_layer+1] == the raw block output the hook captures
        m.norm = torch.nn.Identity()  # type: ignore[assignment]
        m.eval()  # type: ignore[no-untyped-call]
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, 128, (2, 9), generator=g)
        mask = torch.ones_like(ids)
        mask[1, 6:] = 0

        cap = capture_block_outputs(m, [0, 2, 3], ids, mask)
        with torch.no_grad():
            hs = m(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        for ly in (0, 2, 3):
            assert torch.equal(cap[ly], hs[ly + 1]), f"layer {ly} mismatch"
