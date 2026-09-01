"""AO-1: inject_rep per-arm behavior, scale fitting, self-target builder, collate mask, metric."""

from typing import Any

import pytest
import torch
from test_ola_shards import make_pairs
from torch import Tensor

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.ao_train import (
    AOConfig,
    ao_collate,
    build_ao1_examples,
    fit_scale,
    inject_rep,
    reconstruction_cos2,
    rep_source,
)
from oracle_lens.pipeline.jspace import JSpace

D = 8
ALPHA = 8000.0


def identity_whitener(d: int = D) -> Whitener:
    return Whitener(mu=torch.zeros(d), w=torch.eye(d), ridge_c=0.0)


class TinyTokenizer:
    """Minimal tokenizer for the example builder: char-level ids, fixed tag/eos."""

    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


def test_rep_source_mapping() -> None:
    assert rep_source("real_h") == "real_h"
    for rep in ("raw_unit", "wht_unit", "raw_scaled"):
        assert rep_source(rep) == "ar_out"


def test_inject_rep_unit_arms_are_alpha_normed() -> None:
    gen = torch.Generator().manual_seed(0)
    vec = torch.randn(5, D, generator=gen) * 3.7  # arbitrary magnitudes
    for rep in ("raw_unit", "real_h"):
        v = inject_rep(vec, rep, alpha=ALPHA, scale=1.0)
        assert torch.allclose(v.norm(dim=-1), torch.full((5,), ALPHA), atol=1e-2)


def test_inject_rep_wht_unit_identity_matches_unit() -> None:
    gen = torch.Generator().manual_seed(1)
    vec = torch.randn(4, D, generator=gen)
    wht = inject_rep(vec, "wht_unit", alpha=ALPHA, scale=1.0, whitener=identity_whitener())
    unit = inject_rep(vec, "raw_unit", alpha=ALPHA, scale=1.0)
    assert torch.allclose(wht, unit, atol=1e-4)


def test_inject_rep_wht_unit_requires_whitener() -> None:
    with pytest.raises(ValueError, match="whitener"):
        inject_rep(torch.randn(2, D), "wht_unit", alpha=ALPHA, scale=1.0)


def test_inject_rep_j_unit_identity_matches_unit() -> None:
    """JSpace(mu=0, J=I) reduces j_unit to the raw unit arm — the ruler adds nothing."""
    gen = torch.Generator().manual_seed(2)
    vec = torch.randn(4, D, generator=gen)
    jsp = JSpace(mu=torch.zeros(D), j=torch.eye(D))
    jun = inject_rep(vec, "j_unit", alpha=ALPHA, scale=1.0, jspace=jsp)
    unit = inject_rep(vec, "raw_unit", alpha=ALPHA, scale=1.0)
    assert torch.allclose(jun, unit, atol=1e-4)


def test_inject_rep_j_unit_applies_transpose_and_centering() -> None:
    """z = (x - mu) @ J.T — J is not symmetric, so the transpose convention is load-bearing."""
    j = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    jsp = JSpace(mu=torch.tensor([1.0, 0.0]), j=j)
    out = inject_rep(torch.tensor([[2.0, 3.0]]), "j_unit", alpha=1.0, scale=1.0, jspace=jsp)
    # (x - mu) = [1, 3]; @ J.T swaps -> [3, 1]; unit-normalised then scaled by alpha=1
    expect = torch.tensor([[3.0, 1.0]]) / torch.tensor([[3.0, 1.0]]).norm()
    assert torch.allclose(out, expect, atol=1e-6)


def test_inject_rep_j_unit_is_alpha_normed() -> None:
    """The AR matches J-space DIRECTION only — the untrained norm must not reach the AO."""
    gen = torch.Generator().manual_seed(3)
    vec = torch.randn(5, D, generator=gen) * 11.0
    jsp = JSpace(mu=torch.randn(D, generator=gen), j=torch.randn(D, D, generator=gen))
    out = inject_rep(vec, "j_unit", alpha=ALPHA, scale=1.0, jspace=jsp)
    assert torch.allclose(out.norm(dim=-1), torch.full((5,), ALPHA), atol=1e-2)


def test_inject_rep_j_unit_requires_jspace() -> None:
    with pytest.raises(ValueError, match="jspace"):
        inject_rep(torch.randn(2, D), "j_unit", alpha=ALPHA, scale=1.0)


def test_j_unit_rep_source_is_ar_out() -> None:
    assert rep_source("j_unit") == "ar_out"


def test_inject_rep_scaled_preserves_magnitude_spread() -> None:
    vec = torch.tensor([[1.0] + [0.0] * (D - 1), [2.0] + [0.0] * (D - 1)])  # norms 1, 2
    v = inject_rep(vec, "raw_scaled", alpha=ALPHA, scale=3.0)
    # a single global scalar: norms stay in the input's 1:2 ratio (unit would flatten them)
    assert v.norm(dim=-1).tolist() == pytest.approx([3.0, 6.0])


def test_fit_scale_targets_median_norm() -> None:
    gen = torch.Generator().manual_seed(2)
    vecs = torch.randn(101, D, generator=gen) * 5.0
    s = fit_scale(vecs, target_norm=ALPHA)
    scaled_med = float((s * vecs).norm(dim=-1).median())
    assert scaled_med == pytest.approx(ALPHA, rel=1e-5)


def _recover_pair_index(target_ids: list[int], pairs: Any) -> int:
    """Which pair an example came from: the unique pair whose full span is a contiguous slice."""
    spans = {tuple(pairs.row_ids(i).tolist()): i for i in range(len(pairs))}
    for span, i in spans.items():
        n = len(span)
        for j in range(len(target_ids) - n + 1):
            if tuple(target_ids[j : j + n]) == span:
                return i
    raise AssertionError("no span found in target")


def test_build_ao1_examples_real_h_self_target() -> None:
    pairs = make_pairs([[7, 8, 9, 10, 11], [22, 23, 24, 25, 26, 27]])
    tok = TinyTokenizer()
    cfg = AOConfig(run_name="t", rep="real_h", min_len=1, max_len=100, n_examples=10)
    ex = build_ao1_examples(pairs, tok, cfg)
    assert len(ex) == 2
    for e in ex:
        ids = e["target_ids"].tolist()
        assert ids[-1] == tok.eos_token_id
        # the example maps to exactly one pair (its span sits contiguously in the target)...
        i = _recover_pair_index(ids, pairs)
        # ...and real_h injects that pair's stored preceding activation verbatim
        assert torch.allclose(e["inject_vec"], pairs.targets[i].float())


def test_build_ao1_examples_ar_out_needs_tensor() -> None:
    pairs = make_pairs([[7, 8, 9, 10, 11]])
    cfg = AOConfig(run_name="t", rep="raw_unit", min_len=1, max_len=100)
    with pytest.raises(ValueError, match="ar_out"):
        build_ao1_examples(pairs, TinyTokenizer(), cfg)


def test_build_ao1_examples_ar_out_uses_precompute() -> None:
    pairs = make_pairs([[7, 8, 9, 10, 11], [12, 13, 14, 15, 16]])
    ar_out = torch.arange(2 * D, dtype=torch.float32).reshape(2, D)
    cfg = AOConfig(run_name="t", rep="raw_unit", min_len=1, max_len=100)
    ex = build_ao1_examples(pairs, TinyTokenizer(), cfg, ar_out=ar_out)
    injected = torch.stack([e["inject_vec"] for e in ex])
    # every injected vector is a row of the precomputed AR(out), never pairs.targets
    for row in injected:
        assert any(torch.allclose(row, ar_out[i]) for i in range(2))


def test_ao_collate_masks_prompt_and_pad() -> None:
    rows = [
        {"inject_vec": torch.ones(D), "target_ids": torch.tensor([3, 4, 5])},
        {"inject_vec": torch.zeros(D), "target_ids": torch.tensor([6, 7])},
    ]
    batch = ao_collate(rows, prompt_len=4, pad_id=0)
    assert batch["inject_vec"].shape == (2, D)
    labels = batch["labels"]
    assert labels.shape == (2, 4 + 3)
    assert (labels[:, :4] == -100).all()  # prompt masked
    assert labels[0, 4:].tolist() == [3, 4, 5]
    assert labels[1, 4:].tolist() == [6, 7, -100]  # pad masked
    attn = batch["attention_mask"]
    assert attn[1].tolist() == [1, 1, 1, 1, 1, 1, 0]  # pad position off


class _FakeScorer:
    """Stands in for FrozenReconstructor.score with a fixed whitened cosine."""

    def __init__(self, cos: Tensor) -> None:
        self._cos = cos

    def score(
        self, texts_per_act: list[list[str]], targets: Tensor, *, mode: str
    ) -> dict[str, Tensor]:
        return {"whitened_cos": self._cos}


def test_reconstruction_cos2_squares_signed_cosine() -> None:
    cos = torch.tensor([0.5, -0.4, float("nan")])
    got = reconstruction_cos2(_FakeScorer(cos), [["a"], ["b"], ["c"]], torch.zeros(3, D))  # type: ignore[arg-type]
    assert got[0].item() == pytest.approx(0.25)
    assert got[1].item() == pytest.approx(0.16)  # negative cos still explains variance
    assert torch.isnan(got[2])


def test_ao_config_to_dict_roundtrips_rep() -> None:
    cfg = AOConfig(run_name="arm-a", rep="wht_unit")
    d: dict[str, Any] = cfg.to_dict()
    assert d["rep"] == "wht_unit" and d["run_name"] == "arm-a"
