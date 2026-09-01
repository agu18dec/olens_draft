"""CPU tests for the round-2 deflation chain (ola/deflate.py + ablation.project_out_rows)."""

from typing import Any

import torch

from oracle_lens.pipeline.ablation import (
    WhitenedSpace,
    project_out,
    project_out_rows,
    whitened_direction,
)
from oracle_lens.pipeline.deflate import run_deflation_chains
from oracle_lens.pipeline.distill import extract_all_explanations, truncate_phrase


class FakeTok:
    """Whitespace 'tokenizer': ids are the words themselves (slice/decode compatible)."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


def _identity_space(d: int) -> WhitenedSpace:
    """ridge→0 identity moments: whiten ≈ raw, so chain geometry is readable by eye."""
    return WhitenedSpace.from_moments(torch.zeros(d), torch.eye(d), ridge_c=0.0)


def _wrap(s: str) -> str:
    return f"<explanation>\n{s}\n</explanation>"


class ScriptedSampler:
    """Pops the next scripted text per chain index; appending order = draw order per row."""

    def __init__(self, scripts: dict[int, list[str]]) -> None:
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls = 0

    def __call__(self, vecs: torch.Tensor, idxs: list[int]) -> list[str]:
        assert vecs.shape[0] == len(idxs)
        self.calls += 1
        return [self.scripts[i].pop(0) for i in idxs]


def _basis_recon(mapping: dict[str, int], d: int) -> Any:
    def recon(phrases: list[str]) -> torch.Tensor:
        out = torch.zeros(len(phrases), d)
        for j, ph in enumerate(phrases):
            out[j, mapping[ph]] = 1.0
        return out

    return recon


# ---------- project_out_rows ----------


def test_project_out_rows_matches_scalar_project_out() -> None:
    gen = torch.Generator().manual_seed(0)
    d = 6
    a = torch.randn(d, d, generator=gen)
    space = WhitenedSpace.from_moments(
        torch.randn(d, generator=gen), a @ a.T + torch.eye(d), ridge_c=0.1
    )
    h = torch.randn(3, d, generator=gen)
    d_raw = torch.randn(3, d, generator=gen)
    got, coefs = project_out_rows(space, h, d_raw)
    for i in range(3):
        want, coef = project_out(space, h[i : i + 1], whitened_direction(space, d_raw[i]))
        assert torch.allclose(got[i], want[0], atol=1e-3)
        assert torch.allclose(coefs[i], coef[0], atol=1e-4)


def test_project_out_rows_removes_direction_exactly() -> None:
    d = 5
    space = _identity_space(d)
    h = torch.tensor([[2.0, 1.0, 0.0, 0.0, 3.0]])
    d_raw = torch.zeros(1, d)
    d_raw[0, 0] = 4.0  # direction e0 (norm folds out)
    got, _coef = project_out_rows(space, h, d_raw)
    assert abs(float(space.whiten(got)[0, 0])) < 1e-4  # e0 component gone
    assert torch.allclose(got[0, 1:], h[0, 1:], atol=1e-3)  # complement untouched
    # re-projecting the SAME direction is a no-op with ~zero coefficient
    again, coef2 = project_out_rows(space, got, d_raw)
    assert abs(float(coef2[0])) < 1e-4
    assert torch.allclose(again, got, atol=1e-3)


# ---------- the chain ----------


def test_chain_deflates_and_resamples_dups() -> None:
    d = 4
    space = _identity_space(d)
    h0 = torch.tensor([[2.0, 1.0, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]])
    sampler = ScriptedSampler(
        {
            0: [_wrap("alpha"), _wrap("alpha"), _wrap("beta"), _wrap("gamma")],  # step-2 dup
            1: [_wrap("delta"), _wrap("beta"), _wrap("gamma")],
        }
    )
    recon = _basis_recon({"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}, d)
    res = run_deflation_chains(h0, 3, space, sampler, recon, FakeTok(), n=8, resample_cap=2)
    assert [len(s) for s in res.samples] == [3, 3]
    assert res.resamples[0] == [0, 1, 0]  # one redraw at step 2 (the alpha dup)
    assert res.resamples[1] == [0, 0, 0]
    # row 0: coef sequence reads off h0's coords (basis directions are orthogonal)
    assert abs(res.coefs[0][0] - 2.0) < 1e-2  # alpha ≡ e0
    assert abs(res.coefs[0][1] - 1.0) < 1e-2  # beta ≡ e1, untouched by the e0 deflation
    assert abs(res.coefs[0][2] - 0.5) < 1e-2  # gamma ≡ e2
    assert abs(res.wnorm0[0] - h0[0].norm()) < 1e-2
    # explained fraction Σc²/w0² is exact
    frac = sum(c**2 for c in res.coefs[0]) / res.wnorm0[0] ** 2
    assert abs(frac - 1.0) < 1e-3  # h0 row 0 lies fully in span{e0,e1,e2}


def test_chain_accepts_dup_at_cap_and_coef_vanishes() -> None:
    d = 3
    space = _identity_space(d)
    h0 = torch.tensor([[1.0, 2.0, 0.0]])
    sampler = ScriptedSampler({0: [_wrap("alpha")] * 10})  # mode-collapsed teacher
    recon = _basis_recon({"alpha": 0}, d)
    res = run_deflation_chains(h0, 2, space, sampler, recon, FakeTok(), n=8, resample_cap=2)
    assert res.samples[0] == [_wrap("alpha")] * 2  # dup accepted at cap
    assert res.resamples[0] == [0, 2]  # step 2 burned the full cap
    assert abs(res.coefs[0][0] - 1.0) < 1e-2
    assert abs(res.coefs[0][1]) < 1e-4  # re-projecting the removed direction ≈ no-op


def test_chain_empty_phrase_no_deflation() -> None:
    d = 3
    space = _identity_space(d)
    sampler = ScriptedSampler({0: [_wrap("")] * 10})
    recon = _basis_recon({}, d)  # must never be called with an empty phrase
    res = run_deflation_chains(
        torch.ones(1, d), 1, space, sampler, recon, FakeTok(), n=4, resample_cap=2
    )
    assert res.resamples[0] == [2]  # empties are redrawn to the cap, then accepted
    assert res.coefs[0] == [0.0]  # ...with no deflation


def test_chain_truncated_dup_detected() -> None:
    """Two draws differing only past the n-token cap are the SAME phrase for the dup check."""
    d = 3
    space = _identity_space(d)
    sampler = ScriptedSampler(
        {
            0: [_wrap("a b c d"), _wrap("a b c OTHER"), _wrap("x y")],
        }
    )
    recon = _basis_recon({"a b c": 0, "x y": 1}, d)
    res = run_deflation_chains(
        torch.ones(1, d), 2, space, sampler, recon, FakeTok(), n=3, resample_cap=4
    )
    assert res.resamples[0] == [0, 1]
    assert res.samples[0][1] == _wrap("x y")


# ---------- parsing helpers ----------


def test_truncate_phrase() -> None:
    tok = FakeTok()
    assert truncate_phrase("a b c d", tok, n=2) == "a b"
    assert truncate_phrase("a b", tok, n=4) == "a b"
    assert truncate_phrase("", tok, n=4) == ""


def test_extract_all_explanations() -> None:
    text = (
        "<explanation>\nfirst one\n</explanation>\n"
        "<explanation>\nsecond\n</explanation>\n"
        "<explanation>\nthird cut off\n</expl"
    )
    assert extract_all_explanations(text) == ["first one", "second", "third cut off"]
    assert extract_all_explanations("bare, no tags") == ["bare, no tags"]
    assert extract_all_explanations("") == []
    assert extract_all_explanations("<explanation>\n\n</explanation>") == []


def test_chain_min_tokens_redraws_short_and_accepts_longest_at_cap() -> None:
    """k4n256d's >150 floor (scaled down here): short-but-distinct draws are redrawn like
    dups; a long draw is accepted immediately; at cap the LONGEST distinct candidate wins."""
    d = 4
    space = _identity_space(d)
    h0 = torch.tensor([[2.0, 1.0, 0.5, 0.0]])
    long_a = _wrap("alpha one two three")  # 4 FakeTok tokens > floor 3
    sampler = ScriptedSampler(
        {
            0: [
                _wrap("beta"),  # step 1 attempt 0: distinct but short -> redraw
                long_a,  # step 1 attempt 1: long -> accepted
                _wrap("gamma"),  # step 2: all short/distinct -> cap, longest wins
                _wrap("delta one two"),  # 3 tokens, still not > 3
                _wrap("beta"),
            ],
        }
    )
    recon = _basis_recon({"alpha one two three": 0, "beta": 1, "gamma": 2, "delta one two": 3}, d)
    res = run_deflation_chains(
        h0, 2, space, sampler, recon, FakeTok(), n=8, resample_cap=2, min_tokens=3
    )
    assert res.samples[0][0] == long_a
    assert res.resamples[0][0] == 1
    # step 2 burned the cap on shorts; longest distinct candidate ("delta one two") accepted
    assert res.samples[0][1] == _wrap("delta one two")
    assert res.resamples[0][1] == 2
