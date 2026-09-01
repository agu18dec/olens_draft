"""Unit tests for the r2s selection core (``ola.r2_select`` + ``ablation.project_out_rows``)."""

import importlib.util
from pathlib import Path

import torch

from oracle_lens.core.nnomp import nnomp_batch
from oracle_lens.pipeline.ablation import WhitenedSpace, project_out, project_out_rows
from oracle_lens.pipeline.r2_select import (
    best_step_pick,
    omp_select,
    omp_select_staged,
    prefix_candidate_phrases,
    unique_index,
    unit_rows,
)


class FakeTok:
    """Whitespace 'tokenizer': ids are the words themselves (slice/decode compatible)."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


def _space(d: int = 6, seed: int = 0) -> WhitenedSpace:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=gen)
    cov = a @ a.T / d + 0.5 * torch.eye(d)
    mu = torch.randn(d, generator=gen)
    return WhitenedSpace.from_moments(mu, cov, ridge_c=0.1)


def test_unique_index() -> None:
    uniq, idx = unique_index(["a", "", "b", "a", "c", "b"])
    assert uniq == ["a", "b", "c"]
    assert idx == [0, -1, 1, 0, 2, 1]


def test_prefix_candidate_phrases_alignment_and_dedup() -> None:
    """r2sp: cand i*len(samples)+j is sample j at lengths[i]; unique_index collapses repeats
    (a short sample's longer truncations are identical strings)."""
    samples = [
        "<explanation>\nalpha beta gamma delta\n</explanation>",
        "<explanation>\nalpha beta\n</explanation>",
    ]
    cands = prefix_candidate_phrases(samples, FakeTok(), lengths=(2, 4))
    assert cands == ["alpha beta", "alpha beta", "alpha beta gamma delta", "alpha beta"]
    uniq, idx = unique_index(cands)
    assert uniq == ["alpha beta", "alpha beta gamma delta"]
    assert idx == [0, 0, 1, 0]


def test_select_script_base_variant_strips_suffixes() -> None:
    """Dotted pilot/step suffixes AND underscore lane suffixes (_le64) resolve to VARIANTS
    keys — main KeyError'd on '_le64' dirs until r2sp (the le64 round ran pod-side copies)."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "distill" / "olens_r2s_select.py"
    spec = importlib.util.spec_from_file_location("r2sselect", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.base_variant("pool64n32") == "pool64n32"
    assert mod.base_variant("pool64n32.dp") == "pool64n32"
    assert mod.base_variant("pool64n32_le64") == "pool64n32"
    assert mod.base_variant("pool64n32_le64.pfx") == "pool64n32"
    assert mod.base_variant("k4n32d64.p.step1") == "k4n32d64"


def test_omp_select_matches_nnomp_on_shared_dict() -> None:
    gen = torch.Generator().manual_seed(1)
    b, c, d, k = 3, 8, 16, 4
    dictionary = unit_rows(torch.randn(c, d, generator=gen))
    x = torch.randn(b, d, generator=gen)
    ref = nnomp_batch(x, dictionary, max_atoms=k)
    sel, coeffs, fve = omp_select(
        x, dictionary.unsqueeze(0).expand(b, c, d), torch.ones(b, c, dtype=torch.bool), k=k
    )
    assert torch.equal(sel, ref.atoms)
    assert torch.allclose(fve, ref.fve, atol=1e-5)
    assert torch.allclose(coeffs, ref.coeffs, atol=1e-4)


def test_omp_select_respects_valid_mask() -> None:
    gen = torch.Generator().manual_seed(2)
    b, c, d = 2, 5, 8
    dirs = unit_rows(torch.randn(b, c, d, generator=gen))
    x = dirs[:, 0, :].clone()  # row's best atom is index 0 …
    valid = torch.ones(b, c, dtype=torch.bool)
    valid[:, 0] = False  # … but it is masked out
    sel, _, _ = omp_select(x, dirs, valid, k=3)
    assert (sel != 0).all()


def test_omp_select_no_positive_scores_selects_nothing() -> None:
    d = 4
    dirs = unit_rows(torch.eye(d)[:2]).unsqueeze(0)  # e0, e1
    x = -torch.ones(1, d)  # negative correlation with both
    sel, coeffs, fve = omp_select(x, dirs, torch.ones(1, 2, dtype=torch.bool), k=2)
    assert sel.tolist() == [[-1, -1]]
    assert float(fve) == 0.0
    assert coeffs.abs().sum() == 0.0


def test_omp_select_staged_picks_one_per_stage() -> None:
    # 4 candidates, 2 length stages; stage 0 may use {0,1}, stage 1 may use {2,3}.
    d = 6
    gen = torch.Generator().manual_seed(11)
    dirs = unit_rows(torch.randn(1, 4, d, generator=gen))
    # make atom 1 the best for x, atom 3 the best for the residual after removing 1
    x = (2.0 * dirs[0, 1] + 1.0 * dirs[0, 3]).unsqueeze(0)
    sv = [
        torch.tensor([[True, True, False, False]]),
        torch.tensor([[False, False, True, True]]),
    ]
    sel, _coef, fve = omp_select_staged(x, dirs, sv)
    assert sel[0, 0].item() in (0, 1) and sel[0, 1].item() in (2, 3)
    assert float(fve) > 0.5


def test_omp_select_staged_skips_empty_stage() -> None:
    d = 5
    dirs = unit_rows(torch.eye(d)[:3].unsqueeze(0))
    x = dirs[:, 0, :].clone()
    sv = [
        torch.tensor([[True, True, True]]),
        torch.tensor([[False, False, False]]),  # no valid atom -> slot stays -1
    ]
    sel, _c, _f = omp_select_staged(x, dirs, sv)
    assert sel[0, 0].item() == 0 and sel[0, 1].item() == -1


def test_best_step_pick_prefers_largest_positive() -> None:
    d = 4
    dirs = torch.stack([torch.eye(d)[0], torch.eye(d)[1], -torch.eye(d)[0]]).unsqueeze(0)
    h = torch.tensor([[0.5, 2.0, 0.0, 0.0]])
    idx, coef, pos = best_step_pick(h, dirs, torch.ones(1, 3, dtype=torch.bool))
    assert idx.tolist() == [1] and pos.tolist() == [True]
    assert torch.allclose(coef, torch.tensor([2.0]))


def test_best_step_pick_fallback_and_empty() -> None:
    d = 3
    dirs = torch.stack([torch.eye(d)[0], torch.eye(d)[1]]).unsqueeze(0)
    h = torch.tensor([[-1.0, -2.0, 0.0]])
    idx, coef, pos = best_step_pick(h, dirs, torch.ones(1, 2, dtype=torch.bool))
    assert idx.tolist() == [0] and pos.tolist() == [False]  # least-negative fallback
    assert torch.allclose(coef, torch.tensor([-1.0]))
    idx2, coef2, _ = best_step_pick(h, dirs, torch.zeros(1, 2, dtype=torch.bool))
    assert idx2.tolist() == [-1] and float(coef2) == 0.0


def test_nnomp_batch_runs_in_bf16() -> None:
    """The m1 scoring path feeds bf16 — nnomp must not mix dtypes mid-loop (2026-08-01 bug)."""
    gen = torch.Generator().manual_seed(7)
    dictionary = unit_rows(torch.randn(32, 16, generator=gen)).to(torch.bfloat16)
    x = torch.randn(4, 16, generator=gen).to(torch.bfloat16)
    res = nnomp_batch(x, dictionary, max_atoms=4)
    ref = nnomp_batch(x.float(), dictionary.float(), max_atoms=4)
    assert (res.atoms >= -1).all() and res.fve.shape == (4,)
    assert (res.atoms == ref.atoms).float().mean() > 0.9  # bf16 ties may flip an atom rarely


def test_project_out_rows_zeroes_own_direction() -> None:
    gen = torch.Generator().manual_seed(3)
    space = _space()
    h = torch.randn(4, 6, generator=gen)
    d_raw = torch.randn(4, 6, generator=gen)
    h_abl, coef = project_out_rows(space, h, d_raw)
    u = unit_rows(space.whiten(d_raw))
    resid = (space.whiten(h_abl) * u).sum(-1)
    assert resid.abs().max() < 1e-3
    assert torch.allclose(coef, (space.whiten(h) * u).sum(-1), atol=1e-4)


def test_project_out_rows_matches_shared_direction_case() -> None:
    gen = torch.Generator().manual_seed(4)
    space = _space(seed=5)
    h = torch.randn(3, 6, generator=gen)
    d_raw = torch.randn(6, generator=gen)
    ref_h, ref_c = project_out(space, h, space.whiten(d_raw.unsqueeze(0))[0])
    rows_h, rows_c = project_out_rows(space, h, d_raw.unsqueeze(0).expand(3, 6))
    assert torch.allclose(rows_h, ref_h, atol=1e-3)
    assert torch.allclose(rows_c, ref_c, atol=1e-4)


# ---- bullet-prefix candidates + shrink_to_shortest (ptag stage-2, 2026-08-23) ----


def _load_omp_script():  # type: ignore[no-untyped-def]
    script = Path(__file__).resolve().parents[1] / "scripts" / "distill" / "ao_gt_omp_readout.py"
    spec = importlib.util.spec_from_file_location("aogtomp", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bullet_prefix_candidates_ladder_full_and_dedup() -> None:
    mod = _load_omp_script()
    long_bullet = "- " + " ".join(f"w{i}" for i in range(70))
    samples = [
        "- a b c\n- p q r s t",  # bullets of 3 and 5 tokens
        "- a b z",               # 2-prefix collides with sample 0 bullet 0's
        "- x",                   # 1 token -> skipped
        long_bullet,             # 70 tokens -> capped at 64
    ]
    cand_ids, cand_len, cand_src = mod.build_candidates(
        samples, FakeTok(), mode="bullet_prefixes", prefixes=(2, 4, 8, 16, 32, 64),
        is_degen=lambda t: t == "a b c",  # kill ONE full bullet, keep its prefixes
    )
    by_key = {tuple(ids): (cand_len[i], cand_src[i]) for i, ids in enumerate(cand_ids)}
    # ladder union full: 3-token bullet -> {2,3}; 5-token -> {2,4,5}; 70-token -> {2..32,64}
    assert by_key[("a", "b")] == (2, (0, 0))  # first occurrence keeps provenance
    assert ("a", "b", "c") not in by_key  # degeneracy-masked full bullet
    assert by_key[("a", "b", "z")] == (3, (1, 0))
    assert by_key[("p", "q", "r", "s", "t")][0] == 5 and by_key[("p", "q")][1] == (0, 1)
    assert sorted({cand_len[i] for i, s in enumerate(cand_src) if s == (3, 0)}) == [
        2, 4, 8, 16, 32, 64]
    assert not any(s[0] == 2 for s in cand_src)  # the 1-token bullet emitted nothing
    # bullets mode: ONE atom per bullet, whole (capped), same dedup
    _b_ids, b_len, _b_src = mod.build_candidates(
        samples, FakeTok(), mode="bullets", prefixes=(2, 4, 8, 16, 32, 64),
        is_degen=lambda t: False,
    )
    assert sorted(b_len) == [3, 3, 5, 64]
    # prefixes mode unchanged: ladder over raw samples, src bullet_idx == -1
    p_ids, _p_len, p_src = mod.build_candidates(
        ["a b c d e"], FakeTok(), mode="prefixes", prefixes=(2, 4),
        is_degen=lambda t: False,
    )
    assert [tuple(i) for i in p_ids] == [("a", "b"), ("a", "b", "c", "d")]
    assert p_src == [(0, -1), (0, -1)]


def test_shrink_to_shortest_picks_shortest_within_eps() -> None:
    from oracle_lens.pipeline.r2_select import shrink_to_shortest

    d = 4
    e = torch.eye(d)
    cos = 0.99  # 1-atom FVE of the short swap = cos^2 ≈ 0.9801 (drop ≈ 0.0199)
    dirs = torch.stack([e[0], unit_rows(cos * e[0] + (1 - cos**2) ** 0.5 * e[1])])
    cand_ids = [[1, 2, 3, 4], [1, 2]]
    cand_len = [4, 2]
    x_w = e[0].unsqueeze(0)
    sel, fve = shrink_to_shortest(
        [0], 1.0, cand_ids, cand_len, dirs, x_w, ladder=(2, 4), eps=0.05)
    assert sel == [1] and abs(fve - cos**2) < 5e-3
    sel0, fve0 = shrink_to_shortest(
        [0], 1.0, cand_ids, cand_len, dirs, x_w, ladder=(2, 4), eps=0.0)
    assert sel0 == [0] and abs(fve0 - 1.0) < 5e-3


def test_shrink_to_shortest_budget_is_global() -> None:
    from oracle_lens.pipeline.r2_select import shrink_to_shortest

    d, eps = 4, 0.1
    e = torch.eye(d)
    s2 = 2 * 0.6 * eps  # each swap alone costs sin^2/2 = 0.6*eps of joint FVE
    c = (1 - s2) ** 0.5
    dirs = torch.stack([
        e[0],                                        # 0: A full   (ids 1234, len 4)
        unit_rows(c * e[0] + s2**0.5 * e[1]),        # 1: A short  (ids 12,   len 2)
        e[2],                                        # 2: B full   (ids 567,  len 3)
        unit_rows(c * e[2] + s2**0.5 * e[3]),        # 3: B short  (ids 56,   len 2)
    ])
    cand_ids = [[1, 2, 3, 4], [1, 2], [5, 6, 7], [5, 6]]
    cand_len = [4, 2, 3, 2]
    x_w = unit_rows((e[0] + e[2]).unsqueeze(0))
    sel, fve = shrink_to_shortest(
        [0, 2], 1.0, cand_ids, cand_len, dirs, x_w, ladder=(2, 4), eps=eps)
    # longest-first: A (len 4) swaps (costs 0.6*eps); B's swap would land at 1.2*eps -> kept
    assert sel == [1, 2]
    assert 1 - 0.6 * eps - 5e-3 <= fve <= 1 - 0.6 * eps + 5e-3


def test_shrink_to_shortest_never_duplicates_or_creates_prefix_pairs() -> None:
    from oracle_lens.pipeline.r2_select import shrink_to_shortest

    d = 4
    e = torch.eye(d)
    # dup guard: X's 2-prefix IS already the other pick -> skipped, no swap possible
    dirs = torch.stack([e[0], e[1]])
    sel, _ = shrink_to_shortest(
        [0, 1], 1.0, [[1, 2, 3, 4], [1, 2]], [4, 2], dirs,
        unit_rows((e[0] + e[1]).unsqueeze(0)), ladder=(2, 4), eps=1.0)
    assert sel == [0, 1]
    # prefix-pair guard: X's 2-prefix [1,2] is a token-prefix of pick Y=[1,2,9] -> skipped
    # even though eps would allow it (X_short dir == X dir)
    dirs2 = torch.stack([e[0], e[0], e[1]])
    cand_ids2 = [[1, 2, 3, 4], [1, 2], [1, 2, 9]]
    sel2, _ = shrink_to_shortest(
        [0, 2], 1.0, cand_ids2, [4, 2, 3], dirs2,
        unit_rows((e[0] + e[1]).unsqueeze(0)), ladder=(2, 4), eps=1.0)
    assert sel2 == [0, 2]
