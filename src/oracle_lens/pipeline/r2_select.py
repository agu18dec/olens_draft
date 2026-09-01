"""Round-2 SELECTION over a shared teacher-sample pool (the ``r2s`` methods).

The original round 2 (``olens_r2_design.md``) trained students on k iid teacher samples
verbatim; the r2s arms instead SELECT the k=4 phrases (n=32 tokens each) from a 16-sample
pool per activation, four ways:

- ``m1`` dict NNOMP:   greedy NNOMP of whiten(h) against the TRAIN-POOL-WIDE dictionary of AR
                       images of every pool phrase — cross-prompt selection ("on-policy" in the
                       sense that every atom is teacher-generated text; user decision
                       2026-08-01: ONE dictionary built from the train pool only, used for both
                       splits, so no eval-derived text ever enters train targets).
- ``m2`` within NNOMP: same greedy objective, dictionary restricted to the row's OWN pool.
- ``m3`` LM judge:     gpt-5.4-nano picks 4 for diversity + descriptiveness
                       (``scripts/ola/olens_r2s_judge.py``; no math here).
- ``m4`` deflation:    4-step chain, best-of-16 fresh candidates per step by whitened
                       projection onto the CURRENT deflated vector, run step-synchronously
                       through the vllm sampler (mathematically the serial chain of
                       ``ola.deflate``, batched across rows; dup-exclusion replaces the
                       redraw loop).

This module is the pure batched math (CPU-testable); GPU/IO orchestration lives in
``scripts/ola/olens_r2s_select.py``. Conventions follow ``ola.ablation`` /
``oracle_lens.nnomp``: a phrase's direction is the unit vector of whiten(AR(phrase)) (centered
by μ), selection scores are plain dot products against whiten(h).
"""

from collections.abc import Sequence
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from oracle_lens.core.nnomp import nnls_refit
from oracle_lens.pipeline.distill import extract_explanation_content, truncate_phrase

__all__ = [
    "best_step_pick",
    "candidate_phrases",
    "omp_select",
    "omp_select_staged",
    "prefix_candidate_phrases",
    "shrink_to_shortest",
    "unique_index",
    "unit_rows",
]


def candidate_phrases(samples: Sequence[str], tokenizer: Any, *, n: int) -> list[str]:
    """Per-sample truncated phrase, index-aligned with ``samples`` (repeats/empties kept).

    Same extract+truncate as assembly (``phrases_from_samples``) but WITHOUT dedup — selection
    needs the sample-index alignment to map picked atoms back to raw samples.
    """
    return [truncate_phrase(extract_explanation_content(s), tokenizer, n=n) for s in samples]


def prefix_candidate_phrases(
    samples: Sequence[str], tokenizer: Any, *, lengths: Sequence[int]
) -> list[str]:
    """Every prefix truncation of every sample, one length block at a time (the r2sp free
    prefix-selection candidate set, user spec 2026-08-10).

    Index-aligned as ``[L0 of sample 0..S-1, L1 of sample 0..S-1, ...]`` — candidate
    ``i*len(samples)+j`` is sample ``j`` truncated to ``lengths[i]`` tokens — so embed
    sidecars keep the same cand-to-source mapping contract as :func:`candidate_phrases`
    (repeats/empties kept; ``unique_index`` dedups downstream). Exactness: the first L tokens
    of a temp-1.0 sample ARE a max_tokens=L draw, so prefixes are true teacher candidates.
    """
    out: list[str] = []
    for n in lengths:
        out.extend(candidate_phrases(samples, tokenizer, n=n))
    return out


def unique_index(cands: Sequence[str]) -> tuple[list[str], list[int]]:
    """(unique non-empty phrases in first-appearance order, per-candidate index; -1 = empty)."""
    uniq: list[str] = []
    where: dict[str, int] = {}
    idx: list[int] = []
    for c in cands:
        if not c:
            idx.append(-1)
            continue
        if c not in where:
            where[c] = len(uniq)
            uniq.append(c)
        idx.append(where[c])
    return uniq, idx


def unit_rows(x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
    """Rows normalized to unit length (zero rows clamp instead of dividing by 0)."""
    out: Tensor = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return out


def omp_select(
    x_w: Float[Tensor, "b d"],
    dirs_w: Float[Tensor, "b c d"],
    valid: Bool[Tensor, "b c"],
    *,
    k: int,
    min_gain: float = 1e-3,
) -> tuple[Int[Tensor, "b k"], Float[Tensor, "b k"], Float[Tensor, "b"]]:
    """Greedy OMP + NNLS refit of each row against its OWN candidate set (method m2).

    Same objective and stopping rules as ``nnomp_batch`` (positive-correlation greedy pick,
    full non-negative refit each step, stop on no-positive-score or FVE gain < ``min_gain``),
    but the dictionary is per-row (``dirs_w`` — whitened UNIT directions). Returns
    ``(sel, coeffs, fve)``: candidate indices in selection order (-1 unused), their NNLS
    coefficients, and the final fraction of whitened variance explained.
    """
    b, _c, d = dirs_w.shape
    dev = x_w.device
    sel = torch.full((b, k), -1, dtype=torch.long, device=dev)
    coeffs = torch.zeros(b, k, device=dev)
    residual = x_w.clone()
    active = torch.ones(b, dtype=torch.bool, device=dev)
    prev_fve = torch.zeros(b, device=dev)

    for step in range(k):
        scores = torch.bmm(dirs_w, residual.unsqueeze(-1)).squeeze(-1)  # [b, c]
        scores = scores.masked_fill(~valid, float("-inf"))
        for j in range(step):  # never reselect
            prev = sel[:, j]
            ok = prev >= 0
            scores[torch.nonzero(ok).squeeze(-1), prev[ok]] = float("-inf")
        best_score, best_idx = scores.max(dim=-1)
        take = active & (best_score > 0) & (prev_fve < 1.0 - min_gain)
        if not bool(take.any()):
            break
        sel[:, step] = torch.where(take, best_idx, sel[:, step])

        valid_sel = sel >= 0
        dsel = torch.gather(dirs_w, 1, sel.clamp_min(0).unsqueeze(-1).expand(b, k, d))
        w, fve = nnls_refit(dsel, x_w, valid_sel)
        recon = torch.bmm(w.unsqueeze(1), dsel).squeeze(1)
        residual = x_w - recon
        gained = fve - prev_fve
        active = take & (gained > min_gain)
        prev_fve = torch.where(take, fve, prev_fve)
        coeffs = torch.where(valid_sel, w, coeffs)

    return sel, coeffs, prev_fve


def omp_select_staged(
    x_w: Float[Tensor, "b d"],
    dirs_w: Float[Tensor, "b c d"],
    stage_valid: list[Bool[Tensor, "b c"]],
    *,
    min_gain: float = 1e-3,
) -> tuple[Int[Tensor, "b k"], Float[Tensor, "b k"], Float[Tensor, "b"]]:
    """Length-staged greedy OMP (method m5, user spec 2026-08-01): the paper's "one phrase
    length at a time" recipe.

    Identical to :func:`omp_select` except step ``j`` may only pick from atoms allowed by
    ``stage_valid[j]`` (the length-Lⱼ candidates). ``k = len(stage_valid)``. Coarse→fine is just
    the caller's ordering of the ladder. A row with no valid atom at a stage skips that stage
    (its slot stays -1); already-picked atoms are never reselected across stages (so a phrase
    shared between two length buckets isn't taken twice). Refit + FVE + stop rules unchanged.
    """
    k = len(stage_valid)
    b, _c, d = dirs_w.shape
    dev = x_w.device
    sel = torch.full((b, k), -1, dtype=torch.long, device=dev)
    coeffs = torch.zeros(b, k, device=dev)
    residual = x_w.clone()
    prev_fve = torch.zeros(b, device=dev)

    for step in range(k):
        scores = torch.bmm(dirs_w, residual.unsqueeze(-1)).squeeze(-1)  # [b, c]
        scores = scores.masked_fill(~stage_valid[step].to(dev), float("-inf"))
        for j in range(step):  # never reselect an atom picked at an earlier stage
            prev = sel[:, j]
            ok = prev >= 0
            scores[torch.nonzero(ok).squeeze(-1), prev[ok]] = float("-inf")
        best_score, best_idx = scores.max(dim=-1)
        take = best_score > 0
        sel[:, step] = torch.where(take, best_idx, sel[:, step])

        valid_sel = sel >= 0
        dsel = torch.gather(dirs_w, 1, sel.clamp_min(0).unsqueeze(-1).expand(b, k, d))
        w, fve = nnls_refit(dsel, x_w, valid_sel)
        recon = torch.bmm(w.unsqueeze(1), dsel).squeeze(1)
        residual = x_w - recon
        prev_fve = torch.where(valid_sel.any(-1), fve, prev_fve)
        coeffs = torch.where(valid_sel, w, coeffs)

    return sel, coeffs, prev_fve


def _token_prefix_of(a: Sequence[int], b: Sequence[int]) -> bool:
    """True iff ``a`` is a (proper or equal) leading-token prefix of ``b``."""
    return len(a) <= len(b) and list(b[: len(a)]) == list(a)


def shrink_to_shortest(
    sel: Sequence[int],
    base_fve: float,
    cand_ids: Sequence[Sequence[int]],
    cand_len: Sequence[int],
    dirs_w: Float[Tensor, "c d"],
    x_w: Float[Tensor, "1 d"],
    *,
    ladder: Sequence[int],
    eps: float,
) -> tuple[list[int], float]:
    """Shrink-to-shortest over an OMP pick set (Agam 2026-08-11: "the shortest prefix that
    reconstructs it best"; extended 2026-08-23 to bullet-prefix candidates).

    Per pick, longest-first: swap it for the SHORTEST candidate that is a leading-token
    prefix of it (same sample/bullet by construction — token-content lookup) while the
    joint k-atom NNLS-refit FVE stays within ``eps`` of ``base_fve``. The budget is one
    global tolerance around the ORIGINAL FVE, checked on every trial, so k shrinks cannot
    silently stack k epsilons. A swap is skipped when it would leave one pick a token-prefix
    of another (such a set is doomed to the assembler's ``prefix_redundant`` reject).

    Only candidate lengths on ``ladder`` are tried — the candidate builder guarantees those
    prefixes exist (when they survived dedup/degeneracy). Returns the (possibly) shrunk
    selection indices and the refit FVE of the final set.
    """
    idmap = {tuple(ids): ci for ci, ids in enumerate(cand_ids)}
    out = [int(j) for j in sel]
    k = len(out)
    ones = torch.ones(1, k, dtype=torch.bool, device=x_w.device)
    for p in sorted(range(k), key=lambda p: -cand_len[out[p]]):
        for n in ladder:
            if n >= cand_len[out[p]]:
                break
            c = idmap.get(tuple(cand_ids[out[p]][:n]))
            if c is None or c in out:
                continue
            others = [out[q] for q in range(k) if q != p]
            if any(
                _token_prefix_of(cand_ids[c], cand_ids[o])
                or _token_prefix_of(cand_ids[o], cand_ids[c])
                for o in others
            ):
                continue
            trial = [c if q == p else out[q] for q in range(k)]
            _, tfve = nnls_refit(dirs_w[trial].unsqueeze(0), x_w, ones)
            if float(tfve[0]) >= base_fve - eps:
                out[p] = c
                break
    _, fve = nnls_refit(dirs_w[out].unsqueeze(0), x_w, ones)
    return out, float(fve[0])


def best_step_pick(
    h_w: Float[Tensor, "b d"],
    dirs_w: Float[Tensor, "b c d"],
    valid: Bool[Tensor, "b c"],
) -> tuple[Int[Tensor, "b"], Float[Tensor, "b"], Bool[Tensor, "b"]]:
    """Best-of-c deflation-step pick (method m4, user decision (a) 2026-08-01).

    Per row: the candidate with the LARGEST whitened projection coefficient onto the current
    (deflated) vector — which is the largest POSITIVE coef whenever one exists, and otherwise
    the least-negative one (accepted anyway so the chain always advances; the returned
    ``positive`` mask logs the fallback rate). ``idx`` is -1 (coef 0) when a row has no valid
    candidate at all.
    """
    scores = torch.bmm(dirs_w, h_w.unsqueeze(-1)).squeeze(-1)  # [b, c]
    scores = scores.masked_fill(~valid, float("-inf"))
    coef, idx = scores.max(dim=-1)
    has = valid.any(-1)
    idx = torch.where(has, idx, torch.full_like(idx, -1))
    coef = torch.where(has, coef, torch.zeros_like(coef))
    return idx, coef, coef > 0
