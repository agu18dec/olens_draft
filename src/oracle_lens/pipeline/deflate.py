"""Deflation-chain teacher sampling core (the round-2 ``*d`` variants).

Instead of ``k`` iid teacher samples per activation, a CHAIN (user decision 2026-07-30):
sample ONE phrase, map it through the frozen AR reconstructor, project the reconstruction's
whitened unit direction OUT of the activation (:func:`~oracle_lens.pipeline.ablation.
project_out_rows` — the same rank-1 whitened ablation as the multi-token robustness test), and
sample again from the deflated vector — ``k`` steps. Each successive draw is pushed off the
already-explained mode, so the k phrases cover complementary components in greedy dominance
order — the superposition-coverage fix for iid temp-1.0 sampling, which under-covers minority
components of a superposed activation.

Dup policy (user decision 2026-07-30): a draw whose truncated-to-n phrase repeats an earlier
accepted phrase of the SAME chain (or is empty) is redrawn at the unchanged deflated vector,
up to ``resample_cap`` extra draws per step; then the dup is accepted (logged via
``resamples``). An accepted dup still deflates (re-projecting a mostly-removed direction is a
near-no-op); an empty phrase never does (``coef == 0.0``).

This module is the model-free, CPU-testable batched chain; the model-bound sample/reconstruct
closures live in ``scripts/ola/olens_distill_sample_cluster.py`` (``--engine hf`` — the vllm
rollout venv can import neither the AR nor this module, and the chain's serial dependency
doesn't fit vLLM's batched ``n=k'`` sampling anyway).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from oracle_lens.pipeline.ablation import WhitenedSpace, project_out_rows
from oracle_lens.pipeline.distill import extract_explanation_content, truncate_phrase

# (current deflated vectors of the PENDING rows, their chain indices) -> one raw sample each.
SampleFn = Callable[[Float[Tensor, "p d"], list[int]], list[str]]
# (truncated phrases) -> raw-space AR predictions, one row per phrase.
ReconFn = Callable[[list[str]], Float[Tensor, "p d"]]


@dataclass(frozen=True)
class ChainResult:
    """Per chain (= per activation row): the k accepted raw samples in chain order, the signed
    whitened coefficient each step removed (0.0 when the phrase was empty ⇒ no deflation), the
    redraw count each step consumed, and the whitened norm of the ORIGINAL vector (so
    ``coef²/wnorm0²`` is the fraction of whitened variance a step explained)."""

    samples: list[list[str]]
    coefs: list[list[float]]
    resamples: list[list[int]]
    wnorm0: list[float]


def run_deflation_chains(
    h0: Float[Tensor, "b d"],
    k: int,
    space: WhitenedSpace,
    sample_fn: SampleFn,
    recon_fn: ReconFn,
    tokenizer: Any,
    *,
    n: int,
    resample_cap: int,
    min_tokens: int = 0,
) -> ChainResult:
    """Run ``b`` deflation chains of ``k`` steps, batched step-synchronously.

    All rows must share one layer/``space`` (the caller groups by layer). Vectors stay in RAW
    activation space between steps; each projection round-trips through ``space``.

    ``min_tokens`` (user decision 2026-07-30, k4n256d's >150 floor): a distinct-but-short draw
    is redrawn like a dup; at ``resample_cap`` the LONGEST distinct candidate seen this step is
    accepted instead of the last draw (it still deflates and is logged — assembly's
    ``select_phrases`` floor decides whether it enters the target).
    """
    b = int(h0.shape[0])
    h = h0.float().clone()
    wnorm0 = space.whiten(h).norm(dim=-1)
    seen: list[set[str]] = [set() for _ in range(b)]
    samples: list[list[str]] = [[] for _ in range(b)]
    coefs: list[list[float]] = [[] for _ in range(b)]
    resamples: list[list[int]] = [[] for _ in range(b)]

    def _tok_len(key: str) -> int:
        return len(tokenizer(key, add_special_tokens=False)["input_ids"])

    for _step in range(k):
        texts: list[str] = [""] * b
        keys: list[str] = [""] * b
        redraws = [0] * b
        best: list[tuple[int, str, str] | None] = [None] * b  # (len, raw, key), distinct only
        pending = list(range(b))
        attempt = 0
        while pending:
            drawn = sample_fn(h[pending], pending)
            still: list[int] = []
            for bi, raw in zip(pending, drawn, strict=True):
                key = truncate_phrase(extract_explanation_content(raw), tokenizer, n=n)
                distinct = bool(key) and key not in seen[bi]
                if distinct and (not min_tokens or _tok_len(key) > min_tokens):
                    texts[bi], keys[bi] = raw, key
                elif attempt >= resample_cap:
                    kept = best[bi]  # local binds so mypy keeps the not-None narrowing
                    if kept is not None:  # longest distinct-but-short candidate
                        _, texts[bi], keys[bi] = kept
                    else:  # dup/empty — accepted and logged, as before
                        texts[bi], keys[bi] = raw, key
                else:
                    if distinct:
                        cand = (_tok_len(key), raw, key)
                        cur = best[bi]
                        if cur is None or cand[0] > cur[0]:
                            best[bi] = cand
                    redraws[bi] += 1
                    still.append(bi)
            pending = still
            attempt += 1
        for bi in range(b):
            if keys[bi]:
                seen[bi].add(keys[bi])
            samples[bi].append(texts[bi])
            resamples[bi].append(redraws[bi])

        recon_rows = [bi for bi in range(b) if keys[bi]]
        step_coefs = [0.0] * b
        if recon_rows:
            preds = recon_fn([keys[bi] for bi in recon_rows])
            h_new, cf = project_out_rows(space, h[recon_rows], preds.to(h.device))
            h[recon_rows] = h_new
            for bi, c in zip(recon_rows, cf.tolist(), strict=True):
                step_coefs[bi] = float(c)
        for bi in range(b):
            coefs[bi].append(step_coefs[bi])

    return ChainResult(samples=samples, coefs=coefs, resamples=resamples, wnorm0=wnorm0.tolist())
