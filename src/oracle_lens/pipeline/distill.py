"""AO self-distillation: turn k teacher rollouts per activation into a list target.

The current AO emits ONE ``<explanation>...</explanation>`` per sample. To distil a
multiple-outputs-in-one-forward student we sample the frozen teacher AO ``k`` times at temp 1.0,
strip the tag wrapper, drop exact duplicates (keeping first appearance — no frequency ranking, per
the locked decision), and format the surviving uniques as a single list target of ``k`` tagged
blocks. This module is the pure, CPU-testable core (dedup + target formatting + k selection); the
GPU sampling and Modal fan-out live in ``scripts/ola/ao_distill_modal.py``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "VARIANTS",
    "DistillVariant",
    "dedup_keep_order",
    "extract_all_explanations",
    "extract_explanation_content",
    "format_list_target_text",
    "layer_draw",
    "phrases_from_samples",
    "pick_k",
    "repeat_rate",
    "select_phrases",
    "shortfall_rows",
    "strip_explanation",
    "truncate_phrase",
]

_TAG = re.compile(r"</?explanation>", re.IGNORECASE)
_CLOSE = "</explanation>"


def strip_explanation(text: str) -> str:
    """Drop ``<explanation>``/``</explanation>`` tags and surrounding whitespace from one sample."""
    return _TAG.sub("", text).strip()


def extract_explanation_content(text: str) -> str:
    """One teacher sample -> its explanation content, robust to max_tokens truncation.

    Beyond :func:`strip_explanation` (complete tags anywhere), a sample cut off by ``max_new``
    can end mid-way through ``</explanation>`` (e.g. ``…content\\n</expl``) or even mid-way
    through a SECOND ``<explanation`` opener; any trailing prefix of a tag is dropped too.
    Content is whatever precedes the (possibly partial) close tag, tags removed, stripped.
    """
    # keep only the first block's content: cut at the first COMPLETE close tag
    cut = text.find(_CLOSE)
    if cut >= 0:
        text = text[:cut]
    # drop a trailing PARTIAL tag: longest suffix that is a proper prefix of a tag string
    for tag in (_CLOSE, "<explanation>"):
        for j in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:j]):
                text = text[:-j]
                break
    return _TAG.sub("", text).strip()


def extract_all_explanations(text: str) -> list[str]:
    """Every ``<explanation>`` block's content from a LIST generation (student outputs).

    Splits on the opener and reuses :func:`extract_explanation_content` per segment, so a final
    block truncated mid-close-tag survives; empty blocks are dropped. Text with no opener at all
    is treated as one bare block (mirrors the single-sample parser's fallback).
    """
    parts = text.split("<explanation>")
    segments = parts[1:] if len(parts) > 1 else parts
    out = [extract_explanation_content(s) for s in segments]
    return [c for c in out if c]


@dataclass(frozen=True)
class DistillVariant:
    """One round-2 student: k phrases of <= n tokens each, plus its sampling budget."""

    name: str
    k: int  # phrases per target list
    n: int  # per-phrase token cap (enforced by truncation/max_tokens, not the prompt)
    k_prime: int  # teacher samples per activation (oversampled so dedup still yields k)
    max_new: int  # teacher max_new_tokens = n + <explanation> wrapper slack
    sampler: str = "iid"  # "iid" (one n=k' generate) or "deflate" (k-step chain, ola/deflate.py)
    min_phrase_tokens: int = 0  # assembly keeps only phrases LONGER than this (0 = no floor)


# The four iid round-2 students (user spec 2026-07-29). k1n1024 keeps the round-1 single-block
# format (control); the others are list students. k' oversamples where short phrases collide.
# The "d" variants (user spec 2026-07-30) mirror the multi-phrase students but sample a
# DEFLATION CHAIN: k sequential draws, each after projecting the previous phrase's whitened AR
# direction out of the activation — k_prime is 1 draw per step (dups resampled in-loop).
VARIANTS: dict[str, DistillVariant] = {
    v.name: v
    for v in (
        DistillVariant("k1n1024", k=1, n=1024, k_prime=8, max_new=1040, min_phrase_tokens=400),
        DistillVariant("k4n256", k=4, n=256, k_prime=40, max_new=272, min_phrase_tokens=150),
        DistillVariant("k20n4", k=20, n=4, k_prime=48, max_new=12),
        DistillVariant("k20n16", k=20, n=16, k_prime=32, max_new=28),
        DistillVariant(
            "k4n256d", k=4, n=256, k_prime=1, max_new=272, sampler="deflate", min_phrase_tokens=150
        ),
        DistillVariant("k20n4d", k=20, n=4, k_prime=1, max_new=12, sampler="deflate"),
        DistillVariant("k20n16d", k=20, n=16, k_prime=1, max_new=28, sampler="deflate"),
        # Round-2 SELECTION variants (user spec 2026-08-01): "pool*" sample a shared pool of k'
        # iid 32-token samples per activation; the k=4 target phrases are picked DOWNSTREAM by a
        # selection stage (dict-NNOMP / within-prompt-NNOMP / LM judge), NOT by select_phrases'
        # first-k. k4n32d16 is the best-of-16 deflation chain (method 4), run step-synchronously
        # through the vllm sampler: k_prime = candidate draws per chain step, best positive
        # whitened projection wins, dups excluded before the argmax.
        DistillVariant("pool16n32", k=4, n=32, k_prime=16, max_new=48),
        DistillVariant("pool64n32", k=4, n=32, k_prime=64, max_new=48),
        DistillVariant("pool128n32", k=4, n=32, k_prime=128, max_new=48),
        DistillVariant("pool256n32", k=4, n=32, k_prime=256, max_new=48),
        DistillVariant("k4n32d16", k=4, n=32, k_prime=16, max_new=48, sampler="deflate"),
        DistillVariant("k4n32d64", k=4, n=32, k_prime=64, max_new=48, sampler="deflate"),
        DistillVariant("k4n32d128", k=4, n=32, k_prime=128, max_new=48, sampler="deflate"),
        DistillVariant("k4n32d256", k=4, n=32, k_prime=256, max_new=48, sampler="deflate"),
    )
}


def layer_draw(n_rows: int, n_layers: int, *, seed: int, split: str) -> torch.Tensor:
    """Seeded per-row layer assignment (index into LAYERS), shared by every round-2 variant.

    Canonical copy of the sampler script's inline ``layer_draw_for`` (kept inline there for the
    vllm venv); the sync unit test asserts the two produce identical draws.
    """
    gen = torch.Generator().manual_seed(seed + (1 if split == "eval" else 0))
    return torch.randint(n_layers, (n_rows,), generator=gen)


def truncate_phrase(content: str, tokenizer: Any, *, n: int) -> str:
    """Explanation content -> the <= ``n``-token phrase string (the dedup key / target unit).

    The single source of truth for truncation: the deflation chain's in-loop dup check and
    :func:`phrases_from_samples` (assembly) must agree on what counts as "the same phrase".
    """
    if not content:
        return ""
    ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    return str(tokenizer.decode(ids[:n])).strip() if len(ids) > n else content


def select_phrases(samples: Sequence[str], tokenizer: Any, var: DistillVariant) -> list[str]:
    """Assembly-side phrase selection for one row: dedup/truncate, apply the variant's
    length floor (keep only phrases with MORE than ``min_phrase_tokens`` content tokens,
    first-appearance order — approximates the length-conditional draw distribution rather
    than a longest-of-N max bias), then cap at ``k``."""
    cand = phrases_from_samples(samples, tokenizer, n=var.n)
    if var.min_phrase_tokens:
        cand = [
            ph
            for ph in cand
            if len(tokenizer(ph, add_special_tokens=False)["input_ids"]) > var.min_phrase_tokens
        ]
    return cand[: var.k]


def phrases_from_samples(samples: Sequence[str], tokenizer: Any, *, n: int) -> list[str]:
    """Raw teacher samples -> deduped phrase list, each truncated to <= ``n`` tokens.

    Per sample: extract the explanation content, tokenize (no special tokens), cut at ``n``
    ids, re-decode, strip. Dedup (first appearance) runs on the TRUNCATED text — two samples
    that differ only past the cap are one phrase.
    """
    out: list[str] = []
    seen: set[str] = set()
    for s in samples:
        phrase = truncate_phrase(extract_explanation_content(s), tokenizer, n=n)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


def shortfall_rows(unique_counts: Sequence[int], k: int) -> list[int]:
    """Indices whose deduped phrase count is below ``k`` (the top-up re-sampling work list)."""
    return [i for i, c in enumerate(unique_counts) if c < k]


def dedup_keep_order(samples: Sequence[str]) -> list[str]:
    """Strip tag wrappers, drop empties and exact-duplicate strings, keep first appearance.

    First appearance order (NOT frequency-sorted): the locked target-ordering decision.
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in samples:
        c = strip_explanation(s)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def repeat_rate(samples: Sequence[str]) -> float:
    """Fraction of (non-empty) samples that are exact repeats of an earlier one.

    ``1 - n_unique / n_nonempty`` — a degeneracy signal for the pilot inspection (high ⇒ temp 1.0
    is collapsing to a few strings). Returns 0.0 when there are no non-empty samples.
    """
    cleaned = [strip_explanation(s) for s in samples]
    nonempty = [c for c in cleaned if c]
    if not nonempty:
        return 0.0
    return 1.0 - len(set(nonempty)) / len(nonempty)


def format_list_target_text(uniques: Sequence[str], k: int) -> str:
    """The first ``k`` uniques as newline-joined ``<explanation>`` blocks (the CE target text).

    Whole-string formatting (tokenized in one pass by the caller) avoids BPE boundary merges that
    per-segment tokenization of tags/content would introduce.
    """
    blocks = [f"<explanation>\n{u}\n</explanation>" for u in uniques[:k]]
    return "\n".join(blocks)


def pick_k(n_unique: int, k_max: int, generator: torch.Generator) -> int:
    """A random requested count in ``[1, min(k_max, n_unique)]`` (0 if no uniques).

    Randomizing k per example teaches the student to honor a requested list length; the target is
    then the first-k uniques. Uses a torch ``Generator`` for reproducibility (mirrors the seeded
    permutation in ``build_ao1_examples``).
    """
    hi = min(k_max, n_unique)
    if hi <= 0:
        return 0
    return int(torch.randint(1, hi + 1, (1,), generator=generator).item())
