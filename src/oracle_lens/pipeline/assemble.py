"""Rollouts -> candidate span pool for the long oracle lens (pure, seeded; no I/O).

The core of ``scripts/ola/assemble_long_spans.py``, factored out so the conv_index contract,
generated-region masking, two-pass sampling, and split logic are unit-testable on synthetic
rollouts. See that script's docstring for the full pipeline.
"""

import hashlib
import random
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Bool
from torch import Tensor

from oracle_lens.core.sampling import split_of
from oracle_lens.pipeline.span_store import SpanOnlyPairs
from oracle_lens.pipeline.spans import LongSpan, carve_disjoint_spans, sample_long_spans


@dataclass(frozen=True)
class SpanPoolStats:
    n_convs: int
    n_dup_convs: int
    n_spans: int


def build_span_pool(
    rollout_shards: Iterable[list[dict[str, Any]]],
    *,
    seed: int = 0,
    n_max: int = 1024,
    base_per_conv: int = 24,
    long_per_conv: int = 8,
    long_n_min: int = 513,
    mid_per_conv: int = 0,
    mid_n_min: int = 257,
    mid_n_max: int = 512,
    eval_frac: float = 0.005,
    dataset_id: str = "lmsys_long",
    sampler: str = "passes",
    gap_max: int = 32,
) -> tuple[SpanOnlyPairs, Bool[Tensor, " n"], SpanPoolStats]:
    """Sample the candidate span pool from rollout shards (in the given order).

    ``conv_index`` = global row index over the concatenated shards — the exact
    ``dedup.seed_keys_from_rollouts`` contract. Only the FIRST conversation per seed hash
    samples spans (conv-level exact dedup); the mask is the generated region only. Two
    ``sampler`` modes:

    - ``"passes"`` (the original 5M-pair recipe): up to three ``sample_long_spans`` passes
      with distinct rng streams — base log-uniform, targeted long (>= ``long_n_min``), and an
      optional targeted MID pass (``mid_per_conv`` > 0, range [``mid_n_min``, ``mid_n_max``])
      for octaves the base pass under-supplies — with cross-pass duplicates dropped on
      ``(start, n)``. Spans OVERLAP freely (measured 4.26x token redundancy at full harvest,
      the 2026-07-29 val_ce-rise cause).
    - ``"disjoint"``: one ``carve_disjoint_spans`` pass (rng stream ``{seed}:carve:{conv}``) —
      mutually non-overlapping spans, ``long_per_conv``/``mid_per_conv`` reserved placements,
      random inter-span gaps up to ``gap_max``. ``base_per_conv`` is ignored (yield is bounded
      by content, not per-conv quotas).

    Returns ``(pool, eval_mask, stats)`` where ``eval_mask`` is the conv-level ``split_of``
    verdict per row.
    """
    if sampler not in ("passes", "disjoint"):
        raise ValueError(f"unknown sampler {sampler!r}")
    # array('i'/'q'), not Python lists: the flat id pool is ~1e9 elements at full scale and
    # boxed ints would be ~10x the memory of packed 4/8-byte arrays.
    flat_ids = array("i")
    offsets = array("q", [0])
    conv_col = array("q")
    prev_pos_col = array("q")
    prev_tok_col = array("q")
    prev_asst_col = array("b")
    split_col = array("b")  # 1 = eval
    seen_keys: set[bytes] = set()
    n_convs = n_dup = 0

    for shard in rollout_shards:
        for row in shard:
            conv_index = n_convs
            n_convs += 1
            seed_text = str(row.get("seed", ""))
            key = hashlib.blake2b(seed_text.encode(), digest_size=16).digest()
            if key in seen_keys:  # first occurrence wins (dedup.dedup_convs semantics)
                n_dup += 1
                continue
            seen_keys.add(key)
            prompt_ids = list(row["prompt_ids"])
            output_ids = list(row["output_ids"])
            rendered = prompt_ids + output_ids
            n_prompt = len(prompt_ids)
            mask = [i >= n_prompt for i in range(len(rendered))]
            if sampler == "disjoint":
                base: list[LongSpan] = carve_disjoint_spans(
                    mask,
                    rng=random.Random(f"{seed}:carve:{conv_index}"),
                    n_max=n_max,
                    gap_max=gap_max,
                    long_per_conv=long_per_conv,
                    long_n_min=min(long_n_min, n_max),
                    mid_per_conv=mid_per_conv,
                    mid_n_min=min(mid_n_min, n_max),
                    mid_n_max=min(mid_n_max, n_max),
                )
                long_pass: list[LongSpan] = []
                mid_pass: list[LongSpan] = []
            else:
                base = sample_long_spans(
                    mask,
                    max_per_conv=base_per_conv,
                    rng=random.Random(f"{seed}:base:{conv_index}"),
                    n_min=1,
                    n_max=n_max,
                    max_attempts=96,
                )
                long_pass = sample_long_spans(
                    mask,
                    max_per_conv=long_per_conv,
                    rng=random.Random(f"{seed}:long:{conv_index}"),
                    n_min=min(long_n_min, n_max),
                    n_max=n_max,
                    max_attempts=64,
                )
                mid_pass = (
                    sample_long_spans(
                        mask,
                        max_per_conv=mid_per_conv,
                        rng=random.Random(f"{seed}:mid:{conv_index}"),
                        n_min=min(mid_n_min, n_max),
                        n_max=min(mid_n_max, n_max),
                        max_attempts=48,
                    )
                    if mid_per_conv > 0
                    else []
                )
            taken: set[tuple[int, int]] = set()
            # v2 rows carry their source conversation's turn-1 hash as split_key so every
            # user-turn seed of one lmsys conversation lands on the same train/eval side
            # (later turns share the human's topic with a turn-1 rollout that may sit in the
            # reused eval set). Absent (all v1 rows): own seed hash — the historical verdict.
            split_key = str(row.get("split_key") or key.hex())
            is_eval = split_of(f"{dataset_id}:{split_key}", eval_frac=eval_frac) == "eval"
            for span in [*base, *long_pass, *mid_pass]:
                if (span.start, span.n_tokens) in taken:
                    continue
                taken.add((span.start, span.n_tokens))
                flat_ids.extend(rendered[span.start : span.start + span.n_tokens])
                offsets.append(len(flat_ids))
                conv_col.append(conv_index)
                prev_pos_col.append(span.prev_pos)
                prev_tok_col.append(rendered[span.prev_pos])
                prev_asst_col.append(int(span.prev_is_assistant))
                split_col.append(int(is_eval))

    def _t(a: "array[int]", dtype: torch.dtype) -> Tensor:
        return torch.frombuffer(a, dtype=dtype).clone() if len(a) else torch.empty(0, dtype=dtype)

    pool = SpanOnlyPairs(
        span_ids=_t(flat_ids, torch.int32),
        offsets=torch.frombuffer(offsets, dtype=torch.int64).clone(),
        conv_index=_t(conv_col, torch.int64),
        prev_pos=_t(prev_pos_col, torch.int64),
        prev_token_id=_t(prev_tok_col, torch.int64),
        prev_is_assistant=_t(prev_asst_col, torch.int8).bool(),
    )
    eval_mask = _t(split_col, torch.int8).bool()
    return pool, eval_mask, SpanPoolStats(n_convs, n_dup, len(pool))
