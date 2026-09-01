"""Phrase-span sampling for reconstructor pairs (pure, seeded, deterministic).

A pair is (phrase = tokens ``start..start+n_tokens-1`` of a rendered conversation, target = the
residual-stream activation at ``prev_pos = start - 1``). Spans lie entirely inside assistant
turns; ``prev_pos`` itself may be a non-assistant token (e.g. the newline after the assistant
header when the phrase opens a turn) — faithful to "the position immediately preceding", and
recorded via ``prev_is_assistant`` so metrics can be sliced by it later.

N is drawn FIRST, then a uniform valid start: drawing position-first would bias against large N
(long spans fit in fewer places), while N-first keeps the N distribution uniform by construction.
"""

import hashlib
import random
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal


@dataclass(frozen=True)
class PhraseSpan:
    """One sampled phrase span within a rendered conversation."""

    prev_pos: int  # position of the target activation (= start - 1)
    start: int  # first phrase token
    n_tokens: int  # N, uniform in [n_min, n_max]
    prev_is_assistant: bool


def split_of(conv_key: str, *, eval_frac: float = 0.05) -> Literal["train", "eval"]:
    """Deterministic, process-stable train/eval split by conversation key.

    sha256, not ``hash()`` — the builtin is salted per process and would scatter the split
    across shards/reruns. ``conv_key`` convention: ``f"{dataset_id}:{doc_index}"``.
    """
    digest = hashlib.sha256(conv_key.encode()).digest()
    frac = int.from_bytes(digest[:8], "big") / 2**64
    return "eval" if frac < eval_frac else "train"


def sample_phrase_spans(
    assistant_mask: list[bool],
    *,
    max_per_conv: int,
    rng: random.Random,
    n_min: int = 1,
    n_max: int = 32,
    max_attempts: int = 64,
) -> list[PhraseSpan]:
    """Up to ``max_per_conv`` deduplicated spans fully inside assistant turns, ``start >= 1``.

    Per draw: N ~ U{n_min..n_max}, then a uniform start among positions where
    ``assistant_mask[start:start+N]`` is all-True. If the drawn N fits nowhere (short turns),
    the draw is retried with a fresh N (bounded by ``max_attempts`` total per conversation),
    so short-turn conversations still contribute small-N pairs.
    """
    length = len(assistant_mask)
    # prefix[i] = number of True in mask[:i]; span all-True iff prefix[s+n] - prefix[s] == n
    prefix = [0, *accumulate(int(m) for m in assistant_mask)]
    seen: set[tuple[int, int]] = set()
    spans: list[PhraseSpan] = []
    attempts = 0
    while len(spans) < max_per_conv and attempts < max_attempts:
        attempts += 1
        n = rng.randint(n_min, n_max)
        valid = [s for s in range(1, length - n + 1) if prefix[s + n] - prefix[s] == n]
        if not valid:
            continue
        start = valid[rng.randrange(len(valid))]
        if (start, n) in seen:
            continue
        seen.add((start, n))
        spans.append(
            PhraseSpan(
                prev_pos=start - 1,
                start=start,
                n_tokens=n,
                prev_is_assistant=assistant_mask[start - 1],
            )
        )
    return spans
