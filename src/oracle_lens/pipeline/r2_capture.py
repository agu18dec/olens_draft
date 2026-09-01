"""Round-2 fresh-position draw over the held-out lmsys rollouts (pure, seeded, testable).

Round 2 samples the frozen round-1 lens on TRUE residuals at positions the round-1 spans never
used: for each conversation, the candidate read positions are every ``prev_pos`` whose FOLLOWING
token is model-generated (the round-1 span convention: a span starting at ``s`` reads the
residual at ``s - 1``), minus the round-1 exclusion set. The draw is per-conv seeded so the
CPU draw pass and any re-run agree exactly; global selection then walks a seeded shuffle of
eligible conversations until the position quota is met.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DrawRecord",
    "draw_conv_positions",
    "select_quota",
    "valid_prev_positions",
]


def valid_prev_positions(generated_mask: Sequence[bool]) -> list[int]:
    """Every ``prev_pos`` = s-1 for generated positions s >= 1 (the span-start convention).

    ``prev_pos`` itself may be the last prompt token (``prev_is_assistant`` False in round 1) —
    what matters is that the text FOLLOWING the read position is generated.
    """
    return [s - 1 for s in range(1, len(generated_mask)) if generated_mask[s]]


def draw_conv_positions(
    generated_mask: Sequence[bool],
    *,
    per_conv: int,
    rng: random.Random,
    exclude: frozenset[int] | set[int] = frozenset(),
) -> list[int]:
    """Up to ``per_conv`` distinct fresh prev-positions, seeded, sorted ascending.

    ``exclude`` = the conversation's round-1 span ``prev_pos`` set — round-2 positions are
    guaranteed disjoint from what round 1 trained on (user decision 2026-07-29).
    """
    cands = [p for p in valid_prev_positions(generated_mask) if p not in exclude]
    if not cands:
        return []
    return sorted(rng.sample(cands, min(per_conv, len(cands))))


@dataclass(frozen=True)
class DrawRecord:
    """One conversation's draw: where it lives on disk and which positions to capture."""

    shard: str  # rollout shard filename
    row: int  # row index within that shard
    conv_index: int  # global conv index (the round-1 assembly contract)
    positions: tuple[int, ...]


def select_quota(
    records: Sequence[DrawRecord], *, quota: int, seed: int, label: str
) -> list[DrawRecord]:
    """Seeded-shuffle ``records`` and keep just enough conversations to reach ``quota`` positions.

    The last kept record is trimmed so the total is EXACTLY ``quota`` (when supply allows);
    under-supply keeps everything and the caller warns loudly.
    """
    order = list(records)
    random.Random(f"{seed}:r2select:{label}").shuffle(order)
    out: list[DrawRecord] = []
    total = 0
    for rec in order:
        if total >= quota:
            break
        take = min(len(rec.positions), quota - total)
        out.append(
            rec if take == len(rec.positions) else DrawRecord(
                rec.shard, rec.row, rec.conv_index, rec.positions[:take]
            )
        )
        total += take
    return out
