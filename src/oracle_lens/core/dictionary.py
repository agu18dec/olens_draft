"""Dictionary-phrase sampling and storage (oracle-lens stage 2).

Paper recipe: sample start positions from held-out Assistant text, take the 2/4/8/16/32-token
continuation at each, dedupe within each length, and vectorize every phrase with the
reconstructor. This module does the sampling (pure, seeded — positions come from assistant
turns, continuations must stay inside the turn) and the per-length dedup/IO; vectorization
lives in the Modal launcher next to the checkpoint loader.
"""

import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch
from jaxtyping import Float, Int
from safetensors.torch import load_file, save_file
from torch import Tensor

DICT_LENGTHS: tuple[int, ...] = (2, 4, 8, 16, 32)


def fits_fn(assistant_mask: list[bool], limit: int) -> Callable[[int, int], bool]:
    prefix = [0]
    for m in assistant_mask[:limit]:
        prefix.append(prefix[-1] + int(m))

    def fits(start: int, n: int) -> bool:
        return start + n <= limit and prefix[start + n] - prefix[start] == n

    return fits


def sample_dictionary_positions(
    assistant_mask: list[bool],
    rendered_ids: list[int],
    *,
    max_positions: int,
    rng: random.Random,
    lengths: tuple[int, ...] = DICT_LENGTHS,
) -> list[int]:
    """The sampled start positions themselves (the paper's "one million start positions").

    Exposed separately from the phrases because the teacher decomposes the GT activations at
    THESE positions — the positions manifest is what the stage-3 capture pass replays.
    """
    limit = min(len(assistant_mask), len(rendered_ids))
    fits = fits_fn(assistant_mask, limit)
    starts = [s for s in range(limit) if fits(s, min(lengths))]
    rng.shuffle(starts)
    return starts[:max_positions]


def phrases_at_positions(
    assistant_mask: list[bool],
    rendered_ids: list[int],
    positions: list[int],
    *,
    lengths: tuple[int, ...] = DICT_LENGTHS,
) -> dict[int, list[tuple[int, ...]]]:
    """Per position, the continuation at every length that fits.

    A continuation of length L "fits" iff mask[start : start+L] is all True (fully inside one
    assistant turn). Near-turn-end positions therefore contribute only the short lengths —
    deliberately keeping delimiter-adjacent phrases in the dictionary (paper observation).
    """
    limit = min(len(assistant_mask), len(rendered_ids))
    fits = fits_fn(assistant_mask, limit)
    out: dict[int, list[tuple[int, ...]]] = {n: [] for n in lengths}
    for start in positions:
        for n in lengths:
            if fits(start, n):
                out[n].append(tuple(rendered_ids[start : start + n]))
    return out


def sample_dictionary_phrases(
    assistant_mask: list[bool],
    rendered_ids: list[int],
    *,
    max_positions: int,
    rng: random.Random,
    lengths: tuple[int, ...] = DICT_LENGTHS,
) -> dict[int, list[tuple[int, ...]]]:
    """Up to ``max_positions`` start positions; per position, every length that fits."""
    positions = sample_dictionary_positions(
        assistant_mask, rendered_ids, max_positions=max_positions, rng=rng, lengths=lengths
    )
    return phrases_at_positions(assistant_mask, rendered_ids, positions, lengths=lengths)


def merge_deduped(
    shards: list[dict[int, list[tuple[int, ...]]]],
    *,
    lengths: tuple[int, ...] = DICT_LENGTHS,
) -> dict[int, tuple[Int[Tensor, "n len"], Int[Tensor, "n"]]]:
    """Exact token-id dedup per length -> (phrase_ids, occurrence counts)."""
    out: dict[int, tuple[Tensor, Tensor]] = {}
    for n in lengths:
        counts: Counter[tuple[int, ...]] = Counter()
        for shard in shards:
            counts.update(shard.get(n, []))
        if not counts:
            continue
        phrases = list(counts.keys())
        out[n] = (
            torch.tensor(phrases, dtype=torch.long),
            torch.tensor([counts[p] for p in phrases], dtype=torch.long),
        )
    return out


def save_dictionary_length(
    path: Path,
    phrase_ids: Int[Tensor, "n len"],
    counts: Int[Tensor, "n"],
    *,
    meta: dict[str, str],
    vectors: Float[Tensor, "n d"] | None = None,
) -> None:
    tensors = {"phrase_ids": phrase_ids.cpu(), "counts": counts.cpu()}
    if vectors is not None:
        tensors["vectors"] = vectors.to(torch.float16).cpu()
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata=meta)
    tmp.replace(path)


def load_dictionary_length(path: Path, *, device: str = "cpu") -> dict[str, Tensor]:
    return load_file(str(path), device=device)
