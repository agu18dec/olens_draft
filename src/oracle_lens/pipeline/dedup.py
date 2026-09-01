"""Conversation-level exact dedup for the GT-continuation lens (design.md §3).

A deliberate departure from the 2026-07-19 "duplicates stay, we just count them" decision (which
stands for the AR/AO lines): lmsys seeds repeat heavily, and duplicated seeds produce
near-identical on-policy rollouts, so this experiment drops every row of all but the FIRST
conversation carrying a given key. The key is the seed text (the lmsys first user turn the
rollout was generated from), recovered from the rollout JSONs the capture step consumed —
``seed_keys_from_rollouts`` assumes ``conv_index`` enumerates the concatenated rollout rows in
path order (asserted against the pair pool by the fetch script). Rows whose conversation has no
recoverable key are KEPT and counted — dedup must fail open, never silently shrink the pool.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from oracle_lens.pipeline.multilayer import MultiLayerPairs


def seed_keys_from_rollouts(paths: list[Path]) -> dict[int, bytes]:
    """``conv_index -> blake2b(seed text)`` from onpolicy rollout shards (row order = paths)."""
    keys: dict[int, bytes] = {}
    idx = 0
    for p in sorted(paths):
        for row in json.loads(p.read_text()):
            seed = str(row.get("seed", ""))
            keys[idx] = hashlib.blake2b(seed.encode(), digest_size=16).digest()
            idx += 1
    return keys


def dedup_convs(
    pairs: MultiLayerPairs, keys: Mapping[int, bytes], *, label: str = ""
) -> tuple[MultiLayerPairs, dict[str, int]]:
    """Keep rows whose conversation is the first occurrence of its key (or has no key).

    Winner per key = the smallest ``conv_index`` present in ``pairs`` (deterministic, order-free).
    """
    conv = pairs.conv_index.tolist()
    winner: dict[bytes, int] = {}
    for c in sorted(set(conv)):
        k = keys.get(c)
        if k is not None and k not in winner:
            winner[k] = c
    winners = set(winner.values())
    keep_flags = [c in winners if keys.get(c) is not None else True for c in conv]
    keep_mask = torch.tensor(keep_flags, dtype=torch.bool)
    keep = torch.nonzero(keep_mask).squeeze(-1)
    stats = {
        "n_rows": len(pairs),
        "n_kept": int(keep.shape[0]),
        "n_dropped": len(pairs) - int(keep.shape[0]),
        "n_convs": len(set(conv)),
        "n_convs_unkeyed": sum(1 for c in set(conv) if keys.get(c) is None),
    }
    if label:
        pct = 100.0 * stats["n_dropped"] / max(1, stats["n_rows"])
        print(
            f"[conv dedup] {label}: dropped {stats['n_dropped']}/{stats['n_rows']} rows "
            f"({pct:.2f}%), {stats['n_convs_unkeyed']}/{stats['n_convs']} convs unkeyed (kept)"
        )
    return pairs.select(keep), stats
