"""AO-ladder training pool: fresh on-policy crops, provably disjoint from AR training.

Construction (user spec 2026-07-28, `docs/project/experiments/ola/ao_ladder.md`):

1. Sample start positions from the raw on-policy chat rollouts' generated region (``output_ids``
   — produced with ``enable_thinking=False``, so pure assistant text). Starts are drawn from a
   **non-overlapping tiling** of each conversation, so no two windows share a token.
2. Take **ONE** continuation per window at a seeded length from {2,4,8,16,32,64}. Nested crops of
   one window are prefixes of the same text: keeping all six made each text a target ~4.8x (x k
   layers = ~24x per epoch) and produced textbook overfitting inside a single epoch — train loss
   1.05 -> 0.35 while val CE bottomed at 32M tokens and rose to 1.03 (2026-07-29). Length coverage
   is preserved ACROSS windows, not within one.
3. **Dedup within each length** (exact token-id hash, first occurrence wins).
4. **AR-disjointness, config-free:** every AR training crop is a *prefix of a pairs row*, and the
   AR reads ONLY the span ids (no conversation context) — so dropping any AO span whose content
   equals a pairs-row prefix at that length guarantees the AR never fit that exact input,
   regardless of which crop pools the (Modal-side, unfetchable) b512 rungs drew.

The pool carries **no activations**: AO training needs only ``(span ids, AR(span))``; true ``h``
is eval-only (the pairs' first-4096-row carve). Crop index space is the row-major enumeration of
``keep`` — both the AR-output precompute and the trainer derive it identically from the artifact.
"""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Bool, Int
from torch import Tensor

# Legacy defaults (pool v1/v2/iolens/ext1). Pools are SELF-DESCRIBING since 2026-08-04: the
# artifact's `crop_lengths` metadata is the source of truth, read back via `AOPool.lengths` —
# consumers must index lengths through the pool, never through this constant (the long-emission
# ext2 pools go beyond 64). Same pattern as the arout shards' `ao_layers`.
CROP_LENGTHS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
WINDOW = 64

# Same placeholder convention as shards.drop_placeholder_rows (lmsys anonymization tokens that
# leaked into rollouts). Checked once per 64-token window (conservative: a hit drops the start).
PLACEHOLDER_RE = re.compile(r"\b(?:NAME|EMAIL|PHONE|IP_ADDRESS|CREDIT_CARD)_\d+\b")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
        return out or "unknown"
    except OSError:  # pragma: no cover
        return "unknown"


def span_hash(ids: list[int]) -> bytes:
    """Content hash of a token-id span (16-byte blake2b over int32 little-endian bytes)."""
    return hashlib.blake2b(
        torch.tensor(ids, dtype=torch.int32).numpy().tobytes(), digest_size=16
    ).digest()


def load_chat_rollouts(
    rollout_dir: Path, *, glob: str = "rollouts_*.json"
) -> tuple[list[list[int]], list[str]]:
    """All chat rollouts' ``output_ids`` in deterministic (filename, index) order.

    Returns ``(outputs, filenames)``; conv_uid = position in ``outputs``. ``glob`` restricts the
    source files — the DELTA pool (extension data for an already-trained run) is built from
    ``rollouts_gen2_*.json`` only, so its conversations are disjoint from pool v2's by
    construction rather than by later filtering.

    **Completed shards only.** The generator checkpoints in-progress shards to
    ``rollouts_<tag><NNNN>.part.json`` (so preemption costs 256 rollouts, not the whole shard), and
    that name also matches ``rollouts_*.json``. Ingesting a partial would make the pool depend on
    when it happened to be built, and — because a finished shard is a SUPERSET of its own partial —
    a build that caught the brief window between the final write and the partial's unlink would
    contain the same conversations twice. Duplicate source text is exactly what the diversity
    guarantees exist to prevent, so partials are excluded here by construction.
    """
    files = sorted(p for p in rollout_dir.glob(glob) if not p.name.endswith(".part.json"))
    if not files:
        raise FileNotFoundError(f"no {glob} under {rollout_dir}")
    outputs: list[list[int]] = []
    for f in files:
        rolls = json.loads(f.read_text())
        outputs.extend([list(r["output_ids"]) for r in rolls])
    return outputs, [f.name for f in files]


def sample_starts(
    outputs: list[list[int]],
    *,
    n_starts: int,
    seed: int,
    max_per_conv: int = 4,
    window: int = WINDOW,
    non_overlapping: bool = True,
) -> tuple[Int[Tensor, " m"], Int[Tensor, " m"]]:
    """Seeded (conv_uid, start) sample: <= ``max_per_conv`` starts per conversation, each leaving
    >= ``window`` tokens of room; globally shuffled then capped at ``n_starts``.

    ``non_overlapping`` (default) tiles each conversation into disjoint ``window``-token blocks
    and samples whole blocks, so **no two windows share a token**. Overlapping starts silently
    duplicate text across examples — the 2026-07-29 overfit (train 1.05 -> 0.35 while val bottomed
    at 32M tokens and rose) came from exactly this kind of hidden repetition.
    """
    gen = torch.Generator().manual_seed(seed)
    convs: list[int] = []
    starts: list[int] = []
    for uid, out in enumerate(outputs):
        room = len(out) - window + 1
        if room <= 0:
            continue
        if non_overlapping:
            n_blocks = len(out) // window
            if n_blocks == 0:
                continue
            k = min(max_per_conv, n_blocks)
            picks = torch.randperm(n_blocks, generator=gen)[:k] * window
        else:
            k = min(max_per_conv, room)
            picks = torch.randperm(room, generator=gen)[:k]
        convs.extend([uid] * k)
        starts.extend(picks.tolist())
    order = torch.randperm(len(convs), generator=gen)[:n_starts]
    conv_t = torch.tensor(convs, dtype=torch.long)[order]
    start_t = torch.tensor(starts, dtype=torch.long)[order]
    return conv_t, start_t


def pairs_prefix_hashes(
    span_ids: Int[Tensor, " total"],
    offsets: Int[Tensor, " n1"],
    *,
    lengths: tuple[int, ...] = CROP_LENGTHS,
) -> dict[int, set[bytes]]:
    """Content hashes of every pairs-row prefix at each crop length.

    Every AR training crop (any pool, any rung) is a row prefix, so membership here == "the AR
    may have trained on this exact input". ~6 hashes/row over 4.63M rows — minutes on CPU.
    """
    out: dict[int, set[bytes]] = {n: set() for n in lengths}
    for i in range(len(offsets) - 1):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        row_len = hi - lo
        for n in lengths:
            if row_len >= n:
                out[n].add(span_hash(span_ids[lo : lo + n].tolist()))
    return out


def crop_hashes(pool: "AOPool") -> dict[int, set[bytes]]:
    """Hashes of every kept crop of an existing pool AND every shorter prefix of it, per length.

    Used as an ``exclude`` set when building a DELTA pool: window-level dedup only dedups within
    one pool, and the AR-prefix exclusion covers only pairs rows — neither stops the model
    regenerating v2 boilerplate ("Sure, I'd be happy to help ...") from a different seed, which
    would hand the extension run text its parent already trained on.

    **Prefix closure** is the part that blocks the subtle subset case: an N=2 delta crop equal to
    the first 2 tokens of a v2 N=32 crop has a different injected vector (the AR reads the span),
    but its entire supervised target was already inside a parent target. Measured 2026-07-30
    before closure: 5,993/549,131 delta crops (1.09%, 0.18% of span tokens) — cheap to exclude,
    so excluded. The reverse direction (a long delta crop merely *starting* with a short v2 crop)
    is deliberately NOT excluded: v2 kept ~62k distinct N=2 crops, so closing that direction
    would strike most windows that open with any common bigram.
    """
    lengths = pool.lengths
    out: dict[int, set[bytes]] = {n: set() for n in lengths}
    for i in range(len(pool.ids)):
        for j in range(len(lengths)):
            if bool(pool.keep[i, j]):
                for jj in range(j + 1):  # the crop itself and every shorter prefix of it
                    nn = lengths[jj]
                    out[nn].add(span_hash(pool.ids[i, :nn].tolist()))
    return out


def pool_fingerprint(ids: Int[Tensor, "m w"], keep: Bool[Tensor, "m l"]) -> str:
    """Content fingerprint binding ar_out shards to the exact pool they were computed from."""
    h = hashlib.blake2b(digest_size=16)
    h.update(ids.to(torch.int32).numpy().tobytes())
    h.update(keep.to(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def audit_diversity(
    pool: "AOPool", *, layers_per_crop: int, max_repeat: float = 2.5
) -> dict[str, Any]:
    """Prove the pool has no hidden text repetition. Raises if it does.

    Three ways duplicate text sneaks in, all seen for real:
      1. nested crops of one window (prefixes of the same text),
      2. overlapping windows sampled from one conversation,
      3. layer multiplicity k (the same target with a different injected vector).

    ``repeat_factor`` = target appearances per distinct text = (crops / distinct windows) * k.
    An epoch over a pool with repeat_factor R shows each text R times, so R>1 already risks
    memorization; the default ceiling of 2.5 permits k=2 with one crop per window and nothing
    more. Call at pool build AND at trainer startup — a silent regression here costs a whole run.
    """
    rows, _lens = pool.crop_index()
    n_crops = len(rows)
    distinct_windows = int(torch.unique(rows).numel())
    crops_per_window = n_crops / max(1, distinct_windows)
    repeat = crops_per_window * layers_per_crop

    # token overlap between windows of the same conversation (0 under a non-overlapping tiling)
    order = torch.argsort(pool.conv * 10**7 + pool.start)
    conv_s, start_s = pool.conv[order], pool.start[order]
    same_conv = conv_s[1:] == conv_s[:-1]
    gaps = (start_s[1:] - start_s[:-1])[same_conv]
    overlapping = int((gaps < pool.window).sum()) if len(gaps) else 0

    report = {
        "n_crops": n_crops,
        "distinct_windows": distinct_windows,
        "crops_per_window": round(crops_per_window, 3),
        "layers_per_crop": layers_per_crop,
        "repeat_factor": round(repeat, 3),
        "overlapping_window_pairs": overlapping,
        "distinct_texts_per_epoch": distinct_windows,
    }
    problems = []
    if repeat > max_repeat:
        problems.append(
            f"each text is a target {repeat:.1f}x per epoch (ceiling {max_repeat}): "
            f"{crops_per_window:.1f} crops/window x k={layers_per_crop}"
        )
    if overlapping:
        problems.append(f"{overlapping} window pairs in a conversation share tokens")
    if problems:
        raise ValueError("pool diversity audit FAILED: " + "; ".join(problems) + f" | {report}")
    return report


@dataclass
class AOPool:
    """The pool artifact: fixed-width windows + per-length keep masks + provenance.

    Crop index space (shared with the ar_out precompute): row-major over ``keep`` —
    ``crop_idx`` enumerates ``(start_i, length_j)`` pairs with ``keep[i, j]``, i-major.
    ``lengths``/``window`` come from the artifact itself (``crop_lengths`` metadata / the ids
    width) — a consumer that assumes the legacy (2..64) list mislabels every crop of a
    long-emission pool.
    """

    ids: Int[Tensor, "m w"]
    conv: Int[Tensor, " m"]
    start: Int[Tensor, " m"]
    keep: Bool[Tensor, "m l"]
    meta: dict[str, Any]

    @property
    def lengths(self) -> tuple[int, ...]:
        ls = tuple(int(n) for n in self.meta.get("crop_lengths") or CROP_LENGTHS)
        if len(ls) != self.keep.shape[1]:
            raise ValueError(
                f"pool metadata lists {len(ls)} crop lengths but keep has "
                f"{self.keep.shape[1]} columns — corrupt or mislabeled artifact"
            )
        return ls

    @property
    def window(self) -> int:
        return int(self.ids.shape[1])

    def crop_index(self) -> tuple[Int[Tensor, " k"], Int[Tensor, " k"]]:
        """``(start_row, length_idx)`` per kept crop, in canonical order."""
        rows, lens = torch.nonzero(self.keep, as_tuple=True)
        return rows, lens

    def crop_ids(self, row: int, length_idx: int) -> Int[Tensor, " n"]:
        return self.ids[row, : self.lengths[length_idx]]

    def n_crops(self) -> int:
        return int(self.keep.sum())

    def save(self, path: Path) -> None:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        save_file(
            {
                "ids": self.ids.to(torch.int32),
                "conv": self.conv.to(torch.int64),
                "start": self.start.to(torch.int64),
                "keep": self.keep,
            },
            str(tmp),
            metadata={"meta": json.dumps(self.meta)},
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "AOPool":
        from safetensors import safe_open
        from safetensors.torch import load_file

        tensors = load_file(str(path))
        with safe_open(str(path), framework="pt") as f:
            meta = json.loads((f.metadata() or {}).get("meta", "{}"))
        return cls(
            ids=tensors["ids"].to(torch.long),
            conv=tensors["conv"],
            start=tensors["start"],
            keep=tensors["keep"].bool(),
            meta=meta,
        )


def build_eval_pool(
    span_ids: Int[Tensor, " total"],
    offsets: Int[Tensor, " n1"],
    *,
    n_rows: int = 4096,
    lengths: tuple[int, ...] = CROP_LENGTHS,
) -> AOPool:
    """Eval pool from the pairs' leading-row carve (never AR-trained; true h stored alongside).

    Row i of the pairs becomes a window (prefix, zero-padded past the row) with ``keep[i, j]``
    for every crop length the row affords. No dedup/exclusion — these ARE pairs prefixes on
    purpose: the same spans the AR was validated on, so AO FVE is comparable to the AR ceiling.
    """
    window = max(lengths)
    m = min(n_rows, len(offsets) - 1)
    ids = torch.zeros(m, window, dtype=torch.long)
    keep = torch.zeros(m, len(lengths), dtype=torch.bool)
    for i in range(m):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        n = min(hi - lo, window)
        ids[i, :n] = span_ids[lo : lo + n].to(torch.long)
        for j, length in enumerate(lengths):
            keep[i, j] = (hi - lo) >= length
    meta = {
        "crop_lengths": list(lengths),
        "kind": "eval_pairs_carve",
        "n_rows": m,
        "git_commit": _git_commit(),
        "stats": {f"n_kept_N{n}": int(keep[:, j].sum()) for j, n in enumerate(lengths)},
    }
    return AOPool(
        ids=ids,
        conv=torch.arange(m),  # = pairs row index (source-of-truth for true-h lookup in eval)
        start=torch.zeros(m, dtype=torch.long),
        keep=keep,
        meta=meta,
    )


def build_ao_pool(
    outputs: list[list[int]],
    conv: Int[Tensor, " m"],
    start: Int[Tensor, " m"],
    *,
    exclude: dict[int, set[bytes]],
    special_ids: frozenset[int],
    tokenizer: Any = None,
    seed: int = 0,
    all_lengths: bool = False,
    lengths: tuple[int, ...] = CROP_LENGTHS,
    provenance: dict[str, Any] | None = None,
) -> AOPool:
    """Assemble the pool: windows, per-length dedup, AR-prefix exclusion, hygiene filters.

    Filters, in order per start/crop (each counted in the provenance):
    - window contains a special token (chat markers the model emitted) -> drop the START;
    - window text matches a placeholder pattern -> drop the START (checked once per window,
      conservative — cheaper than per-crop decode and drops few starts);
    - within-length exact dup -> drop the crop;
    - content equals a pairs-row prefix at that length -> drop the crop (AR-disjointness).
    """
    window = max(lengths)
    m = len(conv)
    ids = torch.zeros(m, window, dtype=torch.long)
    keep = torch.zeros(m, len(lengths), dtype=torch.bool)
    stats = {
        "n_starts_sampled": m,
        "n_starts_special": 0,
        "n_starts_placeholder": 0,
        "n_starts_dup_window": 0,
        **{f"n_dup_N{n}": 0 for n in lengths},
        **{f"n_excluded_N{n}": 0 for n in lengths},
        **{f"n_kept_N{n}": 0 for n in lengths},
    }
    seen: dict[int, set[bytes]] = {n: set() for n in lengths}
    seen_windows: set[bytes] = set()  # dedup the 64-token TEXT, not just the emitted span
    gen = torch.Generator().manual_seed(seed * 7919 + 3)
    # ONE length per window (default): the six nested crops of a window are prefixes of one text,
    # so keeping them all made each text a target ~4.8x (x k layers = ~24x/epoch) and drove the
    # 2026-07-29 overfit. Sampling one length per window keeps full length coverage across the
    # pool while every text appears exactly once.
    length_choice = torch.randint(len(lengths), (m,), generator=gen)
    for i in range(m):
        win = outputs[int(conv[i])][int(start[i]) : int(start[i]) + window]
        if any(t in special_ids for t in win):
            stats["n_starts_special"] += 1
            continue
        if tokenizer is not None and PLACEHOLDER_RE.search(tokenizer.decode(win)):
            stats["n_starts_placeholder"] += 1
            continue
        # Window-level dedup: two conversations can emit identical boilerplate, so the same text
        # would appear twice (at different sampled lengths, one a prefix of the other). Span-level
        # dedup misses this — it only sees the emitted crop. 5.6% of pool_v2 rows were such dups.
        wh = span_hash(win)
        if wh in seen_windows:
            stats["n_starts_dup_window"] += 1
            continue
        seen_windows.add(wh)
        ids[i] = torch.tensor(win, dtype=torch.long)
        js = range(len(lengths)) if all_lengths else [int(length_choice[i])]
        for j in js:
            n = lengths[j]
            h = span_hash(win[:n])
            if h in seen[n]:
                stats[f"n_dup_N{n}"] += 1
                continue
            seen[n].add(h)
            if h in exclude.get(n, set()):
                stats[f"n_excluded_N{n}"] += 1
                continue
            keep[i, j] = True
            stats[f"n_kept_N{n}"] += 1
    kept_any = keep.any(dim=1)
    meta = {
        "crop_lengths": list(lengths),
        "seed": seed,
        "one_length_per_window": not all_lengths,
        "git_commit": _git_commit(),
        "stats": stats,
        **(provenance or {}),
    }
    return AOPool(
        ids=ids[kept_any].to(torch.int32).to(torch.long),
        conv=conv[kept_any],
        start=start[kept_any],
        keep=keep[kept_any],
        meta=meta,
    )
