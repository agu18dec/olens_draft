"""G9 — reconcile a cell's recorded training examples against the pairs that actually exist.

``StreamingPairDataset`` promises that "every row in the stream is a first-time sample". That was
asserted in a docstring and never measured, and on 2026-08-02 it was false twice over: the
producer re-captured slices already on disk (byte-identical pairs), and non-zero DataLoader
workers re-read shards the rank marker had not yet been stamped for. Between them, ~28% of the
assistant cell's recorded samples and ~21% of the pt cell's were repeats, which silently inflates
the x-axis of the AR scaling curve.

The check is arithmetic and cheap, so it should run routinely rather than after a mystery:

    recorded examples  >  unique pairs ever captured   =>  repetition, no other explanation

``recorded`` comes from the trainer's all-reduced counter (``ml_checkpoints/<run>/*/meta.json``,
or the live ``resume.pt``); ``unique`` comes from the capture reports, keyed on the
``(rollout_shard, conv-hash interval)`` that actually determines a pair's content — NOT on the
shard number, which counts a legitimately disjoint later wave as a duplicate.

    uv run python scripts/datagen/iolens_reconcile_counts.py
    uv run python scripts/datagen/iolens_reconcile_counts.py --cells chat --strict
"""

import argparse
import contextlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DONE_RE = re.compile(r"done_(\d+)(?:s(\d+))?\.json$")


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def capture_totals(pairs_dir: Path) -> dict[str, Any]:
    """Unique vs duplicate train pairs, keyed on (rollout_shard, wave interval start).

    The wave tag encodes ``train_frac_skip * 1000``; an untagged report is the pre-wave-tag
    naming, i.e. skip 0.0. Two reports sharing a key captured the SAME conversations with the
    same span RNG, so their pairs are byte-identical.
    """
    by: dict[tuple[int, float], list[tuple[str, int]]] = defaultdict(list)
    for report in sorted(pairs_dir.glob("done_*.json")):
        m = DONE_RE.search(report.name)
        if not m:
            continue
        meta = json.loads(report.read_text())
        skip = round(int(m.group(2) or 0) / 1000, 4)
        by[(int(m.group(1)), skip)].append((report.name, int(meta["pairs"]["train"])))
    unique = sum(v[0][1] for v in by.values())
    duplicate = sum(n for v in by.values() for _, n in v[1:])
    dup_keys = {k: [n for _, n in v] for k, v in by.items() if len(v) > 1}
    return {"unique": unique, "duplicate": duplicate, "slices": len(by), "dup_keys": dup_keys}


def recorded_examples(ckpt_dir: Path) -> tuple[int, str]:
    """Highest recorded example count for a run, and where it came from."""
    best, src = 0, "none"
    for meta_path in ckpt_dir.glob("*/meta.json"):
        try:
            n = int(json.loads(meta_path.read_text()).get("examples", 0))
        except (json.JSONDecodeError, OSError):
            continue
        if n > best:
            best, src = n, meta_path.parent.name
    return best, src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="chat,pt")
    ap.add_argument("--run-fmt", default="ar.{cell}.mlayer.lc.s0")
    ap.add_argument("--pairs-fmt", default="ml_pairs_iolens_{cell}")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any cell shows repetition (for use as a gate)")
    args = ap.parse_args()

    root = ola_root()
    bad = []
    for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
        pairs_dir = root / args.pairs_fmt.format(cell=cell)
        ckpt_dir = root / "ml_checkpoints" / args.run_fmt.format(cell=cell)
        if not pairs_dir.exists():
            print(f"[G9] {cell}: no pairs dir at {pairs_dir} — skipping", flush=True)
            continue
        cap = capture_totals(pairs_dir)
        rec, src = recorded_examples(ckpt_dir)
        # recorded > unique is proof of repetition: the trainer cannot feed rows that were never
        # captured. It is a LOWER bound, since unconsumed pairs sitting in the buffer mean fewer
        # unique rows were actually consumed than were captured.
        #
        # The bound LOOSENS as the producer captures more, so a single reading understates past
        # repetition -- it binds hardest when consumption has caught up with capture. Persist the
        # high-water mark so the strongest bound ever observed survives; that is the number that
        # says something about the run's history, not the instantaneous one.
        repeats = rec - cap["unique"]
        hw_path = ckpt_dir / "g9_highwater.json"
        hw: dict[str, Any] = {"repeats": 0, "recorded": 0, "unique": 0}
        if hw_path.exists():
            with contextlib.suppress(json.JSONDecodeError):
                hw = json.loads(hw_path.read_text())
        if repeats > int(hw.get("repeats", 0)):
            hw = {"repeats": repeats, "recorded": rec, "unique": cap["unique"], "at": src}
            if ckpt_dir.exists():
                tmp = hw_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(hw, indent=2))
                tmp.replace(hw_path)
        pct = (100.0 * repeats / rec) if rec else 0.0
        print(f"=== {cell}")
        print(f"  capture slices          : {cap['slices']}")
        print(f"  UNIQUE pairs captured   : {cap['unique']:>12,}")
        print(f"  duplicate pairs written : {cap['duplicate']:>12,}")
        print(f"  recorded examples       : {rec:>12,}  (from {src})")
        for (shard, skip), counts in sorted(cap["dup_keys"].items()):
            print(f"    ! shard {shard} skip {skip}: captured {len(counts)}x {counts}")
        if repeats > 0:
            print(f"  REPEATS now (>= bound)  : {repeats:>12,}  ({pct:.0f}% of recorded)")
            bad.append(cell)
        else:
            print(f"  headroom                : {-repeats:>12,} unique pairs not yet consumed")
            print("  OK — no repetition provable from the CURRENT counts (a weak test while the")
            print("       producer is ahead of the trainer; see the high-water mark below)")
        hw_rep = int(hw.get("repeats", 0))
        if hw_rep > 0:
            hw_pct = 100.0 * hw_rep / max(1, int(hw.get("recorded", 1)))
            print(f"  HIGH-WATER repeats      : {hw_rep:>12,}  ({hw_pct:.0f}% of recorded at the")
            print(f"       time, {hw.get('recorded', 0):,} recorded vs {hw.get('unique', 0):,} "
                  f"unique) — this is the number the curve's x-axis is inflated by")
    if bad and args.strict:
        raise SystemExit(f"[G9] repetition detected in: {', '.join(bad)}")


if __name__ == "__main__":
    main()
