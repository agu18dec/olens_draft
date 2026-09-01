"""iolens S3: verify + report the 4-way split over the prepared seeds (assigned in seed prep).

The split is a pure function of seed content (``rollout_store.split_of_key``), stamped on every
seed row BEFORE generation. This script re-derives it, verifies the stamps, checks conversation
cohesion (every turn of a WildChat conversation on the same side), and writes the canonical
``$OLA_ROOT/splits_iolens.json`` report both cells' fleets and gates reconcile against.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def check_cell(seeds_dir: Path, mode: str) -> dict[str, Any]:
    from oracle_lens.pipeline.rollout_store import SPLITS, split_of_key

    counts = [0, 0, 0, 0]
    n = 0
    stamp_mismatch = 0
    conv_sides: dict[str, set[int]] = defaultdict(set)
    first_turn_split: dict[str, int] = {}
    for p in sorted(seeds_dir.glob("seeds_*.json")):
        for r in json.loads(p.read_text()):
            n += 1
            counts[r["split"]] += 1
            if mode == "chat":
                conv = r["key"].rsplit(":", 1)[0]
                turn = int(r["key"].rsplit(":", 1)[1])
                conv_sides[conv].add(int(r["split"]))
                if turn == 0:
                    # only turn 0 is re-derivable from row content (the conv is keyed on it)
                    if split_of_key(r["text"]) != r["split"]:
                        stamp_mismatch += 1
                    first_turn_split[conv] = int(r["split"])
            else:
                if split_of_key(r["key"].split(":", 1)[1]) != r["split"]:
                    stamp_mismatch += 1
    incoherent = sum(1 for sides in conv_sides.values() if len(sides) > 1)
    report: dict[str, Any] = {
        "mode": mode,
        "n_seeds": n,
        "splits": dict(zip(SPLITS, counts, strict=True)),
        "fractions": {k: round(c / max(1, n), 4) for k, c in zip(SPLITS, counts, strict=True)},
        "stamp_mismatches": stamp_mismatch,
        "pass": stamp_mismatch == 0,
    }
    if mode == "chat":
        report["n_conversations"] = len(conv_sides)
        report["conversations_split_across_sides"] = incoherent
        report["pass"] = bool(report["pass"]) and incoherent == 0
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chat-dir", default="seeds_iolens_chat")
    ap.add_argument("--pt-dir", default="seeds_iolens_pt")
    ap.add_argument("--out", default="splits_iolens.json")
    args = ap.parse_args()
    root = ola_root()
    report: dict[str, Any] = {}
    for mode, d in (("chat", args.chat_dir), ("pt", args.pt_dir)):
        p = root / d
        if p.exists():
            report[mode] = check_cell(p, mode)
        else:
            report[mode] = {"mode": mode, "skipped": f"{p} missing"}
    ok = all(bool(r.get("pass", True)) for r in report.values())
    report["pass"] = ok
    (root / args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit("[iolens-splits] FAILED — see report above")


if __name__ == "__main__":
    main()
