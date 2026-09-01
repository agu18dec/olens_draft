"""Lever-1 data: build a reward-variance curriculum from the RAFT harvest (CPU-only).

The 88%-degenerate-groups plateau is driven by activations where every rollout scores
alike (var~0 -> zero GRPO advantage). Here we read the harvest's 16 per-rollout FVEs
per activation, compute the per-activation reward STD, and keep only the LEARNABLE
(high-variance) rows -> rl_train_curriculum.parquet. GRPO on this should have non-zero
advantage on (nearly) every group.

  python build_curriculum.py --harvest-glob 'raft2_harvest/harvest_*.json' \
      --train rl_train_0.parquet --out rl_train_curriculum.parquet [--keep-frac 0.5]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-glob", required=True)
    ap.add_argument("--train", default="artifacts/sc/rl_distill/rl_train_0.parquet")
    ap.add_argument("--out", default="artifacts/sc/rl_distill/rl_train_curriculum.parquet")
    ap.add_argument("--keep-frac", type=float, default=0.5, help="top-frac by reward std")
    ap.add_argument("--min-std", type=float, default=0.0, help="also require std >= this")
    a = ap.parse_args()

    stats = {}  # row_id -> (std, mean)
    for f in glob.glob(a.harvest_glob):
        for rec in json.loads(Path(f).read_text()).get("records", []):
            fv = np.array([r["fve"] for r in rec["rollouts"]], dtype=float)
            if len(fv):
                stats[int(rec["row_id"])] = (float(fv.std()), float(fv.mean()))
    if not stats:
        raise SystemExit("no harvest records found")
    stds = np.array([v[0] for v in stats.values()])
    print(f"[curriculum] {len(stats)} harvested activations")
    for p in (10, 25, 50, 75, 90):
        print(f"   reward-std p{p}: {np.percentile(stds, p):.4f}")
    thr = max(a.min_std, float(np.quantile(stds, 1 - a.keep_frac)))
    keep = {rid for rid, (s, m) in stats.items() if s >= thr}
    dead = sum(1 for s in stds if s < 0.01)
    print(f"[curriculum] std threshold {thr:.4f} -> keep {len(keep)} learnable "
          f"(dropping {len(stats)-len(keep)}; {dead} were ~dead var<0.01)")

    t = pq.read_table(a.train)
    rid = t.column("row_id").to_pylist()
    mask = [r in keep for r in rid]
    out = t.filter(pa.array(mask))
    pq.write_table(out, a.out)
    print(f"[curriculum] wrote {out.num_rows} rows -> {a.out} "
          f"(from {t.num_rows} train rows; harvest covered {len(stats)})")


if __name__ == "__main__":
    main()
