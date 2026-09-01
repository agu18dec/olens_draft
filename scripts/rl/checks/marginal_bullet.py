"""How much does each bullet add? Cumulative joint-FVE over the oracle-picked
bullets (in nnomp selection order) for k=1..4, on the balanced 12x11 eval set.
Marginal(k) = FVE(first k) - FVE(first k-1). Reuses saved picks (no regeneration
of rollouts; only an AR scoring pass on the <=4 picked bullets per item).

Usage: python marginal_bullet.py <run_dir> <per_layer> <reward_device> <ladder.json> [tag]
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from data import load_ao_rl_dataset  # noqa: E402
from sidecar import load_sidecar  # noqa: E402


def main():
    run_dir, per_layer, dev, ladder = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    tag = sys.argv[5] if len(sys.argv) > 5 else Path(ladder).stem
    cfg = yaml.safe_load((Path(run_dir) / "run_config.yaml").read_text())
    cfg["reward_device"] = dev; cfg["reward_agg"] = "joint"
    rargs = SimpleNamespace(**cfg)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rargs.base_ckpt)
    scfg = load_sidecar(rargs.sidecar, tok)
    rows = load_ao_rl_dataset(rargs.eval_parquet, inj_id=scfg.injection_token_id, n_max=None)
    by = {}
    for r in rows:
        by.setdefault(int(r["layer"]), []).append(r)
    eval_rows = [r for ly in sorted(by) for r in by[ly][:per_layer]]

    from train_rl_ao import build_reward_fn
    score_fn, _ = build_reward_fn(rargs, tok)
    recs = json.loads(Path(ladder).read_text())["records"]
    assert len(recs) == len(eval_rows)
    golds = torch.from_numpy(np.stack([r["gold"] for r in eval_rows])).float()
    layers = torch.tensor([r["layer"] for r in eval_rows], dtype=torch.long)

    # cumulative FVE after k picked bullets, per item
    cum = {1: [], 2: [], 3: [], 4: []}
    for i, rec in enumerate(recs):
        picks = [b["text"].strip().lstrip("- ").strip() for b in rec.get("oracle_bullets", [])]
        picks = [p for p in picks if p][:4]
        if not picks:
            for k in cum:
                cum[k].append(0.0)
            continue
        for k in (1, 2, 3, 4):
            sub = picks[:k] if k <= len(picks) else picks
            text = "\n".join(f"- {p}" for p in sub)
            res = score_fn([text], golds[i:i+1], layers[i:i+1], [[]])
            cum[k].append(float(res.fve[0]))

    print(f"\n=== marginal FVE per bullet — {tag} (n={len(recs)}) ===")
    prev = 0.0
    for k in (1, 2, 3, 4):
        m = float(np.mean(cum[k]))
        print(f"  after {k} bullet(s): {m*100:.1f}%   (bullet {k} added +{(m-prev)*100:.1f} pts)")
        prev = m
    Path(ladder).with_name(f"marginal_{tag}.json").write_text(
        json.dumps({k: cum[k] for k in cum}, indent=2))


if __name__ == "__main__":
    main()
