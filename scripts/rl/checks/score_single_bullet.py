"""Single-bullet FVE: re-score each saved rollout's FIRST bullet only (k=1) through
the AR — no regeneration. Gives the 'single bullet' readout number to sit beside the
4-bullet (best@1) numbers. Runs on the SAME balanced eval rows the ladder used.

Usage: python score_single_bullet.py <run_dir> <ladder.json> <per_layer> <reward_device>
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
    run_dir, ladder_json, per_layer, dev = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    cfg = yaml.safe_load((Path(run_dir) / "run_config.yaml").read_text())
    cfg["reward_device"] = dev
    cfg["reward_agg"] = "single"          # single-readout scorer path
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

    from oracle_lens.pipeline.rl_reward import split_bullets
    score_fn, _ = build_reward_fn(rargs, tok)

    data = json.loads(Path(ladder_json).read_text())
    recs = data["records"]
    assert len(recs) == len(eval_rows), f"{len(recs)} recs vs {len(eval_rows)} rows"

    golds = torch.from_numpy(np.stack([r["gold"] for r in eval_rows])).float()
    layers = torch.tensor([r["layer"] for r in eval_rows], dtype=torch.long)

    per_item, per_layer_acc = [], {}
    for i, rec in enumerate(recs):
        firsts = []
        for roll in rec["rollouts"]:
            b = split_bullets(roll["text"], k_max=1)
            firsts.append(b[0] if b else roll["text"])
        n = len(firsts)
        res = score_fn(firsts, golds[i:i+1].repeat(n, 1), layers[i:i+1].repeat(n),
                       [[] for _ in firsts])
        fv = res.fve.float().numpy()
        m = float(fv.mean())          # mean single-bullet FVE over this item's rollouts
        per_item.append(m)
        per_layer_acc.setdefault(int(rec["layer"]), []).append(m)

    arr = np.array(per_item)
    print(f"\n=== single-bullet (first bullet only, k=1) — {Path(ladder_json).name} ===")
    print(f"  mean {arr.mean():.4f}  median {np.median(arr):.4f}  (n={len(arr)})")
    for ly in sorted(per_layer_acc):
        v = per_layer_acc[ly]
        print(f"    L{ly:>2}: {np.mean(v):.3f}")
    out = Path(ladder_json).with_name(Path(ladder_json).stem + "_single1.json")
    out.write_text(json.dumps({"single_bullet_mean": float(arr.mean()),
                               "per_item": per_item,
                               "layer": [int(r["layer"]) for r in recs]}, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
