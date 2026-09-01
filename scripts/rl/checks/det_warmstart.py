"""Measure the FUNCTIONAL bullet diversity of a checkpoint's rollouts: det(Gram) of the
per-bullet AR-image directions (the exact quantity the --diversity-lambda reward uses).

det=0 -> bullets are parallel/redundant (rephrasings in image space); det=1 -> orthogonal
(genuinely different reconstructive directions). Answers whether the SFT warmstart (or RL)
has real functional bullet-diversity for a diversity-reward RL to amplify — the non-lexical
version of the pre-check. Runs the AR, so needs a free GPU.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="artifacts/sc/rl_runs/iolens-rl-final-ddp600")
    p.add_argument("--ladders", nargs="+", default=["ladder_SFT.json", "ladder_RL.json"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--max-rollouts", type=int, default=8, help="rollouts/item to score (speed)")
    args = p.parse_args()

    import yaml
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.ar_loader import load_ladder_whiteners, load_lc_reconstructor
    from oracle_lens.pipeline.multilayer_reconstructor import ml_collate
    from oracle_lens.pipeline.rl_reward import (
        RewardSpace,
        capture_block_outputs,
        reward_text_ids,
        split_bullets,
    )

    rd = Path(args.run_dir)
    rargs = yaml.safe_load((rd / "run_config.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(rargs["base_ckpt"])
    recon = load_lc_reconstructor(Path(rargs["ar_ckpt"]), device=args.device, eager=True)
    whit = load_ladder_whiteners(Path(rargs["whitener_dir"]), prefix=rargs["whitener_prefix"],
                                 ridge_c=rargs["ridge"], layers=tuple(recon.layers))
    space = RewardSpace(whiten=rargs.get("reward_whiten", True),
                        unit_norm=rargs.get("reward_unit_norm", True))
    pos = {int(ly): i for i, ly in enumerate(recon.layers)}
    dev = args.device

    def bullet_dirs(texts, layer):
        w = whit[layer]
        out = []
        for t in texts:
            ids = reward_text_ids(tok, t)
            if not ids:
                continue
            b = ml_collate([{"ids": torch.tensor(ids), "target": torch.zeros(1)}], pad_id=0)
            cap = capture_block_outputs(recon.backbone, [layer], b["input_ids"].to(dev),
                                        b["attention_mask"].to(dev))
            last = int(b["attention_mask"].sum(1) - 1)
            h = cap[layer][0, last].float()
            pv = recon.head(h + recon.layer_emb.weight[pos[layer]]).float().cpu()
            pv = w.whiten(pv.unsqueeze(0))[0] if space.whiten else (pv - w.mu)
            pv = pv / (pv.norm() + 1e-9)
            out.append(pv)
        return torch.stack(out) if out else torch.zeros(0)

    for lad in args.ladders:
        with open(rd / lad) as fh:
            recs = json.load(fh)["records"]
        dets, ks = [], []
        with torch.no_grad():
            for r in recs:
                layer = int(r["layer"])
                for d in r["rollouts"][:args.max_rollouts]:
                    bs = split_bullets(d["text"], args.k_max)
                    if len(bs) < 2:
                        dets.append(1.0 if bs else 0.0)
                        ks.append(len(bs))
                        continue
                    dirs = bullet_dirs(bs, layer)
                    if dirs.shape[0] < 2:
                        dets.append(1.0)
                        ks.append(dirs.shape[0])
                        continue
                    gram = dirs @ dirs.T
                    dets.append(float(torch.det(gram
                                                + 1e-6 * torch.eye(dirs.shape[0])).clamp_min(0.0)))
                    ks.append(dirs.shape[0])
        n = len(dets)
        print(f"\n=== {lad}: det(Gram) of bullet AR-image directions ({n} rollouts) ===")
        print(f"  mean {st.mean(dets):.3f}  median {st.median(dets):.3f}  "
              f"p90 {sorted(dets)[int(.9*n)]:.3f}  max {max(dets):.3f}")
        print(f"  frac det>0.3 (spread bullets): {sum(1 for x in dets if x>0.3)/n*100:.0f}%"
              f"   det>0.5: {sum(1 for x in dets if x>0.5)/n*100:.0f}%")
        print(f"  mean bullets/rollout: {st.mean(ks):.2f}")


if __name__ == "__main__":
    main()
