"""Surface bullets that are semantic REPHRASINGS yet have near-ORTHOGONAL AR-image
directions (low cosine) — the mechanism behind SFT's high det(Gram) despite low faithful
diversity. Prints bullet texts + their pairwise cosine so we can eyeball rephrasing pairs
with low cos.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="artifacts/sc/rl_runs/iolens-rl-final-ddp600")
    p.add_argument("--ladder", default="ladder_SFT.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cos-thresh", type=float, default=0.3,
                   help="show rollouts with a bullet pair below this cos")
    p.add_argument("--n-show", type=int, default=18)
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
    ra = yaml.safe_load((rd / "run_config.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(ra["base_ckpt"])
    recon = load_lc_reconstructor(Path(ra["ar_ckpt"]), device=args.device, eager=True)
    whit = load_ladder_whiteners(Path(ra["whitener_dir"]), prefix=ra["whitener_prefix"],
                                 ridge_c=ra["ridge"], layers=tuple(recon.layers))
    space = RewardSpace(whiten=ra.get("reward_whiten", True),
                        unit_norm=ra.get("reward_unit_norm", True))
    pos = {int(ly): i for i, ly in enumerate(recon.layers)}
    dev = args.device

    def dirs(texts, layer):
        w = whit[layer]
        out = []
        for t in texts:
            ids = reward_text_ids(tok, t)
            if not ids:
                out.append(None)
                continue
            b = ml_collate([{"ids": torch.tensor(ids), "target": torch.zeros(1)}], pad_id=0)
            cap = capture_block_outputs(recon.backbone, [layer],
                                        b["input_ids"].to(dev), b["attention_mask"].to(dev))
            last = int(b["attention_mask"].sum(1) - 1)
            h = cap[layer][0, last].float()
            pv = recon.head(h + recon.layer_emb.weight[pos[layer]]).float().cpu()
            pv = w.whiten(pv.unsqueeze(0))[0] if space.whiten else (pv - w.mu)
            out.append(pv / (pv.norm() + 1e-9))
        return out

    with open(rd / args.ladder) as fh:
        recs = json.load(fh)["records"]
    shown = 0
    with torch.no_grad():
        for r in recs:
            if shown >= args.n_show:
                break
            layer = int(r["layer"])
            for d in r["rollouts"][:6]:
                bs = split_bullets(d["text"], 4)
                if len(bs) < 2:
                    continue
                ds = dirs(bs, layer)
                idx = [i for i in range(len(bs)) if ds[i] is not None]
                if len(idx) < 2:
                    continue
                # find the lowest-cosine pair
                best = None
                for a in range(len(idx)):
                    for b2 in range(a + 1, len(idx)):
                        c = float(ds[idx[a]] @ ds[idx[b2]])
                        if best is None or c < best[0]:
                            best = (c, idx[a], idx[b2])
                if best and best[0] < args.cos_thresh:
                    print("=" * 80)
                    print(f"L{layer} row {r.get('row_id')} | "
                          f"lowest bullet-pair cos = {best[0]:.2f}")
                    print(f"  src: {(r.get('source_text') or '')[:80]!r}")
                    print(f"  [A] {bs[best[1]][:130]!r}")
                    print(f"  [B] {bs[best[2]][:130]!r}")
                    shown += 1
                    break
    print(f"\nshown {shown} rollouts with a bullet pair below cos {args.cos_thresh}")


if __name__ == "__main__":
    main()
