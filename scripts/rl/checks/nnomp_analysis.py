"""How much is extractable, and is the ceiling the POLICY or the TARGET/AR?

On the balanced 132-item eval set, per LAYER (bullet images are layer-specific):
  pooled@N   : nnomp of the gold over the item's OWN bullets pooled from N rollouts,
               for N in {1,2,4,8,16,32} -> the best-across-N curve.
  AO-dict    : nnomp over a GLOBAL dictionary of ALL items' AO bullets (this layer).
  GT-dict    : nnomp over a GLOBAL dictionary of the real source-text sentences
               (non-AO text) at this layer.
Reads ladder_RL.json for rollout texts, rl_gate_0 (balanced) for golds+source_text.

  python nnomp_analysis.py <run_dir> <ladder.json> <device> [--cap 600] [--out x.json]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from data import load_ao_rl_dataset  # noqa: E402
from eval_ceiling import _ar_unit_images, _load_args  # noqa: E402
from sidecar import load_sidecar  # noqa: E402


def sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t or "") if len(s.strip()) > 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("ladder"); ap.add_argument("device")
    ap.add_argument("--per-layer", type=int, default=12)
    ap.add_argument("--cap", type=int, default=600, help="max dict atoms per layer")
    ap.add_argument("--exclude-own", action="store_true",
                    help="AO-dict: reconstruct each item from ONLY other questions' bullets")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from oracle_lens.core.nnomp import nnomp_batch
    from oracle_lens.pipeline.ar_loader import load_ladder_whiteners, load_lc_reconstructor
    from oracle_lens.pipeline.rl_reward import RewardSpace, split_bullets
    rargs = _load_args(a.run_dir, a.device)
    tok = AutoTokenizer.from_pretrained(rargs.base_ckpt)
    scfg = load_sidecar(rargs.sidecar, tok)
    rows = load_ao_rl_dataset(rargs.eval_parquet, inj_id=scfg.injection_token_id, n_max=None)
    by = defaultdict(list)
    for r in rows:
        by[int(r["layer"])].append(r)
    eval_rows = [r for ly in sorted(by) for r in by[ly][:a.per_layer]]
    recs = json.loads(Path(a.ladder).read_text())["records"]
    assert len(recs) == len(eval_rows), f"{len(recs)} vs {len(eval_rows)}"
    dev = a.device
    recon = load_lc_reconstructor(Path(rargs.ar_ckpt), device=dev, eager=True)
    W = load_ladder_whiteners(Path(rargs.whitener_dir), prefix=rargs.whitener_prefix,
                              ridge_c=rargs.ridge, layers=tuple(recon.layers))
    space = RewardSpace(whiten=rargs.reward_whiten, unit_norm=rargs.reward_unit_norm)
    rng = np.random.RandomState(0)

    Ns = [1, 2, 4, 8, 16, 32]
    pooledN = {n: [] for n in Ns}; aodict = []; gtdict = []
    idx_by_layer = defaultdict(list)
    for i, r in enumerate(eval_rows):
        idx_by_layer[int(r["layer"])].append(i)

    def nn(gw, img, k):
        return float(nnomp_batch(gw.unsqueeze(0), img.to(dev), max_atoms=k).fve[0].cpu()) if img.shape[0] else 0.0

    winners = []   # per item: the AO-dict-selected bullets (+ own/foreign provenance) for Opus vetting
    for ly, idxs in sorted(idx_by_layer.items()):
        w = W[ly]
        own = []
        prov = {}        # bullet_key -> set of jj (which layer-local items produced it)
        prov_text = {}   # bullet_key -> original-case text (first occurrence)
        for jj, i in enumerate(idxs):
            perroll = [split_bullets(rr["text"], 99) or [rr["text"]] for rr in recs[i]["rollouts"]]
            own.append(perroll)
            for r in perroll:
                for b in r:
                    key = b.strip().lower()
                    prov.setdefault(key, set()).add(jj)
                    prov_text.setdefault(key, b.strip())
        ao_u = [prov_text[k] for k in prov]  # original-case dedup'd bullets
        if len(ao_u) > a.cap:
            ao_u = [ao_u[j] for j in sorted(rng.choice(len(ao_u), a.cap, replace=False))]
        gt_txt = []
        for i in idxs:
            gt_txt.extend(sents(recs[i].get("source_text") or eval_rows[i].get("source_text", "")))
        gt_u = list(dict.fromkeys(gt_txt))[: a.cap]
        ao_img, ao_kept = _ar_unit_images(recon, tok, ao_u, ly, w, space, dev, mb=rargs.rm_micro_batch)
        ao_texts = [ao_u[k] for k in ao_kept]   # rows of ao_img correspond to these
        atom_jj = [prov.get(t.strip().lower(), set()) for t in ao_texts]  # which items produced each atom
        if gt_u:
            gt_img, _ = _ar_unit_images(recon, tok, gt_u, ly, w, space, dev, mb=rargs.rm_micro_batch)
        else:
            gt_img = torch.zeros(0, ao_img.shape[1])
        for jj, i in enumerate(idxs):
            g = torch.from_numpy(eval_rows[i]["gold"]).float()
            gw = (w.whiten(g.unsqueeze(0))[0] if space.whiten else (g - w.mu)).float().to(dev)
            own_p = 0.0
            for n in Ns:
                rolls = own[jj]
                sub = rolls if len(rolls) <= n else [rolls[k] for k in sorted(rng.choice(len(rolls), n, replace=False))]
                bl = [b for r in sub for b in r]
                img, _ = _ar_unit_images(recon, tok, bl, ly, w, space, dev, mb=rargs.rm_micro_batch)
                fv = nn(gw, img, 4); pooledN[n].append(fv); own_p = fv  # n=32 is last -> own_p
            # AO-dict with atom capture (which bullets were selected, own vs foreign)
            if ao_img.shape[0]:
                mask = None
                if a.exclude_own:  # foreign-only: forbid atoms this item produced
                    mask = torch.tensor([[jj not in atom_jj[k] for k in range(len(ao_texts))]],
                                        dtype=torch.bool, device=dev)
                res_ao = nnomp_batch(gw.unsqueeze(0), ao_img.to(dev), max_atoms=8, mask=mask)
                ao_fve = float(res_ao.fve[0].cpu()); aodict.append(ao_fve)
                picks = []
                for at, co in zip(res_ao.atoms[0].tolist(), res_ao.coeffs[0].tolist()):
                    if at >= 0 and abs(co) > 1e-6:
                        txt = ao_texts[at]
                        picks.append({"text": txt[:200], "coeff": round(float(co), 3),
                                      "own": jj in prov.get(txt.strip().lower(), set())})
                winners.append({"row_id": eval_rows[i]["row_id"], "layer": ly,
                                "target_source": (recs[i].get("source_text") or "")[:300],
                                "own_pooled32": round(own_p, 4), "ao_dict_fve": round(ao_fve, 4),
                                "ao_dict_bullets": picks,
                                "n_foreign": sum(1 for p in picks if not p["own"])})
            else:
                aodict.append(0.0)
            gtdict.append(nn(gw, gt_img, 8))
        print(f"[L{ly}] ao_dict={len(ao_u)} gt_dict={len(gt_u)} done", flush=True)

    res = {"pooledN": {n: float(np.mean(pooledN[n])) for n in Ns},
           "ao_dict": float(np.mean(aodict)), "gt_dict": float(np.mean(gtdict)),
           "n_items": len(eval_rows), "winners": winners}
    print("\n=== nnomp analysis ===")
    for n in Ns:
        print(f"  pooled@{n:>2}: {res['pooledN'][n]*100:.1f}%")
    print(f"  AO global-dict (best across questions): {res['ao_dict']*100:.1f}%")
    print(f"  GT global-dict (real text):             {res['gt_dict']*100:.1f}%")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2)); print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
