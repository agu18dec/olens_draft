"""Gate 5 — matched-vs-shuffled probe through the NEW rollout path (port of check_10).

Greedy rollouts on gate rows via the trainer's hook path (actor cuda:0), scored
by score_texts on the frozen AR (cuda:1) against matched golds vs a
within-layer derangement. The AO must beat shuffled golds decisively:
gap >= --min-gap (0.15), paired win rate >= --min-win (0.65).
Known-good (old stack, same data/AR): matched -1.4994, shuffled -2.0009,
gap +0.502, win 0.917; mean matched FVE ≈ 0.102 (the SFT baseline) — reported
here as the tie to the known-good numbers.

Run: uv run --no-sync python scripts/rl/checks/gate_probe.py \
       --parquet $SC/rl/rl_gate_0.parquet --sidecar $SC/rl/merged/nla_meta.yaml \
       --ao-lora $SC/hf/ckpts/ao/chat/k4.L20plus.s2/step3002/lora_hf \
       --ar-ckpt $SC/hf/ckpts/ar/chat/mlayer.lc.s0/ex16014240 \
       --whitener-dir $SC/whiteners/chat
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from _sc_lib import hook_greedy, load_actor, load_gate_rows, verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--ao-lora", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ar-ckpt", required=True)
    ap.add_argument("--whitener-dir", required=True)
    ap.add_argument("--whitener-prefix", default="whitening_iolens_chat")
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--min-gap", type=float, default=0.15)
    ap.add_argument("--min-win", type=float, default=0.65)
    ap.add_argument("--actor-device", default="cuda:0")
    ap.add_argument("--reward-device", default="cuda:1")
    ap.add_argument("--reward-agg", choices=["single", "joint"], default="single")
    ap.add_argument("--reward-k-max", type=int, default=4)
    args = ap.parse_args()

    from oracle_lens.pipeline.ar_loader import load_ladder_whiteners, load_lc_reconstructor
    from oracle_lens.pipeline.rl_reward import (
        RewardSpace,
        extract_explanation,
        score_texts,
        score_texts_joint,
    )

    recon = load_lc_reconstructor(Path(args.ar_ckpt), device=args.reward_device, eager=True)
    whiteners = load_ladder_whiteners(
        Path(args.whitener_dir), prefix=args.whitener_prefix, ridge_c=args.ridge,
        layers=tuple(recon.layers))
    ar_layers = {int(x) for x in recon.layers}

    # >= 2 usable rows per layer so a within-layer derangement exists
    rows_all = [r for r in load_gate_rows(args.parquet, args.n * 4)
                if int(r["layer"]) in ar_layers]
    by_layer: dict[int, list] = defaultdict(list)
    for r in rows_all:
        by_layer[int(r["layer"])].append(r)
    rows = []
    per_layer = max(2, args.n // max(1, len(by_layer)))
    for ly in sorted(by_layer):
        if len(by_layer[ly]) >= 2:
            rows.extend(by_layer[ly][:per_layer])
    rows = rows[: args.n if args.n % 2 == 0 else args.n + 1]
    assert len(rows) >= 8, f"too few usable rows ({len(rows)}) for a probe"

    # generation arm: the trainer's rollout path (hook injection, greedy)
    actor, tok, _cfg, vectors_ref, eos_ids = load_actor(
        args.base_ckpt, args.ao_lora, args.sidecar, args.actor_device)
    pad_id = tok.eos_token_id
    texts = []
    for r in rows:
        v = torch.tensor(np.asarray(r["activation_vector"], dtype=np.float32))
        resp = hook_greedy(actor, r["prompt_ids"], v, vectors_ref, eos_ids, pad_id,
                           args.max_new, args.actor_device)
        texts.append(extract_explanation(tok.decode(resp, skip_special_tokens=True)))

    golds = torch.tensor(np.asarray([r["gold_vector"] for r in rows], dtype=np.float32))
    lys = torch.tensor([int(r["layer"]) for r in rows])
    # within-layer rotation -> derangement
    perm = list(range(len(rows)))
    groups: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[int(r["layer"])].append(i)
    for g in groups.values():
        for a, b in zip(g, g[1:] + g[:1], strict=True):
            perm[a] = b
    assert all(perm[i] != i for i in range(len(rows))), "not a derangement"

    space = RewardSpace()
    pad0 = int(tok.pad_token_id or 0)
    def _score(g):
        if args.reward_agg == "joint":
            return score_texts_joint(recon, tok, texts, g, lys, whiteners, space=space,
                                     k_max=args.reward_k_max, pad_id=pad0,
                                     device=args.reward_device)
        return score_texts(recon, tok, texts, g, lys, whiteners, space=space,
                           pad_id=pad0, device=args.reward_device)
    res_m = _score(golds)
    res_s = _score(golds[perm])

    r_m = [float(x) for x in res_m.reward]
    r_s = [float(x) for x in res_s.reward]
    n = len(rows)
    gap = sum(r_m) / n - sum(r_s) / n
    wins = sum(1 for a, b in zip(r_m, r_s, strict=True) if a > b) / n
    ok = gap >= args.min_gap and wins >= args.min_win
    return verdict("gate_probe", ok, {
        "n": n, "layers": sorted(groups),
        "mean_matched": round(sum(r_m) / n, 4),
        "mean_shuffled": round(sum(r_s) / n, 4),
        "gap": round(gap, 4), "paired_win_rate": round(wins, 3),
        "mean_fve_matched": round(float(np.nanmean([float(x) for x in res_m.fve])), 4),
        "invalid_extractions": int((~res_m.valid).sum()),
        "known_good": {"matched": -1.4994, "shuffled": -2.0009, "gap": 0.502,
                       "win": 0.917, "fve_sft_baseline": 0.102},
        "samples": [
            {"row": rows[i]["row_id"], "layer": int(lys[i]),
             "r_matched": round(r_m[i], 3), "r_shuffled": round(r_s[i], 3),
             "text": texts[i][:110]}
            for i in range(min(6, n))
        ],
    })


if __name__ == "__main__":
    raise SystemExit(main())
