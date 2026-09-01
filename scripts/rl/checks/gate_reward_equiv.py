"""Gate 2 — reward equivalence, three ways (port of check_06 --real, Ray arm dropped).

score_texts (the live trainer path) ≡ inline hook-capture recompute ≡ pure-numpy
whiten/cosine, atol 2e-3, against the REAL frozen AR (ex16014240) + chat
whiteners; plus the floor contract (empty text -> -4.0, valid=False).

Run: uv run python scripts/rl/checks/gate_reward_equiv.py \
       --parquet $SC/rl/rl_gate_0.parquet \
       --ar-ckpt $SC/hf/ckpts/ar/chat/mlayer.lc.s0/ex16014240 \
       --whitener-dir $SC/whiteners/chat
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from _sc_lib import load_gate_rows, verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ar-ckpt", required=True)
    ap.add_argument("--whitener-dir", required=True)
    ap.add_argument("--whitener-prefix", default="whitening_iolens_chat")
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from oracle_lens.pipeline.ar_loader import load_ladder_whiteners, load_lc_reconstructor
    from oracle_lens.pipeline.rl_reward import (
        RewardSpace,
        capture_block_outputs,
        extract_explanation,
        score_texts,
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    recon = load_lc_reconstructor(Path(args.ar_ckpt), device=args.device, eager=True)
    whiteners = load_ladder_whiteners(
        Path(args.whitener_dir), prefix=args.whitener_prefix, ridge_c=args.ridge,
        layers=tuple(recon.layers))

    rows = [r for r in load_gate_rows(args.parquet, args.n * 2)
            if int(r["layer"]) in set(recon.layers)][: args.n]
    assert len(rows) >= 8, f"too few usable gate rows ({len(rows)})"
    texts = [
        extract_explanation(f"<explanation>toy text about row {r['row_id']} layer "
                            f"{r['layer']} and some more words</explanation>")
        for r in rows
    ]
    golds = torch.tensor(np.asarray([r["gold_vector"] for r in rows], dtype=np.float32))
    lys = torch.tensor([int(r["layer"]) for r in rows])
    pad_id = int(tok.pad_token_id or 0)

    # (1) live path
    res = score_texts(recon, tok, texts, golds, lys, whiteners,
                      space=RewardSpace(), pad_id=pad_id, device=args.device)

    # (2) inline recompute (right-pad, last REAL token via mask) + (3) numpy
    id_rows = [tok(t, add_special_tokens=False)["input_ids"][:64] for t in texts]
    width = max(len(r) for r in id_rows)
    ids = torch.full((len(id_rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_rows), width), dtype=torch.long)
    for i, r in enumerate(id_rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    need = sorted({int(x) for x in lys.tolist()})
    cap = capture_block_outputs(recon.backbone, need, ids.to(args.device), mask.to(args.device))
    last_idx = mask.sum(dim=1) - 1
    li = {int(ly): i for i, ly in enumerate(recon.layers)}
    emb_w = recon.layer_emb.weight
    max_d = 0.0
    per_row = []
    with torch.no_grad():
        for i in range(len(rows)):
            ly_i = int(lys[i])
            h_l = cap[ly_i][i, int(last_idx[i])].float()
            pred = recon.head(h_l + emb_w[li[ly_i]]).float().cpu().numpy()
            w = whiteners[ly_i]
            pw = (pred - w.mu.numpy()) @ w.w.numpy().T
            gw = (golds[i].numpy() - w.mu.numpy()) @ w.w.numpy().T
            cos = float(pw @ gw / (np.linalg.norm(pw) * np.linalg.norm(gw)))
            r_np = -2.0 * (1.0 - cos)
            dd = abs(r_np - float(res.reward[i]))
            max_d = max(max_d, dd)
            per_row.append({"row": rows[i]["row_id"], "r_live": round(float(res.reward[i]), 5),
                            "r_numpy": round(r_np, 5), "abs_diff": round(dd, 6)})

    empty = score_texts(recon, tok, [""], golds[:1], lys[:1], whiteners,
                        pad_id=pad_id, device=args.device)
    floor_ok = math.isclose(float(empty.reward[0]), -4.0) and not bool(empty.valid[0])
    ok = max_d < args.atol and bool(res.valid.all()) and floor_ok
    return verdict("gate_reward_equiv", ok, {
        "max_abs_diff": round(max_d, 6), "atol": args.atol,
        "floor_on_empty": floor_ok, "rows": per_row[:6],
    })


if __name__ == "__main__":
    raise SystemExit(main())
