"""GATE: the precomputed AR reconstructions must reproduce the AR's known quality (CPU-only).

Scores ``FVE = cos^2(whiten(ar_out_eval[c, l]), whiten(h_l))`` over the eval pool against the
TRUE stored residuals, per layer and per crop length, in the AR's own training basis
(on-policy whitener, ridge_c 0.1). Catches — in one shot — a wrong checkpoint, a missed
compile-wrap (scores collapse to ~chance), a layer-index off-by-one (per-layer profile
scrambles), and pool/arout misalignment (fingerprint check).

Read the result against the rung's published val_fve (alldata asst ~= 9.78% mean): our eval
crops are a different draw (deterministic prefixes at 6 lengths, incl. N=64 which the crop32 AR
never trained on), so expect the SAME ballpark on the N<=32 subset, not equality.

    # long job — run in tmux, tee to logs/:
        uv run python scripts/ao/ao_gate_arout.py \
            --ar-run ar.asst.on.mlayer.lc.alldata.crop32.b512.s0
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-run", type=str, required=True)
    ap.add_argument("--eval-pool", type=str, default="ao_pool/eval_pool_v1.safetensors")
    ap.add_argument("--arout-dir", type=str, default="")
    ap.add_argument("--pairs-dir", type=str, default="ml_pairs_onpolicy_chat")
    ap.add_argument("--whitener-dir", type=str, default="hf_ckpts/whiteners")
    ap.add_argument("--whitener-prefix", type=str, default="whitening_onpolicy")
    args = ap.parse_args()

    import torch

    from oracle_lens.pipeline.ao_ladder import load_arout
    from oracle_lens.pipeline.ao_pool import AOPool, pool_fingerprint
    from oracle_lens.pipeline.ar_loader import load_ladder_whiteners
    from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards_lazy

    root = ola_root()
    eval_pool = AOPool.load(root / args.eval_pool)
    efp = pool_fingerprint(eval_pool.ids, eval_pool.keep)
    arout_dir = root / (args.arout_dir or f"ao_arout/{args.ar_run}")
    arout, top = load_arout(arout_dir, split="eval", expect_fingerprint=efp)
    rows, lens = eval_pool.crop_index()
    n = len(rows)
    assert top["n_crops"] == n, f"arout {top['n_crops']} crops != pool {n}"

    pairs_paths = sorted((root / args.pairs_dir).rglob("*.safetensors"))
    pairs, _ = load_multilayer_shards_lazy(pairs_paths)
    # Layer semantics come from the arout shards. `pred` has one row per AO layer (12 for
    # --layer-min 20), while `true` is the AR's full 17-layer pair stack — so pred is indexed by
    # POSITION in ao_layers and true by that layer's position in LAYERS. Indexing both with the
    # same counter (the previous code) compares layer 20's reconstruction against layer 0's truth.
    ao_layers = list(top.get("ao_layers") or []) or list(LAYERS)
    print(f"[gate] arout declares {len(ao_layers)} layers: {ao_layers}", flush=True)
    whiteners = load_ladder_whiteners(
        root / args.whitener_dir, prefix=args.whitener_prefix, ridge_c=0.1,
        layers=tuple(ao_layers),
    )

    pred = torch.stack([torch.as_tensor(arout[i]) for i in range(n)]).float()
    true = torch.stack(
        [torch.as_tensor(pairs.targets[int(eval_pool.conv[int(rows[i])])]) for i in range(n)]
    ).float()
    fve = torch.zeros(n, len(ao_layers))
    for j, ly in enumerate(ao_layers):
        w = whiteners[ly]
        ti = list(LAYERS).index(ly)          # row of this layer in the AR's full pair stack
        pw, tw = w.whiten(pred[:, j]), w.whiten(true[:, ti])
        fve[:, j] = torch.nn.functional.cosine_similarity(pw, tw, dim=-1) ** 2

    print(f"[gate] {args.ar_run}: {n} eval crops")
    in_regime = torch.tensor([eval_pool.lengths[int(j)] <= 32 for j in lens])
    print(
        f"[gate] FVE mean ALL {float(fve.mean()):.4f} | N<=32 (AR regime) "
        f"{float(fve[in_regime].mean()):.4f} | N=64 (OOD) {float(fve[~in_regime].mean()):.4f}"
    )
    print("[gate] per-layer FVE (N<=32):")
    for j, ly in enumerate(ao_layers):
        print(f"  L{ly:<3d} {float(fve[in_regime, j].mean()):.4f}")
    print("[gate] per-length FVE mean:")
    for j, length in enumerate(eval_pool.lengths):
        m = lens == j
        print(f"  N={length:<3d} {float(fve[m].mean()):.4f}  ({int(m.sum())} crops)")
    out = {
        "ar_run": args.ar_run,
        "fve_mean_all": float(fve.mean()),
        "fve_mean_le32": float(fve[in_regime].mean()),
        "fve_mean_64": float(fve[~in_regime].mean()),
        **{f"fve_L{ly}": float(fve[in_regime, j].mean()) for j, ly in enumerate(ao_layers)},
        "ao_layers": ao_layers,
    }
    gate_path = root / "ao_runs" / f"gate_arout_{args.ar_run}.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(out, indent=2))
    print(f"[gate] wrote {gate_path}")


if __name__ == "__main__":
    main()
