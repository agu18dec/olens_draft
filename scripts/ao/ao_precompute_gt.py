"""Precompute GROUND-TRUTH residuals for the AO pool, in arout shard format.

The GT-AO arm (Agam, 2026-08-05): train an AO whose input vectors at TRAINING time are the
same thing it sees at inference time — the model's true residual at ``prev_pos =
span_start - 1`` — instead of the AR's reconstruction. Everything else (pool, picks, val
split, trainer, config) stays byte-identical to the AR-vector runs; these shards drop into
``ao_train_cluster.py --arout-dir`` unchanged because they carry the same tensors
(``targets`` ``[n, k, d]`` + ``layer_pick``) and the same metadata contract (pool
fingerprint, lo/hi alignment, ``ao_layers``).

Coordinate contract (must match iolens_capture_pairs / the AR's training targets): the
residual is taken over the conversation's FULL rendered ids (prompt + output) at absolute
position ``prompt_len + start - 1``. The pool's ``conv`` ids index the ao_train-split,
think-reemission-filtered conversation list in rollout-shard order — reconstructed here with
the exact ``ao_build_pool.py`` filter, then verified PER CROP by matching the pool's stored
span ids against the conversation slice (a single mismatch aborts the shard).

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/ao/ao_precompute_gt.py \
        --pool ao_pool/pool_iolens.safetensors --split train --n-shards 4 --shard 0 \
        --layer-min 20 --layers-per-crop 4
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)


def _hf_offline() -> None:
    if os.environ.get("AO_HF_ONLINE") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    _hf_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--out-dir", type=str, default="", help="default ao_gtout/<pool stem>")
    ap.add_argument("--rollout-store-dir", type=str, default="rollouts_iolens/chat")
    ap.add_argument("--rollout-glob", type=str, default="rollouts_*.safetensors")
    ap.add_argument("--conv-batch", type=int, default=16, help="conversations per forward")
    # pick machinery — MUST mirror ao_precompute_cluster.py exactly
    ap.add_argument("--layers-per-crop", type=int, default=0)
    ap.add_argument("--layer-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--layer-min", type=int, default=20)
    ap.add_argument("--n-val-crops", type=int, default=600)
    ap.add_argument("--extension", action="store_true")
    args = ap.parse_args()
    if args.shard < 0:
        args.shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if args.split == "eval":
        raise SystemExit(
            "[gt-pre] eval pools are carved from PAIRS spans (build_eval_pool), so their "
            "conv/start fields do not reference rollout conversations — this mapping would be "
            "silently wrong. GT eval vectors come directly from the pairs_eval_* shards "
            "(they ARE true residuals); assemble those separately."
        )

    import torch
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.ao_pool import AOPool, pool_fingerprint
    from oracle_lens.pipeline.multilayer import LAYERS
    from oracle_lens.pipeline.rollout_store import SPLITS, load_rollout_shards

    root = ola_root()
    pool_path = root / args.pool
    pool = AOPool.load(pool_path)
    fp = pool_fingerprint(pool.ids, pool.keep)
    rows, lens = pool.crop_index()
    n_use = len(rows)

    # layer universe + seeded picks — verbatim from ao_precompute_cluster.py
    n_rows_expected = len(LAYERS) - 1 if args.layer_min else len(LAYERS)
    ar_layers = list(LAYERS[-n_rows_expected:])
    ao_layers = [ly for ly in ar_layers if ly >= args.layer_min]
    n_universe = len(ao_layers)
    keep_pos = [i for i, ly in enumerate(ar_layers) if ly >= args.layer_min]
    pick_all: "torch.Tensor | None" = None
    if args.layers_per_crop:
        k = args.layers_per_crop
        if k > n_universe:
            raise SystemExit(f"[gt-pre] k={k} exceeds the {n_universe}-layer universe")
        if args.extension:
            gen_t = torch.Generator().manual_seed(args.layer_seed * 999_983 + 17)
            pick_all = (
                torch.rand(n_use, n_universe, generator=gen_t).argsort(dim=1)[:, :k]
            ).to(torch.int8)
        else:
            from oracle_lens.pipeline.ao_ladder import conv_split

            train_idx, val_idx = conv_split(
                pool, n_val_crops=args.n_val_crops, seed=args.split_seed, n_avail=n_use
            )
            gen_t = torch.Generator().manual_seed(args.layer_seed * 999_983 + 17)
            train_pick = (
                torch.rand(len(train_idx), n_universe, generator=gen_t).argsort(dim=1)[:, :k]
            )
            gen_v = torch.Generator().manual_seed((args.layer_seed + 1) * 999_983 + 17)
            val_pick = (
                torch.rand(len(val_idx), n_universe, generator=gen_v).argsort(dim=1)[:, :k]
            )
            pick_all = torch.zeros(n_use, k, dtype=torch.int8)
            pick_all[train_idx] = train_pick.to(torch.int8)
            pick_all[val_idx] = val_pick.to(torch.int8)

    bounds = [round(s * n_use / args.n_shards) for s in range(args.n_shards + 1)]
    lo, hi = bounds[args.shard], bounds[args.shard + 1]
    stem = Path(args.pool).stem
    out_dir = root / (args.out_dir or f"ao_gtout/{stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ao_arout_{args.split}_{args.shard:04d}.safetensors"
    if out_path.exists():
        print(f"[gt-pre] {out_path} exists — skipping (idempotent)", flush=True)
        return
    print(
        f"[gt-pre] GT residuals split={args.split} shard {args.shard}/{args.n_shards} "
        f"crops [{lo}, {hi}) of {n_use} fp={fp} layers={ao_layers}",
        flush=True,
    )

    # --- conv_uid -> rollout-store row: reproduce ao_build_pool.py's filter EXACTLY ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    store_dir = root / args.rollout_store_dir
    shard_paths = sorted(store_dir.glob(args.rollout_glob))
    rolls = load_rollout_shards(shard_paths)
    split_want = SPLITS.index("ao_train" if args.split == "train" else "eval")
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    uid_to_row: list[int] = []
    for i in range(len(rolls)):
        if int(rolls.split_id[i]) != split_want:
            continue
        out_ids = rolls.output_ids(i).tolist()
        if isinstance(think_id, int) and think_id in out_ids:
            continue
        uid_to_row.append(i)
    print(f"[gt-pre] conv map: {len(uid_to_row):,} conversations in split", flush=True)

    # --- group this shard's crops by conversation ---
    conv_of_window = pool.conv  # [n_windows]
    start_of_window = pool.start
    by_conv: dict[int, list[int]] = defaultdict(list)
    for ci in range(lo, hi):
        by_conv[int(conv_of_window[int(rows[ci])])].append(ci)

    from oracle_lens.core.dump import conversations_layers_resid_batched
    from oracle_lens.model import ModelBackend

    backend = ModelBackend(MODEL_ID, device="cuda", dtype=torch.bfloat16)
    d_model = 5120
    n_store = args.layers_per_crop or len(ao_layers)
    out = torch.zeros(hi - lo, n_store, d_model, dtype=torch.bfloat16)
    norms: list[float] = []

    uids = sorted(by_conv)
    done = 0
    for b0 in range(0, len(uids), args.conv_batch):
        chunk = uids[b0 : b0 + args.conv_batch]
        rendered, metas = [], []
        for uid in chunk:
            row = uid_to_row[uid]
            plen = int(rolls.prompt_len[row])
            full = rolls.conv_ids(row).tolist()
            crops = by_conv[uid]
            # truncate at the last position we need (prev of the furthest span start; the
            # span itself is also needed for the ids-match verification)
            last_needed = max(
                plen + int(start_of_window[int(rows[ci])]) + int(pool.lengths[int(lens[ci])])
                for ci in crops
            )
            rendered.append(full[:last_needed])
            metas.append((uid, row, plen, crops))
        resid = conversations_layers_resid_batched(backend, rendered, layers=ao_layers)
        for j, (uid, row, plen, crops) in enumerate(metas):
            full = rolls.conv_ids(row).tolist()
            for ci in crops:
                w = int(rows[ci])
                start = int(start_of_window[w])
                n_tok = int(pool.lengths[int(lens[ci])])  # lens are LENGTH INDICES
                abs_start = plen + start
                # HARD verification: the pool's stored span ids must equal the conv slice
                span = pool.crop_ids(w, int(lens[ci])).tolist()
                if full[abs_start : abs_start + n_tok] != span:
                    raise SystemExit(
                        f"[gt-pre] COORDINATE MISMATCH conv_uid={uid} crop={ci}: pool span != "
                        f"conversation slice at {abs_start} — refusing to write garbage"
                    )
                prev = abs_start - 1
                vec_all = torch.stack(
                    [resid[ly][j, prev] for ly in ao_layers], dim=0
                )  # [n_universe, d]
                if pick_all is not None:
                    sel = pick_all[ci].long()
                    vec = vec_all[sel]
                else:
                    vec = vec_all[keep_pos] if len(keep_pos) != len(ar_layers) else vec_all
                out[ci - lo] = vec.to(torch.bfloat16).cpu()
                norms.append(float(vec_all.norm(dim=-1).median()))
        done += len(chunk)
        if b0 % (args.conv_batch * 10) == 0:
            n_done = sum(len(by_conv[u]) for u in uids[:done])
            print(f"[gt-pre] convs {done}/{len(uids)} crops~{n_done}", flush=True)

    from safetensors.torch import save_file

    med = float(torch.tensor(norms).median())
    meta = {
        "ar_run": "gt-activations",
        "source": "gt",
        "pool": str(pool_path),
        "pool_fingerprint": fp,
        "split": args.split,
        "lo": lo,
        "hi": hi,
        "n_total": n_use,
        "d_model": d_model,
        "pad_width": max(pool.lengths),
        "ar_layers": ar_layers,
        "ao_layers": ao_layers,
        "layer_min": args.layer_min,
        "n_universe": n_universe,
        "gt_median_norm_shard": med,
    }
    if pick_all is not None:
        meta.update(
            layers_per_crop=args.layers_per_crop,
            layer_seed=args.layer_seed,
            split_seed=args.split_seed,
            n_val_crops=args.n_val_crops,
        )
    tmp = out_path.with_suffix(f".tmp.{os.getpid()}")
    tensors = {"targets": out}
    if pick_all is not None:
        tensors["layer_pick"] = pick_all[lo:hi].contiguous()
    save_file(tensors, str(tmp), metadata={"meta": json.dumps(meta)})
    tmp.replace(out_path)
    print(f"[gt-pre] wrote {out_path} ({out.shape}) | shard median |h| = {med:.1f}", flush=True)


if __name__ == "__main__":
    main()
