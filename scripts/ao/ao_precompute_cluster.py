"""Precompute AR reconstructions for the AO pool: ``ar_out[crop, 17, 5120]`` bf16 shards.

One Slurm array task = one contiguous slice of the pool's canonical crop order, one GPU:

    scripts/cluster/submit.sh -J ao-pre-<rung> -g 1 -t 06:00:00 -a 0-7 -- \
        uv run --no-sync python scripts/ao/ao_precompute_cluster.py \
            --ar-run ar.asst.on.mlayer.lc.alldata.crop32.b512.s0 --n-shards 8 --max-crops 700000

Shards are idempotent (atomic ``.tmp -> replace``; an existing complete shard is skipped) and
carry the pool fingerprint — the trainer refuses shards from a stale pool. The eval split
(``--split eval``) runs as a single small task and always precomputes ALL its crops.

The AR checkpoint must already be local (``fetch_ar_checkpoint`` from a login/CPU context, or
the ml_checkpoints sibling) — GPU jobs run HF-offline.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from oracle_lens.hf_offline import hf_offline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _heads_sha(ckpt_dir: Path) -> str:
    """blake2b-16 of heads.pt bytes — the AR identity stamp for arout shards."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    with open(ckpt_dir / "heads.pt", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)




def main() -> None:
    hf_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-run", type=str, required=True)
    ap.add_argument(
        "--pool",
        type=str,
        required=True,
        help="pool artifact the shards are computed FROM (e.g. ao_pool/pool_v2.safetensors; "
        "eval splits take the matching eval pool). REQUIRED: a silent v1 default poisoned all "
        "three small-rung precomputes on 2026-07-30 — 100+ GPU-hours of shards carrying the "
        "wrong fingerprint, caught only by the trainer's load-time check.",
    )
    ap.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--shard", type=int, default=-1, help="-1 = $SLURM_ARRAY_TASK_ID (or 0)")
    ap.add_argument("--max-crops", type=int, default=0, help="0 = all kept crops (prefix cap)")
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--out-dir", type=str, default="", help="default ao_arout/<ar_run>")
    # k-sliced storage (iolens): keep only each crop's k SEEDED layers — x8.5 disk. The pick
    # replicates AOLadderDataset exactly (train picks under --layer-seed over the conv_split
    # train side, val picks under seed+1 over the val side; torch.rand rows are prefix-stable,
    # so trainer-side n_crops caps stay consistent). Train split only; eval keeps all 17.
    ap.add_argument("--layers-per-crop", type=int, default=0, help="0 = store all 17 layers")
    ap.add_argument("--layer-seed", type=int, default=0, help="= the AO run's --seed")
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument(
        "--layer-min", type=int, default=0,
        help="store/serve ONLY layers >= this (Agam, 2026-08-03: AO sees layers 20-63 only). "
        "The stored rows and the layer list recorded in shard metadata are restricted together, "
        "so nothing downstream can mis-map a row to a layer.",
    )
    ap.add_argument(
        "--layer-max", type=int, default=0,
        help="0 = no cap; store/serve ONLY layers <= this (u64 run: --layer-min 20 "
        "--layer-max 60 excludes the calibration-specialized L63). Restricted together with "
        "the metadata layer list, same as --layer-min.",
    )
    ap.add_argument("--n-val-crops", type=int, default=600)
    ap.add_argument(
        "--extension",
        action="store_true",
        help="extension-pool picks: NO conv_split — the AO trains on every crop of this pool "
        "(crop_sel=None => picks are rand(n,17)[layer_seed] over the canonical order)",
    )
    ap.add_argument(
        "--no-group",
        action="store_true",
        help="prompt_tag debug arm: full per-layer sweep + gather instead of the grouped "
        "k-forward path — the smoke proves the two bitwise-equal on real weights",
    )
    args = ap.parse_args()
    if args.shard < 0:
        args.shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if args.layers_per_crop and args.split != "train":
        raise SystemExit("[ao-pre] --layers-per-crop applies to the train split only")

    import torch
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.ao_arout import arout_micro_batch
    from oracle_lens.pipeline.ao_pool import AOPool, pool_fingerprint
    from oracle_lens.pipeline.ar_loader import (
        ar_head_mode,
        ar_layer_set,
        fetch_ar_checkpoint,
        load_reconstructor,
    )
    from oracle_lens.pipeline.multilayer_reconstructor import ml_collate

    root = ola_root()
    pool_path = root / args.pool
    pool = AOPool.load(pool_path)
    fp = pool_fingerprint(pool.ids, pool.keep)
    rows, lens = pool.crop_index()
    n_all = len(rows)
    n_use = min(n_all, args.max_crops) if (args.max_crops and args.split == "train") else n_all

    # The AO's layer universe. Picks MUST be drawn over this, not over len(LAYERS): with
    # --layer-min 20 the arout has 12 rows, so a pick of 12..16 gathers out of bounds and the
    # kernel dies on `idx_dim >= 0 && idx_dim < index_size` (both train shards did). Derived
    # from heads.pt (ar_layer_set) BEFORE the picks are drawn — never guessed from LAYERS: the
    # old `len(LAYERS)-1 if layer_min` inference encoded an lc/--drop-layers 0 assumption that
    # a prompt_tag or layer-max-trained AR silently violates.
    ckpt = fetch_ar_checkpoint(args.ar_run, dest=root / "hf_ckpts")
    print(f"[ao-pre] AR checkpoint: {ckpt}", flush=True)
    head_mode = ar_head_mode(ckpt)
    ar_layers = list(ar_layer_set(ckpt))
    keep_pos = [
        i for i, ly in enumerate(ar_layers)
        if ly >= args.layer_min and (not args.layer_max or ly <= args.layer_max)
    ]
    ao_layers = [ar_layers[i] for i in keep_pos]
    n_universe = len(ao_layers)
    if args.layer_min or args.layer_max:
        print(f"[ao-pre] layer window [{args.layer_min}, {args.layer_max or 'max'}]: storing "
              f"{len(ao_layers)} of {len(ar_layers)} layers -> {ao_layers}", flush=True)

    pick_all: "torch.Tensor | None" = None
    if args.layers_per_crop:
        k = args.layers_per_crop
        if k > n_universe:
            raise SystemExit(
                f"[ao-pre] --layers-per-crop {k} exceeds the {n_universe}-layer universe "
                f"{ao_layers} implied by --layer-min {args.layer_min}"
            )
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
    out_dir = root / (args.out_dir or f"ao_arout/{args.ar_run}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ao_arout_{args.split}_{args.shard:04d}.safetensors"
    if out_path.exists():
        print(f"[ao-pre] {out_path} exists — skipping (idempotent)", flush=True)
        return
    print(
        f"[ao-pre] {args.ar_run} split={args.split} shard {args.shard}/{args.n_shards} "
        f"crops [{lo}, {hi}) of {n_use} (pool {n_all}) fp={fp}",
        flush=True,
    )

    # eager for prompt_tag: the grouped path runs RAGGED sub-batch shapes, and dynamic=False
    # recompiles mid-service on this torch 2.9.1 + Hopper stack produced cos=NaN then a CUDA
    # illegal memory access (repro3b, 2026-08-10). lc keeps its compiled fixed-shape path.
    recon = load_reconstructor(ckpt, eager=(head_mode == "prompt_tag"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    pad_id = int(tokenizer.pad_token_id or 0)
    d_model = recon.head.linear.weight.shape[0]
    if list(recon.layers) != ar_layers:
        raise SystemExit(
            f"[ao-pre] layer set mismatch: heads.pt implies {ar_layers} but the loaded model "
            f"carries {list(recon.layers)} — refusing (picks would mis-map)"
        )
    n_store = args.layers_per_crop or len(ao_layers)
    out = torch.zeros(hi - lo, n_store, d_model, dtype=torch.bfloat16)
    with torch.no_grad():
        for base in range(lo, hi, args.micro_batch):
            top = min(base + args.micro_batch, hi)
            batch_rows = [
                {
                    "ids": pool.crop_ids(int(rows[i]), int(lens[i])),
                    "target": torch.zeros(1),  # unused by the forward
                }
                for i in range(base, top)
            ]
            batch = ml_collate(batch_rows, pad_id=pad_id, width=max(pool.lengths))
            preds = arout_micro_batch(
                recon,
                batch["input_ids"].cuda(),
                batch["attention_mask"].cuda(),
                picks=pick_all[base:top] if pick_all is not None else None,
                keep_pos=keep_pos,
                no_group=args.no_group,
            )
            out[base - lo : top - lo] = preds.to(torch.bfloat16).cpu()
            if (base - lo) % (args.micro_batch * 20) == 0:
                print(f"[ao-pre] {base - lo}/{hi - lo}", flush=True)

    from safetensors.torch import save_file

    meta = {
        "ar_run": args.ar_run,
        "pool": str(pool_path),
        "pool_fingerprint": fp,
        "split": args.split,
        "lo": lo,
        "hi": hi,
        "n_total": n_use,
        "d_model": int(d_model),
        "pad_width": max(pool.lengths),
        # authoritative row->layer mapping for every consumer of these shards
        "ar_layers": ar_layers,
        "ao_layers": ao_layers,
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "n_universe": n_universe,
        # AR identity: the pool fingerprint does NOT encode WHICH AR produced these vectors
        # (b200_ptar_gatechecks trap) — heads_sha does. head_mode makes shards self-describing.
        "head_mode": head_mode,
        "ar_ckpt_dir": str(ckpt),
        "heads_sha": _heads_sha(ckpt),
        "attn_impl": os.environ.get("OLA_ATTN_IMPL", "auto"),
    }
    if pick_all is not None:
        meta.update(
            layers_per_crop=args.layers_per_crop,
            layer_seed=args.layer_seed,
            split_seed=args.split_seed,
            n_val_crops=args.n_val_crops,
        )
    # Unique tmp per writer: shards may be computed by TWO lanes at once (a high-QoS array plus a
    # low-QoS scavenger of the same rung) — a shared .tmp path would interleave their writes and
    # one of them would rename a corrupt file into place. With unique tmps the worst case is a
    # duplicated computation; whichever atomic replace lands last wins with identical content.
    tmp = out_path.with_suffix(f".tmp.{os.getpid()}.{os.environ.get('SLURM_JOB_ID', '0')}")
    tensors = {"targets": out}
    if pick_all is not None:
        tensors["layer_pick"] = pick_all[lo:hi].contiguous()
    save_file(tensors, str(tmp), metadata={"meta": json.dumps(meta)})
    tmp.replace(out_path)
    print(f"[ao-pre] wrote {out_path} ({out.shape})", flush=True)


if __name__ == "__main__":
    main()
