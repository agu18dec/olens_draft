"""Slice an arout bank to a crop prefix [0, hi) — the small-bank companion.

The AR-reconstruction bank (ao_arout_train_*.safetensors) ships as monolithic
131,790-crop shards; the GT bank can be regenerated at any granularity (the
precompute's shard scheme is free). To pair them at a smaller range, slice the
AR side: targets[:hi], layer_pick[:hi], meta.hi = hi. load_arout only requires
shards to tile [0, n) contiguously with matching pool fingerprints, and
prep_rl_data's pick-table equality then compares identically-sliced tables —
the seeded picks are generated over the FULL pool and sliced by shard bounds on
both sides, so a prefix slice preserves alignment exactly.

Usage:
    uv run --no-sync python scripts/rl/slice_arout.py \
        --src artifacts/sc/ao_arout/ar.chat.mlayer.lc.s0/ex16014240 \
        --dst artifacts/sc/ao_arout_small --split train --hi 65895
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--hi", type=int, required=True)
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    paths = sorted(args.src.glob(f"ao_arout_{args.split}_*.safetensors"))
    assert paths, f"no ao_arout_{args.split}_* under {args.src}"
    args.dst.mkdir(parents=True, exist_ok=True)
    out_path = args.dst / f"ao_arout_{args.split}_0000.safetensors"
    if out_path.exists():
        print(f"[slice] {out_path} exists — skipping (idempotent)")
        return

    # shards are sorted by lo; take from the front until hi is covered
    taken = 0
    targets_parts = []
    picks_parts = []
    meta0 = None
    for p in paths:
        with safe_open(str(p), framework="pt") as f:
            meta = json.loads((f.metadata() or {}).get("meta", "{}"))
            if meta0 is None:
                meta0 = meta
                assert int(meta["lo"]) == 0, f"first shard starts at {meta['lo']}, want 0"
            n = f.get_slice("targets").get_shape()[0]
            need = min(n, args.hi - taken)
            if need <= 0:
                break
            targets_parts.append(f.get_slice("targets")[:need])
            picks_parts.append(f.get_slice("layer_pick")[:need])
            taken += need
            print(f"[slice] {p.name}: took {need}/{n} crops (total {taken})", flush=True)
    assert taken == args.hi, f"bank ends at {taken} < requested hi {args.hi}"

    import torch

    targets = torch.cat(targets_parts) if len(targets_parts) > 1 else targets_parts[0]
    picks = torch.cat(picks_parts) if len(picks_parts) > 1 else picks_parts[0]
    meta = dict(meta0)
    meta["lo"], meta["hi"] = 0, args.hi
    meta["sliced_from"] = str(args.src)
    save_file(
        {"targets": targets.contiguous(), "layer_pick": picks.contiguous()},
        str(out_path), metadata={"meta": json.dumps(meta)},
    )
    print(f"[slice] wrote {out_path} ({taken} crops, "
          f"{targets.numel() * targets.element_size() / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
