"""Bare-metal per-layer whitener fit from a multi-layer pairs dir's OWN activations — iolens.

Port of ``ola_modal.py::fit_ml_whitener`` for the standalone box: fits per-layer mean+cov
(fp64 streaming ``MomentAccumulator``) from the FIRST ``--n-shards-use`` train shards of a
pairs dir (gate G10 wants >=1M rows per layer — about 12 iolens sub-shards) and writes
``$OLA_ROOT/{out_prefix}_L{L}.safetensors``. Loads the shards eagerly (fits this box's RAM)
and streams 8192-row chunks through the accumulator on GPU.

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/datagen/iolens_fit_whitener.py \
        --pairs-dir ml_pairs_iolens_chat --out-prefix whitening_iolens_chat
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-dir", required=True, help="OLA_ROOT-relative pairs dir")
    ap.add_argument("--out-prefix", required=True, help="e.g. whitening_iolens_chat")
    ap.add_argument(
        "--n-shards-use", type=int, default=12,
        help="iolens sub-shards hold <=100k pairs each; 12 gives ~1.2M rows/layer (gate G10 "
        "wants >=1M). Eager load ~200 GB RAM at that size — fine on this box.",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from oracle_lens.core.stats import MomentAccumulator
    from oracle_lens.core.whitening import save_moments
    from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards

    root = ola_root()
    paths = sorted((root / args.pairs_dir).glob("pairs_train_*.safetensors"))[: args.n_shards_use]
    if not paths:
        raise SystemExit(f"[iolens-whiten] no pairs_train_*.safetensors in {root}/{args.pairs_dir}")
    pairs, _meta = load_multilayer_shards(paths)  # eager: targets resident (~45 GB/shard, RAM ok)
    d = int(pairs.targets.shape[-1])
    commit = git_commit()
    written: dict[str, int] = {}
    for li, lyr in enumerate(LAYERS):
        acc = MomentAccumulator(d_model=d, device=args.device)
        for start in range(0, len(pairs), 8192):
            acc.update(pairs.targets[start : start + 8192, li].float().to(args.device))
        out = root / f"{args.out_prefix}_L{lyr}.safetensors"
        save_moments(
            out,
            acc.mean().float(),
            acc.covariance().float(),
            meta={
                "model_id": MODEL_ID,
                "dataset_id": args.pairs_dir,
                "layer": str(lyr),
                "n_samples": str(acc.count),
                "git_commit": commit,
            },
        )
        written[str(lyr)] = acc.count
        print(f"[iolens-whiten] L{lyr}: {acc.count} samples -> {out.name}", flush=True)
    print(json.dumps({"prefix": args.out_prefix, "pairs_dir": args.pairs_dir, "layers": written}))


if __name__ == "__main__":
    main()
