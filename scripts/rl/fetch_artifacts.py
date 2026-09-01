"""Fetch + lay out every artifact the self-contained AO RL stack needs on a fresh box.

Idempotent. Downloads from the consolidated HF dataset repo (agu18dec/local-workspace)
plus the base model hub snapshot, then builds an $SC root that mirrors the cluster's
$OLA_ROOT naming (so ao_precompute_gt.py / prep_rl_data.py run unmodified) and writes
the `lora_hf/` copy of the SFT AO adapter:

  - `._orig_mod.` stripped from every safetensors key. Our AO SFTs train with
    compile_blocks=on, so the saved adapter keys carry the torch.compile wrapper
    prefix; PeftModel.from_pretrained on an UNCOMPILED base would silently load
    nothing (the documented inert-LoRA failure — see ar_loader.load_ao_adapter).
    This trainer never compiles (Hopper/Triton NaN bug + skip-lens's own no-compile
    rule), so the strip happens once, here, at fetch time.
  - `lora_dropout: 0` in the copied adapter_config.json — dropout-free by
    construction (rollout and update must sample the same policy; skip-lens
    enforces the same invariant).

Usage:
    uv run python scripts/rl/fetch_artifacts.py

Lays out artifacts/sc by default; override with --sc-root or $SC_ROOT.
"""


import argparse
import json
import os
import sys
from pathlib import Path

DATASET_REPO = "agu18dec/local-workspace"
BASE_MODEL = "Qwen/Qwen3.6-27B"

AO_LORA = "ckpts/ao/chat/k4.L20plus.s2/step3002/lora"
AR_CKPT = "ckpts/ar/chat/mlayer.lc.s0/ex16014240"
INCLUDE = [
    AO_LORA + "/**",
    AR_CKPT + "/**",
    "data/whiteners/chat/**",
    "data/ao/pool/pool_iolens.safetensors",
    "data/ao/arout/ar.chat.mlayer.lc.s0/ex16014240/**",
    "data/rollouts/chat/**",
]

# $SC link name (OLA_ROOT-style, matches docs/project/experiments/ola/iolens_runbook.md §paths)
# -> path under the hf mirror.
LINKS = {
    "ao_pool": "data/ao/pool",
    "ao_arout/ar.chat.mlayer.lc.s0/ex16014240": "data/ao/arout/ar.chat.mlayer.lc.s0/ex16014240",
    "rollouts_iolens/chat": "data/rollouts/chat",
    "whiteners/chat": "data/whiteners/chat",
    "ckpts/ao/chat/k4.L20plus.s2/step3002": "ckpts/ao/chat/k4.L20plus.s2/step3002",
    "ckpts/ar/chat/mlayer.lc.s0/ex16014240": AR_CKPT,
}


def strip_orig_mod(lora_dir: Path, out_dir: Path) -> None:
    """Write the uncompiled-key copy of a (possibly compiled) LoRA adapter."""
    from safetensors.torch import load_file, save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    sd = load_file(lora_dir / "adapter_model.safetensors")
    stripped = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    n_changed = sum(1 for a, b in zip(sd, stripped, strict=True) if a != b)
    save_file(stripped, out_dir / "adapter_model.safetensors")

    cfg = json.loads((lora_dir / "adapter_config.json").read_text())
    dropout_was = cfg.get("lora_dropout")
    cfg["lora_dropout"] = 0.0
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(
        f"[fetch] lora_hf: {len(sd)} tensors, {n_changed} keys stripped of ._orig_mod., "
        f"lora_dropout {dropout_was} -> 0.0 -> {out_dir}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-root", type=Path, default=Path(os.environ.get("SC_ROOT", "artifacts/sc")))
    ap.add_argument("--skip-download", action="store_true", help="layout + strip only")
    args = ap.parse_args()

    sc = args.sc_root
    mirror = sc / "hf"
    mirror.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        from huggingface_hub import snapshot_download

        snapshot_download(BASE_MODEL)  # hub cache; resolve via checks/_lib.resolve_snapshot
        for pattern in INCLUDE:  # one call per pattern — multi-pattern CLI parsing is flaky
            snapshot_download(
                DATASET_REPO, repo_type="dataset", local_dir=mirror, allow_patterns=[pattern]
            )

    missing = [p for p in [AO_LORA, AR_CKPT] if not (mirror / p).exists()]
    for name in ("pool_iolens.safetensors",):
        if not (mirror / "data/ao/pool" / name).exists():
            missing.append(f"data/ao/pool/{name}")
    if missing:
        sys.exit(f"[fetch] FATAL — missing after download: {missing}")

    for link, target in LINKS.items():
        dst = sc / link
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            continue
        dst.symlink_to(mirror / target)
    print(f"[fetch] layout ready under {sc}", flush=True)

    lora_hf = mirror / AO_LORA.replace("/lora", "/lora_hf")
    if not (lora_hf / "adapter_model.safetensors").exists():
        strip_orig_mod(mirror / AO_LORA, lora_hf)
    else:
        print(f"[fetch] {lora_hf} already present — skipping strip", flush=True)

    # heads.pt sanity for the frozen AR (layer count is derived from layer_emb rows)
    ar = mirror / AR_CKPT
    for f in ("heads.pt", "lora/adapter_model.safetensors"):
        assert (ar / f).exists(), f"frozen AR incomplete: missing {ar / f}"
    print("[fetch] all artifacts verified", flush=True)


if __name__ == "__main__":
    main()
