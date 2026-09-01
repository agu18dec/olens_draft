"""Prep for the AO GRPO run: merged VL checkpoint + NLA sidecar + RL/gate parquets.

Runs in the PROJECT venv (renders prompts under its transformers; the miles env never touches
a chat template). Stages are idempotent (skip-if-exists) and independently invokable:

    uv run python scripts/rl/prep_rl_data.py \\
        --ao-lora-dir $OLA_ROOT/ao_checkpoints/<run>/stepN/lora \\
        --ao-meta $OLA_ROOT/ao_checkpoints/<run>/stepN/meta.json \\
        --layers 20,24,28,32,36,40,44,48,52,56,60,63 --global-scale 64.559 \\
        --out-dir $OLA_ROOT/rl/<tag> [--stage merge|sidecar|parquet|all] ...

Artifacts in --out-dir:
    merged/                VL-layout merged AO (policy warm start + KL ref, sglang-servable)
    merged/nla_meta.yaml   sidecar: injection char/ids + injection_scale: null (pass-through —
                           parquet vectors are PRE-TRANSFORMED with the AO's own transform/scale)
    rl_train_<seed>.parquet / rl_gate_<seed>.parquet
        rows: row_id · layer · prompt_ids (prep-rendered, CJK slot inside) · prompt_text ·
              activation_vector (inject_gt-transformed AR-out — what the AO reads) ·
              gold_vector (raw TRUE residual at `layer` — what the reward scores against)
    prep.json              provenance (args, counts, git commit)

The parquet stage is wired at M2 against the target rung's pool/arout/pairs (their exact
paths are rung-specific); merge + sidecar are rung-agnostic and final.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stage_merge(args: argparse.Namespace, out: Path) -> Path:
    """LoRA -> full VL model (the merge_teacher_vl.py recipe; sglang-servable arch)."""
    import shutil

    merged = out / "merged"
    if (merged / "config.json").is_file() and (merged / "merge_provenance.json").is_file():
        print(f"[prep] {merged} already merged — skipping", flush=True)
        return merged

    lora_dir = Path(args.ao_lora_dir)
    if not (lora_dir / "adapter_model.safetensors").is_file():
        raise SystemExit(f"no adapter at {lora_dir}")

    n = -1
    if not (merged / "model.safetensors.index.json").is_file():
        import torch
        from transformers import AutoModelForImageTextToText

        from oracle_lens.pipeline.merge_vl import merge_lora_into_vl

        print("[prep] loading VL model (cpu, bf16) for merge", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cpu"
        )
        if not any("language_model.layers" in n_ for n_, _ in model.named_parameters()):
            raise SystemExit(
                f"{type(model).__name__} has no language_model.layers — wrong VL loader"
            )
        n = merge_lora_into_vl(model, lora_dir)
        merged.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged), safe_serialization=True)
    else:
        print("[prep] merged shards present — completing tokenizer/provenance only", flush=True)

    # tokenizer/processor files come as FILE COPIES from the base snapshot —
    # AutoProcessor.from_pretrained pulls a video-processor backend (torchvision/av)
    # the cluster venv doesn't ship (observed ImportError, job 46305). Serving needs
    # the files, not the python objects.
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.jinja"]))
    for f in snap.iterdir():
        if f.name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            continue  # keep the MERGED model's own configs
        if f.is_file():
            shutil.copy(f, merged / f.name)
    (merged / "merge_provenance.json").write_text(
        json.dumps(
            {"ao_lora_dir": str(lora_dir), "n_deltas": n, "git_commit": _git_commit()}, indent=2
        )
    )
    print(f"[prep] merged ({n} deltas this run) -> {merged}", flush=True)
    return merged


def build_ao_sidecar(
    tokenizer: Any, *, d_model: int, prompt_kind: str, layers: list[int], base_checkpoint: str
) -> dict[str, Any]:
    """NLA model-sidecar (schema v2) for the merged AO.

    injection_scale is null BY DESIGN: parquet vectors are pre-transformed at prep, and
    ``normalize_activation(v, None)`` is the documented pass-through — both the rollout splice
    and the actor's training forward inject the vector as-is (magnitude-preserving, the
    ``raw_scaled`` contract).

    The canonical actor template is the canonical layer's user-turn content with
    ``{injection_char}`` in the slot; the injection char + BOTH neighbor ids are asserted
    IDENTICAL across every trained layer's render (the layer number appears elsewhere in the
    prompt, never beside the slot) so the marker-scan injection can never false-negative.
    """
    from oracle_lens.pipeline import verbalizer as wv

    templates = {
        "explain": wv.WV_EXPLAIN_PROMPT_TEMPLATE,
        "continuation": wv.WV_CONTINUATION_PROMPT_TEMPLATE,
        "continuation_min": wv.WV_CONTINUATION_MIN_PROMPT_TEMPLATE,
        "continuation_raw": wv.WV_CONTINUATION_RAW_PROMPT_TEMPLATE,
        "concepts_raw": wv.WV_CONCEPTS_RAW_PROMPT_TEMPLATE,
        "explanation": wv.WV_PROMPT_TEMPLATE,
    }
    kind = wv.PROMPT_ALIASES.get(prompt_kind, prompt_kind)
    if kind not in templates:
        raise ValueError(f"no sidecar template for prompt kind {prompt_kind!r}")
    render = wv.renderer_for(kind)

    neighbor_sets: set[tuple[str, int, int, int]] = set()
    for ly in layers:
        p = render(tokenizer, layer=ly)
        slots = [i for i, t in enumerate(p.input_ids) if t == p.char_id]
        if slots != [p.slot] or p.slot in (0, len(p.input_ids) - 1):
            raise ValueError(f"layer {ly}: expected one mid-sequence slot, found {slots}")
        neighbor_sets.add(
            (p.char, p.char_id, int(p.input_ids[p.slot - 1]), int(p.input_ids[p.slot + 1]))
        )
    if len(neighbor_sets) != 1:
        raise ValueError(
            f"injection char/neighbors vary across layers: {sorted(neighbor_sets)} — the "
            f"marker-scan injection would false-negative on some layers"
        )
    char, char_id, left_id, right_id = next(iter(neighbor_sets))
    canonical_content = templates[kind].format(layer=layers[0], char="{injection_char}")
    return {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "actor",
        "stage": "iolens-ao-rl-warmstart",
        "base_checkpoint": base_checkpoint,
        "d_model": int(d_model),
        "extraction": {"injection_scale": None, "mse_scale": None},
        "tokens": {
            "injection_char": char,
            "injection_token_id": int(char_id),
            "injection_left_neighbor_id": int(left_id),
            "injection_right_neighbor_id": int(right_id),
            "critic_suffix_ids": None,
        },
        "prompt_templates": {"actor": canonical_content, "critic": None},
        "trained_on": [],
        "parent_checkpoints": [],
        "created_by": "scripts/rl/prep_rl_data.py",
    }


def stage_sidecar(args: argparse.Namespace, out: Path, run_cfg: dict[str, Any]) -> None:
    import yaml
    from transformers import AutoTokenizer

    sidecar_path = out / "merged" / "nla_meta.yaml"
    if sidecar_path.is_file() and not args.force_sidecar:
        print(f"[prep] {sidecar_path} exists — skipping", flush=True)
        return
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_kind = run_cfg.get("extra", {}).get("prompt", "explain")
    layers = [int(x) for x in args.layers.split(",")]
    sidecar = build_ao_sidecar(
        tokenizer, d_model=5120, prompt_kind=prompt_kind, layers=layers, base_checkpoint=MODEL_ID
    )
    sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True))
    print(
        f"[prep] sidecar -> {sidecar_path} (char {sidecar['tokens']['injection_char']!r})",
        flush=True,
    )


def stage_parquet(args: argparse.Namespace, out: Path, run_cfg: dict[str, Any]) -> None:
    """(crop, layer) rows: pre-transformed AR-out inject vectors + TRUE-residual golds.

    Sources are two k-sliced arout banks over the SAME pool with the SAME seeded layer
    picks (verified by hard equality of their pick tables):
      --arout-dir     AR reconstructions (what the AO reads — inject source)
      --gt-arout-dir  ground-truth residuals from ao_precompute_gt (reward gold source)
    Row recipe: sample crops (seeded, per split); pick one of the crop's k seeded layers;
    activation_vector = inject_gt(ar_arout[c][j], run transform/alpha/scale) — EXACTLY the
    SFT injection; gold_vector = gt_arout[c][j] raw; prompt = the run's own renderer at
    that layer. Train rows come from split=train, gate rows from split=eval (the ruler's
    conv-split val) — disjoint by construction.
    """
    import random

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.ao_ladder import load_arout
    from oracle_lens.pipeline.ao_pool import AOPool, pool_fingerprint
    from oracle_lens.pipeline.inject import inject_gt
    from oracle_lens.pipeline.verbalizer import renderer_for

    train_pq = out / f"rl_train_{args.seed}.parquet"
    gate_pq = out / f"rl_gate_{args.seed}.parquet"
    if train_pq.is_file() and gate_pq.is_file() and not args.force_parquet:
        print("[prep] parquets exist — skipping", flush=True)
        return

    transform = run_cfg.get("transform", "scaled")
    alpha = float(run_cfg.get("alpha", 8000.0))
    per_layer_scales = run_cfg.get("scales") or {}

    def scale_for_layer(ly: int) -> float:
        if per_layer_scales:
            return float(per_layer_scales[str(ly)])
        assert args.global_scale > 0, "--global-scale required (no per-layer scales in cfg)"
        return float(args.global_scale)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_kind = run_cfg.get("extra", {}).get("prompt", "explain")
    render = renderer_for(prompt_kind)

    pool = AOPool.load(Path(args.pool))
    fp = pool_fingerprint(pool.ids, pool.keep)

    # Both RL splits come from the banks' TRAIN shards: the AR bank's eval shard is
    # fingerprinted to a DIFFERENT pool (eval_pool_iolens) than the GT eval bank
    # (pool_iolens conv-split val) — no aligned eval pair exists on disk (observed,
    # job 46418). Gate rows are carved from crops DISJOINT from the train sample
    # (same seeded shuffle, tail slice); the science-ruler FVE eval stays in ao_fve_eval.
    for split, n_rows, out_path, tail in (
        ("train", args.n_rows, train_pq, False),
        ("train", args.gate_rows, gate_pq, True),
    ):
        if out_path.is_file() and not args.force_parquet:
            print(f"[prep] {out_path.name} exists — skipping", flush=True)
            continue
        ar_out, ar_top = load_arout(Path(args.arout_dir), split=split, expect_fingerprint=fp)
        gt_out, gt_top = load_arout(Path(args.gt_arout_dir), split=split, expect_fingerprint=fp)
        ao_layers = [int(x) for x in ar_top["ao_layers"]]
        assert ao_layers == [int(x) for x in gt_top["ao_layers"]], "layer universes differ"
        ar_pick = ar_top.get("layer_pick")
        gt_pick = gt_top.get("layer_pick")
        assert ar_pick is not None and gt_pick is not None, "expected k-sliced arout banks"
        assert torch.equal(ar_pick.to(torch.int8), gt_pick.to(torch.int8)), (
            "AR and GT banks carry DIFFERENT seeded layer picks — golds would mismatch "
            "the injected layers"
        )
        n_crops = len(ar_out)
        allowed_layers = {int(x) for x in args.layers.split(",")}
        assert len(gt_out) == n_crops, f"bank sizes differ: {n_crops} vs {len(gt_out)}"

        prompts = {ly: render(tokenizer, layer=ly) for ly in ao_layers}
        # crop -> (window row, length idx): lets each parquet row carry the SOURCE
        # span text the activation was captured before (wandb sample tables).
        crop_rows, crop_lens = pool.crop_index()
        rng = random.Random(args.seed)  # SAME shuffle both passes: head=train, tail=gate
        crop_ids = list(range(n_crops))
        rng.shuffle(crop_ids)
        crop_ids = crop_ids[-n_rows:] if tail else crop_ids[:n_rows]
        assert not tail or args.n_rows + n_rows <= n_crops, "train+gate exceed the bank"

        out_rows = []
        for c in crop_ids:
            picks = ar_pick[int(c)].tolist()
            j = rng.randrange(len(picks))
            ly = ao_layers[int(picks[j])]
            # AO cells trained on a layer SUBSET (e.g. 20-60, no 63) must not emit
            # rows for layers they never saw — re-draw among the crop's picks that
            # ARE in the trained set (--layers, authoritative). Old full-layer metas
            # pass through untouched (every picked layer is allowed). RNG stream is
            # only perturbed for crops that actually need a re-draw.
            if ly not in allowed_layers:
                avail = [jj for jj in range(len(picks))
                         if ao_layers[int(picks[jj])] in allowed_layers]
                assert avail, f"crop {c}: no picked layer in --layers ({picks})"
                j = avail[rng.randrange(len(avail))]
                ly = ao_layers[int(picks[j])]
            vec = torch.as_tensor(ar_out[int(c)])[j].float()
            v = inject_gt(vec.unsqueeze(0), transform, alpha=alpha, scale=scale_for_layer(ly))[0]
            gold = torch.as_tensor(gt_out[int(c)])[j].float()
            wv = prompts[ly]
            out_rows.append(
                {
                    "row_id": int(c),
                    "layer": ly,
                    "prompt_ids": [int(x) for x in wv.input_ids],
                    "prompt_text": tokenizer.decode(wv.input_ids),
                    "activation_vector": [float(x) for x in v],
                    "gold_vector": [float(x) for x in gold],
                    "slot": int(wv.slot),
                    # the span whose preceding residual is the activation (crop text)
                    "source_text": tokenizer.decode(
                        pool.crop_ids(int(crop_rows[int(c)]), int(crop_lens[int(c)]))
                    ),
                }
            )
            if len(out_rows) % 5000 == 0:
                print(f"[prep] {split}: {len(out_rows)}/{len(crop_ids)} rows", flush=True)

        pq.write_table(pa.Table.from_pylist(out_rows), out_path)  # type: ignore[no-untyped-call]
        print(f"[prep] {split}: wrote {len(out_rows)} rows -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ao-lora-dir", type=str, required=True)
    ap.add_argument("--ao-meta", type=str, required=True, help="the checkpoint's meta.json")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument(
        "--stage", type=str, default="all", choices=["merge", "sidecar", "parquet", "all"]
    )
    ap.add_argument("--layers", type=str, required=True, help="csv of the AO's trained layers")
    ap.add_argument(
        "--global-scale",
        type=float,
        default=0.0,
        help="the run's frozen global injection scale (from its cell record)",
    )
    ap.add_argument("--pool", type=str, default="")
    ap.add_argument("--arout-dir", type=str, default="", help="AR bank (inject source)")
    ap.add_argument("--gt-arout-dir", type=str, default="", help="GT bank (gold source)")
    ap.add_argument("--force-parquet", action="store_true")
    ap.add_argument("--n-rows", type=int, default=100_000)
    ap.add_argument("--gate-rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-sidecar", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = json.loads(Path(args.ao_meta).read_text()).get("config", {})

    if args.stage in ("merge", "all"):
        stage_merge(args, out)
    if args.stage in ("sidecar", "all"):
        stage_sidecar(args, out, run_cfg)
    if args.stage == "parquet":
        stage_parquet(args, out, run_cfg)
    (out / "prep.json").write_text(
        json.dumps(
            {"args": vars(args), "run_cfg_keys": sorted(run_cfg), "git_commit": _git_commit()},
            indent=2,
        )
    )
    print("[prep] done", flush=True)


if __name__ == "__main__":
    main()
