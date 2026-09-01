"""Slurm entry for one AO-ladder run (single-node DDP via mp.spawn) — see ao_ladder.md.

The trainer is ``gt_train`` verbatim; the injected vector is the crop's precomputed AR
reconstruction, ``raw_scaled`` with the ONE frozen global constant. Fast kernels are MANDATORY:
this entry hard-fails if the qwen3_5 fla binding (or flash-attn) is missing — no slow fallback.

    # long job — run in tmux, tee to logs/:
        uv run python scripts/ao/ao_train_cluster.py \
            --run-name ao.asst.alldata-b512.s0 \
            --ar-run ar.asst.on.mlayer.lc.alldata.crop32.b512.s0 --resume

Resume: ``--resume`` picks up ``$OLA_ROOT/ao_checkpoints/<run>/resume`` (written every
validation and on SIGUSR1) — exact continuation: LoRA + optimizer state + sampler skip.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from oracle_lens.hf_offline import hf_offline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"
WANDB_PROJECT = "ola"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)




def assert_fast_kernels() -> None:
    """User directive: fast kernels or no training — never the pure-torch fallback."""
    import importlib.util

    # flash-attn 2.x has no wheel for torch 2.9 (cu12 wheels stop at 2.8; the 2.8 wheel is ABI-
    # incompatible), no proven sm_100 build (B200 pods), and flash-attn-4's CuTe path does not
    # import. The kernels that actually matter for Qwen3.6 are the fla gated-delta +
    # causal_conv1d ones checked below — this model is hybrid-Mamba, and THOSE are the
    # pure-torch fallback the no-slow-path rule targets. Attention itself is a small share here
    # (AO crops are <= 64 tokens), so sdpa is allowed when OLA_ATTN_IMPL says so; a bare
    # missing flash_attn with no override still fails.
    if importlib.util.find_spec("flash_attn") is None and "OLA_ATTN_IMPL" not in os.environ:
        raise SystemExit("[ao] flash_attn is NOT installed and OLA_ATTN_IMPL is unset")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    if getattr(m, "chunk_gated_delta_rule", None) is None:
        raise SystemExit(
            "[ao] transformers qwen3_5 did not bind fla gated-delta kernels "
            "(install flash-linear-attention + tilelang) — refusing the pure-torch fallback"
        )
    if getattr(m, "causal_conv1d_fn", None) is None:
        raise SystemExit("[ao] causal_conv1d not bound — install causal-conv1d")


def slurm_topology(local_gpus: int) -> tuple[int, int, int, str]:
    """(global_world, local_gpus, node_rank, master_addr) from the Slurm allocation.

    Multi-node: one task per node (``--ntasks-per-node=1``), each spawning ``local_gpus`` ranks.
    MASTER_ADDR is resolved to an IPv4 literal — hostname resolution across nodes has bitten this
    cluster before (runbook: 'IPv4 master addr' post-mortem).
    """
    import socket
    import subprocess

    nodes = int(os.environ.get("SLURM_NNODES", "1"))
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    # Explicit IPv4, never the string "localhost": it can resolve to ::1 and wedge the TCPStore
    # rendezvous (the same trap env.sh documents for the AR trainer). MASTER_ADDR wins if set,
    # which is how two concurrent single-node runs share a box.
    addr = os.environ.get("MASTER_ADDR") or "127.0.0.1"
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
    if nodes > 1 and nodelist:
        first = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, check=True
        ).stdout.split()[0]
        addr = socket.gethostbyname(first)
    return local_gpus * nodes, local_gpus, node_rank, addr


def _build_spaces(cfg: Any, root: Path) -> list[Any]:
    """One frozen ruler per cfg.layers entry: Whitener for wht_unit, JSpace for j_unit.

    Both satisfy inject.MetricSpace, so inject_gt takes either — the arms differ only in WHICH
    map, which is the whole point of the AO_white vs AO_J comparison.
    """
    wdir = root / "whitening_v2"
    if cfg.transform == "wht_unit":
        # _load_whiteners_cached, NOT load_whitener per layer: the latter runs an fp64 eigh of a
        # 5120^2 covariance EACH time (~10-17 CPU-min for 17 layers, per rank). The cached loader
        # reads the prebuilt {mu, W} matrices in seconds.
        from oracle_lens.pipeline.jobs.train import _load_whiteners_cached
        from oracle_lens.pipeline.multilayer import LAYERS as _WL

        cached = _load_whiteners_cached(wdir, 0.1)
        by_layer = dict(zip(_WL, cached, strict=True))
        return [by_layer[ly].to("cuda") for ly in cfg.layers]
    from oracle_lens.pipeline.jspace import load_jspaces

    jsp = load_jspaces(
        cfg.extra.get("jspace_repo") or "neuronpedia/jacobian-lens",
        cfg.extra.get("jspace_file")
        or "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
        wdir,
        tuple(cfg.layers),
    )
    missing = [ly for ly in cfg.layers if ly not in jsp]
    if missing:
        raise SystemExit(f"[ao] j_unit: no Jacobian for layers {missing} — drop them from layers")
    return [jsp[ly].to("cuda") for ly in cfg.layers]


def _worker(  # type: ignore[no-untyped-def]
    local_rank: int,
    world_size: int,
    local_gpus: int,
    node_rank: int,
    master_addr: str,
    config_json: str,
    resume_json: str,
    train_data,
    val_data,
) -> None:
    import torch
    import wandb
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.soft_token_sft import (
        SoftTokenConfig,
        install_preempt_handler,
        train_soft_token,
    )
    from oracle_lens.pipeline.verbalizer import renderer_for

    install_preempt_handler()
    cfg = SoftTokenConfig.from_dict(json.loads(config_json))
    resume_state: dict[str, Any] | None = json.loads(resume_json) if resume_json else None
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = load_causal_lm(
        MODEL_ID, dtype=torch.bfloat16, device="cuda",
        attn_implementation=os.environ.get("OLA_ATTN_IMPL", "flash_attention_2"),
    )
    # AO sequences are ~100 tokens, so grad-ckpt OFF may win (bench decides); the 4k-token OOM
    # lesson was for the AR's output_hidden_states path, not this one.
    if cfg.extra.get("grad_ckpt", "on") == "on":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if cfg.extra.get("compile_blocks", "off") == "on":
        # The AR ladder's speed lever: per-block torch.compile BEFORE the LoRA wrap fuses the
        # r=16 adapter GEMMs + norms (launch-bound at ~120-token seqs). (layer x length)-pure
        # batches make shapes static (~12 variants), so dynamic=False holds.
        import torch.nn as nn

        for module in model.modules():
            layers = getattr(module, "layers", None)
            if isinstance(layers, nn.ModuleList) and hasattr(module, "norm"):
                for i in range(len(layers)):
                    layers[i] = torch.compile(layers[i], dynamic=False)  # type: ignore[assignment]
                if local_rank == 0:
                    print(f"[ao] compiled {len(layers)} decoder blocks", flush=True)
                break
    peft_model: Any
    if resume_state is not None:
        peft_model = PeftModel.from_pretrained(model, resume_state["lora_path"], is_trainable=True)
        if local_rank == 0:
            print(f"[ao] resumed LoRA from {resume_state['lora_path']}", flush=True)
    elif cfg.init_from:
        peft_model = PeftModel.from_pretrained(
            model, str(ola_root() / "ao_checkpoints" / cfg.init_from / "lora"), is_trainable=True
        )
    else:
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
        peft_model = get_peft_model(model, lora)
    for p_ in peft_model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()

    # registry dispatch (verbalizer.renderer_for) — the hand-rolled dict here was the exact
    # footgun PROMPT_RENDERERS documents (it silently fell back to the continuation prompt
    # for any unknown kind)
    render = renderer_for(cfg.extra.get("prompt", "gt"))
    prompts = [render(tokenizer, layer=ly) for ly in cfg.layers]
    if local_rank == 0:
        # startup audit: per-layer prompt shape + a live injection norm (gate D1/D2 material)
        print(
            f"[ao] prompts: lens={[len(p.input_ids) for p in prompts]} "
            f"slots={[p.slot for p in prompts]}",
            flush=True,
        )
        ex0 = train_data[0]
        v0 = float(ex0["inject_vec"].norm())
        ly0 = cfg.layers[int(ex0["layer_idx"])]
        s = cfg.scale_for(ly0)
        print(f"[ao] inject example0: L{ly0} |ar_out|={v0:.1f} scale={s} |v|={s * v0:.1f}")

    torch.cuda.reset_peak_memory_stats()
    # Per-layer ruler for the metric-space arms. Built HERE (per rank, after cfg is
    # deserialized) rather than shipped through spawn: a Whitener/JSpace pair is ~100 MB/layer
    # and the loaders are cached on disk.
    spaces = None
    if cfg.transform in ("wht_unit", "j_unit"):
        spaces = _build_spaces(cfg, Path(os.environ["OLA_ROOT"]))
        if local_rank == 0:
            print(f"[ao] {cfg.transform}: {len(spaces)} per-layer rulers "
                  f"for layers {cfg.layers}", flush=True)

    summary = train_soft_token(
        peft_model,
        prompts,
        train_data,
        val_data,
        cfg,
        spaces=spaces,
        pad_id=int(tokenizer.pad_token_id or 0),
        wandb_project=WANDB_PROJECT,
        local_rank=local_rank,
        world_size=world_size,
        local_gpus=local_gpus,
        node_rank=node_rank,
        master_addr=master_addr,
        ckpt_dir=ola_root() / "ao_checkpoints" / cfg.run_name,
        resume_state=resume_state,
    )
    summary["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if local_rank != 0 or node_rank != 0:
        return
    print(json.dumps(summary), flush=True)
    if wandb.run is not None:
        wandb.summary.update(summary)
        wandb.finish()
    ckpt_dir = ola_root() / "ao_checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir / "lora"))
    runs_dir = ola_root() / "ao_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{cfg.run_name}.json").write_text(
        json.dumps({"config": cfg.to_dict(), "summary": summary}, indent=2)
    )


def resolve_scale(arout, alpha: float, *, path: Path, fit: bool) -> float:  # type: ignore[no-untyped-def]
    """Load the ladder's frozen global scale, or fit it ONCE (median ‖ar_out‖ across all layers)."""
    import torch

    from oracle_lens.pipeline.inject import fit_scale

    if path.exists():
        rec = json.loads(path.read_text())
        print(f"[ao] frozen global scale: {rec['scale']} (from {path})", flush=True)
        return float(rec["scale"])
    if not fit:
        raise SystemExit(f"[ao] no frozen scale at {path} — run once with --fit-scale")
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(arout), generator=gen)[:4096]
    rows = torch.stack([torch.as_tensor(arout[int(i)]) for i in idx.tolist()]).float()
    per_layer = rows.norm(dim=-1).median(dim=0).values  # [17]
    scale = fit_scale(rows.reshape(-1, rows.shape[-1]), target_norm=alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scale": scale,
                "alpha": alpha,
                "per_layer_median_norm": [round(float(x), 2) for x in per_layer],
                "n_sample": len(rows),
            },
            indent=2,
        )
    )
    print(f"[ao] FIT frozen global scale = {scale:.6f} (target_norm {alpha})", flush=True)
    print(f"[ao] per-layer median |ar_out|: {[round(float(x), 1) for x in per_layer]}", flush=True)
    return float(scale)


def main() -> None:
    hf_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--ar-run", type=str, required=True)
    ap.add_argument("--pool", type=str, default="ao_pool/pool_v1.safetensors")
    ap.add_argument("--eval-pool", type=str, default="ao_pool/eval_pool_v1.safetensors")
    ap.add_argument("--arout-dir", type=str, default="", help="default ao_arout/<ar_run>")
    ap.add_argument("--eval-arout-dir", type=str, default="", help="default = --arout-dir")
    ap.add_argument("--n-crops", type=int, default=0, help="0 = all precomputed crops")
    ap.add_argument(
        "--n-val-crops",
        type=int,
        default=600,
        help="val crops (x17 layers = examples per validation; full eval pool would be 217k)",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-every-steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument(
        "--val-micro-batch",
        type=int,
        default=16,
        help="val batch size; must be <= the smallest (layer,length) group or NO val batches form",
    )
    ap.add_argument("--transform", default="scaled",
                    choices=["scaled", "scaled_clip", "unit", "wht_unit", "j_unit"],
                    help="injection arm. scaled = AO_raw (ladder default, one frozen global "
                         "scale); wht_unit = AO_white; j_unit = AO_J. The metric-space arms "
                         "unit-normalise then scale by alpha, so alpha IS the injected norm and "
                         "is identical across them — only the direction differs.")
    ap.add_argument("--alpha", type=float, default=8000.0)
    ap.add_argument("--per-layer-scale", action="store_true",
                    help="transform=scaled: fit ONE scalar PER LAYER so median ||s*AR(p)|| == "
                         "alpha at that layer, instead of the ladder's single global constant. "
                         "A scalar preserves the AR output's DIRECTION exactly (unlike whitening "
                         "or the J map, which rotate the axes) and only corrects the magnitude. "
                         "Measured 2026-08-03: ||AR(p)||/||h|| runs 0.47x at L0 to 2.42x at L63 "
                         "— a 5x swing that no single global scale can fix, which is what "
                         "gt_train's own docstring warns about.")
    ap.add_argument("--scramble-seed", type=int, default=0,
                    help="CONTROL for the alpha sweep: >0 permutes the AR-output rows so each "
                         "crop is injected with ANOTHER crop's activation. Train loss alone "
                         "cannot pick alpha — a low loss can mean the AO learned an "
                         "unconditional phrase prior and ignored the injection, which is "
                         "exactly what too-small alpha looks like. The usable alpha is the one "
                         "maximising (scrambled loss - real loss).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-len", type=int, default=0,
        help="0 = all pool lengths. >0 = train/val ONLY on crops of length <= this, SUBSET "
        "after the seeded layer-pick draw so a full-length k-sliced arout stays valid "
        "(u64max32 cell, user 2026-08-19)")
    ap.add_argument(
        "--adopt-arout-picks", action="store_true",
        help="TRAIN dataset takes its layer picks from the arout shards' stored layer_pick "
        "table instead of re-drawing them (required for MERGED replay-mix pools, where crop "
        "indices shift vs the source pools and a seeded re-draw can never match). Val "
        "datasets keep the strict seeded-equality check.")
    ap.add_argument("--n-gpu", type=int, default=0, help="0 = all visible")
    ap.add_argument("--grad-ckpt", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--compile-blocks", type=str, default="off", choices=["on", "off"])
    ap.add_argument(
        "--prompt",
        type=str,
        default="explain",  # ladder default (user 2026-07-29): "explains what this encodes"
        # continuation_raw (u64, Agam 2026-08-08): tag-free — <activation> enclosure, target =
        # raw span + EOS (AOLadderDataset wrap_tags=False). explain_raw (register SFT, Agam
        # 2026-08-11): same tag-free target contract, explain wording.
        choices=["gt", "min", "explain", "continuation_raw", "explain_raw"],
    )
    ap.add_argument(
        "--layers-per-crop",
        type=int,
        default=2,
        help="k seeded layers per crop (0 = all 17). k=2 is the ladder standard (user 2026-07-29) "
        "and every rung must match it or the curves are not comparable. It also keeps "
        "repeat_factor = crops_per_window x k at 2.0, under audit_diversity's 2.5 ceiling — the "
        "old default of 4 put it at 4.0, which the audit rejects at startup.",
    )
    ap.add_argument(
        "--val-source",
        type=str,
        default="pool",
        choices=["pool", "pairs"],
        help="pool = in-distribution reserved conversations (default); pairs = the FVE eval pool",
    )
    ap.add_argument("--split-seed", type=int, default=1234, help="conversation split seed")
    ap.add_argument(
        "--target-prefix",
        type=str,
        default="",
        help="literal text inserted as the FIRST supervised tokens before the span (dot-variant: "
        "'.'). Applied to train AND val targets; recorded in extra.target_prefix.",
    )
    ap.add_argument(
        "--val-pool",
        type=str,
        default="",
        help="validate on THIS pool's conversation-split val set instead of --pool's. The "
        "extension run trains on the delta pool but must keep the PARENT run's val set "
        "(pool_v2 + split_seed 1234) — validating on delta's own held-out conversations would "
        "move the val distribution mid-curve and fake a jump at the continuation point.",
    )
    ap.add_argument(
        "--val-arout-dir",
        type=str,
        default="",
        help="arout dir matching --val-pool (required with it)",
    )
    ap.add_argument(
        "--match-config",
        type=str,
        default="",
        help="run name whose recorded config this launch must MATCH (the ladder's comparability "
        "contract). Loads ao_runs/<name>.json and hard-fails on any mismatch in the fields that "
        "make rungs comparable — never rely on a human retyping flags per rung.",
    )
    ap.add_argument(
        "--max-repeat", type=float, default=2.5,
        help="ceiling on how many times each text is a target per epoch "
        "(crops_per_window * layers_per_crop). Default 2.5 permits k=2 at one crop per window. "
        "RAISING THIS ACCEPTS A KNOWN RISK: layer multiplicity is one of three repetition sources "
        "that previously drove val-CE inflection via text memorization, so a run above the "
        "default must be read with that in mind (watch val CE for an inflection, not just its "
        "floor). Set to 4.0 for k=4 on a 1-crop/window pool (Agam, 2026-08-04).",
    )
    ap.add_argument("--fit-scale", action="store_true")
    ap.add_argument(
        "--scale-path",
        default="ao_runs/global_scale.json",
        help="OLA_ROOT-relative frozen-scale record; ablation-arm AOs (different AR output"
        " norms) need a per-arm path so they never clobber the ladder's constant",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--init-from",
        type=str,
        default="",
        help="branch off another run's adapter, e.g. 'ao.asst...k2.g8/step1600' (weights only — "
        "no optimizer state, unlike --resume)",
    )
    ap.add_argument(
        "--exclude-idx-file",
        type=str,
        default="",
        help="exact-remainder continuation: .npy of int64 example indices (OLA_ROOT-relative ok) "
        "excluded from the train sampler, so the run trains only the parent's UNSEEN remainder. "
        "Use when the parent's (micro_batch, world) geometry cannot be reproduced (u64 on H200); "
        "the consumed set comes from scripts/iolens/u64_consumed_set.py. Pair with --init-from; "
        "mutually exclusive with --skip-batches.",
    )
    ap.add_argument(
        "--skip-batches",
        type=int,
        default=0,
        help="skip the first N per-rank micro-batches of the seeded permutation. Paired with "
        "--init-from at the SAME step this replays the exact data suffix the parent run saw, so "
        "an LR branch differs from its control in the LR alone.",
    )
    ap.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="print the exact config + N decoded input/output pairs, then exit (CPU-safe)",
    )
    args = ap.parse_args()

    if not args.dry_run:
        assert_fast_kernels()  # (CUDA-gated, so skipped for a CPU dry run)

    import torch
    import torch.multiprocessing as tmp
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.ao_ladder import (
        AOLadderDataset,
        ao_gt_config,
        conv_split,
        load_arout,
    )
    from oracle_lens.pipeline.ao_pool import AOPool, audit_diversity, pool_fingerprint

    root = ola_root()
    pool = AOPool.load(root / args.pool)
    eval_pool = AOPool.load(root / args.eval_pool)
    arout_dir = root / (args.arout_dir or f"ao_arout/{args.ar_run}")
    fp = pool_fingerprint(pool.ids, pool.keep)
    efp = pool_fingerprint(eval_pool.ids, eval_pool.keep)
    eval_arout_dir = root / args.eval_arout_dir if args.eval_arout_dir else arout_dir
    arout, top = load_arout(arout_dir, split="train", expect_fingerprint=fp)
    if args.scramble_seed:
        # Break the crop<->activation correspondence, keeping the marginal distribution of
        # injected vectors identical. Any loss the AO still achieves is what it could get
        # WITHOUT reading the activation.
        perm = torch.randperm(len(arout), generator=torch.Generator().manual_seed(
            args.scramble_seed))
        arout = arout[perm]
        print(f"[ao] SCRAMBLED control: arout rows permuted (seed {args.scramble_seed})",
              flush=True)
    arout_eval, etop = load_arout(eval_arout_dir, split="eval", expect_fingerprint=efp)
    arout_pick = top.pop("layer_pick", None)
    print(f"[ao] arout: {top} | eval: {etop}", flush=True)
    if arout_pick is not None:
        sm = dict(top["slice_meta"])
        want = {
            "layers_per_crop": args.layers_per_crop,
            "layer_seed": args.seed,
            "split_seed": args.split_seed,
            "n_val_crops": args.n_val_crops,
        }
        # n_universe is informational here: the layer universe is taken FROM the shards
        # (ao_layers) rather than chosen by this run, so it cannot disagree — comparing it would
        # only ever produce a spurious mismatch. Reported below for the record.
        n_universe = sm.pop("n_universe", None)
        if sm != want:
            if args.adopt_arout_picks and sm.get("layers_per_crop") == args.layers_per_crop:
                # merged (replay-mix) dirs: picks come FROM the shards, so the seed provenance
                # recorded there describes the source pools, not this run — k must still agree.
                print(f"[ao] adopt-arout-picks: slice_meta {sm} != args {want} (OK: picks "
                      "are adopted from the shards, not re-drawn)", flush=True)
            else:
                raise SystemExit(
                    f"[ao] k-sliced arout (universe {n_universe}) was precomputed for {sm} "
                    f"but this run wants {want} — "
                    "rerun precompute or match the args"
                )

    # The metric-space / unit arms unit-normalise then multiply by alpha, so alpha IS the
    # injected norm — there is no per-arm scale to fit or freeze. Only the `scaled` family
    # needs the frozen global constant.
    if args.transform in ("wht_unit", "j_unit", "unit") or args.per_layer_scale:
        # per_layer_scale fits its own scalars below, so the frozen GLOBAL constant is moot
        scale = 1.0
        print(f"[ao] {args.transform}: scale-free arm, injected norm = alpha = {args.alpha}",
              flush=True)
    else:
        scale = resolve_scale(
            arout, args.alpha, path=root / args.scale_path, fit=args.fit_scale
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Reserve a tail of the SAME pool as the in-distribution val set: the pairs-derived eval pool
    # is a different corpus, so its CE mixes generalization with domain shift. That pool stays the
    # FVE set (true h lives there) and ao_fve_eval scores it independently.
    if args.val_pool and args.val_source != "pool":
        raise SystemExit("[ao] --val-pool only makes sense with --val-source pool")
    if args.val_pool and not args.val_arout_dir:
        raise SystemExit("[ao] --val-pool requires --val-arout-dir (the matching AR outputs)")
    if args.val_pool:
        # Extension mode: train on EVERY crop of --pool (its text is disjoint from the val pool
        # by the delta build's crop-hash exclusion, so nothing leaks), validate on the val pool's
        # conversation split. Same split_seed + same layer_seed as the parent run means the val
        # examples are IDENTICAL to the parent's — the continued curve stays on the same ruler.
        train_idx = None
        val_pool = AOPool.load(root / args.val_pool)
        vfp = pool_fingerprint(val_pool.ids, val_pool.keep)
        val_arout, vtop = load_arout(
            root / args.val_arout_dir, split="train", expect_fingerprint=vfp
        )
        val_arout_pick = vtop.pop("layer_pick", None)
        # The val dataset is built with the TRAIN arout's ao_layers; a full-storage val arout
        # precomputed under a different --layer-min would silently inject the wrong layers
        # (the pick-equality check only covers the k-sliced path).
        if list(vtop.get("ao_layers") or []) != list(top.get("ao_layers") or []):
            raise SystemExit(
                f"[ao] --val-arout-dir layer universe {vtop.get('ao_layers')} != train "
                f"arout's {top.get('ao_layers')} — precompute the val arout with the same "
                "--layer-min"
            )
        _vtrain, val_idx = conv_split(
            val_pool,
            n_val_crops=args.n_val_crops,
            seed=args.split_seed,
            n_avail=min(len(val_arout), val_pool.n_crops()),
        )
        print(
            f"[ao] EXTENSION val: {args.val_pool} split_seed={args.split_seed} "
            f"({len(val_idx):,} val crops; arout {vtop})",
            flush=True,
        )
    else:
        val_pool, val_arout = pool, arout
        n_avail = min(len(arout), pool.n_crops())
        train_idx, val_idx = conv_split(
            pool, n_val_crops=args.n_val_crops, seed=args.split_seed, n_avail=n_avail
        )
    # Row->layer semantics come from the AROUT SHARDS. The AR drops layer 0 and --layer-min 20
    # restricts further, so assuming the canonical 17-entry LAYERS names the wrong layer in the
    # prompt AND lets picks index off the end of the array.
    ao_layers = list(top.get("ao_layers") or []) or None
    if ao_layers:
        print(f"[ao] arout declares {len(ao_layers)} layers: {ao_layers}", flush=True)
    # j_unit has no Jacobian for the target block, so L63 is dropped from the DATA as well as
    # the config — otherwise crops would be drawn at a layer with no ruler. The drawable set is
    # built from the shards' own universe so it composes with a restricted ao_layers.
    from oracle_lens.pipeline.multilayer import LAYERS as _ALL_LAYERS

    layer_sel = (
        tuple(ly for ly in (ao_layers or _ALL_LAYERS) if ly != 63)
        if args.transform == "j_unit"
        else None
    )
    if layer_sel is not None and arout_pick is not None:
        raise SystemExit(
            "[ao] --transform j_unit cannot run on a k-sliced arout: its stored picks were "
            "drawn over the FULL layer universe, so they can never match picks drawn over the "
            "L63-excluded set — rerun precompute without slicing (or with L63 excluded)"
        )
    wrap_tags = args.prompt not in ("continuation_raw", "explain_raw")
    # val grouping: layer-pure only when the pool's length ladder is wide (u64: 11 layers x 64
    # lengths = 704 (layer,length) groups starve the partial-batch-dropping val sampler; see
    # AOLadderDataset.group_by_length). Train keeps full (layer,length)-purity (zero padding).
    # derive the val grouping from the pool the VAL SET actually comes from — the s1 ext
    # segment trains on a single-length pool (u64ext32) but validates on the 64-length u64
    # pool; keying on the TRAIN pool turned (layer,length) grouping on and starved the val
    # sampler to ZERO batches (2026-08-21).
    _val_len_pool = val_pool if args.val_pool else pool
    val_by_length = len(_val_len_pool.lengths) <= 8
    train_data = AOLadderDataset(
        pool,
        arout,
        tokenizer=tokenizer,
        ao_layers=ao_layers,
        crop_sel=train_idx,
        n_crops=args.n_crops or None,
        layers_per_crop=args.layers_per_crop,
        layer_seed=args.seed,
        target_prefix=args.target_prefix,
        arout_pick=arout_pick,
        layer_sel=layer_sel,
        wrap_tags=wrap_tags,
        max_len=args.max_len or None,
        adopt_arout_picks=args.adopt_arout_picks,
    )
    if args.val_source == "pool":
        # whole conversations reserved — no window/prefix overlap with train. Same k as train so
        # a validation stays ~2 min (all 17 layers would make it a 4x longer stall every eval).
        val_data = AOLadderDataset(
            val_pool,
            val_arout,
            tokenizer=tokenizer,
            ao_layers=ao_layers,
            crop_sel=val_idx,
            layers_per_crop=args.layers_per_crop,
            layer_seed=args.seed + 1,
            target_prefix=args.target_prefix,
            arout_pick=val_arout_pick if args.val_pool else arout_pick,
            layer_sel=layer_sel,
            wrap_tags=wrap_tags,
            group_by_length=val_by_length,
            max_len=args.max_len or None,
        )
    else:
        val_data = AOLadderDataset(
            eval_pool, arout_eval, tokenizer=tokenizer, n_crops=args.n_val_crops or None,
            ao_layers=list(etop.get("ao_layers") or []) or None,
            layer_sel=layer_sel,
            wrap_tags=wrap_tags,
            group_by_length=val_by_length,
            max_len=args.max_len or None,
        )
    audit = audit_diversity(
        pool, layers_per_crop=args.layers_per_crop, max_repeat=args.max_repeat
    )
    if args.max_repeat > 2.5:
        print(
            f"[ao] WARNING diversity ceiling RAISED to {args.max_repeat} (default 2.5): each text "
            f"is a target {audit.get('repeat_factor')}x per epoch. Layer multiplicity previously "
            f"caused val-CE inflection via memorization — watch for an inflection in val CE, not "
            f"just its floor.", flush=True,
        )
    print(f"[ao] diversity audit PASSED: {audit}", flush=True)
    print(
        f"[ao] {len(train_data):,} train examples ({train_data.n_crops:,} crops x k="
        f"{train_data.k}) / {len(val_data):,} val examples "
        f"({args.val_source} source, {val_data.n_crops:,} crops x {val_data.k})",
        flush=True,
    )

    ckpt_dir = root / "ao_checkpoints" / args.run_name
    resume_state: dict[str, Any] | None = None
    skip_batches = 0
    if args.resume and (ckpt_dir / "resume" / "state.json").exists():
        state = json.loads((ckpt_dir / "resume" / "state.json").read_text())
        n_gpu = args.n_gpu or torch.cuda.device_count()
        global_world = slurm_topology(n_gpu)[0]
        if int(state.get("world_size", global_world)) != global_world:
            raise SystemExit(
                f"[ao] resume world_size {state.get('world_size')} != current {global_world} — "
                "sampler skip is only exact at the same world size"
            )
        resume_state = {
            "micro_steps": int(state["micro_steps"]),
            "tokens_span": int(state["tokens_span"]),
            "tokens_sup": int(state["tokens_sup"]),
            "lora_path": str(ckpt_dir / "resume" / "lora"),
            "optimizer_path": str(ckpt_dir / "resume" / "optimizer.pt"),
        }
        skip_batches = int(state["micro_steps"])
        print(f"[ao] RESUME at micro-step {skip_batches}: {state}", flush=True)
    elif args.init_from:
        # Branch: adopt another run's weights and replay the data suffix from the same point.
        # Fresh optimizer state (Adam moments are not carried across a branch) and a fresh
        # warmup, so the arm is "same weights, same remaining data, different LR".
        src = root / "ao_checkpoints" / args.init_from / "lora"
        if not src.exists():
            raise SystemExit(f"--init-from: no adapter at {src}")
        skip_batches = args.skip_batches
        print(f"[ao] BRANCH from {args.init_from} (skip_batches={skip_batches})", flush=True)
    elif args.skip_batches:
        raise SystemExit("--skip-batches without --init-from would silently discard data")

    cfg = ao_gt_config(
        run_name=args.run_name,
        ao_layers=ao_layers,
        ar_run=args.ar_run,
        scale=scale,
        pool_path=args.pool,
        n_examples=len(train_data),
        alpha=args.alpha,
        lora_r=args.lora_r,
        lr=args.lr,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        eval_every_steps=args.eval_every_steps,
        seed=args.seed,
        skip_batches=skip_batches,
        init_from=args.init_from,
        target_prefix=args.target_prefix,
        crop_lengths=pool.lengths,
    )
    cfg.transform = args.transform
    if args.per_layer_scale and args.transform in ("scaled", "scaled_clip"):
        if arout_pick is not None:
            raise SystemExit(
                "[ao] --per-layer-scale needs a full (non k-sliced) arout: sliced rows are "
                "[k, d] in pick order, so a per-layer median would read the wrong layers"
            )
        import torch as _t

        _g = _t.Generator().manual_seed(args.seed)
        _idx = _t.randperm(len(arout), generator=_g)[:4096]
        _rows = _t.stack([_t.as_tensor(arout[int(i)]) for i in _idx.tolist()]).float()
        cfg.scales = {}
        for _li, _ly in enumerate(cfg.layers):
            _med = float(_rows[:, _li, :].norm(dim=-1).median())
            cfg.scales[str(_ly)] = args.alpha / max(_med, 1e-9)
        cfg.extra["scale_mode"] = "per_layer_fit"
        print("[ao] per-layer scales (median ||s*AR(p)|| = alpha): "
              + ", ".join(f"L{ly}={cfg.scales[str(ly)]:.3f}" for ly in cfg.layers), flush=True)
    if layer_sel is not None:
        cfg.layers = layer_sel
    cfg.epochs = args.epochs
    cfg.warmup_steps = args.warmup_steps
    cfg.val_micro_batch = args.val_micro_batch
    cfg.num_workers = args.num_workers
    cfg.extra["grad_ckpt"] = args.grad_ckpt
    cfg.extra["compile_blocks"] = args.compile_blocks
    cfg.extra["prompt"] = args.prompt
    cfg.extra["layers_per_crop"] = str(args.layers_per_crop)
    cfg.extra["val_source"] = args.val_source
    cfg.extra["max_len_filter"] = str(args.max_len or "")
    cfg.extra["adopt_arout_picks"] = str(bool(args.adopt_arout_picks))
    cfg.extra["split_seed"] = str(args.split_seed)
    cfg.extra["arout_dir"] = str(arout_dir)
    cfg.extra["val_pool"] = args.val_pool
    cfg.extra["val_arout_dir"] = args.val_arout_dir

    if args.exclude_idx_file:
        # Exact-remainder continuation: audit the exclusion against THIS dataset before any GPU
        # work — a wrong consumed set would silently train the wrong suffix.
        from oracle_lens.pipeline.soft_token_sft import load_exclude_mask

        if args.skip_batches:
            raise SystemExit(
                "[ao] --exclude-idx-file and --skip-batches are mutually exclusive: pick ONE "
                "continuation mechanism"
            )
        cfg.exclude_idx_file = args.exclude_idx_file
        _mask = load_exclude_mask(cfg, len(train_data))
        assert _mask is not None
        _n_excl = int(_mask.sum())
        print(
            f"[ao] EXACT-REMAINDER: {_n_excl:,} consumed examples excluded -> "
            f"{len(train_data) - _n_excl:,} of {len(train_data):,} remain",
            flush=True,
        )
        cfg.extra["exclude_idx_file"] = args.exclude_idx_file
        cfg.extra["exclude_n"] = str(_n_excl)

    if args.match_config:
        # The scaling curves are only comparable if every rung trained under the SAME recipe.
        # Compare this launch against the reference run's RECORDED config — the artifact, not
        # someone's memory of the flags — and refuse to start on any drift.
        ref_path = root / "ao_runs" / f"{args.match_config}.json"
        ref = json.loads(ref_path.read_text())["config"]
        cur: dict[str, Any] = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg.__dict__)
        fields = [
            "transform",
            "alpha",
            "lr",
            "lr_sched",
            "warmup_steps",
            "micro_batch",
            "grad_accum",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "min_len",
            "max_len",
            "seed",
            "epochs",
            "eval_every_steps",
            "val_micro_batch",
            "clip_mult",
            "layers",
        ]
        extra_fields = [
            "prompt",
            "layers_per_crop",
            "grad_ckpt",
            "compile_blocks",
            "global_scale",
            "pool",
            "split_seed",
            "val_source",
            "target_prefix",  # missing in pre-dot refs -> treated as "" below
        ]
        bad = []
        for k in fields:
            a, b = ref.get(k), cur.get(k)
            if a != b:
                bad.append(f"{k}: ref={a!r} this={b!r}")
        rex, cex = ref.get("extra", {}), cur.get("extra", {})
        for k in extra_fields:
            a, b = rex.get(k) or "", cex.get(k) or ""
            if a != b:
                bad.append(f"extra.{k}: ref={a!r} this={b!r}")
        if bad:
            raise SystemExit(
                f"[ao] --match-config {args.match_config}: this launch DIFFERS from the "
                f"reference recipe — the rung would not be comparable:\n  " + "\n  ".join(bad)
            )
        # World size is deliberately NOT compared: the sampler deals rank-chunks WITHIN each
        # global batch (LayerBucketBatchSampler, r::world slice), so effective batch =
        # micro_batch x grad_accum regardless of world. Verified from the reference run's own
        # arithmetic: 2,266,906 consumed examples / 2,204 optimizer steps = 1,028 = micro 128 x
        # accum 8 — no world factor. A 4-GPU run with identical flags consumes the identical
        # batch sequence, just ~2x slower; only resume requires a consistent world per run.
        print(
            f"[ao] config MATCHES reference {args.match_config} on all compared fields", flush=True
        )

    if args.dry_run:
        from oracle_lens.pipeline.multilayer import LAYERS
        from oracle_lens.pipeline.verbalizer import renderer_for

        print("\n===== EXACT RUN CONFIG =====")
        print(json.dumps(cfg.to_dict(), indent=2))
        eff = cfg.micro_batch * max(1, cfg.grad_accum // (args.n_gpu or 1)) * (args.n_gpu or 1)
        print(
            f"\n===== DERIVED =====\n"
            f"  GPUs {args.n_gpu} -> grad_accum {max(1, cfg.grad_accum // (args.n_gpu or 1))}, "
            f"EFFECTIVE BATCH {eff} examples/optimizer-step\n"
            f"  train {len(train_data):,} ex ({train_data.n_crops:,} crops x k={train_data.k})\n"
            f"  val   {len(val_data):,} ex ({val_data.n_crops:,} crops, {args.val_source})\n"
            f"  optimizer steps/epoch ~ {len(train_data) // eff:,}; "
            f"checkpoint every {cfg.eval_every_steps} micro-steps\n"
            f"  arout {top} | frozen scale {scale}"
        )
        render = renderer_for(args.prompt)
        print("\n===== EXAMPLE INPUT -> OUTPUT PAIRS =====")
        seen: set[int] = set()
        shown = 0
        for i in range(len(train_data)):
            ex = train_data[i]
            n = int(ex["span_len"])
            if n in seen:
                continue
            seen.add(n)
            shown += 1
            ly = (ao_layers or list(LAYERS))[int(ex["layer_idx"])]
            pr = render(tokenizer, layer=ly)
            v = ex["inject_vec"]
            print(f"\n--- pair {shown}: crop length {n}, layer {ly} ---")
            print(f"INPUT  (prompt, loss masked; slot {pr.slot} <- {scale:.3f} * AR(span)[L{ly}]):")
            print(f"   {tokenizer.decode(pr.input_ids)!r}")
            print(
                f"   injected soft token: |AR(span)| = {float(v.norm()):.1f} -> "
                f"|scale*AR| = {scale * float(v.norm()):.1f}"
            )
            print(f"OUTPUT (target, loss on all {len(ex['target_ids'])} tokens):")
            print(f"   {tokenizer.decode(ex['target_ids'].tolist())!r}")
            if shown >= args.dry_run:
                break
        print("\n[dry-run] no training started")
        return

    n_gpu = args.n_gpu or torch.cuda.device_count()
    world, local_gpus, node_rank, master_addr = slurm_topology(n_gpu)
    if world != n_gpu:
        print(
            f"[ao] multi-node: {world} ranks = {local_gpus} GPUs x "
            f"{world // local_gpus} nodes (node_rank {node_rank}, master {master_addr})",
            flush=True,
        )
    config_json = json.dumps(cfg.to_dict())
    resume_json = json.dumps(resume_state) if resume_state else ""
    if world > 1:
        train_data.share_memory_()
        val_data.share_memory_()
        ctx = tmp.spawn(  # type: ignore[attr-defined,no-untyped-call]
            _worker,
            args=(
                world,
                local_gpus,
                node_rank,
                master_addr,
                config_json,
                resume_json,
                train_data,
                val_data,
            ),
            nprocs=n_gpu,
            join=False,
        )

        # Slurm/srun deliver SIGUSR1 to THIS parent (the task leader — `uv run` execs python);
        # the spawned ranks never see it. Forward so their collective save-and-exit fires.
        import signal as _signal

        def _fwd_usr1(signum: int, frame: Any) -> None:
            print("[ao] parent got SIGUSR1 — forwarding to ranks", flush=True)
            import contextlib

            for pid in ctx.pids():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, _signal.SIGUSR1)

        _signal.signal(_signal.SIGUSR1, _fwd_usr1)
        while not ctx.join(timeout=5):
            pass
    else:
        _worker(0, 1, 1, 0, "localhost", config_json, resume_json, train_data, val_data)


if __name__ == "__main__":
    main()
