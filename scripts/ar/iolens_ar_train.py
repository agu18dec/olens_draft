"""Bare-metal entry for one multi-layer AR run (single-node DDP via mp.spawn) — iolens program.

The trainer body is ``ola_modal.py::{train_ml_recon,_ml_recon_worker}`` adapted for a standalone
GPU box (no Slurm, no Modal volumes): paths under ``$OLA_ROOT``, kernels asserted up front,
effective batch asserted against the protocol, milestone checkpoints + lightweight resume via
``multilayer_reconstructor`` / ``ola.resume``. The checkpoint series of ONE constant-LR run is
the AR scaling curve (x = exact all-reduced span tokens).

    CUDA_VISIBLE_DEVICES=0,1,2 MASTER_PORT=29510 PYTHONUNBUFFERED=1 \
        uv run python scripts/ar/iolens_ar_train.py \
            --run-name ar.iolens.chat.mlayer.lc.s0 \
            --pairs-dir ml_pairs_iolens_chat --whitener-prefix whitening_iolens_chat \
            --n-gpu 3 --span-law crop_uniform --pad-width 32 --compile-blocks \
            --micro-batch 8 --grad-accum 63 --expected-effective-batch 504 \
            --ckpt-every-steps 2000 --save-every-steps 500 --resume

Resume: ``--resume`` picks up ``$OLA_ROOT/ml_checkpoints/<run>/resume.pt`` (written at every
validation) — exact continuation: LoRA + heads + optimizer + streaming-order fast-forward.
Refused if world size or worker count changed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"
WANDB_PROJECT = "ola"
STREAM_WORKERS = 8  # DataLoader worker processes in streaming mode (fixed: resume replay depends)
# The J ruler of record for the fair loss-space comparison: neuronpedia n=1000 wikitext lens,
# SNAPSHOT-PINNED (the trainer runs HF-offline; prefetch this exact revision into the HF cache).
JSPACE_REPO_DEFAULT = "neuronpedia/jacobian-lens"
JSPACE_FILE_DEFAULT = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
JSPACE_REVISION_DEFAULT = "a4114d7752d11eb546e6cf372213d7e75526d3a1"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def _hf_offline() -> None:
    if os.environ.get("AO_HF_ONLINE") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def assert_fast_kernels() -> None:
    """Fast kernels or no training — never the pure-torch fallback (48/64 layers are linear
    attention; unbound kernels mean multi-second steps — the documented 7-9 s/step failure).

    flash_attn is required unless ``OLA_ATTN_IMPL=sdpa`` is EXPLICITLY set (B200 pod: no FA2
    wheel for torch 2.9+cu12 and unproven sm_100 support; sdpa costs ~0.12% of per-token MACs
    at N<=32 per env.sh's analysis). The fla/causal-conv1d linear-attention kernels are
    non-negotiable either way — they carry 48 of the 64 layers.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is None:
        if os.environ.get("OLA_ATTN_IMPL") != "sdpa":
            raise SystemExit(
                "[iolens-ar] flash_attn is NOT installed — refusing the sdpa fallback "
                "(set OLA_ATTN_IMPL=sdpa explicitly to accept it; same impl for ALL arms "
                "of a comparison)"
            )
        print("[iolens-ar] attention = sdpa (explicit OLA_ATTN_IMPL; no flash_attn)", flush=True)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    if getattr(m, "chunk_gated_delta_rule", None) is None:
        raise SystemExit(
            "[iolens-ar] transformers qwen3_5 did not bind fla gated-delta kernels "
            "(install flash-linear-attention + tilelang) — refusing the pure-torch fallback"
        )
    if getattr(m, "causal_conv1d_fn", None) is None:
        raise SystemExit("[iolens-ar] causal_conv1d not bound — install causal-conv1d")


def effective_batch(micro_batch: int, grad_accum: int, world_size: int) -> int:
    """Spans per optimizer step = mb x ga_eff x ngpus, with the harness's ga scaling rule."""
    ga_eff = max(1, grad_accum // world_size) if world_size > 1 else grad_accum
    return micro_batch * ga_eff * world_size


def load_whiteners_cached(wdir: Path, prefix: str, ridge_c: float) -> list[Any]:
    """Whitening transforms with a disk cache for the expensive part (17 fp64 eigh of 5120^2,
    ~10 CPU-min). First call per (wdir, prefix, ridge) pays once; later loads are seconds."""
    import torch
    from safetensors.torch import load_file, save_file

    from oracle_lens.core.whitening import Whitener, load_whitener
    from oracle_lens.pipeline.multilayer import LAYERS

    cache = wdir / f"wmatrix_{prefix}_ridge{ridge_c}.safetensors"
    if cache.exists():
        t = load_file(str(cache))
        return [Whitener(mu=t[f"mu_L{lyr}"], w=t[f"w_L{lyr}"], ridge_c=ridge_c) for lyr in LAYERS]
    t0 = time.time()
    whiteners = [
        load_whitener(wdir / f"{prefix}_L{lyr}.safetensors", ridge_c=ridge_c) for lyr in LAYERS
    ]
    tensors: dict[str, torch.Tensor] = {}
    for lyr, w in zip(LAYERS, whiteners, strict=True):
        tensors[f"mu_L{lyr}"] = w.mu.cpu()
        tensors[f"w_L{lyr}"] = w.w.cpu()
    save_file(tensors, str(cache))
    print(
        f"[iolens-ar] built + cached {len(whiteners)} whitening matrices "
        f"in {time.time() - t0:.0f}s -> {cache.name}",
        flush=True,
    )
    return whiteners


def build_loss_spaces(cfg: Any, wdir: Path, whiteners: list[Any]) -> tuple[Any, Any]:
    """J/Mixed loss rulers + pure-J val rulers for the jspace/mixed arms (parent process).

    Returns ``(loss_whiteners, jspaces)`` aligned with ``cfg.layer_indices`` (a ``None`` slot for
    layers the J stack does not cover — L63, the Jacobian target). Mutates ``cfg.extra`` with the
    full J provenance (repo/file/pinned revision, μ dir/prefix, covered/excluded layers) — the
    caller must re-serialize ``config_json`` AFTER this. Tensors are moved to shared memory so
    ``mp.spawn`` ranks map, not copy, the ~1.6 GB fp32 stack.

    Non-J arms (whiten/rawcos/unitnorm) with the ``--jspace-*`` flags set still get ``jspaces``
    (pure-J VAL rulers, so every arm logs val_jfve_* off the same ruler) with
    ``loss_whiteners=None`` — the training loss is untouched. Without the flags: (None, None).
    """
    j_in_loss = cfg.loss_space in ("jspace", "mixed")
    if not j_in_loss and not (cfg.jspace_repo and cfg.jspace_file):
        return None, None
    from oracle_lens.pipeline.jspace import MixedSpace, load_jspaces
    from oracle_lens.pipeline.multilayer import LAYERS as ALL_LAYERS

    if j_in_loss and not (cfg.jspace_repo and cfg.jspace_file):
        raise SystemExit(
            f"[iolens-ar] loss_space={cfg.loss_space} needs --jspace-repo/--jspace-file"
        )
    if cfg.loss_space == "mixed" and not (0.0 <= cfg.loss_mix_lambda <= 1.0):
        raise SystemExit(
            f"[iolens-ar] --loss-mix-lambda must be in [0,1], got {cfg.loss_mix_lambda}"
        )
    trained = tuple(ALL_LAYERS[i] for i in cfg.layer_indices)
    spaces = load_jspaces(
        cfg.jspace_repo, cfg.jspace_file, wdir, trained,
        revision=cfg.jspace_revision or None, mu_prefix=cfg.whitener_prefix,
    )
    for s in spaces.values():
        s.mu.share_memory_()  # type: ignore[no-untyped-call]
        s.j.share_memory_()  # type: ignore[no-untyped-call]
    jspaces = [spaces.get(lyr) for lyr in trained]
    covered = [lyr for lyr in trained if lyr in spaces]
    excluded = [lyr for lyr in trained if lyr not in spaces]
    loss_whiteners: list[Any] | None
    if not j_in_loss:
        loss_whiteners = None  # J is a VAL ruler only for this arm
    elif cfg.loss_space == "jspace":
        loss_whiteners = list(jspaces)
    else:
        loss_whiteners = [
            MixedSpace(whitener=whiteners[i], jspace=s, lam=cfg.loss_mix_lambda)
            if s is not None else None
            for i, s in enumerate(jspaces)
        ]
    cfg.extra = dict(cfg.extra or {})
    cfg.extra.update(
        loss_mix_lambda=str(cfg.loss_mix_lambda) if cfg.loss_space == "mixed" else "",
        jspace_repo=cfg.jspace_repo,
        jspace_file=cfg.jspace_file,
        jspace_revision=cfg.jspace_revision or "unresolved",
        jspace_mu_dir=str(wdir),
        jspace_mu_prefix=cfg.whitener_prefix,
        jspace_layers=",".join(str(x) for x in covered),
        jspace_excluded=",".join(str(x) for x in excluded),
    )
    print(
        f"[iolens-ar] loss_space={cfg.loss_space}"
        + (f" λ={cfg.loss_mix_lambda}" if cfg.loss_space == "mixed" else "")
        + f" | J ruler {cfg.jspace_repo}:{cfg.jspace_file}@{cfg.jspace_revision[:10] or '?'}"
        + (" (VAL ruler only)" if not j_in_loss else "")
        + f" | covered {len(covered)}/{len(trained)} layers"
        + (f" (EXCLUDED from loss: {excluded})" if j_in_loss else f" (no J: {excluded})"),
        flush=True,
    )
    return loss_whiteners, jspaces


def _warm_page_cache() -> None:
    """One sequential read of the model shards so spawned ranks don't fight over cold pages."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    snap = Path(snapshot_download(MODEL_ID, allow_patterns=["*.safetensors"]))
    warmed = 0
    for f in sorted(snap.glob("*.safetensors")):
        with open(f, "rb") as fh:
            while fh.read(1 << 26):
                pass
        warmed += f.stat().st_size
    print(f"[iolens-ar] page-cache warm: {warmed / 2**30:.1f} GiB in {time.time() - t0:.0f}s",
          flush=True)


def resume_meta_json(args: Any, root: Path, run_name: str, n_gpu: int) -> str:
    """The worker's resume/curve-continuation payload — ONE definition for both launch paths.

    Returns ``{"examples_prev", "tokens_prev"}`` for a warm start, ``{"resume": true}`` for an
    exact continuation, or ``""`` for a fresh run. Both the streaming and the non-streaming
    launch sites call this; when this logic lived only at the non-streaming site, ``--resume``
    was a silent no-op under ``--stream-dir`` and the run restarted from scratch at chance loss.
    Module-level (not a closure) so the restart path can be tested without a GPU.
    """
    if args.examples_prev or args.tokens_prev:
        return json.dumps({"examples_prev": args.examples_prev, "tokens_prev": args.tokens_prev})
    if not args.resume:
        return ""
    from oracle_lens.pipeline.resume import load_resume_state

    state = load_resume_state(root / "ml_checkpoints" / run_name)
    if state is None:
        # REFUSE rather than fall back to a fresh start. An auto-restart supervisor relaunches
        # with --resume; if the bundle is missing or unreadable, silently training from scratch
        # would overwrite a good run's checkpoints with chance-level weights and look healthy
        # doing it (that near-miss cost 2.48M samples on 2026-08-02). Deliberate fresh runs pass
        # --allow-fresh.
        if not getattr(args, "allow_fresh", False):
            raise SystemExit(
                f"[iolens-ar] --resume but no readable resume.pt under "
                f"{root / 'ml_checkpoints' / run_name} — refusing to silently train from "
                f"scratch. Pass --allow-fresh if that is really what you want."
            )
        print("[iolens-ar] --resume but no resume.pt — fresh start (--allow-fresh)", flush=True)
        return ""
    for key, want in (("world_size", n_gpu), ("num_workers", STREAM_WORKERS)):
        have = state.get(key)
        if have is not None and int(have) != want:
            raise SystemExit(
                f"[iolens-ar] resume {key}={have} != current {want} — replay would be "
                f"inexact; refuse"
            )
    print(
        f"[iolens-ar] --resume: resume.pt at step {state.get('global_step')} "
        f"({int(state.get('examples', 0)):,} samples)", flush=True,
    )
    return json.dumps({"resume": True})


def _worker(
    local_rank: int,
    world_size: int,
    config_json: str,
    train_pairs: Any,
    eval_pairs: Any,
    root_str: str,
    whiteners: Any,
    loss_whiteners: Any,
    jspaces: Any,
    prev_json: str,
) -> None:
    """One DDP rank of the multi-layer reconstructor."""
    _hf_offline()
    import torch
    import wandb
    from peft import LoraConfig, PeftModel, get_peft_model

    from oracle_lens.core.reconstructor import truncate_backbone
    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.multilayer import LAYERS as ALL_LAYERS
    from oracle_lens.pipeline.multilayer_reconstructor import (
        LayerConditionedReconstructor,
        MLReconConfig,
        MultiLayerReconstructor,
        PromptTagReconstructor,
        build_ml_heads,
        build_ml_training_config,
        head_state,
        train_ml_reconstructor,
    )
    from oracle_lens.pipeline.resume import (
        RestoreTrainerState,
        apply_resume_state,
        load_resume_state,
    )

    root = Path(root_str)
    cfg = MLReconConfig(**json.loads(config_json))
    LAYERS = tuple(ALL_LAYERS[i] for i in cfg.layer_indices)  # noqa: N806 (trained subset)
    resume_meta: dict[str, Any] = json.loads(prev_json) if prev_json else {}
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    model = load_causal_lm(
        MODEL_ID, dtype=torch.bfloat16, device="cuda",
        attn_implementation=os.environ.get("OLA_ATTN_IMPL", "flash_attention_2"),
    )
    inner = truncate_backbone(model, layer=max(LAYERS))  # read the deepest target layer
    if cfg.grad_checkpointing:
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inner.enable_input_require_grads()
    d_model = int(getattr(model.config, "text_config", model.config).hidden_size)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    peft_inner: Any
    if cfg.init_from:
        # warm start (continuation): compile FIRST so the saved `_orig_mod.` LoRA keys match,
        # then load the adapter trainable (same order as ola_modal / load_lc_reconstructor).
        ckpt = root / "ml_checkpoints" / cfg.init_from
        if cfg.compile_blocks:
            for i in range(len(inner.layers)):
                inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
            if local_rank == 0:
                print(f"[iolens-ar] compiled {len(inner.layers)} blocks (static)", flush=True)
        peft_inner = PeftModel.from_pretrained(inner, str(ckpt / "lora"), is_trainable=True)
        if local_rank == 0:
            print(f"[iolens-ar] warm-started LoRA from {cfg.init_from}", flush=True)
    else:
        peft_inner = get_peft_model(inner, lora)
        if cfg.compile_blocks:
            # fresh start compiles AFTER the wrap (ola_modal order); saved keys gain `_orig_mod.`
            for i in range(len(inner.layers)):
                inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
            if local_rank == 0:
                print(f"[iolens-ar] compiled {len(inner.layers)} blocks (static)", flush=True)
    for p in peft_inner.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA fp32; base stays bf16
    if cfg.head_mode == "layer_conditioned":
        recon: Any = LayerConditionedReconstructor(
            peft_inner, LAYERS, d_model, layer_norm=cfg.head_layer_norm
        ).to("cuda")
        if cfg.init_from:
            state = torch.load(
                root / "ml_checkpoints" / cfg.init_from / "heads.pt", map_location="cuda"
            )
            recon.head.load_state_dict(state["head"])
            recon.layer_emb.load_state_dict(state["layer_emb"])
            if local_rank == 0:
                print(f"[iolens-ar] warm-started head+layer_emb from {cfg.init_from}", flush=True)
    elif cfg.head_mode == "prompt_tag":
        tag_ids = torch.tensor(json.loads(cfg.tag_ids_json), dtype=torch.long)
        if tag_ids.shape != (len(LAYERS), cfg.tag_width):
            raise SystemExit(
                f"[iolens-ar] tag_ids {tuple(tag_ids.shape)} != ({len(LAYERS)}, {cfg.tag_width})"
            )
        recon = PromptTagReconstructor(
            peft_inner, LAYERS, d_model, tag_ids, layer_norm=cfg.head_layer_norm
        ).to("cuda")
        if cfg.init_from:
            state = torch.load(
                root / "ml_checkpoints" / cfg.init_from / "heads.pt", map_location="cuda"
            )
            recon.head.load_state_dict(state["head"])
            if local_rank == 0:
                print(f"[iolens-ar] warm-started head from {cfg.init_from}", flush=True)
    else:
        heads = build_ml_heads(d_model, len(LAYERS), layer_norm=cfg.head_layer_norm).to("cuda")
        recon = MultiLayerReconstructor(peft_inner, heads)
    whiteners = [w.to("cuda") for w in whiteners]
    if loss_whiteners is not None:
        # the training loss runs on GPU tensors every step — resident rulers, not per-call H2D
        loss_whiteners = [s.to("cuda") if s is not None else None for s in loss_whiteners]
    # jspaces stay on CPU: they are only read in validation, which computes on CPU anyway

    ckpt_dir = root / "ml_checkpoints" / cfg.run_name
    callbacks: list[Any] = []
    skip_batches = 0
    tokens_prev = 0
    examples_prev = 0
    if resume_meta.get("examples_prev") or resume_meta.get("tokens_prev"):
        # warm start from a milestone: continue the sample/token axes rather than restart them
        examples_prev = int(resume_meta.get("examples_prev", 0))
        tokens_prev = int(resume_meta.get("tokens_prev", 0))
        if local_rank == 0:
            print(f"[iolens-ar] continuing curve from {examples_prev:,} samples / "
                  f"{tokens_prev:,} tokens", flush=True)
    elif resume_meta:
        state = load_resume_state(ckpt_dir)
        if state is None:
            raise SystemExit(f"[iolens-ar] --resume but no {ckpt_dir}/resume.pt")
        saved_step = apply_resume_state(state, recon=recon, peft_inner=peft_inner)
        skip_batches = saved_step  # harness global_step counts micro-batches per rank
        tokens_prev = int(state.get("tokens_span", 0))
        examples_prev = int(state.get("examples", 0))
        tc_probe = build_ml_training_config(cfg, len(train_pairs), world_size=world_size)
        callbacks.append(
            RestoreTrainerState(
                tc_probe,
                restore_step=saved_step,
                optimizer_state=state.get("optimizer") or None,
            )
        )
        if local_rank == 0:
            print(
                f"[iolens-ar] resuming at step {saved_step} ({tokens_prev:,} tokens seen)",
                flush=True,
            )

    torch.cuda.reset_peak_memory_stats()
    summary = train_ml_reconstructor(
        recon,
        cfg,
        train_pairs,
        eval_pairs,
        whiteners,
        loss_whiteners=loss_whiteners,
        jspaces=jspaces,
        wandb_project=WANDB_PROJECT,
        local_rank=local_rank,
        world_size=world_size,
        callbacks=callbacks or None,
        ckpt_dir=ckpt_dir,
        skip_batches=skip_batches,
        tokens_span_prev=tokens_prev,
        examples_prev=examples_prev,
    )
    summary["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 2**30
    summary["tokens"] = int(train_pairs.lengths.sum())  # exact pool span tokens (1 epoch)
    if local_rank != 0:
        return
    print(json.dumps(summary), flush=True)
    if wandb.run is not None:
        wandb.summary.update(summary)
        wandb.finish()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_inner.save_pretrained(str(ckpt_dir / "lora"))
    # one attribute-dispatched save: the old head_mode branch referenced `heads`, which is only
    # bound in the read_final path, so any other non-layer_conditioned arch died with NameError
    # here — AFTER the whole run had been paid for.
    torch.save(head_state(recon), ckpt_dir / "heads.pt")
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{cfg.run_name}.json").write_text(
        json.dumps({"config": cfg.to_dict(), "summary": summary}, indent=2)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--pairs-dir", required=True, help="OLA_ROOT-relative; comma list pools dirs")
    ap.add_argument("--whitener-dir", default="", help="default: $OLA_ROOT")
    ap.add_argument("--whitener-prefix", default="whitening_iolens_chat")
    ap.add_argument("--n-gpu", type=int, required=True)
    ap.add_argument("--n-pairs", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument(
        "--span-law", default="uniform",
        choices=["", "uniform", "crop_uniform", "crop_pow2"],
        help="uniform (iolens default, Agam 2026-08-01): captured spans ARE the samples — the "
        "uniform32 carve tiles each answer with disjoint N~uniform{1..32} spans (no prefix "
        "crops, no shared targets); uniform_length_indices rebalances to exact-equal counts "
        "per length. crop_uniform/crop_pow2: prefix-crop laws for octave-carved captures.",
    )
    ap.add_argument("--crop-seed", type=int, default=1234)
    ap.add_argument("--micro-batch", type=int, default=32,
                    help="benchmarked 2026-08-02: mb 8 = 21 samples/s/GPU (launch-bound), "
                    "mb 32 = 75, mb 64 = 82 at 114 GiB. 32 is the knee.")
    ap.add_argument("--grad-accum", type=int, default=18)
    ap.add_argument("--expected-effective-batch", type=int, default=576,
                    help="mb32 x ga_eff 6 x world 3 = 576; mb32 x ga_eff 9 x world 2 = 576 — "
                    "the same effective batch at both cell shapes")
    ap.add_argument("--drop-layers", default="0",
                    help="csv of LAYERS indices to exclude from training (default: index 0 = "
                    "layer 0, whose target scores at chance)")
    ap.add_argument("--stream-dir", default="",
                    help="train off a live producer-fed buffer (OLA_ROOT-relative) instead of "
                    "a fixed pool; shards are consumed once and marked for the janitor")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from the pool size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-sched", default="constant")
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--head-mode",
        default="layer_conditioned",
        choices=["layer_conditioned", "read_final", "prompt_tag"],
        help="layer_conditioned: matched-depth read + learned layer embedding (arch of "
        "record). prompt_tag: layer named in the PROMPT, final-layer read, no layer "
        "embedding. read_final: final-layer read, one head PER layer (superseded).",
    )
    ap.add_argument(
        "--tag-template",
        default="[Layer {n:02d}]",
        help="prompt_tag only. Zero-padded 2-digit by default so every layer's tag "
        "tokenizes to the SAME length — required for one static shape per batch.",
    )
    ap.add_argument("--loss-space", default="whiten",
                    choices=["whiten", "rawcos", "unitnorm", "jspace", "mixed"])
    ap.add_argument("--loss-mix-lambda", type=float, default=0.25,
                    help="mixed arm: λ weight on the J cosine (0=pure whitened, 1=pure J)")
    ap.add_argument("--jspace-repo", default=JSPACE_REPO_DEFAULT,
                    help="HF repo of the J-lens artifact for jspace/mixed loss rulers")
    ap.add_argument("--jspace-file", default=JSPACE_FILE_DEFAULT)
    ap.add_argument("--jspace-revision", default=JSPACE_REVISION_DEFAULT,
                    help="pinned snapshot — the trainer runs HF-offline, prefetch this revision")
    ap.add_argument("--eval-dir", default="",
                    help="OLA_ROOT-relative dir with pairs_eval_*.safetensors to use as the "
                    "in-train val set (held-out conversations) instead of head-carving the "
                    "train pool — the fair-comparison protocol")
    ap.add_argument("--ridge-c", type=float, default=0.1)
    ap.add_argument("--pad-width", type=int, default=32)
    ap.add_argument("--grad-ckpt", action="store_true", default=False,
                    help="benchmarked no-op at N<=32 (83 GiB peak at mb32) — off by default")
    ap.add_argument("--compile-blocks", action="store_true", default=True)
    ap.add_argument("--no-compile-blocks", dest="compile_blocks", action="store_false")
    ap.add_argument("--stream-buffer-rows", type=int, default=150_000)
    ap.add_argument("--eval-every-steps", type=int, default=200)
    ap.add_argument("--ckpt-every-steps", type=int, default=0)
    ap.add_argument("--ckpt-samples-base", type=int, default=125_000,
                    help="log-spaced sample-axis milestones from this example count (0 = use "
                    "--ckpt-every-steps); log-log scaling curves need log-spaced x points")
    ap.add_argument("--ckpt-samples-factor", type=float, default=1.4142135623730951)
    ap.add_argument("--save-every-steps", type=int, default=200)
    ap.add_argument("--init-from", default="", help="warm start (wave extension), fresh optimizer")
    ap.add_argument("--examples-prev", type=int, default=0,
                    help="samples already trained by the checkpoint being warm-started from — "
                    "seeds the counter so the scaling curve CONTINUES instead of restarting at 0")
    ap.add_argument("--tokens-prev", type=int, default=0, help="ditto for span tokens")
    ap.add_argument("--resume", action="store_true", help="exact continuation from resume.pt")
    ap.add_argument(
        "--allow-fresh", action="store_true",
        help="with --resume, permit starting from scratch when no resume.pt exists. Off by "
        "default so an auto-restart can never silently replace a trained run with chance-level "
        "weights.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", type=int, default=0, help="print config + N sample rows, exit")
    args = ap.parse_args()
    _hf_offline()

    import torch
    import torch.multiprocessing as tmp
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.multilayer import load_multilayer_shards_lazy
    from oracle_lens.pipeline.multilayer_reconstructor import MLReconConfig

    root = ola_root()
    n_gpu = args.n_gpu

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and len(visible.split(",")) != n_gpu:
        raise SystemExit(
            f"[iolens-ar] --n-gpu {n_gpu} but CUDA_VISIBLE_DEVICES={visible!r} — set both"
        )
    if args.resume and args.init_from:
        raise SystemExit("[iolens-ar] --resume and --init-from are mutually exclusive")
    if args.ckpt_every_steps and args.lr_sched != "constant":
        raise SystemExit(
            "[iolens-ar] milestone checkpoints are a scaling curve only under constant LR"
        )

    from oracle_lens.pipeline.multilayer import LAYERS as ALL_LAYERS_MAIN

    ALL_LAYERS = ALL_LAYERS_MAIN  # noqa: N806
    drop = {int(x) for x in args.drop_layers.split(",") if x.strip() != ""}
    layer_indices = tuple(i for i in range(len(ALL_LAYERS)) if i not in drop)
    trained_layers = [ALL_LAYERS[i] for i in layer_indices]
    print(f"[iolens-ar] training {len(layer_indices)}/{len(ALL_LAYERS)} layers: "
          f"{trained_layers}", flush=True)

    # prompt_tag: resolve the layer tags ONCE here, from the TRAINED subset, so (a) the layer set
    # is single-sourced off --drop-layers, (b) all ranks share one tokenization (no tokenizer in
    # the worker => 8 ranks cannot disagree), and (c) the exact ids are provenance in every
    # milestone meta.json. Equal token length across layers is a hard requirement: the tag is
    # prepended, so a ragged tag would shift the read position AND break the one-static-shape
    # contract that torch.compile(dynamic=False) relies on.
    tag_ids_json, tag_width = "", 0
    if args.head_mode == "prompt_tag":
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
        tags = [args.tag_template.format(n=lyr) for lyr in trained_layers]
        ids = [_tok(t, add_special_tokens=False).input_ids for t in tags]
        widths = {len(i) for i in ids}
        if len(widths) != 1:
            detail = ", ".join(f"{t!r}={len(i)}" for t, i in zip(tags, ids, strict=True))
            raise SystemExit(
                f"[iolens-ar] --tag-template {args.tag_template!r} tokenizes to unequal lengths "
                f"({detail}). Use a fixed-width layer field, e.g. '[Layer {{n:02d}}]'."
            )
        tag_width = widths.pop()
        tag_ids_json = json.dumps(ids)
        print(
            f"[iolens-ar] prompt_tag: {len(tags)} tags, width={tag_width} tok "
            f"(backbone width {args.pad_width} + {tag_width} = {args.pad_width + tag_width}); "
            f"first={tags[0]!r} ids={ids[0]} last={tags[-1]!r} ids={ids[-1]}",
            flush=True,
        )

    cfg = MLReconConfig(
        run_name=args.run_name,
        n_pairs=args.n_pairs,
        min_len=args.min_len,
        max_len=args.max_len,
        span_law=args.span_law,
        crop_seed=args.crop_seed,
        init_from=args.init_from,
        head_mode=args.head_mode,
        tag_template=args.tag_template,
        tag_width=tag_width,
        tag_ids_json=tag_ids_json,
        whitener_prefix=args.whitener_prefix,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        lr_sched=args.lr_sched,
        warmup_steps=args.warmup_steps,
        ridge_c=args.ridge_c,
        pad_width=args.pad_width,
        compile_blocks=args.compile_blocks,
        stream_buffer_rows=args.stream_buffer_rows,
        eval_every_steps=args.eval_every_steps,
        loss_space=args.loss_space,
        loss_mix_lambda=args.loss_mix_lambda,
        jspace_repo=args.jspace_repo,
        jspace_file=args.jspace_file,
        jspace_revision=args.jspace_revision,
        layer_indices=layer_indices,
        # ABSOLUTE: worker processes resolve this with a CWD that is not OLA_ROOT
        stream_dir=str(ola_root() / args.stream_dir) if args.stream_dir else "",
        max_steps_override=args.max_steps,
        grad_checkpointing=args.grad_ckpt,
        save_every_steps=args.save_every_steps,
        ckpt_every_steps=args.ckpt_every_steps,
        ckpt_samples_base=args.ckpt_samples_base,
        ckpt_samples_factor=args.ckpt_samples_factor,
        seed=args.seed,
    )
    eff = effective_batch(cfg.micro_batch, cfg.grad_accum, n_gpu)
    if args.expected_effective_batch and eff != args.expected_effective_batch:
        raise SystemExit(
            f"[iolens-ar] effective batch {eff} (mb={cfg.micro_batch} x ga_eff x world={n_gpu})"
            f" != protocol {args.expected_effective_batch} — refusing an incomparable run"
        )

    if not args.dry_run:
        assert_fast_kernels()

    if args.stream_dir:
        # streaming: the live buffer IS the training set; validation uses the captured eval
        # pairs (true activations from held-out conversations, never streamed/consumed).
        from oracle_lens.pipeline.multilayer import load_multilayer_shards_lazy as _load

        eval_paths = sorted((root / args.stream_dir).glob("pairs_eval_*.safetensors"))
        if not eval_paths:
            raise SystemExit(f"[iolens-ar] no pairs_eval_* under {root / args.stream_dir}")
        eval_all, _ = _load(eval_paths)
        n_ev = min(cfg.max_eval_rows, len(eval_all))
        eval_pairs_s: Any = eval_all.select(torch.arange(n_ev))
        train_pairs_s: Any = eval_pairs_s  # placeholder; configure_training_dl swaps the stream in
        cfg.extra = dict(cfg.extra or {})
        cfg.extra.update(
            effective_batch=str(eff), world_size=str(n_gpu), stream_dir=args.stream_dir,
            layers=",".join(str(ALL_LAYERS_MAIN[i]) for i in cfg.layer_indices),
            loss_space=cfg.loss_space,
        )
        whiteners_all_s = load_whiteners_cached(
            Path(args.whitener_dir) if args.whitener_dir else root,
            cfg.whitener_prefix, cfg.ridge_c,
        )
        whiteners_s = [whiteners_all_s[i] for i in cfg.layer_indices]
        for w in whiteners_s:
            w.mu.share_memory_()
            w.w.share_memory_()
        loss_whiteners_s, jspaces_s = build_loss_spaces(
            cfg, Path(args.whitener_dir) if args.whitener_dir else root, whiteners_s
        )
        # serialize AFTER build_loss_spaces — it writes the J provenance into cfg.extra
        config_json_s = json.dumps(cfg.to_dict())
        print(f"[iolens-ar] STREAMING from {root / args.stream_dir} | eval {len(eval_pairs_s)} "
              f"pairs | eff_batch {eff} | max_steps {cfg.max_steps_override}", flush=True)
        _warm_page_cache()
        # --resume must be honoured here too. It used to be handled only in the non-streaming
        # path below, so in streaming mode it was a SILENT no-op: the run restarted from scratch
        # at chance loss (2.0) while looking healthy, and would have overwritten resume.pt with
        # the untrained state at the next save. Caught 2026-08-02 after a crash-restart.
        prev_json = resume_meta_json(args, root, cfg.run_name, n_gpu)
        if n_gpu > 1:
            train_pairs_s.share_memory_()
            tmp.spawn(  # type: ignore[attr-defined,no-untyped-call]
                _worker,
                args=(n_gpu, config_json_s, train_pairs_s, eval_pairs_s, str(root), whiteners_s,
                      loss_whiteners_s, jspaces_s, prev_json),
                nprocs=n_gpu, join=True,
            )
        else:
            _worker(0, 1, config_json_s, train_pairs_s, eval_pairs_s, str(root), whiteners_s,
                    loss_whiteners_s, jspaces_s, prev_json)
        return

    # ---- data prep (mirrors ola_modal.train_ml_recon) ----
    _im = AutoTokenizer.from_pretrained(MODEL_ID).convert_tokens_to_ids("<|im_start|>")
    assert isinstance(_im, int), f"<|im_start|> resolved to {_im!r}"
    im_start = _im
    paths = sorted(
        p
        for d in args.pairs_dir.split(",")
        for p in (root / d.strip()).glob("pairs_train_*.safetensors")
    )
    if not paths:
        raise SystemExit(f"[iolens-ar] no pairs_train_*.safetensors under {root}/{args.pairs_dir}")
    pairs, _meta = load_multilayer_shards_lazy(paths)  # targets stay mmap'd
    hits = torch.nonzero(pairs.span_ids == im_start).squeeze(-1)
    keep = torch.ones(len(pairs), dtype=torch.bool)
    if len(hits):
        bad = torch.unique(torch.searchsorted(pairs.offsets, hits, right=True) - 1)
        keep[bad] = False
    lens = pairs.lengths
    if cfg.span_law not in ("crop_uniform", "crop_pow2"):
        # crop modes keep LONG rows — they are the crop sources
        keep &= (lens >= cfg.min_len) & (lens <= cfg.max_len)
    pairs = pairs.select(torch.nonzero(keep).squeeze(-1))
    n = len(pairs)
    if cfg.span_law in ("crop_uniform", "crop_pow2"):
        from oracle_lens.pipeline.multilayer import CroppedPairs, stratified_crop_pool_set

        if cfg.span_law == "crop_pow2":
            ns = tuple(2**k for k in range(6) if 2**k <= cfg.max_len)  # (1,2,4,8,16,32)
        else:
            ns = tuple(range(cfg.min_len, cfg.max_len + 1))
        eval_src = pairs.select(torch.arange(min(4096, n // 10)))
        er, el = stratified_crop_pool_set(eval_src.lengths, ns, 128, seed=4321)
        eval_pairs: Any = CroppedPairs(eval_src, er, el)
        train_src = pairs.select(torch.arange(len(eval_src), n))
        per_n = max(1, (cfg.n_pairs or len(train_src)) // len(ns))
        tr, tl = stratified_crop_pool_set(train_src.lengths, ns, per_n, seed=cfg.crop_seed)
        train_pairs: Any = CroppedPairs(train_src, tr, tl)
        print(
            f"[iolens-ar] span_law={cfg.span_law} lengths={ns}: "
            f"train {len(train_pairs)} crops ({per_n}/N) / eval {len(eval_pairs)} crops "
            f"from {len(train_src)} source rows",
            flush=True,
        )
    else:
        if args.eval_dir:
            # fair-comparison protocol: eval = the capture's own pairs_eval_* shards (held-out
            # CONVERSATIONS, written before training) — train keeps the WHOLE pool, no head-carve
            eval_paths = sorted((root / args.eval_dir).glob("pairs_eval_*.safetensors"))
            if not eval_paths:
                raise SystemExit(
                    f"[iolens-ar] --eval-dir: no pairs_eval_* under {root / args.eval_dir}"
                )
            eval_all, _ = load_multilayer_shards_lazy(eval_paths)
            ehits = torch.nonzero(eval_all.span_ids == im_start).squeeze(-1)
            ekeep = torch.ones(len(eval_all), dtype=torch.bool)
            if len(ehits):
                ebad = torch.unique(torch.searchsorted(eval_all.offsets, ehits, right=True) - 1)
                ekeep[ebad] = False
            ekeep &= (eval_all.lengths >= cfg.min_len) & (eval_all.lengths <= cfg.max_len)
            eval_all = eval_all.select(torch.nonzero(ekeep).squeeze(-1))
            eval_pairs = eval_all.select(torch.arange(min(args.n_eval, len(eval_all))))
            train_all = pairs
        else:
            eval_pairs = pairs.select(torch.arange(min(args.n_eval, n // 10)))
            train_all = pairs.select(torch.arange(len(eval_pairs), n))
        if cfg.n_pairs:
            order = torch.randperm(len(train_all), generator=torch.Generator().manual_seed(1234))
            train_pairs = train_all.select(order[: cfg.n_pairs])
        else:
            train_pairs = train_all

    cfg.extra = dict(cfg.extra or {})
    cfg.extra.update(
        effective_batch=str(eff),
        pairs_dir=args.pairs_dir,
        world_size=str(n_gpu),
        train_tokens_exact=str(int(train_pairs.lengths.sum())),
        eval_tokens_exact=str(int(eval_pairs.lengths.sum())),
        train_crops=str(len(train_pairs)),
        eval_crops=str(len(eval_pairs)),
        loss_space=cfg.loss_space,
        eval_dir=args.eval_dir,
        stream_workers=str(STREAM_WORKERS),
    )
    print(
        f"[iolens-ar] train {len(train_pairs)} / eval {len(eval_pairs)} "
        f"({int(train_pairs.lengths.sum()):,} tokens), n_gpu={n_gpu}, eff_batch={eff}, "
        f"loss_space={cfg.loss_space}",
        flush=True,
    )

    wdir = Path(args.whitener_dir) if args.whitener_dir else root
    whiteners_all = load_whiteners_cached(wdir, cfg.whitener_prefix, cfg.ridge_c)
    whiteners = [whiteners_all[i] for i in cfg.layer_indices]  # match the trained layer subset
    for w in whiteners:
        w.mu.share_memory_()
        w.w.share_memory_()

    loss_whiteners, jspaces = build_loss_spaces(cfg, wdir, whiteners)
    # serialize AFTER build_loss_spaces — it writes the J provenance into cfg.extra
    config_json = json.dumps(cfg.to_dict())

    if args.dry_run:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        for i in range(min(args.dry_run, len(train_pairs))):
            ids = train_pairs.row_ids(i)
            print(f"--- row {i} (N={len(ids)}) ---")
            print(repr(tok.decode(ids)))
        print(f"[iolens-ar] config: {config_json}")
        print("[iolens-ar] dry run complete — config above, no training")
        return

    prev_json = resume_meta_json(args, root, cfg.run_name, n_gpu)

    _warm_page_cache()

    if n_gpu > 1:
        train_pairs.share_memory_()
        eval_pairs.share_memory_()
        tmp.spawn(  # type: ignore[attr-defined,no-untyped-call]
            _worker,
            args=(n_gpu, config_json, train_pairs, eval_pairs, str(root), whiteners,
                  loss_whiteners, jspaces, prev_json),
            nprocs=n_gpu,
            join=True,
        )
    else:
        _worker(0, 1, config_json, train_pairs, eval_pairs, str(root), whiteners,
                loss_whiteners, jspaces, prev_json)
    result = json.loads((root / "runs" / f"{cfg.run_name}.json").read_text())["summary"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
