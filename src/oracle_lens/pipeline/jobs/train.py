"""Multi-layer AR training — the launcher-agnostic body (v2 protocol).

Lifted from ``scripts/ola/ola_modal.py::{_ml_recon_worker, train_ml_recon}`` (which now delegate
here). Changes vs the lift source, all deliberate and recorded in ``runs/<name>.json``:

- ``root`` / ``whiteners_dir`` are explicit parameters (Modal passes its volume paths, Slurm
  passes ``$OLA_ROOT`` — e.g. ``whitening_v2/`` for the refit on-policy whiteners).
- attention impl comes from ``$OLA_ATTN_IMPL`` (default ``flash_attention_2``; ``sdpa`` fallback).
- the effective batch ``micro_batch x ga_eff x world_size`` is computed, recorded, and — when
  ``expected_effective_batch`` is given — ASSERTED before the model loads, so a mis-sized run
  fails in seconds, not after training. (v1's ``grad_accum // world_size`` silently doubled the
  global batch at 8 GPUs; this makes that class of bug loud.)
- the exact token count and the span-length octave fingerprint are recorded per run.
"""

import json
import os
from pathlib import Path
from typing import Any

from oracle_lens.pipeline.jobs.settings import MODEL_ID, WANDB_PROJECT


def effective_batch(
    micro_batch: int, grad_accum: int, world_size: int, *, max_batch_rows: int | None = None
) -> int:
    """spans per optimizer step = bs x ga_eff x ngpus, with the harness's ga scaling rule.

    ``max_batch_rows`` is the TokenBudgetSampler's row cap (crop32 spans are short, so the cap
    binds long before the token budget does). When bucketing is on, THAT is the real per-rank
    batch — not ``micro_batch``, which the sampler ignores. Passing it keeps the
    ``--expected-effective-batch`` guard honest: it defaults to 8, so a run asking for mb=128
    silently trained at eff-batch 32 instead of 512 while the assertion still passed.
    """
    ga_eff = max(1, grad_accum // world_size) if world_size > 1 else grad_accum
    rows = micro_batch if max_batch_rows is None else max_batch_rows
    return rows * ga_eff * world_size


def _load_whiteners_cached(wdir: Path, ridge_c: float) -> list[Any]:
    """Whitening transforms with a disk cache for the expensive part.

    ``load_whitener`` runs an fp64 eigh of a 5120x5120 covariance PER LAYER (~10 CPU-min for all
    17). The first call per (wdir, ridge) pays that once and caches {mu, W}; every later run —
    and every spawned rank, which receives these tensors via shared memory — loads in seconds.
    """
    import time

    import torch
    from safetensors.torch import load_file, save_file

    from oracle_lens.core.whitening import Whitener, load_whitener
    from oracle_lens.pipeline.multilayer import LAYERS

    cache = wdir / f"wmatrix_ridge{ridge_c}.safetensors"
    if cache.exists():
        t = load_file(str(cache))
        return [
            Whitener(mu=t[f"mu_L{lyr}"], w=t[f"w_L{lyr}"], ridge_c=ridge_c) for lyr in LAYERS
        ]
    t0 = time.time()
    whiteners = [
        load_whitener(wdir / f"whitening_L{lyr}.safetensors", ridge_c=ridge_c) for lyr in LAYERS
    ]
    tensors: dict[str, torch.Tensor] = {}
    for lyr, w in zip(LAYERS, whiteners, strict=True):
        tensors[f"mu_L{lyr}"] = w.mu.cpu()
        tensors[f"w_L{lyr}"] = w.w.cpu()
    save_file(tensors, str(cache))
    print(f"[ml-recon] built + cached {len(whiteners)} whitening matrices "
          f"in {time.time() - t0:.0f}s -> {cache.name}", flush=True)
    return whiteners


def ml_recon_worker(
    local_rank: int,
    world_size: int,
    config_json: str,
    train_pairs: Any,
    eval_pairs: Any,
    root_str: str,
    whiteners: Any,
    loss_whiteners: Any = None,
    jspaces: Any = None,
) -> None:
    """One DDP rank of the multi-layer reconstructor (fresh from base, NO compile)."""
    import torch
    import wandb
    from peft import LoraConfig, get_peft_model

    from oracle_lens.core.reconstructor import truncate_backbone
    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.multilayer import LAYERS
    from oracle_lens.pipeline.multilayer_reconstructor import (
        MLReconConfig,
        MultiLayerReconstructor,
        build_ml_heads,
        train_ml_reconstructor,
    )

    root = Path(root_str)
    cfg = MLReconConfig(**json.loads(config_json))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    attn_impl = os.environ.get("OLA_ATTN_IMPL", "flash_attention_2")
    model = load_causal_lm(
        MODEL_ID, dtype=torch.bfloat16, device="cuda", attn_implementation=attn_impl
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
    peft_inner = get_peft_model(inner, lora)
    for p in peft_inner.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA fp32; base stays bf16 (no compile)
    heads: Any = None
    if cfg.head_mode == "layer_conditioned":
        from oracle_lens.pipeline.multilayer_reconstructor import LayerConditionedReconstructor

        recon: Any = LayerConditionedReconstructor(
            peft_inner, LAYERS, d_model, layer_norm=cfg.head_layer_norm
        ).to("cuda")
    else:
        heads = build_ml_heads(d_model, len(LAYERS), layer_norm=cfg.head_layer_norm).to("cuda")
        recon = MultiLayerReconstructor(peft_inner, heads)
    if cfg.compile_blocks:
        # per-block static-shape compile -- the single-layer trainer's documented 17x lever;
        # bucketed edge-padding gives each block exactly len(PAD_EDGES) shapes.
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 32
        for i in range(len(inner.layers)):
            # OLA_COMPILE_MODE=reduce-overhead wraps each block in CUDA graphs — targets the
            # measured kernel-launch floor of short-span steps; single static shape is ideal.
            inner.layers[i] = torch.compile(
                inner.layers[i], dynamic=False,
                mode=os.environ.get("OLA_COMPILE_MODE") or "default",
            )
        if local_rank == 0:
            print(f"[ml-recon] compiled {len(inner.layers)} blocks (static bucket shapes)",
                  flush=True)
    # Whitening matrices are prebuilt ONCE in the parent (17 eigh of 5120^2 take 10-17 CPU-min;
    # 8 ranks each doing them oversubscribed the node and staggered rank arrival by >10 min,
    # wedging the DDP store barrier -- job 1919393). Ranks only move them to their device.
    whiteners = [w.to("cuda") for w in whiteners]
    if loss_whiteners is not None:
        # None = layer excluded from the loss (e.g. L63 under the J-space ruler)
        loss_whiteners = [w.to("cuda") if w is not None else None for w in loss_whiteners]
    if jspaces is not None:
        jspaces = [s.to("cuda") if s is not None else None for s in jspaces]

    callbacks: list[Any] = []
    ckpt_dir = root / "ml_checkpoints" / cfg.run_name
    if cfg.save_every_steps > 0:
        from oracle_lens.pipeline.jobs.resume import (
            ResumeCheckpoint,
            apply_resume_state,
            load_resume_state,
        )
        from oracle_lens.pipeline.multilayer_reconstructor import build_ml_training_config

        state = load_resume_state(ckpt_dir)
        if state is not None:
            saved_step = apply_resume_state(state, recon=recon, peft_inner=peft_inner)
            if local_rank == 0:
                # The streaming sampler has no step-exact fast-forward (the old bucketed path's
                # SkipableBucketSampler went away with it), so weights/optimizer resume but the
                # data order restarts — those first saved_step batches are seen twice. Loud on
                # purpose: silently re-training on them would bias a scaling-curve rung.
                print(
                    f"[resume] loaded step {saved_step} from {ckpt_dir}/resume.pt\n"
                    f"[resume] WARNING: data order RESTARTS from batch 0 — the first "
                    f"{saved_step} micro-batches will be seen a second time. For a clean "
                    f"scaling point, train from scratch instead of resuming.",
                    flush=True,
                )
        tc_probe = build_ml_training_config(cfg, len(train_pairs), world_size=world_size)
        cb = ResumeCheckpoint(
            tc_probe, recon=recon, peft_inner=peft_inner, ckpt_dir=ckpt_dir,
            save_every_steps=cfg.save_every_steps, rank=local_rank,
        )
        if state is not None:
            cb.restore_step = int(state["global_step"])  # applied in on_train_start
            cb.optimizer_state = state.get("optimizer") or None
        callbacks.append(cb)

    torch.cuda.reset_peak_memory_stats()
    summary = train_ml_reconstructor(
        recon,
        cfg,
        train_pairs,
        eval_pairs,
        whiteners,
        wandb_project=WANDB_PROJECT,
        local_rank=local_rank,
        world_size=world_size,
        callbacks=callbacks or None,
        loss_whiteners=loss_whiteners,
        jspaces=jspaces,
    )
    summary["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 2**30
    summary["tokens"] = int(train_pairs.lengths.sum())  # exact real training tokens (span x-axis)
    if local_rank != 0:
        return
    print(json.dumps(summary), flush=True)
    if wandb.run is not None:
        wandb.summary.update(summary)
        wandb.finish()
    ckpt_dir = root / "ml_checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_inner.save_pretrained(str(ckpt_dir / "lora"))
    if cfg.head_mode == "layer_conditioned":
        import torch as _torch

        _torch.save(
            {"head": recon.head.state_dict(), "layer_emb": recon.layer_emb.state_dict()},
            ckpt_dir / "heads.pt",
        )
    else:
        import torch as _torch

        _torch.save(heads.state_dict(), ckpt_dir / "heads.pt")
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{cfg.run_name}.json").write_text(
        json.dumps({"config": cfg.to_dict(), "summary": summary}, indent=2)
    )


def train_ml(
    config_json: str,
    *,
    n_gpu: int,
    pairs_dir: str,
    root: Path,
    whiteners_dir: Path | None = None,
    n_eval: int = 2000,
    expected_effective_batch: int | None = None,
) -> dict[str, Any]:
    """Load the pool, filter, split, and run ``n_gpu``-way DDP training. See module docstring."""
    import torch
    import torch.multiprocessing as tmp
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.matching import histogram_fingerprint
    from oracle_lens.pipeline.multilayer import load_multilayer_shards_lazy
    from oracle_lens.pipeline.multilayer_reconstructor import MLReconConfig

    cfg = MLReconConfig(**json.loads(config_json))
    bucketing = cfg.bucket_by_length and not cfg.pad_width
    eff = effective_batch(
        cfg.micro_batch, cfg.grad_accum, n_gpu,
        max_batch_rows=cfg.max_batch_rows if bucketing else None,
    )
    if expected_effective_batch is not None and eff != expected_effective_batch:
        raise SystemExit(
            f"effective batch {eff} (mb={cfg.micro_batch} x ga_eff x world={n_gpu}) != "
            f"protocol {expected_effective_batch} — refusing to train an incomparable run"
        )
    wdir = whiteners_dir if whiteners_dir is not None else root
    # convert_tokens_to_ids is typed int|list[int]; a single-token lookup is always int, but
    # mypy needs the assert to know it (and the Tensor comparison below needs an int).
    _im = AutoTokenizer.from_pretrained(MODEL_ID).convert_tokens_to_ids("<|im_start|>")
    assert isinstance(_im, int), f"<|im_start|> resolved to {_im!r}"
    im_start = _im
    # pairs_dir may be a comma-separated list of dirs -> pooled; the seed-1234 shuffle over the
    # combined pool keeps nested n_pairs rungs consistent (rung N = first N of the shuffle).
    paths = sorted(
        p
        for d in pairs_dir.split(",")
        for p in (root / d.strip()).glob("pairs_train_*.safetensors")
    )
    if not paths:
        raise FileNotFoundError(f"no pairs_train_*.safetensors under {root} / {pairs_dir}")
    pairs, _meta = load_multilayer_shards_lazy(paths)  # targets stay mmap'd (pool >> RAM)
    # boundary-leak filter (spans containing <|im_start|>) + length filter
    hits = torch.nonzero(pairs.span_ids == im_start).squeeze(-1)
    keep = torch.ones(len(pairs), dtype=torch.bool)
    if len(hits):
        bad = torch.unique(torch.searchsorted(pairs.offsets, hits, right=True) - 1)
        keep[bad] = False
    lens = pairs.lengths
    keep &= (lens >= cfg.min_len) & (lens <= cfg.max_len)
    pairs = pairs.select(torch.nonzero(keep).squeeze(-1))
    n = len(pairs)
    eval_src = pairs.select(torch.arange(min(n_eval, n // 10)))
    train_src = pairs.select(torch.arange(len(eval_src), n))
    if cfg.crop_max:
        from oracle_lens.pipeline.multilayer import CroppedPairs, stratified_crop_pool
        from oracle_lens.pipeline.multilayer_reconstructor import PairsLike

        # Short-span regime: stratified-uniform prefix crops (N exactly uniform in
        # {1..crop_max}); the pool ordering is rank-within-N major, so a rung that is a prefix
        # of the pool (n_pairs rounded down to a multiple of crop_max) keeps N exactly uniform.
        # Eval is cropped from its own carved-out source rows — targets stay disjoint.
        # ``stratified_crop_pool`` takes crops-per-N; the old make_stratified_crops took a total.
        er, el = stratified_crop_pool(
            eval_src.lengths, 1, cfg.crop_max,
            max(1, len(eval_src) // cfg.crop_max), seed=cfg.seed + 1,
        )
        eval_pairs: PairsLike = CroppedPairs(eval_src, er, el)
        tr, tl = stratified_crop_pool(
            train_src.lengths, 1, cfg.crop_max,
            max(1, len(train_src) // cfg.crop_max), seed=cfg.seed,
        )
        train_all: PairsLike = CroppedPairs(train_src, tr, tl)
        rung = (cfg.n_pairs // cfg.crop_max) * cfg.crop_max if cfg.n_pairs else 0
        train_pairs = train_all.select(torch.arange(rung)) if rung else train_all
    else:
        eval_pairs = eval_src
        if cfg.n_pairs:
            order = torch.randperm(len(train_src), generator=torch.Generator().manual_seed(1234))
            train_pairs = train_src.select(order[: cfg.n_pairs])
        else:
            train_pairs = train_src
    # protocol provenance: recorded before training so a crashed run still leaves its config trail
    cfg.extra = dict(cfg.extra or {})
    cfg.extra.update(
        effective_batch=str(eff),
        whiteners_dir=str(wdir),
        train_len_fingerprint=histogram_fingerprint(train_pairs.lengths),
        eval_len_fingerprint=histogram_fingerprint(eval_pairs.lengths),
        pairs_dir=pairs_dir,
        world_size=str(n_gpu),
        attn_impl=os.environ.get("OLA_ATTN_IMPL", "flash_attention_2"),
    )
    if cfg.crop_max:
        cfg.extra.update(
            crop_max=str(cfg.crop_max),
            crop_scheme="stratified_uniform",
            train_crops=str(len(train_pairs)),
            train_tokens_exact=str(int(train_pairs.lengths.sum())),
            eval_crops=str(len(eval_pairs)),
            eval_tokens_exact=str(int(eval_pairs.lengths.sum())),
            n_source_rows=str(n),
        )
    config_json = json.dumps(cfg.to_dict())
    print(
        f"[ml-recon] train {len(train_pairs)} / eval {len(eval_pairs)} "
        f"({int(train_pairs.lengths.sum()):,} tokens), n_gpu={n_gpu}, eff_batch={eff}",
        flush=True,
    )
    # Build the whitening transforms ONCE (17 eigh of 5120^2 ~ 10 CPU-min), cache the result on
    # disk beside the moments, and hand ranks shared-memory tensors -- see ml_recon_worker note.
    whiteners = _load_whiteners_cached(wdir, cfg.ridge_c)
    for w in whiteners:
        w.mu.share_memory_()
        w.w.share_memory_()
    loss_whiteners = None
    jspaces: list[Any] | None = None  # PURE-J rulers, for val metrics in every J-bearing arm
    if cfg.loss_space in ("jspace", "mixed") and cfg.loss_ridge_c:
        raise SystemExit(f"loss_space={cfg.loss_space} and loss_ridge_c are mutually exclusive")
    if cfg.loss_ridge_c and cfg.loss_ridge_c != cfg.ridge_c:
        loss_whiteners = _load_whiteners_cached(wdir, cfg.loss_ridge_c)
        for w in loss_whiteners:
            w.mu.share_memory_()
            w.w.share_memory_()
        cfg.extra["loss_ridge_c"] = str(cfg.loss_ridge_c)
        print(f"[ml-recon] LOSS whitener ridge={cfg.loss_ridge_c} (metrics stay {cfg.ridge_c})",
              flush=True)
    if cfg.loss_space in ("jspace", "mixed"):
        from oracle_lens.pipeline.jspace import MixedSpace, load_jspaces
        from oracle_lens.pipeline.multilayer import LAYERS

        if not (cfg.jspace_repo and cfg.jspace_file):
            raise SystemExit(f"loss_space={cfg.loss_space} needs --jspace-repo and --jspace-file")
        spaces = load_jspaces(
            cfg.jspace_repo, cfg.jspace_file, wdir, LAYERS,
            revision=cfg.jspace_revision or None,
        )
        for s in spaces.values():
            s.mu.share_memory_()  # type: ignore[no-untyped-call]
            s.j.share_memory_()  # type: ignore[no-untyped-call]
        jspaces = [spaces.get(lyr) for lyr in LAYERS]  # pure J, for val_jfve in BOTH arms
        if cfg.loss_space == "jspace":
            loss_whiteners = list(jspaces)
        else:
            # Mixed: (1-lam)*whitened + lam*J as one ruler. L63 is EXCLUDED (not whitened-only)
            # so the mixed arm covers exactly the same 16 layers as the pure-J arm and the two
            # are directly comparable; the whitened arm keeps all 17 as always.
            lam = cfg.loss_mix_lambda
            if not 0.0 <= lam <= 1.0:
                raise SystemExit(f"loss_mix_lambda must be in [0,1], got {lam}")
            loss_whiteners = [
                MixedSpace(whitener=whiteners[i], jspace=s, lam=lam) if s is not None else None
                for i, s in enumerate(jspaces)
            ]
        covered = [lyr for lyr in LAYERS if lyr in spaces]
        excluded = [lyr for lyr in LAYERS if lyr not in spaces]
        try:  # pin the artifact revision (resolves from the local HF cache when offline)
            from huggingface_hub import snapshot_download

            resolved = Path(
                snapshot_download(cfg.jspace_repo, allow_patterns=[cfg.jspace_file])
            ).name
        except Exception:
            resolved = cfg.jspace_revision or "unresolved"
        cfg.extra.update(
            loss_space=cfg.loss_space,
            loss_mix_lambda=str(cfg.loss_mix_lambda) if cfg.loss_space == "mixed" else "",
            jspace_repo=cfg.jspace_repo,
            jspace_file=cfg.jspace_file,
            jspace_revision=resolved,
            jspace_mu_dir=str(wdir),
            jspace_layers=",".join(str(lyr) for lyr in covered),
            jspace_excluded=",".join(str(lyr) for lyr in excluded),
        )
        mix = (
            f" MIXED lam={cfg.loss_mix_lambda} ((1-lam)*whitened + lam*J)"
            if cfg.loss_space == "mixed" else ""
        )
        print(
            f"[ml-recon] {cfg.loss_space} loss:{mix} {len(covered)}/{len(LAYERS)} layers "
            f"(L{','.join(str(lyr) for lyr in excluded)} EXCLUDED — no Jacobian for the target "
            f"block); whitened metrics keep all {len(LAYERS)} layers. "
            f"artifact={cfg.jspace_repo}/{cfg.jspace_file}@{resolved}",
            flush=True,
        )
    # re-serialize: the loss-space branches above add provenance to cfg.extra AFTER the first
    # dump, and the worker's run JSON is written from ITS deserialized cfg
    config_json = json.dumps(cfg.to_dict())

    # Warm the node's page cache with ONE sequential read of the model shards. Without this,
    # 8 spawned ranks each pull the 52 GB snapshot from NFS simultaneously, starve each other,
    # and reach the DDP rendezvous up to ~27 min apart — past the 10-min store timeout
    # (job 1918792). After one warm read, the other loads are RAM-speed page-cache hits.
    from huggingface_hub import snapshot_download

    t0 = __import__("time").time()
    snap = Path(snapshot_download(MODEL_ID, allow_patterns=["*.safetensors"]))
    warmed = 0
    for f in sorted(snap.glob("*.safetensors")):
        with open(f, "rb") as fh:
            while fh.read(1 << 26):
                pass
        warmed += f.stat().st_size
    print(f"[ml-recon] page-cache warm: {warmed / 2**30:.1f} GiB in "
          f"{__import__('time').time() - t0:.0f}s", flush=True)

    if n_gpu > 1:
        train_pairs.share_memory_()
        eval_pairs.share_memory_()
        tmp.spawn(  # type: ignore[no-untyped-call,attr-defined]  # torch.multiprocessing has no stubs
            ml_recon_worker,
            args=(n_gpu, config_json, train_pairs, eval_pairs, str(root), whiteners,
                  loss_whiteners, jspaces),
            nprocs=n_gpu,
            join=True,
        )
    else:
        ml_recon_worker(0, 1, config_json, train_pairs, eval_pairs, str(root), whiteners,
                        loss_whiteners, jspaces)
    result: dict[str, Any] = json.loads((root / "runs" / f"{cfg.run_name}.json").read_text())[
        "summary"
    ]
    return result
