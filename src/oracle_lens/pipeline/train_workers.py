"""Reconstructor / AO training worker bodies, shared by the Modal and Slurm entrypoints.

Lifted out of ``scripts/ola/ola_modal.py`` so the training logic is importable, type-checked
and testable rather than living inside ``@app.function`` decorators. The Modal-specific parts
(image, volumes, GPU request, ``.commit()``) stay in the launcher; what the launcher passes in
is the artifact root and the run identifiers that used to be module constants:

    root           the artifact root (a Modal volume path, or $OLA_ROOT on the cluster)
    model_id       the base model
    wandb_project  the W&B project the run reports to
    dataset_id     the source corpus (conversation iteration only)

Heavy imports (torch, peft, wandb) stay inside the bodies so importing this module from a
launcher client-side stays cheap.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from oracle_lens.hf_offline import hf_offline


def _compile_blocks_if_ckpt_needs(inner: Any, lora_dir: Any) -> None:
    """Compile-wrap backbone blocks BEFORE PeftModel.from_pretrained when the saved adapter keys
    carry the `_orig_mod.` prefix (checkpoint trained with cfg.compile_blocks) — otherwise the
    LoRA keys silently fail to match. No-op for adapters saved from uncompiled training."""
    import torch
    from safetensors import safe_open

    st = next(iter(sorted(Path(str(lora_dir)).glob("adapter_model.safetensors"))), None)
    if st is None:
        return
    with safe_open(str(st), framework="pt") as f:
        needs = any("_orig_mod." in k for k in f.keys())  # noqa: SIM118 — not a dict
    if needs:
        for i in range(len(inner.layers)):
            inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
        print(f"[eval-load] compiled {len(inner.layers)} blocks (_orig_mod ckpt)", flush=True)

def _load_ar_out(ar_dir: Path) -> tuple[Any, Any]:
    """Concatenate the AR(out) precompute shards in row order -> (ar_out [N,d], valid [N])."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    paths = sorted(ar_dir.glob("ar_out_*.safetensors"))
    if not paths:
        raise ValueError(f"no AR(out) shards in {ar_dir}; run ao_precompute first")

    def lo_of(p: Path) -> int:
        with safe_open(str(p), framework="pt") as f:
            return int(f.metadata()["lo"])

    paths.sort(key=lo_of)
    parts = [load_file(str(p), device="cpu") for p in paths]
    ar_out = torch.cat([t["ar_out"] for t in parts])
    valid = torch.cat([t["valid"] for t in parts])
    return ar_out, valid

def _iter_region_conversations(
    shard: int,
    n_shards: int,
    region_start: int,
    region_end: int,
    max_convs: int,
    *,
    dataset_id: str,
) -> "Iterator[tuple[int, list[dict[str, str]]]]":
    """Yield (doc_index, conversation) for this shard's slice of a corpus region."""
    from oracle_lens.data import load_chat_messages

    yielded = 0
    for doc_index, conversation in enumerate(load_chat_messages(dataset_id)):
        if doc_index < region_start:
            continue
        if doc_index >= region_end:
            return
        if doc_index % n_shards != shard:
            continue
        yield doc_index, conversation
        yielded += 1
        if yielded >= max_convs:
            return

def _build_recon(
    cfg: Any, device: str, *, root: Path, model_id: str
) -> tuple[Any, Any, Any]:
    """Truncated + LoRA(fp32) + per-block-compiled reconstructor (shared by train/probe)."""
    import torch
    from peft import LoraConfig, get_peft_model

    from oracle_lens.core.reconstructor import (
        Reconstructor,
        ReconstructorHead,
        truncate_backbone,
    )
    from oracle_lens.model import load_causal_lm

    model = load_causal_lm(model_id, dtype=torch.bfloat16, device=device)
    inner = truncate_backbone(model, layer=getattr(cfg, "truncate_at", 0) or cfg.layer)
    if cfg.grad_checkpointing:
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inner.enable_input_require_grads()
    config = model.config
    d_model = int(getattr(config, "text_config", config).hidden_size)
    if cfg.compile_blocks:
        # six bucket shapes -> more graphs than the default cache limit of 8
        torch._dynamo.config.cache_size_limit = 24
    init_from = getattr(cfg, "init_from", "")
    if init_from:
        # warm start: compile FIRST so the saved `_orig_mod.` LoRA keys match, then load
        from peft import PeftModel

        if cfg.compile_blocks:
            for i in range(len(inner.layers)):
                inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
        ckpt = root / "checkpoints" / init_from
        peft_inner: Any = PeftModel.from_pretrained(inner, str(ckpt / "lora"), is_trainable=True)
        for p in peft_inner.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        state = torch.load(ckpt / "head.pt", map_location=device, weights_only=True)
        head = ReconstructorHead(d_model, layer_norm=any(k.startswith("norm.") for k in state))
        head.load_state_dict(state)
        head = head.to(device)
        print(f"warm-started from {init_from}", flush=True)
        return Reconstructor(peft_inner, head), peft_inner, head
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
            p.data = p.data.float()  # LoRA fp32; base stays bf16
    if cfg.compile_blocks:
        for i in range(len(inner.layers)):
            inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
        print(f"compiled {len(inner.layers)} blocks (static bucket shapes)", flush=True)
    head = ReconstructorHead(d_model, layer_norm=cfg.head_layer_norm).to(device)
    return Reconstructor(peft_inner, head), peft_inner, head

def _recon_worker(
    local_rank: int,
    world_size: int,
    config_json: str,
    train_pairs: Any,
    eval_pairs: Any,
    *,
    root: Path,
    model_id: str,
    wandb_project: str,
) -> None:
    """One DDP rank (also the single-GPU path with world_size=1). Rank 0 saves artifacts."""
    import torch
    import wandb

    from oracle_lens.core.whitening import load_whitener
    from oracle_lens.pipeline.train_recon import (
        LongSpanReconConfig,
        train_longspan_reconstructor,
    )

    hf_offline()
    cfg = LongSpanReconConfig(**json.loads(config_json))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    recon, peft_inner, head = _build_recon(cfg, "cuda", root=root, model_id=model_id)
    whitener = load_whitener(
        root / f"whitening_L{cfg.layer}.safetensors", ridge_c=cfg.ridge_c, device="cuda"
    )
    torch.cuda.reset_peak_memory_stats()
    summary = train_longspan_reconstructor(
        recon,
        train_pairs,
        eval_pairs,
        whitener,
        cfg,
        wandb_project=wandb_project,
        local_rank=local_rank,
        world_size=world_size,
    )
    summary["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if local_rank != 0:
        return
    print(json.dumps(summary), flush=True)
    if wandb.run is not None:
        wandb.summary.update(summary)
        wandb.finish()
    ckpt_dir = root / "checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_inner.save_pretrained(str(ckpt_dir / "lora"))
    torch.save(head.state_dict(), ckpt_dir / "head.pt")
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{cfg.run_name}.json").write_text(
        json.dumps({"config": cfg.to_dict(), "summary": summary}, indent=2)
    )

def _ml_recon_worker(
    local_rank: int,
    world_size: int,
    config_json: str,
    train_pairs: Any,
    eval_pairs: Any,
    *,
    root: Path,
    model_id: str,
    wandb_project: str,
) -> None:
    """One DDP rank of the multi-layer reconstructor (fresh from base, NO compile)."""
    hf_offline()
    import torch
    import wandb
    from peft import LoraConfig, get_peft_model

    from oracle_lens.core.reconstructor import truncate_backbone
    from oracle_lens.core.whitening import load_whitener
    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.multilayer import LAYERS
    from oracle_lens.pipeline.multilayer_reconstructor import (
        MLReconConfig,
        MultiLayerReconstructor,
        build_ml_heads,
        train_ml_reconstructor,
    )

    cfg = MLReconConfig(**json.loads(config_json))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    model = load_causal_lm(
        model_id, dtype=torch.bfloat16, device="cuda", attn_implementation="flash_attention_2"
    )
    inner = truncate_backbone(model, layer=max(LAYERS))  # read the layer-60 residual (deepest)
    if cfg.grad_checkpointing:
        # spans are short (~54 tok) so activation memory is tiny — checkpointing's recompute is
        # ~2x wasted here; default it OFF for the multi-layer AR (see cfg.grad_checkpointing).
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
    if cfg.init_from:
        # warm start (continuation): compile FIRST so the saved `_orig_mod.` LoRA keys match,
        # then load the adapter trainable (same order as the single-layer _build_recon).
        from peft import PeftModel

        ckpt = root / "ml_checkpoints" / cfg.init_from
        if cfg.compile_blocks:
            for i in range(len(inner.layers)):
                inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
            print(f"[ml-recon] compiled {len(inner.layers)} blocks (static shapes)", flush=True)
        peft_inner: Any = PeftModel.from_pretrained(inner, str(ckpt / "lora"), is_trainable=True)
        for p in peft_inner.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        print(f"[ml-recon] warm-started LoRA from {cfg.init_from}", flush=True)
    else:
        peft_inner = get_peft_model(inner, lora)
        for p in peft_inner.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # LoRA fp32; base stays bf16
        if cfg.compile_blocks:
            # per-block torch.compile(dynamic=False) — the single-layer AR's big speed lever; needs
            # STATIC shapes (cfg.pad_width fixed-width collate + fixed micro_batch, drop_last).
            # Saved LoRA keys gain `_orig_mod.`; eval loaders compile-wrap first when they see it.
            for i in range(len(inner.layers)):
                inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
            print(f"[ml-recon] compiled {len(inner.layers)} blocks (static shapes)", flush=True)
    if cfg.head_mode == "layer_conditioned":
        from oracle_lens.pipeline.multilayer_reconstructor import LayerConditionedReconstructor

        recon: Any = LayerConditionedReconstructor(
            peft_inner, LAYERS, d_model, layer_norm=cfg.head_layer_norm
        ).to("cuda")
        if cfg.init_from:
            state = torch.load(
                root / "ml_checkpoints" / cfg.init_from / "heads.pt", map_location="cuda"
            )
            recon.head.load_state_dict(state["head"])
            recon.layer_emb.load_state_dict(state["layer_emb"])
            print(f"[ml-recon] warm-started head+layer_emb from {cfg.init_from}", flush=True)
    else:
        heads = build_ml_heads(d_model, len(LAYERS), layer_norm=cfg.head_layer_norm).to("cuda")
        recon = MultiLayerReconstructor(peft_inner, heads)
    whiteners = [
        load_whitener(
            root / f"{cfg.whitener_prefix}_L{lyr}.safetensors", ridge_c=cfg.ridge_c, device="cuda"
        )
        for lyr in LAYERS
    ]

    torch.cuda.reset_peak_memory_stats()
    summary = train_ml_reconstructor(
        recon,
        cfg,
        train_pairs,
        eval_pairs,
        whiteners,
        wandb_project=wandb_project,
        local_rank=local_rank,
        world_size=world_size,
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
        torch.save(
            {"head": recon.head.state_dict(), "layer_emb": recon.layer_emb.state_dict()},
            ckpt_dir / "heads.pt",
        )
    else:
        torch.save(heads.state_dict(), ckpt_dir / "heads.pt")
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{cfg.run_name}.json").write_text(
        json.dumps({"config": cfg.to_dict(), "summary": summary}, indent=2)
    )

def _ao_worker(
    local_rank: int,
    world_size: int,
    config_json: str,
    train_pairs: Any,
    val_pairs: Any,
    ar_train: Any,
    ar_val: Any,
    *,
    root: Path,
    model_id: str,
    wandb_project: str,
) -> None:
    """One AO-1 DDP rank (also the single-GPU path). Builds the full base model + LoRA per rank."""
    hf_offline()
    from dataclasses import replace

    import torch
    import wandb
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from oracle_lens.core.whitening import load_whitener
    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.ao_train import AOConfig, AODataset, build_ao1_examples, train_ao
    from oracle_lens.pipeline.verbalizer import render_wv_prompt

    cfg = AOConfig(**json.loads(config_json))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    whitener = None
    if cfg.rep == "wht_unit":
        whitener = load_whitener(
            root / f"whitening_L{cfg.layer}.safetensors", ridge_c=0.1, device="cuda"
        )
    train_ex = build_ao1_examples(train_pairs, tokenizer, cfg, ar_out=ar_train)
    # skip_examples applies to TRAIN only; the small val hold-out must not be skipped past (empty)
    val_cfg = replace(cfg, n_examples=len(val_pairs) + 1, skip_examples=0)
    val_ex = build_ao1_examples(val_pairs, tokenizer, val_cfg, ar_out=ar_val)
    train_data = AODataset(train_ex)
    val_data = AODataset(val_ex)
    if local_rank == 0:
        print(f"[ao] {len(train_data)} train / {len(val_data)} val examples", flush=True)

    model = load_causal_lm(
        model_id, dtype=torch.bfloat16, device="cuda", attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    if cfg.init_from:  # resume: load the saved LoRA (trainable) instead of a fresh one
        from peft import PeftModel

        peft_model: Any = PeftModel.from_pretrained(
            model, str(root / "wv_checkpoints" / cfg.init_from / "lora"), is_trainable=True
        )
        if local_rank == 0:
            print(f"[ao] resumed LoRA from {cfg.init_from}", flush=True)
    else:
        peft_model = get_peft_model(model, lora)
    for p_ in peft_model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()
    prompt = render_wv_prompt(tokenizer, layer=cfg.layer)
    if local_rank == 0:
        from oracle_lens.pipeline.ao_train import inject_rep  # type: ignore[attr-defined]

        iv = inject_rep(
            train_data[0]["inject_vec"].unsqueeze(0),
            cfg.rep,
            alpha=cfg.alpha,
            scale=cfg.scale,
            whitener=whitener.to("cpu") if whitener is not None else None,
        )
        print(
            f"[ao] inject: slot={prompt.slot} char={prompt.char!r} "
            f"prompt_len={len(prompt.input_ids)} rep={cfg.rep} "
            f"|v|={float(iv.norm()):.1f} (alpha={cfg.alpha}, scale={cfg.scale:.4g})",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats()
    summary = train_ao(
        peft_model,
        prompt,
        train_data,
        val_data,
        cfg,
        pad_id=int(tokenizer.pad_token_id or 0),
        whitener=whitener,
        wandb_project=wandb_project,
        local_rank=local_rank,
        world_size=world_size,
        ckpt_dir=root / "wv_checkpoints" / cfg.run_name,
    )
    summary["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if local_rank != 0:
        return
    print(json.dumps(summary), flush=True)
    if wandb.run is not None:
        wandb.summary.update(summary)
        wandb.finish()
    ckpt_dir = root / "wv_checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir / "lora"))
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{cfg.run_name}.json").write_text(
        json.dumps({"config": cfg.to_dict(), "summary": summary}, indent=2)
    )
