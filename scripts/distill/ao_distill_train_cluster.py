"""Round-1 AO distill student: bullet-list targets from NNOMP selections, warm-started.

The ``olens_r2_train_cluster.py`` shape pointed at the AO lineage: shards are ``distill_v1``
(``ao_assemble_distill.py``) under ``$OLA_ROOT/<out-dir>/shards/<variant>/{train,eval}``, the
LoRA warm-starts from the u64 AO checkpoint (``ao_checkpoints``, LoRA-only — the base model
stays frozen), and the prompt comes from the shard meta's ``prompt_mode`` via the verbalizer
registry (``concepts_raw``: tag-free '- ' bullets, no count stated).

Injection contract (differs from the u64 parent ON PURPOSE): ``transform=unit, alpha=16000`` —
norm-matched GT residuals, exactly how the teacher samples were generated (probe 2026-08-11:
norm-matched 203/1800 vs frozen-scale 138/1800 on GT injection) and how the RL step will read.
The stored shard ``vec`` is the RAW residual; the trainer normalizes at batch time.

    uv run --no-sync python scripts/distill/ao_distill_train_cluster.py \
        --out-dir distill_u64/pilot --variant omp4 --run-name ao.iolens.distill.omp4.s0 \
        --init-from ao.iolens.chat.k4.L20-60.cont.u64.s0/step28000 --dry-run 3
"""

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"
WANDB_PROJECT = "ola"
DEFAULT_INIT = "ao.iolens.chat.k4.L20-60.cont.u64.s0/step28000"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)


def parent_config(init_from: str) -> dict[str, Any]:
    """The warm-start checkpoint's own recorded config (layers/lora come from HERE, not flags)."""
    meta = ola_root() / "ao_checkpoints" / init_from / "meta.json"
    if not meta.exists():
        raise SystemExit(f"[dt] no meta.json at {meta} — warm start needs the parent's config")
    return dict(json.loads(meta.read_text())["config"])


def _worker(  # type: ignore[no-untyped-def]
    local_rank: int,
    world_size: int,
    config_json: str,
    meta_json: str,
    train_pairs,
    val_pairs,
    dist_json: str,
) -> None:
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoTokenizer

    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.distill_shards import DistillShardMeta
    from oracle_lens.pipeline.gt_train import (
        DistillDataset,
        GTConfig,
        build_distill_examples,
        train_gt,
    )
    from oracle_lens.pipeline.verbalizer import renderer_for

    cfg = GTConfig.from_dict(json.loads(config_json))
    meta = DistillShardMeta.from_metadata(json.loads(meta_json))
    dist = json.loads(dist_json)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ex = build_distill_examples(train_pairs, cfg)
    val_cfg = replace(cfg, n_examples=len(val_pairs) + 1, skip_examples=0, skip_batches=0)
    val_ex = build_distill_examples(val_pairs, val_cfg)
    train_data = DistillDataset(train_pairs, train_ex)
    val_data = DistillDataset(val_pairs, val_ex)
    if local_rank == 0:
        print(
            f"[dt] {len(train_data)} train / {len(val_data)} val examples "
            f"(variant={meta.variant} prompt_mode={meta.prompt_mode})",
            flush=True,
        )

    model = load_causal_lm(
        MODEL_ID, dtype=torch.bfloat16, device="cuda",
        attn_implementation=os.environ.get("OLA_ATTN_IMPL", "flash_attention_2"),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if not cfg.init_from:
        raise SystemExit("[dt] distill students must warm-start (--init-from)")
    peft_model: Any = PeftModel.from_pretrained(
        model,
        str(ola_root() / "ao_checkpoints" / cfg.init_from / "lora"),
        is_trainable=True,
    )
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if local_rank == 0:
        print(f"[dt] warm-started LoRA from {cfg.init_from} "
              f"({n_train / 1e6:.1f}M trainable params, base frozen)", flush=True)
    for p_ in peft_model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()

    render = renderer_for(meta.prompt_mode)
    prompts = [render(tokenizer, layer=ly) for ly in cfg.layers]
    pad_id = int(tokenizer.pad_token_id or 0)
    if local_rank == 0:
        print(f"[dt] prompt ({meta.prompt_mode}): "
              f"{tokenizer.decode(prompts[0].input_ids)!r}", flush=True)
        ex0 = train_data[0]
        v0 = float(ex0["inject_vec"].norm())
        print(f"[dt] inject example0: |h|={v0:.1f} -> unit*alpha={cfg.alpha} | "
              f"target preview: {tokenizer.decode(ex0['target_ids'][:60].tolist())!r}",
              flush=True)

    torch.cuda.reset_peak_memory_stats()
    summary = train_gt(
        peft_model,
        prompts,
        train_data,
        val_data,
        cfg,
        pad_id=pad_id,
        wandb_project=WANDB_PROJECT,
        tokenizer=tokenizer,
        local_rank=local_rank,
        world_size=world_size,
        ckpt_dir=ola_root() / "ao_checkpoints" / cfg.run_name,
        nodes=int(dist["nodes"]),
        node_rank=int(dist["node_rank"]),
        master_addr=str(dist["master_addr"]),
        master_port=int(dist["master_port"]),
    )
    summary["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if local_rank != 0 or int(dist["node_rank"]) != 0:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="OLA_ROOT-relative dir holding shards/")
    ap.add_argument("--variant", default="omp4")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--init-from", default=DEFAULT_INIT)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--val-token-budget", type=int, default=512,
                    help="small val batches so tiny eval splits don't starve the "
                         "partial-dropping (layer,octave) sampler (r1: 41 rows -> 0 batches)")
    ap.add_argument("--mb-max", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-every-steps", type=int, default=25)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=16000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-gpu", type=int, default=0)
    ap.add_argument("--dry-run", type=int, default=0,
                    help="print config + N decoded prompt/target pairs, then exit (CPU-safe)")
    args = ap.parse_args()

    import torch
    import torch.multiprocessing as tmp

    from oracle_lens.pipeline.distill_shards import load_distill_shards
    from oracle_lens.pipeline.gt_train import GTConfig
    from oracle_lens.pipeline.multilayer import LAYERS

    root = ola_root()
    shards_dir = root / args.out_dir / "shards" / args.variant
    train_paths = sorted((shards_dir / "train").glob("*.safetensors"))
    val_paths = sorted((shards_dir / "eval").glob("*.safetensors"))
    if not train_paths or not val_paths:
        raise SystemExit(f"[dt] missing distill shards under {shards_dir}/{{train,eval}}")
    train_pairs, train_meta = load_distill_shards(train_paths)
    val_pairs, _ = load_distill_shards(val_paths)
    pcfg = parent_config(args.init_from)
    layers = tuple(int(x) for x in pcfg["layers"])
    max_len = int(train_pairs.lengths.max().item()) + 1
    print(
        f"[dt] {len(train_pairs)} train / {len(val_pairs)} val rows "
        f"(teacher={train_meta.teacher_run}, prompt_mode={train_meta.prompt_mode}); "
        f"layers={list(layers)}; max target len {max_len - 1}",
        flush=True,
    )

    cfg = GTConfig(
        run_name=args.run_name,
        transform="unit",  # norm-matched: injected norm == alpha (matches generation + RL read)
        layers=layers,
        alpha=args.alpha,
        n_examples=len(train_pairs),
        min_len=1,
        max_len=max_len,
        lr=args.lr,
        lora_r=int(pcfg["lora_r"]),
        lora_alpha=int(pcfg.get("lora_alpha", 32)),
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        eval_every_steps=args.eval_every_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        init_from=args.init_from,
        token_budget=args.token_budget,
        val_token_budget=args.val_token_budget,
        mb_max=args.mb_max,
        num_workers=args.num_workers,
        extra={
            "variant": args.variant,
            "teacher_run": train_meta.teacher_run,
            "prompt": train_meta.prompt_mode,
            "shards_dir": str(shards_dir),
            "parent_transform": str(pcfg.get("transform")),
        },
    )

    if args.dry_run:
        from transformers import AutoTokenizer

        from oracle_lens.pipeline.verbalizer import renderer_for

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        render = renderer_for(train_meta.prompt_mode)
        print("\n===== EXACT RUN CONFIG =====")
        print(json.dumps(cfg.to_dict(), indent=2))
        print("\n===== EXAMPLE INPUT -> OUTPUT PAIRS =====")
        for i in range(min(args.dry_run, len(train_pairs))):
            ly = LAYERS[int(train_pairs.layer_idx[i])]
            pr = render(tok, layer=ly)
            v = train_pairs.vec[i].float()
            print(f"\n--- row {i}: layer {ly} ---")
            print(f"INPUT  (loss masked; slot {pr.slot} <- {args.alpha:g} * h/|h|, "
                  f"|h|={float(v.norm()):.1f}):")
            print(f"   {tok.decode(pr.input_ids)!r}")
            print(f"OUTPUT (target, loss on all {int(train_pairs.lengths[i])} tokens):")
            print(f"   {tok.decode(train_pairs.row_target(i).tolist())!r}")
        print("\n[dry-run] no training started")
        return

    n_gpu = args.n_gpu or torch.cuda.device_count()
    dist_json = json.dumps(
        {"nodes": 1, "node_rank": 0, "master_addr": "127.0.0.1",
         "master_port": int(os.environ.get("MASTER_PORT", "0"))}
    )
    config_json = json.dumps(cfg.to_dict())
    meta_json = json.dumps(train_meta.to_metadata())
    if n_gpu > 1:
        train_pairs.share_memory_()
        val_pairs.share_memory_()
        tmp.spawn(  # type: ignore[attr-defined,no-untyped-call]
            _worker,
            args=(n_gpu, config_json, meta_json, train_pairs, val_pairs, dist_json),
            nprocs=n_gpu,
            join=True,
        )
    else:
        _worker(0, 1, config_json, meta_json, train_pairs, val_pairs, dist_json)


if __name__ == "__main__":
    main()
