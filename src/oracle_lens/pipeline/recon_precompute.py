"""Frozen-AR precompute: span ids -> reconstructed activations (the long-oracle-lens targets).

The long oracle lens trains on (RECONSTRUCTED activation, layer, subsequent text): the frozen
layer-conditioned multilayer AR maps each bare span to its predicted residual at ``prev_pos``
for all 17 ``LAYERS``, and those predictions are stored as the ``targets`` stack of ordinary
``multilayer_v1`` shards (``targets_kind="recon"``), so ``load_multilayer_shards_lazy`` and the
whole GT-training path consume them unchanged.

AR input convention (exactly ``gt_fve_ml.py``/training): BARE span token ids, right-padded, no
chat template, no rollout context — the AR reads the span text alone.
"""

from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor

from oracle_lens.pipeline.multilayer import LAYERS
from oracle_lens.pipeline.multilayer_reconstructor import LayerConditionedReconstructor
from oracle_lens.pipeline.span_store import SpanOnlyPairs


def load_layer_conditioned_ar(
    ar_ckpt: Path, *, device: str = "cuda", attn_implementation: str | None = None
) -> tuple[LayerConditionedReconstructor, Any, int]:
    """Load the frozen AR — either checkpoint layout, returned frozen and eval'd.

    Two on-disk layouts are supported:
    - ``lora/`` (PEFT ``save_pretrained``) + ``heads.pt`` {"head", "layer_emb"} — the interim
      ``ar.asst.on.mlayer.lc.*`` layout (the exact ``gt_fve_ml.py`` recipe);
    - ``resume.pt`` {"lora", "head", "layer_emb", "optimizer", ...} — the training-resume
      bundle of the crop-trained ARs (e.g. ``crop32/1500k``). The LoRA state dict carries
      ``._orig_mod.`` fragments from torch.compile-wrapped blocks; they are stripped before
      loading into the (uncompiled) inference backbone, and the load is verified complete.

    Returns ``(reconstructor.eval(), tokenizer, d_model)``.
    """
    import importlib.util

    from peft import PeftModel, get_peft_model, set_peft_model_state_dict
    from transformers import AutoTokenizer

    from oracle_lens.core.reconstructor import truncate_backbone
    from oracle_lens.model import load_causal_lm

    model_id = "Qwen/Qwen3.6-27B"
    if attn_implementation is None:
        attn_implementation = (
            "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = load_causal_lm(
        model_id, dtype=torch.bfloat16, device=device, attn_implementation=attn_implementation
    )
    inner = truncate_backbone(base, layer=max(LAYERS))
    resume_path = ar_ckpt / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        lora_sd = {k.replace("._orig_mod.", "."): v for k, v in state["lora"].items()}
        lora_cfg = _lora_config_from_run_json(ar_ckpt)
        backbone = get_peft_model(inner, lora_cfg)
        result = set_peft_model_state_dict(backbone, lora_sd)
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise ValueError(f"{len(unexpected)} unexpected LoRA keys, e.g. {unexpected[:3]}")
        n_loaded = sum(1 for k in lora_sd if "lora_" in k)
        n_slots = sum(1 for n, _ in backbone.named_parameters() if "lora_" in n)
        if n_loaded != n_slots:
            raise ValueError(f"LoRA key count mismatch: {n_loaded} in ckpt vs {n_slots} slots")
    else:
        backbone = PeftModel.from_pretrained(inner, str(ar_ckpt / "lora"))
        state = torch.load(ar_ckpt / "heads.pt", map_location=device, weights_only=True)
    backbone.eval()
    d_model = int(getattr(base.config, "text_config", base.config).hidden_size)
    recon = LayerConditionedReconstructor(backbone, LAYERS, d_model, layer_norm=True)
    recon.head.load_state_dict(state["head"])
    recon.layer_emb.load_state_dict(state["layer_emb"])
    if recon.layer_emb.weight.shape[0] != len(LAYERS):
        raise ValueError(
            f"layer_emb covers {recon.layer_emb.weight.shape[0]} layers, expected {len(LAYERS)}"
        )
    recon = recon.to(device)
    recon.eval()
    for p in recon.parameters():
        p.requires_grad_(False)
    return recon, tokenizer, d_model


def _lora_config_from_run_json(ar_ckpt: Path) -> Any:
    """LoraConfig matching the AR's training run (falls back to the r16/a32 house recipe)."""
    import json

    from peft import LoraConfig

    r, alpha, dropout = 16, 32, 0.05
    run_json = ar_ckpt / "run.json"
    if run_json.exists():
        cfg = json.loads(run_json.read_text()).get("config", {})
        r = int(cfg.get("lora_r", r))
        alpha = int(cfg.get("lora_alpha", alpha))
        dropout = float(cfg.get("lora_dropout", dropout))
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules="all-linear", bias="none"
    )


def token_budget_batches(
    lengths: Int[Tensor, " n"], *, token_budget: int, mb_max: int = 256
) -> list[list[int]]:
    """Length-sorted greedy batches whose PADDED size (rows x batch-max len) <= token_budget.

    Sorting by length makes padding waste ~0; each batch is a contiguous run of the sorted
    order. A single row longer than the budget still forms its own batch.
    """
    order = torch.argsort(lengths).tolist()
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        length = int(lengths[i])
        new_max = max(cur_max, length)
        if cur and ((len(cur) + 1) * new_max > token_budget or len(cur) >= mb_max):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = length
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def precompute_targets(
    recon: LayerConditionedReconstructor,
    pairs: SpanOnlyPairs,
    *,
    pad_id: int,
    token_budget: int = 24_576,
    device: str = "cuda",
    log_every: int = 200,
) -> Float[Tensor, "n n_layers d"]:
    """Frozen-AR predictions for every span, returned in ROW ORDER as bf16 ``[n, 17, d]``."""
    n = len(pairs)
    d = int(recon.layer_emb.weight.shape[1])
    out = torch.empty(n, len(recon.layers), d, dtype=torch.bfloat16)
    batches = token_budget_batches(pairs.lengths, token_budget=token_budget)
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            rows = [pairs.row_ids(i) for i in batch]
            width = max(int(r.shape[0]) for r in rows)
            ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
            mask = torch.zeros(len(rows), width, dtype=torch.long)
            for j, r in enumerate(rows):
                ids[j, : int(r.shape[0])] = r
                mask[j, : int(r.shape[0])] = 1
            preds: Tensor = recon(ids.to(device), mask.to(device))  # [b, n_layers, d] fp32
            out[torch.tensor(batch, dtype=torch.long)] = preds.to(torch.bfloat16).cpu()
            if log_every and (bi + 1) % log_every == 0:
                done = sum(len(b) for b in batches[: bi + 1])
                peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                print(
                    f"[precompute] batch {bi + 1}/{len(batches)} ({done}/{n} rows, "
                    f"peak {peak:.1f} GB)",
                    flush=True,
                )
    return out
