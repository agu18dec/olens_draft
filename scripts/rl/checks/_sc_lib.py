"""Shared helpers for the rl_sc gate ladder (engine-free: everything in-process).

Loads the SAME actor construction as train_rl_ao.py (base + AO LoRA as 'default'
+ frozen 'reference' + embedding-replacement hook) so the gates exercise the
real code path, not a simulacrum. Reuses scripts/rl/checks/_lib.py for
resolve_snapshot / spliced_embeds / hf_greedy / verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                      # this dir
sys.path.insert(0, str(_HERE.parent))               # rl_sc/
sys.path.insert(0, str(_HERE.parents[2] / "src"))   # global_workspace

from _lib import (  # noqa: E402, F401  (re-exported)
    hf_greedy,
    resolve_snapshot,
    spliced_embeds,
    verdict,
)
from hooks import register_embed_injection_hook  # noqa: E402
from sidecar import load_sidecar  # noqa: E402


def load_actor(base_ckpt: str, ao_lora: str, sidecar_path: str, device: str = "cuda:0"):
    """Actor exactly as the trainer builds it. Returns (actor, tok, cfg, vectors_ref, eos_ids)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_ckpt)
    cfg = load_sidecar(sidecar_path, tok)
    base = AutoModelForCausalLM.from_pretrained(
        resolve_snapshot(base_ckpt), dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    actor = PeftModel.from_pretrained(base, ao_lora, adapter_name="default", is_trainable=True)
    actor.load_adapter(ao_lora, adapter_name="reference")
    actor.set_adapter("default")
    for _n, _p in actor.named_parameters():
        if ".reference." in _n:
            _p.requires_grad_(False)
    actor.eval()
    vectors_ref = [None]
    register_embed_injection_hook(actor, vectors_ref, cfg.injection_token_id,
                                  cfg.left_neighbor_id, cfg.right_neighbor_id)
    eos_ids = {tok.eos_token_id}
    _gc = getattr(getattr(actor, "generation_config", None), "eos_token_id", None)
    if _gc is not None:
        eos_ids.update(_gc if isinstance(_gc, (list, tuple)) else [_gc])
    eos_ids.discard(None)
    eos_ids.add(cfg.injection_token_id)
    return actor, tok, cfg, vectors_ref, eos_ids


@torch.no_grad()
def hook_greedy(actor, prompt_ids, vec, vectors_ref, eos_ids, pad_id, max_new, device):
    """Greedy continuation ids through the trainer's rollout path (hook injection)."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    vectors_ref[0] = vec.unsqueeze(0).to(device).float() if vec is not None else None
    try:
        out = actor.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new, do_sample=False,
            top_p=1.0, top_k=0, repetition_penalty=1.0,
            pad_token_id=pad_id, eos_token_id=sorted(eos_ids),
            return_dict_in_generate=True,
        )
    finally:
        vectors_ref[0] = None
    resp = out.sequences[0, ids.shape[1]:].tolist()
    n_real = next((i + 1 for i, t in enumerate(resp) if t in eos_ids), len(resp))
    return resp[:n_real]


def load_gate_rows(parquet: str, n: int, *, per_layer_min: int = 0) -> list[dict]:
    """First n rows of the gate parquet (all columns, incl. slot/prompt_text)."""
    import pyarrow.parquet as pq

    t = pq.read_table(parquet)
    rows = []
    for i in range(min(n, t.num_rows)):
        rows.append({c: t.column(c)[i].as_py() for c in t.column_names})
    return rows
