"""Embedding-replacement injection hook for the self-contained AO RL trainer.

Written in the style of skip-lens @ 13547ae nla/utils/hooks.py, but the design is
inverted per the audit's P1: ONE forward hook on `get_input_embeddings()` doing
embedding REPLACEMENT with pre-transformed vectors (our contract), replacing the
source's two-hook design (embed_hook stashing ids + layers[1] Karvonen ADD).

Why the embedding seam:
  - it sees ids and embeddings in one place (no cross-hook state),
  - it is BELOW both PEFT adapters, so `default` and `reference` forwards inject
    identically (the KL anchor sees the same input),
  - `all-linear` LoRA never targets nn.Embedding, so the module is a plain
    frozen embedding and the overwrite kills no trainable gradient,
  - it is exactly where the sglang rollout stack splices (ao_generate.py calls
    the same-ancestry inject_at_marked_positions on the embedding table output),
    so rollout ≡ training injection by code identity.
"""

from __future__ import annotations

from sl_injection import inject_at_marked_positions


def register_embed_injection_hook(model, vectors_ref, inj_id, left_id, right_id):
    """Register the replacement hook on the model's input embedding module.

    vectors_ref: single-element list; [0] is a [N, d] tensor while a forward
    that should inject is running, else None. The count assert inside
    inject_at_marked_positions requires N == (# neighbor-valid marker sites in
    the batch) — zero markers with vectors set is a hard error by design.

    Decode steps (cached generation, ids shape [B, 1]) are skipped: a single
    token can never satisfy the left/right neighbor check, and the marker is a
    stop token so it cannot recur mid-response anyway.
    """
    embed = model.get_input_embeddings()

    def _hook(module, args, kwargs, output):
        v = vectors_ref[0]
        if v is None:
            return output
        ids = None
        if args:
            ids = args[0]
        elif kwargs:
            ids = kwargs.get("input")
        if ids is None or ids.ndim < 2 or ids.shape[-1] < 2:
            return output  # decode step under KV cache — nothing to inject
        return inject_at_marked_positions(
            ids.to(output.device), output, v, inj_id, left_id, right_id,
        )

    handle = embed.register_forward_hook(_hook, with_kwargs=True)
    print(f"[hook] embedding-replacement injection on {type(embed).__name__} "
          f"(inj_id={inj_id}, neighbors=({left_id},{right_id}))", flush=True)
    return handle
