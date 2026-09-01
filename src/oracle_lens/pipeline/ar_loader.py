"""Canonical loader for the layer-conditioned multilayer AR (the checkpoints AOs train from).

One copy of the load recipe that was previously duplicated across ``ola_modal.py`` eval jobs
(``bucket_eval`` / ``metric_grid`` / ``fve_attribution``) and ``gt_fve_ml.py``:

    load bf16 27B -> truncate_backbone(max(LAYERS)) -> compile-wrap IF the adapter carries
    ``_orig_mod.`` keys -> PeftModel.from_pretrained -> LayerConditionedReconstructor + heads.pt

The compile-wrap step is load-bearing for every crop32 rung (trained with
``cfg.compile_blocks=True``): without it PEFT silently matches nothing and the checkpoint scores
at chance (``scorer.py`` gotcha). ``fetch_ar_checkpoint`` resolves a run local-first, else pulls
it from the HF checkpoint repo — call it from a login/CPU context, never inside HF-offline GPU
jobs.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from oracle_lens.pipeline.multilayer import LAYERS

if TYPE_CHECKING:
    from oracle_lens.core.whitening import Whitener
    from oracle_lens.pipeline.multilayer_reconstructor import LayerConditionedReconstructor

# `multilayer_reconstructor` imports the vendored TRAINER (mytorch_lightning -> pydra, wandb,
# loguru, tabulate, torchdata), and `whitening`/`reconstructor` pull their own weight. Only
# `load_lc_reconstructor` and `load_ladder_whiteners` need any of that — the adapter helpers
# (`compile_blocks_for_adapter`, `verify_adapter_live`, `load_ao_adapter`) do not. Importing it
# at module level made every AO INFERENCE path depend on the training stack, which is how a
# readout job on a slim image died with `No module named 'wandb'`. So these imports are local to
# the functions that use them; the annotations above are TYPE_CHECKING-only.

AR_REPO = "agu18dec/qwen3.6-27b-mlayer-ar-checkpoints"
MODEL_ID = "Qwen/Qwen3.6-27B"


def compile_blocks_if_ckpt_needs(inner: Any, lora_dir: Path) -> bool:
    """Compile-wrap backbone blocks BEFORE ``PeftModel.from_pretrained`` when the saved adapter
    keys carry the ``_orig_mod.`` prefix (checkpoint trained with ``cfg.compile_blocks``) —
    otherwise the LoRA keys silently fail to match. No-op for uncompiled-training adapters."""
    from safetensors import safe_open

    st = next(iter(sorted(lora_dir.glob("adapter_model.safetensors"))), None)
    if st is None:
        return False
    with safe_open(str(st), framework="pt") as f:
        needs = any("_orig_mod." in k for k in f.keys())  # noqa: SIM118 — not a dict
    if needs:
        for i in range(len(inner.layers)):
            inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
        print(f"[ar-load] compiled {len(inner.layers)} blocks (_orig_mod ckpt)", flush=True)
    return needs


def compile_blocks_for_adapter(base: Any, lora_dir: Path) -> bool:
    """Compile-wrap a FULL causal LM's decoder blocks iff ``lora_dir``'s adapter needs it.

    The sibling ``compile_blocks_if_ckpt_needs`` takes the already-truncated AR backbone; this one
    finds the decoder stack inside a whole model the way the trainer did (``ao_train_cluster.py``:
    the ``ModuleList`` whose parent also has ``norm``). Matching the trainer structurally, not by
    attribute path, is what makes the compiled key names line up.

    Call BEFORE any ``PeftModel.from_pretrained`` / ``load_adapter`` on that model.
    """
    import torch.nn as nn
    from safetensors import safe_open

    st = lora_dir / "adapter_model.safetensors"
    if not st.exists():
        return False
    with safe_open(str(st), framework="pt") as f:
        needs = any("_orig_mod." in k for k in f.keys())  # noqa: SIM118 — not a dict
    if not needs:
        return False
    for module in base.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and hasattr(module, "norm"):
            for i in range(len(layers)):
                layers[i] = torch.compile(layers[i], dynamic=False)  # type: ignore[assignment]
            print(f"[ao-load] compiled {len(layers)} blocks (_orig_mod ckpt)", flush=True)
            return True
    raise RuntimeError(
        f"{lora_dir}: adapter needs compiled blocks but no decoder ModuleList was found — "
        "loading it would silently match nothing."
    )


def verify_adapter_live(peft: Any, lora_dir: Path, *, adapter: str = "") -> int:
    """Assert a loaded adapter's tensors actually map onto the model; return how many matched.

    ``PeftModel.from_pretrained`` does not complain when the checkpoint's key names miss the
    model's (compiled-vs-uncompiled blocks), so without this an inert LoRA is indistinguishable
    from a working one at inference — no loss, just plausible base-model text.
    """
    from safetensors import safe_open

    live = {n for n, _ in peft.named_parameters() if "lora_" in n}
    if not live:
        raise RuntimeError(f"{lora_dir}: model exposes NO lora_ parameters — the adapter is inert.")
    st = lora_dir / "adapter_model.safetensors"
    if not st.exists():
        return len(live)
    with safe_open(str(st), framework="pt") as f:
        saved = {k for k in f.keys() if "lora_" in k}  # noqa: SIM118 — not a dict

    def stem(k: str) -> str:  # PEFT decorates names with a base_model prefix + adapter suffix
        s = k.split("base_model.model.")[-1].replace(".weight", "")
        for suffix in (".default", f".{adapter}" if adapter else "\0"):
            s = s.replace(suffix, "")
        return s

    matched = {stem(k) for k in saved} & {stem(n) for n in live}
    if len(matched) < 0.9 * len(saved):
        raise RuntimeError(
            f"{lora_dir}: only {len(matched)}/{len(saved)} saved LoRA tensors map onto the model — "
            "a key-prefix mismatch would leave most of the adapter unused."
        )
    print(f"[ao-load] adapter verified: {len(matched)}/{len(saved)} tensors live", flush=True)
    return len(matched)


def load_ao_adapter(base: Any, lora_dir: Path) -> Any:
    """Attach a trained AO adapter to a full causal LM — compile-wrapped and VERIFIED.

    Every AO inference path needs this. An adapter trained with ``compile_blocks=on`` stores keys
    like ``...layers.0._orig_mod.linear_attn.in_proj_a.lora_A.weight``; loading it onto an
    uncompiled model matches **nothing**, and ``PeftModel.from_pretrained`` reports no error — you
    get the base model wearing a LoRA-shaped hat. That failure is invisible in the loss (there is
    none at inference) and reads as "the oracle ignored the activation": the 2026-07-30 probe came
    back with base-model essays about what activation vectors are, one noting the concept tags
    looked empty.

    So: wrap first, then assert the adapter actually landed.
    """
    from peft import PeftModel

    compile_blocks_for_adapter(base, lora_dir)
    peft = PeftModel.from_pretrained(base, str(lora_dir))
    verify_adapter_live(peft, lora_dir)
    peft.eval()
    return peft


def fetch_ar_checkpoint(run_name: str, *, dest: Path, repo: str = AR_REPO) -> Path:
    """Resolve an AR run's checkpoint dir (``lora/`` + ``heads.pt``), local-first.

    Order: ``dest/<run>`` (already fetched) -> sibling ``ml_checkpoints/<run>`` (trained on this
    cluster) -> ``snapshot_download`` from the HF repo into ``dest``. Idempotent.
    """
    local = dest / run_name
    if (local / "heads.pt").exists():
        return local
    sibling = dest.parent / "ml_checkpoints" / run_name
    if (sibling / "heads.pt").exists():
        return sibling
    from huggingface_hub import snapshot_download

    print(f"[ar-load] fetching {repo}/{run_name}", flush=True)
    snapshot_download(repo, local_dir=str(dest), allow_patterns=[f"{run_name}/*"])
    if not (local / "heads.pt").exists():
        raise FileNotFoundError(f"{repo} has no run '{run_name}' (or it lacks heads.pt)")
    return local


def build_reconstructor_from_heads(
    peft: Any,
    layers: tuple[int, ...],
    d_model: int,
    state: dict[str, Any],
    *,
    head_mode: str,
    layer_norm: bool = True,
) -> Any:
    """Construct the right reconstructor head family from a ``heads.pt`` state dict.

    The ONE construction recipe (was inlined in ``iolens_rawfve_rungs.py``): ``prompt_tag``
    -> ``PromptTagReconstructor`` (tag ids from the state, no layer embedding); anything else
    -> ``LayerConditionedReconstructor`` (head + layer_emb). ``layers`` must be the TRAINED
    layer NUMBERS (12 rows for {20,24,...,63}), not indices, not all 17 — checkpoint truth.
    """
    from oracle_lens.pipeline.multilayer_reconstructor import (
        LayerConditionedReconstructor,
        PromptTagReconstructor,
    )

    if head_mode == "prompt_tag":
        recon: Any = PromptTagReconstructor(
            peft, layers, d_model, state["tag_ids"], layer_norm=layer_norm
        )
        recon.head.load_state_dict(state["head"])
    else:
        recon = LayerConditionedReconstructor(peft, layers, d_model, layer_norm=layer_norm)
        recon.head.load_state_dict(state["head"])
        recon.layer_emb.load_state_dict(state["layer_emb"])
    return recon


def _unwrap_compiled_blocks(inner: Any) -> None:
    """Swap torch.compile'd blocks back to eager AFTER an adapter load (see the eager note in
    ``load_lc_reconstructor`` — recompile-mid-service on this stack corrupts memory)."""
    n_unwrapped = 0
    for i in range(len(inner.layers)):
        block = inner.layers[i]
        if hasattr(block, "_orig_mod"):
            inner.layers[i] = block._orig_mod
            n_unwrapped += 1
    if n_unwrapped:
        print(f"[ar-load] eager: unwrapped {n_unwrapped} compiled blocks", flush=True)


def _heads_state_cpu(ckpt_dir: Path) -> dict[str, Any]:
    """CPU-only ``heads.pt`` read — the cheap, authoritative view of an AR's head family."""
    state: dict[str, Any] = torch.load(
        ckpt_dir / "heads.pt", map_location="cpu", weights_only=True
    )
    return state


def ar_head_mode(ckpt_dir: Path) -> str:
    """Head family from ``heads.pt`` keys alone (no 27B load, no meta.json required).

    Dispatch mirrors ``multilayer_reconstructor.head_state``: ``tag_ids`` => prompt_tag,
    ``layer_emb`` => layer_conditioned, a bare ``heads`` dict => read_final. Key inspection is
    the only classifier that also works for the meta-less ``mlayer.lc.*`` finals.
    """
    state = _heads_state_cpu(ckpt_dir)
    if "tag_ids" in state:
        return "prompt_tag"
    if "layer_emb" in state:
        return "layer_conditioned"
    return "read_final"


def ar_layer_set(ckpt_dir: Path) -> tuple[int, ...]:
    """The AR's own layer list from ``heads.pt`` ONLY — cheap enough to call before drawing
    layer picks, so consumers never have to GUESS the universe (the old
    ``len(LAYERS)-1 if layer_min`` inference in the precompute encoded an lc/--drop-layers 0
    assumption that a prompt_tag or layer-max-trained AR silently violates).

    prompt_tag checkpoints carry an explicit ``layers`` list (``head_state``); older saves fall
    back to the tag-row count. lc derives from the embedding row count (the --drop-layers 0
    convention: rows are the LAST n of LAYERS). read_final derives from the head count.
    """
    state = _heads_state_cpu(ckpt_dir)
    if "tag_ids" in state:
        layers = state.get("layers")
        if layers:
            return tuple(int(x) for x in layers)
        return tuple(LAYERS[-int(state["tag_ids"].shape[0]):])
    if "layer_emb" in state:
        n = int(state["layer_emb"]["weight"].shape[0])
        return tuple(LAYERS[-n:]) if n != len(LAYERS) else tuple(LAYERS)
    heads = state.get("heads", state)
    n = 1 + max(int(k.split(".")[0]) for k in heads)
    return tuple(LAYERS[-n:]) if n != len(LAYERS) else tuple(LAYERS)


def _head_layer_norm(ckpt_dir: Path) -> bool:
    """``config.head_layer_norm`` from the rung's meta.json, defaulting True (every iolens rung).

    This CANNOT be inferred from the state dict: the head LayerNorm is
    ``elementwise_affine=False`` (no parameters), so ``load_state_dict`` matches either way and
    a mismatch silently changes the output scale by orders of magnitude.
    """
    import json

    meta = ckpt_dir / "meta.json"
    if meta.exists():
        cfg = json.loads(meta.read_text()).get("config", {})
        return bool(cfg.get("head_layer_norm", True))
    return True


def load_ptag_reconstructor(
    ckpt_dir: Path,
    *,
    model_id: str = MODEL_ID,
    device: str = "cuda",
    attn_implementation: str | None = None,
    eager: bool = False,
) -> Any:
    """Load a frozen prompt-tag AR for inference: ``forward(ids, mask, layer_idx=li) -> [b,1,d]``
    (or ``layer_idx=None`` -> ``[b, n_layers, d]``, one forward per layer).

    Mirrors ``load_lc_reconstructor``; the trained layer set comes from the rung's
    ``meta.json`` (``config.layer_indices`` into the global ``LAYERS``) — it is NOT derivable
    from ``heads.pt`` (no layer embedding), so a rung dir without ``meta.json`` is an error.
    """
    import importlib.util
    import json

    from peft import PeftModel

    from oracle_lens.core.reconstructor import truncate_backbone
    from oracle_lens.model import load_causal_lm

    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{ckpt_dir}: prompt-tag AR needs the rung's meta.json (config.layer_indices) — "
            "point at a rung dir like <run>/ex<N>, not the run root."
        )
    cfg = json.loads(meta_path.read_text())["config"]
    layers = tuple(LAYERS[int(i)] for i in cfg["layer_indices"])

    if attn_implementation is None:
        attn_implementation = (
            "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        )
    base = load_causal_lm(
        model_id, dtype=torch.bfloat16, device=device, attn_implementation=attn_implementation
    )
    inner = truncate_backbone(base, layer=max(LAYERS))
    compile_blocks_if_ckpt_needs(inner, ckpt_dir / "lora")
    peft_inner = PeftModel.from_pretrained(inner, str(ckpt_dir / "lora"))
    if eager:
        _unwrap_compiled_blocks(inner)
    peft_inner.eval()
    d_model = int(getattr(base.config, "text_config", base.config).hidden_size)
    state = torch.load(ckpt_dir / "heads.pt", map_location=device, weights_only=True)
    recon = build_reconstructor_from_heads(
        peft_inner,
        layers,
        d_model,
        state,
        head_mode="prompt_tag",
        layer_norm=bool(cfg.get("head_layer_norm", True)),
    )
    recon = recon.to(device)
    recon.eval()
    return recon


def load_reconstructor(
    ckpt_dir: Path,
    *,
    model_id: str = MODEL_ID,
    device: str = "cuda",
    attn_implementation: str | None = None,
    eager: bool = False,
) -> Any:
    """Load a frozen AR of ANY head family for inference, dispatched on ``heads.pt`` keys.

    ``layer_emb`` => LayerConditionedReconstructor (forward(ids, mask) -> [b, n_layers, d]);
    ``tag_ids``  => PromptTagReconstructor (same all-layer shape with layer_idx=None, or ONE
    tagged forward -> [b, 1, d] with layer_idx=i); anything else raises (read_final finals are
    not consumed by the AO pipeline).

    ``attn_implementation=None`` prefers ``$OLA_ATTN_IMPL`` (the backend the box's env.sh
    resolved and the AR trained under), then auto-picks flash-attention when installed. Pass
    ``eager=True`` when the caller will run RAGGED batch shapes (e.g. the grouped prompt_tag
    arout path): the blocks compile dynamic=False, and a mid-service recompile on this
    torch 2.9.1 + Triton 3.4 + Hopper stack produced cos=NaN then a CUDA illegal memory access
    (repro3b, 2026-08-10). The unwrap preserves the LoRA modules exactly.
    """
    import importlib.util
    import os

    from peft import PeftModel

    from oracle_lens.core.reconstructor import truncate_backbone
    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.multilayer_reconstructor import (
        LayerConditionedReconstructor,
        PromptTagReconstructor,
    )

    if attn_implementation is None:
        attn_implementation = os.environ.get("OLA_ATTN_IMPL") or (
            "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        )
    state = _heads_state_cpu(ckpt_dir)
    layers = ar_layer_set(ckpt_dir)
    base = load_causal_lm(
        model_id, dtype=torch.bfloat16, device=device, attn_implementation=attn_implementation
    )
    # Truncate at the DEEPEST layer this AR reads. For prompt_tag the read is
    # last_hidden_state, so truncation depth IS the read depth — max(layers), not max(LAYERS).
    inner = truncate_backbone(base, layer=max(layers))
    compile_blocks_if_ckpt_needs(inner, ckpt_dir / "lora")
    peft_inner = PeftModel.from_pretrained(inner, str(ckpt_dir / "lora"))
    if eager:
        # Unwrap compiled blocks back to eager AFTER the adapter load (the compile wrapper
        # exists only so the checkpoint's ``_orig_mod.`` keys match; the LoRA modules live
        # inside ``_orig_mod``, so the swap preserves them exactly). See the docstring for why.
        n_unwrapped = 0
        for i in range(len(inner.layers)):
            block = inner.layers[i]
            if hasattr(block, "_orig_mod"):
                inner.layers[i] = block._orig_mod
                n_unwrapped += 1
        if n_unwrapped:
            print(f"[ar-load] eager: unwrapped {n_unwrapped} compiled blocks", flush=True)
    peft_inner.eval()
    d_model = int(getattr(base.config, "text_config", base.config).hidden_size)
    ln = _head_layer_norm(ckpt_dir)
    if len(layers) != len(LAYERS):
        print(
            f"[ar-load] checkpoint has {len(layers)} layer rows (not {len(LAYERS)}) "
            f"-> layers {layers}",
            flush=True,
        )
    recon: Any
    if "tag_ids" in state:
        tag_ids = state["tag_ids"].long()
        recon = PromptTagReconstructor(
            peft_inner, layers, d_model, tag_ids, layer_norm=ln
        )
        recon.head.load_state_dict(state["head"])
        print(
            f"[ar-load] prompt_tag: {tag_ids.shape[0]} tags, width={tag_ids.shape[1]} tok",
            flush=True,
        )
    elif "layer_emb" in state:
        recon = LayerConditionedReconstructor(peft_inner, layers, d_model, layer_norm=ln)
        recon.head.load_state_dict(state["head"])
        recon.layer_emb.load_state_dict(state["layer_emb"])
    else:
        raise ValueError(
            f"{ckpt_dir}/heads.pt has neither tag_ids nor layer_emb — a read_final "
            "checkpoint; the AO pipeline does not consume these"
        )
    recon = recon.to(device)
    recon.eval()
    return recon


def load_lc_reconstructor(
    ckpt_dir: Path,
    *,
    model_id: str = MODEL_ID,
    device: str = "cuda",
    attn_implementation: str | None = None,
    eager: bool = False,
) -> "LayerConditionedReconstructor":
    """Load a frozen layer-conditioned AR for inference: ``forward(ids, mask) -> [b, 17, d]``.

    ``attn_implementation=None`` auto-picks flash-attention when installed (it always is in the
    cluster venv; the fallback exists only for laptops/tests).
    """
    import importlib.util

    from peft import PeftModel

    from oracle_lens.core.reconstructor import truncate_backbone
    from oracle_lens.model import load_causal_lm
    from oracle_lens.pipeline.multilayer_reconstructor import LayerConditionedReconstructor

    if attn_implementation is None:
        attn_implementation = (
            "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        )
    base = load_causal_lm(
        model_id, dtype=torch.bfloat16, device=device, attn_implementation=attn_implementation
    )
    inner = truncate_backbone(base, layer=max(LAYERS))
    compile_blocks_if_ckpt_needs(inner, ckpt_dir / "lora")
    peft_inner = PeftModel.from_pretrained(inner, str(ckpt_dir / "lora"))
    if eager:
        # Unwrap torch.compile'd blocks back to eager AFTER the adapter load (the compile
        # wrapper exists only so the checkpoint's ``_orig_mod.`` keys match; the LoRA
        # modules live inside ``_orig_mod``, so the swap preserves them exactly).
        # Load-bearing for the RL reward worker: the blocks are compiled dynamic=False,
        # every new (batch, width) triggers a full Dynamo recompile, and on this
        # torch 2.9.1 + Triton 3.4 + Hopper stack a recompile mid-service produced
        # cos=NaN then CUDA illegal memory access (repro3b, 2026-08-10 — first compiled
        # call fine, first re-shaped call dead; the same toolchain family as fla #640).
        # Fixed-shape batch evals never recompile, hence eval pipelines never saw it.
        _unwrap_compiled_blocks(inner)
    peft_inner.eval()
    d_model = int(getattr(base.config, "text_config", base.config).hidden_size)
    state = torch.load(ckpt_dir / "heads.pt", map_location=device, weights_only=True)
    # Derive the layer set FROM THE CHECKPOINT. The iolens AR trains with --drop-layers 0 (layer
    # 0's activations are essentially "which token preceded this span" and score at chance), so its
    # layer embedding is [16, 5120] while LAYERS has 17 entries. Assuming 17 made every AO
    # component fail with a size-mismatch RuntimeError at load; assuming it silently would be
    # worse, since the layer<->row mapping would be off by one for every layer above the dropped
    # one. The embedding's row count is authoritative.
    n_emb = int(state["layer_emb"]["weight"].shape[0])
    layers = tuple(LAYERS[-n_emb:]) if n_emb != len(LAYERS) else tuple(LAYERS)
    if n_emb != len(LAYERS):
        print(f"[ar-load] checkpoint has {n_emb} layer rows (not {len(LAYERS)}) -> layers {layers}",
              flush=True)
    recon: LayerConditionedReconstructor = build_reconstructor_from_heads(
        peft_inner, layers, d_model, state, head_mode="lc", layer_norm=True
    )
    recon = recon.to(device)
    recon.eval()
    return recon


def load_ladder_whiteners(
    whitener_dir: Path,
    *,
    prefix: str,
    layers: tuple[int, ...] = LAYERS,
    ridge_c: float = 0.1,
    device: str = "cpu",
) -> dict[int, "Whitener"]:
    """One frozen whitener per layer from ``<dir>/<prefix>_L{layer}.safetensors``.

    ``prefix`` must be the family the AR rung TRAINED with (recorded in its run config as
    ``whitener_prefix``) — FVE numbers in a mismatched basis are incomparable (runbook,
    metric-freeze entry).
    """
    from oracle_lens.core.whitening import Whitener, load_whitener

    out: dict[int, Whitener] = {}
    for layer in layers:
        path = whitener_dir / f"{prefix}_L{layer}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"whitener missing: {path}")
        out[layer] = load_whitener(path, ridge_c=ridge_c, device=device)
    return out
