"""Merge a text-arch LoRA (AO / OLens) into the FULL VL model so SGLang can load it.

Both oracles' LoRAs were trained on ``AutoModelForCausalLM`` = ``Qwen3_5ForCausalLM`` (a text-only
extraction: layers at ``model.layers.{i}.…``, no vision tower), so ``save_pretrained`` after a merge
writes config arch ``Qwen3_5ForCausalLM`` — which this SGLang build has no implementation for. The
base repo serves only because its arch is ``Qwen3_5ForConditionalGeneration`` (language model at
``model.language_model.layers.{i}.…`` plus a ``model.visual.*`` tower), which the input-embeds patch
handles.

So we merge the LoRA deltas into the FULL VL model at the **language-model-mapped keys** and save it
with the VL config: the language-model weights get the adapter, the vision tower is untouched, and
the arch stays servable.

This module is the pure, testable core (key mapping + delta assembly). ``merge_lora_into_vl`` (which
needs the loaded model) lives here too but runs only inside the Modal driver.
"""

import re
from typing import Any

from torch import Tensor

__all__ = ["compute_deltas", "merge_lora_into_vl", "text_module_to_vl_suffix"]

_PEFT_PREFIX = "base_model.model."
# a LoRA tensor key: base_model.model.<module path>.lora_(A|B).weight
_LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")


def text_module_to_vl_suffix(module_path: str) -> str:
    """``base_model.model.model.layers.{i}.{sub}.{mod}`` -> VL weight suffix
    ``language_model.layers.{i}.{sub}.{mod}.weight`` (matched against the VL model by ``endswith``).

    The text extraction nests the decoder directly under ``model.layers``; the VL model nests it
    under ``model.language_model.layers`` — so we insert ``language_model.`` before ``layers``.

    Checkpoints trained with ``compile_blocks`` carry a ``._orig_mod.`` segment in every key
    (``layers.{i}._orig_mod.{sub}.{mod}`` — the repo-wide LoRA-key convention, see
    ``ola/ar_loader.py``); it names torch.compile's wrapper, not a real module, so it is
    stripped here — otherwise every delta of a compiled checkpoint (e.g. the AO ladder runs)
    matches no VL weight and the merge dies with "N deltas never matched".
    """
    m = module_path.replace("._orig_mod.", ".")
    if m.startswith(_PEFT_PREFIX):
        m = m[len(_PEFT_PREFIX) :]
    if not m.startswith("model.layers."):
        raise ValueError(f"unexpected LoRA module path {module_path!r} (want model.layers.…)")
    inner = m[len("model.") :]  # layers.{i}.{sub}.{mod}
    return f"language_model.{inner}.weight"


def compute_deltas(
    lora_sd: dict[str, Tensor], *, scaling: float
) -> dict[str, Tensor]:
    """Pair each module's ``lora_A``/``lora_B`` -> ``{vl_suffix: scaling*(B@A)}`` ([out, in])."""
    a: dict[str, Tensor] = {}
    b: dict[str, Tensor] = {}
    for key, t in lora_sd.items():
        mo = _LORA_KEY.match(key)
        if mo is None:
            continue
        (a if mo["ab"] == "A" else b)[mo["module"]] = t
    if set(a) != set(b):
        raise ValueError(f"lora_A/lora_B modules differ: {set(a) ^ set(b)}")
    deltas: dict[str, Tensor] = {}
    for module, ta in a.items():
        tb = b[module]  # [out, r]
        delta = scaling * (tb.float() @ ta.float())  # [out, in]
        deltas[text_module_to_vl_suffix(module)] = delta
    return deltas


def merge_lora_into_vl(model: Any, lora_dir: Any, *, verbose: bool = True) -> int:
    """Add the LoRA deltas onto ``model``'s VL language-model weights IN PLACE. Returns #merged.

    ``model`` is a loaded ``Qwen3_5ForConditionalGeneration``; ``lora_dir`` holds
    ``adapter_config.json`` + ``adapter_model.safetensors``. Every delta must map to exactly one VL
    parameter of matching shape (asserted) — a silent miss would corrupt the merge.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    lora_dir = Path(lora_dir)
    cfg = json.loads((lora_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    lora_sd = load_file(str(lora_dir / "adapter_model.safetensors"))
    deltas = compute_deltas(lora_sd, scaling=scaling)

    # match VL params by name suffix (robust to the leading ``model.``)
    params = dict(model.named_parameters())
    merged = 0
    with torch.no_grad():
        for suffix, delta in deltas.items():
            matches = [n for n in params if n.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(
                    f"suffix {suffix!r} matched {len(matches)} VL params: {matches[:3]}"
                )
            p = params[matches[0]]
            if tuple(p.shape) != tuple(delta.shape):
                raise ValueError(
                    f"shape {matches[0]}: {tuple(p.shape)} vs {tuple(delta.shape)}")
            p.add_(delta.to(p.dtype).to(p.device))
            merged += 1
    if verbose:
        print(f"[merge-vl] scaling={scaling} merged {merged}/{len(deltas)} deltas", flush=True)
    if merged != len(deltas):
        raise ValueError(f"only merged {merged}/{len(deltas)} deltas")
    return merged
