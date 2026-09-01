"""CPU tests for the VL-remap key mapping + delta assembly (the model merge is Modal-only)."""

import pytest
import torch

from oracle_lens.pipeline.merge_vl import compute_deltas, text_module_to_vl_suffix


def test_text_module_to_vl_suffix() -> None:
    assert (
        text_module_to_vl_suffix("base_model.model.model.layers.0.linear_attn.in_proj_a")
        == "language_model.layers.0.linear_attn.in_proj_a.weight"
    )
    assert (
        text_module_to_vl_suffix("base_model.model.model.layers.44.self_attn.q_proj")
        == "language_model.layers.44.self_attn.q_proj.weight"
    )
    # already-stripped path also works
    assert (
        text_module_to_vl_suffix("model.layers.3.mlp.gate_proj")
        == "language_model.layers.3.mlp.gate_proj.weight"
    )


def test_text_module_to_vl_suffix_rejects_unexpected() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        text_module_to_vl_suffix("base_model.model.lm_head")


def test_compute_deltas_pairs_and_scales() -> None:
    mod = "base_model.model.model.layers.0.self_attn.q_proj"
    a = torch.ones(2, 3)  # [r, in]
    b = torch.ones(4, 2) * 2.0  # [out, r]
    sd = {f"{mod}.lora_A.weight": a, f"{mod}.lora_B.weight": b}
    deltas = compute_deltas(sd, scaling=2.0)
    key = "language_model.layers.0.self_attn.q_proj.weight"
    assert set(deltas) == {key}
    # scaling * (B @ A) = 2 * ([4,2]@[2,3]) with all-ones*2 and ones -> each entry 2*(2*1+2*1)=8
    assert deltas[key].shape == (4, 3)
    assert torch.allclose(deltas[key], torch.full((4, 3), 8.0))


def test_compute_deltas_rejects_orphan() -> None:
    sd = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 3)}
    with pytest.raises(ValueError, match="differ"):
        compute_deltas(sd, scaling=1.0)
