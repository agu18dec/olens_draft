"""An inert LoRA must never load quietly.

A checkpoint trained with ``compile_blocks=on`` stores keys like
``model.layers.0._orig_mod.q_proj.lora_A.weight``. Loaded onto an uncompiled model those match
nothing and ``PeftModel.from_pretrained`` says nothing — at inference there is no loss to look
wrong, so the readout is base-model text that reads exactly like the interesting negative result
"the oracle ignores the injected activation". That happened on 2026-07-30 (ao-probe returned
essays about what activation vectors are, and an FVE eval scored a bare base model for 1h45).

These tests pin the two guards that make it loud instead: the prefix DETECTION, and the
post-load VERIFICATION.
"""

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

from oracle_lens.pipeline.ar_loader import (
    compile_blocks_for_adapter,
    verify_adapter_live,
)


class _Blocks(torch.nn.Module):
    """Minimal stand-in for a decoder stack: a `layers` ModuleList beside a `norm`, which is how
    the trainer locates the blocks to compile."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(n))
        self.norm = torch.nn.LayerNorm(4)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Blocks()


def _write_adapter(tmp_path: Path, keys: list[str]) -> Path:
    d = tmp_path / "lora"
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.zeros(2, 4) for k in keys}, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2}))
    return d


def test_compile_wrap_triggers_only_for_orig_mod_checkpoints(tmp_path: Path) -> None:
    plain = _write_adapter(tmp_path / "a", ["base_model.model.layers.0.q_proj.lora_A.weight"])
    compiled = _write_adapter(
        tmp_path / "b", ["base_model.model.layers.0._orig_mod.q_proj.lora_A.weight"]
    )

    m1 = _Model()
    assert compile_blocks_for_adapter(m1, plain) is False
    # an uncompiled checkpoint must be left strictly alone
    assert type(m1.model.layers[0]).__name__ == "Linear"

    m2 = _Model()
    assert compile_blocks_for_adapter(m2, compiled) is True
    assert type(m2.model.layers[0]).__name__ != "Linear", "blocks were not compile-wrapped"


def test_missing_adapter_file_is_not_treated_as_needing_compile(tmp_path: Path) -> None:
    d = tmp_path / "lora"
    d.mkdir(parents=True)
    assert compile_blocks_for_adapter(_Model(), d) is False


def test_no_decoder_stack_raises_rather_than_loading_inert(tmp_path: Path) -> None:
    """If the blocks can't be found, loading would silently match nothing — that must be an error,
    not a shrug."""
    compiled = _write_adapter(
        tmp_path, ["base_model.model.layers.0._orig_mod.q_proj.lora_A.weight"]
    )
    with pytest.raises(RuntimeError, match="no decoder ModuleList"):
        compile_blocks_for_adapter(torch.nn.Linear(4, 4), compiled)


class _FakePeft:
    """Stands in for a loaded PeftModel exposing a given set of lora parameter names."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def named_parameters(self) -> list[tuple[str, torch.Tensor]]:
        return [(n, torch.zeros(1)) for n in self._names]


def test_verify_rejects_a_model_with_no_lora_params(tmp_path: Path) -> None:
    d = _write_adapter(tmp_path, ["base_model.model.layers.0.q_proj.lora_A.weight"])
    with pytest.raises(RuntimeError, match="NO lora_ parameters"):
        verify_adapter_live(_FakePeft([]), d)


def test_verify_rejects_a_prefix_mismatch(tmp_path: Path) -> None:
    """The real bug: saved keys carry _orig_mod, the live model's do not."""
    saved = [f"base_model.model.layers.{i}._orig_mod.q_proj.lora_A.weight" for i in range(10)]
    d = _write_adapter(tmp_path, saved)
    live = [f"base_model.model.layers.{i}.q_proj.lora_A.default.weight" for i in range(10)]
    with pytest.raises(RuntimeError, match="map onto the model"):
        verify_adapter_live(_FakePeft(live), d)


def test_verify_accepts_matching_keys(tmp_path: Path) -> None:
    saved = [f"base_model.model.layers.{i}._orig_mod.q_proj.lora_A.weight" for i in range(10)]
    d = _write_adapter(tmp_path, saved)
    live = [
        f"base_model.model.layers.{i}._orig_mod.q_proj.lora_A.default.weight" for i in range(10)
    ]
    assert verify_adapter_live(_FakePeft(live), d) == 10


def test_load_chat_rollouts_excludes_partial_shards(tmp_path: Path) -> None:
    """A partial shard must never enter a pool build.

    The generator checkpoints in-progress work to `rollouts_<tag><NNNN>.part.json`, which also
    matches the `rollouts_*.json` glob. A finished shard is a superset of its own partial, so
    ingesting both would duplicate source text — the one thing the diversity guarantees exist to
    rule out.
    """
    import json as _json

    from oracle_lens.pipeline.ao_pool import load_chat_rollouts

    d = tmp_path / "chat"
    d.mkdir()
    (d / "rollouts_gen2_0000.json").write_text(
        _json.dumps([{"seed": "a", "prompt_ids": [1], "output_ids": [10, 11]}])
    )
    # same conversation, still in flight — must be ignored
    (d / "rollouts_gen2_0001.part.json").write_text(
        _json.dumps([{"seed": "b", "prompt_ids": [2], "output_ids": [20, 21]}])
    )
    outputs, names = load_chat_rollouts(d)
    assert names == ["rollouts_gen2_0000.json"], names
    assert outputs == [[10, 11]]


def test_crop_hashes_covers_exactly_the_kept_crops() -> None:
    """The delta-build exclusion set must contain a hash for every kept (window, length) crop and
    nothing else — a miss lets the extension run re-see its parent's text."""
    import torch as _t

    from oracle_lens.pipeline.ao_pool import CROP_LENGTHS, AOPool, crop_hashes, span_hash

    ids = _t.arange(2 * 64, dtype=_t.long).reshape(2, 64)
    keep = _t.zeros(2, len(CROP_LENGTHS), dtype=_t.bool)
    keep[0, 1] = True  # row 0 kept at N=4
    keep[1, 3] = True  # row 1 kept at N=16
    pool = AOPool(
        ids=ids,
        conv=_t.tensor([0, 1]),
        start=_t.zeros(2, dtype=_t.long),
        keep=keep,
        meta={},
    )
    h = crop_hashes(pool)
    # prefix CLOSURE: a kept crop contributes its own hash AND every shorter prefix's — that is
    # what blocks a delta crop from being a subset of a parent target (the v1 epoching, cross-pool)
    assert h[4] == {span_hash(ids[0, :4].tolist()), span_hash(ids[1, :4].tolist())}
    assert h[16] == {span_hash(ids[1, :16].tolist())}
    assert h[2] == {span_hash(ids[0, :2].tolist()), span_hash(ids[1, :2].tolist())}
    assert h[8] == {span_hash(ids[1, :8].tolist())}
    for n in (32, 64):
        assert h[n] == set(), f"unexpected hashes at N={n}"


def test_load_chat_rollouts_glob_restricts_sources(tmp_path: Path) -> None:
    import json as _json

    from oracle_lens.pipeline.ao_pool import load_chat_rollouts

    d = tmp_path / "chat"
    d.mkdir()
    (d / "rollouts_0000.json").write_text(
        _json.dumps([{"seed": "old", "prompt_ids": [1], "output_ids": [1, 2]}])
    )
    (d / "rollouts_gen2_0000.json").write_text(
        _json.dumps([{"seed": "new", "prompt_ids": [2], "output_ids": [3, 4]}])
    )
    outputs, names = load_chat_rollouts(d, glob="rollouts_gen2_*.json")
    assert names == ["rollouts_gen2_0000.json"]
    assert outputs == [[3, 4]]


def test_target_prefix_inserts_between_open_tag_and_span() -> None:
    """Dot-variant contract: a target_prefix becomes the FIRST supervised tokens after the
    opening tag, span accounting is untouched, and the empty prefix reproduces the baseline
    target byte-identically (the ladder's runs must be unaffected). Stub tokenizer keeps the
    test hermetic — the dataset only calls tokenizer(text)["input_ids"] and eos_token_id."""
    import torch as _t

    from oracle_lens.pipeline.ao_ladder import AOLadderDataset
    from oracle_lens.pipeline.ao_pool import CROP_LENGTHS, AOPool

    class _Tok:
        eos_token_id = 9999

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            return {"input_ids": [1000 + b for b in text.encode()]}

    tok = _Tok()
    ids = _t.arange(100, 100 + 64, dtype=_t.long).reshape(1, 64)
    keep = _t.zeros(1, len(CROP_LENGTHS), dtype=_t.bool)
    keep[0, 2] = True  # N=8
    pool = AOPool(
        ids=ids, conv=_t.tensor([0]), start=_t.zeros(1, dtype=_t.long), keep=keep, meta={}
    )

    class _FakeArout:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> _t.Tensor:
            return _t.zeros(17, 8)

    base = AOLadderDataset(pool, cast(Any, _FakeArout()), tokenizer=tok, layers_per_crop=1)
    dot = AOLadderDataset(
        pool, cast(Any, _FakeArout()), tokenizer=tok, layers_per_crop=1, target_prefix="."
    )

    t0 = base[0]["target_ids"].tolist()
    t1 = dot[0]["target_ids"].tolist()
    dot_ids = tok(".")["input_ids"]
    open_ids = tok("<explanation>\n")["input_ids"]
    assert t1 == [*open_ids, *dot_ids, *t0[len(open_ids) :]]
    assert int(base[0]["span_len"]) == int(dot[0]["span_len"]) == 8
    base2 = AOLadderDataset(
        pool, cast(Any, _FakeArout()), tokenizer=tok, layers_per_crop=1, target_prefix=""
    )
    assert base2[0]["target_ids"].tolist() == t0
