"""Parity between the standalone gen worker's inline helpers and ``ola.rollout_store``.

``iolens_rollout_gen.py`` runs in the sglang venv (no ``global_workspace`` install), so its
seed-hash and shard layout are inlined. These tests pin the inline copies to the canonical
module — if either drifts, capture would silently mispair seeds with splits or hashes.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from oracle_lens.pipeline.rollout_store import (
    SPLITS,
    RolloutShardMeta,
    load_rollout_shards,
    seed_hash64,
)

REPO = Path(__file__).resolve().parents[1]


def _load_gen_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "iolens_rollout_gen", REPO / "scripts" / "datagen" / "iolens_rollout_gen.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["iolens_rollout_gen"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_inline_seed_hash_matches_rollout_store() -> None:
    gen = _load_gen_module()
    for key in ("wildchat:abc123:0", "fineweb-edu:doc-42", "日本語のシード"):
        assert gen.seed_hash64(key) == seed_hash64(key)
    assert gen.SPLITS == SPLITS


def test_gen_pack_shard_loads_via_rollout_store(tmp_path: Path) -> None:
    """A shard packed by the gen worker's inline writer round-trips through the canonical
    loader with identical ids, prompt/output boundaries, splits, and exact counts."""
    gen = _load_gen_module()
    rows = [
        {
            "key": f"wildchat:conv{i}:0",
            "split": i % len(SPLITS),
            "prompt_ids": list(range(10 + i)),
            "output_ids": list(range(100, 100 + 20 + i)),
            "output_logprobs": [-0.5] * (20 + i),
            "degen": False,
        }
        for i in range(6)
    ]
    # one degenerate row that must be dropped and counted
    rows.append(
        {
            "key": "wildchat:degen:0",
            "split": 0,
            "prompt_ids": [1, 2, 3],
            "output_ids": [7] * 5,
            "output_logprobs": [-0.1] * 5,
            "degen": True,
        }
    )
    part = tmp_path / "rollouts_0000.part0000.json"
    part.write_text(json.dumps(rows))
    meta_base = {
        "model_id": "test-model",
        "mode": "chat",
        "engine": "sglang",
        "engine_version": "0",
        "tokenizer_sha": "cafebabe",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new": 512,
        "git_commit": "test",
    }
    out = tmp_path / "rollouts_0000.safetensors"
    gen.pack_shard(
        out, [part], meta_base, max_new=512, degen_fail_rate=0.5,
        report_path=tmp_path / "reports" / "rollouts_0000.json",
    )
    rolls = load_rollout_shards([out])
    assert len(rolls) == 6  # degen row dropped
    for i in range(6):
        assert rolls.prompt_ids(i).tolist() == rows[i]["prompt_ids"]
        assert rolls.output_ids(i).tolist() == rows[i]["output_ids"]
        assert int(rolls.split_id[i]) == rows[i]["split"]
        assert int(rolls.seed_hash[i]) == seed_hash64(str(rows[i]["key"]))
    counted = rolls.counts()
    meta = rolls.metas[0]
    assert counted["n_prompt_tokens"] == meta.n_prompt_tokens
    assert counted["n_output_tokens"] == meta.n_output_tokens
    assert isinstance(meta, RolloutShardMeta)
    report = json.loads((tmp_path / "reports" / "rollouts_0000.json").read_text())
    assert report["n_degenerate_dropped"] == 1 and report["n_kept"] == 6
    # engine logprobs stored ragged in conv order
    from safetensors import safe_open

    with safe_open(str(out), framework="pt") as f:
        lp = f.get_tensor("out_logprob")
    assert int(lp.shape[0]) == int(rolls.output_lengths().sum())
    assert torch.allclose(lp.float(), torch.full_like(lp.float(), -0.5))


def test_degeneracy_detector() -> None:
    gen = _load_gen_module()
    assert gen.is_degenerate([1] * 10, 16)  # too short
    loop = list(range(30)) + list(range(50, 70)) * 3  # immediate 20-gram repeat
    assert gen.is_degenerate(loop, 16)
    healthy = list(range(200))
    assert not gen.is_degenerate(healthy, 16)
