"""CPU tests for the round-2 self-distillation pipeline (parse/truncate, shards, examples)."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch

from oracle_lens.pipeline.distill import (
    VARIANTS,
    extract_explanation_content,
    phrases_from_samples,
    shortfall_rows,
)
from oracle_lens.pipeline.distill_shards import (
    DistillPairs,
    DistillShardMeta,
    load_distill_shards,
    save_distill_shard,
)
from oracle_lens.pipeline.gt_train import (
    DistillDataset,
    GTConfig,
    buckets_for,
    build_distill_examples,
    make_batch_sampler,
)
from oracle_lens.pipeline.multilayer import LAYERS
from oracle_lens.pipeline.r2_capture import (
    DrawRecord,
    draw_conv_positions,
    select_quota,
    valid_prev_positions,
)
from oracle_lens.pipeline.shards import LENGTH_BUCKETS


class FakeTok:
    """Whitespace 'tokenizer': ids are the words themselves (slice/decode compatible)."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


# ---------- parsing / truncation ----------


def test_extract_complete_tags() -> None:
    assert extract_explanation_content("<explanation>\na cat\n</explanation>") == "a cat"
    assert extract_explanation_content("no tags at all") == "no tags at all"


def test_extract_truncated_close_tag() -> None:
    assert extract_explanation_content("<explanation>\na cat\n</expl") == "a cat"
    assert extract_explanation_content("<explanation>\na cat\n<") == "a cat"


def test_extract_keeps_first_block_only() -> None:
    two = "<explanation>\nfirst\n</explanation>\n<explanation>\nsecond\n"
    assert extract_explanation_content(two) == "first"


def test_extract_empty() -> None:
    assert extract_explanation_content("") == ""
    assert extract_explanation_content("<explanation>\n</explanation>") == ""


def test_phrases_truncate_and_dedup() -> None:
    tok = FakeTok()
    samples = [
        "<explanation>\nalpha beta gamma delta\n</explanation>",
        "<explanation>\nalpha beta gamma EPSILON\n</explanation>",  # same after 3-token cut
        "<explanation>\nzeta eta\n</explanation>",
        "<explanation>\n\n</explanation>",  # empty -> dropped
    ]
    out = phrases_from_samples(samples, tok, n=3)
    assert out == ["alpha beta gamma", "zeta eta"]


def test_phrases_short_sample_kept_verbatim() -> None:
    out = phrases_from_samples(["<explanation>\nhi\n</explanation>"], FakeTok(), n=16)
    assert out == ["hi"]


def test_shortfall_rows() -> None:
    assert shortfall_rows([3, 4, 0, 4], k=4) == [0, 2]


def test_variants_table() -> None:
    for v in VARIANTS.values():
        if v.sampler == "iid":  # chain variants draw k_prime per STEP, so k_prime < k is fine
            assert v.k_prime >= v.k
        assert v.max_new > v.n  # wrapper slack so n content tokens are reachable


def test_sampler_script_constants_in_sync() -> None:
    """The vllm-venv sampler keeps inline copies (no repo imports there) — they must match."""
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/distill/olens_distill_sample_cluster.py"
    )
    spec = importlib.util.spec_from_file_location("r2sampler", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.LAYERS == LAYERS
    for name, var in VARIANTS.items():
        inline = mod.VARIANTS[name]
        assert (inline["k"], inline["n"], inline["k_prime"], inline["max_new"]) == (
            var.k,
            var.n,
            var.k_prime,
            var.max_new,
        )


# ---------- distill shards ----------


def _mk_pairs(n: int, d: int = 8, seed: int = 0) -> DistillPairs:
    gen = torch.Generator().manual_seed(seed)
    lens = torch.randint(3, 9, (n,), generator=gen)
    offsets = torch.zeros(n + 1, dtype=torch.int64)
    torch.cumsum(lens, dim=0, out=offsets[1:])
    return DistillPairs(
        target_ids=torch.randint(5, 1000, (int(lens.sum()),), generator=gen).to(torch.int32),
        offsets=offsets,
        vec=torch.randn(n, d, generator=gen),
        layer_idx=torch.randint(0, len(LAYERS), (n,), generator=gen),
        src_row=torch.arange(n),
    )


def _mk_meta(n_rows: int, **kw: Any) -> DistillShardMeta:
    base: dict[str, Any] = {
        "model_id": "m",
        "teacher_run": "t",
        "variant": "k20n16",
        "k": 20,
        "n": 16,
        "prompt_mode": "list",
        "pairs_dir": "p",
        "split": "train",
        "seed": 0,
        "git_commit": "abc",
        "n_rows": n_rows,
    }
    base.update(kw)
    return DistillShardMeta(**base)


def test_distill_shard_roundtrip(tmp_path: Path) -> None:
    pairs = _mk_pairs(7)
    save_distill_shard(tmp_path / "a.safetensors", pairs, _mk_meta(7))
    save_distill_shard(tmp_path / "b.safetensors", _mk_pairs(5, seed=1), _mk_meta(5))
    loaded, meta = load_distill_shards(sorted(tmp_path.glob("*.safetensors")))
    assert len(loaded) == 12
    assert meta.k == 20 and meta.prompt_mode == "list"
    assert torch.equal(loaded.row_target(3), pairs.row_target(3))
    assert torch.allclose(loaded.vec[:7].float(), pairs.vec.to(torch.bfloat16).float(), atol=0)
    assert loaded.layer_of(0) == LAYERS[int(pairs.layer_idx[0])]


def test_distill_shard_meta_mismatch(tmp_path: Path) -> None:
    save_distill_shard(tmp_path / "a.safetensors", _mk_pairs(3), _mk_meta(3))
    save_distill_shard(tmp_path / "b.safetensors", _mk_pairs(3), _mk_meta(3, variant="k20n4", n=4))
    with pytest.raises(ValueError, match="mismatch"):
        load_distill_shards(sorted(tmp_path.glob("*.safetensors")))


def test_distill_shard_rowcount_guard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_rows"):
        save_distill_shard(tmp_path / "a.safetensors", _mk_pairs(3), _mk_meta(4))


# ---------- example builder / dataset / bucketing ----------


def _cfg(**kw: Any) -> GTConfig:
    base: dict[str, Any] = {
        "run_name": "t",
        "layers": LAYERS,
        "n_examples": 1000,
        "min_len": 1,
        "max_len": 2048,
        "token_budget": 256,
        "mb_max": 8,
    }
    base.update(kw)
    return GTConfig(**base)


def test_build_distill_examples_verbatim_targets() -> None:
    pairs = _mk_pairs(20)
    cfg = _cfg()
    ex = build_distill_examples(pairs, cfg)
    assert len(ex) == 20
    for e in ex:
        row = int(e["row"])
        assert torch.equal(e["target_ids"], pairs.row_target(row))  # no re-wrap
        assert int(e["n_span"]) == int(e["target_ids"].shape[0])
        assert cfg.layers[int(e["layer_idx"])] == pairs.layer_of(row)


def test_build_distill_examples_layer_subset() -> None:
    pairs = _mk_pairs(50)
    cfg = _cfg(layers=(44,))
    ex = build_distill_examples(pairs, cfg)
    want = sum(1 for i in range(50) if pairs.layer_of(i) == 44)
    assert len(ex) == want
    assert all(int(e["layer_idx"]) == 0 for e in ex)


def test_build_distill_examples_skip_resume_disjoint() -> None:
    pairs = _mk_pairs(30)
    all_rows = [int(e["row"]) for e in build_distill_examples(pairs, _cfg(n_examples=30))]
    head = [int(e["row"]) for e in build_distill_examples(pairs, _cfg(n_examples=10))]
    tail = [
        int(e["row"]) for e in build_distill_examples(pairs, _cfg(n_examples=20, skip_examples=10))
    ]
    assert all_rows == head + tail  # same seeded permutation, disjoint continuation


def test_distill_dataset_item() -> None:
    pairs = _mk_pairs(10)
    ex = build_distill_examples(pairs, _cfg())
    ds = DistillDataset(pairs, ex)
    item = ds[0]
    assert item["inject_vec"].shape == (8,)
    assert item["inject_vec"].dtype == torch.float32


def test_buckets_for_extension() -> None:
    assert buckets_for(_cfg(max_len=1024)) == LENGTH_BUCKETS
    ext = buckets_for(_cfg(max_len=2048))
    assert ext[:-1] == LENGTH_BUCKETS and ext[-1] == (1025, 2048)


def test_sampler_accepts_long_list_targets() -> None:
    """A >1024-token target must land in the extended bucket, not raise."""
    n = 6
    lens = torch.full((n,), 1100, dtype=torch.int64)
    offsets = torch.zeros(n + 1, dtype=torch.int64)
    torch.cumsum(lens, dim=0, out=offsets[1:])
    pairs = DistillPairs(
        target_ids=torch.randint(5, 99, (int(lens.sum()),)).to(torch.int32),
        offsets=offsets,
        vec=torch.randn(n, 8),
        layer_idx=torch.zeros(n, dtype=torch.int64),
        src_row=torch.arange(n),
    )
    cfg = _cfg(max_len=2048, token_budget=2400, mb_max=4)
    ex = build_distill_examples(pairs, cfg)
    sampler = make_batch_sampler(cfg, ex, prompt_lens=[40] * len(LAYERS))
    batches = list(sampler)
    assert batches  # at least one full batch, no bucket_index_of raise


# ---------- fresh-position draw ----------


def test_valid_prev_positions() -> None:
    #                          0      1      2     3      4
    mask = [False, False, True, True, False]
    assert valid_prev_positions(mask) == [1, 2]  # s=2,3 generated -> prev 1,2


def test_draw_conv_positions_excludes_and_caps() -> None:
    import random

    mask = [False] + [True] * 10  # prev positions 0..9
    out = draw_conv_positions(mask, per_conv=6, rng=random.Random(0), exclude={0, 1, 2, 3, 4, 5, 6})
    assert set(out) <= {7, 8, 9} and len(out) == 3
    assert out == sorted(out)


def test_draw_conv_positions_deterministic() -> None:
    import random

    mask = [False] + [True] * 50
    a = draw_conv_positions(mask, per_conv=5, rng=random.Random(7))
    b = draw_conv_positions(mask, per_conv=5, rng=random.Random(7))
    assert a == b


def test_select_quota_exact_trim() -> None:
    recs = [DrawRecord("s", i, i, tuple(range(4))) for i in range(10)]
    picked = select_quota(recs, quota=10, seed=0, label="train")
    assert sum(len(r.positions) for r in picked) == 10
    again = select_quota(recs, quota=10, seed=0, label="train")
    assert picked == again  # seeded determinism
    under = select_quota(recs[:2], quota=100, seed=0, label="train")
    assert sum(len(r.positions) for r in under) == 8  # under-supply keeps everything
