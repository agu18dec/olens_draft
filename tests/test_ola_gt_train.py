"""GT-continuation lens: inject_gt arms, example builder, layer-bucketed sampler/collate, dedup."""

import hashlib

import pytest
import torch

from oracle_lens.pipeline.dedup import dedup_convs
from oracle_lens.pipeline.gt_train import GTConfig, build_gt_examples, fit_layer_scales
from oracle_lens.pipeline.inject import fit_scale, inject_gt
from oracle_lens.pipeline.multilayer import MultiLayerPairs
from oracle_lens.pipeline.soft_token_sft import (
    LayerBucketBatchSampler,
    SoftTokenConfig,
    soft_token_collate,
)

D = 8
LAYERS = (0, 4, 44)
ALPHA = 8000.0


def make_ml_pairs(
    id_rows: list[list[int]], *, seed: int = 0, conv: list[int] | None = None
) -> MultiLayerPairs:
    gen = torch.Generator().manual_seed(seed)
    n = len(id_rows)
    offsets = torch.zeros(n + 1, dtype=torch.int64)
    for i, row in enumerate(id_rows):
        offsets[i + 1] = offsets[i] + len(row)
    return MultiLayerPairs(
        span_ids=torch.tensor([t for row in id_rows for t in row], dtype=torch.int32),
        offsets=offsets,
        targets=torch.randn(n, len(LAYERS), D, generator=gen).to(torch.bfloat16),
        layers=LAYERS,
        conv_index=torch.tensor(conv if conv is not None else list(range(n)), dtype=torch.int64),
        prev_pos=torch.arange(1, n + 1, dtype=torch.int64),
        prev_token_id=torch.arange(100, 100 + n, dtype=torch.int64),
        prev_is_assistant=torch.ones(n, dtype=torch.bool),
    )


class TinyTokenizer:
    """Minimal tokenizer for the example builder: char-level ids, fixed eos."""

    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


# ---- inject_gt arms ------------------------------------------------------------------------


def test_inject_gt_unit_is_alpha_normed() -> None:
    gen = torch.Generator().manual_seed(0)
    vec = torch.randn(5, D, generator=gen) * 3.7
    v = inject_gt(vec, "unit", alpha=ALPHA, scale=123.0)  # scale ignored by unit
    assert torch.allclose(v.norm(dim=-1), torch.full((5,), ALPHA), atol=1e-2)


def test_inject_gt_scaled_preserves_magnitude_spread() -> None:
    gen = torch.Generator().manual_seed(1)
    vec = torch.randn(6, D, generator=gen)
    s = fit_scale(vec, target_norm=ALPHA)
    v = inject_gt(vec, "scaled", alpha=ALPHA, scale=s)
    assert torch.allclose(v, s * vec, atol=1e-4)
    assert abs(float(v.norm(dim=-1).median()) - ALPHA) / ALPHA < 1e-3


def test_inject_gt_scaled_clip_caps_outliers_only() -> None:
    gen = torch.Generator().manual_seed(2)
    vec = torch.randn(4, D, generator=gen)
    vec[0] *= 1000.0  # massive-activation outlier row
    s = fit_scale(vec[1:], target_norm=ALPHA)
    v = inject_gt(vec, "scaled_clip", alpha=ALPHA, scale=s, clip_mult=4.0)
    cap = 4.0 * ALPHA
    assert float(v[0].norm()) == pytest.approx(cap, rel=1e-3)
    ref = inject_gt(vec[1:], "scaled", alpha=ALPHA, scale=s)
    assert torch.allclose(v[1:], ref, atol=1e-3)  # sub-cap rows untouched


# ---- example builder ------------------------------------------------------------------------


def cfg_for(n: int, layers: tuple[int, ...] = LAYERS) -> GTConfig:
    return GTConfig(run_name="t", layers=layers, n_examples=n, min_len=1, max_len=16)


def test_build_gt_examples_target_is_tag_wrapped_span_plus_eos() -> None:
    tok = TinyTokenizer()
    pairs = make_ml_pairs([[7, 8, 9]])
    ex = build_gt_examples(pairs, tok, cfg_for(1, layers=(44,)))
    open_ids = tok("<explanation>\n", add_special_tokens=False)["input_ids"]
    close_ids = tok("\n</explanation>", add_special_tokens=False)["input_ids"]
    expected = [*open_ids, 7, 8, 9, *close_ids, 99]
    assert ex[0]["target_ids"].tolist() == expected
    assert int(ex[0]["layer_idx"]) == 0  # single-layer cfg -> only index 0


def test_build_gt_examples_seeded_layer_draw_is_deterministic() -> None:
    tok = TinyTokenizer()
    pairs = make_ml_pairs([[1, 2], [3], [4, 5, 6], [7], [8, 9]])
    a = build_gt_examples(pairs, tok, cfg_for(5))
    b = build_gt_examples(pairs, tok, cfg_for(5))
    assert [int(e["layer_idx"]) for e in a] == [int(e["layer_idx"]) for e in b]
    assert [int(e["row"]) for e in a] == [int(e["row"]) for e in b]


def test_build_gt_examples_rejects_missing_layer() -> None:
    pairs = make_ml_pairs([[1, 2]])
    with pytest.raises(ValueError, match="layer 63"):
        build_gt_examples(pairs, TinyTokenizer(), cfg_for(1, layers=(63,)))


def test_fit_layer_scales_median_norm_hits_alpha() -> None:
    pairs = make_ml_pairs([[1]] * 64)
    scales = fit_layer_scales(pairs, LAYERS, alpha=ALPHA, sample_n=64)
    for ly in LAYERS:
        pos = LAYERS.index(ly)
        norms = (pairs.targets.float()[:, pos, :] * scales[str(ly)]).norm(dim=-1)
        assert abs(float(norms.median()) - ALPHA) / ALPHA < 1e-2


# ---- sampler + collate ----------------------------------------------------------------------


def test_layer_bucket_sampler_batches_are_layer_pure_and_ranks_disjoint() -> None:
    layer_idx = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # 13 examples, last 0 dropped (partial)
    per_rank: list[set[int]] = []
    for rank in range(2):
        sampler = LayerBucketBatchSampler(layer_idx, micro_batch=2, world=2, rank=rank, seed=3)
        batches = list(sampler)
        assert len(batches) == len(sampler)
        for b in batches:
            assert len({layer_idx[i] for i in b}) == 1  # layer-pure
        per_rank.append({i for b in batches for i in b})
    assert per_rank[0] & per_rank[1] == set()  # disjoint across ranks
    sampler_a = LayerBucketBatchSampler(layer_idx, micro_batch=2, world=2, rank=0, seed=3)
    assert [sorted(b) for b in sampler_a] == [sorted(b) for b in sampler_a]  # epoch-stable


def test_gt_collate_masks_prompt_and_pad_and_rejects_mixed_layers() -> None:
    rows = [
        {
            "inject_vec": torch.randn(D),
            "layer_idx": torch.tensor(1),
            "target_ids": torch.tensor([5, 6, 7]),
            "span_len": torch.tensor(3),
        },
        {
            "inject_vec": torch.randn(D),
            "layer_idx": torch.tensor(1),
            "target_ids": torch.tensor([8]),
            "span_len": torch.tensor(1),
        },
    ]
    prompt_lens = [4, 6, 5]
    out = soft_token_collate(rows, prompt_lens=prompt_lens, pad_id=0)
    p = prompt_lens[1]
    assert out["labels"].shape == (2, p + 3)
    assert (out["labels"][:, :p] == -100).all()
    assert out["labels"][0, p : p + 3].tolist() == [5, 6, 7]
    assert out["labels"][1, p].item() == 8 and (out["labels"][1, p + 1 :] == -100).all()
    assert out["attention_mask"][1, p + 1 :].sum() == 0  # pad not attended
    assert int(out["layer_idx"]) == 1
    rows[1]["layer_idx"] = torch.tensor(2)
    with pytest.raises(ValueError, match="mixed-layer"):
        soft_token_collate(rows, prompt_lens=prompt_lens, pad_id=0)


# ---- conversation dedup ---------------------------------------------------------------------


def _key(s: str) -> bytes:
    return hashlib.blake2b(s.encode(), digest_size=16).digest()


def test_dedup_convs_keeps_first_conv_per_key_and_fails_open() -> None:
    # convs: 0 and 2 share a seed; conv 3 has no key (kept); rows 2 per conv
    pairs = make_ml_pairs([[1], [2], [3], [4], [5], [6], [7], [8]], conv=[0, 0, 1, 1, 2, 2, 3, 3])
    keys = {0: _key("hello"), 1: _key("other"), 2: _key("hello")}
    out, stats = dedup_convs(pairs, keys)
    assert sorted(set(out.conv_index.tolist())) == [0, 1, 3]  # conv 2 dropped, unkeyed 3 kept
    assert stats["n_dropped"] == 2
    assert stats["n_convs_unkeyed"] == 1


# ---- exact token accounting + mid-epoch resume (AO-ladder additions) ------------------------


def test_gt_collate_counts_span_and_supervised_tokens_exactly() -> None:
    rows = [
        {
            "inject_vec": torch.randn(D),
            "layer_idx": torch.tensor(0),
            "target_ids": torch.tensor([5, 6, 7]),  # e.g. tags(2) + span(1) shapes vary by test
            "span_len": torch.tensor(1),
        },
        {
            "inject_vec": torch.randn(D),
            "layer_idx": torch.tensor(0),
            "target_ids": torch.tensor([8, 9]),
            "span_len": torch.tensor(2),
        },
    ]
    out = soft_token_collate(rows, prompt_lens=[4], pad_id=0)
    assert int(out["n_span_tokens"]) == 3  # 1 + 2, never an estimate
    # supervised = every non-pad target position (labels != -100): 3 + 2
    assert int(out["n_sup_tokens"]) == 5
    assert int(out["n_sup_tokens"]) == int((out["labels"] != -100).sum())


def test_layer_bucket_sampler_skip_batches_resumes_exactly() -> None:
    layer_idx = [i % 3 for i in range(60)]
    kw: dict = {"micro_batch": 4, "world": 2, "rank": 1, "seed": 7, "shuffle": True}
    full = LayerBucketBatchSampler(layer_idx, **kw)
    skipped = LayerBucketBatchSampler(layer_idx, **kw, skip_batches=3)
    full_batches = list(full)
    skipped_batches = list(skipped)
    # the skipped sampler yields exactly the tail of the identical deterministic batch list
    assert skipped_batches == full_batches[3:]
    assert len(skipped) == len(full) - 3
    # epoch bump changes the order but the skip contract still holds
    full.set_epoch(1)
    skipped.set_epoch(1)
    assert list(skipped) == list(full)[3:]


def test_gt_config_roundtrips_skip_batches() -> None:
    cfg = SoftTokenConfig(run_name="t", skip_batches=17)
    assert SoftTokenConfig.from_dict(cfg.to_dict()).skip_batches == 17
    # old JSONs without the field still load (default 0)
    d = cfg.to_dict()
    del d["skip_batches"]
    assert SoftTokenConfig.from_dict(d).skip_batches == 0


def test_val_micro_batch_must_admit_at_least_one_batch_per_group() -> None:
    """Regression: (layer,length)-pure grouping drops partial batches, so a val batch size above
    the smallest group size yields ZERO val batches — validation and checkpoints silently never
    fire (cost a 29-minute run on 2026-07-29)."""
    groups = [i % 102 for i in range(6000)]  # ~59 examples per group, like 1497 crops x k=4
    assert len(LayerBucketBatchSampler(groups, micro_batch=128, shuffle=False)) == 0
    assert len(LayerBucketBatchSampler(groups, micro_batch=16, shuffle=False)) > 0
    cfg = SoftTokenConfig(run_name="t", micro_batch=128, val_micro_batch=16)
    assert cfg.val_micro_batch == 16
    assert SoftTokenConfig.from_dict(cfg.to_dict()).val_micro_batch == 16
    d = cfg.to_dict()
    del d["val_micro_batch"]  # old JSONs still load
    assert SoftTokenConfig.from_dict(d).val_micro_batch == 0


def test_deal_key_synchronizes_length_across_ranks() -> None:
    """DDP straggler fix: without deal_key each rank draws an independent group, so a step costs
    the longest sequence any rank drew (measured 1.38x). With it, all ranks share a bucket."""
    n_lengths, n_layers, per_group = 6, 17, 40
    groups = [
        ly * n_lengths + j
        for ly in range(n_layers)
        for j in range(n_lengths)
        for _ in range(per_group)
    ]
    dk = {ly * n_lengths + j: j for ly in range(n_layers) for j in range(n_lengths)}
    world = 8
    samplers = [
        LayerBucketBatchSampler(groups, micro_batch=8, world=world, rank=r, seed=3, deal_key=dk)
        for r in range(world)
    ]
    per_rank = [list(s) for s in samplers]
    assert len({len(x) for x in per_rank}) == 1 and len(per_rank[0]) > 0
    for step in range(len(per_rank[0])):
        buckets = {dk[groups[per_rank[r][step][0]]] for r in range(world)}
        assert len(buckets) == 1, f"step {step}: ranks span lengths {buckets}"
        # every example within a rank's batch is one group (layer+length pure)
        for r in range(world):
            assert len({groups[i] for i in per_rank[r][step]}) == 1
    # layers still vary across ranks within a step (diversity preserved)
    layers_step0 = {groups[per_rank[r][0][0]] // n_lengths for r in range(world)}
    assert len(layers_step0) > 1
    # no example is used by two ranks in the same step
    flat = [i for r in range(world) for i in per_rank[r][0]]
    assert len(flat) == len(set(flat))


# ---- activation dedup (one row per (conv, prev_pos) per epoch) ------------------------------


def test_build_gt_examples_dedup_activations_one_row_per_activation() -> None:
    from dataclasses import replace

    tok = TinyTokenizer()
    # rows 0/1 share (conv=0, prev_pos=5) at different lengths; rows 2/3 are unique activations
    pairs = make_ml_pairs([[1, 2], [3, 4, 5, 6], [7], [8, 9, 10]], conv=[0, 0, 0, 1])
    pairs = replace(pairs, prev_pos=torch.tensor([5, 5, 9, 2], dtype=torch.int64))
    ex = build_gt_examples(pairs, tok, replace(cfg_for(10), dedup_activations=True))
    rows = [int(e["row"]) for e in ex]
    assert len(rows) == 3
    assert {2, 3} <= set(rows)  # unique activations always survive
    assert (0 in rows) != (1 in rows)  # exactly one of the same-activation pair
    keys = {(int(pairs.conv_index[r]), int(pairs.prev_pos[r])) for r in rows}
    assert len(keys) == 3
    # deterministic given the seed
    ex2 = build_gt_examples(pairs, tok, replace(cfg_for(10), dedup_activations=True))
    assert rows == [int(e["row"]) for e in ex2]
    # off by default: all 4 rows train
    assert len(build_gt_examples(pairs, tok, cfg_for(10))) == 4


def test_inject_gt_metric_space_arms_are_alpha_normed_and_use_the_transpose() -> None:
    """wht_unit/j_unit map into a metric space, unit-normalise, then scale by alpha. The norm is
    exactly alpha (the AR matches DIRECTION only, so the untrained norm must not reach the AO),
    and the map is the caller's ruler — J is not symmetric, so `@ J.T` is load-bearing."""
    import torch

    from oracle_lens.pipeline.inject import inject_gt
    from oracle_lens.pipeline.jspace import JSpace

    j = JSpace(mu=torch.tensor([1.0, 0.0]), j=torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    out = inject_gt(torch.tensor([[2.0, 3.0]]), "j_unit", alpha=1.0, scale=1.0, space=j)
    want = torch.tensor([[3.0, 1.0]]) / torch.tensor([[3.0, 1.0]]).norm()
    torch.testing.assert_close(out, want)  # (x-mu)=[1,3] -> @J.T -> [3,1] -> unit

    gen = torch.Generator().manual_seed(0)
    big = JSpace(mu=torch.randn(8, generator=gen), j=torch.randn(8, 8, generator=gen))
    v = inject_gt(torch.randn(5, 8, generator=gen) * 40, "wht_unit",
                  alpha=8000.0, scale=1.0, space=big)
    torch.testing.assert_close(v.norm(dim=-1), torch.full((5,), 8000.0), atol=1e-2, rtol=1e-4)


def test_inject_gt_metric_space_arms_require_a_space() -> None:
    """A missing per-layer ruler must raise, not silently inject an un-mapped vector."""
    import pytest
    import torch

    from oracle_lens.pipeline.inject import inject_gt

    for t in ("wht_unit", "j_unit"):
        with pytest.raises(ValueError, match="metric space"):
            inject_gt(torch.randn(2, 8), t, alpha=8000.0, scale=1.0)  # type: ignore[arg-type]


def test_scale_for_returns_one_for_every_scale_free_arm() -> None:
    """The KeyError fallthrough: scale-free arms carry no per-layer fit_scale entry, so
    scale_for must NOT index `scales` for them."""
    from oracle_lens.pipeline.gt_train import GTConfig
    from oracle_lens.pipeline.inject import SCALE_FREE_GT

    for t in sorted(SCALE_FREE_GT):
        cfg = GTConfig(run_name="t", transform=t, layers=(44,), scales={})  # type: ignore[arg-type]
        assert cfg.scale_for(44) == 1.0


def test_layer_bucket_sampler_exclude_mask_trains_only_the_remainder() -> None:
    """Exact-remainder continuation: excluded examples never appear, remainder fully regroups."""
    layer_idx = [i % 3 for i in range(60)]
    kw: dict = {"micro_batch": 4, "world": 2, "seed": 7, "shuffle": True}
    mask = torch.zeros(60, dtype=torch.bool)
    mask[::2] = True  # exclude every even example
    yielded: set[int] = set()
    for rank in range(2):
        s = LayerBucketBatchSampler(layer_idx, **kw, rank=rank, exclude_mask=mask)
        for batch in s:
            yielded.update(batch)
    assert yielded, "remainder must still form full batches"
    assert not (yielded & set(range(0, 60, 2))), "an excluded example was yielded"
    # deterministic: two constructions yield the identical stream
    a = list(LayerBucketBatchSampler(layer_idx, **kw, rank=0, exclude_mask=mask))
    b = list(LayerBucketBatchSampler(layer_idx, **kw, rank=0, exclude_mask=mask))
    assert a == b
    # guards: skip_batches is a different mechanism; a wrong-sized mask is a wrong dataset
    with pytest.raises(ValueError, match="mutually exclusive"):
        LayerBucketBatchSampler(layer_idx, **kw, rank=0, exclude_mask=mask, skip_batches=1)
    with pytest.raises(ValueError, match="entries for"):
        short_mask = torch.zeros(3, dtype=torch.bool)
        LayerBucketBatchSampler(layer_idx, **kw, rank=0, exclude_mask=short_mask)


def test_gt_config_roundtrips_exclude_idx_file() -> None:
    cfg = SoftTokenConfig(run_name="t", exclude_idx_file="ao_runs/u64_consumed_idx.npy")
    assert SoftTokenConfig.from_dict(cfg.to_dict()).exclude_idx_file == cfg.exclude_idx_file
    d = cfg.to_dict()
    del d["exclude_idx_file"]  # old JSONs without the field still load
    assert SoftTokenConfig.from_dict(d).exclude_idx_file == ""
