"""Unit tests for the layer+length token-budget batch sampler (long oracle lens trainer)."""

import pytest
import torch

from oracle_lens.pipeline.gt_train import (
    LayerLengthBucketBatchSampler,
    bucket_index_of,
    gt_collate,
)
from oracle_lens.pipeline.shards import LENGTH_BUCKETS

OVERHEAD = [60, 62]  # prompt + wrapper, two layers


def _sampler(**kw: object) -> LayerLengthBucketBatchSampler:
    """600 examples over 2 layers x buckets {1, 33-64, 513-1024}."""
    layer_idx: list[int] = []
    span_len: list[int] = []
    for i in range(600):
        layer_idx.append(i % 2)
        span_len.append([1, 40, 800][i % 3])
    defaults: dict[str, object] = {
        "overhead_per_layer": OVERHEAD,
        "token_budget": 4096,
        "world": 1,
        "rank": 0,
        "seed": 0,
        "shuffle": True,
    }
    defaults.update(kw)
    return LayerLengthBucketBatchSampler(layer_idx, span_len, **defaults)  # type: ignore[arg-type]


def _example_meta(i: int) -> tuple[int, int]:
    return i % 2, [1, 40, 800][i % 3]


def test_batches_pure_in_layer_and_bucket_and_budget() -> None:
    s = _sampler()
    for batch in s:
        layers = {_example_meta(i)[0] for i in batch}
        buckets = {bucket_index_of(_example_meta(i)[1], LENGTH_BUCKETS) for i in batch}
        assert len(layers) == 1 and len(buckets) == 1
        ly, bi = layers.pop(), buckets.pop()
        assert len(batch) == s.rows_for_group(ly, bi)
        seq = OVERHEAD[ly] + LENGTH_BUCKETS[bi][1]
        assert len(batch) * seq <= 4096 or len(batch) == 1


def test_rows_scale_with_bucket() -> None:
    s = _sampler()
    bi_small = bucket_index_of(1, LENGTH_BUCKETS)
    bi_large = bucket_index_of(800, LENGTH_BUCKETS)
    assert s.rows_for_group(0, bi_small) > s.rows_for_group(0, bi_large)
    assert s.rows_for_group(0, bi_large) >= 1


def test_mb_max_caps_rows() -> None:
    s = _sampler(mb_max=4)
    assert s.rows_for_group(0, bucket_index_of(1, LENGTH_BUCKETS)) == 4


def test_ddp_ranks_partition_disjoint_equal() -> None:
    s0 = _sampler(world=2, rank=0)
    s1 = _sampler(world=2, rank=1)
    b0 = list(s0)
    b1 = list(s1)
    assert len(b0) == len(b1) == len(s0)
    seen0 = {i for b in b0 for i in b}
    seen1 = {i for b in b1 for i in b}
    assert not (seen0 & seen1)


def test_skip_batches_resumes_exact_suffix() -> None:
    full = list(_sampler(seed=7))
    skipped = list(_sampler(seed=7, skip_batches=5))
    assert skipped == full[5:]
    assert len(_sampler(seed=7, skip_batches=5)) == len(full) - 5


def test_skip_batches_past_end_raises() -> None:
    n = len(_sampler())
    with pytest.raises(ValueError):
        _sampler(skip_batches=n)


def test_epoch_changes_order_deterministically() -> None:
    s = _sampler(seed=3)
    e0 = list(s)
    s.set_epoch(1)
    e1 = list(s)
    assert e0 != e1
    s2 = _sampler(seed=3)
    assert list(s2) == e0  # epoch 0 reproducible


def test_collate_emits_n_span() -> None:
    rows = [
        {
            "inject_vec": torch.zeros(8),
            "layer_idx": torch.tensor(1),
            "target_ids": torch.tensor([5, 6, 7]),
            "n_span": torch.tensor(2),
        }
        for _ in range(3)
    ]
    batch = gt_collate(rows, prompt_lens=[4, 6], pad_id=0)
    assert batch["n_span"].tolist() == [2, 2, 2]
    assert batch["labels"].shape[1] == 6 + 3  # layer 1 prompt + target


def test_length_packed_respects_budget_with_actual_lengths() -> None:
    s = _sampler(length_packed=True)
    total_padded = 0
    for batch in s:
        layers = {_example_meta(i)[0] for i in batch}
        buckets = {bucket_index_of(_example_meta(i)[1], LENGTH_BUCKETS) for i in batch}
        assert len(layers) == 1 and len(buckets) == 1
        seqs = [OVERHEAD[_example_meta(i)[0]] + _example_meta(i)[1] for i in batch]
        padded = len(batch) * max(seqs)
        assert padded <= 4096 or len(batch) == 1
        total_padded += padded
    # packed batches must waste less than the fixed-mb path on the same data
    fixed = _sampler(length_packed=False)
    fixed_padded = sum(
        len(b) * (OVERHEAD[_example_meta(b[0])[0]] + LENGTH_BUCKETS[
            bucket_index_of(_example_meta(b[0])[1], LENGTH_BUCKETS)][1])
        for b in fixed
    )
    assert total_padded <= fixed_padded


def test_length_packed_deterministic_and_resumable() -> None:
    full = list(_sampler(length_packed=True, seed=9))
    again = list(_sampler(length_packed=True, seed=9))
    assert full == again
    skipped = list(_sampler(length_packed=True, seed=9, skip_batches=3))
    assert skipped == full[3:]
