"""Bucket boundaries, static-shape collate, DDP-disjoint batch sampler."""

import pytest
import torch
from test_ola_shards import make_pairs

from oracle_lens.pipeline.buckets import (
    BUCKETS,
    Bucket,
    BucketBatchSampler,
    RaggedPairsDataset,
    batches_per_rank,
    bucket_collate,
    bucket_index,
)


def test_bucket_index_boundaries() -> None:
    assert bucket_index(1) == 0
    assert bucket_index(32) == 0
    assert bucket_index(33) == 1
    assert bucket_index(64) == 1
    assert bucket_index(512) == 4
    assert bucket_index(1024) == 5
    with pytest.raises(ValueError):
        bucket_index(1025)
    with pytest.raises(ValueError):
        bucket_index(0)


def test_collate_pads_left_to_bucket_width() -> None:
    pairs = make_pairs([[1, 2, 3], [4, 5, 6, 7], [8]])
    data = RaggedPairsDataset(pairs)
    batch = bucket_collate([data[0], data[1], data[2]], pad_id=0)
    assert batch["input_ids"].shape == (3, 32)  # bucket 0 (max len 4 <= 32)
    assert batch["input_ids"][0, -3:].tolist() == [1, 2, 3]
    assert batch["input_ids"][0, :-3].tolist() == [0] * 29
    assert batch["attention_mask"][1].sum() == 4
    assert batch["input_ids"][2, -1] == 8  # last real token at index -1
    assert batch["target"].shape == (3, 8)
    assert batch["span_len"].tolist() == [3, 4, 1]


def test_sampler_batches_are_bucket_homogeneous_and_full() -> None:
    lengths = [5] * 300 + [100] * 70 + [900] * 9
    sampler = BucketBatchSampler(lengths, seed=0)
    batches = list(sampler)
    for b in batches:
        idx = bucket_index(max(lengths[i] for i in b))
        assert all(bucket_index(lengths[i]) == idx for i in b)
        assert len(b) == BUCKETS[idx].batch_rows
    # len 5 -> bucket(32,128): 300//128=2; len 100 -> bucket(128,32): 70//32=2;
    # len 900 -> bucket(1024,4): 9//4=2
    assert len(batches) == 6


def test_sampler_ddp_slices_are_disjoint_and_cover_pool() -> None:
    lengths = [10] * (128 * 9) + [200] * (16 * 7)
    per_rank = [list(BucketBatchSampler(lengths, seed=3, num_replicas=4, rank=r)) for r in range(4)]
    n_total = 9 + 7  # bucket batches
    usable = (n_total // 4) * 4
    assert all(len(b) == usable // 4 for b in per_rank)
    flat = [i for rank_batches in per_rank for batch in rank_batches for i in batch]
    assert len(flat) == len(set(flat)), "ranks overlap"
    assert batches_per_rank(lengths, num_replicas=4) == usable // 4


def test_sampler_deterministic_per_seed_and_epoch() -> None:
    lengths = [10] * 512 + [40] * 130
    a = list(BucketBatchSampler(lengths, seed=11))
    b = list(BucketBatchSampler(lengths, seed=11))
    assert a == b
    sampler = BucketBatchSampler(lengths, seed=11)
    first, second = list(sampler), list(sampler)
    assert first == a  # epoch 0 matches a fresh instance
    assert first != second  # epoch 1 reshuffles


def test_custom_tiny_buckets() -> None:
    tiny = (Bucket(4, 2), Bucket(8, 2))
    lengths = [2, 3, 4, 7, 8, 5]
    sampler = BucketBatchSampler(lengths, buckets=tiny, seed=0)
    batches = list(sampler)
    assert len(batches) == 2  # 3 short rows -> 1 batch of 2; 3 long rows -> 1 batch of 2
    widths = set()
    pairs = make_pairs([[1] * n for n in lengths])
    data = RaggedPairsDataset(pairs)
    for b in batches:
        out = bucket_collate([data[i] for i in b], pad_id=0, buckets=tiny)
        widths.add(int(out["input_ids"].shape[1]))
    assert widths <= {4, 8}


def test_collate_rejects_overlong_rows() -> None:
    pairs = make_pairs([[1] * 1025])
    data = RaggedPairsDataset(pairs)
    with pytest.raises(ValueError):
        bucket_collate([data[0]], pad_id=0)
    with pytest.raises(ValueError):
        BucketBatchSampler([1025], seed=0)


def test_ragged_dataset_row_view() -> None:
    pairs = make_pairs([[9, 8, 7], [6]])
    data = RaggedPairsDataset(pairs)
    row = data[0]
    assert row["ids"].tolist() == [9, 8, 7]
    assert int(row["length"]) == 3
    assert row["target"].dtype == torch.bfloat16
