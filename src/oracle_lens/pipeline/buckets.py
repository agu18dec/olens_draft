"""Static-shape bucketing: constant-token micro-batches over a 32..1024 length range.

The oracle-lens reconstructor's 17x speedup comes from per-block ``torch.compile(dynamic=False)``
+ ONE static batch shape (mb128 x pad32 = 4096 padded tokens). Long spans can't share one shape,
so this generalizes to six ``(pad_width, batch_rows)`` buckets that all hold the padded token
count at 4096/micro-batch — six compiled shapes total (bump ``torch._dynamo.config
.cache_size_limit`` to >= 2 x len(BUCKETS) and warm each bucket up once before the timed loop).

``BucketBatchSampler`` is the DDP-aware batch sampler (adapted from the oracle branch's
``LengthBucketSampler``): batches are formed WITHIN a bucket, batch order is shuffled per epoch,
and with ``num_replicas > 1`` each rank takes the disjoint ``batches[rank::num_replicas]`` slice
of an identically-shuffled list — this replaces the harness's DistributedSampler, which is
discarded when a batch_sampler is set.
"""

import random
from collections.abc import Iterator
from typing import NamedTuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from oracle_lens.pipeline.shards import LongSpanPairs


class Bucket(NamedTuple):
    pad_width: int
    batch_rows: int


# Constant 4096 padded tokens per micro-batch — the proven memory/throughput point
# (mb128 x pad32 + grad checkpointing peaked at 47.5 GB on H100 for the oracle-lens recon).
BUCKETS: tuple[Bucket, ...] = (
    Bucket(32, 128),
    Bucket(64, 64),
    Bucket(128, 32),
    Bucket(256, 16),
    Bucket(512, 8),
    Bucket(1024, 4),
)


def bucket_index(length: int, *, buckets: tuple[Bucket, ...] = BUCKETS) -> int:
    """Smallest bucket whose pad_width fits ``length``; raises if nothing fits."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    for i, b in enumerate(buckets):
        if length <= b.pad_width:
            return i
    raise ValueError(f"length {length} exceeds the largest bucket ({buckets[-1].pad_width})")


class RaggedPairsDataset(Dataset[dict[str, Tensor]]):
    """Row view over in-RAM LongSpanPairs (targets stay bf16 until the training step)."""

    def __init__(self, pairs: LongSpanPairs) -> None:
        self.pairs = pairs
        self._lengths = pairs.lengths.tolist()

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def lengths(self) -> list[int]:
        return list(self._lengths)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return {
            "ids": self.pairs.row_ids(i),
            "target": self.pairs.targets[i],
            "length": torch.tensor(self._lengths[i], dtype=torch.long),
        }


def bucket_collate(
    rows: list[dict[str, Tensor]],
    *,
    pad_id: int,
    buckets: tuple[Bucket, ...] = BUCKETS,
) -> dict[str, Tensor]:
    """LEFT-pad a same-bucket batch to its bucket's fixed width (static compile shape).

    Left-padding keeps the last real token at index -1 (the reconstructor reads
    ``last_hidden_state[:, -1, :]``). The sampler guarantees all rows share a bucket, so the
    width inferred from the batch max is that bucket's pad_width — deterministic per bucket.
    """
    lens = torch.stack([r["length"] for r in rows])
    width = buckets[bucket_index(int(lens.max()), buckets=buckets)].pad_width
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for row, r in enumerate(rows):
        n = int(r["length"])
        input_ids[row, width - n :] = r["ids"]
        attention[row, width - n :] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "target": torch.stack([r["target"] for r in rows]),
        "span_len": lens,
    }


class BucketBatchSampler:
    """Bucket-homogeneous batches, DDP-sharded, deterministic under (seed, epoch).

    Within each bucket the row order is shuffled per epoch and chunked into batches of the
    bucket's ``batch_rows`` (final partial chunk dropped — fixed shapes). All buckets' batches
    are pooled, the pool order shuffled, trimmed to a multiple of ``num_replicas``, and each
    rank yields its ``[rank::num_replicas]`` slice. All ranks use ``seed + epoch``, so slices
    partition the pool disjointly.
    """

    def __init__(
        self,
        lengths: list[int],
        *,
        buckets: tuple[Bucket, ...] = BUCKETS,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.buckets = buckets
        self.seed = seed
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self._epoch = 0
        self._by_bucket: list[list[int]] = [[] for _ in buckets]
        for i, n in enumerate(lengths):
            self._by_bucket[bucket_index(n, buckets=buckets)].append(i)
        total = sum(
            len(idxs) // b.batch_rows for idxs, b in zip(self._by_bucket, buckets, strict=True)
        )
        self._len = total // self.num_replicas

    def __len__(self) -> int:
        return self._len

    def _all_batches(self, rng: random.Random) -> list[list[int]]:
        batches: list[list[int]] = []
        for idxs, bucket in zip(self._by_bucket, self.buckets, strict=True):
            order = list(idxs)
            rng.shuffle(order)
            batches.extend(
                order[i : i + bucket.batch_rows]
                for i in range(0, len(order) - bucket.batch_rows + 1, bucket.batch_rows)
            )
        rng.shuffle(batches)
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        return batches[:usable]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        yield from self._all_batches(rng)[self.rank :: self.num_replicas]


def batches_per_rank(
    lengths: list[int], *, buckets: tuple[Bucket, ...] = BUCKETS, num_replicas: int = 1
) -> int:
    """Micro-batches one rank sees per epoch (drives the harness's max_steps)."""
    sampler = BucketBatchSampler(lengths, buckets=buckets, num_replicas=num_replicas, rank=0)
    return len(sampler)


def synthetic_bucket_batch(
    bucket: Bucket, *, d_model: int, vocab_size: int, pad_id: int = 0, seed: int = 0
) -> dict[str, Tensor]:
    """A full-width synthetic batch for one bucket (the per-bucket memory probe)."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab_size, (bucket.batch_rows, bucket.pad_width), generator=gen)
    target: Float[Tensor, "b d"] = torch.randn(
        bucket.batch_rows, d_model, generator=gen, dtype=torch.float32
    )
    lens: Int[Tensor, "b"] = torch.full((bucket.batch_rows,), bucket.pad_width, dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "target": target,
        "span_len": lens,
    }
