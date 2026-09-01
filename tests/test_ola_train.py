"""Whitened-MSE loss known values, harness-config translation, whole-set val metrics."""

from typing import Any, cast

import pytest
import torch
from test_ola_shards import make_pairs
from torch import Tensor, nn

from oracle_lens.core.reconstructor import Reconstructor
from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.buckets import Bucket
from oracle_lens.pipeline.train_recon import (
    LongSpanReconConfig,
    LongSpanReconMydule,
    build_longspan_training_config,
    whitened_mse_loss,
)

D = 8


def identity_whitener(d: int = D) -> Whitener:
    return Whitener(mu=torch.zeros(d), w=torch.eye(d), ridge_c=0.0)


def test_whitened_mse_loss_identity_whitener_equals_mse() -> None:
    gen = torch.Generator().manual_seed(0)
    pred = torch.randn(5, D, generator=gen)
    target = torch.randn(5, D, generator=gen)
    loss = whitened_mse_loss(pred, target, identity_whitener())
    assert loss.item() == pytest.approx(torch.nn.functional.mse_loss(pred, target).item())


def test_whitened_mse_loss_known_value() -> None:
    pred = torch.zeros(1, D)
    target = torch.ones(1, D) * 2.0
    # identity whitening: mean over d of (0-2)^2 = 4
    assert whitened_mse_loss(pred, target, identity_whitener()).item() == pytest.approx(4.0)


def test_training_config_keep_global_batch_divides_grad_accum() -> None:
    cfg = LongSpanReconConfig(run_name="t", grad_accum=8, epochs=1, warmup_steps=0)
    lengths = [10] * (128 * 64)  # 64 bucket-0 batches
    tc = build_longspan_training_config(cfg, lengths, world_size=8)
    assert tc.gradient_accumulation_steps == 1  # 8 // 8
    assert tc.ngpu == 8
    assert tc.max_steps == (64 // 8) // 1  # per-rank batches / grad_accum_eff
    assert tc.lr == pytest.approx(cfg.lr)  # growth 1.0 -> no scaling


def test_training_config_single_gpu_steps() -> None:
    cfg = LongSpanReconConfig(run_name="t", grad_accum=2, epochs=1, warmup_steps=0)
    lengths = [10] * (128 * 10) + [100] * (32 * 4)
    tc = build_longspan_training_config(cfg, lengths, world_size=1)
    assert tc.max_steps == (10 + 4) // 2
    assert tc.gradient_accumulation_steps == 2


class StubRecon(nn.Module):
    """Deterministic stand-in for the 27B reconstructor: embeds ids, mean-pools, projects."""

    def __init__(self, d: int = D, vocab: int = 64) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, d)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        h = self.embed(input_ids) * attention_mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        out: Tensor = self.proj(pooled)
        return out


def test_mydule_whole_set_metrics_shape_and_keys() -> None:
    tiny = (Bucket(4, 2), Bucket(8, 2))
    rows = [[i % 60 + 1] * (1 + i % 4) for i in range(24)]
    pairs = make_pairs(rows)
    cfg = LongSpanReconConfig(run_name="t", retrieval_pool=5, max_eval_rows=0, seed=0)
    recon = cast(Reconstructor, StubRecon())
    mydule = LongSpanReconMydule(recon, identity_whitener(), cfg, pairs, pairs, buckets=tiny)
    # drive validation_step over exactly the sampler's batches
    from oracle_lens.pipeline.buckets import (
        BucketBatchSampler,
        RaggedPairsDataset,
        bucket_collate,
    )

    data = RaggedPairsDataset(pairs)
    sampler = BucketBatchSampler(data.lengths, buckets=tiny, seed=0)
    mydule.model = mydule._recon  # what the harness sets on finalize_model
    batch_info: Any = None
    for batch_idx in sampler:
        mydule.validation_step(bucket_collate([data[i] for i in batch_idx], pad_id=0), batch_info)
    metrics = mydule.last_val_metrics
    assert {"whitened_mse", "whitened_r2", "raw_r2", "retrieval_top1", "predict_mean_r2"} <= set(
        metrics
    )
    assert any(key.startswith("whitened_r2/N") for key in metrics)
    assert metrics["whitened_mse"] >= 0.0


def test_mydule_training_step_counts_tokens() -> None:
    tiny = (Bucket(4, 2),)
    pairs = make_pairs([[1, 2], [3, 4, 5], [6], [7, 8]])
    cfg = LongSpanReconConfig(run_name="t", max_eval_rows=0)
    recon = cast(Reconstructor, StubRecon())
    mydule = LongSpanReconMydule(recon, identity_whitener(), cfg, pairs, pairs, buckets=tiny)
    mydule.model = mydule._recon

    logged: dict[str, float] = {}
    mydule.log = lambda key, value, **kw: logged.__setitem__(key, float(value))
    from oracle_lens.pipeline.buckets import RaggedPairsDataset, bucket_collate

    data = RaggedPairsDataset(pairs)
    batch = bucket_collate([data[0], data[1]], pad_id=0, buckets=tiny)
    loss = mydule.training_step(batch, None)
    assert loss.requires_grad
    assert logged["tokens_consumed"] == 5.0  # 2 + 3 real tokens
