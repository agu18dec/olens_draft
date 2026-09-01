"""CPU tests for multilayer_retrieval_top1: perfect predictions retrieve their own target (1.0),
random predictions sit at chance (~1/pool), and short-of-a-pool inputs return 0 not crash."""

import torch

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.multilayer_reconstructor import multilayer_retrieval_top1


def _identity_whitener(d: int) -> Whitener:
    return Whitener(mu=torch.zeros(d), w=torch.eye(d), ridge_c=0.1)


def test_perfect_predictions_retrieve_1() -> None:
    torch.manual_seed(0)
    n, d = 200, 16
    targets = torch.randn(n, 2, d)
    ret = multilayer_retrieval_top1(targets.clone(), targets, [_identity_whitener(d)] * 2, pool=100)
    assert ret.shape == (2,)
    assert float(ret.min()) == 1.0


def test_random_predictions_near_chance() -> None:
    torch.manual_seed(1)
    n, d = 1000, 32
    preds = torch.randn(n, 1, d)
    targets = torch.randn(n, 1, d)
    ret = multilayer_retrieval_top1(preds, targets, [_identity_whitener(d)], pool=100)
    assert float(ret[0]) < 0.05  # chance = 0.01


def test_too_few_rows_returns_zero() -> None:
    preds = torch.randn(50, 1, 8)
    ret = multilayer_retrieval_top1(preds, preds.clone(), [_identity_whitener(8)], pool=100)
    assert float(ret[0]) == 0.0
