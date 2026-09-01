"""Pins for the jspace/mixed loss modes on the iolens dispatcher (fair loss-space comparison).

The control arm's invariance is the load-bearing test: `mode="whiten"` through the extended
dispatcher must be BIT-IDENTICAL to the plain whitened-cosine loss, or the whiten arm of the
comparison silently shifts and every cross-arm number is void.
"""

import json

import pytest
import torch

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.jspace import JSpace, MixedSpace
from oracle_lens.pipeline.multilayer_reconstructor import (
    MLReconConfig,
    multilayer_recon_loss,
    multilayer_whitened_cosine_loss,
)

D = 16
N_LAYERS = 4
B = 8


def _whiteners(g: torch.Generator) -> list[Whitener]:
    out = []
    for _ in range(N_LAYERS):
        a = torch.randn(D, D, generator=g)
        out.append(Whitener(mu=torch.randn(D, generator=g), w=a @ a.T + torch.eye(D),
                            ridge_c=0.1))
    return out


def _jspaces(g: torch.Generator, *, none_last: bool = True) -> list[JSpace | None]:
    out: list[JSpace | None] = [
        JSpace(mu=torch.randn(D, generator=g), j=torch.randn(D, D, generator=g))
        for _ in range(N_LAYERS)
    ]
    if none_last:
        out[-1] = None  # the L63 analogue: no Jacobian for the target block
    return out


@pytest.fixture()
def data() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return (torch.randn(B, N_LAYERS, D, generator=g),
            torch.randn(B, N_LAYERS, D, generator=g))


def test_whiten_mode_bit_identical(data: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Control-arm pin: the dispatcher's whiten path IS the plain whitened loss, bit for bit."""
    preds, targets = data
    w = _whiteners(torch.Generator().manual_seed(1))
    direct = multilayer_whitened_cosine_loss(preds, targets, w)
    via = multilayer_recon_loss(preds, targets, w, mode="whiten")
    assert torch.equal(direct, via)


def test_jspace_mode_uses_loss_spaces(data: tuple[torch.Tensor, torch.Tensor]) -> None:
    preds, targets = data
    w = _whiteners(torch.Generator().manual_seed(1))
    js = _jspaces(torch.Generator().manual_seed(2))
    got = multilayer_recon_loss(preds, targets, w, mode="jspace", loss_spaces=js)
    want = multilayer_whitened_cosine_loss(preds, targets, js)
    assert torch.equal(got, want)


def test_jspace_mode_requires_loss_spaces(data: tuple[torch.Tensor, torch.Tensor]) -> None:
    preds, targets = data
    w = _whiteners(torch.Generator().manual_seed(1))
    with pytest.raises(ValueError, match="needs loss_spaces"):
        multilayer_recon_loss(preds, targets, w, mode="jspace")


def test_mixed_lambda_endpoints(data: tuple[torch.Tensor, torch.Tensor]) -> None:
    """MixedSpace λ=0 == whitened loss and λ=1 == jspace loss, over the covered layers."""
    preds, targets = data
    w = _whiteners(torch.Generator().manual_seed(1))
    js = _jspaces(torch.Generator().manual_seed(2))
    cov = [li for li, s in enumerate(js) if s is not None]

    def mixed(lam: float) -> torch.Tensor:
        spaces = [MixedSpace(whitener=w[i], jspace=s, lam=lam) if s is not None else None
                  for i, s in enumerate(js)]
        return multilayer_recon_loss(preds, targets, w, mode="mixed", loss_spaces=spaces)

    w_cov = multilayer_whitened_cosine_loss(
        preds[:, cov], targets[:, cov], [w[i] for i in cov]
    )
    j_cov = multilayer_whitened_cosine_loss(
        preds[:, cov], targets[:, cov], [s for s in js if s is not None]
    )
    assert torch.allclose(mixed(0.0), w_cov, atol=1e-6)
    assert torch.allclose(mixed(1.0), j_cov, atol=1e-6)


def test_mixed_is_lambda_weighted_sum(data: tuple[torch.Tensor, torch.Tensor]) -> None:
    preds, targets = data
    w = _whiteners(torch.Generator().manual_seed(1))
    js = _jspaces(torch.Generator().manual_seed(2))
    cov = [li for li, s in enumerate(js) if s is not None]
    lam = 0.25
    spaces = [MixedSpace(whitener=w[i], jspace=s, lam=lam) if s is not None else None
              for i, s in enumerate(js)]
    got = multilayer_recon_loss(preds, targets, w, mode="mixed", loss_spaces=spaces)
    w_cov = multilayer_whitened_cosine_loss(preds[:, cov], targets[:, cov], [w[i] for i in cov])
    j_cov = multilayer_whitened_cosine_loss(
        preds[:, cov], targets[:, cov], [s for s in js if s is not None]
    )
    assert torch.allclose(got, (1 - lam) * w_cov + lam * j_cov, atol=1e-5)


def test_none_slot_excludes_layer(data: tuple[torch.Tensor, torch.Tensor]) -> None:
    """A None ruler (L63) excludes that layer: the mean runs over covered layers only."""
    preds, targets = data
    js = _jspaces(torch.Generator().manual_seed(2))
    cov = [li for li, s in enumerate(js) if s is not None]
    full = multilayer_whitened_cosine_loss(preds, targets, js)
    sliced = multilayer_whitened_cosine_loss(
        preds[:, cov], targets[:, cov], [s for s in js if s is not None]
    )
    assert torch.equal(full, sliced)


def test_config_json_roundtrip_new_fields() -> None:
    """Resume-path compatibility: the new fields survive to_dict -> json -> MLReconConfig."""
    cfg = MLReconConfig(
        run_name="t", loss_space="mixed", loss_mix_lambda=0.25,
        jspace_repo="neuronpedia/jacobian-lens", jspace_file="x.pt", jspace_revision="abc",
    )
    back = MLReconConfig(**json.loads(json.dumps(cfg.to_dict())))
    assert back.loss_space == "mixed"
    assert back.loss_mix_lambda == 0.25
    assert back.jspace_repo == "neuronpedia/jacobian-lens"
    assert back.jspace_revision == "abc"
    # old JSONs (no new fields) still construct, defaults applied
    d = cfg.to_dict()
    for k in ("loss_mix_lambda", "jspace_repo", "jspace_file", "jspace_revision"):
        d.pop(k)
    old = MLReconConfig(**d)  # type: ignore[arg-type]
    assert old.loss_mix_lambda == 0.5
    assert old.jspace_repo == ""
