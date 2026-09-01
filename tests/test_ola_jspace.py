"""JSpace metric ruler: reduction to centered cosine, transpose convention, None-skip, loader."""

from pathlib import Path

import pytest
import torch

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.jspace import JSpace, MetricSpace, load_jspaces
from oracle_lens.pipeline.multilayer_reconstructor import (
    multilayer_fve,
    multilayer_whitened_cosine_loss,
)

D = 8
NL = 3


def ident_whiteners(n: int = NL, d: int = D) -> list[Whitener]:
    return [Whitener(mu=torch.zeros(d), w=torch.eye(d), ridge_c=0.0) for _ in range(n)]


def ident_jspaces(n: int = NL, d: int = D) -> list[JSpace]:
    return [JSpace(mu=torch.zeros(d), j=torch.eye(d)) for _ in range(n)]


def test_identity_jspace_matches_identity_whitener_loss() -> None:
    gen = torch.Generator().manual_seed(0)
    p = torch.randn(4, NL, D, generator=gen)
    t = torch.randn(4, NL, D, generator=gen)
    loss_w = multilayer_whitened_cosine_loss(p, t, ident_whiteners())
    loss_j = multilayer_whitened_cosine_loss(p, t, ident_jspaces())
    assert loss_j.item() == pytest.approx(loss_w.item(), abs=1e-6)


def test_jspace_known_value_transpose_and_centering() -> None:
    # z = (x - μ) @ Jᵀ with J = [[0,1],[1,0]], μ = [1,0]: x=[2,3] → (x-μ)=[1,3] → z=[3,1]
    s = JSpace(mu=torch.tensor([1.0, 0.0]), j=torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    z = s.whiten(torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(z, torch.tensor([[3.0, 1.0]]))


def test_asymmetric_j_uses_row_convention() -> None:
    # J = [[1,2],[0,1]] (not symmetric): z = J @ (x-μ) per row, i.e. x @ Jᵀ.
    s = JSpace(mu=torch.zeros(2), j=torch.tensor([[1.0, 2.0], [0.0, 1.0]]))
    z = s.whiten(torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(z, torch.tensor([[3.0, 1.0]]))  # [1*1+2*1, 0*1+1*1]


def test_loss_none_skip_matches_sliced_loss() -> None:
    gen = torch.Generator().manual_seed(1)
    p = torch.randn(5, NL, D, generator=gen)
    t = torch.randn(5, NL, D, generator=gen)
    spaces = ident_jspaces()
    with_hole = multilayer_whitened_cosine_loss(p, t, [spaces[0], None, spaces[2]])
    sliced = multilayer_whitened_cosine_loss(p[:, [0, 2]], t[:, [0, 2]], [spaces[0], spaces[2]])
    assert with_hole.item() == pytest.approx(sliced.item(), abs=1e-6)


def test_loss_all_none_raises() -> None:
    t = torch.randn(2, NL, D, generator=torch.Generator().manual_seed(2))
    with pytest.raises(ValueError, match="all layers excluded"):
        multilayer_whitened_cosine_loss(t, t, [None, None, None])


def test_fve_accepts_jspaces() -> None:
    t = torch.randn(5, NL, D, generator=torch.Generator().manual_seed(3))
    fve = multilayer_fve(t * 2.0, t, ident_jspaces())  # scaled → cos=1 → cos²=1
    assert fve.shape == (NL,)
    assert torch.allclose(fve, torch.ones(NL), atol=1e-5)


def test_both_rulers_satisfy_metric_space_protocol() -> None:
    assert isinstance(ident_whiteners(1)[0], MetricSpace)
    assert isinstance(ident_jspaces(1)[0], MetricSpace)


def test_load_jspaces_covers_only_stacked_layers(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from jlens.lens import JacobianLens

    d = 4
    gen = torch.Generator().manual_seed(4)
    jac = {0: torch.randn(d, d, generator=gen), 1: torch.randn(d, d, generator=gen)}
    lens_path = tmp_path / "lens.pt"
    JacobianLens(jac, n_prompts=1, d_model=d).save(str(lens_path))
    for lyr in (0, 1):
        save_file(
            {"mu": torch.full((d,), float(lyr)), "cov": torch.eye(d)},
            str(tmp_path / f"whitening_L{lyr}.safetensors"),
        )
    # layer 63 has no J row and no mu file — must be ABSENT, not defaulted
    spaces = load_jspaces(str(lens_path), "ignored", tmp_path, (0, 1, 63))
    assert sorted(spaces) == [0, 1]
    assert torch.allclose(spaces[1].mu, torch.ones(d))
    # fp16 round-trip through save(): compare loosely
    assert torch.allclose(spaces[0].j, jac[0], atol=2e-3)


def test_mixed_space_equals_lambda_weighted_sum_of_losses() -> None:
    """MixedSpace exists so the mixed objective needs NO change to the loss fn. That only holds
    if 2(1-cos) through it equals (1-lam)*L_whitened + lam*L_J exactly — pin it."""
    import torch

    from oracle_lens.pipeline.jspace import JSpace, MixedSpace
    from oracle_lens.pipeline.multilayer_reconstructor import multilayer_whitened_cosine_loss

    torch.manual_seed(0)
    d, lam = 16, 0.5
    mu = torch.randn(d)
    w = JSpace(mu=mu, j=torch.randn(d, d))  # any MetricSpace works as the "whitened" block
    j = JSpace(mu=mu, j=torch.randn(d, d))
    preds, targets = torch.randn(6, 1, d), torch.randn(6, 1, d)

    lw = multilayer_whitened_cosine_loss(preds, targets, [w])
    lj = multilayer_whitened_cosine_loss(preds, targets, [j])
    lmix = multilayer_whitened_cosine_loss(
        preds, targets, [MixedSpace(whitener=w, jspace=j, lam=lam)]
    )
    torch.testing.assert_close((1 - lam) * lw + lam * lj, lmix, atol=1e-5, rtol=1e-4)


def test_mixed_space_endpoints_reduce_to_each_arm() -> None:
    """lam=0 must be exactly the whitened arm and lam=1 exactly the J arm, so the mixed run's
    endpoints ARE the two runs we already have."""
    import torch

    from oracle_lens.pipeline.jspace import JSpace, MixedSpace
    from oracle_lens.pipeline.multilayer_reconstructor import multilayer_whitened_cosine_loss

    torch.manual_seed(1)
    d = 16
    mu = torch.randn(d)
    w = JSpace(mu=mu, j=torch.randn(d, d))
    j = JSpace(mu=mu, j=torch.randn(d, d))
    preds, targets = torch.randn(5, 1, d), torch.randn(5, 1, d)
    for lam, ref in ((0.0, w), (1.0, j)):
        got = multilayer_whitened_cosine_loss(
            preds, targets, [MixedSpace(whitener=w, jspace=j, lam=lam)]
        )
        want = multilayer_whitened_cosine_loss(preds, targets, [ref])
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
