"""arout_micro_batch + the heads.pt-derived loader helpers — the ptag AO-ladder port.

The decisive claim: the GROUPED prompt_tag path (k tagged forwards per crop, batched by layer
position) is exactly equal to the full-sweep-then-gather reference, for any seeded pick table.
"""

import json
from pathlib import Path

import pytest
import torch

from oracle_lens.pipeline.ao_arout import arout_micro_batch
from oracle_lens.pipeline.multilayer import LAYERS

D = 8


class _MockPtagRecon(torch.nn.Module):
    """Encodes (row-identity, layer) into the output so any mis-mapping is visible.

    forward(ids, mask, layer_idx=i) -> [b, 1, D] where out[b, 0] = f(ids[b], i);
    forward(ids, mask, layer_idx=None) -> [b, n_layers, D] (the full sweep).
    """

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.tag_ids = torch.zeros(n_layers, 6, dtype=torch.long)  # presence = ptag dispatch
        self.layers = list(LAYERS[-n_layers:])
        head = torch.nn.Module()
        head.linear = torch.nn.Linear(D, D)
        self.head = head

    def _one(self, ids: torch.Tensor, li: int) -> torch.Tensor:
        base = ids.float().sum(dim=1, keepdim=True)  # row identity
        return base * 1000.0 + li + torch.arange(D).float() / 100.0

    def forward(self, input_ids, attention_mask, layer_idx=None):  # type: ignore[no-untyped-def]
        if layer_idx is not None:
            return self._one(input_ids, int(layer_idx)).unsqueeze(1)
        return torch.stack(
            [self._one(input_ids, li) for li in range(self.n_layers)], dim=1
        )


class _MockLcRecon(torch.nn.Module):
    """lc shape: forward(ids, mask) -> [b, n_layers, D]; no tag_ids attribute."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = list(LAYERS[-n_layers:])

    def forward(self, input_ids, attention_mask):  # type: ignore[no-untyped-def]
        base = input_ids.float().sum(dim=1, keepdim=True)
        return torch.stack(
            [base * 1000.0 + li + torch.arange(D).float() / 100.0 for li in range(self.n_layers)],
            dim=1,
        )


def _picks(b: int, k: int, n_universe: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 999_983 + 17)
    return torch.rand(b, n_universe, generator=g).argsort(dim=1)[:, :k].to(torch.int8)


@pytest.mark.parametrize("b,k,n_ar,keep", [
    (7, 4, 12, list(range(12))),          # the ladder cell: 12-row AR, identity keep_pos
    (5, 2, 16, list(range(4, 16))),       # 16-row AR trimmed to L20+ (non-identity keep_pos)
    (3, 1, 12, list(range(12))),
])
def test_grouped_ptag_equals_gather_reference(b, k, n_ar, keep) -> None:
    recon = _MockPtagRecon(n_ar)
    ids = torch.arange(b * 3).reshape(b, 3)
    mask = torch.ones_like(ids)
    picks = _picks(b, k, len(keep))
    grouped = arout_micro_batch(recon, ids, mask, picks=picks, keep_pos=keep)
    # reference: full sweep over the AR's rows, then gather through keep_pos
    full = recon(ids, mask, layer_idx=None)
    kp = torch.as_tensor(keep, dtype=torch.long)
    idx = kp[picks.long()]
    ref = full.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
    assert torch.equal(grouped, ref)
    # and the --no-group debug arm is that reference verbatim
    ng = arout_micro_batch(recon, ids, mask, picks=picks, keep_pos=keep, no_group=True)
    assert torch.equal(ng, ref)


def test_lc_path_byte_identical_to_historical_expressions() -> None:
    b, k, n_ar = 6, 4, 12
    keep = list(range(12))
    recon = _MockLcRecon(n_ar)
    ids = torch.arange(b * 3).reshape(b, 3)
    mask = torch.ones_like(ids)
    picks = _picks(b, k, len(keep), seed=3)
    out = arout_micro_batch(recon, ids, mask, picks=picks, keep_pos=keep)
    preds = recon(ids, mask)
    idx = picks.long()
    ref = preds.gather(1, idx.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
    assert torch.equal(out, ref)
    # full-storage path with a trimmed window == keep_pos slice
    keep2 = list(range(4, 12))
    recon2 = _MockLcRecon(n_ar)
    out2 = arout_micro_batch(recon2, ids, mask, picks=None, keep_pos=keep2)
    assert torch.equal(out2, recon2(ids, mask)[:, keep2, :])


def test_duplicate_pick_trips_coverage_assert() -> None:
    recon = _MockPtagRecon(12)
    ids = torch.ones(2, 3, dtype=torch.long)
    mask = torch.ones_like(ids)
    bad = torch.tensor([[0, 0], [1, 2]], dtype=torch.int8)  # row 0 picks layer 0 twice
    with pytest.raises(AssertionError, match="duplicate"):
        arout_micro_batch(recon, ids, mask, picks=bad, keep_pos=list(range(12)))


def _write_heads(tmp: Path, state: dict) -> Path:
    d = tmp / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(state, d / "heads.pt")
    return d


def test_ar_layer_set_and_head_mode(tmp_path: Path) -> None:
    import sys

    pre = "mytorch_lightning" in sys.modules

    from oracle_lens.pipeline.ar_loader import ar_head_mode, ar_layer_set

    # lc, 16 rows (--drop-layers 0)
    lc = _write_heads(tmp_path / "a", {
        "head": {}, "layer_emb": {"weight": torch.zeros(16, D)},
    })
    assert ar_layer_set(lc) == tuple(LAYERS[-16:])
    assert ar_head_mode(lc) == "layer_conditioned"
    # ptag with explicit layers (head_state contract)
    pt = _write_heads(tmp_path / "b", {
        "head": {}, "tag_ids": torch.zeros(12, 6, dtype=torch.long),
        "layers": [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63],
    })
    assert ar_layer_set(pt) == (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63)
    assert ar_head_mode(pt) == "prompt_tag"
    # older ptag save without `layers` -> tag-row-count fallback
    pt2 = _write_heads(tmp_path / "c", {
        "head": {}, "tag_ids": torch.zeros(12, 6, dtype=torch.long),
    })
    assert ar_layer_set(pt2) == tuple(LAYERS[-12:])
    # the helpers must stay import-light (ar_loader invariant: no trainer stack) — delta
    # check, since sibling tests in the same process may already have loaded it
    assert pre or "mytorch_lightning" not in sys.modules


def test_head_layer_norm_from_meta(tmp_path: Path) -> None:
    from oracle_lens.pipeline.ar_loader import _head_layer_norm

    d = tmp_path / "r"
    d.mkdir(parents=True)
    assert _head_layer_norm(d) is True  # absent meta -> the iolens default
    (d / "meta.json").write_text(json.dumps({"config": {"head_layer_norm": False}}))
    assert _head_layer_norm(d) is False


def _mini_arout_dir(tmp: Path, shas: list[str]) -> Path:
    from safetensors.torch import save_file

    d = tmp / "arout"
    d.mkdir(parents=True)
    n_per = 3
    for s, sha in enumerate(shas):
        meta = {
            "lo": s * n_per, "hi": (s + 1) * n_per, "pool_fingerprint": "fp", "ar_run": "r",
            "d_model": D, "ao_layers": list(LAYERS), "ar_layers": list(LAYERS),
            "head_mode": "prompt_tag", "heads_sha": sha,
        }
        save_file(
            {"targets": torch.zeros(n_per, len(LAYERS), D, dtype=torch.bfloat16)},
            str(d / f"ao_arout_train_{s:04d}.safetensors"),
            metadata={"meta": json.dumps(meta)},
        )
    return d


def test_load_arout_surfaces_ar_identity_and_refuses_mixed(tmp_path: Path) -> None:
    from oracle_lens.pipeline.ao_ladder import load_arout

    d = _mini_arout_dir(tmp_path / "ok", ["sha-a", "sha-a"])
    _, top = load_arout(d, split="train", expect_fingerprint="fp")
    assert top["head_mode"] == "prompt_tag"
    assert top["heads_sha"] == "sha-a"

    bad = _mini_arout_dir(tmp_path / "bad", ["sha-a", "sha-b"])
    with pytest.raises(ValueError, match="DIFFERENT AR"):
        load_arout(bad, split="train", expect_fingerprint="fp")


def test_max_len_subsets_after_pick_draw(tmp_path: Path) -> None:
    """--max-len must SUBSET the seeded picks, not re-draw them: every surviving crop keeps the
    exact pick row (and arout row index) it had unfiltered, so a full-length k-sliced arout
    stays valid for a length-restricted run (the u64max32 cell)."""
    from test_ola_ao_ladder import _pool, _Tok, _write_arout_sliced

    from oracle_lens.pipeline.ao_ladder import AOLadderDataset, conv_split, load_arout

    pool = _pool(6)
    d = tmp_path / "arout"
    d.mkdir()
    full, pick_all = _write_arout_sliced(d, pool, k=2)
    arout, top = load_arout(d, split="train")
    train_idx, _ = conv_split(pool, n_val_crops=4, seed=1234)
    kw = {"tokenizer": _Tok(), "layers_per_crop": 2, "layer_seed": 0, "crop_sel": train_idx,
          "arout_pick": top["layer_pick"], "wrap_tags": False}
    base = AOLadderDataset(pool, arout, **kw)
    cut = int(pool.lengths[len(pool.lengths) // 2])
    filt = AOLadderDataset(pool, arout, max_len=cut, **kw)
    assert 0 < filt.n_crops < base.n_crops
    assert all(pool.lengths[int(li)] <= cut for li in filt.lens)
    # every surviving crop keeps its unfiltered pick row (keyed by original crop index)
    base_by_sel = {int(c): base.layer_pick[i] for i, c in enumerate(base.crop_sel)}
    for i, c in enumerate(filt.crop_sel):
        assert torch.equal(filt.layer_pick[i], base_by_sel[int(c)])
    # and the injected vector for a filtered item is the arout row of the ORIGINAL crop index
    it = filt[0]
    c0 = int(filt.crop_sel[0])
    expect = full[c0].gather(
        0, pick_all[c0, 0:1].long().unsqueeze(-1).expand(1, full.shape[-1])
    )[0].float()
    assert torch.equal(it["inject_vec"], expect)
