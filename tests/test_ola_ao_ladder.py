"""AO-ladder dataset/config wiring: arout alignment, arithmetic layer indexing, injection."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from oracle_lens.pipeline.ao_ladder import AOLadderDataset, ao_gt_config, load_arout
from oracle_lens.pipeline.ao_pool import CROP_LENGTHS, AOPool, build_ao_pool, pool_fingerprint
from oracle_lens.pipeline.inject import inject_gt
from oracle_lens.pipeline.multilayer import LAYERS
from oracle_lens.pipeline.soft_token_sft import layer_indices, soft_token_collate

D = 16  # small d_model for tests


class _Tok:
    """Minimal tokenizer stub: fixed tag ids, eos 2."""

    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [3, 4] if "</" not in text else [5, 6]}

    def decode(self, ids: list[int]) -> str:
        return " ".join(map(str, ids))


def _pool(n_starts: int = 5) -> AOPool:
    outs = [[100 + (c * 11 + t) % 80 for t in range(200)] for c in range(n_starts)]
    conv = torch.arange(n_starts)
    start = torch.zeros(n_starts, dtype=torch.long)
    return build_ao_pool(
        outs, conv, start, exclude={}, special_ids=frozenset(), seed=0, all_lengths=True
    )


def _write_arout(dir_: Path, pool: AOPool, *, n_shards: int = 2) -> torch.Tensor:
    rows, _lens = pool.crop_index()
    k = len(rows)
    full = torch.randn(k, len(LAYERS), D).to(torch.bfloat16)
    fp = pool_fingerprint(pool.ids, pool.keep)
    bounds = [0, k // n_shards, k]
    for s in range(n_shards):
        lo, hi = bounds[s], bounds[s + 1]
        meta = {"lo": lo, "hi": hi, "pool_fingerprint": fp, "ar_run": "test-ar", "d_model": D}
        save_file(
            {"targets": full[lo:hi]},
            str(dir_ / f"ao_arout_train_{s:04d}.safetensors"),
            metadata={"meta": json.dumps(meta)},
        )
    return full


def test_load_arout_verifies_contiguity_and_fingerprint(tmp_path: Path) -> None:
    pool = _pool()
    full = _write_arout(tmp_path, pool)
    fp = pool_fingerprint(pool.ids, pool.keep)
    arout, top = load_arout(tmp_path, split="train", expect_fingerprint=fp)
    assert top["n_crops"] == len(full) and top["ar_run"] == "test-ar"
    assert torch.equal(torch.as_tensor(arout[0]), full[0])
    with pytest.raises(ValueError, match="DIFFERENT pool"):
        load_arout(tmp_path, split="train", expect_fingerprint="deadbeef")


def test_dataset_indexing_layers_and_vectors(tmp_path: Path) -> None:
    pool = _pool()
    full = _write_arout(tmp_path, pool)
    arout, _ = load_arout(tmp_path, split="train")
    ds = AOLadderDataset(pool, arout, tokenizer=_Tok())
    assert len(ds) == pool.n_crops() * len(LAYERS)
    lys = layer_indices(ds)
    assert lys[:18] == [*range(17), 0]  # arithmetic assignment, crop-major
    # item: correct layer slice of the correct crop's reconstruction, span assembled with tags
    idx = 3 * len(LAYERS) + 7  # crop 3, layer_idx 7
    it = ds[idx]
    assert torch.equal(it["inject_vec"], full[3, 7].float())
    rows, lens = pool.crop_index()
    n = CROP_LENGTHS[int(lens[3])]
    assert int(it["span_len"]) == n
    assert it["target_ids"].tolist() == [3, 4, *pool.ids[int(rows[3]), :n].tolist(), 5, 6, 2]
    # n_crops cap = nested prefix
    ds2 = AOLadderDataset(pool, arout, tokenizer=_Tok(), n_crops=4)
    assert len(ds2) == 4 * len(LAYERS)


def test_ao_config_injects_one_frozen_constant_at_every_layer(tmp_path: Path) -> None:
    cfg = ao_gt_config(
        run_name="ao.test",
        ar_run="test-ar",
        scale=33.5,
        pool_path="pool.safetensors",
        n_examples=100,
    )
    assert cfg.transform == "scaled" and cfg.layers == LAYERS
    assert all(cfg.scale_for(ly) == 33.5 for ly in LAYERS)
    v = torch.randn(2, D)
    out = inject_gt(v, cfg.transform, alpha=cfg.alpha, scale=cfg.scale_for(44))
    assert torch.allclose(out, 33.5 * v)  # raw_scaled: raw reconstruction x one constant
    assert cfg.extra["ar_run"] == "test-ar" and cfg.extra["scale_mode"] == "global_frozen"
    # collate path works end-to-end with the dataset items (layer-pure batch)
    pool = _pool()
    _write_arout(tmp_path, pool)
    arout, _ = load_arout(tmp_path, split="train")
    ds = AOLadderDataset(pool, arout, tokenizer=_Tok())
    rows = [ds[0 * 17 + 5], ds[1 * 17 + 5]]  # two crops, same layer 5
    batch = soft_token_collate(rows, prompt_lens=[4] * 17, pad_id=0)
    assert int(batch["layer_idx"]) == 5
    assert int(batch["n_span_tokens"]) == int(rows[0]["span_len"]) + int(rows[1]["span_len"])


def test_group_indices_make_batches_length_pure(tmp_path: Path) -> None:
    from oracle_lens.pipeline.soft_token_sft import LayerBucketBatchSampler, group_indices

    pool = _pool(6)
    _write_arout(tmp_path, pool)
    arout, _ = load_arout(tmp_path, split="train")
    ds = AOLadderDataset(pool, arout, tokenizer=_Tok())
    groups = group_indices(ds)
    assert len(groups) == len(ds)
    # group encodes (layer, length): layer = g // 6 matches the arithmetic layer index
    from oracle_lens.pipeline.soft_token_sft import layer_indices

    lys = layer_indices(ds)
    assert all(g // 6 == ly for g, ly in zip(groups, lys, strict=True))
    sampler = LayerBucketBatchSampler(groups, micro_batch=2, seed=0)
    for batch in sampler:
        items = [ds[i] for i in batch]
        # layer-pure (collate contract) AND length-pure (zero padding)
        assert len({int(it["layer_idx"]) for it in items}) == 1
        assert len({int(it["span_len"]) for it in items}) == 1
        assert len({int(it["target_ids"].shape[0]) for it in items}) == 1


def test_layers_per_crop_subsampling_is_deterministic_and_diverse(tmp_path: Path) -> None:
    pool = _pool(8)
    _write_arout(tmp_path, pool)
    arout, _ = load_arout(tmp_path, split="train")
    ds4 = AOLadderDataset(pool, arout, tokenizer=_Tok(), layers_per_crop=4, layer_seed=0)
    assert len(ds4) == pool.n_crops() * 4
    # deterministic across constructions (exact-resume requirement)
    ds4b = AOLadderDataset(pool, arout, tokenizer=_Tok(), layers_per_crop=4, layer_seed=0)
    assert torch.equal(ds4.layer_pick, ds4b.layer_pick)
    # different seed -> different picks; per-crop picks are distinct layers
    ds4c = AOLadderDataset(pool, arout, tokenizer=_Tok(), layers_per_crop=4, layer_seed=1)
    assert not torch.equal(ds4.layer_pick, ds4c.layer_pick)
    for c in range(pool.n_crops()):
        picks = ds4.layer_pick[c].tolist()
        assert len(set(picks)) == 4
    # item layer matches the pick table and the vec is that layer's slice
    it = ds4[5 * 4 + 2]
    assert int(it["layer_idx"]) == int(ds4.layer_pick[5, 2])
    # aggregate layer coverage is roughly uniform (each layer ~ n_crops*4/17)
    counts = torch.bincount(ds4.layer_pick.long().reshape(-1), minlength=17).float()
    assert counts.min() > 0
    # k=0 -> full cross product unchanged
    ds17 = AOLadderDataset(pool, arout, tokenizer=_Tok(), layers_per_crop=0)
    assert len(ds17) == pool.n_crops() * 17


def test_conv_split_reserves_whole_conversations(tmp_path: Path) -> None:
    """Val must reserve whole CONVERSATIONS: a window's crops are nested prefixes of one text,
    and two windows of a conversation can overlap, so crop/window splits leak text."""
    from oracle_lens.pipeline.ao_ladder import conv_split

    # 10 windows over 5 conversations (2 windows each) -> a crop-level split WOULD leak
    outs = [[100 + (c * 11 + t) % 80 for t in range(300)] for c in range(5)]
    conv = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    start = torch.tensor([0, 100, 0, 100, 0, 100, 0, 100, 0, 100])
    pool = build_ao_pool(
        outs, conv, start, exclude={}, special_ids=frozenset(), seed=0, all_lengths=True
    )
    full = _write_arout(tmp_path, pool)
    arout, _ = load_arout(tmp_path, split="train")
    train_idx, val_idx = conv_split(pool, n_val_crops=6, seed=0)

    rows, _lens = pool.crop_index()
    tr_convs = set(pool.conv[rows[train_idx]].tolist())
    va_convs = set(pool.conv[rows[val_idx]].tolist())
    assert va_convs and not (tr_convs & va_convs)  # whole conversations, no overlap
    assert len(train_idx) + len(val_idx) == len(rows)  # partition

    tr = AOLadderDataset(pool, arout, tokenizer=_Tok(), crop_sel=train_idx, layers_per_crop=2)
    va = AOLadderDataset(pool, arout, tokenizer=_Tok(), crop_sel=val_idx, layers_per_crop=2)
    assert tr.n_crops == len(train_idx) and va.n_crops == len(val_idx)
    # arout lookup follows crop_sel, not position: val item 0 reads arout[val_idx[0]]
    it = va[0]
    assert torch.equal(it["inject_vec"], full[int(val_idx[0]), int(it["layer_idx"])].float())


def _write_arout_sliced(
    dir_: Path,
    pool: AOPool,
    *,
    k: int = 2,
    layer_seed: int = 0,
    split_seed: int = 1234,
    n_val_crops: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicates ao_precompute_cluster's k-sliced storage: [n, k, D] + layer_pick, 2 shards."""
    from oracle_lens.pipeline.ao_ladder import conv_split

    rows, _lens = pool.crop_index()
    n = len(rows)
    full = torch.randn(n, len(LAYERS), D).to(torch.bfloat16)
    train_idx, val_idx = conv_split(pool, n_val_crops=n_val_crops, seed=split_seed, n_avail=n)
    gen_t = torch.Generator().manual_seed(layer_seed * 999_983 + 17)
    train_pick = torch.rand(len(train_idx), len(LAYERS), generator=gen_t).argsort(dim=1)[:, :k]
    gen_v = torch.Generator().manual_seed((layer_seed + 1) * 999_983 + 17)
    val_pick = torch.rand(len(val_idx), len(LAYERS), generator=gen_v).argsort(dim=1)[:, :k]
    pick_all = torch.zeros(n, k, dtype=torch.int8)
    pick_all[train_idx] = train_pick.to(torch.int8)
    pick_all[val_idx] = val_pick.to(torch.int8)
    sliced = full.gather(1, pick_all.long().unsqueeze(-1).expand(-1, -1, D))
    fp = pool_fingerprint(pool.ids, pool.keep)
    bounds = [0, n // 2, n]
    for s in range(2):
        lo, hi = bounds[s], bounds[s + 1]
        meta = {
            "lo": lo, "hi": hi, "pool_fingerprint": fp, "ar_run": "test-ar", "d_model": D,
            "layers_per_crop": k, "layer_seed": layer_seed, "split_seed": split_seed,
            "n_val_crops": n_val_crops,
        }
        save_file(
            {"targets": sliced[lo:hi], "layer_pick": pick_all[lo:hi].contiguous()},
            str(dir_ / f"ao_arout_train_{s:04d}.safetensors"),
            metadata={"meta": json.dumps(meta)},
        )
    return full, pick_all


def test_rand_rows_are_prefix_stable() -> None:
    """The k-slicing contract: torch.rand fills row-major, so a shorter draw with the same seed
    is exactly the prefix — trainer-side n_crops caps never change surviving crops' picks."""
    g1 = torch.Generator().manual_seed(17)
    g2 = torch.Generator().manual_seed(17)
    a = torch.rand(100, len(LAYERS), generator=g1)
    b = torch.rand(60, len(LAYERS), generator=g2)
    assert torch.equal(a[:60], b)


def test_k_sliced_arout_roundtrip(tmp_path: Path) -> None:
    """A k-sliced arout feeds the dataset the SAME vectors as full storage would."""
    from oracle_lens.pipeline.ao_ladder import conv_split

    pool = _pool()
    full, pick_all = _write_arout_sliced(tmp_path, pool, k=2)
    arout, top = load_arout(tmp_path, split="train")
    assert top["slice_meta"]["layers_per_crop"] == 2
    pick = top["layer_pick"]
    assert torch.equal(pick, pick_all)
    n = pool.n_crops()
    train_idx, val_idx = conv_split(pool, n_val_crops=4, seed=1234, n_avail=n)
    ds = AOLadderDataset(
        pool, arout, tokenizer=_Tok(), crop_sel=train_idx, layers_per_crop=2, layer_seed=0,
        arout_pick=pick,
    )
    for idx in range(min(len(ds), 8)):
        crop, j = divmod(idx, 2)
        it = ds[idx]
        canonical = int(train_idx[crop])
        layer_idx = int(it["layer_idx"])
        assert layer_idx == int(pick_all[canonical, j])
        assert torch.equal(it["inject_vec"], full[canonical, layer_idx].float())
    # val side (layer_seed + 1) round-trips too
    ds_v = AOLadderDataset(
        pool, arout, tokenizer=_Tok(), crop_sel=val_idx, layers_per_crop=2, layer_seed=1,
        arout_pick=pick,
    )
    it = ds_v[0]
    canonical = int(val_idx[0])
    assert torch.equal(it["inject_vec"], full[canonical, int(it["layer_idx"])].float())


def test_k_sliced_arout_refuses_pick_drift(tmp_path: Path) -> None:
    """A precompute made under different seeds must be refused, not silently mis-injected."""
    from oracle_lens.pipeline.ao_ladder import conv_split

    pool = _pool()
    _write_arout_sliced(tmp_path, pool, k=2, layer_seed=0)
    arout, top = load_arout(tmp_path, split="train")
    n = pool.n_crops()
    train_idx, _ = conv_split(pool, n_val_crops=4, seed=1234, n_avail=n)
    with pytest.raises(ValueError, match="does not match"):
        AOLadderDataset(
            pool, arout, tokenizer=_Tok(), crop_sel=train_idx, layers_per_crop=2,
            layer_seed=99, arout_pick=top["layer_pick"],
        )


def test_adopt_arout_picks_takes_stored_table_and_feeds_matching_vectors(
    tmp_path: Path,
) -> None:
    """Merged replay-mix pools shift crop indices, so the seeded re-draw can never match the
    stored picks — adopt_arout_picks must take the stored table wholesale and keep row j of the
    sliced storage paired with stored pick j."""
    pool = _pool()
    full, pick_all = _write_arout_sliced(tmp_path, pool, k=2, layer_seed=0)
    arout, top = load_arout(tmp_path, split="train")
    # a "wrong" seed stands in for the merged-pool index shift: the re-draw differs from stored
    ds = AOLadderDataset(
        pool, arout, tokenizer=_Tok(), layers_per_crop=2, layer_seed=99,
        arout_pick=top["layer_pick"], adopt_arout_picks=True,
    )
    assert torch.equal(ds.layer_pick, pick_all)
    for idx in range(min(len(ds), 8)):
        crop, j = divmod(idx, 2)
        it = ds[idx]
        assert int(it["layer_idx"]) == int(pick_all[crop, j])
        assert torch.equal(it["inject_vec"], full[crop, int(pick_all[crop, j])].float())
    # same table under a crop_sel subset
    sel = torch.tensor([1, 3, 4])
    ds_s = AOLadderDataset(
        pool, arout, tokenizer=_Tok(), crop_sel=sel, layers_per_crop=2, layer_seed=99,
        arout_pick=top["layer_pick"], adopt_arout_picks=True,
    )
    assert torch.equal(ds_s.layer_pick, pick_all[sel])
    # stored picks outside the dataset's layer universe are refused, not silently mis-mapped
    bad = top["layer_pick"].clone()
    bad[0, 0] = 99
    with pytest.raises(ValueError, match="layer universe"):
        AOLadderDataset(
            pool, arout, tokenizer=_Tok(), layers_per_crop=2, layer_seed=0,
            arout_pick=bad, adopt_arout_picks=True,
        )


def test_soft_token_scale_for_handles_every_scale_free_arm() -> None:
    """SoftTokenConfig.scale_for is a SECOND copy of gt_train's (soft_token_sft is the shared
    core the AO ladder actually instantiates). Patching only one left the AO arms raising
    KeyError — pin both so the duplicate cannot drift again."""
    from dataclasses import replace
    from typing import Any, cast

    from oracle_lens.pipeline.ao_ladder import ao_gt_config
    from oracle_lens.pipeline.inject import SCALE_FREE_GT

    base = ao_gt_config(run_name="t", ar_run="a", scale=1.0, pool_path="p", n_examples=8)
    for t in sorted(SCALE_FREE_GT):
        cfg = replace(base, transform=cast(Any, t), scales={})
        assert cfg.scale_for(44) == 1.0, f"{t} must not index the empty scales dict"


def test_ao_arms_are_multilayer_and_j_excludes_l63() -> None:
    """The ARs are layer-conditioned over 17 layers, so the AO must be too — one model, the
    prompt names the layer. The J arm covers exactly the 16 J-covered layers."""
    from dataclasses import replace

    from oracle_lens.pipeline.ao_ladder import ao_gt_config
    from oracle_lens.pipeline.multilayer import LAYERS

    base = ao_gt_config(run_name="t", ar_run="a", scale=1.0, pool_path="p", n_examples=8)
    assert base.layers == LAYERS and len(LAYERS) == 17
    assert replace(base, transform="wht_unit", scales={}).layers == LAYERS
    jcov = tuple(ly for ly in LAYERS if ly != 63)
    assert len(jcov) == 16 and 63 not in jcov
