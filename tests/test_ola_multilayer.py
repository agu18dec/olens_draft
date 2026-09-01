"""Multi-layer shard round-trip, per-layer target stack, vectorized select, concat load."""

from pathlib import Path

import torch

from oracle_lens.pipeline.multilayer import (
    LAYERS,
    MultiLayerPairs,
    MultiLayerShardMeta,
    load_multilayer_shards,
    save_multilayer_shard,
)

D = 8
NL = 3
TEST_LAYERS = (20, 24, 28)


def make_ml_pairs(id_rows: list[list[int]], *, seed: int = 0) -> MultiLayerPairs:
    gen = torch.Generator().manual_seed(seed)
    n = len(id_rows)
    offsets = torch.zeros(n + 1, dtype=torch.int64)
    for i, row in enumerate(id_rows):
        offsets[i + 1] = offsets[i] + len(row)
    return MultiLayerPairs(
        span_ids=torch.tensor([t for row in id_rows for t in row], dtype=torch.int32),
        offsets=offsets,
        targets=torch.randn(n, NL, D, generator=gen).to(torch.bfloat16),
        layers=TEST_LAYERS,
        conv_index=torch.arange(n, dtype=torch.int64),
        prev_pos=torch.arange(1, n + 1, dtype=torch.int64),
        prev_token_id=torch.arange(100, 100 + n, dtype=torch.int64),
        prev_is_assistant=torch.ones(n, dtype=torch.bool),
    )


def make_meta(n_pairs: int, split: str = "train") -> MultiLayerShardMeta:
    return MultiLayerShardMeta(
        model_id="test/model",
        dataset_id="test/data",
        layers=TEST_LAYERS,
        t_max=2048,
        n_max=1024,
        n_min=1,
        length_law="loguniform",
        region_start=0,
        region_end=1000,
        split=split,
        seed=0,
        git_commit="deadbeef",
        n_pairs=n_pairs,
    )


def test_default_layers_are_0_to_60_every_4th_plus_last() -> None:
    # Widened from the original 11-layer L20-60 band to the full stack: every 4th layer from 0,
    # plus the final block 63 (the output residual), which is not on the every-4th grid.
    assert LAYERS == (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63)
    assert len(LAYERS) == 17


def test_roundtrip_preserves_targets_and_layers(tmp_path: Path) -> None:
    pairs = make_ml_pairs([[7, 8, 9], [10, 11, 12, 13]])
    path = tmp_path / "ml_train_0000.safetensors"
    save_multilayer_shard(path, pairs, make_meta(len(pairs)))
    loaded, meta = load_multilayer_shards([path])
    assert meta.layers == TEST_LAYERS
    assert loaded.targets.shape == (2, NL, D)
    assert torch.equal(loaded.row_ids(1), torch.tensor([10, 11, 12, 13]))
    assert loaded.lengths.tolist() == [3, 4]


def test_meta_layers_roundtrip_through_metadata() -> None:
    meta = make_meta(5)
    back = MultiLayerShardMeta.from_metadata(meta.to_metadata())
    assert back.layers == TEST_LAYERS and back.n_pairs == 5


def test_select_keeps_layer_axis(tmp_path: Path) -> None:
    pairs = make_ml_pairs([[1, 2], [3, 4, 5], [6]])
    sub = pairs.select(torch.tensor([0, 2]))
    assert len(sub) == 2
    assert sub.targets.shape == (2, NL, D)
    assert torch.equal(sub.row_ids(0), torch.tensor([1, 2]))
    assert torch.equal(sub.row_ids(1), torch.tensor([6]))


def test_concat_load_shifts_offsets(tmp_path: Path) -> None:
    p0, p1 = tmp_path / "ml_train_0000.safetensors", tmp_path / "ml_train_0001.safetensors"
    save_multilayer_shard(p0, make_ml_pairs([[1, 2], [3]]), make_meta(2))
    save_multilayer_shard(p1, make_ml_pairs([[4, 5, 6]]), make_meta(1))
    loaded, _ = load_multilayer_shards([p0, p1])
    assert len(loaded) == 3
    assert torch.equal(loaded.row_ids(2), torch.tensor([4, 5, 6]))
