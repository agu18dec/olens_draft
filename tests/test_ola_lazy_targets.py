"""CPU tests for LazyTargets / load_multilayer_shards_lazy: the mmap'd per-row target reader must
match the eager loader byte-for-byte through select() chains (the >RAM ladder depends on it)."""

import torch

from oracle_lens.pipeline.multilayer import (
    MultiLayerPairs,
    MultiLayerShardMeta,
    load_multilayer_shards,
    load_multilayer_shards_lazy,
    save_multilayer_shard,
)


def _shard(path, n, layers=(0, 4), d=6, base=0):
    """A tiny shard: row i has span [base+i] (len 1) and targets filled with (base+i)."""
    offsets = torch.arange(n + 1, dtype=torch.int64)
    span_ids = torch.arange(base, base + n, dtype=torch.int64)
    targets = torch.stack(
        [torch.full((len(layers), d), float(base + i)) for i in range(n)]
    ).to(torch.bfloat16)
    pairs = MultiLayerPairs(
        span_ids=span_ids,
        offsets=offsets,
        targets=targets,
        layers=layers,
        conv_index=torch.zeros(n, dtype=torch.int64),
        prev_pos=torch.zeros(n, dtype=torch.int64),
        prev_token_id=torch.zeros(n, dtype=torch.int64),
        prev_is_assistant=torch.zeros(n, dtype=torch.bool),
    )
    meta = MultiLayerShardMeta(
        model_id="m", dataset_id="d", layers=layers, t_max=1, n_max=1, n_min=1,
        length_law="u", region_start=0, region_end=1, split="train", seed=0,
        git_commit="x", n_pairs=n,
    )
    save_multilayer_shard(path, pairs, meta)


def test_lazy_matches_eager(tmp_path):
    p0, p1 = tmp_path / "pairs_train_0.safetensors", tmp_path / "pairs_train_1.safetensors"
    _shard(p0, 5, base=0)
    _shard(p1, 7, base=100)  # second shard rows read as 100..106 -> checks the cross-shard offset
    eager, _ = load_multilayer_shards([p0, p1])
    lazy, _ = load_multilayer_shards_lazy([p0, p1])
    assert len(lazy) == len(eager) == 12
    torch.testing.assert_close(lazy.span_ids, eager.span_ids)
    torch.testing.assert_close(lazy.offsets, eager.offsets)
    for i in range(12):
        torch.testing.assert_close(lazy.targets[i].float(), eager.targets[i].float())


def test_select_view_composition(tmp_path):
    p0, p1 = tmp_path / "pairs_train_0.safetensors", tmp_path / "pairs_train_1.safetensors"
    _shard(p0, 5, base=0)
    _shard(p1, 7, base=100)
    lazy, _ = load_multilayer_shards_lazy([p0, p1])
    # two chained selects (mirror train_ml_recon: filter -> arange subset)
    keep = lazy.select(torch.tensor([0, 2, 4, 6, 8, 10, 11]))
    sub = keep.select(torch.tensor([1, 3, 5]))  # keep-idx 1,3,5 -> global rows 2, 6, 10
    # shard0 rows 0..4 hold value=row; shard1 rows 5..11 hold value=100+local -> row6=101, row10=105
    assert [sub.targets[i][0, 0].item() for i in range(3)] == [2.0, 101.0, 105.0]


def test_pickle_drops_handles(tmp_path):
    import pickle

    p0 = tmp_path / "pairs_train_0.safetensors"
    _shard(p0, 3, base=0)
    lazy, _ = load_multilayer_shards_lazy([p0])
    _ = lazy.targets[0]  # opens a handle
    restored = pickle.loads(pickle.dumps(lazy.targets))  # must survive the mp.spawn / worker fork
    assert restored[2][0, 0].item() == 2.0
