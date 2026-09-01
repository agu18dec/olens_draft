"""CPU tests for the stratified-uniform crop construction (user 2026-07-27): prefix-crops of
existing rows are valid uniform-N pairs (same target), the pool is exact-uniform per N, ordered
round-robin so every prefix is a nested exact-uniform rung, and no (row, N) repeats."""

import torch

from oracle_lens.pipeline.multilayer import (
    CroppedPairs,
    MultiLayerPairs,
    stratified_crop_pool,
)


def _pairs(lengths: list[int]) -> MultiLayerPairs:
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor(lengths), 0, out=offsets[1:])
    total = int(offsets[-1])
    return MultiLayerPairs(
        span_ids=torch.arange(total),
        offsets=offsets,
        targets=torch.randn(len(lengths), 2, 4),
        layers=(0, 4),
        conv_index=torch.zeros(len(lengths), dtype=torch.int64),
        prev_pos=torch.zeros(len(lengths), dtype=torch.int64),
        prev_token_id=torch.zeros(len(lengths), dtype=torch.int64),
        prev_is_assistant=torch.zeros(len(lengths), dtype=torch.bool),
    )


def test_pool_exact_uniform_and_nested() -> None:
    torch.manual_seed(0)
    lengths = [4] * 50 + [2] * 30 + [1] * 20  # every row len>=1; 50 rows >=4
    rows, lens = stratified_crop_pool(torch.tensor(lengths), 1, 4, per_n=10, seed=0)
    assert len(rows) == 40
    # exact uniform per N overall AND on every 4k prefix (round-robin nesting)
    for r in (8, 20, 40):
        c = torch.bincount(lens[:r], minlength=5)[1:5]
        assert (c == r // 4).all(), (r, c)
    # no (row, N) duplicates; crops only from rows long enough
    assert len({(int(a), int(b)) for a, b in zip(rows, lens, strict=True)}) == 40
    src_lens = torch.tensor(lengths)
    assert (src_lens[rows] >= lens).all()


def test_scarce_length_capped() -> None:
    lengths = [1] * 100 + [3] * 2  # only 2 rows can serve N=3
    _rows, lens = stratified_crop_pool(torch.tensor(lengths), 1, 3, per_n=5, seed=0)
    assert int((lens == 3).sum()) == 2  # capped at available
    assert int((lens == 1).sum()) == 5


def test_cropped_pairs_prefix_and_same_target() -> None:
    src = _pairs([5, 3, 8])
    cp = CroppedPairs(src, torch.tensor([2, 0]), torch.tensor([4, 2]))
    assert len(cp) == 2 and cp.lengths.tolist() == [4, 2]
    # crop 0 = first 4 ids of source row 2; SAME target tensor as source row 2
    assert torch.equal(cp.row_ids(0), src.row_ids(2)[:4])
    assert torch.equal(cp.targets[0], src.targets[2])
    sub = cp.select(torch.tensor([1]))
    assert torch.equal(sub.row_ids(0), src.row_ids(0)[:2])
    assert torch.equal(sub.targets[0], src.targets[0])


def test_localize_crops_identical_targets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """localize_crops must return byte-identical targets via the local file, with ids untouched."""
    from safetensors.torch import save_file

    from oracle_lens.pipeline.multilayer import LazyTargets, localize_crops

    torch.manual_seed(0)
    # two fake shards of 6+5 rows, [2, 4] targets
    t1, t2 = torch.randn(6, 2, 4).bfloat16(), torch.randn(5, 2, 4).bfloat16()
    p1, p2 = tmp_path / "s1.safetensors", tmp_path / "s2.safetensors"
    save_file({"targets": t1}, str(p1))
    save_file({"targets": t2}, str(p2))
    lazy = LazyTargets([p1, p2], [6, 5], 2, 4)
    src = _pairs([3, 5, 8, 2, 4, 6, 7, 3, 5, 9, 4])
    src = MultiLayerPairs(**{**src.__dict__, "targets": lazy})
    cp = CroppedPairs(src, torch.tensor([9, 1, 6, 9]), torch.tensor([4, 2, 3, 1]))
    ref = [cp.targets[i].clone() for i in range(4)]
    ref_ids = [cp.row_ids(i).clone() for i in range(4)]
    (loc,) = localize_crops([cp], str(tmp_path / "local.bin"), chunk_rows=3)
    for i in range(4):
        assert torch.equal(loc.targets[i], ref[i])
        assert torch.equal(loc.row_ids(i), ref_ids[i])
    sub = loc.select(torch.tensor([3, 0]))  # select keeps the override + remap
    assert torch.equal(sub.targets[0], ref[3]) and torch.equal(sub.targets[1], ref[0])


def test_streaming_dataset_coverage_and_disjoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Every crop yielded exactly once per epoch; ranks disjoint; full coverage."""
    from safetensors.torch import save_file

    from oracle_lens.pipeline.multilayer import LazyTargets, StreamingCropDataset

    torch.manual_seed(0)
    t1 = torch.randn(11, 2, 4).bfloat16()
    p1 = tmp_path / "s.safetensors"
    save_file({"targets": t1}, str(p1))
    src = _pairs([3, 5, 8, 2, 4, 6, 7, 3, 5, 9, 4])
    src = MultiLayerPairs(**{**src.__dict__, "targets": LazyTargets([p1], [11], 2, 4)})
    row_idx = torch.tensor([0, 2, 4, 6, 8, 10, 1, 3, 9])
    crops = CroppedPairs(src, row_idx, torch.ones(9, dtype=torch.int64))
    seen: list[tuple] = []
    for rank in range(2):
        ds = StreamingCropDataset(crops, rank=rank, world=2, buffer_rows=4, seed=0)
        rows = list(ds)
        assert len(rows) == len(ds)
        seen += [tuple(r["target"].flatten().tolist()) for r in rows]
    assert len(seen) == 9  # full coverage across ranks, nothing dropped
    # targets match the source rows exactly (as a multiset)
    want = [tuple(src.targets[int(r)].float().flatten().tolist()) for r in row_idx]
    assert sorted(seen) == sorted(want)


def test_streaming_exact_shuffle_buffer_covers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from safetensors.torch import save_file

    from oracle_lens.pipeline.multilayer import LazyTargets, StreamingCropDataset

    t1 = torch.randn(6, 1, 2).bfloat16()
    p1 = tmp_path / "s.safetensors"
    save_file({"targets": t1}, str(p1))
    src = _pairs([2, 2, 2, 2, 2, 2])
    src = MultiLayerPairs(**{**src.__dict__, "targets": LazyTargets([p1], [6], 1, 2)})
    crops = CroppedPairs(src, torch.arange(6), torch.ones(6, dtype=torch.int64))
    ds = StreamingCropDataset(crops, rank=0, world=1, buffer_rows=100, seed=1)
    a = [tuple(r["target"].flatten().tolist()) for r in ds]
    b = [tuple(r["target"].flatten().tolist()) for r in ds]
    assert sorted(a) == sorted(b) and len(a) == 6  # same multiset, full coverage both epochs


def test_streaming_worker_substride_coverage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Worker sub-strides (wid, wn) partition each rank's stream: union = full, no overlap."""
    from safetensors.torch import save_file

    from oracle_lens.pipeline.multilayer import LazyTargets, StreamingCropDataset

    t1 = torch.randn(12, 1, 2).bfloat16()
    p1 = tmp_path / "s.safetensors"
    save_file({"targets": t1}, str(p1))
    src = _pairs([2] * 12)
    src = MultiLayerPairs(**{**src.__dict__, "targets": LazyTargets([p1], [12], 1, 2)})
    crops = CroppedPairs(src, torch.arange(12), torch.ones(12, dtype=torch.int64))
    ds = StreamingCropDataset(crops, rank=0, world=1, buffer_rows=4, seed=0)
    seen: list[tuple] = []
    for wid in range(3):
        rows = list(ds._iter_worker(wid, 3))
        seen += [tuple(r["target"].flatten().tolist()) for r in rows]
    want = [tuple(src.targets[i].float().flatten().tolist()) for i in range(12)]
    assert sorted(seen) == sorted(want)  # exact partition across workers


def test_second_view_pool_disjoint_from_prior() -> None:
    """A second-view pool built with exclude=prior shares ZERO (row, N) pairs with the prior."""
    torch.manual_seed(0)
    lengths = torch.tensor([4] * 200 + [2] * 100 + [1] * 100)
    prior = stratified_crop_pool(lengths, 1, 4, per_n=30, seed=1234)
    second = stratified_crop_pool(lengths, 1, 4, per_n=30, seed=5678, exclude=prior)
    a = {(int(r), int(n)) for r, n in zip(*[t.tolist() for t in prior], strict=True)}
    b = {(int(r), int(n)) for r, n in zip(*[t.tolist() for t in second], strict=True)}
    assert not (a & b)  # fully disjoint
    assert len(b) == 120  # still full-size (pool has room after exclusion)
