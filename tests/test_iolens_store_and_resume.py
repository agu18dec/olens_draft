"""iolens store round-trips, loss-space arms, and the streaming resume fast-forward."""

from pathlib import Path
from typing import Any

import pytest
import torch

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.multilayer import (
    CroppedPairs,
    MultiLayerPairs,
    StreamingCropDataset,
)
from oracle_lens.pipeline.multilayer_reconstructor import multilayer_recon_loss
from oracle_lens.pipeline.rollout_store import (
    SPLIT_FRACTIONS,
    SPLITS,
    RolloutShardMeta,
    load_rollout_shards,
    save_rollout_shard,
    seed_hash64,
    split_of_key,
)

D = 8
N_LAYERS = 3


def test_rollout_store_roundtrip(tmp_path: Path) -> None:
    convs = [[1, 2, 3, 10, 11], [4, 5, 20, 21, 22, 23]]
    meta = RolloutShardMeta(
        model_id="m", mode="chat", engine="sglang", engine_version="0", tokenizer_sha="x",
        temperature=1.0, top_p=1.0, max_new=512, n_convs=2, n_prompt_tokens=5,
        n_output_tokens=6, git_commit="t",
    )
    p = tmp_path / "rollouts_0000.safetensors"
    save_rollout_shard(
        p, conv_ids=convs, prompt_lens=[3, 2], seed_hashes=[seed_hash64("a"), seed_hash64("b")],
        split_ids=[0, 3], meta=meta,
    )
    r = load_rollout_shards([p])
    assert r.prompt_ids(0).tolist() == [1, 2, 3] and r.output_ids(0).tolist() == [10, 11]
    assert r.prompt_ids(1).tolist() == [4, 5] and r.output_ids(1).tolist() == [20, 21, 22, 23]
    assert r.counts() == {"n_convs": 2, "n_prompt_tokens": 5, "n_output_tokens": 6}
    # drifted meta counts refuse to save
    bad = RolloutShardMeta(**{**meta.__dict__, "n_output_tokens": 99})
    with pytest.raises(ValueError, match="drift"):
        save_rollout_shard(
            tmp_path / "bad.safetensors", conv_ids=convs, prompt_lens=[3, 2],
            seed_hashes=[1, 2], split_ids=[0, 0], meta=bad,
        )


def test_split_of_key_deterministic_and_reasonable() -> None:
    keys = [f"seed-{i}" for i in range(20_000)]
    splits = [split_of_key(k) for k in keys]
    assert splits == [split_of_key(k) for k in keys]  # pure
    for s, name in enumerate(SPLITS):
        frac = splits.count(s) / len(splits)
        assert abs(frac - SPLIT_FRACTIONS[name]) < 0.02, (name, frac)


def _loss_inputs() -> tuple[torch.Tensor, torch.Tensor, list[Whitener]]:
    g = torch.Generator().manual_seed(0)
    preds = torch.randn(4, N_LAYERS, D, generator=g)
    targets = torch.randn(4, N_LAYERS, D, generator=g)
    whiteners = [
        Whitener(mu=torch.zeros(D), w=torch.eye(D) * (li + 1.0), ridge_c=0.1)
        for li in range(N_LAYERS)
    ]
    return preds, targets, whiteners


def test_loss_space_arms_differ_and_dispatch() -> None:
    preds, targets, whiteners = _loss_inputs()
    l_wh = multilayer_recon_loss(preds, targets, whiteners, mode="whiten")
    l_rc = multilayer_recon_loss(preds, targets, whiteners, mode="rawcos")
    l_un = multilayer_recon_loss(preds, targets, whiteners, mode="unitnorm")
    # rawcos must ignore the whitener entirely: identity whiteners give the same value
    ident = [Whitener(mu=torch.zeros(D), w=torch.eye(D), ridge_c=0.1) for _ in range(N_LAYERS)]
    assert torch.allclose(l_rc, multilayer_recon_loss(preds, targets, ident, mode="rawcos"))
    # under identity whitening, whiten == rawcos (same 2(1-cos) form)
    assert torch.allclose(multilayer_recon_loss(preds, targets, ident, mode="whiten"), l_rc)
    # unitnorm is the MSE-to-unit-target arm: penalizes the pred norm, so it must differ from
    # rawcos (which is norm-free) and move when preds are rescaled
    assert not torch.allclose(l_un, l_rc)
    assert not torch.allclose(
        multilayer_recon_loss(preds * 5, targets, whiteners, mode="unitnorm"), l_un
    )
    assert torch.allclose(
        multilayer_recon_loss(preds * 5, targets, whiteners, mode="rawcos"), l_rc
    )
    assert l_wh.shape == () and l_rc.shape == () and l_un.shape == ()
    with pytest.raises(ValueError, match="loss_space"):
        multilayer_recon_loss(preds, targets, whiteners, mode="nope")


def _crops(n_src: int = 40) -> CroppedPairs:
    lens = torch.full((n_src,), 8, dtype=torch.long)
    flat = torch.arange(int(lens.sum()), dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)])
    targets = torch.randn(n_src, N_LAYERS, D).to(torch.bfloat16)
    pairs = MultiLayerPairs(
        span_ids=flat, offsets=offsets, targets=targets,
        layers=tuple(range(N_LAYERS)),
        conv_index=torch.arange(n_src), prev_pos=torch.zeros(n_src, dtype=torch.long),
        prev_token_id=torch.zeros(n_src, dtype=torch.long),
        prev_is_assistant=torch.ones(n_src, dtype=torch.bool),
    )
    row_idx = torch.arange(n_src)
    crop_len = torch.full((n_src,), 4, dtype=torch.long)
    return CroppedPairs(pairs, row_idx, crop_len)


def test_streaming_skip_batches_is_exact_suffix() -> None:
    """skip_batches drops EXACTLY the first K micro-batches' rows and preserves the rest of the
    stream order — the resume replay contract."""
    crops = _crops()
    mb = 4
    full = [
        int(r["ids"][0])
        for r in StreamingCropDataset(crops, rank=0, world=1, buffer_rows=8, seed=3)
    ]
    for k in (1, 3):
        resumed = [
            int(r["ids"][0])
            for r in StreamingCropDataset(
                crops, rank=0, world=1, buffer_rows=8, seed=3, skip_batches=k, micro_batch=mb
            )
        ]
        assert resumed == full[k * mb :]


def test_streaming_skip_defers_but_materializes_targets() -> None:
    crops = _crops()
    rows = list(
        StreamingCropDataset(
            crops, rank=0, world=1, buffer_rows=8, seed=3, skip_batches=2, micro_batch=4
        )
    )
    assert all("target" in r and r["target"].shape == (N_LAYERS, D) for r in rows)
    assert all("_defer" not in r for r in rows)


def test_rollout_store_multi_shard_concat(tmp_path: Path) -> None:
    """Two shards concatenate with correct offset rebasing — every accessor stays row-exact."""
    metas = []
    all_convs = []
    for si, convs in enumerate([[[1, 2, 10], [3, 20, 21]], [[5, 6, 7, 30]]]):
        plens = [2, 1] if si == 0 else [3]
        n_prompt = sum(plens)
        n_out = sum(len(c) for c in convs) - n_prompt
        meta = RolloutShardMeta(
            model_id="m", mode="chat", engine="sglang", engine_version="0", tokenizer_sha="x",
            temperature=1.0, top_p=1.0, max_new=512, n_convs=len(convs),
            n_prompt_tokens=n_prompt, n_output_tokens=n_out, git_commit="t",
        )
        save_rollout_shard(
            tmp_path / f"rollouts_{si:04d}.safetensors",
            conv_ids=convs, prompt_lens=plens,
            seed_hashes=[seed_hash64(f"s{si}-{i}") for i in range(len(convs))],
            split_ids=[0] * len(convs), meta=meta,
        )
        metas.append(meta)
        all_convs.extend(zip(convs, plens, strict=True))
    r = load_rollout_shards(sorted(tmp_path.glob("rollouts_*.safetensors")))
    assert len(r) == 3
    for i, (conv, plen) in enumerate(all_convs):
        assert r.conv_ids(i).tolist() == conv
        assert r.prompt_ids(i).tolist() == conv[:plen]
        assert r.output_ids(i).tolist() == conv[plen:]
    assert r.counts() == {
        "n_convs": 3,
        "n_prompt_tokens": sum(m.n_prompt_tokens for m in metas),
        "n_output_tokens": sum(m.n_output_tokens for m in metas),
    }
    assert r.output_lengths().tolist() == [1, 2, 1]


def test_stratified_crop_pool_set_pow2() -> None:
    """The crop_pow2 law: crops only at N in {1,2,4,8,16,32}, equal counts, round-robin prefix
    rungs, and no (row, N) pair drawn twice."""
    from oracle_lens.pipeline.multilayer import stratified_crop_pool_set

    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 1025, (5000,), generator=g)
    ns = (1, 2, 4, 8, 16, 32)
    rows, lens = stratified_crop_pool_set(lengths, ns, per_n=200, seed=7)
    assert set(lens.tolist()) == set(ns)  # ONLY power-of-2 lengths
    for n in ns:
        assert int((lens == n).sum()) == 200  # equal counts per length
    # crops never exceed the source row's length
    assert bool((lengths[rows] >= lens).all())
    # no duplicate (row, N)
    keys = {(int(r), int(n)) for r, n in zip(rows.tolist(), lens.tolist(), strict=True)}
    assert len(keys) == len(rows)
    # round-robin: any prefix that is a multiple of len(ns) is exactly uniform
    for rung in (6, 60, 600):
        prefix = lens[:rung]
        for n in ns:
            assert int((prefix == n).sum()) == rung // len(ns)
    # delegation wrapper unchanged: integer range law still works
    from oracle_lens.pipeline.multilayer import stratified_crop_pool

    _r2, l2 = stratified_crop_pool(lengths, 1, 32, per_n=10, seed=3)
    assert set(l2.tolist()) == set(range(1, 33))


def test_carve_uniform_spans_disjoint_and_uniform() -> None:
    """The iolens AR carve: disjoint spans, N in 1..32, near-uniform lengths, high coverage,
    no span a prefix of another (distinct starts guaranteed by disjointness)."""
    import random as _random

    from oracle_lens.pipeline.spans import carve_uniform_spans

    rng = _random.Random(7)
    plen, total = 40, 40 + 1200
    mask = [i >= plen for i in range(total)]
    spans = carve_uniform_spans(mask, rng=rng, n_lo=1, n_hi=32)
    assert len(spans) > 25
    # disjoint + inside the region + prev_pos = start-1
    taken: set[int] = set()
    for s in spans:
        assert s.start >= plen and s.start + s.n_tokens <= total
        assert s.prev_pos == s.start - 1
        assert 1 <= s.n_tokens <= 32
        span_positions = set(range(s.start, s.start + s.n_tokens))
        assert not (span_positions & taken)
        taken |= span_positions
    # coverage: gaps are small (gap_max=8), so most of the region is used
    assert len(taken) / (total - plen) > 0.6
    # near-uniform N over many conversations
    counts = [0] * 33
    for seed in range(200):
        for s in carve_uniform_spans(mask, rng=_random.Random(seed)):
            counts[s.n_tokens] += 1
    inner = counts[1:29]  # tail clamp only affects the largest lengths' exactness
    assert max(inner) < 2.0 * min(inner)
    # deterministic under the same seed
    a = carve_uniform_spans(mask, rng=_random.Random(3))
    b = carve_uniform_spans(mask, rng=_random.Random(3))
    assert [(s.start, s.n_tokens) for s in a] == [(x.start, x.n_tokens) for x in b]


def _write_pair_shard(
    path: Path, n_rows: int, n_layers: int = 3, d: int = 4, id_offset: int = 0
) -> None:
    """Minimal multilayer_v1 shard for streaming tests (row i's id encodes offset+i, so rows
    are globally distinguishable across shards)."""
    from oracle_lens.pipeline.multilayer import (
        MultiLayerPairs,
        MultiLayerShardMeta,
        save_multilayer_shard,
    )

    ids = [[1000 + id_offset + i] for i in range(n_rows)]
    pairs = MultiLayerPairs(
        span_ids=torch.tensor([t for row in ids for t in row], dtype=torch.int32),
        offsets=torch.arange(n_rows + 1, dtype=torch.int64),
        targets=torch.arange(n_rows * n_layers * d, dtype=torch.float32).reshape(
            n_rows, n_layers, d
        ).to(torch.bfloat16),
        layers=tuple(range(n_layers)),
        conv_index=torch.zeros(n_rows, dtype=torch.long),
        prev_pos=torch.zeros(n_rows, dtype=torch.long),
        prev_token_id=torch.zeros(n_rows, dtype=torch.long),
        prev_is_assistant=torch.ones(n_rows, dtype=torch.bool),
    )
    meta = MultiLayerShardMeta(
        model_id="m", dataset_id="d", layers=tuple(range(n_layers)), t_max=8, n_max=32,
        n_min=1, length_law="uniform32_disjoint", region_start=0, region_end=0,
        split="train", seed=0, git_commit="t", n_pairs=n_rows,
    )
    save_multilayer_shard(path, pairs, meta)


def test_streaming_pair_dataset_partitions_and_marks(tmp_path: Path) -> None:
    """Ranks read disjoint ROWS of every shard, each row exactly once, and the janitor frees a
    shard only once every rank has marked it."""
    from oracle_lens.pipeline.stream_pairs import (
        StreamingPairDataset,
        buffer_bytes,
        shard_marker,
        sweep_consumed,
    )

    world = 2
    for s in range(4):
        _write_pair_shard(
            tmp_path / f"pairs_train_{s:04d}.safetensors", n_rows=25, id_offset=100 * s
        )
    per_rank: list[list[int]] = []
    for rank in range(world):
        ds = StreamingPairDataset(
            tmp_path, rank=rank, world=world, shuffle_rows=4, max_wait_s=0.0, wait_s=0.0
        )
        per_rank.append([int(r["ids"][0]) for r in ds])
    seen = [x for rows in per_rank for x in rows]
    assert len(set(seen)) == len(seen)  # no row twice
    assert len(seen) == 4 * (25 // world) * world  # stride-truncated, whole shards
    # Rank balance is the property that keeps DDP alive: an imbalance means the short rank blocks
    # in the producer wait while its peers step on, and the next ALLREDUCE times out.
    assert len(per_rank[0]) == len(per_rank[1]), f"unbalanced: {[len(r) for r in per_rank]}"
    # every rank reads every shard now, so a shard is spent only when ALL of them have marked it
    for s in range(4):
        shard = tmp_path / f"pairs_train_{s:04d}.safetensors"
        assert all(shard_marker(shard, r).exists() for r in range(world))
    assert sweep_consumed(tmp_path, world) > 0
    assert buffer_bytes(tmp_path) == 0


def test_streaming_partial_rank_completion_is_not_sweepable(tmp_path: Path) -> None:
    """A shard only one rank has finished must survive — the other rank's rows are still owed."""
    from oracle_lens.pipeline.stream_pairs import (
        StreamingPairDataset,
        buffer_bytes,
        sweep_consumed,
    )

    world = 2
    for s in range(2):
        _write_pair_shard(
            tmp_path / f"pairs_train_{s:04d}.safetensors", n_rows=20, id_offset=100 * s
        )
    ds = StreamingPairDataset(
        tmp_path, rank=0, world=world, shuffle_rows=4, max_wait_s=0.0, wait_s=0.0
    )
    rank0 = [int(r["ids"][0]) for r in ds]
    assert rank0, "rank 0 read nothing"
    assert sweep_consumed(tmp_path, world) == 0, "swept a shard rank 1 has not read"
    assert buffer_bytes(tmp_path) > 0


def test_streaming_non_zero_worker_does_not_reread_shards(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A worker with id != 0 must read each shard exactly once.

    The consumed marker is written by worker 0 only (one per rank) but every worker filters on
    it, so before the per-worker `seen` set a worker that finished its snapshot first re-listed
    the same shards and re-read its whole stride — byte-identical rows the trainer counted as
    fresh samples. Measured on the live runs as ~497k repeated rows in the pt cell (21% of its
    recorded examples), with no duplicate capture involved.

    The pre-fix code never terminates here (nothing ever marks, so the shard list never empties),
    which is why max_rows caps the read: the assertion is that we stop at the real row count with
    no repeats, not at the cap with them.
    """
    import torch.utils.data as _tud

    from oracle_lens.pipeline.stream_pairs import StreamingPairDataset

    n_shards, rows_per = 4, 25
    for s in range(n_shards):
        _write_pair_shard(
            tmp_path / f"pairs_train_{s:04d}.safetensors", n_rows=rows_per, id_offset=100 * s
        )

    class _Info:
        id, num_workers = 1, 2

    monkeypatch.setattr(_tud, "get_worker_info", lambda: _Info())
    ds = StreamingPairDataset(
        tmp_path, rank=0, world=1, shuffle_rows=4, max_wait_s=0.0, wait_s=0.0, max_rows=200
    )
    got = [int(r["ids"][0]) for r in ds]
    expected = n_shards * len(range(1, rows_per, 2))  # this worker's stride only
    assert len(got) == expected, f"expected {expected} rows, got {len(got)} (re-read shards?)"
    assert len(set(got)) == len(got), "worker yielded the same row twice"


def test_streaming_resume_skips_consumed_and_layer_subset(tmp_path: Path) -> None:
    """A restarted rank skips shards it already marked; layer_indices slices the target stack."""
    from oracle_lens.pipeline.stream_pairs import StreamingPairDataset, shard_marker

    for s in range(2):
        _write_pair_shard(tmp_path / f"pairs_train_{s:04d}.safetensors", n_rows=10, n_layers=3)
    kw = {"rank": 0, "world": 1, "shuffle_rows": 2, "max_wait_s": 0.0, "wait_s": 0.0}
    first = list(StreamingPairDataset(tmp_path, **kw))  # type: ignore[arg-type]
    assert len(first) == 20
    assert first[0]["target"].shape == (3, 4)
    assert all(shard_marker(tmp_path / f"pairs_train_{s:04d}.safetensors", 0).exists()
               for s in range(2))
    again = list(StreamingPairDataset(tmp_path, **kw))  # type: ignore[arg-type]
    assert again == []  # resume: nothing re-read
    _write_pair_shard(tmp_path / "pairs_train_0002.safetensors", n_rows=10, n_layers=3)
    sub = list(StreamingPairDataset(tmp_path, layer_indices=(1, 2), **kw))  # type: ignore[arg-type]
    assert len(sub) == 10 and sub[0]["target"].shape == (2, 4)  # layer 0 dropped


def test_streaming_dataset_reports_length(tmp_path: Path) -> None:
    """The vendored harness does `global_step % len(dataloader)`, so the stream MUST have a
    length; it reflects this rank's unconsumed rows and shrinks as shards are consumed."""
    from oracle_lens.pipeline.stream_pairs import StreamingPairDataset, shard_marker

    world = 2
    shards = []
    for i in range(4):
        p = tmp_path / f"pairs_train_{i:04d}.safetensors"
        _write_pair_shard(p, n_rows=25, id_offset=100 * i)
        shards.append(p)
    # every rank sees all 4 shards but reads its 1/world stride, so lengths are EQUAL — that
    # equality is the same property that keeps the ranks from starving each other under DDP
    lens = [len(StreamingPairDataset(tmp_path, rank=r, world=world, min_len=1))
            for r in range(world)]
    assert lens == [4 * 25 // world] * world
    # a shard this rank has finished drops out of its own length
    shard_marker(shards[0], 0).touch()
    assert len(StreamingPairDataset(tmp_path, rank=0, world=world, min_len=1)) == 3 * 25 // world
    assert len(StreamingPairDataset(tmp_path, rank=1, world=world, min_len=1)) == 4 * 25 // world
    # an exhausted stream returns the floor, so len(dataloader) can never round to 0 batches
    for sh in shards:
        for r in range(world):
            shard_marker(sh, r).touch()
    assert len(StreamingPairDataset(tmp_path, rank=0, world=world, min_len=100_000)) == 100_000
