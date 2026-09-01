"""AO-ladder pool construction: sampling, nested crops, dedup, AR-prefix exclusion."""

import itertools
from pathlib import Path

import pytest
import torch

from oracle_lens.pipeline.ao_pool import (
    CROP_LENGTHS,
    WINDOW,
    AOPool,
    build_ao_pool,
    pairs_prefix_hashes,
    sample_starts,
    span_hash,
)


def _outputs(n_convs: int = 8, length: int = 200, base: int = 100) -> list[list[int]]:
    # distinct deterministic token streams, no special ids
    return [[base + (c * 7 + t) % 90 for t in range(length)] for c in range(n_convs)]


def test_sample_starts_is_deterministic_and_leaves_window_room() -> None:
    outs = _outputs()
    c1, s1 = sample_starts(outs, n_starts=20, seed=3, max_per_conv=4)
    c2, s2 = sample_starts(outs, n_starts=20, seed=3, max_per_conv=4)
    assert torch.equal(c1, c2) and torch.equal(s1, s2)
    for c, s in zip(c1.tolist(), s1.tolist(), strict=True):
        assert s + WINDOW <= len(outs[c])
    # per-conv cap holds
    for c in set(c1.tolist()):
        assert c1.tolist().count(c) <= 4
    c3, _ = sample_starts(outs, n_starts=20, seed=4, max_per_conv=4)
    assert not torch.equal(c1, c3)  # seed matters


def test_build_pool_dedups_within_length_and_excludes_prefixes() -> None:
    # two windows that share their first 4 tokens but differ later: window dedup keeps both,
    # span dedup must still drop the second's N=2 and N=4 crops as exact-content duplicates
    a = [10 + t % 50 for t in range(100)]
    b = a[:4] + [700 + t for t in range(96)]
    outs = [a, b]
    conv = torch.tensor([0, 1])
    start = torch.tensor([0, 0])
    pool = build_ao_pool(
        outs, conv, start, exclude={}, special_ids=frozenset(), seed=0, all_lengths=True
    )
    assert pool.keep.shape[1] == len(CROP_LENGTHS)
    assert len(pool.ids) == 2  # distinct windows, both kept
    assert pool.keep[0].all()  # first occurrence kept at every length
    stats = pool.meta["stats"]
    assert stats["n_dup_N2"] == 1 and stats["n_kept_N2"] == 1  # second window's 2-prefix is a dup
    assert stats["n_dup_N4"] == 1 and stats["n_kept_N8"] == 2  # diverges by N=8

    # exclusion: pretend the AR trained on this window's 4-prefix
    excl = {4: {span_hash(outs[0][:4])}}
    pool2 = build_ao_pool(
        outs, conv[:1], start[:1], exclude=excl, special_ids=frozenset(), all_lengths=True
    )
    j4 = CROP_LENGTHS.index(4)
    assert not pool2.keep[0, j4]  # excluded at N=4
    assert pool2.keep[0, j4 + 1]  # longer crops unaffected
    assert pool2.meta["stats"]["n_excluded_N4"] == 1


def test_build_pool_drops_starts_with_special_tokens() -> None:
    outs = _outputs(2)
    outs[1][10] = 999  # special token inside the window
    conv = torch.tensor([0, 1])
    start = torch.tensor([0, 0])
    pool = build_ao_pool(outs, conv, start, exclude={}, special_ids=frozenset({999}))
    assert len(pool.ids) == 1  # start 1 dropped entirely
    assert pool.meta["stats"]["n_starts_special"] == 1


def test_pairs_prefix_hashes_cover_exactly_row_prefixes() -> None:
    # rows: [1,2,3,4,5] and [7,8] -> prefixes at N=2: (1,2), (7,8); at N=4: (1,2,3,4)
    span_ids = torch.tensor([1, 2, 3, 4, 5, 7, 8], dtype=torch.int32)
    offsets = torch.tensor([0, 5, 7], dtype=torch.long)
    h = pairs_prefix_hashes(span_ids, offsets, lengths=(2, 4))
    assert span_hash([1, 2]) in h[2] and span_hash([7, 8]) in h[2]
    assert span_hash([1, 2, 3, 4]) in h[4]
    assert span_hash([2, 3]) not in h[2]  # mid-row substrings are NOT AR inputs
    assert len(h[4]) == 1  # short row has no 4-prefix


def test_pool_save_load_roundtrip_and_crop_index(tmp_path: Path) -> None:
    outs = _outputs(4)
    conv, start = sample_starts(outs, n_starts=6, seed=0, max_per_conv=2)
    pool = build_ao_pool(
        outs, conv, start, exclude={}, special_ids=frozenset(), seed=0, all_lengths=True
    )
    p = tmp_path / "pool.safetensors"
    pool.save(p)
    back = AOPool.load(p)
    assert torch.equal(back.ids, pool.ids) and torch.equal(back.keep, pool.keep)
    assert back.meta["stats"] == pool.meta["stats"]
    rows, lens = back.crop_index()
    assert len(rows) == back.n_crops()
    # canonical order: i-major, then length
    flat = rows * len(CROP_LENGTHS) + lens
    assert torch.equal(flat, flat.sort().values)
    # crop_ids returns the right prefix
    n0 = CROP_LENGTHS[int(lens[0])]
    assert torch.equal(back.crop_ids(int(rows[0]), int(lens[0])), back.ids[int(rows[0]), :n0])


def test_non_overlapping_starts_never_share_tokens() -> None:
    """Overlapping windows silently duplicate text — one of the three repetition paths that
    produced the 2026-07-29 single-epoch overfit."""
    outs = [[100 + t % 70 for t in range(600)] for _ in range(6)]
    conv, start = sample_starts(outs, n_starts=200, seed=1, max_per_conv=8)
    per_conv: dict[int, list[int]] = {}
    for c, s in zip(conv.tolist(), start.tolist(), strict=True):
        per_conv.setdefault(c, []).append(s)
    for c, starts in per_conv.items():
        starts.sort()
        for a, b in itertools.pairwise(starts):
            assert b - a >= WINDOW, f"conv {c}: windows at {a},{b} overlap"
        assert all(s % WINDOW == 0 for s in starts)  # block-aligned tiling
    # the unsafe mode is still reachable, and does overlap
    _c2, s2 = sample_starts(outs, n_starts=200, seed=1, max_per_conv=8, non_overlapping=False)
    assert any(v % WINDOW != 0 for v in s2.tolist())


def test_one_length_per_window_and_diversity_audit() -> None:
    from oracle_lens.pipeline.ao_pool import audit_diversity

    outs = [[100 + (c * 13 + t) % 90 for t in range(600)] for c in range(20)]
    conv, start = sample_starts(outs, n_starts=60, seed=0, max_per_conv=4)
    pool = build_ao_pool(outs, conv, start, exclude={}, special_ids=frozenset(), seed=0)
    # exactly one crop per surviving window
    assert int(pool.keep.sum(dim=1).max()) == 1
    rep = audit_diversity(pool, layers_per_crop=2)
    assert rep["crops_per_window"] == 1.0
    assert rep["repeat_factor"] == 2.0 and rep["overlapping_window_pairs"] == 0
    # k=4 exceeds the ceiling -> audit must refuse
    with pytest.raises(ValueError, match="diversity audit FAILED"):
        audit_diversity(pool, layers_per_crop=4)
    # the OLD construction (all 6 nested crops) is exactly what the audit is there to catch
    nested = build_ao_pool(
        outs, conv, start, exclude={}, special_ids=frozenset(), seed=0, all_lengths=True
    )
    assert int(nested.keep.sum(dim=1).max()) > 1
    with pytest.raises(ValueError, match="target"):
        audit_diversity(nested, layers_per_crop=4)


def test_window_level_dedup_drops_identical_texts() -> None:
    """Two conversations emitting identical text must yield ONE window: span-level dedup misses
    this when the duplicates draw different lengths (one span is then a prefix of the other)."""
    body = [200 + t % 40 for t in range(300)]
    outs = [list(body), list(body), [900 + t % 50 for t in range(300)]]
    conv = torch.tensor([0, 1, 2])
    start = torch.zeros(3, dtype=torch.long)
    pool = build_ao_pool(outs, conv, start, exclude={}, special_ids=frozenset(), seed=0)
    assert len(pool.ids) == 2  # the duplicate conversation's window is dropped
    assert pool.meta["stats"]["n_starts_dup_window"] == 1
    seen = {span_hash(pool.ids[i].tolist()) for i in range(len(pool.ids))}
    assert len(seen) == len(pool.ids)  # every kept window is a distinct text


def test_custom_lengths_are_self_describing_and_roundtrip(tmp_path: Path) -> None:
    """Long-emission pools (lengths beyond 64) carry their own length list end to end."""
    lengths = (8, 16, 32, 64, 96, 128, 192, 256)
    outs = _outputs(n_convs=6, length=600)
    conv, start = sample_starts(outs, n_starts=12, seed=1, max_per_conv=2, window=max(lengths))
    pool = build_ao_pool(
        outs, conv, start, exclude={}, special_ids=frozenset(), seed=0, lengths=lengths
    )
    assert pool.lengths == lengths
    assert pool.window == 256 and pool.ids.shape[1] == 256
    assert pool.keep.shape[1] == len(lengths)
    # crop_ids slices by the POOL's lengths, not the legacy constant
    rows, lens = pool.crop_index()
    for c in range(len(rows)):
        assert len(pool.crop_ids(int(rows[c]), int(lens[c]))) == lengths[int(lens[c])]
    p = tmp_path / "long.safetensors"
    pool.save(p)
    loaded = AOPool.load(p)
    assert loaded.lengths == lengths and loaded.window == 256

    # a mislabeled artifact (lengths list != keep columns) must refuse, not mislabel crops
    loaded.meta["crop_lengths"] = [2, 4, 8]
    with pytest.raises(ValueError, match="crop lengths"):
        _ = loaded.lengths


def test_legacy_pool_without_lengths_meta_defaults_to_crop_lengths() -> None:
    outs = _outputs()
    conv, start = sample_starts(outs, n_starts=6, seed=2, max_per_conv=2)
    pool = build_ao_pool(outs, conv, start, exclude={}, special_ids=frozenset(), seed=0)
    pool.meta.pop("crop_lengths", None)  # pre-2026-08 artifacts lack the key
    assert pool.lengths == CROP_LENGTHS
    assert pool.window == WINDOW
