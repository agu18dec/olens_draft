"""Unit tests for the rollout -> span-pool assembly core (long oracle lens)."""

import hashlib
import itertools
import json
from pathlib import Path

import torch

from oracle_lens.pipeline.assemble import build_span_pool
from oracle_lens.pipeline.dedup import seed_keys_from_rollouts

PROMPT_TOK = 1  # sentinel: prompt-region token id
GEN_BASE = 100  # generated-region ids start here


def _rollout(seed: str, n_prompt: int, n_gen: int) -> dict[str, object]:
    return {
        "seed": seed,
        "prompt_ids": [PROMPT_TOK] * n_prompt,
        "output_ids": [GEN_BASE + i for i in range(n_gen)],
    }


def _shards() -> list[list[dict[str, object]]]:
    return [
        [_rollout("alpha", 5, 60), _rollout("beta", 3, 40)],
        [_rollout("alpha", 5, 60), _rollout("gamma", 4, 700), _rollout("tiny", 2, 0)],
    ]


def test_conv_index_matches_seed_keys_contract(tmp_path: Path) -> None:
    """conv_index enumerates concatenated rollout rows exactly as seed_keys_from_rollouts does."""
    shards = _shards()
    for i, shard in enumerate(shards):
        (tmp_path / f"rollouts_{i:04d}.json").write_text(json.dumps(shard))
    keys = seed_keys_from_rollouts(sorted(tmp_path.glob("rollouts_*.json")))
    pool, _, _ = build_span_pool(shards, seed=0, base_per_conv=4, long_per_conv=0)
    seeds_by_conv = {0: "alpha", 1: "beta", 2: "alpha", 3: "gamma", 4: "tiny"}
    for c in pool.conv_index.tolist():
        expect = hashlib.blake2b(seeds_by_conv[c].encode(), digest_size=16).digest()
        assert keys[c] == expect


def test_duplicate_seed_conv_never_samples() -> None:
    pool, _, stats = build_span_pool(_shards(), seed=0, base_per_conv=8, long_per_conv=2)
    assert stats.n_convs == 5
    assert stats.n_dup_convs == 1
    assert 2 not in set(pool.conv_index.tolist())  # conv 2 duplicates conv 0's seed


def test_spans_live_in_generated_region_only() -> None:
    pool, _, _ = build_span_pool(_shards(), seed=1, base_per_conv=8, long_per_conv=2)
    assert len(pool) > 0
    assert not (pool.span_ids == PROMPT_TOK).any()  # no prompt token inside any span
    # prev_pos may sit on the last prompt token but never earlier
    for i in range(len(pool)):
        conv = int(pool.conv_index[i])
        n_prompt = {0: 5, 1: 3, 3: 4, 4: 2}[conv]
        assert int(pool.prev_pos[i]) >= n_prompt - 1


def test_no_duplicate_spans_across_passes() -> None:
    # long_n_min=1 forces both passes into the same (start, n) space; the union must dedup
    pool, _, _ = build_span_pool(
        _shards(), seed=2, base_per_conv=16, long_per_conv=16, long_n_min=1
    )
    triples = list(
        zip(pool.conv_index.tolist(), pool.prev_pos.tolist(), pool.lengths.tolist(), strict=True)
    )
    assert len(triples) == len(set(triples))


def test_long_pass_only_on_qualifying_convs() -> None:
    pool, _, _ = build_span_pool(_shards(), seed=0, base_per_conv=0, long_per_conv=4)
    # only gamma (700 gen tokens) can host n >= 513 spans
    assert set(pool.conv_index.tolist()) <= {3}
    assert (pool.lengths >= 513).all()


def test_mid_pass_fills_mid_octaves_only() -> None:
    pool, _, _ = build_span_pool(
        _shards(), seed=0, base_per_conv=0, long_per_conv=0, mid_per_conv=4
    )
    assert len(pool) > 0
    assert (pool.lengths >= 257).all() and (pool.lengths <= 512).all()
    # off by default: existing two-pass behavior unchanged
    pool_off, _, _ = build_span_pool(_shards(), seed=0, base_per_conv=0, long_per_conv=0)
    assert len(pool_off) == 0


def test_split_is_conv_level_and_deterministic() -> None:
    pool_a, eval_a, _ = build_span_pool(_shards(), seed=0, eval_frac=0.5)
    pool_b, eval_b, _ = build_span_pool(_shards(), seed=0, eval_frac=0.5)
    assert torch.equal(pool_a.offsets, pool_b.offsets)
    assert torch.equal(eval_a, eval_b)
    verdict: dict[int, bool] = {}
    for i in range(len(pool_a)):
        c = int(pool_a.conv_index[i])
        v = bool(eval_a[i])
        assert verdict.setdefault(c, v) == v  # one verdict per conversation


def test_disjoint_sampler_no_overlap_within_conv() -> None:
    """sampler="disjoint": no two rows of a conv share a token position (cross-octave)."""
    pool, _, _ = build_span_pool(_shards(), seed=0, sampler="disjoint")
    by_conv: dict[int, list[tuple[int, int]]] = {}
    lens = pool.lengths
    for i in range(len(pool)):
        c = int(pool.conv_index[i])
        start = int(pool.prev_pos[i]) + 1
        by_conv.setdefault(c, []).append((start, start + int(lens[i])))
    for c, ivs in by_conv.items():
        ivs.sort()
        for (_, e0), (s1, _) in itertools.pairwise(ivs):
            assert s1 >= e0, f"conv {c}: overlapping spans"


def test_disjoint_sampler_spans_in_generated_region() -> None:
    pool, _, _ = build_span_pool(_shards(), seed=0, sampler="disjoint")
    for i in range(len(pool)):
        ids = pool.row_ids(i).tolist()
        assert all(t >= GEN_BASE for t in ids), "span leaked into the prompt region"


def test_disjoint_sampler_deterministic_and_distinct_from_passes() -> None:
    a, ea, _ = build_span_pool(_shards(), seed=0, sampler="disjoint")
    b, eb, _ = build_span_pool(_shards(), seed=0, sampler="disjoint")
    assert torch.equal(a.span_ids, b.span_ids) and torch.equal(a.offsets, b.offsets)
    assert torch.equal(ea, eb)
    c, _, _ = build_span_pool(_shards(), seed=0)
    assert not (
        len(a) == len(c)
        and torch.equal(a.offsets, c.offsets)
        and torch.equal(a.span_ids, c.span_ids)
    )


def test_disjoint_sampler_reserves_long_span_on_qualifying_conv() -> None:
    """The 700-token conv ("gamma") must contribute a >=513 span; split stays conv-level."""
    pool, eval_mask, _ = build_span_pool(_shards(), seed=0, sampler="disjoint")
    lens = pool.lengths
    gamma_rows = [i for i in range(len(pool)) if int(pool.conv_index[i]) == 3]
    assert any(int(lens[i]) >= 513 for i in gamma_rows)
    for c in set(pool.conv_index.tolist()):
        rows = [i for i in range(len(pool)) if int(pool.conv_index[i]) == c]
        verdicts = {bool(eval_mask[i]) for i in rows}
        assert len(verdicts) == 1, f"conv {c} straddles train/eval"


def test_disjoint_sampler_rejects_unknown_sampler() -> None:
    try:
        build_span_pool(_shards(), seed=0, sampler="nope")
    except ValueError as e:
        assert "sampler" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_split_key_overrides_own_seed_hash_for_split_verdict() -> None:
    # two distinct seeds carrying the SAME split_key (later turns of one conversation) must
    # get identical split verdicts, and that verdict must equal the split_key owner's own
    row_a = _rollout("turn one text", 5, 60)
    row_b = _rollout("a later user turn", 5, 60)
    row_b["split_key"] = hashlib.blake2b(b"turn one text", digest_size=16).hexdigest()
    # sweep eval_frac so both sides of the verdict are exercised
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        pool, eval_mask, _ = build_span_pool(
            [[row_a, row_b]], seed=0, base_per_conv=4, long_per_conv=0, eval_frac=frac
        )
        verdicts = {int(c): bool(eval_mask[i]) for i, c in enumerate(pool.conv_index.tolist())}
        assert len(verdicts) == 2  # both convs sampled spans
        assert verdicts[0] == verdicts[1], f"split diverged at eval_frac={frac}"


def test_absent_split_key_keeps_historical_verdict() -> None:
    # rows without split_key (all v1 rollouts) must split exactly as before the field existed
    pool, eval_mask, _ = build_span_pool(_shards(), seed=0, eval_frac=0.5)
    with_key = [dict(r) for shard in _shards() for r in shard]
    for r in with_key:
        r["split_key"] = hashlib.blake2b(str(r["seed"]).encode(), digest_size=16).hexdigest()
    pool_k, eval_k, _ = build_span_pool([with_key], seed=0, eval_frac=0.5)
    assert eval_mask.tolist() == eval_k.tolist()
    assert pool.conv_index.tolist() == pool_k.conv_index.tolist()
