"""CPU tests for token-budget bucketing: the packing must bound padded tokens per batch, cover every
row once, and (under DDP) deal equal, disjoint batch counts to every rank so step counts never
diverge (a DDP deadlock)."""

from oracle_lens.pipeline.multilayer import (
    TokenBudgetSampler,
    pack_token_budget,
    token_budget_nbatches,
)


def _lengths() -> list[int]:
    # a realistic long-tailed span mix: many short, a few up to 1024
    return [1] * 50 + [8] * 40 + [64] * 20 + [256] * 8 + [1024] * 4


def test_pack_respects_budget() -> None:
    lengths = _lengths()
    budget, max_rows = 4096, 256
    batches = pack_token_budget(lengths, budget, max_rows, seed=0)
    for b in batches:
        maxlen = max(lengths[i] for i in b)
        assert len(b) * maxlen <= budget  # padded tokens bounded (max span 1024 < budget => always)
        assert len(b) <= max_rows


def test_pack_covers_every_row_once() -> None:
    lengths = _lengths()
    batches = pack_token_budget(lengths, 4096, 256, seed=3)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))  # partition: every index exactly once


def test_max_rows_caps_short_batches() -> None:
    lengths = [1] * 1000  # would be one 1000-row batch without the cap
    batches = pack_token_budget(lengths, 4096, max_rows=64, seed=0)
    assert all(len(b) <= 64 for b in batches)
    assert sum(len(b) for b in batches) == 1000


def test_sampler_ddp_equal_disjoint() -> None:
    lengths = _lengths()
    world, budget = 4, 4096
    all_batches = pack_token_budget(lengths, budget, 256, seed=0)
    expected = len(all_batches) // world
    seen: list[tuple[int, ...]] = []
    for rank in range(world):
        s = TokenBudgetSampler(lengths, budget, world=world, rank=rank, seed=0, max_rows=256)
        assert len(s) == expected  # every rank runs the SAME number of steps
        batches = list(s)
        assert len(batches) == expected
        seen.extend(tuple(sorted(b)) for b in batches)
    # ranks get disjoint batches (no batch dealt to two ranks)
    assert len(seen) == len(set(seen)) == expected * world


def test_nbatches_matches_sampler_len() -> None:
    lengths = _lengths()
    for world in (1, 2, 8):
        nb = token_budget_nbatches(lengths, 4096, world, 256, seed=0)
        s = TokenBudgetSampler(lengths, 4096, world=world, rank=0, seed=0, max_rows=256)
        assert nb == len(s)
