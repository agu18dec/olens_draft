"""The prompt-tag layer schedule must cover every layer evenly within ONE optimizer step.

Why this matters for fairness: the layer is a prompt, so a micro-batch can supervise only one
layer. The harness divides the loss by ``gradient_accumulation_steps`` and DDP averages over ranks,
so if the ``ga * world`` micro-batches of one optimizer step hit each layer equally, the
optimizer-step gradient is *mean over layers of mean over rows* — the same estimator form as the
layer-conditioned arch. If coverage were uneven, the two arms would be optimizing different
objectives and the comparison would be confounded.
"""

from collections import Counter

from oracle_lens.pipeline.multilayer_reconstructor import MLReconConfig, MLReconMydule


class _Cfg:
    def __init__(self, ga: int, world: int) -> None:
        self.gradient_accumulation_steps = ga
        self._world = world

    def world_size(self) -> int:
        return self._world


class _Trainer:
    def __init__(self, ga: int, world: int, rank: int, step: int) -> None:
        self.config = _Cfg(ga, world)
        self.global_rank = rank
        self.global_step = step


def _sched(n_layers: int, ga: int, world: int, rank: int, step: int) -> int:
    m = MLReconMydule.__new__(MLReconMydule)
    m._cfg = MLReconConfig(run_name="t", layer_indices=tuple(range(n_layers)))
    m.trainer = _Trainer(ga, world, rank, step)
    return m._schedule_layer_idx()


def test_one_optimizer_step_covers_all_12_layers_evenly() -> None:
    """ga_eff=3, world=8 -> 24 micro-batches per step over 12 layers = exactly 2 each."""
    n, ga, world = 12, 3, 8
    seen = Counter(
        _sched(n, ga, world, rank, opt_step * ga + micro)
        for opt_step in [0]
        for micro in range(ga)
        for rank in range(world)
    )
    assert sorted(seen) == list(range(n))
    assert set(seen.values()) == {2}


def test_coverage_stays_even_across_many_optimizer_steps() -> None:
    n, ga, world = 12, 3, 8
    for opt_step in range(7):
        seen = Counter(
            _sched(n, ga, world, rank, opt_step * ga + micro)
            for micro in range(ga)
            for rank in range(world)
        )
        assert set(seen.values()) == {2}, f"opt_step {opt_step}: {seen}"


def test_rank_layer_pairing_rotates_between_steps() -> None:
    """No rank may be pinned to one layer, or a rank's LoRA shard would see a biased layer mix."""
    n, ga, world = 12, 3, 8
    first = [_sched(n, ga, world, r, 0) for r in range(world)]
    later = [_sched(n, ga, world, r, ga) for r in range(world)]
    assert first != later


def test_is_pure_function_of_step_rank_so_resume_is_stable() -> None:
    a = _sched(12, 3, 8, rank=5, step=41)
    b = _sched(12, 3, 8, rank=5, step=41)
    assert a == b


def test_no_trainer_is_safe() -> None:
    m = MLReconMydule.__new__(MLReconMydule)
    m._cfg = MLReconConfig(run_name="t", layer_indices=tuple(range(12)))
    m.trainer = None
    assert m._schedule_layer_idx() == 0
