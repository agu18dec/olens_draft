"""Regression tests for the AR restart path.

On 2026-08-02 a crash-restart of the assistant cell hit three separate defects in a row, each of
which had been latent because every previous restart used ``--init-from`` (warm start) rather
than ``--resume`` (exact continuation). The worst of them silently restarted training FROM
SCRATCH and would have overwritten ``resume.pt`` with the untrained state at its next save.

A resume path that is never exercised is not a resume path — so these tests exercise it. All are
CPU-only: they cover the decision logic and the state plumbing, not the GPU training step.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from oracle_lens.pipeline.resume import RestoreTrainerState, save_resume_state
from oracle_lens.pipeline.stream_pairs import StreamingPairDataset

# scripts/ are not a package; load the entry point by path like the other script tests do.
_SPEC = importlib.util.spec_from_file_location(
    "iolens_ar_train", Path(__file__).resolve().parents[1] / "scripts/ar/iolens_ar_train.py"
)
assert _SPEC and _SPEC.loader
_AR = importlib.util.module_from_spec(_SPEC)
sys.modules["iolens_ar_train"] = _AR
_SPEC.loader.exec_module(_AR)


def _args(**kw: Any) -> SimpleNamespace:
    base = {"resume": False, "examples_prev": 0, "tokens_prev": 0, "allow_fresh": False}
    base.update(kw)
    return SimpleNamespace(**base)


class _Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


def _write_resume(ckpt_dir: Path, **kw: Any) -> None:
    mod = _Mod()
    opt = torch.optim.SGD(mod.parameters(), lr=0.1)
    save_resume_state(
        ckpt_dir, recon=mod, peft_inner=mod, optimizer=opt,
        global_step=kw.pop("global_step", 18005), tokens_span=kw.pop("tokens_span", 40_238_609),
        examples=kw.pop("examples", 2_477_760), **kw,
    )


def test_fresh_run_returns_empty(tmp_path: Path) -> None:
    assert _AR.resume_meta_json(_args(), tmp_path, "run", 3) == ""


def test_warm_start_carries_the_curve_axes(tmp_path: Path) -> None:
    out = _AR.resume_meta_json(_args(examples_prev=749_280, tokens_prev=12_158_173), tmp_path,
                               "run", 3)
    assert json.loads(out) == {"examples_prev": 749_280, "tokens_prev": 12_158_173}


def test_resume_without_a_bundle_refuses(tmp_path: Path) -> None:
    """--resume with no bundle must NOT silently train from scratch.

    The supervisor relaunches with --resume; if the bundle were missing, a fresh start would
    overwrite a trained run's checkpoints with chance-level weights, unattended, while every log
    line looked healthy.
    """
    with pytest.raises(SystemExit, match="refusing to silently train from scratch"):
        _AR.resume_meta_json(_args(resume=True), tmp_path, "run", 3)


def test_resume_without_a_bundle_allows_fresh_when_asked(tmp_path: Path, capsys: Any) -> None:
    assert _AR.resume_meta_json(_args(resume=True, allow_fresh=True), tmp_path, "run", 3) == ""
    assert "fresh start" in capsys.readouterr().out


def test_resume_with_a_bundle_is_not_a_no_op(tmp_path: Path, capsys: Any) -> None:
    """THE regression: under --stream-dir this returned "" and the run restarted from scratch."""
    ckpt = tmp_path / "ml_checkpoints" / "run"
    _write_resume(ckpt, world_size=3, num_workers=_AR.STREAM_WORKERS)
    out = _AR.resume_meta_json(_args(resume=True), tmp_path, "run", 3)
    assert json.loads(out) == {"resume": True}
    # a silent no-op must be impossible to miss in the log
    assert "18005" in capsys.readouterr().out


def test_resume_refuses_a_changed_world_size(tmp_path: Path) -> None:
    ckpt = tmp_path / "ml_checkpoints" / "run"
    _write_resume(ckpt, world_size=3, num_workers=_AR.STREAM_WORKERS)
    with pytest.raises(SystemExit, match="world_size"):
        _AR.resume_meta_json(_args(resume=True), tmp_path, "run", 2)


def test_warm_start_wins_over_resume(tmp_path: Path) -> None:
    """--examples-prev is the explicit request; it must not be silently overridden by a bundle."""
    ckpt = tmp_path / "ml_checkpoints" / "run"
    _write_resume(ckpt, world_size=3, num_workers=_AR.STREAM_WORKERS)
    out = _AR.resume_meta_json(_args(resume=True, examples_prev=100), tmp_path, "run", 3)
    assert json.loads(out)["examples_prev"] == 100


def test_every_launch_path_routes_through_the_helper() -> None:
    """The original bug was not IN the helper — it was the streaming launch path not CALLING it.

    So pin the wiring: every ``prev_json`` the launcher hands a worker must come from
    ``resume_meta_json(...)``. A new launch path that hand-rolls the payload (which is exactly
    how --resume became a no-op under --stream-dir) fails here.
    """
    import ast

    src = (Path(__file__).resolve().parents[1] / "scripts/ar/iolens_ar_train.py").read_text()
    assigns = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "prev_json" for t in node.targets
        )
    ]
    assert len(assigns) >= 2, "expected a prev_json assignment per launch path"
    for node in assigns:
        assert isinstance(node.value, ast.Call), (
            f"prev_json assigned a non-call at line {node.lineno}"
        )
        assert isinstance(node.value.func, ast.Name)
        assert node.value.func.id == "resume_meta_json", (
            f"prev_json at line {node.lineno} bypasses resume_meta_json()"
        )


def test_restore_seeds_step_start_time() -> None:
    """Fast-forwarding global_step can land on a log boundary; the harness then computes
    `time.time() - step_start_time` before any step has been timed and dies on None."""
    trainer = SimpleNamespace(global_step=0, optimizer=None, step_start_time=None)
    cb = RestoreTrainerState.__new__(RestoreTrainerState)
    cb.trainer = trainer
    cb._restore_step = 18005
    cb._optimizer_state = None
    cb.on_train_start()
    assert trainer.global_step == 18005
    assert isinstance(trainer.step_start_time, float)


def test_g9_keys_duplicates_on_the_interval_not_the_shard(tmp_path: Path) -> None:
    """A later wave on the same rollout shard is DISJOINT data, not a duplicate.

    My first hand-analysis of this keyed on the shard number and reported pt with ~520k duplicate
    pairs; ``done_0000s0100.json`` is skip 0.100, i.e. the interval [0.10, 0.20), which shares no
    conversations with [0.00, 0.10). The true duplicate count was 260,821. Only an exact interval
    re-capture produces byte-identical pairs, so that is what the gate must key on.
    """
    spec = importlib.util.spec_from_file_location(
        "iolens_reconcile_counts",
        Path(__file__).resolve().parents[1] / "scripts/datagen/iolens_reconcile_counts.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def write(name: str, shard: int, pairs: int) -> None:
        (tmp_path / name).write_text(json.dumps({"shard": shard, "pairs": {"train": pairs}}))

    write("done_0000.json", 0, 260_821)          # original, skip 0.0
    write("done_0000s0000.json", 0, 260_821)     # exact re-capture of skip 0.0 -> DUPLICATE
    write("done_0000s0100.json", 0, 259_263)     # skip 0.100, disjoint -> unique
    out = mod.capture_totals(tmp_path)
    assert out["duplicate"] == 260_821
    assert out["unique"] == 260_821 + 259_263
    assert list(out["dup_keys"]) == [(0, 0.0)]


def test_adjacent_wave_intervals_do_not_count_as_overlapping() -> None:
    """Wave N+1 starts exactly where wave N ends; that is adjacency, not overlap.

    `skip + frac` accumulates float error (0.2 + 0.1 == 0.30000000000000004), so a naive
    `lo < e_hi` reads [0.30,0.40) as overlapping [0.20,0.30) by 5.5e-17. That fired for real: the
    pt producer refused every remaining wave and sat in a retry loop while its buffer drained.
    """
    eps = 1e-9

    def overlaps(lo: float, hi: float, e_lo: float, e_hi: float) -> bool:
        return lo < round(e_hi, 6) - eps and round(e_lo, 6) < hi - eps

    # the exact failing case, with the float error reproduced rather than assumed
    assert 0.2 + 0.1 != 0.3
    assert not overlaps(0.3, 0.4, 0.2, 0.2 + 0.1)
    assert not overlaps(0.14, 0.28, 0.0, 0.14)          # chat's wave spacing
    # genuine overlaps must still be caught
    assert overlaps(0.05, 0.15, 0.0, 0.1)               # partial
    assert overlaps(0.0, 0.1, 0.0, 0.1)                 # exact re-capture
    assert overlaps(0.02, 0.04, 0.0, 0.1)               # contained


def test_streaming_skips_a_shard_that_vanished(tmp_path: Path) -> None:
    """The janitor can sweep a shard between listing it and opening it (an out-of-band consumed
    mark races a live worker). That killed a 3-GPU run; a missing shard must be skipped."""
    ds = StreamingPairDataset(tmp_path, rank=0, world=1)
    rows = list(ds._rows(tmp_path / "pairs_train_0000_00.safetensors", 0, 1))
    assert rows == []
