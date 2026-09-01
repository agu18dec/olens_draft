# Extraction manifest

This repo was extracted 2026-09-01 from the `global-workspace` monorepo at commit
`0a5e30b9e1077560eee233a7ae2008a55d365a68`. It contains ONLY the oracle-lens pipeline
(data generation, AR/AO SFT, distillation, GRPO RL); everything else in the monorepo
(evals, dashboards, plotting, other research lines) was deliberately left behind.
Runtime code is byte-identical to the monorepo modulo the transforms listed below.

## Source → destination map

| Monorepo source | Here | Notes |
|---|---|---|
| `src/global_workspace/ola/` | `src/oracle_lens/pipeline/` | minus `grpo_probe.py`, `oracle_compare.py`, `wsbench_fve.py`, `playground.py` (eval/abandoned) |
| `src/global_workspace/oracle_lens/{whitening,reconstructor,nnomp,dump,sampling,evals,eval_items,masks,stats,dictionary}.py` | `src/oracle_lens/core/` | the single-layer primitive closure; the single-layer trainer (`train_recon.py` etc.) was not carried |
| `src/global_workspace/{model,data,hf_offline,readout_text}.py` | `src/oracle_lens/` | |
| `src/global_workspace/lens.py` | `src/oracle_lens/jlens_readout.py` | renamed to avoid `oracle_lens.lens` stutter |
| `src/jlens/` | `src/jlens/` | vendored anthropics/jacobian-lens (Apache-2.0), minus `vis.py`, `examples.py` |
| `vendor/mytorch-lightning/` | `vendor/mytorch-lightning/` | PRIVATE upstream — see its `VENDORED.md` before sharing publicly |
| `scripts/ola/iolens_{seed_prep,splits,rollout_gen,rollout_validate,capture_pairs,produce_loop,reconcile_counts,fit_whitener}.py` | `scripts/datagen/` | |
| `scripts/ola/iolens_ar_train.py` | `scripts/ar/` | |
| `scripts/ola/ao_{build_pool,precompute_cluster,precompute_gt,gate_arout,train_cluster}.py` | `scripts/ao/` | |
| `scripts/ola/{ao_distill_sample,ao_gt_omp_readout,ao_assemble_distill,ao_distill_train_cluster,ao_raft_select,olens_distill_sample_cluster,olens_r2s_select}.py` | `scripts/distill/` | |
| `scripts/ola/rl_sc/` | `scripts/rl/` | minus `viz/diag_beforeafter_omp4long.{html,json}` (run artifacts) and `checks/bench_{readout,aggregate}.py` (read monorepo eval banks) |
| `scripts/ola/rl/prep_rl_data.py`, `scripts/ola/rl/checks/_lib.py` | `scripts/rl/`, `scripts/rl/checks/` | the only two files the rl stack reused from the Miles stack |
| `tests/{conftest,test_iolens_*,test_ola_*,test_rl_reward,test_ao_adapter_load,test_oracle_{recon,nnomp,dump,masks,sampling}}.py` | `tests/` | minus `test_ola_miles_reward.py`, `test_ola_oracle_compare.py` (dropped modules) |

## Transforms applied

1. **Import rename** (mechanical, word-bounded, longest-prefix-first):
   `global_workspace.oracle_lens → oracle_lens.core`, `global_workspace.ola → oracle_lens.pipeline`,
   `global_workspace.lens → oracle_lens.jlens_readout`, `global_workspace.{model,data,hf_offline,readout_text}
   → oracle_lens.{model,data,hf_offline,readout_text}`. Verified: zero `global_workspace` references remain.
2. **Path-string rewrite** in docstrings/launchers/tests: `scripts/ola/rl_sc/ → scripts/rl/`,
   `scripts/ola/<name>.py → scripts/<stage>/<name>.py`; `__file__`-relative depth fixed where
   `rl_sc/` lost a directory level.
3. **Root scrub**: `paths.py` default is now `./artifacts` (`$OLA_ROOT` to override); the RL root
   `/workspace/sc → artifacts/sc` (`$SC_ROOT` / `--sc-root`); hardcoded `cd /workspace/global-workspace`
   → repo-relative; cluster cache candidates, other-user paths, and venv paths removed or made generic.
4. **Fresh `__init__.py`s** for `oracle_lens` / `oracle_lens.core` / `oracle_lens.pipeline`
   (the monorepo's lazy-symbol maps referenced dropped modules); `core/__init__` no longer lazily
   re-exports the uncarried single-layer trainer.
5. `scripts/distill/olens_distill_sample_cluster.py --prompt-variant` now errors clearly
   (its prompt table lived in a dropped research script).
6. **pyproject**: dependency list trimmed to what the carried code imports (plus `torchdata`, an
   undeclared runtime dep of vendored mytorch-lightning); ruff/mypy-strict/pytest configs ported;
   per-file ruff ignores added for `scripts/rl/{viz,checks}` (pre-existing style debt — those files
   were never under the monorepo's ruff gate and are byte-identical to the runs of record).
7. `WANDB_PROJECT` defaults to `oracle-lens`, env-overridable.

## Not carried (and why)

- **Miles+SGLang RL stack** (`scripts/ola/rl/` minus the two files above): requires two external
  vendored repos (miles @051cd15, nla) that are gitignored in the monorepo and not on disk.
  Superseded in practice by the self-contained stack in `scripts/rl/`.
- All eval/plotting/report-site/judge scripts, dashboards, and the other research lines
  (template lens, RelP, workspace-bench, …).
- Cluster/pod launch tooling (`scripts/cluster/`, `scripts/iolens/`): Slurm/pod-specific;
  replaced by generic tmux guidance in `docs/pipeline.md`.

## Verification at extraction time

`uv run ruff check .` clean; `uv run mypy src` strict clean (scripts/tests carry the monorepo's
pre-existing annotation debt — they were never under its `mypy src` gate); full CPU test suite
377 passed / 1 skipped; runtime import sweep of every `src` module clean; secrets grep clean.
