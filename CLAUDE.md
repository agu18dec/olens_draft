# oracle-lens — agent notes

Standalone extraction of the oracle-lens training pipeline (provenance: `MANIFEST.md`).
Start with `README.md`; stage-by-stage commands in `docs/pipeline.md`; RL in `docs/rl.md`.

- **Read `docs/failure_modes.md` before running or modifying anything expensive.** Every entry
  cost a real run once. Highlights: layer set comes from `heads.pt`/shard metadata (never
  hardcode 17); load adapters via `oracle_lens.pipeline.ar_loader` (bare
  `PeftModel.from_pretrained` silently loads nothing on `_orig_mod.` keys); whiteners and
  injection scales are frozen — never refit; `--dry-run 1` first (`--dry-run 0` is NOT a dry
  run); monitor `val_epoch/val_ce`, not train loss.
- Env: `bash setup.sh --dev`, then `uv run …`. Python ≥3.12, mypy strict on `src`, ruff line
  length 100 (F722/F821/UP037 ignored on purpose — jaxtyping). Gate before claiming done:
  `uv run ruff check . && uv run mypy src && uv run pytest`.
- `src/jlens/` and `vendor/mytorch-lightning/` are vendored — never edit them.
  mytorch-lightning's upstream is PRIVATE; see its `VENDORED.md` before publishing this repo.
- Artifact roots: `$OLA_ROOT` (pipeline; scripts refuse to run with it unset),
  `artifacts/sc` / `$SC_ROOT` (RL). Published data + checkpoints:
  HF dataset `agu18dec/local-workspace`. `HF_TOKEN` via env only.
- Rollout generation (sglang) and distill teacher sampling (vllm) run in separate venvs you
  build yourself — deliberately not declared in `pyproject.toml`.
- Long GPU jobs: tmux + `PYTHONUNBUFFERED=1` + tee to a timestamped `logs/` file.
- After ANY change to RL injection/reward/data/update math: `bash scripts/rl/checks/run_gates.sh`.
