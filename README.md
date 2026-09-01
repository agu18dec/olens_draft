# oracle-lens

Train a model to read another model's activations in natural language. Extracted as a
standalone repo from a larger research codebase (see `MANIFEST.md` for provenance); contains
the complete data-generation + SFT-distillation + RL training pipeline, and nothing else.

Two models, both LoRA adapters (r=16, all-linear) on a frozen **Qwen/Qwen3.6-27B** base:

- **AR — activation reconstructor** (text → activation): given a text span, predict the residual
  activation that preceded it, at ~16 layers (L20–63) in one forward pass, via a truncated
  backbone + per-layer conditioning (`oracle_lens.pipeline.multilayer_reconstructor`). Trained
  first on the model's own rollouts, then **frozen forever** — it is the reward model for
  everything downstream.
- **AO — activation oracle** (activation → text): one activation vector replaces the embedding
  of a marker slot in the prompt; the model is trained to verbalize what the activation encodes
  (`oracle_lens.pipeline.soft_token_sft`). Trained by SFT to invert the frozen AR, distilled to
  a multi-bullet student, then RL-tuned with GRPO against the frozen AR (`scripts/rl/`).

## Quickstart

```bash
bash setup.sh --dev        # repo-local .venv via uv (--dev adds ruff/mypy/pytest)
uv run pytest              # CPU test suite — no GPU or model download needed

# smallest end-to-end GPU thing: fetch published artifacts and run the RL gate ladder
uv run python scripts/rl/fetch_artifacts.py     # → artifacts/sc (AO LoRA, frozen AR, whiteners, pool)
bash scripts/rl/checks/run_gates.sh
```

Every training stage can start from published artifacts instead of regenerating data — the HF
dataset repo **`agu18dec/local-workspace`** holds rollouts (`data/rollouts/{chat,pt}/`), whiteners
(`data/whiteners/`), AO pools/arout (`data/ao/`), and checkpoints (`ckpts/ar/…`, `ckpts/ao/…`,
including the distill student `ckpts/ao/distill/…` and the RL result of record
`ckpts/ao/rl/iolens.final.ddp600.s0`).

## Pipeline

```
seed prep + 4-way split          scripts/datagen/iolens_seed_prep.py, iolens_splits.py
  → on-policy rollouts (SGLang)  scripts/datagen/iolens_rollout_gen.py     [separate sglang venv]
  → validate                     scripts/datagen/iolens_rollout_validate.py
  → activation capture           scripts/datagen/iolens_capture_pairs.py (17-layer residuals)
  → whiteners                    scripts/datagen/iolens_fit_whitener.py    [fit ONCE, then frozen]
→ AR training                    scripts/ar/iolens_ar_train.py             [multi-GPU DDP]
→ AO pool / AR-outputs / gate    scripts/ao/ao_build_pool.py, ao_precompute_cluster.py, ao_gate_arout.py
→ AO SFT                         scripts/ao/ao_train_cluster.py
→ distillation                   scripts/distill/ (teacher sampling [vllm venv] → NNOMP select → SFT)
→ RL (GRPO vs frozen AR)         scripts/rl/train_rl_ao.py via run_ddp.sh
```

Stage-by-stage commands: **`docs/pipeline.md`**. RL: **`docs/rl.md`**. Read
**`docs/failure_modes.md`** before running anything expensive — every entry is a failure that
was already paid for once.

## Layout

- `src/oracle_lens/pipeline/` — the multilayer pipeline library (data gen, AR, AO, distill, RL reward)
- `src/oracle_lens/core/` — single-layer primitives (whitening, reconstructor head, NNOMP, shards)
- `src/oracle_lens/` — model loading (`model.py`), chat data (`data.py`), J-lens readout bridge
- `src/jlens/` — vendored [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache-2.0)
- `vendor/mytorch-lightning/` — vendored trainer harness (**private upstream — see its VENDORED.md
  before sharing this repo publicly**)
- `scripts/{datagen,ar,ao,distill,rl}/` — stage launchers; `tests/` — CPU test suite
- Artifacts land under `./artifacts` by default; set `OLA_ROOT` (pipeline) / `SC_ROOT` (RL) to
  point at a big disk.

## Environment notes

- Python ≥3.12, [`uv`](https://docs.astral.sh/uv/); `bash setup.sh` builds the repo-local venv.
- **Two extra venvs are deliberately not declared here**: rollout generation runs under a
  [SGLang](https://github.com/sgl-project/sglang) venv you build separately (the script has no
  repo imports), and distill teacher sampling runs under a vLLM venv (needs prompt-embeds
  support). Both scripts document this in their docstrings.
- Long jobs: run in tmux with `PYTHONUNBUFFERED=1`, tee output to a timestamped file under `logs/`.
- `HF_TOKEN` / `WANDB_API_KEY` via environment only — never on argv, never committed.
- Checks: `uv run ruff check . && uv run mypy src && uv run pytest`.
