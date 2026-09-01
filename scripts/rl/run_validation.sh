#!/bin/bash
# Phase-3 validation run: 50 steps, 2 GPUs (actor cuda:0, frozen AR cuda:1),
# 8 prompts x G=64 = 512 completions/step. Pass criteria (audit plan):
# reward/mean up from ~-1.50; eval/fve from ~0.102 rising toward 0.1437;
# extraction >= 0.95; truncation < 2%; kl smooth; grads finite.
set -uo pipefail
[ -f "${SC_ENV:-}" ] && source "$SC_ENV"  # optional: box-specific env (secrets, caches)
cd "$(dirname "$0")/../.."   # repo root

RUN=${AO_RUN_NAME:-iolens-rl-sc-val}
OUT=artifacts/sc/rl_runs/$RUN
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=${GPUS:-0,1} uv run python scripts/rl/train_rl_ao.py \
    --config scripts/rl/configs/rl_ao_2gpu.yaml \
    --save-dir "$OUT" \
    --wandb-name "$RUN" \
    "$@" 2>&1 | tee -a "$OUT/console.log"
