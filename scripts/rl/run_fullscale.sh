#!/bin/bash
# 4 independent 2-GPU pairs (actor cuda:2k, AR cuda:2k+1), lr-focused de-risk
# sweep (STEPS=50 first pass, then extend winners): s0 = run100 control
# (lr 1.41e-5), s1 = the lr-5e-5 hypothesis (miles run100b/c flat-reward ->
# LoRA-lr-too-low, commit 2f7e7433), s2 = lr 1e-4 (skip-lens's own LoRA-RL
# default at matched effective adapter scale), s3 = lr 5e-5 with the looser
# beta 0.02 anchor. ~2048 completions/step box-wide. Zero distributed code by
# design (audit P3).
set -uo pipefail
[ -f "${SC_ENV:-}" ] && source "$SC_ENV"  # optional: box-specific env (secrets, caches)
cd "$(dirname "$0")/../.."   # repo root

STEPS=${STEPS:-300}
CONFIG=${CONFIG:-scripts/rl/configs/rl_ao_2gpu.yaml}
PREFIX=${PREFIX:-iolens-rl-sc100}
BETAS=(${BETAS_OVERRIDE:-0.05 0.05 0.05 0.02})
LRS=(${LRS_OVERRIDE:-1.41e-5 5e-5 1e-4 5e-5})

# 4 concurrent trainers x torch's default nthreads==ncores livelocks the box on
# CPU ops (whitener factorizations spun 73 min at 260 threads/proc on 256 cores).
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32

for k in 0 1 2 3; do
  BETA=${BETAS[$k]}
  LR=${LRS[$k]}
  RUN="${PREFIX}-b${BETA}-lr${LR}-s${k}"
  OUT=artifacts/sc/rl_runs/$RUN
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=$((2 * k)),$((2 * k + 1)) nohup uv run --no-sync python \
      scripts/rl/train_rl_ao.py \
      --config "$CONFIG" \
      --save-dir "$OUT" \
      --num-steps "$STEPS" \
      --kl-beta "$BETA" \
      --lr "$LR" \
      --seed "$k" \
      --wandb-name "$RUN" \
      > "$OUT/console.log" 2>&1 &
  echo "pair $k: $RUN on GPUs $((2 * k)),$((2 * k + 1)) (pid $!)"
done
wait
