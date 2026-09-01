#!/bin/bash
# 8-GPU DDP training: 7 actor ranks (GPUs 0-6) + the frozen reward AR on GPU 7
# (hosted by rank 0). One policy, batch_prompts*G completions per optimizer step
# (default 14 x 64 = 896). Weights stay identical across ranks via grad
# all-reduce; on-policy validity is per-rank exact (each rank samples from the
# synchronized weights).
#
#   STEPS=50 CONFIG=scripts/rl/configs/rl_ao_u64_2gpu.yaml \
#     RUN=my-ddp-run bash scripts/rl/run_ddp.sh
set -uo pipefail
[ -f "${SC_ENV:-}" ] && source "$SC_ENV"  # optional: box-specific env (secrets, caches)
cd "$(dirname "$0")/../.."   # repo root

NPROC=${NPROC:-7}
REWARD_DEV=${REWARD_DEV:-cuda:7}   # AR GPU (last visible device); cuda:3 for a 4-GPU subset
STEPS=${STEPS:-300}
CONFIG=${CONFIG:-scripts/rl/configs/rl_ao_u64_2gpu.yaml}
RUN=${RUN:-iolens-rl-ddp}
BP=${BP:-14}                 # global prompts/step; must divide by NPROC
OUT=artifacts/sc/rl_runs/$RUN
mkdir -p "$OUT"

export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24

uv run torchrun --rdzv-backend=c10d \
    --rdzv-endpoint="localhost:${RDZV_PORT:-29500}" --nproc-per-node="$NPROC" \
    scripts/rl/train_rl_ao.py \
    --config "$CONFIG" \
    --save-dir "$OUT" \
    --num-steps "$STEPS" \
    --batch-prompts "$BP" \
    --reward-device "$REWARD_DEV" \
    --wandb-name "$RUN" \
    "$@" 2>&1 | tee -a "$OUT/console.log"
