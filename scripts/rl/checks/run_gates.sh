#!/bin/bash
# The rl_sc gate ladder, in order; stops at the first FAIL.
# Gates 04/05/07 + fidelity + TIS of the miles ladder are dropped BY CONSTRUCTION
# (no engine, no cache, no weight sync, sampler == trainer) — see rl_sc_runbook.md.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root
[ -f "${SC_ENV:-}" ] && source "$SC_ENV"  # optional: box-specific env (secrets, caches)

SC=${SC:-artifacts/sc}
AO_LORA=$SC/hf/ckpts/ao/chat/k4.L20plus.s2/step3002/lora_hf
AR=$SC/hf/ckpts/ar/chat/mlayer.lc.s0/ex16014240
WHIT=$SC/whiteners/chat
SIDE=$SC/rl/merged/nla_meta.yaml
GATE_PQ=${GATE_PQ:-$SC/rl/rl_gate_0.parquet}
PY="uv run --no-sync python"

run() { echo; echo "=== $1 ==="; shift; time "$@" || { echo "GATE FAILED"; exit 1; }; }

run gate_injection     $PY scripts/rl/checks/gate_injection.py \
    --parquet "$GATE_PQ" --sidecar "$SIDE" --ao-lora "$AO_LORA"
run gate_reward_equiv  $PY scripts/rl/checks/gate_reward_equiv.py \
    --parquet "$GATE_PQ" --ar-ckpt "$AR" --whitener-dir "$WHIT"
run gate_padding       $PY scripts/rl/checks/gate_padding.py \
    --parquet "$GATE_PQ" --sidecar "$SIDE" --ao-lora "$AO_LORA"
run gate_direction_noop $PY scripts/rl/checks/gate_direction_noop.py \
    --parquet "$GATE_PQ" --sidecar "$SIDE" --ao-lora "$AO_LORA"
run gate_probe         $PY scripts/rl/checks/gate_probe.py \
    --parquet "$GATE_PQ" --sidecar "$SIDE" --ao-lora "$AO_LORA" \
    --ar-ckpt "$AR" --whitener-dir "$WHIT"

echo; echo "ALL GATES PASS"
