# rl_sc — self-contained AO GRPO (RL training quickstart)

Truly on-policy RL for the Inverted-OLens AO on **2 GPUs per run** (actor + frozen
reward AR), no sglang/vllm/ray, no weight sync, no fidelity/TIS machinery — the
sampler IS the trainer. Fork of skip-lens @13547ae per the plan-B migration in
`docs/project/experiments/ola/skip_lens_audit.md` §5; execution record + results in
`docs/project/experiments/ola/rl_sc_runbook.md`.

## TL;DR to a training run on a fresh 8-GPU box

```bash
# 0. env (standalone box; ~5 min)
bash bootstrap_gpu_box.sh                      # project venv (torch 2.9.1+cu128, tf 5.11)
source scripts/cluster/env.sh && source scripts/cluster/pod_env.sh
export HF_HOME=<your hf cache> OLA_ROOT=artifacts/sc SC=artifacts/sc
export WANDB_MODE=online PYTORCH_ALLOC_CONF=expandable_segments:True
# GDN fast kernels (3x fwd+bwd; gated - see "Speed" below). nvcc must be on PATH.
uv pip install --python "$UV_PROJECT_ENVIRONMENT" flash-linear-attention tilelang
export FLA_TILELANG=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH

# 1. artifacts (~90 GB total; ~5 min on a fast pipe)
uv run --no-sync python scripts/rl/fetch_artifacts.py --sc-root $SC
#    base 27B + SFT AO lora (+ lora_hf stripped copy) + frozen AR + whiteners
#    + pool/arout/rollout banks from HF agu18dec/local-workspace

# 2. data: GT bank (~35 min on 8 GPUs) + sidecar + parquets (~5 min)
for s in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$s \
  uv run --no-sync python scripts/ao/ao_precompute_gt.py \
    --pool ao_pool/pool_iolens.safetensors --split train \
    --n-shards 64 --shard $s --layers-per-crop 4 --layer-min 20 & done; wait
uv run --no-sync python scripts/rl/slice_arout.py \
  --src $SC/ao_arout/ar.chat.mlayer.lc.s0/ex16014240 \
  --dst $SC/ao_arout_small --split train --hi 65895
uv run --no-sync python scripts/rl/prep_rl_data.py \
  --ao-lora-dir <ckpt>/lora --ao-meta <ckpt>/meta.json \
  --out-dir $SC/rl_u64 --stage sidecar --layers 20,24,28,32,36,40,44,48,52,56,60
uv run --no-sync python scripts/rl/prep_rl_data.py \
  --ao-lora-dir <ckpt>/lora --ao-meta <ckpt>/meta.json \
  --out-dir $SC/rl_u64 --stage parquet \
  --layers 20,24,28,32,36,40,44,48,52,56,60 \
  --pool $SC/ao_pool/pool_iolens.safetensors \
  --arout-dir $SC/ao_arout_small --gt-arout-dir $SC/ao_gtout/pool_iolens \
  --n-rows 20000 --gate-rows 512 --seed 0

# 2c. (optional, ~3 min) diagnostic probe bank — acetaminophen/GFP/suppression
#     items from oracle_lens.eval_items.OPEN_ITEMS; the trainer logs greedy AO
#     readouts on them as the wandb diag/samples table every eval
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/rl/build_diag_bank.py \
  --ao-meta <ckpt>/meta.json --sidecar $SC/rl_u64/merged/nla_meta.yaml \
  --out $SC/rl_u64/rl_diag.parquet --layers 24,40,56

# 3. gates (run BEFORE trusting any training; ~15 min)
GATE_PQ=$SC/rl_u64/rl_gate_0.parquet bash scripts/rl/checks/run_gates.sh

# 4. train (one 2-GPU run; ~230 s/step at 512 completions/step)
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python scripts/rl/train_rl_ao.py \
  --config scripts/rl/configs/rl_ao_u64_2gpu.yaml \
  --save-dir $SC/rl_runs/myrun --wandb-name myrun

# 4b. or a 4-arm sweep across 8 GPUs (independent 2-GPU pairs)
STEPS=50 CONFIG=scripts/rl/configs/rl_ao_u64_2gpu.yaml PREFIX=myrun \
  LRS_OVERRIDE="1.41e-5 5e-5 1e-4 3e-4" BETAS_OVERRIDE="0.05 0.05 0.05 0.05" \
  bash scripts/rl/run_fullscale.sh

# 4c. or ONE policy on the whole box: DDP, 7 actor ranks + the AR on GPU 7
#     (LoRA-grad all-reduce; weights bitwise-identical across ranks; noop gate
#     re-verified under torchrun). 896 completions/step at ~33 s/step.
STEPS=50 RUN=myrun-ddp BP=14 bash scripts/rl/run_ddp.sh
```

Resume: `--resume-from-lora <save_dir>/iter_NNNNNN` (Adam moments + data cursor +
wandb step restore automatically; the KL reference always stays the SFT init).

## What the trainer is

`train_rl_ao.py` — GRPO with **one optimizer step per rollout batch**, so the
importance ratio ≡ 1 and no ratio/clip/IS machinery exists. Per step:
`batch_prompts × group_size` rollouts via HF `generate` on the training weights
(zero padding: each prompt expands to its own group), reward = whitened cosine
against the frozen LC-AR on the second GPU (`rl_reward.score_texts`, r ∈ [−4, 0],
floor −4.0), per-prompt group-normalized advantages, micro-batched
forward+backward with selective fp32 logprobs, exact analytic KL(policy‖ref) via a
linearized surrogate (gradient exactly zero at policy≡ref — see the runbook for
why naive autograd of the KL drifts under Adam).

Invariants the code enforces (don't fight them):
- `temperature == 1.0` hard assert + unclamped sampler — required for the
  no-ratio surrogate's validity. Eval-only greedy via `--eval-temperature 0`.
- the injection marker is a **stop token** — a marker echo halts generation
  (crash-impossible by construction), and marker-terminated rollouts train like
  any EOS-terminated one.
- prompts come pre-rendered as `prompt_ids` in the parquet — the trainer never
  re-renders or re-tokenizes; sidecar asserts catch tokenizer drift at startup.
- eval generation is bucketed by exact prompt length (GDN arch: pads upstream of
  the recurrence corrupt state — never left-pad).
- adapter liveness assert at load (≥90% of saved LoRA tensors must map) — PEFT
  silently loads NOTHING on key-schema drift.

## Configs

`configs/rl_ao_2gpu.yaml` (explain ckpt) / `configs/rl_ao_u64_2gpu.yaml` (u64
continuation ckpt). Keys are argparse dests; CLI overrides YAML. The load-bearing
ones and their measured operating points (8×H200, 27B):

| knob | value | why |
|---|---|---|
| batch_prompts × group_size | 8 × 64 = 512/step | G=64 gives tight group baselines |
| max_new_tokens | 128 | 96 truncated ~20% of T=1 rollouts |
| logp_micro_batch | 4 | 8 OOMs at mnt=128 (graph + ref forward on 141 GB) |
| lr / kl_beta | 5e-5 / 0.05 | best-supported operating point (runbook sweep) |
| actor/reward device | cuda:0 / cuda:1 | 54 GB + 50 GB never co-reside |

Multi-run: `run_fullscale.sh` launches 4 independent 2-GPU pairs (no distributed
code anywhere — parallelism = independent seeds, per the audit's P3). It exports
`OMP_NUM_THREADS=32` etc — 4 trainers × torch's default nthreads==ncores
livelocks the box.

## Gates (`checks/`)

Run `run_gates.sh` after ANY change to injection, reward, data, or the update
math. `gate_injection` (mechanism triple + hook≡splice exact), `gate_reward_equiv`
(score_texts ≡ inline ≡ numpy, diff 0.0), `gate_padding` (padding adds nothing
over the GDN batch-numerics baseline), `gate_direction_noop` (toy reward: one step
moves reward the right way; constant reward ⇒ gnorm exactly 0), `gate_probe`
(matched-vs-shuffled golds: gap ≥ 0.15, win ≥ 0.65 — ties the whole path to
known-good numbers). The miles ladder's 02/04/05/07 + fidelity + TIS have no
referent here (no engine) — that deletion is the point of this stack.

## Speed (measured 2026-08-11, 27B, H200)

| config | completions/step | s/step | compl/GPU·s |
|---|---|---|---|
| miles/sglang run100 (8 GPUs) | 128 | 116 | 0.14-0.20 |
| 2-GPU pair, torch-fallback GDN | 512 | ~280 | 0.91 |
| 2-GPU pair, fla+tilelang + gen-batching | 512 | 114 | 2.25 |
| DDP 8-GPU (7 actors + 1 AR) | 896 | 33 | 3.4 |

fla+tilelang (FLA_TILELANG=1; needs nvcc) is 3.0x on fwd+bwd and is GATED:
grads vs fallback cos >= 0.9956 all 992 LoRA tensors, hook==splice still
token-exact, no-op gnorm still exactly 0. Generation batching
(--gen-prompts-per-call, same-length prompts only, zero padding) is greedy
token-identical to per-prompt calls. Topology note: 7+1 beats 4 actors + 4 ARs
decisively - scoring is ~6-10 s vs ~100 s of actor work; 1:1 ARs would trade
43% of rollout throughput for a ~4% scoring saving. Boot ~2 min. Eval (128) ~35 s. And the headline
finding you should know before launching anything: with the current frozen AR
reward, held-out reward is **flat across a 20× lr range** while KL/entropy show
massive policy movement — the reward model's own reconstruction ceiling binds
(u64 greedy FVE 0.148 ≈ the AR's 0.1437). Tune the *ceiling* (stronger/refreshed
AR, different reward space), not the optimizer.
