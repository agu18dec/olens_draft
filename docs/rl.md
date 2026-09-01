# GRPO RL on the AO — the self-contained stack (`scripts/rl/`)

On-policy GRPO that trains the activation oracle against the **frozen** AR as a reward model.
Everything runs on one node from this repo: no serving engine, no ray, no weight sync. This
doc is the execution reference; the quickstart in `scripts/rl/README.md` points here.

The result of record: warm-started from the distillation student
`ckpts/ao/distill/final.s0/step105`, 600 DDP steps → **joint FVE 0.120 → 0.155 (peak 0.159)**
— the first positive RL result on the AO line, beating the RAFT-1 rejection-sampling bar
(**0.1237**) and clearing the single-readout AR ceiling (~0.144). Published as
`ckpts/ao/rl/iolens.final.ddp600.s0` (`iter_000600`) in `agu18dec/local-workspace`.

## Design: the sampler IS the trainer

`scripts/rl/train_rl_ao.py` takes **one optimizer step per rollout batch**, sampling with HF
`generate` on the training weights themselves. Because the sampling policy and the updated
policy are the same weights, the importance ratio ≡ 1 — there is no ratio, no clipping, no
importance sampling, no trainer↔engine weight sync, and no fidelity/TIS machinery to drift.
Per step: `batch_prompts × group_size` rollouts → reward on the frozen AR (second GPU) →
per-prompt group-normalized advantages → micro-batched forward+backward with selective fp32
logprobs → **exact analytic KL(policy‖ref)** via a linearized surrogate whose gradient is
exactly zero at policy≡ref (naive autograd of the KL leaves a machine-epsilon gradient that
Adam amplifies to full lr-scale steps — found by the gate ladder).

Invariants the code enforces (don't fight them):

- `temperature == 1.0` hard assert + unclamped sampler — required for the no-ratio surrogate's
  validity. Eval-only greedy via `--eval-temperature 0`.
- **The injection marker is a stop token**: a marker echo halts generation (crash-impossible
  by construction) and marker-terminated rollouts train like EOS-terminated ones.
- **Prompts come pre-rendered as `prompt_ids` in the parquet** — the trainer never re-renders
  or re-tokenizes; sidecar asserts catch tokenizer drift at startup.
- Eval generation is bucketed by exact prompt length (GDN architecture: pads upstream of the
  recurrence corrupt state — never left-pad).
- Adapter liveness assert at load (≥90% of saved LoRA tensors must map) — PEFT silently loads
  nothing on key-schema drift (see failure_modes.md).

The injection itself is one forward hook on `get_input_embeddings()` that REPLACES the marker
slot's embedding with the pre-transformed parquet vector (`scripts/rl/sl_injection.py` — the
same-ancestry function the SFT generation path splices with; the gate ladder proves
hook ≡ splice token-exactly). The contract is the distill lineage's: `unit` transform,
`16000 × h/‖h‖`, layers 20–60, marker `㈜` (id 158983). Sidecar `injection_scale: null` is
asserted at startup — parquet vectors are already transformed.

## The reward, exactly

Given a generated readout and the injected TRUE residual `h` at layer `ly`
(`oracle_lens.pipeline` reward code; `--reward-agg joint` for the bullet student):

1. Parse bullets: lines starting `"- "`, continuation lines appended, strip, drop empties,
   cap at `--reward-k-max` (4), drop bullets <2 tokens.
2. Embed each bullet with the **frozen chat-final AR** (`ar.chat.mlayer.lc.s0/ex16014240`),
   take the prediction at `ly`.
3. Whiten with the layer's frozen whitener (`whitening_iolens_chat_L{ly}`, ridge 0.1). Atoms =
   unit-normalized whitened embeddings; query = whitened `h`, NOT normalized.
4. Non-negative NNLS refit → reward = **joint FVE** `1 − ‖x_w − Σcᵢaᵢ‖²/‖x_w‖²`. Joint, not
   mean per-bullet — per-bullet rewards 4 copies of the best bullet; joint penalizes
   redundancy for free.

Failure floor: r = −4.0 (cos = −1) for truncated/unparseable rollouts, trained on
(anti-runaway); zero parseable bullets → 0. Reward validity is pre-certified: two
independently trained same-scale ARs agree on within-group rankings at Spearman 0.988 /
top-1 94.3%, so advantages have ~0.99 test-retest reliability. Rollouts, reward, and the
policy's SFT training must all inject identically — a transform/scale mismatch scores a
different model (measured −47% at the wrong scale).

## Setup

```bash
bash setup.sh                     # repo-local .venv
export OLA_ROOT=artifacts/sc SC=artifacts/sc
export WANDB_MODE=online PYTORCH_ALLOC_CONF=expandable_segments:True
# optional GDN fast kernels (3.0x fwd+bwd, gated by the ladder): needs nvcc on PATH
uv pip install flash-linear-attention tilelang
export FLA_TILELANG=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
```

### 1. Fetch artifacts (`scripts/rl/fetch_artifacts.py`)

```bash
uv run python scripts/rl/fetch_artifacts.py --sc-root $SC     # ~90 GB, idempotent
```

Downloads from `agu18dec/local-workspace` + the `Qwen/Qwen3.6-27B` hub snapshot, and lays out
`$SC` (default `artifacts/sc`, override `--sc-root` / `$SC_ROOT`) mirroring the pipeline's
`$OLA_ROOT` naming so the datagen/AO scripts run unmodified: the SFT AO LoRA
(`ckpts/ao/chat/k4.L20plus.s2/step3002`), the frozen reward AR
(`ckpts/ar/chat/mlayer.lc.s0/ex16014240`), the chat whiteners, `pool_iolens` + its arout, and
the chat rollouts. It also writes the **`lora_hf/` copy** of the AO adapter:

- every safetensors key stripped of `._orig_mod.` — the SFT runs train with
  `compile_blocks=on`, so saved keys carry the torch.compile wrapper prefix, and
  `PeftModel.from_pretrained` on this trainer's UNCOMPILED base would silently load nothing
  (the documented inert-LoRA failure). This trainer never compiles, so the strip happens once,
  at fetch time.
- `lora_dropout: 0` in the copied `adapter_config.json` — rollout and update must sample the
  same policy, so the policy is dropout-free by construction.

For the run of record's warm start, additionally download the distill student:

```bash
hf download agu18dec/local-workspace --repo-type dataset \
    --include 'ckpts/ao/distill/final.s0/step105/**' --local-dir $SC/hf
```

(then strip it the same way, or point `--ao-lora` at a `lora_hf` you produce with
`fetch_artifacts.py --skip-download` after placing it — the config
`scripts/rl/configs/rl_ao_iolensfinal_2gpu.yaml` expects
`$SC/hf/ckpts/ao/distill/final.s0/step105/lora_hf`).

### 2. Data prep (`prep_rl_data.py` + the small-bank GT strategy)

RL items are (true residual, layer) pairs. The GT bank only needs to tile the same crop range
as the AR bank it's paired with, and 20k rows sample fine from far fewer crops — so build 8 of
the 64 GT shards (crops [0, 65,895), one shard per GPU, ~35 min on 8 GPUs) and prefix-slice
the published arout to match (`load_arout` requires only contiguous-from-0 tiling +
fingerprint match; the seeded pick tables are generated over the full pool then sliced, so a
prefix slice preserves pick alignment exactly):

```bash
for s in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$s \
  uv run python scripts/ao/ao_precompute_gt.py \
    --pool ao_pool/pool_iolens.safetensors --split train \
    --n-shards 64 --shard $s --layers-per-crop 4 --layer-min 20 & done; wait

uv run python scripts/rl/slice_arout.py \
  --src $SC/ao_arout/ar.chat.mlayer.lc.s0/ex16014240 \
  --dst $SC/ao_arout_small --split train --hi 65895
```

Then the sidecar (tokenizer-only; no merge stage — a merged VL checkpoint was only ever needed
for external serving) and the parquets (a 512-row gate parquet + the 20k training parquet):

```bash
uv run python scripts/rl/prep_rl_data.py \
  --ao-lora-dir <ckpt>/lora --ao-meta <ckpt>/meta.json \
  --out-dir $SC/rl_u64 --stage sidecar --layers 20,24,28,32,36,40,44,48,52,56,60
uv run python scripts/rl/prep_rl_data.py \
  --ao-lora-dir <ckpt>/lora --ao-meta <ckpt>/meta.json \
  --out-dir $SC/rl_u64 --stage parquet \
  --layers 20,24,28,32,36,40,44,48,52,56,60 \
  --pool $SC/ao_pool/pool_iolens.safetensors \
  --arout-dir $SC/ao_arout_small --gt-arout-dir $SC/ao_gtout/pool_iolens \
  --n-rows 20000 --gate-rows 512 --seed 0
```

Parquet rows carry: `row_id · layer · prompt_ids` (prep-rendered, marker slot inside) ·
`prompt_text` · `activation_vector` (the transformed vector the AO reads) · `gold_vector`
(raw TRUE residual the reward scores against). Prompt kind, transform, and alpha all come
from the checkpoint's own `meta.json` — prep for the distill-final policy therefore renders
`concepts_raw` prompts automatically. Optionally build the diagnostic probe bank
(`scripts/rl/build_diag_bank.py`) — the trainer logs greedy readouts on it to wandb every eval.

### 3. The gate ladder (`scripts/rl/checks/run_gates.sh`)

```bash
GATE_PQ=$SC/rl_u64/rl_gate_0.parquet bash scripts/rl/checks/run_gates.sh   # ~15 min
```

**Run it after ANY change to injection, reward, data, or the update math** — it exists because
each gate has already caught a real bug:

| gate | proves |
|---|---|
| `gate_injection` | mechanism triple + hook ≡ splice token-exact |
| `gate_reward_equiv` | `score_texts` ≡ inline ≡ numpy, max diff 0.0; floor contract |
| `gate_padding` | padding adds nothing over the GDN batch-numerics baseline |
| `gate_direction_noop` | toy reward moves the right way in one step; constant reward ⇒ gnorm **exactly 0** |
| `gate_probe` | matched-vs-shuffled golds: gap ≥ 0.15, win ≥ 0.65 — ties the whole path to known-good numbers |

The ladder itself found: the epsilon-KL × Adam drift (hence the linearized KL), a truncation
floor contaminating toy rewards, and that mb1-vs-mb8 logprob deltas are kernel batch numerics,
not padding. An engine-mismatch ladder (serve/radix/logprob/sync gates, fidelity, TIS) has no
referent here — that deletion is the point of this stack.

### 4. Launch

```bash
# validation run first: 50 steps, 2 GPUs (actor cuda:0, frozen AR cuda:1), 8×64 = 512 compl/step.
# Pass: reward/mean rising, extraction ≥ 0.95, truncation < 2%, kl smooth, grads finite.
bash scripts/rl/run_validation.sh

# single 2-GPU run with explicit config:
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 uv run python scripts/rl/train_rl_ao.py \
  --config scripts/rl/configs/rl_ao_iolensfinal_2gpu.yaml \
  --save-dir $SC/rl_runs/myrun --wandb-name myrun 2>&1 | tee logs/rl_$(date +%Y%m%d_%H%M).log

# 4-arm sweep: 4 independent 2-GPU pairs (no distributed code; parallelism = seeds)
STEPS=50 CONFIG=scripts/rl/configs/rl_ao_iolensfinal_2gpu.yaml PREFIX=myrun \
  LRS_OVERRIDE="1.41e-5 5e-5 1e-4 3e-4" BETAS_OVERRIDE="0.05 0.05 0.05 0.05" \
  bash scripts/rl/run_fullscale.sh

# the full-scale shape (the run of record's): ONE policy on the whole box via torchrun DDP —
# 7 actor ranks on GPUs 0–6 + the frozen AR on GPU 7 (hosted by rank 0). LoRA-grad
# all-reduce keeps weights bitwise-identical across ranks; on-policy validity is per-rank
# exact. 14 prompts × G=64 = 896 completions/step at ~33 s/step on 8×H200.
STEPS=600 CONFIG=scripts/rl/configs/rl_ao_iolensfinal_2gpu.yaml \
  RUN=myrun-ddp BP=14 bash scripts/rl/run_ddp.sh
```

Configs are in `scripts/rl/configs/` (keys are argparse dests; CLI overrides YAML). Measured
operating points (27B, H200): `logp_micro_batch 4` (8 OOMs at `max_new_tokens 128` with the
graph + ref forward on 141 GB), `max_new_tokens 128` (96 truncated ~20% of T=1 rollouts),
lr 5e-5 / β 0.05 as the best-supported 2-GPU point (the ddp600 run used β 0.02). The 7+1
topology beats 4 actors + 4 ARs decisively: scoring is ~6–10 s vs ~100 s of actor work.
`run_fullscale.sh`/`run_ddp.sh` export `OMP_NUM_THREADS=32/24` — multiple trainers × torch's
default nthreads==ncores livelocks the box. Resume with
`--resume-from-lora <save_dir>/iter_NNNNNN` (Adam moments + data cursor + wandb step restore;
the KL reference always stays the SFT init).

Throughput anchors: 2-GPU pair with fla+tilelang ≈ 114 s / 512 completions; DDP-8 ≈ 33 s / 896.
The fla+tilelang path is gated (grad cosine ≥ 0.9956 vs fallback on all 992 LoRA tensors,
hook≡splice still exact, no-op gnorm still exactly 0).

## The result of record, and what to expect

**`ckpts/ao/rl/iolens.final.ddp600.s0`** (`iter_000600`): GRPO from `distill/final.s0/step105`
(prompt `concepts_raw`, joint reward), 600 DDP-8 steps, β 0.02 → eval joint FVE
**0.120 → 0.155, peak 0.159**. Beats RAFT-1 (0.1237) and the single-readout AR ceiling
(~0.144), and beats literal source-text reconstruction on this eval set (GT 0.129 mean /
0.075 median — the set is genuinely hard). **No reward hacking**: bullet_diversity rose
0.15 → 0.34 and truncation fell to 0; the eventual plateau is *advantage* saturation (88% of
groups at zero within-group spread), not corruption.

Honest caveats — read before planning follow-ups:

- **Reward-model headroom binds, not optimization.** On the u64 single-readout warm start,
  held-out reward was flat across a **20× lr sweep** while KL (→0.34) and entropy (−66%)
  showed massive policy movement; the policy's greedy FVE (0.148) already ≈ the frozen AR's
  own reconstruction ceiling (0.1437). Gains must come from raising the ceiling (a stronger /
  refreshed / context-conditioned reward AR, or a different reward space), not tuning the
  optimizer.
- **Readout jank at low KL.** The ddp600 checkpoint's readouts drift toward continuations
  (KL reached 0.60), 4–39% of rollouts hit the 128-token cap, and its "diversity" is partly
  fake — 4 bullets are often rephrasings of one reading (pooled-gain decomposition: the +5.5
  pooled gain is ~70% a better single bullet, only +1.6 real complementarity). A staged fix
  plan (Dr.GRPO `--dr-grpo` for the /std noise amplification + length bias, KL/token-budget
  A/B, anti-repetition and fluency terms, multi-target reconstruction reward) was **in
  flight, not complete** — treat those levers as designed, not validated. Known trap: a
  64-token budget under Dr.GRPO collapses length (the −4 truncation floor becomes a cliff);
  keep the budget above the ~81-token mean.
- True residuals are intrinsically ~3× harder to read than AR images (0.12 vs 0.31 on the
  same models) — don't interpret absolute FVE against 1.0. pass@16 (0.199) is the honest
  near-term target region.
- Watch the pass@k curve: RL success = pass@1 climbing toward the old pass@16 WITHOUT the
  curve flattening (pass@16 dropping toward pass@1 = entropy collapse, not learning).

## Alternative stack

A full-scale Miles+SGLang GRPO stack (server-based rollouts, weight sync, the fidelity/TIS
gate ladder) exists but was **not carried into this repo** — it depends on two external
private-ish repos. It is documented in the source monorepo
(`docs/project/experiments/ola/` — the skip-lens audit and miles runbooks). Its measured
throughput was ~7×/GPU *worse* than this stack for this problem; use it only if you
specifically need engine-served rollouts.
