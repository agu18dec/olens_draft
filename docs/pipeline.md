# The oracle-lens pipeline: data generation → AR → AO → distillation

End-to-end runbook for this repo. Every command below runs from the repo root with the
repo-local venv (`bash setup.sh` once, then `uv run …`). Artifacts live under `$OLA_ROOT`:
the package default is `./artifacts` (`src/oracle_lens/pipeline/paths.py`), but the stage
scripts refuse to run with the variable unset (a guard against writing terabytes to a surprise
location) — so `export OLA_ROOT=$PWD/artifacts` first, or point it at a large disk for real
runs. Long jobs: run in tmux with `PYTHONUNBUFFERED=1`, tee to a timestamped file in `logs/`
(e.g. `… 2>&1 | tee logs/ar_$(date +%Y%m%d_%H%M).log`). Block-buffered output makes a healthy
job look hung.

Base model: `Qwen/Qwen3.6-27B`. Published artifacts (data + checkpoints): HF dataset repo
[`agu18dec/local-workspace`](https://huggingface.co/datasets/agu18dec/local-workspace).
The RL stage that follows all of this is documented in [rl.md](rl.md); standing gotchas in
[failure_modes.md](failure_modes.md).

## What the two halves are

| | AR — Activation Reconstructor | AO — Activation Oracle |
|---|---|---|
| direction | text span → activation | activation → text |
| input | token ids of a span | ONE activation vector, injected as a soft token (marker-slot embedding replacement) |
| output | `[b, 17, 5120]` residual reconstruction at all target layers in one forward | the span's text |
| trained by | whitened cosine loss vs true residuals | masked cross-entropy on the target tokens |
| adapter | LoRA on frozen Qwen3.6-27B + a small head (`heads.pt`) | LoRA on frozen Qwen3.6-27B |

The activation a span maps to is the residual at `prev_pos = span_start − 1` — the state
immediately *before* the span, so the span is that state's continuation. Target layers are
`(0, 4, 8, …, 60, 63)` = 17 layers, but the trained set is whatever `heads.pt` says (the
chat AR of record drops layer 0 → 16 trained layers). **Never hardcode 17** — derive the
layer universe from `heads.pt` / shard metadata (see failure_modes.md).

You do NOT tell the AR which layer to reconstruct: one forward returns all layers, with the
conditioning internal (per-layer learned embedding rows). The AO is the opposite — the prompt
names the layer of the single injected vector.

Pipeline at a glance:

```
seeds ──► rollouts ──► pairs capture ──► whiteners ──► AR training (frozen at final rung)
 (split stamped          │                                   │
  BEFORE generation)     └──► AO pool ──► AR outputs (arout) ┴─► gate ─► scale ─► AO training
                                                                                     │
                                                    distillation (bullet student) ◄──┘
                                                                                     │
                                                                          GRPO RL (docs/rl.md)
```

---

## Stage 1 — Seeds + the 4-way split

`scripts/datagen/iolens_seed_prep.py` writes seed shards to
`$OLA_ROOT/seeds_iolens_{chat,pt}/`: the chat cell uses WildChat-1M user turns, the pt cell
uses 256-token FineWeb-Edu prefixes. **The 4-way split (`ar_train` / `ao_train` / `ao_val` /
`eval`) is stamped on every seed BEFORE any generation**, as a pure content hash of the
conversation's first user turn (chat) / document id (pt) — so a requeued or regenerated task
can never move a conversation across the AR/AO boundary, and every turn of a conversation
lands on the same side.

```bash
uv run python scripts/datagen/iolens_seed_prep.py --mode chat     # and --mode pt
uv run python scripts/datagen/iolens_splits.py                    # verify stamps + write splits_iolens.json
```

`iolens_splits.py` re-derives the split, checks conversation cohesion, and writes the
canonical `$OLA_ROOT/splits_iolens.json` that later gates reconcile against. Exact counts go
to `seeds_iolens_<mode>/report.json` (gate G9 reads them).

## Stage 2 — On-policy rollouts (separate sglang venv)

`scripts/datagen/iolens_rollout_gen.py` is a standalone SGLang worker: one GPU, one seed-shard
stride, **no repo imports** — it runs in an sglang venv you build yourself (sglang is
deliberately not a dependency of this repo; its torch pin conflicts). Each worker launches a
stock `sglang.launch_server`, generates with `return_logprob` on so the shard stores the EXACT
output token ids (never re-tokenized text), writes atomic chunk parts (a crash loses ≤1
chunk), and packs `rollouts_{NNNN}.safetensors` + an exact-count report into
`$OLA_ROOT/rollouts_iolens/<mode>/`.

```bash
# one worker per GPU, in tmux, unbuffered, logged:
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 <your-sglang-venv>/bin/python \
    scripts/datagen/iolens_rollout_gen.py --mode chat --shard 0 --n-shards 12 --port 30100 \
    2>&1 | tee logs/gen_c0_$(date +%Y%m%d_%H%M).log
```

Validate the FIRST shard(s) before scaling the fleet — gates G1–G6:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/datagen/iolens_rollout_validate.py \
    --mode chat --shards 0 --g3-convs 100
```

G1 tokenizer identity, G2 prompt-render identity (100% exact-id match), G3 engine↔HF logprob
parity (mean |Δ| < 0.25 nats, r > 0.95), G4 length/truncation, G5 degeneracy (≤3% per shard),
G6 seed freshness (zero duplicate seed hashes). The chat cell of record is 12 shards
(1,485,997 convs / 798.4M output tokens); pt is 13 shards (991,520 / 680.1M).

**Rollouts are the canonical artifact.** AR training pairs are ~174 KB each and deterministically
re-derivable, so they are never published — the published `data/rollouts/{chat,pt}/` shards +
`data/meta/splits_iolens.json` are.

## Stage 3 — Pair capture (AR training data)

`scripts/datagen/iolens_capture_pairs.py`: one invocation = one rollout shard, one GPU. Per
conversation, ONE forward over prompt+output; spans are carved disjointly ONLY inside the
generated region; **targets = the 17-layer residual at `prev_pos = span_start − 1`**
(`multilayer_v1` schema). Split routing is inherited from the seeds: `ar_train` rows →
`pairs_train`, `eval` rows → `pairs_eval`; **`ao_train`/`ao_val` rows are SKIPPED by design**
(the AO consumes AR reconstructions of pool text, never true activations; `--ao-val-as-train`
exists to capture the ao_val split when you need true-residual items, e.g. for distillation).

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/datagen/iolens_capture_pairs.py \
    --mode chat --rollout-shard 0 --out-dir ml_pairs_iolens_chat --train-frac 0.14
```

`--train-frac` is the disk valve: the FULL ar_train split is ~3–4 TB of pairs. Capture a
hash-disjoint wave, train on it, then take the next disjoint wave with `--train-frac-skip` —
or run the producer loop and never think about it:

```bash
# throttled producer + janitor: keeps the buffer between disk watermarks,
# sweeps consumed shards, never captures the same slice twice (persistent cursor)
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 uv run python scripts/datagen/iolens_produce_loop.py \
    --cells chat:6:0.14 --worlds chat:3 2>&1 | tee logs/produce_$(date +%Y%m%d_%H%M).log
```

Pairs are flushed to sub-shards with atomic tmp→rename writes; a storage preflight (G11)
refuses to start under 2× the projected volume. Periodically run gate **G9** — recorded
training examples must never exceed unique pairs ever captured (silent repeats flatten the
scaling curve; this happened, ~28% inflation, before the guards):

```bash
uv run python scripts/datagen/iolens_reconcile_counts.py --cells chat --strict
```

## Stage 4 — Whiteners

Per-layer mean+cov fitted from the pairs' own activations (gate G10: fit only once ≥1M
rows/layer exist):

```bash
uv run python scripts/datagen/iolens_fit_whitener.py \
    --pairs-dir ml_pairs_iolens_chat --out-prefix whitening_iolens_chat
```

Output: `$OLA_ROOT/whitening_iolens_chat_L{L}.safetensors`, one per layer.

**Whitener reuse rule: if you are training against the published lineage, download the
published whiteners (`data/whiteners/{chat,pt}/`) and NEVER refit.** A refit silently moves
the FVE basis and every number shifts — new runs stop being comparable to the published
curves. Fit fresh whiteners only for a genuinely new cell. FVE everywhere in this repo =
mean cos² in the whitened unit-norm basis, held-out, reported as percent.

## Stage 5 — AR training

`scripts/ar/iolens_ar_train.py`, single-node DDP via `mp.spawn`. The chat command of record
(exact final configuration; drop `--resume` for a fresh start — LR starts at 1e-3, the value
below is where the staleness policy had cut it by the end):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONUNBUFFERED=1 uv run python scripts/ar/iolens_ar_train.py \
    --run-name ar.chat.mlayer.lc.s0 --n-gpu 3 \
    --pairs-dir ml_pairs_iolens_chat --stream-dir ml_pairs_iolens_chat \
    --whitener-prefix whitening_iolens_chat \
    --micro-batch 32 --grad-accum 18 --expected-effective-batch 576 \
    --lr 1e-5 --lr-sched constant --warmup-steps 100 --max-steps 400000 \
    --drop-layers 0 --eval-every-steps 100 --save-every-steps 200 \
    --ckpt-samples-base 125000 --ckpt-samples-factor 1.4142135623730951 \
    --resume 2>&1 | tee logs/ar_chat_$(date +%Y%m%d_%H%M).log
```

Load-bearing details:

- **Effective batch 576 is an invariant** (`--expected-effective-batch` hard-fails on the
  `grad_accum // world` silent-change bug). Adjust `--micro-batch`/`--grad-accum` to your GPU
  count so `micro_batch × per_rank_accum × n_gpu = 576`; lr 1e-3 was tuned at 576.
- `--drop-layers 0`: layer 0 is untrainable noise at `span_start − 1`, so the layer embedding
  has 16 rows. Downstream consumers derive the layer set from `heads.pt`, never a constant.
- **Milestones land at √2-spaced sample counts as `ex<examples>/` directories**
  (`lora/`, `heads.pt`, `meta.json` with exact all-reduced `examples`/`tokens_span`). The
  checkpoint series of one constant-LR run IS the scaling curve; `ex` counts are cumulative
  and monotone across warm restarts (step numbers reset — never use them for naming).
- **LR staleness policy (measured):** constant LR converges to a noise floor, not the data's
  ceiling. Cut ~3× only when (a) no material gain at any horizon of the dense curve AND
  (b) several million *fresh* samples were consumed at the current LR. The first cut was
  transformative (+37% relative); later cuts +3–5%. A cut = kill, edit LR, relaunch with
  `--resume`.
- **Stop rule:** freeze when a full data doubling moves the tracked metrics ~+1%. That froze
  the chat AR at **`ex16014240`** (16.0M spans / 259.7M span tokens): val FVE mean **14.37%**,
  band L20–56 **15.36%**, L63 **28.20%**, ret@1 85.23%. The pt AR (`ex16013824`) reached
  8.32% — genuinely harder, not under-trained. Monitor `val_fve_*` on the milestone
  `meta.json`s, never train loss.

Checkpoints of record: `ckpts/ar/chat/mlayer.lc.s0/ex16014240` (chat FINAL, the frozen AR
every AO and the RL reward use) and `ckpts/ar/pt/mlayer.lc.s0/ex16013824` on
`agu18dec/local-workspace`. Always load via `oracle_lens.pipeline.ar_loader.load_lc_reconstructor`
(never bare PEFT — see failure_modes.md).

## Stage 6 — AO data: pool → arout → gate → frozen injection scale

The AO learns *activation → text*: the AR's reconstruction is scaled and injected as a soft
token (replacing a placeholder's embedding at one slot of the prompt, forward via
`inputs_embeds`); CE on the target tokens only. The AO consumes the **raw scaled** AR output —
no whitening on the injection path.

```bash
# 1) AO pool from the ao_train split of the rollouts (CPU; crop windows,
#    conversation-disjoint from AR, diversity-audited)
uv run python scripts/ao/ao_build_pool.py \
    --rollout-store-dir rollouts_iolens/chat --rollout-glob 'rollouts_*.safetensors' \
    --pairs-dir ml_pairs_iolens_chat --out-name pool_iolens --seed 1 --skip-fetch

# 2) AR reconstructions (arout): k=4 seeded layers per crop, layers >= 20, sharded over GPUs
for s in 0 1; do CUDA_VISIBLE_DEVICES=$s PYTHONUNBUFFERED=1 uv run python \
    scripts/ao/ao_precompute_cluster.py \
    --ar-run ar.chat.mlayer.lc.s0/ex16014240 --pool ao_pool/pool_iolens.safetensors \
    --split train --n-shards 2 --shard $s --layer-min 20 --layers-per-crop 4 & done; wait
# same with --split eval --pool ao_pool/eval_pool_iolens.safetensors for the FVE eval set

# 3) GATE: arout FVE must reproduce the AR's own quality (catches wrong-pool /
#    wrong-layer / inert-adapter bugs in one shot)
uv run python scripts/ao/ao_gate_arout.py \
    --ar-run ar.chat.mlayer.lc.s0/ex16014240 --whitener-dir ""
```

Pool hygiene (all enforced in code — do not disable): one crop per non-overlapping window,
one seeded length per window, dedup at span AND window level, AR-disjointness by hashing every
pairs-row prefix, conversation-granularity val split. `audit_diversity()` raises above a 2.5×
text-repeat factor at build and at trainer startup (the k=4 iolens runs override it explicitly
with `--max-repeat 4.0` — a deliberate, recorded trade). Arout shards are **self-describing**
(`ao_layers`, pool fingerprint, pick seeds); the trainer, gate, and dataset all read the layer
universe from shard metadata and refuse a fingerprint mismatch. Never bypass this.

**Injection scale: fit ONCE, then frozen forever.** `alpha` is the target slot norm; the scale
is `alpha / median ‖ar_out‖`. Val CE was insensitive across 64× of alpha (LayerNorm absorbs
magnitude), so the value is a comparability contract, not a tuning knob. Frozen values of
record at **α = 16000**:

| cell | injected vector | frozen scale |
|---|---|---|
| chat AO (AR reconstructions) | `scale × AR(span)` | **64.559** (`ao_runs/scale_iolens_chat_final.json`) |
| pt AO (pt-AR reconstructions) | `scale × AR_pt(span)` | **34.651** (pt norms ≈1.9× chat's — reusing 64.559 would inject ~2× hot) |
| GT AO (true residuals) | `scale × h` | **177.133** (true norms ~3× below AR outputs) |
| distill lineage onward | `16000 × h/‖h‖` (`transform=unit`) | scale-free — see Stage 8 |

Refit (with `--fit-scale`, then freeze) only when a NEW cell's median norm differs materially;
never refit within a lineage.

## Stage 7 — AO training

`scripts/ao/ao_train_cluster.py` (single-node DDP via `mp.spawn`; fast attention kernels are
checked at startup). Command of record (chat s0, 6 GPUs):

```bash
PYTHONUNBUFFERED=1 uv run python scripts/ao/ao_train_cluster.py \
    --run-name ao.iolens.chat.k4.L20plus.s0 --ar-run ar.chat.mlayer.lc.s0/ex16014240 \
    --pool ao_pool/pool_iolens.safetensors --eval-pool ao_pool/eval_pool_iolens.safetensors \
    --arout-dir ao_arout/ar.chat.mlayer.lc.s0/ex16014240 \
    --layers-per-crop 4 --max-repeat 4.0 \
    --scale-path ao_runs/scale_iolens_chat_final.json --alpha 16000 --prompt explain \
    --n-gpu 6 --micro-batch 128 --grad-accum 6 --lr 3e-4 --warmup-steps 100 \
    2>&1 | tee logs/ao_chat_$(date +%Y%m%d_%H%M).log
```

- **`--dry-run 1` first, before every launch.** It prints the full config, effective batch,
  example counts, and decoded prompt/target pairs, CPU-safe. (`--dry-run 0` is NOT a dry run —
  it means "zero dry-run rows", i.e. train.)
- **Effective batch 768 is the AO invariant.** The trainer floors per-rank accumulation to
  `grad_accum // n_gpu`, so the 6-GPU flags above (128×6) silently become eff **512** on
  4 GPUs. On 4 GPUs use `--micro-batch 64 --grad-accum 12` (= 64×3×4 = 768). Read the
  effective-batch line in the dry-run banner, always.
- **Monitor `val_epoch/val_ce`, never train loss and never `val_epoch/loss`** (the latter is
  a mis-swept train metric from the vendored harness; batches are length-pure so a single
  train-loss reading swings 0.33–3.06 with the crop length it drew). The k=4 repeat factor
  makes a mid-epoch val-CE inflection (memorization onset) the expected end-of-useful-training
  signal — every segment of the lineage was deliberately stopped at its argmin + a few rising
  validations. Checkpoints land as `stepN/`; `resume/` (written every validation) gives exact
  continuation with `--resume`.

**Extension segments (more data on one ruler).** More AO data = fresh crops of the same (or
new) rollouts, **exclusion-deduped** against every parent pool (`--exclude-pool`; the model
regenerates boilerplate across seeds, so conversation-disjointness alone is not enough). Then
warm-start from the parent's best-val-CE checkpoint and — critically — **keep the parent's
exact val set**:

```bash
PYTHONUNBUFFERED=1 uv run python scripts/ao/ao_train_cluster.py \
    ... --init-from ao.iolens.chat.k4.L20plus.s0/step<argmin> \
    --pool ao_pool/pool_iolens_ext1.safetensors --arout-dir ao_arout_ext1/<ar-run>/<rung> \
    --val-source pool --val-pool ao_pool/pool_iolens.safetensors \
    --val-arout-dir ao_arout/ar.chat.mlayer.lc.s0/ex16014240
```

`--val-pool` pins the identical val examples (same split/layer seeds), so the continued val-CE
curve stays on ONE ruler across segments — validating on the new pool's own held-out crops
would move the distribution mid-curve and fake a jump at the seam. Extension arout always goes
to an explicit fresh `--out-dir` (never mixed into the parent's arout dir), precomputed with
`--extension`.

**Lineage results (one ruler, chat cell):** s0 argmin 0.779 → s1 0.7177 → s2 0.7019. The
tag-free continuation reframe (`cont.u64`, prompt `continuation_raw`, raw span + EOS targets,
uniform 1–64-token crops, layer universe L20–60, ONE data pass at eff batch 1024) starts its
own curve: argmin 1.4203 (cont2), 1.4104 (ext.u64w2) — its `step28000`/`step28500` checkpoints
are the distillation teacher. Best checkpoints per family: the registry rows in the source
monorepo's `docs/project/checkpoints.md`; on HF: `ckpts/ao/{chat,gt,pt}/…` in
`agu18dec/local-workspace`.

**GT arm:** `scripts/ao/ao_precompute_gt.py` produces TRUE residuals at
`prev_pos = prompt_len + start − 1` for every pool crop, in arout shard format (same picks,
same metadata contract) — the shards drop into `--arout-dir` unchanged. Every crop is
ids-verified against the conversation slice; a single mismatch aborts the shard. Trained with
the frozen GT scale 177.133.

## Stage 8 — Distillation: the multi-bullet student

Turns the single-continuation u64 AO into a student that reads a TRUE residual and emits ~4
distinct `- ` bullet concepts (prompt `concepts_raw`, count-free). This is the RL warm start.

```bash
# 1) draw items + render prompts + write injected vectors (project venv)
uv run python scripts/distill/ao_distill_sample.py --mode prompts \
    --pairs-dir ml_pairs_aoval --out-dir distill_u64/pilot --n-rows 2046

# 2) teacher sampling (vLLM, SEPARATE venv you build yourself — prompt-embeds on a
#    merged teacher checkpoint; per-512-row resumable part writes)
<your-vllm-venv>/bin/python scripts/distill/ao_distill_sample.py \
    --mode sample --out-dir distill_u64/pilot --merged-dir <merged-teacher-dir> \
    --arm normmatched --n-shards 4 --shard 0

# 3) NNOMP selection vs the TRUE activation (shard over GPUs)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/distill/ao_gt_omp_readout.py \
    --out-dir distill_u64/pilot --arm normmatched --n-shards 4 --shard 0

# 4) assemble bullet-list targets -> distill_v1 shards (filter of record applied here)
uv run python scripts/distill/ao_assemble_distill.py \
    --out-dir distill_u64/pilot --arm normmatched --variant omp4

# 5) train the student (LoRA-only warm start from the u64 teacher)
PYTHONUNBUFFERED=1 uv run python scripts/distill/ao_distill_train_cluster.py \
    --out-dir distill_u64/pilot --variant omp4 --run-name ao.iolens.distill.omp4.s0 \
    --init-from ao.iolens.chat.k4.L20-60.cont.u64.s0/step28000 --dry-run 3
```

The recipe of record:

- **Teacher sampling: 64 samples × 100 tokens per item**, temp 1.0 / top_p 0.95, injected
  vectors = TRUE residuals from the held-out ao_val pairs, **norm-matched** (`α·v/‖v‖`,
  α=16000 — measured +47% read accuracy over the frozen scale on GT injection).
- **Selection: NNOMP k=4 + NNLS refit.** Candidates = prefix cuts (or, later rounds, the
  student's own parsed bullets — `--cand-mode bullets`, which beat prefixes on FVE at half the
  generation cost); atoms = unit rows of whitened AR embeddings; query = whitened `h`,
  un-normalized. Selection FVE ~0.31–0.34. The `--shrink` shortest-prefix rule was tried and
  **retired from the data path** (good compression, worse generation — the unshrunk "long"
  targets won on the fixed ruler).
- **The injection contract changes at the distill boundary — a regime change, not a detail.**
  The u64 parent injected `scaled` (64.559 × AR output); ALL distill students (and RL) inject
  **`unit`: `16000 × h/‖h‖` of the true residual**. `scales` is empty in these checkpoints'
  metas BY DESIGN. The warm start across the change was safe because the slot norm the model
  sees (~16000) is unchanged. Reward/readout code must inject the SAME way or it scores a
  different model.
- **LoRA-only warm start** from the u64 teacher (base frozen); training data grew across
  rounds r1 (775 rows) → r2 (5,256) → r3 (14,266) → final (29,564 rows, val_ce **1.6765**,
  `ckpts/ao/distill/final.s0/step105`, registry id `iolens.final`).
- **Never pair an input with a label anchored to information the input cannot contain**: an
  AR-image input with GT-anchored bullet targets collapsed 6× (0.020 vs 0.120) — that was
  label noise, not the manifold gap (self-consistent AR-everywhere training reaches ~92% of
  the GT lineage). GT-input training is the contract.
- **RAFT-1 baseline** (`scripts/distill/ao_raft_select.py`): 16 rollouts × 15,400 items, whole
  readouts scored by joint bullet-FVE (byte-identical to the RL reward), top half kept, one
  SFT epoch → **pass@1 0.1237 on the fixed 220-item ruler**. This is the bar the RL stage had
  to beat (it did: 0.155; see [rl.md](rl.md)). Anchors: final student pass@1 0.1202 /
  pass@16 0.199; literal-true-span 0.178; best-4-of-pool ~0.34.

---

## Training from published artifacts vs from scratch

You almost never need Stages 1–2 (and often not 3–4 either). Everything a new run needs is
under `data/` in `agu18dec/local-workspace`:

```bash
# rollouts (canonical), whiteners (REUSE — never refit), split manifest
mkdir -p $OLA_ROOT/rollouts_iolens/chat
hf download agu18dec/local-workspace --repo-type dataset \
    --include 'data/rollouts/chat/*' --local-dir /tmp/dl && \
    mv /tmp/dl/data/rollouts/chat/* $OLA_ROOT/rollouts_iolens/chat/
hf download agu18dec/local-workspace --repo-type dataset \
    --include 'data/whiteners/chat/*' 'data/meta/splits_iolens.json' --local-dir /tmp/dl && \
    mv /tmp/dl/data/whiteners/chat/* $OLA_ROOT/ && mv /tmp/dl/data/meta/splits_iolens.json $OLA_ROOT/
```

The capture (Stage 3) is fully seeded and reproduces the program's pairs bit-identically from
the published rollouts + split manifest. Also published: AO pools (`data/ao/pool/`), arout
(`data/ao/arout/…`), frozen scales/meta (`data/meta/`, `ao/runs/`), and every checkpoint
family (`ckpts/ar/…`, `ckpts/ao/…`, `ckpts/ao/rl/iolens.final.ddp600.s0`) — so you can start
at any stage: AO training needs only pool + arout + the frozen AR rung; distillation needs
only the u64 teacher + ao_val pairs; RL needs only what `scripts/rl/fetch_artifacts.py`
downloads. Three rules keep a new run comparable to the published curves: reuse the published
whiteners, keep the effective-batch invariants (AR 576 / AO 768), and keep `--drop-layers 0`.
