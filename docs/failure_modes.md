# Failure modes already paid for — don't rediscover

Every item here cost a real run, a real day, or a real conclusion. Read before training
anything; the guards mentioned are in the code, but knowing *why* they exist is what keeps you
from routing around them.

## From the original runbook

1. **The 17-layer assumption.** The AR trains 16 layers (layer 0 dropped), the AO sees 12 (or
   11 in the L20–60 universe). Anything that hardcodes `len(LAYERS)` breaks — it did, in six
   places at once (loader crash, CUDA scatter assert, and worst: the AO being *told* "layer 0"
   while injected with layer 20). Layer semantics come from `heads.pt` / arout shard metadata,
   nowhere else. See the dedicated section below.
2. **Buffered logs look like hangs.** Run every long job with `PYTHONUNBUFFERED=1` (and tee to
   a timestamped `logs/` file). A block-buffered healthy job is indistinguishable from a dead
   one until the buffer flushes.
3. **tqdm parsing must anchor on "Training Epoch".** The validation progress bar has the same
   shape and has mis-parsed four separate monitors into false stall alerts.
4. **A killed capture leaves partial sub-shards.** Wait out in-flight captures
   (`iolens_capture_pairs.py` / the produce loop); never kill them mid-flush. Writes are
   atomic tmp→rename, but a kill between sub-shards still costs the in-memory batch.
5. **Streamed pairs must be duplicate-guarded.** Capture ledger + per-worker seen set + the G9
   reconciliation (`scripts/datagen/iolens_reconcile_counts.py`). Repeats silently flatten the
   scaling curve — measured at ~28%/~21% inflation before the guards existed.
6. **One server per port/app.** Leftover SGLang/vLLM servers from a previous session fight the
   new one. Kill leftover children by process (`pkill -f`), not just by tmux session — killing
   the tmux pane leaves the engine processes holding GPU memory (check
   `nvidia-smi --query-compute-apps` and kill by PID).
7. **Injection sweeps must compare at matched validation indices.** Arm A at step 600 vs arm B
   at step 200 once produced a fake "monotone with scale" conclusion.
8. **Secrets:** env-var only (`HF_TOKEN=… uv run …`), `chmod 600`, never argv, never committed;
   rotate anything that lands in chat or logs.

## Layer semantics come from `heads.pt` — never hardcode 17

The layer universe is data: the AR's `heads.pt` (`layer_emb` row count) and the arout shards'
`ao_layers` metadata are the only sources of truth. The chat AR of record trains 16 layers
(`--drop-layers 0`), the `L20plus` AOs see 12, the u64/distill/RL lineage sees 11 (L20–60).
Every consumer in this repo derives the set from the artifact it loads; new code must too. A
wrong assumption here is not a crash — it can silently mislabel which layer a vector came from.

## Effective-batch invariants: AR 576, AO 768

Both trainers were tuned at a fixed effective batch and their published curves are only
comparable at it.

- AR: `--expected-effective-batch 576` hard-fails a mismatch — keep the flag, don't "fix" the
  assert.
- AO: **the trainer floors per-rank accumulation to `grad_accum // n_gpu`**, so the 6-GPU
  command of record (`--micro-batch 128 --grad-accum 6`) silently becomes effective batch
  **512** on 4 GPUs. The right 4-GPU shape is `--micro-batch 64 --grad-accum 12` (= 768).
  The dry-run banner prints the effective batch — read it before every launch.

## `ex<examples>` milestone naming — cumulative, never step numbers

AR milestones are named by the exact all-reduced cumulative example count (`ex16014240`),
which is monotone across warm restarts. Step numbers reset on every restart and mean nothing
across segments (AO checkpoint dirs use `stepN`, but note optimizer steps ≠ dir numbers when
per-rank accumulation changes: step3002 = opt 1000 on a 4-GPU run). When recording or
comparing rungs, use the `examples`/`tokens_span` counters in `meta.json`, never a step.

## LazyTargets stays mmap-backed

The 17-layers-per-pair target tensors must be memory-mapped (`LazyTargets`), not loaded
resident. Loading them resident works at small scale and **SIGSEGVs at >RAM rungs** — the
failure appears only after you've scaled up, hours into a run. Don't "simplify" the lazy
loading away.

## Injection scales are frozen once — never refit

α = 16000 everywhere; chat scale 64.559, pt 34.651, GT 177.133, and the distill/RL lineage is
scale-free (`unit`: `16000 × h/‖h‖`, `scales` legitimately empty in the meta). Val CE is
insensitive to the scale across 64× — the value is a *comparability contract* between
training, generation, readouts, and reward. Refitting mid-lineage (or reusing another cell's
scale: the chat scale on pt vectors injects ~2× hot, on GT vectors ~2.7× cold — measured −47%
read accuracy) silently scores a different model. Read each checkpoint's own meta via
`InjectSpec`-style code rather than assuming a scale.

## `--dry-run 1` first (`--dry-run 0` is NOT a dry run)

The flag is a row count: `--dry-run N` prints config + N decoded input/output pairs and exits;
`--dry-run 0` means zero dry-run rows, i.e. **train**. Before every AO/distill launch, run
`--dry-run 1` (or more) and read: effective batch, example/val counts, arout fingerprint
acceptance, and the decoded pairs (layer named == layer injected, sane `|scale·vec|`). The
dry-run gate has caught wrong `--eval-arout-dir`s and batch-geometry drift that nothing else
would have surfaced before hours of training.

## Monitor `val_epoch/val_ce`, never train loss (and never `val_epoch/loss`)

`val_epoch/loss` is NOT validation loss: the vendored harness's log accumulator sweeps pending
*train*-step metrics in under the `val_epoch/` prefix (give-away: `span_tokens_per_sec_rank`
appears beside it, which is only logged from `training_step`). And per-batch train loss is
meaningless here — batches are length-pure, so one reading swings 0.33–3.06 purely with the
crop length drawn. Read `val_epoch/val_ce` (+ `val_ce_L{layer}` / `val_ce_N{len}` breakdowns,
`train_ce`, `train_val_gap`). For the k=4 pools, the thing to watch is the **val-CE
inflection** (memorization onset), not the floor.

## Always load adapters via `oracle_lens.pipeline.ar_loader`

Use `load_lc_reconstructor` (AR) / `load_ao_adapter` (AO). These runs train with
`compile_blocks=on`, so every saved adapter tensor key carries the torch.compile
`_orig_mod.` prefix. Bare `PeftModel.from_pretrained` on an uncompiled base matches **none**
of them and raises nothing: at inference there is no loss to look wrong, so an inert LoRA is
indistinguishable from a working one — the model then scores at chance and perfectly mimics
an *interesting* negative result (it cost a probe run and 1h45 of FVE eval on what turned out
to be the bare base model). The loaders compile-wrap first and **error unless ≥90% of saved
LoRA tensors map onto the model** ("adapter verified: N/M tensors live"). The RL stack's
`fetch_artifacts.py` instead writes a stripped `lora_hf/` copy once, at fetch time — same
invariant, other direction.

## Read a checkpoint only with its own contract

Prompt kind (`explain` / `continuation_raw` / `concepts_raw`), transform (`scaled` / `unit`),
alpha, and layer universe are part of a checkpoint's identity — they live in its `meta.json`.
Reading a checkpoint under another checkpoint's contract (wrong prompt wording, wrong
transform, wrong scale, a layer outside its universe) is **a different experiment, not a
noisier one**: the numbers you get are internally consistent and quietly about the wrong
thing. Derive every injection/readout parameter from the checkpoint's own meta.

## Val-set starvation and the one-ruler rule

Tiny eval splits starve the (layer,length)-pure val sampler — a val batch size larger than the
smallest group drops every partial batch and validation silently never fires (guarded now: the
trainer refuses a zero-batch val loader; `val_token_budget` exists for small distill splits —
still check for "0it" val logs). And when extending a run on new data, keep the **parent's
exact val set** (`--val-source pool --val-pool <parent pool>`): validating on the new pool's
own carve moves the distribution mid-curve and fakes a jump at the seam. Accept a new eval
carve only after printing its per-length kept histogram — a rebuilt carve once had zero rows
above N=32 and every long-emission number would have been fiction.
