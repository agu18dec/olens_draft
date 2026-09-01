# scripts/rl — self-contained AO GRPO

On-policy GRPO for the activation oracle against the frozen AR reward: the sampler IS the
trainer (one optimizer step per rollout batch ⇒ importance ratio ≡ 1; no engine, no weight
sync). **Full documentation: [docs/rl.md](../../docs/rl.md)** — design, injection/reward
contracts, data prep, gate ladder, results of record, and caveats.

Quickstart (details and prerequisites in docs/rl.md):

```bash
export OLA_ROOT=artifacts/sc SC=artifacts/sc
uv run python scripts/rl/fetch_artifacts.py --sc-root $SC        # base 27B + ckpts + banks
# GT bank + sidecar + parquets: docs/rl.md §"Data prep"
GATE_PQ=$SC/rl_u64/rl_gate_0.parquet bash scripts/rl/checks/run_gates.sh   # MUST pass first
bash scripts/rl/run_validation.sh                                # 50-step 2-GPU shakeout
STEPS=600 RUN=my-ddp BP=14 bash scripts/rl/run_ddp.sh            # 8-GPU: 7 actors + AR on GPU 7
```

Re-run the gate ladder after ANY change to injection, reward, data, or the update math.
