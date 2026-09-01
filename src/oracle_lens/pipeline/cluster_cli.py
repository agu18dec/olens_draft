"""Slurm entrypoint for Inverted OLens jobs: the same bodies the Modal app calls, argv-dispatched.

Modal owns argv via ``@app.local_entrypoint``; Slurm owns it via argparse — one body, two
launchers. Run from an sbatch/srun context only (the bodies expect a CUDA device):

    scripts/cluster/submit.sh -J pton-dump -g 1 -a 0-15%8 -m 160G -t 06:00:00 -- \\
        uv run --no-sync python -m oracle_lens.pipeline.cluster_cli dump-onpolicy \\
        --mode pt --n-shards 16 --rollouts-dir $OLA_ROOT/rollouts/pt

Array tasks read ``SLURM_ARRAY_TASK_ID`` as the shard index unless ``--shard`` is given.
"""

import argparse
import json
import os
from pathlib import Path

from oracle_lens.hf_offline import hf_offline
from oracle_lens.pipeline.paths import ola_root


def _shard_from_env(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise SystemExit("no --shard given and SLURM_ARRAY_TASK_ID is unset")
    return int(env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ola-cluster")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump-onpolicy", help="one shard of the on-policy multi-layer dump")
    d.add_argument("--mode", choices=["chat", "pt"], required=True)
    d.add_argument("--n-shards", type=int, required=True)
    d.add_argument("--shard", type=int, default=None, help="default: SLURM_ARRAY_TASK_ID")
    d.add_argument("--rollouts-dir", type=Path, default=None,
                   help="default: $OLA_ROOT/onpolicy/<mode>")
    d.add_argument("--out-dir", default="")
    d.add_argument("--max-per-conv", type=int, default=16)
    d.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("gen-onpolicy", help="one shard of on-policy rollout generation (vLLM)")
    g.add_argument("--mode", choices=["chat", "pt"], required=True)
    g.add_argument("--n-shards", type=int, required=True)
    g.add_argument("--shard", type=int, default=None, help="default: SLURM_ARRAY_TASK_ID")
    g.add_argument("--n-per-shard", type=int, default=1250)
    g.add_argument("--max-new", type=int, default=512)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed-offset", type=int, default=0, help="drop N raw stream rows (top-ups)")
    g.add_argument("--out-tag", default="")
    g.add_argument("--gpu-mem-util", type=float, default=0.92)

    t = sub.add_parser("train-ml", help="multi-layer AR training run (v2 protocol)")
    t.add_argument("--run-name", required=True)
    t.add_argument("--pairs-dir", required=True)
    t.add_argument("--n-gpu", type=int, default=8)
    t.add_argument("--n-pairs", type=int, default=0, help="0 = all (nested prefix otherwise)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--micro-batch", type=int, default=32)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-sched", default="constant", help="constant | cosine (final_lr=0)")
    t.add_argument("--bucketed", type=int, default=1,
                   help="1: length-grouped fixed-row batches (edge padding, static shapes)")
    t.add_argument("--compile-blocks", type=int, default=1,
                   help="1: per-block torch.compile (needs bucketed for static shapes)")
    t.add_argument("--save-every-steps", type=int, default=250,
                   help="resume-blob cadence in optimizer steps; 0 disables save/resume")
    t.add_argument("--grad-ckpt", type=int, default=1,
                   help="ON: measured 2026-07-26 that no-ckpt OOMs a 140GiB H200 even at "
                        "mb16/pad256 (~20MB/token of uncheckpointed intermediates across 64 "
                        "blocks, MLP width 17408). Not a tuning preference - a hard wall.")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--eval-every-steps", type=int, default=25)
    t.add_argument("--whiteners-dir", type=Path, default=None,
                   help="default: $OLA_ROOT (v1 whiteners); ladder uses $OLA_ROOT/whitening_v2")
    t.add_argument("--expected-effective-batch", type=int, default=256,
                   help="protocol assertion; job refuses to start on mismatch. 0 disables")
    t.add_argument("--crop-max", type=int, default=0,
                   help=">0: short-span regime — stratified-uniform prefix crops with N exactly "
                        "uniform in {1..crop_max}; n_pairs then counts crops (rounded to a "
                        "multiple of crop_max) and rungs nest as pool prefixes")
    t.add_argument("--warmup-steps", type=int, default=20,
                   help="constant-lr warmup cap (actual = min(this, steps//10))")
    t.add_argument("--loss-ridge-c", type=float, default=0.0,
                   help=">0: whiten the TRAINING loss with this ridge (val metrics keep 0.1)")
    t.add_argument("--loss-space", default="whiten", choices=["whiten", "jspace", "mixed"],
                   help="jspace: TRAINING loss in the Jacobian-lens space z=(x-μ)@Jᵀ (pure "
                        "J-cosine, no whitening; J-covered layers only, L63 excluded). Val "
                        "keeps all whitened metrics and adds val_jfve_*/val_jloss")
    t.add_argument("--loss-mix-lambda", type=float, default=0.5,
                   help="loss-space=mixed: (1-lam)*whitened + lam*J. Same 16 J-covered layers "
                        "as the pure-J arm (L63 excluded) so the two are directly comparable")
    t.add_argument("--jspace-repo", default="neuronpedia/jacobian-lens",
                   help="HF repo of the JacobianLens artifact (loss-space=jspace)")
    t.add_argument("--jspace-file", default="qwen3.6-27b/jlens/Salesforce-wikitext/"
                                            "Qwen3.6-27B_jacobian_lens_n1000.pt",
                   help="file inside the repo (loss-space=jspace)")
    t.add_argument("--jspace-revision", default="",
                   help="optional HF revision pin; resolved snapshot is recorded either way")

    r = sub.add_parser("refit-whiteners", help="fit v2 per-layer whiteners from train rows")
    r.add_argument("--pairs-dir", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--max-rows", type=int, default=400_000)

    args = parser.parse_args(argv)
    # gen-onpolicy STREAMS its seed corpus (lmsys / fineweb-edu) from the Hub, so it cannot run
    # offline; every other stage reads only the pinned, fully-cached model snapshot.
    if args.cmd != "gen-onpolicy":
        hf_offline()

    if args.cmd == "refit-whiteners":
        from oracle_lens.pipeline.jobs.refit import refit_whiteners

        out = refit_whiteners(root=ola_root(), pairs_dir=args.pairs_dir,
                              out_dir=args.out_dir, max_rows=args.max_rows)
        print(json.dumps(out), flush=True)
        return 0

    if args.cmd == "train-ml":
        from oracle_lens.pipeline.jobs.train import train_ml

        config = {
            "run_name": args.run_name,
            "n_pairs": args.n_pairs,
            "seed": args.seed,
            "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "lr_sched": args.lr_sched,
            "epochs": args.epochs,
            "eval_every_steps": args.eval_every_steps,
            "head_mode": "layer_conditioned",
            "bucket_by_length": bool(args.bucketed),
            # The TokenBudgetSampler batches by max_batch_rows, NOT micro_batch — its default of
            # 8 would quietly shrink a mb=128 run to eff-batch 32. Keep them equal (the config's
            # own comment says "= micro_batch") so --expected-effective-batch means what it says.
            "max_batch_rows": args.micro_batch,
            "compile_blocks": bool(args.compile_blocks),
            "save_every_steps": args.save_every_steps,
            "grad_checkpointing": bool(args.grad_ckpt),
            "crop_max": args.crop_max,
            "warmup_steps": args.warmup_steps,
            "loss_ridge_c": args.loss_ridge_c,
            "loss_space": args.loss_space,
            "loss_mix_lambda": args.loss_mix_lambda,
            "jspace_repo": args.jspace_repo,
            "jspace_file": args.jspace_file,
            "jspace_revision": args.jspace_revision,
        }
        summary = train_ml(
            json.dumps(config),
            n_gpu=args.n_gpu,
            pairs_dir=args.pairs_dir,
            root=ola_root(),
            whiteners_dir=args.whiteners_dir,
            expected_effective_batch=args.expected_effective_batch or None,
        )
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    if args.cmd == "dump-onpolicy":
        from oracle_lens.pipeline.jobs.dump import dump_onpolicy_shard

        result = dump_onpolicy_shard(
            _shard_from_env(args.shard),
            args.n_shards,
            root=ola_root(),
            mode=args.mode,
            rollouts_dir=args.rollouts_dir,
            out_dir=args.out_dir,
            max_per_conv=args.max_per_conv,
            seed=args.seed,
        )
        print(json.dumps(result), flush=True)
    if args.cmd == "gen-onpolicy":
        from oracle_lens.pipeline.jobs.gen import gen_onpolicy

        result = gen_onpolicy(
            root=ola_root(),
            mode=args.mode,
            shard=_shard_from_env(args.shard),
            n_shards=args.n_shards,
            n_per_shard=args.n_per_shard,
            max_new=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            seed_offset=args.seed_offset,
            out_tag=args.out_tag,
            gpu_memory_utilization=args.gpu_mem_util,
        )
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
