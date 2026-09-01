"""Build the AO-ladder training pool + eval pool (CPU-only, deterministic, one-time).

Steps (design: docs/project/experiments/ola/ao_ladder.md):
1. Fetch the full chat rollouts (`agu18dec/qwen3.6-27b-onpolicy-rollouts` chat/, ~1 GB JSON)
   into ``$OLA_ROOT/rollouts_full/chat`` (idempotent).
2. Hash every pairs-row prefix at the 6 crop lengths from ``$OLA_ROOT/ml_pairs_onpolicy_chat``
   (the config-free AR-disjointness set: every AR training crop is a pairs-row prefix).
3. Sample ~1M start positions (seeded), build nested {2,4,8,16,32,64}-token crops, dedup within
   each length, drop AR-prefix collisions / special-token / placeholder windows.
4. Save ``$OLA_ROOT/ao_pool/pool_v1.safetensors`` (+ ``eval_pool_v1`` from the pairs' first-4096
   row carve) and print the full accounting.

    # long job — run in tmux, tee to logs/:
        uv run python scripts/ao/ao_build_pool.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"
ROLLOUTS_REPO = "agu18dec/qwen3.6-27b-onpolicy-rollouts"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-starts", type=int, default=1_050_000)
    ap.add_argument("--max-per-conv", type=int, default=8)
    ap.add_argument("--all-lengths", action="store_true", help="keep all 6 nested crops (unsafe)")
    ap.add_argument("--overlapping", action="store_true", help="allow overlapping windows (unsafe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs-dir", type=str, default="ml_pairs_onpolicy_chat")
    ap.add_argument("--n-eval-rows", type=int, default=4096)
    ap.add_argument(
        "--lengths",
        type=str,
        default="2,4,8,16,32,64",
        help="csv of crop lengths; the pool records them (self-describing) and every consumer "
        "reads them back from the artifact. Long-emission pools (Agam 2026-08-04) go beyond 64 "
        "— window size = max(lengths), so conversations must have that many output tokens.",
    )
    ap.add_argument("--out-name", type=str, default="pool_v1")
    ap.add_argument(
        "--rollout-glob",
        type=str,
        default="rollouts_*.json",
        help="restrict source files, e.g. 'rollouts_gen2_*.json' for the delta pool",
    )
    ap.add_argument(
        "--exclude-pool",
        type=str,
        default="",
        help="csv of existing pools whose kept crops are also excluded (delta builds: pass "
        "every prior segment's pool so the extension run never re-sees trained text — the "
        "model regenerates boilerplate across different seeds, so disjoint conversations "
        "alone do not guarantee disjoint crops). Lengths the parent lacks are unaffected.",
    )
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument(
        "--splits",
        type=str,
        default="ao_train",
        help="csv of rollout-store splits to crop from (default ao_train). The u64 scaling run "
        "(Agam 2026-08-08) adds ar_train — overlap with AR training text is accepted there in "
        "exchange for ~3.5x the token budget; eval/ao_val stay reserved.",
    )
    ap.add_argument(
        "--no-ar-exclude",
        action="store_true",
        help="skip the AR-prefix exclusion (pairs_prefix_hashes). Pairs are still required for "
        "the eval carve. Use with --splits including ar_train, where the exclusion is moot and "
        "its cost at wide length ladders is hours.",
    )
    ap.add_argument(
        "--rollout-store-dir",
        type=str,
        default="",
        help="iolens path: read rollout_store safetensors shards from this OLA_ROOT-relative dir"
        " and use ONLY the ao_train split's conversations (splits were assigned before"
        " generation). Replaces the JSON fetch/load; --rollout-glob then matches"
        " rollouts_*.safetensors.",
    )
    ap.add_argument(
        "--exclude-seed-hash-file",
        type=str,
        default="",
        help="OLA_ROOT-relative torch file holding an int64 tensor of rollout seed_hash values "
        "to DROP (rollout-store path only). Use case: a wave-2 re-roll shares seed prompts "
        "with wave 1, so conversations whose seeds back the frozen val ruler must never "
        "enter a training pool, whatever text the re-roll produced.",
    )
    args = ap.parse_args()
    lengths = tuple(sorted(int(n) for n in args.lengths.split(",")))
    assert len(lengths) == len(set(lengths)) and all(n > 0 for n in lengths)

    from transformers import AutoTokenizer

    from oracle_lens.pipeline.ao_pool import (
        AOPool,
        audit_diversity,
        build_ao_pool,
        build_eval_pool,
        crop_hashes,
        load_chat_rollouts,
        pairs_prefix_hashes,
        pool_fingerprint,
        sample_starts,
    )
    from oracle_lens.pipeline.multilayer import load_multilayer_shards_lazy

    root = ola_root()
    roll_dir = root / "rollouts_full" / "chat"
    if not args.skip_fetch and not args.rollout_store_dir:
        from huggingface_hub import snapshot_download

        print(f"[ao-pool] fetching {ROLLOUTS_REPO}/chat -> {roll_dir}", flush=True)
        snapshot_download(
            ROLLOUTS_REPO,
            repo_type="dataset",
            local_dir=str(root / "rollouts_full"),
            allow_patterns=["chat/*"],
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    marker_ids = (
        tokenizer.convert_tokens_to_ids(t)
        for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    )
    special_ids = frozenset(int(i) for i in tokenizer.all_special_ids) | frozenset(
        int(i) for i in marker_ids if isinstance(i, int)
    )
    print(f"[ao-pool] {len(special_ids)} special ids filtered", flush=True)

    pairs_paths = sorted((root / args.pairs_dir).rglob("*.safetensors"))
    pairs, meta = load_multilayer_shards_lazy(pairs_paths)
    print(f"[ao-pool] pairs: {len(pairs)} rows from {len(pairs_paths)} shards", flush=True)
    if args.no_ar_exclude:
        exclude: dict[int, set[bytes]] = {n: set() for n in lengths}
        print("[ao-pool] AR-prefix exclusion SKIPPED (--no-ar-exclude)", flush=True)
    else:
        exclude = pairs_prefix_hashes(pairs.span_ids, pairs.offsets, lengths=lengths)
        for n, s in exclude.items():
            print(f"[ao-pool] AR-prefix hashes N={n}: {len(s):,}", flush=True)

    for prev_path in (p for p in args.exclude_pool.split(",") if p.strip()):
        prev = AOPool.load(root / prev_path.strip())
        prev_hashes = crop_hashes(prev)  # keyed by the PARENT pool's own lengths
        for n in exclude:
            before = len(exclude[n])
            exclude[n] |= prev_hashes.get(n, set())
            print(
                f"[ao-pool] +{len(exclude[n]) - before:,} crop hashes N={n} "
                f"from {prev_path.strip()}",
                flush=True,
            )

    if args.rollout_store_dir:
        from oracle_lens.pipeline.rollout_store import SPLITS, load_rollout_shards

        store_dir = root / args.rollout_store_dir
        glob = args.rollout_glob
        if glob == "rollouts_*.json":  # the JSON default doesn't match safetensors shards
            glob = "rollouts_*.safetensors"
        shard_paths = sorted(store_dir.glob(glob))
        rolls = load_rollout_shards(shard_paths)
        want_splits = {SPLITS.index(s.strip()) for s in args.splits.split(",") if s.strip()}
        held_out = {SPLITS.index("eval"), SPLITS.index("ao_val")}
        assert not (want_splits & held_out), "--splits must never include eval/ao_val (leak)"
        # same content policy as capture (Agam 2026-08-01): whole conversations with a
        # re-emitted <think> in the output are excluded from the AO pool too
        think_id = tokenizer.convert_tokens_to_ids("<think>")
        drop_hashes: set[int] = set()
        if args.exclude_seed_hash_file:
            import torch as _t

            drop_hashes = {
                int(h) for h in _t.load(root / args.exclude_seed_hash_file, weights_only=True)
            }
            print(f"[ao-pool] seed-hash exclusion list: {len(drop_hashes)} hashes", flush=True)
        n_think = 0
        n_seed_dropped = 0
        outputs = []
        for i in range(len(rolls)):
            if int(rolls.split_id[i]) not in want_splits:
                continue
            if drop_hashes and int(rolls.seed_hash[i]) in drop_hashes:
                n_seed_dropped += 1
                continue
            out = rolls.output_ids(i).tolist()
            if isinstance(think_id, int) and think_id in out:
                n_think += 1
                continue
            outputs.append(out)
        print(f"[ao-pool] think-reemission convs excluded: {n_think}", flush=True)
        if drop_hashes:
            print(f"[ao-pool] seed-hash-excluded convs dropped: {n_seed_dropped}", flush=True)
        files = [p.name for p in shard_paths]
        print(
            f"[ao-pool] rollout store {store_dir}: {len(rolls):,} convs, "
            f"{len(outputs):,} in splits [{args.splits}]",
            flush=True,
        )
    else:
        outputs, files = load_chat_rollouts(roll_dir, glob=args.rollout_glob)
    n_tokens = sum(len(o) for o in outputs)
    print(f"[ao-pool] rollouts: {len(outputs):,} convs / {n_tokens:,} tokens", flush=True)
    conv, start = sample_starts(
        outputs,
        n_starts=args.n_starts,
        seed=args.seed,
        max_per_conv=args.max_per_conv,
        window=max(lengths),
        non_overlapping=not args.overlapping,
    )
    print(f"[ao-pool] sampled {len(conv):,} starts", flush=True)

    pool = build_ao_pool(
        outputs,
        conv,
        start,
        exclude=exclude,
        special_ids=special_ids,
        tokenizer=tokenizer,
        seed=args.seed,
        all_lengths=args.all_lengths,
        lengths=lengths,
        provenance={
            "rollout_files": files,
            "rollout_glob": args.rollout_glob,
            "exclude_pool": args.exclude_pool,
            "rollouts_repo": ROLLOUTS_REPO,
            "pairs_dir": args.pairs_dir,
            "n_pairs_rows": len(pairs),
            "max_per_conv": args.max_per_conv,
            # seed/n_starts/splits were historically NOT recorded — recovering the u64 pool's
            # seed (2026-08-15) took a 10-seed rebuild sweep against the runbook's crop counts.
            # Record everything a byte-identical rebuild needs.
            "seed": args.seed,
            "n_starts": args.n_starts,
            "n_starts_sampled": len(conv),
            "splits": args.splits,
            "no_ar_exclude": args.no_ar_exclude,
            "lengths": list(lengths),
        },
    )
    out = root / "ao_pool" / f"{args.out_name}.safetensors"
    pool.save(out)
    fp = pool_fingerprint(pool.ids, pool.keep)
    print(f"[ao-pool] saved {out} fingerprint={fp}", flush=True)
    stats = pool.meta["stats"]
    total_tokens = 0
    for n in lengths:
        kept = stats[f"n_kept_N{n}"]
        total_tokens += kept * n
        print(
            f"[ao-pool] N={n:<3d} kept {kept:>9,}  dup {stats[f'n_dup_N{n}']:>7,}  "
            f"AR-excluded {stats[f'n_excluded_N{n}']:>7,}",
            flush=True,
        )
    print(
        f"[ao-pool] TOTAL crops {pool.n_crops():,}  span tokens {total_tokens:,} "
        f"(x17 layers = {17 * total_tokens:,} trainable span-token-examples)",
        flush=True,
    )

    for k in (1, 2, 4):
        try:
            print(f"[ao-pool] diversity audit @ k={k}: {audit_diversity(pool, layers_per_crop=k)}")
        except ValueError as e:
            print(f"[ao-pool] diversity audit @ k={k}: {e}")

    eval_pool = build_eval_pool(
        pairs.span_ids, pairs.offsets, n_rows=args.n_eval_rows, lengths=lengths
    )
    eval_out = root / "ao_pool" / f"eval_{args.out_name}.safetensors"
    eval_pool.save(eval_out)
    efp = pool_fingerprint(eval_pool.ids, eval_pool.keep)
    print(
        f"[ao-pool] eval pool {eval_out} rows={len(eval_pool.ids)} "
        f"crops={eval_pool.n_crops():,} fingerprint={efp}",
        flush=True,
    )
    # §7 standing rule (the 2026-08-04 ext2 empty-carve bug): accept an eval carve only after
    # seeing its per-length kept histogram — a zero at any length means the source is wrong.
    eval_hist = {n: int(eval_pool.keep[:, j].sum()) for j, n in enumerate(eval_pool.lengths)}
    print(f"[ao-pool] eval pool kept-per-length: {eval_hist}", flush=True)
    if any(v == 0 for v in eval_hist.values()):
        print(
            "[ao-pool] WARNING: eval carve has ZERO kept rows at some length — "
            "wrong/short-window source?",
            flush=True,
        )
    # loud sanity: the eval pool rows must be the pairs' leading rows (true-h lookup contract)
    assert isinstance(eval_pool, AOPool) and int(eval_pool.conv[0]) == 0
    print(f"[ao-pool] dataset meta: {meta}", flush=True)


if __name__ == "__main__":
    main()
