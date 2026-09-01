"""iolens S6: 17-layer activation capture over fresh rollout shards (AR training data).

One task = one rollout shard (``rollouts_{N:04d}.safetensors`` from ``iolens_rollout_gen.py``).
Per conversation: ONE forward over prompt+output, spans carved ONLY inside the generated region
with ``carve_disjoint_spans`` (mutually disjoint — the 2026-07-29 redundancy fix; ~5x cheaper
per span token than the loguniform walk), plus an optional raw log-uniform slice of
conversations (``--raw-slice-frac``) for backward comparability. Targets = the 17-layer residual
at ``prev_pos = span_start - 1``, ``multilayer_v1`` schema.

Split routing (assigned at seed prep, BEFORE generation): ``ar_train`` rows -> ``pairs_train``,
``eval`` rows -> ``pairs_eval`` (true-h FVE pool); ``ao_train``/``ao_val`` rows are SKIPPED —
the AO consumes AR reconstructions of pool text, never true activations.

Memory/disk discipline: pairs are flushed to sub-shards of ``--flush-pairs`` rows (atomic
tmp->rename; the accumulate-then-cat dumper OOM is a paid-for lesson), and a storage preflight
(G11) extrapolates the pair volume from the first conversations and refuses to start under 2x
the projection. A per-shard ``done`` marker with exact counts makes reruns idempotent.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/datagen/iolens_capture_pairs.py \
        --mode chat --rollout-shard 0 --out-dir ml_pairs_iolens_chat
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"
MIN_RENDERED_TOKENS = 16
T_MAX = 2048
PAIR_BYTES = 17 * 5120 * 2  # bf16 target stack — the disk term that matters


def is_chunk_repetitive(text: str, *, window: int = 50, max_frac: float = 0.3) -> bool:
    """Char-level near-loop detector (catches loops the token-20-gram pack check misses,
    e.g. combining-mark symbol soup). Measured hit-rate ~0.01% chat / 0.00% pt."""
    if len(text) <= 4 * window:
        return False
    chunks = [text[k : k + window] for k in range(0, len(text) - window, window)]
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c] = counts.get(c, 0) + 1
    return max(counts.values()) / len(chunks) > max_frac


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path.cwd()
        ).stdout.strip()
        return out or "unknown"
    except OSError:  # pragma: no cover
        return "unknown"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def _hf_offline() -> None:
    if os.environ.get("AO_HF_ONLINE") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class SubShardWriter:
    """Accumulate pair rows and flush every ``flush_pairs`` to an atomic sub-shard."""

    def __init__(
        self, pairs_dir: Path, split: str, shard: int, flush_pairs: int, meta_kw: dict[str, Any],
        tag: str = "",
    ) -> None:
        self.pairs_dir = pairs_dir
        self.split = split
        self.shard = shard
        self.tag = tag
        self.flush_pairs = flush_pairs
        self.meta_kw = meta_kw
        self.rows: list[dict[str, Any]] = []
        self.buffered = 0
        self.sub = 0
        self.total_pairs = 0
        self.total_span_tokens = 0

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        n = len(row["prev_pos"])
        self.buffered += n
        self.total_pairs += n
        self.total_span_tokens += sum(len(s) for s in row["ids"])
        if self.buffered >= self.flush_pairs:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import torch

        from oracle_lens.pipeline.multilayer import (
            LAYERS,
            MultiLayerPairs,
            MultiLayerShardMeta,
            save_multilayer_shard,
        )

        ids = [seq for r in self.rows for seq in r["ids"]]
        pairs = MultiLayerPairs(
            span_ids=torch.tensor([t for seq in ids for t in seq], dtype=torch.int32),
            offsets=torch.tensor(
                [0, *torch.tensor([len(s) for s in ids]).cumsum(0).tolist()], dtype=torch.int64
            ),
            targets=torch.cat([r["targets"] for r in self.rows]).to(torch.bfloat16),
            layers=tuple(LAYERS),
            conv_index=torch.cat([r["conv_index"] for r in self.rows]),
            prev_pos=torch.cat([r["prev_pos"] for r in self.rows]),
            prev_token_id=torch.cat([r["prev_token_id"] for r in self.rows]),
            prev_is_assistant=torch.cat([r["prev_is_assistant"] for r in self.rows]),
        )
        meta = MultiLayerShardMeta(
            layers=tuple(LAYERS),
            split=cast(Literal["train", "eval"], self.split),
            n_pairs=len(pairs),
            **self.meta_kw,
        )
        out = (
            self.pairs_dir
            / f"pairs_{self.split}_{self.shard:04d}{self.tag}_{self.sub:02d}.safetensors"
        )
        tmp = out.with_suffix(f".tmp.{os.getpid()}")
        save_multilayer_shard(tmp, pairs, meta)
        tmp.replace(out)
        print(f"[capture] wrote {out.name} ({len(pairs)} pairs)", flush=True)
        self.rows = []
        self.buffered = 0
        self.sub += 1


def main() -> None:
    _hf_offline()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["chat", "pt"])
    ap.add_argument("--rollout-shard", type=int, required=True)
    ap.add_argument("--rollouts-dir", default="", help="default rollouts_iolens/<mode>")
    ap.add_argument("--out-dir", required=True, help="e.g. ml_pairs_iolens_chat")
    ap.add_argument("--n-max", type=int, default=1024, help="octave mode only")
    ap.add_argument(
        "--carve-mode", default="uniform32", choices=["uniform32", "octave"],
        help="uniform32 (iolens AR law): tile the answer with disjoint N~uniform{1..32} spans "
        "— many independent (text, target) pairs per conversation, uniform position coverage. "
        "octave: the legacy long-span cascade (crop-source captures for >32 experiments).",
    )
    ap.add_argument("--raw-slice-frac", type=float, default=0.05,
                    help="fraction of convs sampled with the OLD loguniform law (comparability)")
    ap.add_argument("--ao-val-as-train", action="store_true",
                    help="capture the ao_val split into the train writer INSTEAD of ar_train "
                    "(distillation source data — the untouched 5%%). Dedicated --out-dir only.")
    ap.add_argument("--ao-train-as-train", action="store_true",
                    help="capture the ao_train split into the train writer INSTEAD of ar_train "
                    "(asst scaling extension, user decision 2026-08-19: WildChat seeds are "
                    "exhausted, so the AR curve extends into the AO reserve — future AOs on "
                    "this pool LOSE their held-out-from-AR conv split). Dedicated --out-dir.")
    ap.add_argument("--span-n-lo", type=int, default=1,
                    help="uniform32 mode: min span length (default 1 = the published law)")
    ap.add_argument("--span-n-hi", type=int, default=32,
                    help="uniform32 mode: max span length (default 32 = the published law; "
                    "--span-n-lo 4 --span-n-hi 4 = the fixed-4-token carve)")
    ap.add_argument("--max-eval-convs", type=int, default=300,
                    help="cap on eval-split conversations captured per shard")
    ap.add_argument(
        "--train-frac", type=float, default=1.0,
        help="wave fraction: capture only this deterministic (seeded, conv-hash) fraction of"
        " ar_train conversations. Disk budgeting: FULL ar_train capture is ~3-4 TB of pairs;"
        " a wave captures a slice, trains, evicts to HF, then the next wave uses"
        " --train-frac-skip to take the NEXT slice.",
    )
    ap.add_argument(
        "--train-frac-skip", type=float, default=0.0,
        help="skip conversations whose hash falls below this (wave 2 = --train-frac-skip <wave1"
        " frac>); [skip, skip+frac) is captured — waves are disjoint by construction",
    )
    ap.add_argument("--flush-pairs", type=int, default=100_000)
    ap.add_argument("--batch-tokens", type=int, default=16_384,
                    help="padded tokens per capture forward (length-sorted batches)")
    ap.add_argument("--batch-rows", type=int, default=16, help="max conversations per forward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--allow-overlap", action="store_true",
        help="permit a capture whose conv-hash interval overlaps one already in the slice "
        "ledger. Those pairs are byte-identical duplicates of pairs already produced — only "
        "use this to deliberately re-derive a lost shard, never to add training data.",
    )
    args = ap.parse_args()
    if not (1 <= args.span_n_lo <= args.span_n_hi):
        raise SystemExit(
            f"--span-n-lo/-hi must satisfy 1 <= lo <= hi, got {args.span_n_lo}/{args.span_n_hi}"
        )

    import torch

    from oracle_lens.core.dump import conversations_layers_resid_batched
    from oracle_lens.model import ModelBackend
    from oracle_lens.pipeline.multilayer import LAYERS
    from oracle_lens.pipeline.rollout_store import SPLITS, load_rollout_shards
    from oracle_lens.pipeline.spans import (
        carve_disjoint_spans,
        carve_uniform_spans,
        delimiter_mask,
        sample_long_spans,
    )

    root = ola_root()
    rdir = root / (args.rollouts_dir or f"rollouts_iolens/{args.mode}")
    shard_path = rdir / f"rollouts_{args.rollout_shard:04d}.safetensors"
    pairs_dir = root / args.out_dir
    pairs_dir.mkdir(parents=True, exist_ok=True)
    # the wave tag keeps every (shard, slice) distinct: filenames and done-markers from a later
    # wave must never collide with (or be mistaken for) an earlier wave's on the same shard
    wave_tag = f"s{round(args.train_frac_skip * 1000):04d}"
    done_marker = pairs_dir / f"done_{args.rollout_shard:04d}{wave_tag}.json"
    if done_marker.exists():
        print(f"[capture] {done_marker} exists — skipping (idempotent)", flush=True)
        return
    # Slice ledger. The done-marker above only catches an identical (shard, wave) FILENAME; it
    # cannot see a slice captured under a different naming scheme, nor a partially-overlapping
    # --train-frac. The invariant that actually matters is on content: pairs are a pure function
    # of (rollout_shard, conv-hash interval, seed), because the span RNG is seeded by
    # (seed, rollout_shard, conv_index) and does NOT depend on train_frac_skip. So re-capturing
    # any overlapping [skip, skip+frac) on the same shard reproduces byte-identical pairs, which
    # the streaming trainer would consume as if they were fresh samples -- breaking the
    # "every row is a first-time sample" contract in stream_pairs. Observed 2026-08-02: the
    # producer re-captured chat shards 0-2 at skip 0.0 (already captured before the producer
    # loop existed, under the untagged naming) and held-out FVE stalled for ~576k samples.
    ledger_path = pairs_dir / "capture_ledger.json"
    ledger: list[dict[str, Any]] = (
        json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    )
    lo, hi = args.train_frac_skip, args.train_frac_skip + args.train_frac
    # Compare on a rounded grid with a tolerance. ADJACENT waves share an endpoint, and
    # `skip + frac` accumulates float error (0.2 + 0.1 = 0.30000000000000004), so a naive
    # `lo < e_hi` reads wave 3 [0.30,0.40) as overlapping wave 2 [0.20,0.30) by 5.5e-17 and
    # refuses it. That deadlocked the pt producer in a retry loop while its buffer drained.
    # eps is far below any real wave fraction, so genuine overlaps are still caught.
    eps = 1e-9
    for e in ledger:
        if e["rollout_shard"] != args.rollout_shard or e["seed"] != args.seed:
            continue
        e_lo, e_hi = round(float(e["lo"]), 6), round(float(e["hi"]), 6)
        if lo < e_hi - eps and e_lo < hi - eps:  # half-open interval overlap
            msg = (
                f"shard {args.rollout_shard} interval [{lo:.4f},{hi:.4f}) overlaps "
                f"already-captured [{e['lo']:.4f},{e['hi']:.4f}) at seed {args.seed} — those "
                f"pairs would be byte-identical duplicates. Pick a disjoint --train-frac-skip."
            )
            if not args.allow_overlap:
                raise SystemExit(f"[capture] REFUSING: {msg}")
            print(f"[capture] WARNING (--allow-overlap): {msg}", flush=True)
    rolls = load_rollout_shards([shard_path])
    # --ao-val-as-train (distillation, Agam 2026-08-11): source the TRAIN writer from the
    # ao_val split — the only 5% untouched by every AR and AO — instead of ar_train. Use a
    # DEDICATED --out-dir; schema/filenames/wave hashing/ledger all behave identically.
    if args.ao_val_as_train and args.ao_train_as_train:
        raise SystemExit("[capture] --ao-val-as-train and --ao-train-as-train are exclusive")
    split_ar = SPLITS.index(
        "ao_val" if args.ao_val_as_train else "ao_train" if args.ao_train_as_train else "ar_train"
    )
    split_eval = SPLITS.index("eval")
    dataset_id = f"iolens-{args.mode}"

    # ---- G11 storage preflight: carve the first 64 AR-split convs, extrapolate, demand 2x ----
    probe_spans = 0
    probe_convs = 0
    import hashlib as _hb

    def _u(i: int) -> float:
        h = _hb.blake2b(int(rolls.seed_hash[i]).to_bytes(8, "big", signed=True),
                        digest_size=8).digest()
        return int.from_bytes(h, "big") / 2**64

    wlo, whi = args.train_frac_skip, args.train_frac_skip + args.train_frac
    n_ar_convs = sum(
        1 for i in range(len(rolls))
        if int(rolls.split_id[i]) == split_ar and wlo <= _u(i) < whi
    )
    for i in range(len(rolls)):
        if int(rolls.split_id[i]) != split_ar or not (wlo <= _u(i) < whi):
            continue
        plen = int(rolls.prompt_len[i])
        total = min(int(rolls.offsets[i + 1] - rolls.offsets[i]), T_MAX)
        if total <= plen:
            continue
        mask = [j >= plen for j in range(total)]
        rng = random.Random(args.seed * 100_003 + i)
        if args.carve_mode == "uniform32":
            probe_spans += len(
                carve_uniform_spans(mask, rng=rng, n_lo=args.span_n_lo, n_hi=args.span_n_hi)
            )
        else:
            probe_spans += len(carve_disjoint_spans(mask, rng=rng, n_max=args.n_max))
        probe_convs += 1
        if probe_convs >= 64:
            break
    est_pairs = int(probe_spans / max(1, probe_convs) * n_ar_convs)
    est_bytes = est_pairs * PAIR_BYTES
    free = shutil.disk_usage(pairs_dir).free
    print(
        f"[capture] G11 preflight: ~{probe_spans / max(1, probe_convs):.1f} spans/conv x "
        f"{n_ar_convs} convs ≈ {est_pairs:,} pairs ≈ {est_bytes / 2**30:.1f} GiB "
        f"(free {free / 2**30:.0f} GiB)",
        flush=True,
    )
    if free < 2 * est_bytes:
        raise SystemExit("[capture] G11 REFUSED: free disk < 2x projected pair bytes")

    backend = ModelBackend(MODEL_ID, device="cuda", dtype=torch.bfloat16)
    # Capture-time content filters (Agam, 2026-08-01): drop whole conversations whose OUTPUT
    # (a) re-emits a <think> block (chat; ~7.8% — off-distribution for the AO's explain task)
    # or (b) is char-chunk repetitive (~0.01%). Rollout shards on HF stay complete; these
    # filters are capture-local and revisable. Counts reported exactly.
    think_id = backend.tokenizer.convert_tokens_to_ids("<think>")
    filtered = {"think_reemit": 0, "chunk_repeat": 0}
    meta_kw = {
        "model_id": MODEL_ID,
        "dataset_id": dataset_id,
        "t_max": T_MAX,
        "n_max": args.n_max,
        "n_min": args.span_n_lo,
        # the default string stays byte-identical to the published shards; only a non-default
        # span window changes it (metadata consumers must not see a spurious law diff)
        "length_law": (
            (
                "uniform32_disjoint"
                if (args.span_n_lo, args.span_n_hi) == (1, 32)
                else f"uniform32_disjoint_n{args.span_n_lo}-{args.span_n_hi}"
            )
            if args.carve_mode == "uniform32"
            else f"carve_disjoint+raw{args.raw_slice_frac}"
        ),
        "region_start": 0,
        "region_end": 0,
        "seed": args.seed,
        "git_commit": _git_commit(),
    }
    writers = {
        "train": SubShardWriter(pairs_dir, "train", args.rollout_shard, args.flush_pairs,
                                meta_kw, tag=wave_tag),
        "eval": SubShardWriter(pairs_dir, "eval", args.rollout_shard, args.flush_pairs,
                               meta_kw, tag=wave_tag),
    }
    conv_base = args.rollout_shard * 1_000_000
    used = {"train": 0, "eval": 0}

    # Pass 1 (CPU): carve spans per conversation, build the work list.
    work: list[dict[str, Any]] = []  # {i, split, rendered, mask, spans}
    import hashlib as _hashlib

    def _conv_u(i: int) -> float:
        """Deterministic uniform in [0,1) per conversation (seed-hash keyed — wave-stable)."""
        h = _hashlib.blake2b(int(rolls.seed_hash[i]).to_bytes(8, "big", signed=True),
                             digest_size=8).digest()
        return int.from_bytes(h, "big") / 2**64

    lo_f, hi_f = args.train_frac_skip, args.train_frac_skip + args.train_frac
    for i in range(len(rolls)):
        sp = int(rolls.split_id[i])
        if sp == split_ar:
            if not (lo_f <= _conv_u(i) < hi_f):
                continue  # another wave's slice
            split = "train"
        elif sp == split_eval and used["eval"] < args.max_eval_convs:
            split = "eval"
        else:
            continue
        rendered = rolls.conv_ids(i).tolist()[:T_MAX]
        plen = min(int(rolls.prompt_len[i]), len(rendered))
        if len(rendered) < MIN_RENDERED_TOKENS or plen >= len(rendered):
            continue
        out_ids = rendered[plen:]
        if args.mode == "chat" and isinstance(think_id, int) and think_id in out_ids:
            filtered["think_reemit"] += 1
            continue
        if is_chunk_repetitive(backend.tokenizer.decode(out_ids)):
            filtered["chunk_repeat"] += 1
            continue
        mask = [j >= plen for j in range(len(rendered))]
        rng = random.Random(args.seed * 100_003 + args.rollout_shard * 10_007 + i)
        if args.carve_mode == "uniform32":
            spans = carve_uniform_spans(mask, rng=rng, n_lo=args.span_n_lo, n_hi=args.span_n_hi)
        elif rng.random() < args.raw_slice_frac:
            delim = delimiter_mask(rendered, backend.tokenizer)
            spans = sample_long_spans(
                mask, max_per_conv=16, rng=rng, n_min=1, n_max=args.n_max, max_attempts=64,
                delimiter_prev=delim, delimiter_upsample=1.0, length_law="loguniform",
            )
        else:
            spans = carve_disjoint_spans(mask, rng=rng, n_max=args.n_max)
        if not spans:
            continue
        work.append({"i": i, "split": split, "rendered": rendered, "mask": mask, "spans": spans})
        used[split] += 1
    print(f"[capture] shard {args.rollout_shard}: {len(work)} convs to capture "
          f"({used})", flush=True)

    # Pass 2 (GPU): length-sorted batched forwards under a token budget — the throughput lever
    # (a lone ~1k-token forward leaves the H200 mostly idle; ~16k-token batches run prefill
    # compute-bound). Length sorting keeps padding waste minimal; causality means right-pad
    # garbage can never leak into the positions we read.
    work.sort(key=lambda w: len(w["rendered"]))
    batches: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in work:
        widest = max(len(w["rendered"]), *(len(x["rendered"]) for x in cur)) if cur else len(
            w["rendered"]
        )
        if cur and (widest * (len(cur) + 1) > args.batch_tokens or len(cur) >= args.batch_rows):
            batches.append(cur)
            cur = []
        cur.append(w)
    if cur:
        batches.append(cur)

    import time as _time

    t0 = _time.time()
    done_tokens = 0
    for bi, batch in enumerate(batches):
        resid = conversations_layers_resid_batched(
            backend, [w["rendered"] for w in batch], layers=list(LAYERS)
        )  # {L: [b, pos, d]}
        for j, w in enumerate(batch):
            spans = w["spans"]
            rendered = w["rendered"]
            prev_idx = torch.tensor([s.prev_pos for s in spans], dtype=torch.long)
            pidx = prev_idx.to(backend.device)  # index on the resid's own device (device gotcha)
            tgt = torch.stack([resid[lyr][j, pidx] for lyr in LAYERS], dim=1).cpu()
            writers[w["split"]].add(
                {
                    "ids": [rendered[s.start : s.start + s.n_tokens] for s in spans],
                    "targets": tgt,
                    "conv_index": torch.full((len(spans),), conv_base + w["i"], dtype=torch.long),
                    "prev_pos": prev_idx,
                    "prev_token_id": torch.tensor(
                        [rendered[s.prev_pos] for s in spans], dtype=torch.long
                    ),
                    "prev_is_assistant": torch.tensor(
                        [w["mask"][s.prev_pos] for s in spans], dtype=torch.bool
                    ),
                }
            )
        done_tokens += sum(len(w["rendered"]) for w in batch)
        if bi % 50 == 0:
            rate = done_tokens / max(1.0, _time.time() - t0)
            print(
                f"[capture] shard {args.rollout_shard}: batch {bi}/{len(batches)}, "
                f"{writers['train'].total_pairs + writers['eval'].total_pairs} pairs, "
                f"{rate:,.0f} rendered tok/s",
                flush=True,
            )

    for writer in writers.values():
        writer.flush()
    report = {
        "shard": args.rollout_shard,
        "mode": args.mode,
        "convs_used": used,
        "convs_filtered": filtered,
        "pairs": {k: w.total_pairs for k, w in writers.items()},
        "span_tokens": {k: w.total_span_tokens for k, w in writers.items()},
        "sub_shards": {k: w.sub for k, w in writers.items()},
        "gb": round(
            (writers["train"].total_pairs + writers["eval"].total_pairs) * PAIR_BYTES / 2**30, 2
        ),
    }
    tmp = done_marker.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(done_marker)
    # Record the conv-hash interval so no later capture can re-derive these pairs. Written after
    # the shard is complete and re-read here (not from the copy loaded at startup) so concurrent
    # captures of different shards don't clobber each other's entries.
    cur = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    cur.append({
        "mode": args.mode, "rollout_shard": args.rollout_shard, "seed": args.seed,
        "lo": lo, "hi": hi, "carve_mode": args.carve_mode,
        "pairs": writers["train"].total_pairs, "wave_tag": wave_tag,
    })
    ltmp = ledger_path.with_suffix(f".tmp.{os.getpid()}")
    ltmp.write_text(json.dumps(cur, indent=2))
    ltmp.replace(ledger_path)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
