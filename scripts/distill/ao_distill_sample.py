"""Bulk AO sampling for distillation: GROUND-TRUTH activations in, k' samples out (vLLM).

The AO-side port of ``olens_distill_sample_cluster.py`` (same two-venv split, same
prompt-embeds contract) for the u64 tag-free AO. Differences from the olens sampler:

- Injected vectors are TRUE residuals from the ao_val pairs (never AR reconstructions —
  Agam 2026-08-11), precomputed per arm at the prompts stage:
    * ``normmatched``: alpha * v/||v||  (the probe_obvious winner: 203/1800 vs 138 frozen)
    * ``frozen``:      64.559 * v       (the run's trained scale on raw GT norms, ~2.7x cold)
- Prompt = the AO's own training prompt (``continuation_raw``) rendered per layer — generation
  matches training exactly; no stop string (tag-free targets end at EOS).
- Rows are drawn with EXACT per-layer floors over the training universe [20..60] and filtered
  for trivial following-text (r2_filter.is_degenerate) before anything is sampled.

    # 1) project venv: filter + draw + render + write vectors
    uv run python scripts/distill/ao_distill_sample.py --mode prompts \
        --pairs-dir ml_pairs_aoval --out-dir distill_u64/pilot --n-rows 2046
    # 2) vllm venv (no repo imports):
    <your-vllm-venv>/bin/python scripts/distill/ao_distill_sample.py \
        --mode sample --out-dir distill_u64/pilot --merged-dir /opt/ola/merged/ao-u64-step28000 \
        --arm normmatched --n-shards 4 --shard 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

AO_LAYERS = list(range(20, 61, 4))  # the u64 training universe (11 layers, no 63)
ALPHA = 16000.0
FROZEN_SCALE = 64.55908784774493
K_PRIME = 64
MAX_NEW = 100
PROMPT_KIND = "continuation_raw"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def do_prompts(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards_lazy
    from oracle_lens.pipeline.r2_filter import is_degenerate
    from oracle_lens.pipeline.verbalizer import renderer_for

    root = ola_root()
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

    paths = sorted((root / args.pairs_dir).glob("pairs_train_*.safetensors"))
    pairs, _meta = load_multilayer_shards_lazy(paths)
    lens = pairs.offsets[1:] - pairs.offsets[:-1]

    # triviality filter on the true following-text (Agam: don't build items whose target is
    # trivial); seeded shuffle then greedy fill with EXACT per-layer floors
    used: set[int] = set()
    if args.exclude_pconf:
        for pc in args.exclude_pconf.split(","):
            prev = json.loads((root / pc).read_text())
            used |= {int(it["pair_row"]) for it in prev["items"]}
        print(f"[ao-distill] excluding {len(used)} pair_rows from {args.exclude_pconf}",
              flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(pairs), generator=gen).tolist()
    per_layer = args.n_rows // len(AO_LAYERS)
    assert per_layer * len(AO_LAYERS) == args.n_rows, "--n-rows must divide by 11 layers"
    items: list[dict] = []
    counts = dict.fromkeys(AO_LAYERS, 0)
    n_trivial = 0
    for r in order:
        if len(items) >= args.n_rows:
            break
        if r in used:
            continue
        span = pairs.span_ids[int(pairs.offsets[r]) : int(pairs.offsets[r + 1])].tolist()
        text = tok.decode(span)
        if is_degenerate(text):
            n_trivial += 1
            continue
        ly = next((x for x in AO_LAYERS if counts[x] < per_layer), None)
        if ly is None:
            break
        counts[ly] += 1
        items.append({"pair_row": int(r), "layer": int(ly), "span_len": int(lens[r])})
    assert all(v == per_layer for v in counts.values()), f"layer floors unmet: {counts}"
    print(f"[ao-distill] {len(items)} items ({per_layer}/layer x {len(AO_LAYERS)}), "
          f"trivial killed {n_trivial}", flush=True)

    # per-arm FINAL injected vectors (raw GT residual at the item's layer)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    vecs = {a: torch.zeros(len(items), 5120) for a in arms}
    for i, it in enumerate(items):
        v = torch.as_tensor(pairs.targets[it["pair_row"]][LAYERS.index(it["layer"])]).float()
        for a in arms:
            vecs[a][i] = ALPHA * v / v.norm() if a == "normmatched" else args.frozen_scale * v
    for a in arms:
        n = vecs[a].norm(dim=-1)
        print(f"[ao-distill] arm {a}: |inject| median {n.median():.0f} "
              f"(p10 {n.quantile(0.1):.0f} / p90 {n.quantile(0.9):.0f})", flush=True)
        from safetensors.torch import save_file

        save_file({"vecs": vecs[a].to(torch.bfloat16)}, str(out / f"vecs_{a}.safetensors"))

    render = renderer_for(args.prompt_kind)
    prompts = {}
    for ly in AO_LAYERS:
        p = render(tok, layer=ly)
        prompts[str(ly)] = {"input_ids": list(p.input_ids), "slot": int(p.slot)}
    (out / "pconf.json").write_text(json.dumps({
        "prompt_kind": args.prompt_kind, "alpha": ALPHA, "frozen_scale": args.frozen_scale,
        "k_prime": args.k_prime, "max_new": args.max_new, "arms": arms, "seed": args.seed,
        "pairs_dir": args.pairs_dir, "teacher": args.teacher_tag,
        "prompts": prompts, "items": items,
    }))
    print(f"[ao-distill] wrote {out}/pconf.json + vecs_*.safetensors", flush=True)


def do_sample(args: argparse.Namespace) -> None:
    # vllm venv: NO repo imports past this point
    import torch
    from safetensors import safe_open
    from vllm import LLM, SamplingParams

    root = ola_root()
    out = root / args.out_dir
    pconf = json.loads((out / "pconf.json").read_text())
    items = pconf["items"]
    lo = len(items) * args.shard // args.n_shards
    hi = len(items) * (args.shard + 1) // args.n_shards
    dst = out / f"texts_{args.arm}_{args.shard:04d}.json"
    if dst.exists():
        print(f"[ao-distill] {dst.name} exists — skipping", flush=True)
        return

    merged = Path(args.merged_dir)
    index = json.loads((merged / "model.safetensors.index.json").read_text())["weight_map"]
    emb_key = next(k for k in index if k.endswith("embed_tokens.weight"))
    with safe_open(str(merged / index[emb_key]), framework="pt") as f:
        emb = f.get_tensor(emb_key)
    base_pe = {
        ly: emb[torch.tensor(p["input_ids"], dtype=torch.long)].clone()
        for ly, p in pconf["prompts"].items()
    }
    with safe_open(str(out / f"vecs_{args.arm}.safetensors"), framework="pt") as f:
        vecs = f.get_tensor("vecs")

    # Per-PART atomic writes + resume (Agam 2026-08-11 after the b4 kill lost 45 min of
    # memory-buffered generation): every --part-rows items land in their own
    # texts_<arm>_<shard>.part<base>.json; on restart, done parts are skipped; the final
    # texts_<arm>_<shard>.json is stitched from parts (reader-compatible) and parts removed.
    part_rows = max(args.chunk, args.part_rows - args.part_rows % args.chunk)

    def part_path(pbase: int) -> Path:
        return out / f"texts_{args.arm}_{args.shard:04d}.part{pbase:06d}.json"

    llm = LLM(model=str(merged), enable_prompt_embeds=True,
              max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(n=pconf["k_prime"], temperature=1.0, top_p=0.95,
                        max_tokens=pconf["max_new"])
    for pbase in range(lo, hi, part_rows):
        pend = min(pbase + part_rows, hi)
        if part_path(pbase).exists():
            print(f"[ao-distill] part {pbase} exists — resuming past it", flush=True)
            continue
        results = []
        for base in range(pbase, pend, args.chunk):
            part = items[base : min(base + args.chunk, pend)]
            prompts = []
            for j, it in enumerate(part):
                pe = base_pe[str(it["layer"])].clone()
                # vecs are indexed by GLOBAL item position (written over all items at prompts)
                pe[pconf["prompts"][str(it["layer"])]["slot"]] = vecs[base + j].to(pe.dtype)
                prompts.append({"prompt_embeds": pe})
            outs = llm.generate(prompts, sp)
            for it, o in zip(part, outs, strict=True):
                results.append({**it, "samples": [c.text for c in o.outputs]})
            print(f"[ao-distill] {min(base + args.chunk, pend) - lo}/{hi - lo} rows", flush=True)
        ptmp = part_path(pbase).with_suffix(".tmp")
        ptmp.write_text(json.dumps({"arm": args.arm, "lo": pbase, "hi": pend, "rows": results}))
        ptmp.replace(part_path(pbase))
    rows: list[dict] = []
    parts = sorted(out.glob(f"texts_{args.arm}_{args.shard:04d}.part*.json"))
    for p in parts:
        rows += json.loads(p.read_text())["rows"]
    if len(rows) != hi - lo:
        raise SystemExit(f"[ao-distill] stitched {len(rows)} rows != shard size {hi - lo}")
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps({"arm": args.arm, "lo": lo, "hi": hi, "rows": rows}))
    tmp.replace(dst)
    for p in parts:
        p.unlink()
    print(f"[ao-distill] wrote {dst} (stitched {len(parts)} parts)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["prompts", "sample"])
    ap.add_argument("--pairs-dir", default="ml_pairs_aoval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-rows", type=int, default=2046, help="must divide by the 11 layers")
    ap.add_argument("--exclude-pconf", default="",
                    help="csv of OLA_ROOT-relative pconf.json paths whose pair_rows to skip "
                         "(batch-2+ draws must not reuse earlier batches' rows)")
    ap.add_argument("--arms", default="normmatched,frozen")
    ap.add_argument("--frozen-scale", type=float, default=FROZEN_SCALE,
                    help="the teacher cell's frozen global scale for the 'frozen' arm "
                         "(default = the chat u64 AO's 64.559; ptag u64max32 cell: 122.3225 "
                         "from ao_runs/scale_iolens_ptag_u64_final.json)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--teacher-tag", default="ao.iolens.chat.k4.L20-60.cont.u64.s0/step28000")
    ap.add_argument("--merged-dir", default="")
    ap.add_argument("--arm", default="normmatched")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--part-rows", type=int, default=512,
                    help="rows per intermediate part file (resume granularity)")
    ap.add_argument("--prompt-kind", default="continuation_raw",
                    help="verbalizer prompt for generation (concepts_raw for bullet students)")
    ap.add_argument("--k-prime", type=int, default=K_PRIME)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()
    if args.mode == "prompts":
        do_prompts(args)
    else:
        do_sample(args)


if __name__ == "__main__":
    main()
