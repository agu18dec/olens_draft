"""Round-2 teacher sampling: frozen round-1 lens x k' independent samples per activation.

The teacher is ALWAYS sampled with its own round-1 continuation prompt (one per layer), k'
independent completions at temp 1.0 / top_p 0.95 — never prompted for k things at once (the
k-block list is assembled later for the STUDENT; ``assemble_distill.py``). Per-phrase length n
is enforced by ``max_new`` here and token truncation at assembly (never by prompt wording).

Three modes:

- ``--mode prompts`` (project venv): render the 17 per-layer continuation prompts with
  ``ola/verbalizer.render_wv_continuation_prompt`` and dump ``{layer: {input_ids, slot}}`` +
  the teacher's injection contract (transform/alpha from ``olens_runs/<run>.json``) to
  ``prompts.json``. Single source of truth — the vLLM engine below never imports repo modules
  (the rollout venv lacks jaxtyping/peft).
- ``--mode sample --engine vllm`` (ROLLOUT venv): serve the MERGED teacher
  (``merge_teacher_vl.py``) with ``enable_prompt_embeds=True``; per row build the full prompt
  embedding from the merged checkpoint's ``embed_tokens`` rows with ``alpha * h/||h||`` written
  into the slot, and generate ``n=k'`` samples with stop="</explanation>". Self-contained.
- ``--mode sample --engine hf`` (project venv): PEFT adapter + ``generate(inputs_embeds=…)``
  (the ``olens_fve_eval.py`` pattern) — the pilot/parity fallback, ~8x slower.

Rows come from ``$OLA_ROOT/r2_resid_pairs/{train,eval}`` (true residuals); the layer per row is
a seeded draw shared by every variant (same activations, same layers — matched comparison).
Sharded by contiguous row slice; skip-if-exists; atomic writes. ``--topup <shortfall.json>``
re-samples only the listed rows at 2x k' into ``topup<round>_`` files.

    # after merge: one worker per GPU (shard i of N), e.g. variant k20n16, train split
    VLLM=<your-vllm-venv>/bin/python
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \\
        $VLLM scripts/distill/olens_distill_sample_cluster.py \\
            --mode sample --engine vllm --variant k20n16 --split train \\
            --teacher-run olens.n1024.recon.main.a8000.lr3e-4.r32.s0 --n-shards 16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"
PAIRS_DIR = "r2_resid_pairs"
OUT_ROOT = "distill_n1024"
LAYERS: tuple[int, ...] = (*range(0, 61, 4), 63)  # == ola.multilayer.LAYERS (17)

# k' oversampling + max_new per variant — mirror of ola.distill.VARIANTS (kept inline so the
# vllm engine needs no repo imports; the unit test asserts the two stay in sync).
VARIANTS: dict[str, dict[str, int]] = {
    "k1n1024": {"k": 1, "n": 1024, "k_prime": 8, "max_new": 1040},
    "k4n256": {"k": 4, "n": 256, "k_prime": 40, "max_new": 272},
    "k20n4": {"k": 20, "n": 4, "k_prime": 48, "max_new": 12},
    "k20n16": {"k": 20, "n": 16, "k_prime": 32, "max_new": 28},
    # "*d*" chain variants sample through the HF engine or the step-synchronous driver, never
    # a plain iid vllm pass — listed here only so the sync test sees one canonical table
    # (k_prime = draws per chain step for these).
    "k4n256d": {"k": 4, "n": 256, "k_prime": 1, "max_new": 272},
    "k20n4d": {"k": 20, "n": 4, "k_prime": 1, "max_new": 12},
    "k20n16d": {"k": 20, "n": 16, "k_prime": 1, "max_new": 28},
    # selection-pool variants (round-2 methods 1-3 share one pool; k picked downstream):
    "pool16n32": {"k": 4, "n": 32, "k_prime": 16, "max_new": 48},
    "pool64n32": {"k": 4, "n": 32, "k_prime": 64, "max_new": 48},
    "pool128n32": {"k": 4, "n": 32, "k_prime": 128, "max_new": 48},
    "pool256n32": {"k": 4, "n": 32, "k_prime": 256, "max_new": 48},
    "k4n32d16": {"k": 4, "n": 32, "k_prime": 16, "max_new": 48},
    "k4n32d64": {"k": 4, "n": 32, "k_prime": 64, "max_new": 48},
    "k4n32d128": {"k": 4, "n": 32, "k_prime": 128, "max_new": 48},
    "k4n32d256": {"k": 4, "n": 32, "k_prime": 256, "max_new": 48},
}


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


class PairShards:
    """Minimal lazy reader over multilayer_v1 shards (no repo imports — vllm-venv safe)."""

    def __init__(self, split_dir: Path) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.paths = sorted(split_dir.glob("*.safetensors"))
        if not self.paths:
            raise SystemExit(f"no pair shards under {split_dir}")
        self.rows_per = []
        self.layers: tuple[int, ...] = ()
        for p in self.paths:
            with safe_open(str(p), framework="pt") as f:
                meta = f.metadata()
                if meta.get("targets_kind") != "resid":
                    raise SystemExit(f"{p}: targets_kind={meta.get('targets_kind')!r} != resid")
                self.rows_per.append(int(meta["n_pairs"]))
                self.layers = tuple(int(x) for x in meta["layers"].split(","))
        if self.layers != LAYERS:
            raise SystemExit(f"shard layers {self.layers} != expected {LAYERS}")
        self._cum = [0]
        for r in self.rows_per:
            self._cum.append(self._cum[-1] + r)
        self._handles: dict[int, object] = {}

    def __len__(self) -> int:
        return self._cum[-1]

    def vec(self, row: int, layer_pos: int):  # type: ignore[no-untyped-def]
        import bisect

        s = bisect.bisect_right(self._cum, row) - 1
        if s not in self._handles:
            self._handles[s] = self._safe_open(str(self.paths[s]), framework="pt")
        local = row - self._cum[s]
        return self._handles[s].get_slice("targets")[local][layer_pos]  # type: ignore[attr-defined]


def layer_draw_for(n_rows: int, *, seed: int, split: str) -> Any:
    """Seeded per-row layer assignment (index into LAYERS), shared across all variants."""
    import torch

    gen = torch.Generator().manual_seed(seed + (1 if split == "eval" else 0))
    return torch.randint(len(LAYERS), (n_rows,), generator=gen)


def teacher_tag(run: str, step: str) -> str:
    return run + (f".{step}" if step else "")


def do_prompts(args: argparse.Namespace) -> None:
    """Project-venv step: render per-layer prompts + injection contract -> prompts.json."""
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.verbalizer import render_wv_continuation_prompt

    root = ola_root()
    run_json = root / "olens_runs" / f"{args.teacher_run}.json"
    if not run_json.is_file():
        raise SystemExit(f"{run_json} missing — teacher run record required")
    cfg = json.loads(run_json.read_text())["config"]
    if cfg["transform"] != "unit":
        raise SystemExit(f"teacher transform {cfg['transform']!r} unsupported (unit only)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if args.prompt_variant:
        # r2sf-dp pilot prompts were part of a research line not carried into this repo
        # (monorepo scripts/ola/olens_r2sf_desc_prompt_pilot.py).
        raise SystemExit("--prompt-variant is not supported in this repo; use the default prompt")
    else:
        prompts = {
            str(ly): {"input_ids": p.input_ids, "slot": p.slot}
            for ly in LAYERS
            for p in [render_wv_continuation_prompt(tok, layer=ly)]
        }
    out_dir = root / OUT_ROOT / teacher_tag(args.teacher_run, args.step)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "teacher_run": args.teacher_run,
        "step": args.step,
        "transform": cfg["transform"],
        "alpha": float(cfg["alpha"]),
        "layers": list(LAYERS),
        "prompts": prompts,
    }
    fname = f"prompts.{args.prompt_variant}.json" if args.prompt_variant else "prompts.json"
    payload["prompt_variant"] = args.prompt_variant
    (out_dir / fname).write_text(json.dumps(payload))
    print(f"[r2s] wrote {out_dir / fname} ({len(prompts)} layers)", flush=True)


def _work_rows(args: argparse.Namespace, n_rows: int) -> list[int]:
    """This shard's row list: a contiguous slice, or the topup file's rows sharded round-robin."""
    if args.topup:
        listed = json.loads(Path(args.topup).read_text())["rows"]
        return [int(r) for i, r in enumerate(listed) if i % args.n_shards == args.shard]
    chunk = (n_rows + args.n_shards - 1) // args.n_shards
    lo = args.shard * chunk
    hi = min(n_rows, lo + chunk)
    if args.max_rows > 0:
        hi = min(hi, lo + args.max_rows)
    return list(range(lo, hi))


def do_sample(args: argparse.Namespace) -> None:
    root = ola_root()
    tag = teacher_tag(args.teacher_run, args.step)
    var = VARIANTS[args.variant]
    k_prime = (2 * var["k_prime"]) if args.topup else var["k_prime"]
    shard = args.shard if args.shard >= 0 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    args.shard = shard
    pf = f"prompts.{args.prompts_tag}.json" if args.prompts_tag else "prompts.json"
    prompts_file = root / OUT_ROOT / tag / pf
    if not prompts_file.is_file():
        raise SystemExit(f"{prompts_file} missing — run --mode prompts first (project venv)")
    pconf = json.loads(prompts_file.read_text())

    out_dir = root / OUT_ROOT / tag / "samples" / (args.variant + args.out_suffix) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"topup{args.round}_" if args.topup else ""
    out_path = out_dir / f"{prefix}samples_{shard:04d}.json"
    if out_path.exists():
        print(f"[r2s] {out_path} exists — skipping", flush=True)
        return

    pairs_dir = Path(args.pairs_dir) if args.pairs_dir else root / PAIRS_DIR / args.split
    pairs = PairShards(pairs_dir)
    draw = layer_draw_for(len(pairs), seed=args.seed, split=args.split)
    if getattr(args, "layer_map", ""):
        # chain steps over sparse-row le64 pools: honor each row's ORIGINAL layer
        import json as _json

        _lm = _json.loads(Path(args.layer_map).read_text())
        import torch as _torch

        draw = _torch.tensor([LAYERS.index(int(_lm[str(i)])) for i in range(len(pairs))])
        print(f"[r2s] layer map override: {args.layer_map} ({len(_lm)} rows)", flush=True)
    rows = _work_rows(args, len(pairs))
    print(
        f"[r2s] shard {shard}: {len(rows)} rows, variant={args.variant} k'={k_prime} "
        f"max_new={var['max_new']} engine={args.engine}",
        flush=True,
    )
    if not rows:
        _atomic_write(out_path, [])
        return

    if args.engine == "vllm":
        results = _sample_vllm(args, root, tag, pconf, pairs, draw, rows, k_prime, var["max_new"])
    else:
        results = _sample_hf(args, root, pconf, pairs, draw, rows, k_prime, var["max_new"])

    payload = [
        {"row": r, "layer": LAYERS[int(draw[r])], "samples": s}
        for r, s in zip(rows, results, strict=True)
    ]
    _atomic_write(out_path, payload)
    n_tok = sum(len(x) for s in results for x in s)  # chars, cheap proxy
    print(
        f"[r2s] shard {shard}: saved {len(payload)} rows -> {out_path} (~{n_tok} chars)", flush=True
    )
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        _atomic_write_obj(
            meta_path,
            {
                "variant": args.variant,
                "teacher": tag,
                "seed": args.seed,
                "k_prime": var["k_prime"],
                "max_new": var["max_new"],
                "engine": args.engine,
                "n_rows_split": len(pairs),
                "temperature": args.temp,
                "top_p": 0.95,
                "out_suffix": args.out_suffix,
                "pairs_dir": str(pairs_dir),
                "prompts_tag": args.prompts_tag,
            },
        )


def _atomic_write(path: Path, rows: list) -> None:  # type: ignore[type-arg]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False))
    tmp.replace(path)


def _atomic_write_obj(path: Path, obj: dict) -> None:  # type: ignore[type-arg]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _unit_inject(vec, alpha: float):  # type: ignore[no-untyped-def]
    v = vec.float()
    return alpha * v / v.norm().clamp_min(1e-9)


def _sample_vllm(  # type: ignore[no-untyped-def]
    args,
    root: Path,
    tag: str,
    pconf: dict[str, Any],
    pairs: PairShards,
    draw,
    rows: list[int],
    k_prime: int,
    max_new: int,
) -> list[list[str]]:
    import torch
    from safetensors import safe_open
    from vllm import LLM, SamplingParams

    merged = Path(args.merged_dir) if args.merged_dir else root / "merged" / tag
    if not (merged / "config.json").is_file():
        raise SystemExit(f"merged teacher missing at {merged} — run merge_teacher_vl.py")

    # embed_tokens rows straight from the merged checkpoint (no 27B HF load): the embedding is
    # a plain lookup, so weight rows == get_input_embeddings() output for the same ids.
    index = json.loads((merged / "model.safetensors.index.json").read_text())["weight_map"]
    emb_key = next(k for k in index if k.endswith("embed_tokens.weight"))
    with safe_open(str(merged / index[emb_key]), framework="pt") as f:
        emb = f.get_tensor(emb_key)  # [vocab, d] bf16, cpu
    base_pe = {
        int(ly): emb[torch.tensor(p["input_ids"], dtype=torch.long)].clone()
        for ly, p in pconf["prompts"].items()
    }
    slots = {int(ly): int(p["slot"]) for ly, p in pconf["prompts"].items()}

    llm = LLM(
        model=str(merged),
        enable_prompt_embeds=True,
        max_num_seqs=args.max_num_seqs,  # hybrid-Mamba cache cap (~310) — onpolicy_cluster.py
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=k_prime,
        temperature=args.temp,
        top_p=0.95,
        max_tokens=max_new,
        stop=["</explanation>"],
    )
    results: list[list[str]] = []
    chunk = max(1, args.chunk)
    for lo in range(0, len(rows), chunk):
        part = rows[lo : lo + chunk]
        prompts = []
        for r in part:
            ly = LAYERS[int(draw[r])]
            pe = base_pe[ly].clone()
            pe[slots[ly]] = _unit_inject(
                torch.as_tensor(pairs.vec(r, int(draw[r]))), alpha=float(pconf["alpha"])
            ).to(pe.dtype)
            prompts.append({"prompt_embeds": pe})
        outs = llm.generate(prompts, sp)
        results.extend([[o.text for o in out.outputs] for out in outs])
        print(f"[r2s] {min(lo + chunk, len(rows))}/{len(rows)} rows sampled", flush=True)
    return results


def _sample_hf(  # type: ignore[no-untyped-def]
    args,
    root: Path,
    pconf: dict[str, Any],
    pairs: PairShards,
    draw,
    rows: list[int],
    k_prime: int,
    max_new: int,
) -> list[list[str]]:
    """Project-venv fallback: PEFT teacher + batched generate(inputs_embeds=…) per row."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from oracle_lens.model import load_causal_lm

    lora_dir = root / "olens_checkpoints" / args.teacher_run
    if args.step:
        lora_dir = lora_dir / args.step
    lora_dir = lora_dir / "lora"
    import importlib.util

    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    model = load_causal_lm(MODEL_ID, dtype=torch.bfloat16, device="cuda", attn_implementation=attn)
    peft = PeftModel.from_pretrained(model, str(lora_dir)).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = int(tok.pad_token_id or 0)
    embed = model.get_input_embeddings()
    alpha = float(pconf["alpha"])
    results: list[list[str]] = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            ly = LAYERS[int(draw[r])]
            p = pconf["prompts"][str(ly)]
            ids = torch.tensor([p["input_ids"]], device="cuda")
            pe = embed(ids)[0].clone()
            vec = torch.as_tensor(pairs.vec(r, int(draw[r])))
            pe[int(p["slot"])] = _unit_inject(vec, alpha=alpha).cuda().to(pe.dtype)
            batch = pe.unsqueeze(0).expand(k_prime, -1, -1)
            out = peft.generate(  # type: ignore[no-untyped-call]
                inputs_embeds=batch,
                attention_mask=torch.ones(k_prime, pe.shape[0], dtype=torch.long, device="cuda"),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=args.temp,
                top_p=0.95,
                use_cache=True,
                pad_token_id=pad_id,
            )
            results.append([str(tok.decode(o, skip_special_tokens=True)) for o in out])
            if (i + 1) % 20 == 0:
                print(f"[r2s-hf] {i + 1}/{len(rows)} rows", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=("prompts", "sample"))
    ap.add_argument("--engine", type=str, default="vllm", choices=("vllm", "hf"))
    ap.add_argument("--variant", type=str, default="", choices=("", *VARIANTS))
    ap.add_argument("--split", type=str, default="train", choices=("train", "eval"))
    ap.add_argument("--teacher-run", type=str, required=True)
    ap.add_argument("--step", type=str, default="", help="stepN subdir; empty = final lora")
    ap.add_argument("--merged-dir", type=str, default="", help="default $OLA_ROOT/merged/<tag>")
    ap.add_argument("--shard", type=int, default=-1, help="-1 = $SLURM_ARRAY_TASK_ID (0 outside)")
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=-1, help="cap rows this shard (pilots)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=512, help="rows per llm.generate call (vllm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topup", type=str, default="", help="shortfall json from assemble_distill")
    ap.add_argument("--round", type=int, default=1, help="topup round number (output prefix)")
    ap.add_argument("--temp", type=float, default=1.0, help="sampling temperature (pilot arms)")
    ap.add_argument(
        "--prompt-variant",
        type=str,
        default="",
        help="prompts mode: render a desc-pilot prompt variant (e.g. precise) "
        "-> prompts.<variant>.json instead of the continuation prompt",
    )
    ap.add_argument(
        "--prompts-tag",
        type=str,
        default="",
        help="sample mode: read prompts.<tag>.json (r2sf-dp desc pools)",
    )
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="suffix on the samples/<variant> dir (temp arms, deflation steps)",
    )
    ap.add_argument(
        "--pairs-dir",
        type=str,
        default="",
        help="override the vec shard dir (deflation-step vectors); default r2 pairs",
    )
    ap.add_argument("--layer-map", type=str, default="")
    args = ap.parse_args()
    if args.mode == "prompts":
        do_prompts(args)
        return
    if not args.variant:
        raise SystemExit("--variant required for --mode sample")
    do_sample(args)


if __name__ == "__main__":
    main()
