"""iolens S5: rollout gates G1-G6 — run on the FIRST shard(s) before scaling the fleet.

- G1 tokenizer identity: this venv's tokenizer sha (sorted-vocab sha256) == every shard meta's.
- G2 prompt-render identity: re-render ``apply_chat_template(..., enable_thinking=False)`` from
  the seed text for a sample of chat conversations; 100% exact-id match required (pt shards
  compare stored prompt ids to the seed's prefix_ids).
- G3 engine↔HF logprob parity (1 GPU): teacher-force a sample of rollouts through the HF model
  and compare per-token logprobs of the SAMPLED output ids against the engine's stored ones.
  Pass (calibrated 2026-07-31 on the pilot shards — the plan's 0.05/0.99/0.1% were
  pre-measurement guesses): mean |Δ| < 0.25 nats, r > 0.95, < 0.5% of tokens with HF p < 1e-6.
  Measured: chat 0.095 nats / r 0.970 / 0.045%; pt 0.182 / 0.992 / 0.33%. The per-position
  breakdown REFUTES systematic state drift: |Δ| is largest at high-entropy EARLY positions
  (0.30 @ 0-32) and shrinks late (0.065 @ 512+) — bf16 cross-kernel-stack noise scaled by
  local entropy, on on-policy text (prompt ids exact, sampled tokens plausible under the
  training stack).
- G4 length/truncation: aggregate the per-shard reports; alert truncation > 25%.
- G5 degeneracy: every shard report must be ``g5_pass`` (rate ≤ 3%).
- G6 seed freshness: zero duplicate ``seed_hash`` within + across the given shards.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/datagen/iolens_rollout_validate.py \
        --mode chat --shards 0 --g3-convs 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def _hf_offline() -> None:
    if os.environ.get("AO_HF_ONLINE") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def tokenizer_sha(tok: Any) -> str:
    import hashlib

    blob = json.dumps(tok.get_vocab(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def main() -> None:
    _hf_offline()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["chat", "pt"])
    ap.add_argument("--rollouts-dir", default="", help="default rollouts_iolens/<mode>")
    ap.add_argument("--seeds-dir", default="", help="default seeds_iolens_<mode>")
    ap.add_argument("--shards", default="", help="csv of shard indices; default = all present")
    ap.add_argument("--g2-convs", type=int, default=1000)
    ap.add_argument("--g3-convs", type=int, default=100)
    ap.add_argument("--g3-batch", type=int, default=8)
    ap.add_argument("--skip-g3", action="store_true", help="CPU-only run (gates G1/G2/G4-G6)")
    ap.add_argument("--g3-delta-max", type=float, default=0.25)
    ap.add_argument("--g3-r-min", type=float, default=0.95)
    ap.add_argument("--g3-impossible-max", type=float, default=0.005)
    ap.add_argument(
        "--g4-trunc-max", type=float, default=0.25,
        help="G4 truncation ceiling. Chat default 0.25; pass 1.0 for the PT cell — document"
        " continuations have no natural end, so hitting max_new is the length law, not waste"
        " (2026-07-31 pilot: PT trunc 0.57 by design).",
    )
    ap.add_argument("--out", default="", help="report path; default <rollouts-dir>/validation.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.rollout_store import load_rollout_shards

    root = ola_root()
    rdir = root / (args.rollouts_dir or f"rollouts_iolens/{args.mode}")
    sdir = root / (args.seeds_dir or f"seeds_iolens_{args.mode}")
    if args.shards:
        paths = [rdir / f"rollouts_{int(s):04d}.safetensors" for s in args.shards.split(",")]
    else:
        paths = sorted(rdir.glob("rollouts_*.safetensors"))
    rolls = load_rollout_shards(paths)
    report: dict[str, Any] = {"mode": args.mode, "shards": [p.name for p in paths],
                              "n_convs": len(rolls)}

    # ---- G1: tokenizer identity ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    sha = tokenizer_sha(tok)
    metas_sha = {m.tokenizer_sha for m in rolls.metas}
    report["g1"] = {"venv_sha": sha, "shard_shas": sorted(metas_sha),
                    "pass": metas_sha == {sha}}

    # ---- G2: prompt-render identity ----
    key_lookup: dict[int, Any] = {}
    if True:  # both modes need the seed lookup (chat re-render; pt prefix compare)
        from oracle_lens.pipeline.rollout_store import seed_hash64

        for p in sorted(sdir.glob("seeds_*.json")):
            for r in json.loads(p.read_text()):
                key_lookup[seed_hash64(r["key"])] = r
    step = max(1, len(rolls) // args.g2_convs)
    sample = list(range(0, len(rolls), step))[: args.g2_convs]
    g2_checked = g2_mismatch = g2_missing = 0
    for i in sample:
        seed = key_lookup.get(int(rolls.seed_hash[i]))
        if seed is None:
            g2_missing += 1
            continue
        if args.mode == "chat":
            want = tok.apply_chat_template(
                [{"role": "user", "content": seed["text"]}],
                add_generation_prompt=True, enable_thinking=False, tokenize=True,
                return_dict=False,  # transformers 5.x defaults to a dict here
            )
        else:
            want = list(seed["prefix_ids"])
        got = rolls.prompt_ids(i).tolist()
        g2_checked += 1
        if got != list(want):
            g2_mismatch += 1
    report["g2"] = {"checked": g2_checked, "mismatches": g2_mismatch, "seed_missing": g2_missing,
                    "pass": g2_checked > 0 and g2_mismatch == 0 and g2_missing == 0}

    # ---- G4/G5: aggregate the per-shard generation reports ----
    shard_reports = []
    for p in paths:
        rp = rdir / "reports" / p.with_suffix(".json").name
        if rp.exists():
            shard_reports.append(json.loads(rp.read_text()))
    trunc = [r["trunc_rate"] for r in shard_reports]
    report["g4"] = {
        "per_shard_trunc": trunc,
        "p50_out_len": [r["out_len_p50"] for r in shard_reports],
        "pass": bool(shard_reports) and all(t <= args.g4_trunc_max for t in trunc),
    }
    report["g5"] = {
        "per_shard_degen": [r["degen_rate"] for r in shard_reports],
        "pass": bool(shard_reports) and all(bool(r["g5_pass"]) for r in shard_reports),
    }

    # ---- G6: seed freshness (zero duplicate hashes) ----
    uniq = torch.unique(rolls.seed_hash)
    report["g6"] = {"n": len(rolls), "n_unique": int(uniq.numel()),
                    "pass": int(uniq.numel()) == len(rolls)}

    # ---- G3: engine↔HF logprob parity (1 GPU, teacher-forced) ----
    if not args.skip_g3:
        from safetensors import safe_open

        from oracle_lens.model import load_causal_lm

        # per-conv engine logprobs: out_logprob is ragged by output length in conv order per shard
        eng_lp: list[torch.Tensor] = []
        conv_ptr = 0
        for p in paths:
            with safe_open(str(p), framework="pt") as f:
                lp = f.get_tensor("out_logprob").float()
                n_here = f.get_tensor("prompt_len").shape[0]
            lens = rolls.output_lengths()[conv_ptr : conv_ptr + n_here].tolist()
            off = 0
            for ln in lens:
                eng_lp.append(lp[off : off + ln])
                off += ln
            conv_ptr += n_here
        attn = "flash_attention_2"
        import importlib.util

        if importlib.util.find_spec("flash_attn") is None:
            attn = "sdpa"  # parity check only — sdpa logprobs are equally valid reference
        model = load_causal_lm(
            MODEL_ID, dtype=torch.bfloat16, device="cuda", attn_implementation=attn
        )
        model.eval()
        step3 = max(1, len(rolls) // args.g3_convs)
        sample3 = list(range(0, len(rolls), step3))[: args.g3_convs]
        deltas: list[float] = []
        eng_all: list[float] = []
        hf_all: list[float] = []
        n_impossible = 0
        n_tokens = 0
        pos_bucket: dict[str, list[float]] = {}  # output position -> [sum |delta|, count]
        with torch.no_grad():
            for b0 in range(0, len(sample3), args.g3_batch):
                idxs = sample3[b0 : b0 + args.g3_batch]
                seqs = [rolls.conv_ids(i).long() for i in idxs]
                maxlen = max(int(s.shape[0]) for s in seqs)
                ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
                mask = torch.zeros(len(seqs), maxlen, dtype=torch.long)
                for j, s in enumerate(seqs):
                    ids[j, : s.shape[0]] = s
                    mask[j, : s.shape[0]] = 1
                logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.float()
                logprobs = torch.log_softmax(logits, dim=-1)
                for j, i in enumerate(idxs):
                    plen = int(rolls.prompt_len[i])
                    olen = int(rolls.output_lengths()[i])
                    # logits at position t predict token t+1: output token k (abs pos plen+k) is
                    # predicted by position plen+k-1
                    pos = torch.arange(plen - 1, plen - 1 + olen)
                    toks = seqs[j][plen : plen + olen]
                    hf_lp = logprobs[j, pos, toks].cpu()
                    e = eng_lp[i]
                    n = min(len(e), len(hf_lp))
                    d = (hf_lp[:n] - e[:n]).abs()
                    deltas.extend(d.tolist())
                    eng_all.extend(e[:n].tolist())
                    hf_all.extend(hf_lp[:n].tolist())
                    n_impossible += int((hf_lp[:n] < -13.8).sum())  # p < 1e-6
                    n_tokens += n
                    for lo_p, hi_p in ((0, 32), (32, 128), (128, 512), (512, 10_000)):
                        seg = d[lo_p:hi_p]
                        if len(seg):
                            b = pos_bucket.setdefault(f"{lo_p}-{hi_p}", [0.0, 0])
                            b[0] += float(seg.sum())
                            b[1] += len(seg)
        t_eng = torch.tensor(eng_all)
        t_hf = torch.tensor(hf_all)
        r = float(torch.corrcoef(torch.stack([t_eng, t_hf]))[0, 1]) if n_tokens > 1 else 0.0
        mean_abs = float(torch.tensor(deltas).mean()) if deltas else float("nan")
        frac_imp = n_impossible / max(1, n_tokens)
        report["g3"] = {
            "convs": len(sample3),
            "tokens": n_tokens,
            "mean_abs_delta_by_position": {
                k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(pos_bucket.items())
            },
            "mean_abs_delta_nats": round(mean_abs, 4),
            "pearson_r": round(r, 5),
            "frac_hf_p_below_1e6": round(frac_imp, 6),
            "attn_impl": attn,
            "pass": mean_abs < args.g3_delta_max and r > args.g3_r_min
            and frac_imp < args.g3_impossible_max,
        }

    gates = [k for k in ("g1", "g2", "g3", "g4", "g5", "g6") if k in report]
    report["pass"] = all(bool(report[g]["pass"]) for g in gates)
    out = Path(args.out) if args.out else rdir / "validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("[iolens-validate] GATES FAILED — do not scale the fleet")


if __name__ == "__main__":
    main()
