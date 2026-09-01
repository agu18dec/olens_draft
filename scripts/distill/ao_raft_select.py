"""RAFT selection: score WHOLE readouts by joint bullet-FVE, keep the best one per item.

The rejection-sampling baseline (Agam 2026-08-12): unlike the distill assembler — which stitches
the best BULLETS across many samples into composite targets — RAFT keeps each item's single best
COMPLETE readout, verbatim, so SFT reinforces trajectories the policy actually produced. Score =
whitened non-negative joint NNLS-FVE of the readout's parsed bullets vs the true activation
(identical math to ao_student_fve.py; this IS the future RL reward).

    CUDA_VISIBLE_DEVICES=g uv run --no-sync python scripts/distill/ao_raft_select.py \
        --out-dir distill_u64/raft --n-shards 4 --shard g
    # then --aggregate prints the pass@k table and writes raft training rows
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

AR_RUN = "ar.chat.mlayer.lc.s0/ex16014240"
MAX_BULLETS = 6


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)


def parse_bullets(text: str) -> list[str]:
    out: list[str] = []
    cur: str | None = None
    for ln in text.split("\n"):
        if ln.startswith("- "):
            if cur is not None:
                out.append(cur)
            cur = ln[2:]
        elif cur is not None:
            cur += "\n" + ln
    if cur is not None:
        out.append(cur)
    return [b.strip() for b in out if b.strip()][:MAX_BULLETS]


def run_shard(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from oracle_lens.core.nnomp import nnls_refit
    from oracle_lens.pipeline.ablation import WhitenedSpace
    from oracle_lens.pipeline.ar_loader import fetch_ar_checkpoint, load_lc_reconstructor
    from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards_lazy
    from oracle_lens.pipeline.multilayer_reconstructor import ml_collate
    from oracle_lens.pipeline.r2_select import unit_rows

    root = ola_root()
    out = root / args.out_dir
    pconf = json.loads((out / "pconf.json").read_text())
    items = pconf["items"]
    lo = len(items) * args.shard // args.n_shards
    hi = len(items) * (args.shard + 1) // args.n_shards

    texts: dict[int, list[str]] = {}
    for p in sorted(out.glob("texts_normmatched_*.json")):
        d = json.loads(p.read_text())
        for base, row in enumerate(d["rows"]):
            texts[d["lo"] + base] = row["samples"]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    pairs, _ = load_multilayer_shards_lazy(
        sorted((root / pconf["pairs_dir"]).glob("pairs_train_*.safetensors"))
    )
    recon = load_lc_reconstructor(fetch_ar_checkpoint(AR_RUN, dest=root / "hf_ckpts"))
    ar_layers = list(LAYERS[-int(recon.layer_emb.weight.shape[0]):])
    spaces: dict[int, WhitenedSpace] = {}

    def space(ly: int) -> WhitenedSpace:
        if ly not in spaces:
            t = load_file(str(root / f"whitening_iolens_chat_L{ly}.safetensors"), device="cpu")
            spaces[ly] = WhitenedSpace.from_moments(t["mu"], t["cov"], ridge_c=0.1).to("cuda")
        return spaces[ly]

    def ar_embed(ids_list: list[list[int]]) -> torch.Tensor:
        preds = torch.zeros(len(ids_list), len(ar_layers), 5120)
        with torch.no_grad():
            for b in range(0, len(ids_list), 256):
                chunk = ids_list[b : b + 256]
                batch = ml_collate(
                    [{"ids": torch.tensor(r), "target": torch.zeros(1)} for r in chunk], pad_id=0
                )
                preds[b : b + len(chunk)] = (
                    recon(batch["input_ids"].cuda(), batch["attention_mask"].cuda())
                    .float().cpu()
                )
        return preds

    results = []
    for gi in range(lo, hi):
        it = items[gi]
        samples = texts.get(gi) or []
        if not samples:
            continue
        ly = it["layer"]
        li = ar_layers.index(ly)
        sp = space(ly)
        h = torch.as_tensor(pairs.targets[it["pair_row"]][LAYERS.index(ly)]).float().cuda()
        x_w = sp.whiten(h.unsqueeze(0))
        # embed every readout's bullets in ONE batch per item
        per_sample: list[tuple[int, int]] = []  # (start, n_bullets) into the flat ids list
        flat_ids: list[list[int]] = []
        for s in samples:
            bs = [b for b in parse_bullets(s)
                  if len(tok(b, add_special_tokens=False)["input_ids"]) >= 2]
            ids = [tok(b, add_special_tokens=False)["input_ids"][:64] for b in bs]
            per_sample.append((len(flat_ids), len(ids)))
            flat_ids.extend(ids)
        if not flat_ids:
            continue
        vecs = ar_embed(flat_ids)[:, li].cuda()
        fves = []
        for start, nb in per_sample:
            if nb == 0:
                fves.append(0.0)
                continue
            dirs_w = unit_rows(sp.whiten(vecs[start : start + nb]))
            mask = torch.ones(1, nb, dtype=torch.bool, device="cuda")
            _, fve = nnls_refit(dirs_w.unsqueeze(0), x_w, mask)
            fves.append(round(float(fve[0]), 4))
        best = max(range(len(fves)), key=lambda i: fves[i])
        results.append({"item": gi, "layer": ly, "pair_row": it["pair_row"],
                        "best_idx": best, "best_fve": fves[best], "fves": fves,
                        "best_text": samples[best]})
        if (gi - lo) % 200 == 0:
            print(f"[raft] {gi - lo}/{hi - lo}", flush=True)
    dst = out / f"raft_select_{args.shard:04d}.json"
    dst.write_text(json.dumps(results))
    print(f"[raft] wrote {dst} ({len(results)} items)", flush=True)


def aggregate(args: argparse.Namespace) -> None:
    import statistics as st

    root = ola_root()
    out = root / args.out_dir
    rows = []
    for p in sorted(out.glob("raft_select_*.json")):
        rows += json.loads(p.read_text())
    ks = (1, 2, 4, 8, 16)
    print(f"{len(rows)} items — pass@k mean FVE (best of first k samples):")
    for k in ks:
        vals = [max(r["fves"][:k]) for r in rows if len(r["fves"]) >= k]
        print(f"  k={k:2d}: {st.mean(vals):.4f}")
    print(f"  mean single-sample FVE (all): "
          f"{st.mean(f for r in rows for f in r['fves']):.4f}")


def assemble(args: argparse.Namespace) -> None:
    """Winners -> distill_v1 shards: the best COMPLETE readout per item, VERBATIM (that is the
    whole point of RAFT — no re-formatting, no trim, no bullet surgery). Kept: parses >=2
    bullets and best_fve in the top --keep-frac of winners (rejection sampling keeps the top
    of the reward distribution, not everything that won its own pool)."""
    import subprocess

    import torch
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.distill_shards import (
        DistillPairs,
        DistillShardMeta,
        save_distill_shard,
    )
    from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards_lazy

    root = ola_root()
    out = root / args.out_dir
    pconf = json.loads((out / "pconf.json").read_text())
    rows = []
    for p in sorted(out.glob("raft_select_*.json")):
        rows += json.loads(p.read_text())
    rows = [r for r in rows if len(parse_bullets(r["best_text"])) >= 2]
    rows.sort(key=lambda r: -r["best_fve"])
    kept = rows[: int(len(rows) * args.keep_frac)]
    import statistics as st

    print(f"[raft-asm] {len(rows)} parseable winners -> kept top {len(kept)} "
          f"(mean best_fve {st.mean(r['best_fve'] for r in kept):.4f})")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    eos = int(tok.eos_token_id)
    pairs, _ = load_multilayer_shards_lazy(
        sorted((root / pconf["pairs_dir"]).glob("pairs_train_*.safetensors"))
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).parent).strip()
    except Exception:
        commit = "unknown"
    splits: dict[str, list[dict]] = {"train": [], "eval": []}
    for j, r in enumerate(kept):
        splits["eval" if j % 20 == 0 else "train"].append(r)
    for split, entries in splits.items():
        out_dir = out / "shards" / "raftv1" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        flat: list[int] = []
        offsets = [0]
        vecs, layer_col, src_col = [], [], []
        for r in entries:
            ids = [*tok(r["best_text"], add_special_tokens=False)["input_ids"], eos]
            flat.extend(ids)
            offsets.append(len(flat))
            vecs.append(torch.as_tensor(
                pairs.targets[int(r["pair_row"])])[LAYERS.index(int(r["layer"]))].float())
            layer_col.append(LAYERS.index(int(r["layer"])))
            src_col.append(int(r["pair_row"]))
        dp = DistillPairs(
            target_ids=torch.tensor(flat, dtype=torch.int32),
            offsets=torch.tensor(offsets, dtype=torch.int64),
            vec=torch.stack(vecs),
            layer_idx=torch.tensor(layer_col, dtype=torch.int64),
            src_row=torch.tensor(src_col, dtype=torch.int64),
        )
        meta = DistillShardMeta(
            model_id="Qwen/Qwen3.6-27B", teacher_run=pconf["teacher"], variant="raftv1",
            k=4, n=64, prompt_mode="concepts_raw", pairs_dir=pconf["pairs_dir"], split=split,
            seed=pconf["seed"], git_commit=commit, n_rows=len(dp),
        )
        save_distill_shard(out_dir / "distill_0000.safetensors", dp, meta)
        print(f"[raft-asm] {split}: {len(dp)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--keep-frac", type=float, default=0.5)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args)
    elif args.assemble:
        assemble(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
