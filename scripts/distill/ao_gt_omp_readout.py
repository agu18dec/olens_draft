"""Distillation selection + FVE report: 64 samples -> k=4 targets vs the TRUE activation.

Consumes the pilot sampler's output (``ao_distill_sample.py``): per item (GT activation at one
layer, 64 AO samples), cut prefixes at {2,4,8,16,32,64} tokens, F1-mask degenerate candidates,
AR-embed the uniques, and select k=4 by three strategies on identical rows:

  - ``omp``    non-negative OMP + NNLS refit over the item's own candidates (r2_select.omp_select)
  - ``staged`` one atom per prefix-length stage (4, 8, 16, 32) (omp_select_staged)
  - ``iid4``   first 4 distinct non-degenerate samples at the 32-token prefix (the r2s baseline)

Candidate units (``--cand-mode``): ``prefixes`` = the {2,4,8,16,32,64} ladder over each raw
sample; ``bullets`` = one atom per parsed '- ' bullet (whole, capped at 64 tokens);
``bullet_prefixes`` (Agam 2026-08-22) = the ladder over EACH bullet plus the full bullet, so
NNOMP + ``--shrink`` lands on the shortest bullet-prefix that gives the highest joint FVE.

Scoring (conventions of record): atoms = unit_rows(whiten(AR(p)))[layer], query = whiten(h_true)
un-normalized; FVE = nnls_refit (nonneg). Ceiling = 1-atom NNLS of AR(true span); floor =
rolled-target NNLS at the same k.

    # shard the AR embedding over GPUs:
    CUDA_VISIBLE_DEVICES=g uv run --no-sync python scripts/distill/ao_gt_omp_readout.py \
        --out-dir distill_u64/pilot --arm normmatched --n-shards 4 --shard g
    # then: --aggregate (CPU) prints the tables from the per-shard JSONs
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

AR_RUN = "ar.chat.mlayer.lc.s0/ex16014240"
PREFIXES = (2, 4, 8, 16, 32, 64)
STAGES = (4, 8, 16, 32)
K = 4
SHRINK_EPS = 0.005  # joint-FVE budget for the whole shrink pass (see --shrink)
BANDS = {"L20-32": range(20, 33), "L36-48": range(36, 49), "L52-60": range(52, 61)}


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)


def parse_bullets(text: str) -> list[str]:
    """Split a bullet-student generation into its '- ' items (continuation lines attached).

    The candidate unit for ``--cand-mode bullets`` (Agam 2026-08-11: sample the ROUND-3 bullet
    student and run NNOMP over its bullets instead of continuation prefixes — candidates become
    native concept units, no mid-word/leading-punct fragments)."""
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
    return [b.strip() for b in out if b.strip()]


def build_candidates(
    samples: list[str],
    tok: object,
    *,
    mode: str,
    prefixes: tuple[int, ...],
    is_degen: object = None,
) -> tuple[list[list[int]], list[int], list[tuple[int, int]]]:
    """Candidate atoms for one item: (token-id lists, lengths, (sample_idx, bullet_idx)).

    ``bullet_idx`` is -1 in ``prefixes`` mode. Dedup is on the token-id tuple (first
    occurrence keeps its provenance); degenerate decodes are masked. ``bullet_prefixes``
    emits, per bullet, every ladder length below the bullet's (64-capped) length PLUS the
    full bullet — so a shorter bullet-prefix is always available for shrink to land on.
    ``is_degen`` is injectable for CPU tests (defaults to ``r2_filter.is_degenerate``).
    """
    if is_degen is None:
        from oracle_lens.pipeline.r2_filter import is_degenerate as is_degen
    cand_ids: list[list[int]] = []
    cand_len: list[int] = []
    cand_src: list[tuple[int, int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(ids: list[int], src: tuple[int, int]) -> None:
        key = tuple(ids)
        if key in seen:
            return
        txt = tok.decode(list(key))  # type: ignore[attr-defined]
        if is_degen(txt):  # type: ignore[operator]
            return
        seen.add(key)
        cand_ids.append(list(ids))
        cand_len.append(len(ids))
        cand_src.append(src)

    cap = max(prefixes)
    for si, s in enumerate(samples):
        if mode in ("bullets", "bullet_prefixes"):
            for bi, b in enumerate(parse_bullets(s or "")):
                ids = tok(b, add_special_tokens=False)["input_ids"][:cap]  # type: ignore[operator]
                if len(ids) < 2:
                    continue
                if mode == "bullets":
                    add(ids, (si, bi))
                else:
                    for n in sorted({p for p in prefixes if p < len(ids)} | {len(ids)}):
                        add(ids[:n], (si, bi))
        else:
            ids = tok(s or "", add_special_tokens=False)["input_ids"]  # type: ignore[operator]
            for n in prefixes:
                if len(ids) < n:
                    break
                add(ids[:n], (si, -1))
    return cand_ids, cand_len, cand_src


def run_shard(args: argparse.Namespace) -> None:
    prefixes = tuple(int(x) for x in args.prefixes.split(",")) if args.prefixes else PREFIXES
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from oracle_lens.core.nnomp import nnls_refit
    from oracle_lens.pipeline.ablation import WhitenedSpace
    from oracle_lens.pipeline.ar_loader import fetch_ar_checkpoint, load_reconstructor
    from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards_lazy
    from oracle_lens.pipeline.multilayer_reconstructor import PromptTagReconstructor, ml_collate
    from oracle_lens.pipeline.r2_filter import is_degenerate
    from oracle_lens.pipeline.r2_select import (
        omp_select,
        omp_select_staged,
        shrink_to_shortest,
        unit_rows,
    )

    root = ola_root()
    out = root / args.out_dir
    pconf = json.loads((out / "pconf.json").read_text())
    items = pconf["items"]
    lo = len(items) * args.shard // args.n_shards
    hi = len(items) * (args.shard + 1) // args.n_shards

    # final shard files AND resumable .part files (a capped generation run leaves only parts;
    # items with no texts are skipped downstream as "too few candidates")
    texts: dict[int, list[str]] = {}
    for p in sorted(out.glob(f"texts_{args.arm}_*.json")):
        d = json.loads(p.read_text())
        for base, row in enumerate(d["rows"]):
            texts[d["lo"] + base] = row["samples"]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    pairs, _ = load_multilayer_shards_lazy(
        sorted((root / pconf["pairs_dir"]).glob("pairs_train_*.safetensors"))
    )
    recon = load_reconstructor(fetch_ar_checkpoint(args.ar_run, dest=root / "hf_ckpts"))
    ar_layers = list(recon.layers)
    is_ptag = isinstance(recon, PromptTagReconstructor)
    spaces: dict[int, WhitenedSpace] = {}

    def space(ly: int) -> WhitenedSpace:
        if ly not in spaces:
            t = load_file(str(root / f"{args.whitener_prefix}_L{ly}.safetensors"), device="cpu")
            spaces[ly] = WhitenedSpace.from_moments(t["mu"], t["cov"], ridge_c=0.1).to("cuda")
        return spaces[ly]

    def ar_embed(ids_list: list[list[int]], li: int) -> torch.Tensor:
        """Raw AR images at ONE trained layer (index ``li`` into ``recon.layers``) -> [n, d]."""
        preds = torch.zeros(len(ids_list), 5120)
        with torch.no_grad():
            for b in range(0, len(ids_list), 256):
                chunk = ids_list[b : b + 256]
                batch = ml_collate(
                    [{"ids": torch.tensor(r), "target": torch.zeros(1)} for r in chunk], pad_id=0
                )
                ids_c = batch["input_ids"].cuda()
                mask_c = batch["attention_mask"].cuda()
                if is_ptag:
                    # one tagged forward at li ([b,1,d]); layer_idx=None would cost
                    # n_layers forwards per batch for rows we'd immediately slice away
                    pred = recon(ids_c, mask_c, layer_idx=li)[:, 0]
                else:
                    pred = recon(ids_c, mask_c)[:, li]
                preds[b : b + len(chunk)] = pred.float().cpu()
        return preds

    results = []
    for gi in range(lo, hi):
        it = items[gi]
        ly = it["layer"]
        samples = texts.get(gi) or []
        # candidates: see build_candidates (ladder/bullets/bullet-prefixes; F1 mask; id dedup)
        cand_ids, cand_len, cand_src = build_candidates(
            samples, tok, mode=args.cand_mode, prefixes=prefixes, is_degen=is_degenerate
        )
        cand_sample = [s for s, _ in cand_src]
        # per-source (sample, bullet) longest candidate — the "how much was shrunk" denominator
        full_len: dict[tuple[int, int], int] = {}
        for ci, src in enumerate(cand_src):
            full_len[src] = max(full_len.get(src, 0), cand_len[ci])
        if len(cand_ids) < K:
            results.append({"item": gi, "layer": ly, "skipped": "too few candidates"})
            continue

        li = ar_layers.index(ly)
        vecs = ar_embed(cand_ids, li)  # [c, d] raw AR images at the item's layer
        sp = space(ly)
        if args.query_source == "ar":
            # end-to-end AR control (Agam 2026-08-12): the selection target is the AR image of
            # the source span, never the true residual — the fully self-consistent proxy arm
            span_ids = pairs.span_ids[
                int(pairs.offsets[it["pair_row"]]) : int(pairs.offsets[it["pair_row"] + 1])
            ].tolist()[:64]
            h = ar_embed([span_ids], li)[0].cuda()
        else:
            h = torch.as_tensor(pairs.targets[it["pair_row"]][LAYERS.index(ly)]).float().cuda()
        x_w = sp.whiten(h.unsqueeze(0))  # [1, d]
        dirs_w = unit_rows(sp.whiten(vecs.cuda()))  # [c, d]
        valid = torch.ones(1, len(cand_ids), dtype=torch.bool, device="cuda")

        row: dict = {"item": gi, "layer": ly, "n_cand": len(cand_ids), "span_len": it["span_len"]}
        # omp trajectory: k=1..3 are diagnostics (skipped by --lean); k=4 is the selection
        traj = []
        if not args.lean:
            for k in range(1, K):
                _sel, _coef, fve = omp_select(
                    x_w, dirs_w.unsqueeze(0), valid, k=k, min_gain=args.min_gain
                )
                traj.append(round(float(fve[0]), 4))
        sel4, _, fve4 = omp_select(x_w, dirs_w.unsqueeze(0), valid, k=K, min_gain=args.min_gain)
        traj.append(round(float(fve4[0]), 4))
        row["omp_fve_traj"] = traj
        def sel_entry(
            j: int, ids=cand_ids, lens=cand_len, srcs=cand_src, full=full_len
        ) -> dict:
            return {
                "text": tok.decode(ids[j]),
                "len": lens[j],
                "src": list(srcs[j]),
                "full_len": full[srcs[j]],
            }

        picks = [int(j) for j in sel4[0].tolist() if j >= 0]
        row["omp_sel"] = [sel_entry(j) for j in picks]
        if args.shrink and args.cand_mode in ("prefixes", "bullet_prefixes") and picks:
            # Shrink-to-shortest (Agam 2026-08-11 "the shortest prefix that reconstructs it
            # best"; bullet_prefixes 2026-08-23 "the shortest possible bullet which gives the
            # highest fve") — math in r2_select.shrink_to_shortest, one global eps budget.
            sel, sh_fve = shrink_to_shortest(
                picks, float(traj[-1]),
                cand_ids, cand_len, dirs_w, x_w,
                ladder=prefixes, eps=args.shrink_eps,
            )
            row["shrunk_fve"] = round(sh_fve, 4)
            row["shrunk_sel"] = [sel_entry(j) for j in sel]
        # staged: one atom per length stage
        stage_valid = [
            (torch.tensor([n == st for n in cand_len], device="cuda").unsqueeze(0) & valid)
            for st in STAGES
        ]
        if not args.lean and all(bool(sv.any()) for sv in stage_valid):
            ssel, _, sfve = omp_select_staged(x_w, dirs_w.unsqueeze(0), stage_valid)
            row["staged_fve"] = round(float(sfve[0]), 4)
            row["staged_sel"] = [
                {"text": tok.decode(cand_ids[int(j)]), "len": cand_len[int(j)]}
                for j in ssel[0].tolist() if j >= 0
            ]
        if not args.lean:
            # iid-4: first 4 distinct samples' 32-token prefixes
            iid_idx, used = [], set()
            for ci in range(len(cand_ids)):
                if cand_len[ci] == 32 and cand_sample[ci] not in used:
                    iid_idx.append(ci)
                    used.add(cand_sample[ci])
                if len(iid_idx) == K:
                    break
            if len(iid_idx) == K:
                v = dirs_w[iid_idx].unsqueeze(0)
                _, fve = nnls_refit(v, x_w, torch.ones(1, K, dtype=torch.bool, device="cuda"))
                row["iid4_fve"] = round(float(fve[0]), 4)
            # ceiling: 1-atom NNLS of AR(true span); floor: omp sel vs ROLLED target
            span = pairs.span_ids[
                int(pairs.offsets[it["pair_row"]]) : int(pairs.offsets[it["pair_row"] + 1])
            ].tolist()[:64]
            tv = unit_rows(sp.whiten(ar_embed([span], li).cuda()))
            ones1 = torch.ones(1, 1, dtype=torch.bool, device="cuda")
            _, cfve = nnls_refit(tv.unsqueeze(0), x_w, ones1)
            row["ceiling_fve"] = round(float(cfve[0]), 4)
        results.append(row)
        if (gi - lo) % 50 == 0:
            print(f"[omp] {gi - lo}/{hi - lo} items", flush=True)

    dst = out / f"select_{args.arm}{args.tag}_{args.shard:04d}.json"
    dst.write_text(json.dumps(results))
    print(f"[omp] wrote {dst} ({len(results)} items)", flush=True)


def aggregate(args: argparse.Namespace) -> None:
    import statistics as st

    root = ola_root()
    out = root / args.out_dir
    for arm in args.arms.split(","):
        rows = []
        for p in sorted(out.glob(f"select_{arm}_*.json")):
            rows += json.loads(p.read_text())
        ok = [r for r in rows if "omp_fve_traj" in r]
        skipped = len(rows) - len(ok)

        def m(key: str, sel: list) -> float:
            vals = [r[key] if not isinstance(r.get(key), list) else r[key][-1]
                    for r in sel if r.get(key) is not None]
            return st.mean(vals) if vals else float("nan")

        print(f"\n===== arm {arm}: {len(ok)} items (skipped {skipped}) =====")
        print(f"  FVE mean:  omp(k4) {m('omp_fve_traj', ok):.4f} | "
              f"shrunk {m('shrunk_fve', ok):.4f} | "
              f"staged {m('staged_fve', ok):.4f} | iid4 {m('iid4_fve', ok):.4f} | "
              f"ceiling(1-atom true-span) {m('ceiling_fve', ok):.4f}")
        traj = [st.mean([r["omp_fve_traj"][k] for r in ok]) for k in range(K)]
        print(f"  omp trajectory k=1..4: {[round(t, 4) for t in traj]}")
        for band, rng in BANDS.items():
            sel = [r for r in ok if r["layer"] in rng]
            print(f"  {band} (n={len(sel)}): omp {m('omp_fve_traj', sel):.4f} | "
                  f"staged {m('staged_fve', sel):.4f} | iid4 {m('iid4_fve', sel):.4f} | "
                  f"ceil {m('ceiling_fve', sel):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", default="normmatched")
    ap.add_argument("--prefixes", default="", help="csv override of prefix lengths (e.g. 2,4,8)")
    ap.add_argument("--tag", default="", help="suffix for output files (restriction experiments)")
    ap.add_argument("--query-source", default="gt", choices=["gt", "ar"],
                    help="selection target: true residual (gt) or the span's AR image (ar — "
                         "the end-to-end proxy control)")
    ap.add_argument("--cand-mode", default="prefixes",
                    choices=["prefixes", "bullets", "bullet_prefixes"],
                    help="candidate units: continuation prefixes (teacher gens), parsed '- ' "
                         "bullets whole (bullet-student gens; shrink is a no-op there), or "
                         "every bullet's prefix ladder + full bullet (bullet_prefixes — "
                         "shrink lands on the shortest bullet-prefix at highest FVE)")
    ap.add_argument("--lean", action="store_true",
                    help="production mode: k=4 NNOMP (+ --shrink) only — skip the k<4 "
                         "trajectory, staged, iid4 and ceiling diagnostic arms")
    ap.add_argument("--shrink", action="store_true",
                    help="post-OMP: swap each pick for its shortest same-source token-prefix "
                         "that keeps the joint refit-FVE within --shrink-eps of the OMP "
                         "set's (prefixes / bullet_prefixes modes)")
    ap.add_argument("--shrink-eps", type=float, default=SHRINK_EPS,
                    help="one global joint-FVE budget for the whole shrink pass "
                         "(0 = only swaps that never drop below the OMP set's FVE)")
    ap.add_argument("--min-gain", type=float, default=1e-3,
                    help="OMP stop threshold: a pick must add at least this much joint FVE "
                         "(raise it, e.g. 0.005, to let items settle at 3 or fewer bullets "
                         "when the AR says the 4th doesn't matter — pair with the "
                         "assembler's --min-picks)")
    ap.add_argument("--ar-run", default=AR_RUN,
                    help="AR checkpoint for atoms+query embeddings — a rung dir like "
                         "ar.asst.ptag.pooled.s0/ex<N> (ptag rungs need their meta.json)")
    ap.add_argument("--whitener-prefix", default="whitening_iolens_chat",
                    help="whitener family — MUST match the AR's training basis "
                         "(pooled/ptag cell: whitening_iolens_pooled)")
    ap.add_argument("--arms", default="normmatched,frozen", help="aggregate mode")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
