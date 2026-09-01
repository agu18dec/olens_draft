"""r2s selection pipeline (GPU stages): AR phrase embedding + NNOMP/deflation selection.

Operates on the shared teacher-sample pool written by ``olens_distill_sample_cluster.py``
(variant ``pool16n32`` / ``pool64n32`` and the deflation-step candidate dirs
``k4n32d16.step{S}``). See ``ola.r2_select`` for the math and the method glossary.

Modes (all skip-if-exists, resumable; project venv, 1 GPU):

- ``embed``      one sample shard -> AR images of its unique phrases at ALL 17 layers.
                 Output ``embeds/<vdir>/<split>/embed_XXXX.safetensors`` ("dirs" [17,U,d] fp16,
                 RAW activation space — whitening happens at selection time) + sidecar
                 ``embed_XXXX.json`` {"phrases", "rows": [{row, layer, cand}]}.
                 ``--prefix-lengths 2,4,8,16,32`` (r2sp) expands every sample into its prefix
                 truncations at those lengths — same shard format, one cand slot per
                 (length, sample), written to ``embeds/<vdir>.pfx`` so m1/m2/deflate-step get
                 free any-length selection with no selection-code change.
- ``m2``         within-pool greedy OMP (k=4) per row -> ``selections/m2/<split>.jsonl``.
- ``m1``         pool-wide dict NNOMP (k=4): ONE dictionary from the TRAIN pool (both splits
                 select against it; user decision 2026-08-01) -> ``selections/m1/<split>.jsonl``.
- ``deflate-step`` one chain step S in {0..3}: best-of-16 candidate pick against the current
                 (deflated) vector, project it out, write next-step vec shards for the vllm
                 sampler (S<3) or the final chain record (S=3) -> ``m4/<split>/...`` +
                 ``selections/m4/<split>.jsonl``.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from oracle_lens.core.nnomp import nnls_refit, nnomp_batch
from oracle_lens.pipeline.ablation import WhitenedSpace, project_out_rows
from oracle_lens.pipeline.distill import VARIANTS, layer_draw, truncate_phrase
from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards_lazy
from oracle_lens.pipeline.r2_filter import DescLabels, is_degenerate, load_desc_labels
from oracle_lens.pipeline.r2_select import (
    best_step_pick,
    candidate_phrases,
    omp_select,
    omp_select_staged,
    prefix_candidate_phrases,
    unique_index,
    unit_rows,
)

M5_LENGTHS = (4, 8, 16, 32)  # coarse->fine ladder for the staged multi-length NNOMP (m5)

MODEL_ID = "Qwen/Qwen3.6-27B"
PAIRS_DIR = "r2_resid_pairs"
OUT_ROOT = "distill_n1024"
DEFAULT_AR = "ar.asst.on.mlayer.lc.alldata.v2cont.lr1e4.crop32.b512.s0"
DEFL_VARIANT = "k4n32d16"
K = 4


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def base_variant(vdir: str) -> str:
    """samples-dir name -> VARIANTS key (suffixes like '.step1'/'.t12' fall away).

    Underscore suffixes ('pool64n32_le64') fall away too, but only when the dotted base is
    not itself a VARIANTS key — the le64 round ran with pod-side script copies and main's
    lookup KeyError'd on the '_le64' dirs (found 2026-08-10, r2sp planning).
    """
    base = vdir.split(".")[0]
    return base if base in VARIANTS else base.split("_", 1)[0]


_DESC_CACHE: "DescLabels | None" = None


def _desc(args: argparse.Namespace) -> "DescLabels | None":
    """The r2sf-dp desc-band filter (gpt-5.5 CONT/DESC/JUNK labels), loaded once per run."""
    global _DESC_CACHE
    if not getattr(args, "desc_filter", ""):
        return None
    if _DESC_CACHE is None:
        _DESC_CACHE = load_desc_labels(args.desc_filter, band=args.desc_band)
        print(
            f"[r2s] desc-band filter: {len(_DESC_CACHE.by_row)} labeled band rows "
            f"(band {args.desc_band}) from {args.desc_filter}",
            flush=True,
        )
    return _DESC_CACHE


def _desc_keep_content(
    labels: "DescLabels | None", row: int, layer: int, content: str, tokenizer: Any
) -> bool:
    """Band-filter a FULL-length candidate by its n=32 judged-text key (the F3 judge labels
    phrases at the n=32 truncation, so full contents are keyed by their own truncation)."""
    if labels is None or not labels.in_band(int(layer)):
        return True
    return labels.keep(int(row), int(layer), truncate_phrase(content, tokenizer, n=32))


def tag_dir(args: argparse.Namespace) -> Path:
    return ola_root() / OUT_ROOT / str(args.teacher_run)


def load_space(layer: int, *, ridge_c: float, device: str) -> WhitenedSpace:
    from safetensors.torch import load_file

    t = load_file(str(ola_root() / f"whitening_L{layer}.safetensors"), device="cpu")
    return WhitenedSpace.from_moments(t["mu"], t["cov"], ridge_c=ridge_c).to(device)


def _atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sample_files(args: argparse.Namespace, vdir: str, split: str) -> list[Path]:
    d = tag_dir(args) / "samples" / vdir / split
    files = sorted(d.glob("samples_*.json"))
    if not files:
        raise SystemExit(f"no sample shards under {d}")
    return files


def embed_dir(args: argparse.Namespace, vdir: str, split: str) -> Path:
    return tag_dir(args) / "embeds" / vdir / split


# --------------------------------------------------------------------------- mode: embed


def do_embed(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.recon_precompute import load_layer_conditioned_ar
    from oracle_lens.pipeline.scorer import bare_token_ids

    shard = args.shard if args.shard >= 0 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    files = sample_files(args, args.variant_dir, args.split)
    if shard >= len(files):
        print(f"[r2s-embed] shard {shard} >= {len(files)} sample files — nothing to do")
        return
    src = files[shard]
    # r2sp free prefix selection: every prefix truncation of every sample is a candidate;
    # the expanded shards live under a '.pfx' dir so they never alias the plain n=32 embeds.
    plens = [int(x) for x in args.prefix_lengths.split(",")] if args.prefix_lengths else []
    out_vdir = args.variant_dir + (".pfx" if plens else "")
    out_dir = embed_dir(args, out_vdir, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem.replace("samples_", "embed_")
    out_st, out_js = out_dir / f"{stem}.safetensors", out_dir / f"{stem}.json"
    if out_st.exists() and out_js.exists():
        print(f"[r2s-embed] {out_st} exists — skipping")
        return

    n_cap = VARIANTS[base_variant(args.variant_dir)].n
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    payload = json.loads(src.read_text())
    phrases: list[str] = []
    where: dict[str, int] = {}
    rows_meta: list[dict[str, Any]] = []
    for row in payload:
        cands = (
            prefix_candidate_phrases(row["samples"], tokenizer, lengths=plens)
            if plens
            else candidate_phrases(row["samples"], tokenizer, n=n_cap)
        )
        uniq, idx = unique_index(cands)
        base: list[int] = []
        for u in uniq:
            if u not in where:
                where[u] = len(phrases)
                phrases.append(u)
            base.append(where[u])
        rows_meta.append(
            {
                "row": row["row"],
                "layer": row["layer"],
                "cand": [(base[i] if i >= 0 else -1) for i in idx],
            }
        )
    print(f"[r2s-embed] {src.name}: {len(payload)} rows, {len(phrases)} unique phrases")

    recon, _tok, d_model = load_layer_conditioned_ar(ola_root() / "ar_ckpts" / args.ar_run)
    pad_id = int(tokenizer.pad_token_id or 0)
    dirs = torch.zeros(len(LAYERS), len(phrases), d_model, dtype=torch.float16)
    with torch.no_grad():
        for lo in range(0, len(phrases), args.batch):
            chunk = phrases[lo : lo + args.batch]
            id_rows = [bare_token_ids(tokenizer, t) or [pad_id] for t in chunk]
            width = max(len(r) for r in id_rows)
            ids = torch.full((len(id_rows), width), pad_id, dtype=torch.long)
            mask = torch.zeros(len(id_rows), width, dtype=torch.long)
            for j, r in enumerate(id_rows):
                ids[j, : len(r)] = torch.tensor(r)
                mask[j, : len(r)] = 1
            out = recon(ids.to("cuda"), mask.to("cuda"))  # [b, n_layers, d]
            dirs[:, lo : lo + len(id_rows)] = out.transpose(0, 1).to(torch.float16).cpu()
            if (lo // args.batch) % 20 == 0:
                print(
                    f"[r2s-embed] {min(lo + args.batch, len(phrases))}/{len(phrases)}", flush=True
                )

    from safetensors.torch import save_file

    tmp = out_st.with_suffix(".tmp")
    meta = {"ar_run": args.ar_run, "n_phrases": str(len(phrases))}
    if plens:
        meta["prefix_lengths"] = args.prefix_lengths
    save_file({"dirs": dirs}, str(tmp), metadata=meta)
    tmp.replace(out_st)
    _atomic_json(out_js, {"phrases": phrases, "rows": rows_meta})
    print(f"[r2s-embed] saved {out_st} ({len(phrases)} x {len(LAYERS)} x {d_model})")


# --------------------------------------------------------------- mode: embed-mlen (for m6)


def do_embed_mlen(args: argparse.Namespace) -> None:
    """Per-length AR embeddings of the pool truncations, for the DICT multi-length arm (m6).

    For each L in ``--lengths`` (default 4,8,16 — L=32 reuses the plain ``embed`` shards) this
    truncates every pool sample to L tokens, dedups, and embeds the uniques → one shard under
    ``embeds_mlen/<vdir>/<split>/L{L}/embed_XXXX.{safetensors,json}`` (same {dirs,phrases}
    format as ``embed``, so ``EmbedShards`` reads it and ``do_m6`` builds a whole-pool per-length
    dictionary just like ``do_m1`` does for L=32)."""
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.recon_precompute import load_layer_conditioned_ar
    from oracle_lens.pipeline.scorer import bare_token_ids

    shard = args.shard if args.shard >= 0 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    files = sample_files(args, args.variant_dir, args.split)
    if shard >= len(files):
        print(f"[r2s-emlen] shard {shard} >= {len(files)} files — nothing to do")
        return
    src = files[shard]
    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else [4, 8, 16]
    payload = json.loads(src.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = int(tokenizer.pad_token_id or 0)
    contents = [candidate_phrases(row["samples"], tokenizer, n=10_000) for row in payload]
    if args.f1_filter:
        contents = [[c for c in cs if not is_degenerate(c)] for cs in contents]
    labels = _desc(args)
    if labels is not None:
        # band rows keep only DESC-labeled sources; the truncations inherit the source's label
        contents = [
            [
                c
                for c in cs
                if _desc_keep_content(labels, int(row["row"]), int(row["layer"]), c, tokenizer)
            ]
            for row, cs in zip(payload, contents, strict=True)
        ]

    recon = None
    d_model = 0
    from safetensors.torch import save_file

    for length in lengths:
        mlen_dir = (
            args.variant_dir
            + (".f1" if args.f1_filter else "")
            + (".w" if labels is not None else "")
        )
        base = tag_dir(args) / "embeds_mlen" / mlen_dir / args.split / f"L{length}"
        base.mkdir(parents=True, exist_ok=True)
        stem = src.stem.replace("samples_", "embed_")
        out_st, out_js = base / f"{stem}.safetensors", base / f"{stem}.json"
        if out_st.exists() and out_js.exists():
            print(f"[r2s-emlen] {out_st} exists — skipping")
            continue
        phrases: list[str] = []
        seen: set[str] = set()
        for cs in contents:
            for c in cs:
                ph = truncate_phrase(c, tokenizer, n=length)
                if ph and ph not in seen:
                    seen.add(ph)
                    phrases.append(ph)
        if recon is None:
            recon, _tok, d_model = load_layer_conditioned_ar(ola_root() / "ar_ckpts" / args.ar_run)
        dirs = torch.zeros(len(LAYERS), len(phrases), d_model, dtype=torch.float16)
        with torch.no_grad():
            for lo in range(0, len(phrases), args.batch):
                chunk = phrases[lo : lo + args.batch]
                idr = [bare_token_ids(tokenizer, t) or [pad_id] for t in chunk]
                w = max(len(r) for r in idr)
                ids = torch.full((len(idr), w), pad_id, dtype=torch.long)
                mask = torch.zeros(len(idr), w, dtype=torch.long)
                for j, r in enumerate(idr):
                    ids[j, : len(r)] = torch.tensor(r)
                    mask[j, : len(r)] = 1
                out = recon(ids.to("cuda"), mask.to("cuda"))
                dirs[:, lo : lo + len(idr)] = out.transpose(0, 1).to(torch.float16).cpu()
        tmp = out_st.with_suffix(".tmp")
        save_file({"dirs": dirs}, str(tmp), metadata={"length": str(length)})
        tmp.replace(out_st)
        _atomic_json(out_js, {"phrases": phrases})
        print(f"[r2s-emlen] L{length} {src.name}: {len(phrases)} uniques -> {out_st}", flush=True)


# ------------------------------------------------------------------- shared embed reading


class EmbedShards:
    """Lazy reader over one variant/split's embed shards (dirs stay on disk until sliced).

    ``drop_degenerate=True`` applies the F1 heuristics (r2sf round) at READ time: flagged
    phrases are masked out of every row's ``cand`` list and reported bad by :meth:`is_bad`
    (dict builders must skip them). The embed shards themselves are untouched/reused — index
    alignment with the fp16 slabs is preserved.
    """

    def __init__(
        self, d: Path, *, drop_degenerate: bool = False, desc_labels: "DescLabels | None" = None
    ) -> None:
        self.sts = sorted(d.glob("embed_*.safetensors"))
        self.sides = [json.loads(p.with_suffix(".json").read_text()) for p in self.sts]
        if not self.sts:
            raise SystemExit(f"no embed shards under {d}")
        self.drop_degenerate = drop_degenerate
        if drop_degenerate:
            from oracle_lens.pipeline.r2_filter import is_degenerate

            n_bad = n_all = 0
            for side in self.sides:
                bad = [is_degenerate(p) for p in side["phrases"]]
                side["_bad"] = bad
                n_bad += sum(bad)
                n_all += len(bad)
                for r in side["rows"]:
                    r["cand"] = [(-1 if (c >= 0 and bad[c]) else c) for c in r["cand"]]
            print(f"[r2s] F1 filter: {n_bad}/{n_all} pool phrases masked ({d})")
        self.desc_labels = desc_labels
        if desc_labels is not None:
            # r2sf-dp band filter: mask CONT/JUNK-labeled candidate slots for in-band rows
            # (labels are per phrase TEXT; unlabeled rows/phrases stay, fail-open).
            n_masked = 0
            for side in self.sides:
                phr = side["phrases"]
                for r in side["rows"]:
                    if not desc_labels.in_band(int(r["layer"])):
                        continue
                    cand = []
                    for c in r["cand"]:
                        if c >= 0 and not desc_labels.keep(int(r["row"]), int(r["layer"]), phr[c]):
                            cand.append(-1)
                            n_masked += 1
                        else:
                            cand.append(c)
                    r["cand"] = cand
            print(f"[r2s] desc-band filter: {n_masked} candidate slots masked ({d})")

    def is_bad(self, si: int, ui: int) -> bool:
        """True iff phrase ``ui`` of shard ``si`` is F1-masked (always False unfiltered)."""
        bad = self.sides[si].get("_bad")
        return bool(bad and bad[ui])

    def is_desc_bad(self, si: int, ui: int) -> bool:
        """True iff phrase ``ui`` of shard ``si`` is labeled CONT/JUNK in every band row it
        was judged in (dict-method band semantics: drop KNOWN non-DESC atoms, keep unlabeled).
        Always False without a desc filter."""
        if self.desc_labels is None:
            return False
        side = self.sides[si]
        db = side.get("_descbad")
        if db is None:
            db = [self.desc_labels.labeled_bad(p) for p in side["phrases"]]
            side["_descbad"] = db
        return bool(db[ui])

    def layer_slab(self, si: int, lp: int) -> torch.Tensor:
        """One shard's [U, d] fp16 raw AR images at layer position ``lp``."""
        from safetensors import safe_open

        with safe_open(str(self.sts[si]), framework="pt") as f:
            slab: torch.Tensor = f.get_slice("dirs")[lp]
        return slab

    def rows(self) -> list[tuple[int, dict[str, Any]]]:
        """All (shard_index, row_meta) pairs across shards."""
        return [(si, r) for si, side in enumerate(self.sides) for r in side["rows"]]


def _pairs(args: argparse.Namespace, split: str) -> Any:
    paths = sorted((ola_root() / args.pairs_dir / split).glob("*.safetensors"))
    pairs, meta = load_multilayer_shards_lazy(paths)
    if meta.targets_kind != "resid":
        raise SystemExit(f"targets_kind={meta.targets_kind!r} != resid")
    return pairs


# ----------------------------------------------------------------------------- mode: m2


def do_m2(args: argparse.Namespace) -> None:
    out = tag_dir(args) / "selections" / f"m2{args.out_tag}" / f"{args.split}.jsonl"
    if out.exists():
        print(f"[r2s-m2] {out} exists — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = EmbedShards(
        embed_dir(args, args.variant_dir, args.split),
        drop_degenerate=args.f1_filter,
        desc_labels=_desc(args),
    )
    by_layer: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for si, r in shards.rows():
        by_layer.setdefault(LAYERS.index(int(r["layer"])), []).append((si, r))

    results: list[dict[str, Any]] = []
    for lp in sorted(by_layer):
        space = load_space(LAYERS[lp], ridge_c=args.ridge_c, device="cuda")
        slabs = {si: shards.layer_slab(si, lp) for si in {si for si, _ in by_layer[lp]}}
        items = by_layer[lp]
        for lo in range(0, len(items), args.chunk):
            part = items[lo : lo + args.chunk]
            c_max = max(len(r["cand"]) for _, r in part)
            dirs = torch.zeros(len(part), c_max, slabs[part[0][0]].shape[-1])
            valid = torch.zeros(len(part), c_max, dtype=torch.bool)
            uidx_of: list[list[int]] = []
            for j, (si, r) in enumerate(part):
                uniq_here = sorted({c for c in r["cand"] if c >= 0})
                uidx_of.append(uniq_here)
                if uniq_here:
                    dirs[j, : len(uniq_here)] = slabs[si][torch.tensor(uniq_here)].float()
                    valid[j, : len(uniq_here)] = True
            h = torch.stack(
                [torch.as_tensor(_PAIRS.targets[r["row"]])[lp] for _, r in part]
            ).float()
            x_w = space.whiten(h.to("cuda"))
            dirs_w = unit_rows(space.whiten(dirs.to("cuda").reshape(-1, dirs.shape[-1])))
            dirs_w = dirs_w.reshape(len(part), c_max, -1)
            sel, coeffs, fve = omp_select(x_w, dirs_w, valid.to("cuda"), k=K)
            for j, (si, r) in enumerate(part):
                picked = [uidx_of[j][int(s)] for s in sel[j] if s >= 0]
                results.append(
                    {
                        "row": r["row"],
                        "layer": r["layer"],
                        "phrases": [shards.sides[si]["phrases"][u] for u in picked],
                        "coeffs": [round(float(c), 4) for c in coeffs[j][: len(picked)]],
                        "fve": round(float(fve[j]), 4),
                        "n_uniq": len(uidx_of[j]),
                    }
                )
        print(f"[r2s-m2] L{LAYERS[lp]}: {len(items)} rows done", flush=True)
    results.sort(key=lambda r: int(r["row"]))
    _write_jsonl(out, results)
    print(f"[r2s-m2] wrote {out} ({len(results)} rows)")


# ----------------------------------------------------------------------------- mode: m1


def do_m1(args: argparse.Namespace) -> None:
    # per-layer sharding (m5 pattern): --only-layer-idx N writes a .part.L<N> file so the big
    # prefix-expanded dicts can run as a 17-task array; merge via olens_r2s_m5_merge --method m1
    part_tag = f".part.L{args.only_layer_idx}" if args.only_layer_idx >= 0 else ""
    out = tag_dir(args) / "selections" / f"m1{args.out_tag}" / f"{args.split}{part_tag}.jsonl"
    if out.exists():
        print(f"[r2s-m1] {out} exists — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    # dictionary = TRAIN pool only, both splits select against it (user decision 2026-08-01)
    labels = _desc(args)
    dict_shards = EmbedShards(
        embed_dir(args, args.variant_dir, "train"),
        drop_degenerate=args.f1_filter,
        desc_labels=labels,
    )

    def build_picks(desc_only: bool) -> tuple[dict[int, list[int]], list[str]]:
        first: dict[str, tuple[int, int]] = {}
        for si, side in enumerate(dict_shards.sides):
            for ui, ph in enumerate(side["phrases"]):
                if dict_shards.is_bad(si, ui):
                    continue
                if desc_only and dict_shards.is_desc_bad(si, ui):
                    continue
                first.setdefault(ph, (si, ui))
        picks: dict[int, list[int]] = {}
        for _ph, (si, ui) in first.items():
            picks.setdefault(si, []).append(ui)
        # phrases in the SAME shard-by-shard order the dict rows are built -> no GPU reorder
        # (materializing a reordered copy of a multi-GB dict was a second OOM source).
        phrases = [dict_shards.sides[si]["phrases"][ui] for si in sorted(picks) for ui in picks[si]]
        return picks, phrases

    # r2sf-dp band semantics for the DICT method: in-band layers select against the dictionary
    # minus KNOWN CONT/JUNK atoms (band-judged, never DESC); out-of-band layers (and runs
    # without --desc-filter) use the full dictionary.
    dict_variants = {False: build_picks(False)}
    if labels is not None:
        dict_variants[True] = build_picks(True)
    print(
        f"[r2s-m1] dictionary: {len(dict_variants[False][1])} unique train-pool phrases "
        f"from {len(dict_shards.sts)} shards"
        + (f" (band dict: {len(dict_variants[True][1])})" if labels is not None else "")
    )

    # query rows come from the TARGET split's pool metadata (row/layer assignments)
    q_shards = (
        dict_shards
        if args.split == "train"
        else EmbedShards(embed_dir(args, args.variant_dir, "eval"), drop_degenerate=args.f1_filter)
    )
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for _si, r in q_shards.rows():
        by_layer.setdefault(LAYERS.index(int(r["layer"])), []).append(r)
    if args.only_layer_idx >= 0:  # array-shard mode: this task owns exactly LAYERS[idx]
        by_layer = {args.only_layer_idx: by_layer.get(args.only_layer_idx, [])}

    results: list[dict[str, Any]] = []
    for lp in sorted(by_layer):
        band = labels is not None and labels.in_band(LAYERS[lp])
        picks_per_shard, dict_phrases = dict_variants[band]
        space = load_space(LAYERS[lp], ridge_c=args.ridge_c, device="cuda")
        # Stream each shard's slab straight into a preallocated bf16 dict (whiten -> unit ->
        # cast per shard). A full-fp32 dict (6M x d ~ 120GB) OOMs both the 140GB card and, via a
        # 2x torch.cat peak, host RAM; this keeps only per-shard fp32 slabs + the final bf16 dict.
        # --dict-device cpu holds the dict in host RAM (r2sp prefix dicts are ~215GB > VRAM);
        # nnomp_batch streams column chunks to the GPU on device mismatch.
        n_atoms = sum(len(picks_per_shard[si]) for si in picks_per_shard)
        dictionary: torch.Tensor | None = None
        off = 0
        for si in sorted(picks_per_shard):
            slab = dict_shards.layer_slab(si, lp)[torch.tensor(picks_per_shard[si])].float()
            g = unit_rows(space.whiten(slab.to("cuda"))).to(torch.bfloat16)
            if dictionary is None:
                dictionary = torch.empty(
                    n_atoms, g.shape[1], dtype=torch.bfloat16, device=args.dict_device
                )
            dictionary[off : off + g.shape[0]] = g
            off += g.shape[0]
            del slab, g
        torch.cuda.empty_cache()
        if dictionary is None:
            raise SystemExit(f"[r2s-m1] empty dictionary at L{LAYERS[lp]}")

        items = by_layer[lp]
        for lo in range(0, len(items), args.chunk):
            part = items[lo : lo + args.chunk]
            h = torch.stack([torch.as_tensor(_PAIRS.targets[r["row"]])[lp] for r in part]).float()
            x_w = space.whiten(h.to("cuda"))
            # bf16 scores (fp32 accum on H200) ~4x the fp32 walltime at a ~1.4M-atom dict;
            # exact coefficients are recomputed in fp32 by the pilot's joint-refit metric
            res = nnomp_batch(
                x_w.to(torch.bfloat16), dictionary, max_atoms=K, chunk=args.score_chunk
            )
            for j, r in enumerate(part):
                picked = [int(a) for a in res.atoms[j] if a >= 0]
                results.append(
                    {
                        "row": r["row"],
                        "layer": r["layer"],
                        "phrases": [dict_phrases[a] for a in picked],
                        "atom_idx": picked,
                        "coeffs": [round(float(c), 4) for c in res.coeffs[j][: len(picked)]],
                        "fve": round(float(res.fve[j]), 4),
                    }
                )
            print(
                f"[r2s-m1] L{LAYERS[lp]}: {min(lo + args.chunk, len(items))}/{len(items)}",
                flush=True,
            )
        del dictionary
        torch.cuda.empty_cache()
    results.sort(key=lambda r: int(r["row"]))
    _write_jsonl(out, results)
    print(f"[r2s-m1] wrote {out} ({len(results)} rows)")


# --------------------------------------------------------------------- mode: deflate-step


def do_deflate_step(args: argparse.Namespace) -> None:
    step = args.step
    vdir = args.variant_dir if step == 0 else f"{args.defl_variant}.step{step}"
    m4_dir = tag_dir(args) / f"m4{args.out_tag}" / args.split
    m4_dir.mkdir(parents=True, exist_ok=True)
    state_path = m4_dir / f"state_step{step}.json"
    if state_path.exists():
        print(f"[r2s-defl] {state_path} exists — skipping")
        return

    shards = EmbedShards(embed_dir(args, vdir, args.split), drop_degenerate=args.f1_filter)
    all_rows = shards.rows()
    n_rows = len(all_rows)
    draw = layer_draw(n_rows, len(LAYERS), seed=args.seed, split=args.split)

    prev_state: dict[str, Any] | None = None
    if step > 0:
        prev_state = json.loads((m4_dir / f"state_step{step - 1}.json").read_text())
    accepted: list[list[str]] = (
        prev_state["accepted"] if prev_state else [[] for _ in range(n_rows)]
    )
    coefs: list[list[float]] = prev_state["coefs"] if prev_state else [[] for _ in range(n_rows)]
    pos_flags: list[list[bool]] = (
        prev_state["positive"] if prev_state else [[] for _ in range(n_rows)]
    )

    # current vectors: raw pairs at step 0, previous step's deflated shard after
    if step == 0:
        vec_of = _PAIRS.targets

        def get_vec(row: int, lp: int) -> torch.Tensor:
            return torch.as_tensor(vec_of[row])[lp]

        wnorm0: list[float] = [0.0] * n_rows
    else:
        from safetensors.torch import load_file

        vecs = load_file(str(m4_dir / f"step{step}" / "vecs_0000.safetensors"))["targets"]

        def get_vec(row: int, lp: int) -> torch.Tensor:
            return vecs[row][lp]

        wnorm0 = prev_state["wnorm0"] if prev_state else [0.0] * n_rows

    by_layer: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for si, r in all_rows:
        lp = LAYERS.index(int(r["layer"]))
        if int(draw[int(r["row"])]) != lp:
            raise SystemExit(
                f"layer draw mismatch at row {r['row']}: shard says L{r['layer']}, "
                f"draw says L{LAYERS[int(draw[int(r['row'])])]} — seed/split misaligned"
            )
        by_layer.setdefault(lp, []).append((si, r))

    next_vecs = (
        torch.zeros(n_rows, len(LAYERS), shards.layer_slab(0, 0).shape[-1], dtype=torch.float16)
        if step < K - 1
        else None
    )
    n_fallback = n_empty = 0
    for lp in sorted(by_layer):
        space = load_space(LAYERS[lp], ridge_c=args.ridge_c, device="cuda")
        slabs = {si: shards.layer_slab(si, lp) for si in {si for si, _ in by_layer[lp]}}
        items = by_layer[lp]
        for lo in range(0, len(items), args.chunk):
            part = items[lo : lo + args.chunk]
            c_max = max(len(r["cand"]) for _, r in part)
            d_model = slabs[part[0][0]].shape[-1]
            dirs_raw = torch.zeros(len(part), c_max, d_model)
            valid = torch.zeros(len(part), c_max, dtype=torch.bool)
            cand_uidx: list[list[int]] = []
            for j, (si, r) in enumerate(part):
                row = int(r["row"])
                taken = set(accepted[row])
                uniq_here = sorted(
                    {c for c in r["cand"] if c >= 0 and shards.sides[si]["phrases"][c] not in taken}
                )
                cand_uidx.append(uniq_here)
                if uniq_here:
                    dirs_raw[j, : len(uniq_here)] = slabs[si][torch.tensor(uniq_here)].float()
                    valid[j, : len(uniq_here)] = True
            h = torch.stack([get_vec(int(r["row"]), lp) for _, r in part]).float().to("cuda")
            h_w = space.whiten(h)
            if step == 0:
                for j, (_si, r) in enumerate(part):
                    wnorm0[int(r["row"])] = round(float(h_w[j].norm()), 4)
            dirs_dev = dirs_raw.to("cuda")
            dirs_w = unit_rows(space.whiten(dirs_dev.reshape(-1, d_model)))
            dirs_w = dirs_w.reshape(len(part), c_max, -1)
            idx, coef, positive = best_step_pick(h_w, dirs_w, valid.to("cuda"))

            chosen_raw = torch.zeros(len(part), d_model, device="cuda")
            has = idx >= 0
            if bool(has.any()):
                rows_sel = torch.nonzero(has).squeeze(-1)
                chosen_raw[rows_sel] = dirs_dev[rows_sel, idx[rows_sel].clamp_min(0)]
            h_new = h.clone()
            if bool(has.any()):
                abl, _cf = project_out_rows(space, h[rows_sel], chosen_raw[rows_sel])
                h_new[rows_sel] = abl
            for j, (si, r) in enumerate(part):
                row = int(r["row"])
                if int(idx[j]) < 0:
                    n_empty += 1
                    accepted[row].append("")
                    coefs[row].append(0.0)
                    pos_flags[row].append(False)
                else:
                    if not bool(positive[j]):
                        n_fallback += 1
                    accepted[row].append(shards.sides[si]["phrases"][cand_uidx[j][int(idx[j])]])
                    coefs[row].append(round(float(coef[j]), 4))
                    pos_flags[row].append(bool(positive[j]))
                if next_vecs is not None:
                    next_vecs[row, lp] = h_new[j].to(torch.float16).cpu()
        print(f"[r2s-defl] step {step} L{LAYERS[lp]}: {len(items)} rows", flush=True)

    if next_vecs is not None:
        from safetensors.torch import save_file

        step_dir = m4_dir / f"step{step + 1}"
        step_dir.mkdir(parents=True, exist_ok=True)
        tmp = step_dir / "vecs_0000.tmp"
        save_file(
            {"targets": next_vecs},
            str(tmp),
            metadata={
                "targets_kind": "resid",
                "n_pairs": str(n_rows),
                "layers": ",".join(str(x) for x in LAYERS),
            },
        )
        tmp.replace(step_dir / "vecs_0000.safetensors")
        print(f"[r2s-defl] wrote {step_dir}/vecs_0000.safetensors")
    _atomic_json(
        state_path,
        {
            "accepted": accepted,
            "coefs": coefs,
            "positive": pos_flags,
            "wnorm0": wnorm0,
            "n_fallback": n_fallback,
            "n_empty": n_empty,
        },
    )
    print(f"[r2s-defl] step {step}: fallback={n_fallback} empty={n_empty} -> {state_path}")

    if step == K - 1:
        sel_dir = tag_dir(args) / "selections" / f"m4{args.out_tag}"
        sel_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for _si, r in all_rows:
            row = int(r["row"])
            w0 = max(wnorm0[row], 1e-9)
            results.append(
                {
                    "row": row,
                    "layer": r["layer"],
                    "phrases": [p for p in accepted[row] if p],
                    "coefs": coefs[row],
                    "positive": pos_flags[row],
                    "wnorm0": wnorm0[row],
                    "fve_chain": round(sum(c * c for c in coefs[row]) / (w0 * w0), 4),
                }
            )
        results.sort(key=lambda r: int(r["row"]))
        _write_jsonl(sel_dir / f"{args.split}.jsonl", results)
        print(f"[r2s-defl] wrote {sel_dir / (args.split + '.jsonl')} ({len(results)} rows)")


# ----------------------------------------------------------------------------- mode: m5


def do_m5(args: argparse.Namespace) -> None:
    """Multi-length staged NNOMP (user spec 2026-08-01): derive per-length candidate pools by
    TRUNCATING the pool samples to each L in ``M5_LENGTHS`` (exact for an AR teacher — the first
    L tokens of a temp-1.0 sample ARE a max_tokens=L draw), embed each unique truncation through
    the AR at the row's layer, then greedy OMP picking one atom per length in coarse->fine order.
    Reads raw samples (not the n=32 embed shards, whose texts are all length 32)."""
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.recon_precompute import load_layer_conditioned_ar
    from oracle_lens.pipeline.scorer import bare_token_ids

    lengths = tuple(int(x) for x in args.lengths.split(",")) if args.lengths else M5_LENGTHS
    # per-layer sharding: --only-layer-idx N processes just LAYERS[N] into a part file, so the
    # (slow, 64-candidate) pool64 selection can run as a job array (one layer per task) and be
    # merged afterward. Default -1 = all layers into <split>.jsonl (original behavior).
    part_tag = f".part.L{args.only_layer_idx}" if args.only_layer_idx >= 0 else ""
    out = tag_dir(args) / "selections" / f"m5{args.out_tag}" / f"{args.split}{part_tag}.jsonl"
    if out.exists():
        print(f"[r2s-m5] {out} exists — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = int(tokenizer.pad_token_id or 0)

    files = sample_files(args, args.variant_dir, args.split)
    rows: list[dict[str, Any]] = []
    for p in files:
        rows.extend(json.loads(p.read_text()))
    # per row: unique truncation per length + which length-bucket(s) each unique phrase is in
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_layer.setdefault(LAYERS.index(int(r["layer"])), []).append(r)
    if args.only_layer_idx >= 0:  # process only this layer's rows (array-shard mode)
        by_layer = {args.only_layer_idx: by_layer.get(args.only_layer_idx, [])}

    recon, _t, d_model = load_layer_conditioned_ar(ola_root() / "ar_ckpts" / args.ar_run)

    def ar_imgs(texts: list[str], lp: int) -> torch.Tensor:
        preds = torch.zeros(len(texts), d_model)
        with torch.no_grad():
            for lo in range(0, len(texts), args.batch):
                chunk = texts[lo : lo + args.batch]
                idr = [bare_token_ids(tokenizer, t) or [pad_id] for t in chunk]
                w = max(len(x) for x in idr)
                ids = torch.full((len(idr), w), pad_id, dtype=torch.long)
                mask = torch.zeros(len(idr), w, dtype=torch.long)
                for j, x in enumerate(idr):
                    ids[j, : len(x)] = torch.tensor(x)
                    mask[j, : len(x)] = 1
                preds[lo : lo + len(idr)] = recon(ids.cuda(), mask.cuda())[:, lp].float().cpu()
        return preds

    results: list[dict[str, Any]] = []
    for lp in sorted(by_layer):
        space = load_space(LAYERS[lp], ridge_c=args.ridge_c, device="cuda")
        items = by_layer[lp]
        for lo in range(0, len(items), args.chunk):
            part = items[lo : lo + args.chunk]
            # build per-row candidate universe (union over lengths) + per-length membership
            per_uni: list[list[str]] = []
            per_bucket: list[list[set[int]]] = []  # [row][length_idx] = {cand indices}
            all_texts: list[str] = []
            for r in part:
                uni: list[str] = []
                where: dict[str, int] = {}
                buckets: list[set[int]] = [set() for _ in lengths]
                # extract each sample's full content once, then truncate to every length
                contents = candidate_phrases(r["samples"], tokenizer, n=10_000)
                if args.f1_filter:
                    contents = [c for c in contents if not is_degenerate(c)]
                if _desc(args) is not None:
                    contents = [
                        c
                        for c in contents
                        if _desc_keep_content(
                            _desc(args), int(r["row"]), int(r["layer"]), c, tokenizer
                        )
                    ]
                for li, length in enumerate(lengths):
                    seen: set[str] = set()
                    for content in contents:
                        ph = truncate_phrase(content, tokenizer, n=length)
                        if not ph or ph in seen:
                            continue
                        seen.add(ph)
                        if ph not in where:
                            where[ph] = len(uni)
                            uni.append(ph)
                        buckets[li].add(where[ph])
                per_uni.append(uni)
                per_bucket.append(buckets)
                all_texts.extend(uni)
            # embed the union once per chunk
            imgs = ar_imgs(all_texts, lp).cuda() if all_texts else torch.zeros(0, d_model).cuda()
            imgs_w = space.whiten(imgs) if all_texts else imgs
            c_max = max((len(u) for u in per_uni), default=1)
            c_max = max(c_max, 1)
            dirs_w = torch.zeros(len(part), c_max, d_model, device="cuda")
            stage_valid = [torch.zeros(len(part), c_max, dtype=torch.bool) for _ in lengths]
            off = 0
            for ri, uni in enumerate(per_uni):
                if uni:
                    dirs_w[ri, : len(uni)] = unit_rows(imgs_w[off : off + len(uni)])
                    for li in range(len(lengths)):
                        for ci in per_bucket[ri][li]:
                            stage_valid[li][ri, ci] = True
                off += len(uni)
            h = (
                torch.stack([torch.as_tensor(_PAIRS.targets[r["row"]])[lp] for r in part])
                .float()
                .to("cuda")
            )
            x_w = space.whiten(h)
            sel, coeffs, fve = omp_select_staged(x_w, dirs_w, [m.cuda() for m in stage_valid])
            for ri, r in enumerate(part):
                picks = [
                    (int(li), int(sel[ri, li])) for li in range(len(lengths)) if sel[ri, li] >= 0
                ]
                results.append(
                    {
                        "row": int(r["row"]),
                        "layer": int(r["layer"]),
                        "phrases": [per_uni[ri][ci] for _li, ci in picks],
                        "lengths": [lengths[li] for li, _ci in picks],
                        "coeffs": [round(float(coeffs[ri, li]), 4) for li, _ci in picks],
                        "fve": round(float(fve[ri]), 4),
                    }
                )
        print(f"[r2s-m5] L{LAYERS[lp]}: {len(items)} rows", flush=True)
    results.sort(key=lambda r: int(r["row"]))
    _write_jsonl(out, results)
    print(f"[r2s-m5] wrote {out} ({len(results)} rows)")


# ----------------------------------------------------------------------------- mode: m6


def _whole_pool_dict_layer(
    shards: "EmbedShards",
    lp: int,
    space: WhitenedSpace,
    dtype: torch.dtype,
    desc_only: bool = False,
) -> tuple[list[str], torch.Tensor]:
    """Dedup a pool's phrases across shards (first appearance) and return (phrases,
    unit-whitened AR directions at layer-pos lp) aligned by index — the do_m1 dict build,
    factored so m6 can call it per length. ``desc_only=True`` additionally drops atoms the
    desc-band labels mark as KNOWN CONT/JUNK (band-pass dict semantics)."""
    first: dict[str, tuple[int, int]] = {}
    for si, side in enumerate(shards.sides):
        for ui, ph in enumerate(side["phrases"]):
            if shards.is_bad(si, ui):
                continue
            if desc_only and shards.is_desc_bad(si, ui):
                continue
            first.setdefault(ph, (si, ui))
    picks: dict[int, list[int]] = {}
    for _ph, (si, ui) in first.items():
        picks.setdefault(si, []).append(ui)
    # phrases in shard-by-shard build order -> the dict rows already match, no GPU reorder needed
    # (a reordered copy of a multi-GB dict was a second OOM source alongside the fp32 cat).
    phrases = [shards.sides[si]["phrases"][ui] for si in sorted(picks) for ui in picks[si]]
    # stream each shard straight into the preallocated bf16 dict — never a full fp32 cat (OOM'd
    # both GPU and, via torch.cat's 2x peak, host RAM).
    n_atoms = sum(len(picks[si]) for si in picks)
    dirs_w: torch.Tensor | None = None
    off = 0
    for si in sorted(picks):
        slab = shards.layer_slab(si, lp)[torch.tensor(picks[si])].float()
        g = unit_rows(space.whiten(slab.to("cuda"))).to(dtype)
        if dirs_w is None:
            dirs_w = torch.empty(n_atoms, g.shape[1], dtype=dtype, device="cuda")
        dirs_w[off : off + g.shape[0]] = g
        off += g.shape[0]
        del slab, g
    torch.cuda.empty_cache()
    if dirs_w is None:
        raise SystemExit("[r2s-m6] empty per-length dictionary")
    return phrases, dirs_w


def _best_atom(
    residual: torch.Tensor, dictionary: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row argmax of residual·atom over a big [N,d] dictionary, scored in column chunks."""
    b = residual.shape[0]
    best_s = torch.full((b,), float("-inf"), device=residual.device)
    best_i = torch.full((b,), -1, dtype=torch.long, device=residual.device)
    rq = residual.to(dictionary.dtype)
    for lo in range(0, dictionary.shape[0], chunk):
        s = rq @ dictionary[lo : lo + chunk].T  # [b, cs]
        sc, si = s.float().max(dim=-1)
        upd = sc > best_s
        best_s = torch.where(upd, sc, best_s)
        best_i = torch.where(upd, si + lo, best_i)
    return best_i, best_s


def do_m6(args: argparse.Namespace) -> None:
    """DICT multi-length staged NNOMP (user 2026-08-01): the m5 ladder but selecting each
    length's bullet from the WHOLE TRAIN POOL's length-L truncations (cross-prompt), not the
    row's own samples. One greedy pick per length, coarse→fine, NNLS-refit + deflate between.
    Needs `embed` (L=32) + `embed-mlen` (L=4/8/16) TRAIN shards; both splits query the train dicts.
    """
    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else [4, 8, 16, 32]
    out = tag_dir(args) / "selections" / f"m6{args.out_tag}" / f"{args.split}.jsonl"
    if out.exists():
        print(f"[r2s-m6] {out} exists — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16

    labels = _desc(args)

    def dict_shards(length: int, band: bool = False) -> EmbedShards:
        # band dicts (r2sf-dp): L32 masks KNOWN CONT/JUNK atoms at read time (labels are keyed
        # at the n=32 truncation); L<32 truncation texts can't match the label keys, so their
        # band dicts come from the build-time-filtered ".w" emlen dirs instead.
        if length == 32:
            return EmbedShards(
                embed_dir(args, args.variant_dir, "train"),
                drop_degenerate=args.f1_filter,
                desc_labels=labels if band else None,
            )
        return EmbedShards(
            tag_dir(args)
            / "embeds_mlen"
            / (args.variant_dir + (".f1" if args.f1_filter else "") + (".w" if band else ""))
            / "train"
            / f"L{length}"
        )

    dshards = {length: dict_shards(length) for length in lengths}
    dshards_band = (
        {length: dict_shards(length, band=True) for length in lengths}
        if labels is not None
        else dshards
    )
    q = EmbedShards(embed_dir(args, args.variant_dir, args.split), drop_degenerate=args.f1_filter)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for _si, r in q.rows():
        by_layer.setdefault(LAYERS.index(int(r["layer"])), []).append(r)

    results: list[dict[str, Any]] = []
    for lp in sorted(by_layer):
        band = labels is not None and labels.in_band(LAYERS[lp])
        layer_dicts = dshards_band if band else dshards
        space = load_space(LAYERS[lp], ridge_c=args.ridge_c, device="cuda")
        items = by_layer[lp]
        # whiten ALL of this layer's query rows up front, then keep only per-row staged-NNOMP
        # state resident. The length loop is OUTER: one whole-pool dict is built, scored against
        # every row, and freed before the next length — so at most ONE length-dict (not all four,
        # which was ~230GB for pool64) is on the GPU at a time.
        xs = [
            space.whiten(
                torch.stack(
                    [
                        torch.as_tensor(_PAIRS.targets[r["row"]])[lp]
                        for r in items[lo : lo + args.chunk]
                    ]
                )
                .float()
                .to("cuda")
            )
            for lo in range(0, len(items), args.chunk)
        ]
        x_w = torch.cat(xs)
        del xs
        b, d = x_w.shape
        residual = x_w.clone()
        sel_vecs = torch.zeros(b, len(lengths), d, device="cuda")
        valid = torch.zeros(b, len(lengths), dtype=torch.bool, device="cuda")
        coefs = torch.zeros(b, len(lengths), device="cuda")
        picks: list[list[tuple[int, str]]] = [[] for _ in range(b)]
        for j, length in enumerate(lengths):
            phrases_l, dirs_l = _whole_pool_dict_layer(
                layer_dicts[length], lp, space, dtype, desc_only=band and length == 32
            )
            for lo in range(0, b, args.chunk):
                sl = slice(lo, min(lo + args.chunk, b))
                idx, score = _best_atom(residual[sl], dirs_l, args.score_chunk)
                take = score > 0
                sel_vecs[sl, j] = torch.where(
                    take.unsqueeze(-1), dirs_l[idx.clamp_min(0)].float(), sel_vecs[sl, j]
                )
                valid[sl, j] = take
                w, _fve = nnls_refit(sel_vecs[sl], x_w[sl], valid[sl])
                coefs[sl] = torch.where(valid[sl], w, coefs[sl])
                residual[sl] = x_w[sl] - torch.bmm(w.unsqueeze(1), sel_vecs[sl]).squeeze(1)
                for ri in range(take.shape[0]):
                    if bool(take[ri]):
                        picks[lo + ri].append((length, phrases_l[int(idx[ri])]))
            del phrases_l, dirs_l
            torch.cuda.empty_cache()
            print(f"[r2s-m6] L{LAYERS[lp]} length {length}: {b} rows scored", flush=True)
        for lo in range(0, b, args.chunk):
            sl = slice(lo, min(lo + args.chunk, b))
            _wf, fve = nnls_refit(sel_vecs[sl], x_w[sl], valid[sl])
            for ci, r in enumerate(items[lo : lo + args.chunk]):
                ri = lo + ci
                results.append(
                    {
                        "row": int(r["row"]),
                        "layer": int(r["layer"]),
                        "phrases": [p for _le, p in picks[ri]],
                        "lengths": [le for le, _p in picks[ri]],
                        "coeffs": [round(float(coefs[ri, jj]), 4) for jj in range(len(picks[ri]))],
                        "fve": round(float(fve[ci]), 4),
                    }
                )
        del x_w, residual, sel_vecs, valid, coefs
        torch.cuda.empty_cache()
    results.sort(key=lambda r: int(r["row"]))
    _write_jsonl(out, results)
    print(f"[r2s-m6] wrote {out} ({len(results)} rows)")


_PAIRS: Any = None


def main() -> None:
    global _PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=("embed", "embed-mlen", "m1", "m2", "m5", "m6", "deflate-step"),
    )
    ap.add_argument("--teacher-run", required=True)
    ap.add_argument(
        "--variant-dir",
        default="pool16n32",
        help="samples/<dir> to read (pool16n32, pool64n32, pool16n32.t12, ...)",
    )
    ap.add_argument("--split", default="train", choices=("train", "eval"))
    ap.add_argument("--ar-run", default=DEFAULT_AR)
    ap.add_argument("--shard", type=int, default=-1, help="embed mode: sample-shard index")
    ap.add_argument("--step", type=int, default=0, help="deflate-step mode: chain step 0..3")
    ap.add_argument("--batch", type=int, default=256, help="AR forward batch (embed)")
    ap.add_argument("--chunk", type=int, default=1024, help="selection row chunk")
    ap.add_argument("--score-chunk", type=int, default=65536, help="nnomp dict column chunk")
    ap.add_argument("--ridge-c", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lengths", default="", help="m5 length ladder, e.g. 4,8,16,32")
    ap.add_argument(
        "--pairs-dir",
        default=PAIRS_DIR,
        help="resid pairs dir under $OLA_ROOT (e.g. r2_resid_pairs_le64)",
    )
    ap.add_argument(
        "--prefix-lengths",
        default="",
        help="embed mode: emit EVERY prefix truncation at these token lengths as candidates "
        "(r2sp free prefix selection, e.g. 2,4,8,16,32); writes embeds/<variant>.pfx",
    )
    ap.add_argument(
        "--dict-device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="m1: where the dictionary lives; cpu holds a >VRAM dict in host RAM and "
        "nnomp streams column chunks to the GPU (size the job's --mem to the dict)",
    )
    ap.add_argument(
        "--only-layer-idx",
        type=int,
        default=-1,
        help="m5/m1: process only LAYERS[idx] into a .part.L<idx> file (array sharding)",
    )
    ap.add_argument("--out-tag", default="", help="suffix on selections/<mX><tag> (e.g. _64)")
    ap.add_argument(
        "--f1-filter",
        action="store_true",
        help="r2sf: mask F1-degenerate pool phrases at read time (r2_filter)",
    )
    ap.add_argument(
        "--desc-filter",
        default="",
        help="r2sf-dp: F3-judge labels jsonl; drop CONT/JUNK candidates for rows "
        "in the --desc-band (emlen builds get a .w dir suffix)",
    )
    ap.add_argument(
        "--desc-band", default="20-60", help="inclusive layer band the desc filter applies to"
    )
    ap.add_argument("--defl-variant", default=DEFL_VARIANT, help="deflation step sampling variant")
    args = ap.parse_args()
    if args.mode == "embed":
        do_embed(args)
        return
    if args.mode == "embed-mlen":
        do_embed_mlen(args)
        return
    _PAIRS = _pairs(args, args.split)
    if args.mode == "m2":
        do_m2(args)
    elif args.mode == "m1":
        do_m1(args)
    elif args.mode == "m5":
        do_m5(args)
    elif args.mode == "m6":
        do_m6(args)
    else:
        do_deflate_step(args)


if __name__ == "__main__":
    main()
