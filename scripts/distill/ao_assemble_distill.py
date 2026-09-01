"""Assemble the AO-distill pilot selections into ``distill_v1`` training shards.

Consumes ``ao_gt_omp_readout.py --shrink`` output (``select_<arm><tag>_*.json``) plus the
sampler's ``pconf.json``: per kept item, the 4 shrunk NNOMP picks become ONE bullet-list target

    - <shortest pick>\n- ...\n- <longest pick><EOS>

(shortest-first, prompt_mode ``concepts_raw`` — tag-free, no count in the wording) and the
injected vector is the item's TRUE residual from the ao_val pairs pool (raw; the trainer's
``unit`` transform norm-matches it to alpha at train time, matching how the samples were
generated).

Filter of record (Opus judge 2026-08-11, ~70% kept at ~3% residual defect, plus Agam's
prefix-redundancy rule): n_cand >= 25, exactly 4 picks, every pick >= 2 units (latin words +
CJK chars), max pairwise Jaccard < 0.45 (word keys + CJK bigrams), no U+FFFD, no 1/2/3-gram
repeated >= 3x within a pick, letter-ratio >= 0.30, no pick a string-prefix of another, and —
for parseability of the '- ' format — no pick containing a line that starts with "- ".

    uv run python scripts/distill/ao_assemble_distill.py \
        --out-dir distill_u64/pilot --arm normmatched --tag _shrunk --variant omp4
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from oracle_lens.pipeline.distill_shards import (
    DistillPairs,
    DistillShardMeta,
    save_distill_shard,
)
from oracle_lens.pipeline.multilayer import LAYERS, load_multilayer_shards_lazy

MODEL_ID = "Qwen/Qwen3.6-27B"
SHARD_ROWS = 4096
K = 4
EVAL_EVERY = 20  # every 20th kept item -> eval split (~5%)

_WORD = re.compile(r"[A-Za-z0-9À-ɏЀ-ӿ]+")
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def units(text: str) -> list[str]:
    """Unicode-aware content units: latin/cyrillic words + individual CJK chars."""
    return _WORD.findall(text) + _CJK.findall(text)


def jaccard_keys(text: str) -> set[str]:
    """Similarity keys: lowercased words + CJK bigrams (the judge's convention)."""
    words = {w.lower() for w in _WORD.findall(text)}
    cjk = _CJK.findall(text)
    return words | {a + b for a, b in pairwise(cjk)}


def has_ngram_loop(text: str) -> bool:
    """True if any 1/2/3-gram of units repeats >= 3x (the degenerate-loop debris rule)."""
    u = [w.lower() for w in units(text)]
    for n in (1, 2, 3):
        if len(u) >= 3 * n:
            grams = Counter(tuple(u[i : i + n]) for i in range(len(u) - n + 1))
            if grams and grams.most_common(1)[0][1] >= 3:
                return True
    return False


def letter_ratio(text: str) -> float:
    compact = "".join(text.split())
    if not compact:
        return 0.0
    letters = sum(1 for c in compact if c.isalpha())
    return letters / len(compact)


def reject_reason(picks: list[str], n_cand: int, *, min_picks: int = K) -> str:
    """First filter rule the set violates, or '' if it passes (the filter of record).

    ``min_picks < K`` (Agam 2026-08-23) admits variable-size sets: NNOMP already stops
    early when the AR says a further bullet adds < min-gain FVE, so a 3-pick row is
    "3 bullets were enough", not a failure — the chat rounds' exactly-4 rule stays the
    default."""
    if n_cand < 25:
        return "n_cand<25"
    if not (min_picks <= len(picks) <= K):
        return "n_sel!=4" if min_picks == K else f"n_sel<{min_picks}"
    keys = [jaccard_keys(p) for p in picks]
    for i, p in enumerate(picks):
        if len(units(p)) < 2:
            return "min_units<2"
        if "�" in p:
            return "replchar"
        if has_ngram_loop(p):
            return "ngram_loop"
        if letter_ratio(p) < 0.30:
            return "low_letter"
        if any(ln.lstrip().startswith("- ") for ln in p.split("\n")):
            return "bullet_collision"
        for j in range(i + 1, len(picks)):
            a, b = picks[i].strip(), picks[j].strip()
            if a.startswith(b) or b.startswith(a):
                return "prefix_redundant"
            inter = keys[i] & keys[j]
            union = keys[i] | keys[j]
            if union and len(inter) / len(union) >= 0.45:
                return "jaccard>=0.45"
    return ""


# leading chars to strip: ascii punct + common CJK/typographic marks (by codepoint to stay
# ruff-RUF001-clean)
_LEAD_PUNCT = (
    "'`\".,;:!?)]}>* \t\n-"
    + "".join(map(chr, (0x2019, 0x3002, 0xFF0C, 0x3001, 0xFF1B, 0xFF1A, 0xFF09, 0x3011,
                        0x300D, 0x201D, 0x2026, 0x2013, 0x2014)))
)


def trim_lead_punct(text: str) -> str:
    """Strip leading punctuation/space (Agam 2026-08-11: readouts mirrored training picks that
    START with boundary fragments like ". " / apostrophe-s / ") " — trim at assembly so the
    student never learns punctuation-first bullets). Mid-word fragments starting with letters
    are genuine continuation semantics and stay."""
    return text.lstrip(_LEAD_PUNCT)


def format_bullet_target(picks: list[str]) -> str:
    return "\n".join(f"- {p.strip()}" for p in picks)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ola_root() -> Path:
    import os

    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT unset — export it first (see docs/pipeline.md)")
    return Path(root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="OLA_ROOT-relative pilot dir (pconf.json)")
    ap.add_argument("--arm", default="normmatched")
    ap.add_argument("--tag", default="_shrunk", help="selection-file suffix to consume")
    ap.add_argument("--variant", default="omp4")
    ap.add_argument("--sel-key", default="",
                    help="which stored selection to train on: shrunk_sel | staged_sel | omp_sel "
                         "(default: shrunk_sel when present, else omp_sel)")
    ap.add_argument("--no-trim-lead-punct", action="store_true",
                    help="keep picks' leading punctuation (pre-r4 behavior)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-picks", type=int, default=K,
                    help="accept selections with this many picks or more (up to 4). The chat "
                         "rounds required exactly 4; 3 admits items where NNOMP stopped early "
                         "because the AR scored no 4th bullet worth --min-gain")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    root = ola_root()
    out = root / args.out_dir
    pconf = json.loads((out / "pconf.json").read_text())
    items = pconf["items"]
    rows: list[dict[str, Any]] = []
    for p in sorted(out.glob(f"select_{args.arm}{args.tag}_*.json")):
        rows += json.loads(p.read_text())
    if not rows:
        raise SystemExit(f"no select_{args.arm}{args.tag}_*.json under {out}")

    sel_key = args.sel_key or (
        "shrunk_sel" if any("shrunk_sel" in r for r in rows) else "omp_sel"
    )
    print(f"[asm] selection key: {sel_key}")
    kept: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    reasons: Counter[str] = Counter()
    n_trimmed = 0
    for r in rows:
        sel = r.get(sel_key)
        if not sel:
            reasons["no_selection"] += 1
            continue
        if not args.no_trim_lead_punct:
            trimmed = [dict(s, text=trim_lead_punct(s["text"])) for s in sel]
            n_trimmed += sum(
                1 for a, b in zip(sel, trimmed, strict=True) if a["text"] != b["text"]
            )
            sel = trimmed
        why = reject_reason([s["text"] for s in sel], int(r.get("n_cand", 0)),
                            min_picks=args.min_picks)
        if why:
            reasons[why] += 1
            continue
        kept.append((r, sorted(sel, key=lambda s: (int(s["len"]), len(s["text"])))))
    if n_trimmed:
        print(f"[asm] trimmed leading punctuation on {n_trimmed} picks")

    n = len(rows)
    print(f"[asm] {n} selection rows -> kept {len(kept)} ({100 * len(kept) / n:.1f}%)")
    print(f"[asm] rejects: {dict(reasons.most_common())}")
    len_hist = Counter(int(s["len"]) for _, sel in kept for s in sel)
    print(f"[asm] pick-length histogram (post-shrink): {dict(sorted(len_hist.items()))}")
    layer_hist = Counter(int(r["layer"]) for r, _ in kept)
    print(f"[asm] kept per layer: {dict(sorted(layer_hist.items()))}")
    if "shrunk_fve" in rows[0]:
        import statistics as st

        ok = [r for r in rows if "shrunk_fve" in r]
        print(
            f"[asm] FVE omp(k4) {st.mean(r['omp_fve_traj'][-1] for r in ok):.4f} -> "
            f"shrunk {st.mean(r['shrunk_fve'] for r in ok):.4f} (n={len(ok)})"
        )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eos = int(tok.eos_token_id)
    pairs, _ = load_multilayer_shards_lazy(
        sorted((root / pconf["pairs_dir"]).glob("pairs_train_*.safetensors"))
    )

    splits: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {"train": [], "eval": []}
    for j, entry in enumerate(kept):
        splits["eval" if j % EVAL_EVERY == 0 else "train"].append(entry)

    commit = git_commit()
    for split, entries in splits.items():
        out_dir = out / "shards" / args.variant / split
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = 0
        total_tok = 0
        for lo in range(0, len(entries), SHARD_ROWS):
            part = entries[lo : lo + SHARD_ROWS]
            flat: list[int] = []
            offsets = [0]
            vecs: list[torch.Tensor] = []
            layer_col: list[int] = []
            src_col: list[int] = []
            for r, sel in part:
                text = format_bullet_target([s["text"] for s in sel])
                ids = [*tok(text, add_special_tokens=False)["input_ids"], eos]
                flat.extend(ids)
                offsets.append(len(flat))
                total_tok += len(ids)
                it = items[int(r["item"])]
                assert int(it["layer"]) == int(r["layer"])
                vecs.append(
                    torch.as_tensor(pairs.targets[int(it["pair_row"])])[
                        LAYERS.index(int(r["layer"]))
                    ].float()
                )
                layer_col.append(LAYERS.index(int(r["layer"])))
                src_col.append(int(it["pair_row"]))
            dp = DistillPairs(
                target_ids=torch.tensor(flat, dtype=torch.int32),
                offsets=torch.tensor(offsets, dtype=torch.int64),
                vec=torch.stack(vecs),
                layer_idx=torch.tensor(layer_col, dtype=torch.int64),
                src_row=torch.tensor(src_col, dtype=torch.int64),
            )
            meta = DistillShardMeta(
                model_id=MODEL_ID,
                teacher_run=pconf["teacher"],
                variant=args.variant,
                k=K,
                n=64,
                prompt_mode="concepts_raw",
                pairs_dir=pconf["pairs_dir"],
                split=split,
                seed=args.seed,
                git_commit=commit,
                n_rows=len(dp),
            )
            save_distill_shard(out_dir / f"distill_{shard_idx:04d}.safetensors", dp, meta)
            shard_idx += 1
        print(f"[asm] {split}: {len(entries)} rows, {shard_idx} shard(s), {total_tok:,} tokens")


if __name__ == "__main__":
    main()
