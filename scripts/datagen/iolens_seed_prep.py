"""iolens S2/S3: seed prep — WildChat user turns (chat cell) + FineWeb-Edu prefixes (PT cell).

Writes seed shards under ``$OLA_ROOT/seeds_iolens_{chat,pt}/seeds_{shard:04d}.json`` where each
row is ``{"key": str, "text": str | null, "prefix_ids": [int] | null, "split": int}``:

- chat: every stripped user turn of every WildChat-1M conversation (length-filtered, content
  deduped). ``key = "wildchat:<conv_hash>:<turn_index>"``; the 4-way split is derived from the
  CONVERSATION's first user turn's content hash (``rollout_store.split_of_key``), so every turn
  of a conversation lands on the same split side — assigned BEFORE any generation (ordering
  rule S3<S4: a requeued task can never move a conversation across the AR/AO boundary).
- pt: the first ``--pt-prefix-tokens`` (256) Qwen tokens of FineWeb-Edu ``sample-10BT``
  documents (documents shorter than the prefix are skipped — a full-document "prefix" would
  make the continuation start off-distribution). ``key = "fineweb-edu:<doc_id>"``; split from
  the document id hash.

Stdlib + huggingface_hub + pyarrow + tokenizers only (same streaming pattern as
``lmsys_stream.py`` — no ``datasets`` dependency). Exact counts + split histograms are written
to ``seeds_iolens_{mode}/report.json`` (gate G9 reconciles against them).
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_ID = "Qwen/Qwen3.6-27B"
WILDCHAT_REPO = "allenai/WildChat-1M"
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG_PREFIX = "sample/10BT/"  # the sample-10BT parquet shards live under this prefix


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — export it first (see docs/pipeline.md)")
    return Path(root)


def parquet_paths(repo: str, prefix: str = "", limit: int = 0) -> list[str]:
    """Local paths of the repo's parquet shards (download-once, then cache hits)."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(repo)
    siblings = info.siblings or []
    names = sorted(
        s.rfilename
        for s in siblings
        if s.rfilename.endswith(".parquet") and s.rfilename.startswith(prefix)
    )
    if not names:
        raise RuntimeError(f"no parquet shards under {repo}/{prefix}")
    if limit:
        names = names[:limit]
    return [
        api.hf_hub_download(repo_id=repo, filename=n, repo_type="dataset") for n in names
    ]


def iter_wildchat(batch_rows: int = 512) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield (conversation_hash, conversation turns) rows in deterministic shard order."""
    import pyarrow.parquet as pq

    for path in parquet_paths(WILDCHAT_REPO, prefix="data/"):
        pf = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
        for batch in pf.iter_batches(  # type: ignore[no-untyped-call]
            batch_size=batch_rows, columns=["conversation_hash", "conversation"]
        ):
            hashes = batch.column("conversation_hash").to_pylist()
            convs = batch.column("conversation").to_pylist()
            yield from zip(hashes, convs, strict=True)


def iter_fineweb(batch_rows: int = 512, shard_limit: int = 0) -> Iterator[tuple[str, str]]:
    """Yield (doc id, text) from FineWeb-Edu sample-10BT in deterministic shard order."""
    import pyarrow.parquet as pq

    for path in parquet_paths(FINEWEB_REPO, prefix=FINEWEB_CONFIG_PREFIX, limit=shard_limit):
        pf = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["id", "text"]):  # type: ignore[no-untyped-call]
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            yield from zip(ids, texts, strict=True)


def _content_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def prep_chat(out_dir: Path, *, n_seeds: int, per_shard: int, max_chars: int) -> dict[str, Any]:
    from oracle_lens.pipeline.rollout_store import split_of_key

    seen: set[bytes] = set()
    rows: list[dict[str, Any]] = []
    shard = 0
    split_counts = [0, 0, 0, 0]
    n_convs = 0
    dropped_len = dropped_dup = 0

    def flush() -> None:
        nonlocal shard, rows
        if not rows:
            return
        p = out_dir / f"seeds_{shard:04d}.json"
        tmp = p.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(rows))
        tmp.replace(p)
        shard += 1
        rows = []

    for conv_hash, conv in iter_wildchat():
        turns = [
            (m.get("content") or "").strip()
            for m in (conv or [])
            if m.get("role") == "user" and (m.get("content") or "").strip()
        ]
        if not turns:
            continue
        n_convs += 1
        split = split_of_key(turns[0])  # conversation cohesion: first turn keys the whole conv
        for ti, text in enumerate(turns):
            if len(text) > max_chars or "NAME_" in text:
                dropped_len += 1
                continue
            h = _content_hash(text)
            if h in seen:
                dropped_dup += 1
                continue
            seen.add(h)
            rows.append({"key": f"wildchat:{conv_hash}:{ti}", "text": text, "split": split})
            split_counts[split] += 1
            if len(rows) >= per_shard:
                flush()
            if n_seeds and sum(split_counts) >= n_seeds:
                flush()
                return {
                    "mode": "chat", "n_seeds": sum(split_counts), "n_convs_scanned": n_convs,
                    "splits": split_counts, "dropped_len_or_name": dropped_len,
                    "dropped_dup": dropped_dup, "n_shards": shard,
                }
    flush()
    return {
        "mode": "chat", "n_seeds": sum(split_counts), "n_convs_scanned": n_convs,
        "splits": split_counts, "dropped_len_or_name": dropped_len,
        "dropped_dup": dropped_dup, "n_shards": shard,
    }


def prep_pt(
    out_dir: Path, *, n_seeds: int, per_shard: int, prefix_tokens: int, shard_limit: int
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from oracle_lens.pipeline.rollout_store import split_of_key

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    seen: set[bytes] = set()
    rows: list[dict[str, Any]] = []
    shard = 0
    split_counts = [0, 0, 0, 0]
    dropped_short = dropped_dup = 0

    def flush() -> None:
        nonlocal shard, rows
        if not rows:
            return
        p = out_dir / f"seeds_{shard:04d}.json"
        tmp = p.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(rows))
        tmp.replace(p)
        shard += 1
        rows = []

    for doc_id, text in iter_fineweb(shard_limit=shard_limit):
        h = _content_hash(text[:2000])
        if h in seen:
            dropped_dup += 1
            continue
        seen.add(h)
        # 4000 chars safely bounds 257 tokens (worst case ~1 token/char); tokenizing whole
        # multi-KB documents for a 256-token prefix would dominate the scan
        ids = tok(text[:4000], add_special_tokens=False)["input_ids"][: prefix_tokens + 1]
        if len(ids) <= prefix_tokens:  # doc shorter than the prefix — skip, no partial prefixes
            dropped_short += 1
            continue
        split = split_of_key(doc_id)
        rows.append({"key": f"fineweb-edu:{doc_id}", "prefix_ids": ids[:prefix_tokens],
                     "split": split})
        split_counts[split] += 1
        if len(rows) >= per_shard:
            flush()
        if n_seeds and sum(split_counts) >= n_seeds:
            break
    flush()
    return {
        "mode": "pt", "n_seeds": sum(split_counts), "splits": split_counts,
        "prefix_tokens": prefix_tokens, "dropped_short": dropped_short,
        "dropped_dup": dropped_dup, "n_shards": shard,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["chat", "pt"])
    ap.add_argument("--n-seeds", type=int, default=0, help="0 = the whole source")
    ap.add_argument("--per-shard", type=int, default=50_000)
    ap.add_argument("--max-chars", type=int, default=4000, help="chat: user-turn length cap")
    ap.add_argument("--pt-prefix-tokens", type=int, default=256)
    ap.add_argument("--pt-shard-limit", type=int, default=0, help="cap parquet shards (pilot)")
    ap.add_argument("--out-dir", default="", help="default seeds_iolens_<mode>")
    args = ap.parse_args()
    os.environ.setdefault("AO_HF_ONLINE", "1")  # seed prep is a fetch step by definition

    root = ola_root()
    out_dir = root / (args.out_dir or f"seeds_iolens_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "chat":
        report = prep_chat(
            out_dir, n_seeds=args.n_seeds, per_shard=args.per_shard, max_chars=args.max_chars
        )
    else:
        report = prep_pt(
            out_dir,
            n_seeds=args.n_seeds,
            per_shard=args.per_shard,
            prefix_tokens=args.pt_prefix_tokens,
            shard_limit=args.pt_shard_limit,
        )
    report["split_names"] = ["ar_train", "ao_train", "ao_val", "eval"]
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
