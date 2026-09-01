"""On-policy rollout generation — the launcher-agnostic body.

Qwen generates its OWN text; we later capture activations over THAT (``jobs.dump`` reads the
``rollouts_*.json`` written here — it does not generate). Kept out of the Modal app for the same
reason ``jobs.train`` is: the local CLI and the Modal launcher must not drift.

Faithful to ``scripts/ola/onpolicy_modal.py``: same seed corpora and filters, same sampling
(temp 1.0, top_p 1.0, max_tokens 512, seed 0), same row schema and filenames, so rollouts
produced either way are interchangeable.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
_NAME = re.compile(r"NAME_\d+")  # lmsys anonymization placeholder — exclude (user request)


def load_seeds(mode: str, n: int, *, shard: int, n_shards: int, skip: int = 0) -> list[str]:
    """Stream real seeds, taking every ``n_shards``-th row from ``shard`` (disjoint shards).

    ``skip`` drops the first ``skip`` raw stream rows before sharding — for a top-up run that must
    use FRESH prompts disjoint from an earlier run. chat: lmsys first USER turn (skip NAME_\\d+
    anonymized convs). pt: FineWeb-Edu doc prefix (first ~1200 chars).
    """
    from datasets import load_dataset

    seeds: list[str] = []
    if mode == "chat":
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i < skip or (i - skip) % n_shards != shard:
                continue
            conv = row.get("conversation") or []
            if conv and conv[0].get("role") == "user":
                txt = (conv[0].get("content") or "").strip()
                if txt and not _NAME.search(txt) and len(txt) < 4000:
                    seeds.append(txt)
            if len(seeds) >= n:
                break
    else:  # pt
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i < skip or (i - skip) % n_shards != shard:
                continue
            txt = (row.get("text") or "").strip()
            if len(txt) > 400:
                seeds.append(txt[:1200])  # a document prefix to continue from
            if len(seeds) >= n:
                break
    return seeds


def gen_onpolicy(
    *,
    root: Path,
    mode: str = "chat",
    shard: int = 0,
    n_shards: int = 1,
    n_per_shard: int = 1250,
    max_new: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed_offset: int = 0,
    out_tag: str = "",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.92,
) -> dict[str, Any]:
    """Generate one shard of Qwen's own rollouts -> ``root/onpolicy/<mode>/rollouts_<shard>.json``.

    Skip-if-exists: a completed shard is left alone, so a re-run only fills the gaps (the local
    array runner has no scheduler to do this for us).
    """
    out_dir = root / "onpolicy" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"rollouts_{out_tag}{shard:04d}.json" if out_tag else f"rollouts_{shard:04d}.json"
    dst = out_dir / fname
    if dst.exists():
        rows_existing = json.loads(dst.read_text())
        print(f"[gen {mode} shard {shard}] exists ({len(rows_existing)} rollouts) — skipping")
        return {"mode": mode, "shard": shard, "n": len(rows_existing), "skipped": True}

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    seeds = load_seeds(mode, n_per_shard, shard=shard, n_shards=n_shards, skip=seed_offset)
    print(f"[gen {mode} shard {shard}] {len(seeds)} seeds (skip={seed_offset})", flush=True)

    # Qwen3.6 hybrid-Mamba: max_num_seqs must stay under the ~310 mamba-cache limit.
    llm = LLM(
        model=MODEL_ID, dtype="bfloat16", max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=256,
    )
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new, seed=0)

    if mode == "chat":
        outs = llm.chat(
            [[{"role": "user", "content": s}] for s in seeds], sp,
            chat_template_kwargs={"enable_thinking": False}, use_tqdm=True,
        )
    else:
        outs = llm.generate(list(seeds), sp, use_tqdm=True)

    rows, lengths = [], []
    for seed, o in zip(seeds, outs, strict=True):
        gen = o.outputs[0]
        rows.append({
            "seed": seed,
            "prompt_ids": list(o.prompt_token_ids),
            "output_ids": list(gen.token_ids),
        })
        lengths.append(len(gen.token_ids))

    # atomic tmp+rename: a killed shard must never leave a half-written json that the
    # skip-if-exists check above would then treat as complete.
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False))
    tmp.rename(dst)
    tot = sum(lengths)
    print(f"[gen {mode} shard {shard}] saved {len(rows)} rollouts, {tot:,} gen tokens", flush=True)
    return {"mode": mode, "shard": shard, "n": len(rows), "gen_tokens": tot}
