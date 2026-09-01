"""On-policy multi-layer activation dump — the launcher-agnostic body.

Extracted verbatim from ``scripts/ola/ola_modal.py::dump_multilayer_onpolicy_shard`` (which now
delegates here); the only changes are the explicit ``root``/``rollouts_dir`` parameters and the
removal of the Modal-only lines (``_hf_offline()``, ``data_vol.commit()`` — the callers own those).

Reads ``rollouts_*.json`` (``{prompt_ids, output_ids}``); spans are sampled ONLY inside the
GENERATED region (Qwen's own text), and the target is the 17-layer residual at
``prev_pos = start-1`` captured in ONE forward over ``prompt_ids + output_ids``. Whitening reuses
the assistant-fit ``whitening_L{L}`` artifacts (common yardstick), so no stats are collected here.

Idempotent per shard (skips if the train file exists), so a preempted array task requeues safely.
"""

import json
import random
from pathlib import Path
from typing import Any

from oracle_lens.pipeline.jobs.settings import MIN_RENDERED_TOKENS, MODEL_ID, N_MAX, T_MAX
from oracle_lens.pipeline.paths import git_commit


def dump_onpolicy_shard(
    shard: int,
    n_shards: int,
    *,
    root: Path,
    mode: str = "chat",
    rollouts_dir: Path | None = None,
    t_max: int = T_MAX,
    max_per_conv: int = 16,
    n_max: int = N_MAX,
    n_min: int = 1,
    seed: int = 0,
    out_dir: str = "",
    length_law: str = "loguniform",
) -> dict[str, Any]:
    """One shard of the on-policy multi-layer dump. See module docstring."""
    import torch

    from oracle_lens.core.dump import conversation_layers_resid
    from oracle_lens.core.sampling import split_of
    from oracle_lens.model import ModelBackend
    from oracle_lens.pipeline.multilayer import (
        LAYERS,
        MultiLayerPairs,
        MultiLayerShardMeta,
        save_multilayer_shard,
    )
    from oracle_lens.pipeline.spans import delimiter_mask, sample_long_spans

    layers = list(LAYERS)
    out_dir = out_dir or f"ml_pairs_onpolicy_{mode}"
    pairs_dir = root / out_dir
    pairs_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "train": pairs_dir / f"pairs_train_{shard:04d}.safetensors",
        "eval": pairs_dir / f"pairs_eval_{shard:04d}.safetensors",
    }
    if out["train"].exists():
        return {"shard": shard, "skipped": True}

    src = rollouts_dir if rollouts_dir is not None else root / "onpolicy" / mode
    files = sorted(src.glob("rollouts_*.json"))[shard::n_shards]
    if not files:
        raise FileNotFoundError(f"no rollouts_*.json under {src}")
    dataset_id = f"qwen-onpolicy-{mode}"
    backend = ModelBackend(MODEL_ID, device="cuda", dtype=torch.bfloat16)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    used = 0
    total_spans = 0

    for fpath in files:
        for row_i, roll in enumerate(json.loads(fpath.read_text())):
            rendered = (list(roll["prompt_ids"]) + list(roll["output_ids"]))[:t_max]
            n_prompt = min(len(roll["prompt_ids"]), len(rendered))
            if len(rendered) < MIN_RENDERED_TOKENS or n_prompt >= len(rendered):
                continue  # nothing generated survived truncation -> no on-policy span
            mask = [i >= n_prompt for i in range(len(rendered))]  # spans only in Qwen's own gen
            delim = delimiter_mask(rendered, backend.tokenizer)
            rng = random.Random(seed * 100_003 + shard * 10_007 + row_i)
            spans = sample_long_spans(
                mask,
                max_per_conv=max_per_conv,
                rng=rng,
                n_min=n_min,
                n_max=n_max,
                max_attempts=64,
                delimiter_prev=delim,
                delimiter_upsample=1.0,
                length_law=length_law,
            )
            if not spans:
                continue
            resid = conversation_layers_resid(backend, rendered, layers=layers)  # {L: [pos, d]}
            split = split_of(f"{dataset_id}:{fpath.stem}:{row_i}")
            prev_idx = torch.tensor(
                [s.prev_pos for s in spans], dtype=torch.long, device=backend.device
            )
            tgt = torch.stack([resid[lyr][prev_idx] for lyr in layers], dim=1).cpu()
            rows[split].append(
                {
                    "ids": [rendered[s.start : s.start + s.n_tokens] for s in spans],
                    "targets": tgt,
                    "conv_index": torch.full((len(spans),), row_i, dtype=torch.long),
                    "prev_pos": prev_idx.cpu(),
                    "prev_token_id": torch.tensor(
                        [rendered[s.prev_pos] for s in spans], dtype=torch.long
                    ),
                    "prev_is_assistant": torch.tensor(
                        [mask[s.prev_pos] for s in spans], dtype=torch.bool
                    ),
                }
            )
            total_spans += len(spans)
            used += 1
            if used % 100 == 0:
                print(
                    f"[onpolicy {mode} shard {shard}] {used} rollouts, {total_spans} pairs",
                    flush=True,
                )

    counts: dict[str, int] = {}
    for split_name, split_rows in rows.items():
        if not split_rows:
            continue
        id_rows = [ids for r in split_rows for ids in r["ids"]]
        offsets = torch.zeros(len(id_rows) + 1, dtype=torch.int64)
        for i, ids in enumerate(id_rows):
            offsets[i + 1] = offsets[i] + len(ids)
        pairs = MultiLayerPairs(
            span_ids=torch.tensor([t for ids in id_rows for t in ids], dtype=torch.int32),
            offsets=offsets,
            targets=torch.cat([r["targets"] for r in split_rows]).to(torch.bfloat16),
            layers=LAYERS,
            conv_index=torch.cat([r["conv_index"] for r in split_rows]),
            prev_pos=torch.cat([r["prev_pos"] for r in split_rows]),
            prev_token_id=torch.cat([r["prev_token_id"] for r in split_rows]),
            prev_is_assistant=torch.cat([r["prev_is_assistant"] for r in split_rows]),
        )
        meta = MultiLayerShardMeta(
            model_id=MODEL_ID,
            dataset_id=dataset_id,
            layers=LAYERS,
            t_max=t_max,
            n_max=n_max,
            n_min=n_min,
            length_law=length_law,
            region_start=0,
            region_end=0,
            split=split_name,
            seed=seed,
            git_commit=git_commit(),
            n_pairs=len(pairs),
        )
        save_multilayer_shard(out[split_name], pairs, meta)
        counts[split_name] = len(pairs)
    return {"shard": shard, "rollouts_used": used, "pairs": counts}
