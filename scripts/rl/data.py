"""Parquet loader for the self-contained AO RL trainer — OUR schema, no re-render.

Reads the parquets written by scripts/rl/prep_rl_data.py:
  row_id, layer, prompt_ids (PRE-RENDERED token ids, exactly one marker),
  prompt_text (human-readable, unused here), activation_vector (the PRE-TRANSFORMED
  inject vector — AR reconstruction), gold_vector (true residual — reward target).

Streaming/row-group + numpy-fast-path structure follows skip-lens @ 13547ae
train_rl_self_contained.py:load_rl_dataset (their docstring documents the 60GB
`.to_pylist()` disaster this shape avoids); the per-row exactly-one-marker assert
is ported from scripts/rl/ao_data_source.py:83-87 (the prep/tokenizer-drift
tripwire).
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

REQUIRED_COLS = ("row_id", "layer", "prompt_ids", "activation_vector", "gold_vector")


def load_ao_rl_dataset(parquet_path, *, inj_id: int, n_max: int | None = None) -> list[dict]:
    """Rows: {row_id, layer, prompt_ids: list[int], inject: np[d] f32, gold: np[d] f32}."""
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    for required in REQUIRED_COLS:
        assert required in names, (
            f"{parquet_path!r} missing column {required!r} — re-run prep_rl_data.py"
        )
    has_source = "source_text" in names  # crop span text (wandb sample tables)
    cols = list(REQUIRED_COLS) + (["source_text"] if has_source else [])
    rows: list[dict] = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if n_max is None else min(n_max - len(rows), rg.num_rows)
        rg = rg.slice(0, take)
        n = rg.num_rows
        row_ids = rg.column("row_id").to_pylist()
        layers = rg.column("layer").to_pylist()
        prompt_ids = rg.column("prompt_ids").to_pylist()
        sources = rg.column("source_text").to_pylist() if has_source else [""] * n
        # list<float> -> flat arrow values -> [n, d] float32 view (zero-copy;
        # .flatten() respects slice offsets).
        inj_col = rg.column("activation_vector").combine_chunks()
        injects = np.asarray(inj_col.flatten(), dtype=np.float32).reshape(n, -1)
        gold_col = rg.column("gold_vector").combine_chunks()
        golds = np.asarray(gold_col.flatten(), dtype=np.float32).reshape(n, -1)
        for i in range(n):
            pids = prompt_ids[i]
            n_markers = sum(1 for t in pids if t == inj_id)
            assert n_markers == 1, (
                f"{parquet_path!r} row_id={row_ids[i]}: prompt_ids carry {n_markers} "
                f"injection tokens (id {inj_id}), want exactly 1 — prep/tokenizer drift."
            )
            rows.append({
                "row_id": int(row_ids[i]),
                "layer": int(layers[i]),
                "prompt_ids": [int(t) for t in pids],
                "inject": injects[i],
                "gold": golds[i],
                "source_text": sources[i] or "",
            })
    return rows
