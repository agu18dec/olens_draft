"""Sidecar (nla_meta.yaml) loader for the self-contained AO RL trainer.

Loads the schema-v2 model sidecar written by scripts/rl/prep_rl_data.py
(`build_ao_sidecar`) and re-verifies every token-level fact against the LIVE
tokenizer — the tokenizer-drift tripwire. Assert logic ported from
scripts/rl/ao_data_source.py:54-57 and skip-lens nla/config.py (@13547ae).

The `injection_scale is None` assert is load-bearing: parquet vectors are
PRE-TRANSFORMED at prep, and pass-through injection is the contract that makes
the training-forward splice identical to what the vectors were built for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SidecarConfig:
    injection_char: str
    injection_token_id: int
    left_neighbor_id: int
    right_neighbor_id: int
    d_model: int
    base_checkpoint: str
    actor_template: str
    raw: dict


def load_sidecar(path: str | Path, tokenizer) -> SidecarConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    assert raw.get("kind") == "nla_model" and raw.get("schema_version") == 2, (
        f"{path}: expected nla_model schema v2 sidecar, got "
        f"kind={raw.get('kind')!r} schema_version={raw.get('schema_version')!r}"
    )
    ext = raw["extraction"]
    assert ext.get("injection_scale") is None, (
        f"AO parquets carry PRE-TRANSFORMED vectors; the sidecar must say "
        f"injection_scale: null (pass-through), got {ext.get('injection_scale')!r}."
    )
    tok = raw["tokens"]
    char = tok["injection_char"]
    inj_id = int(tok["injection_token_id"])
    left_id = int(tok["injection_left_neighbor_id"])
    right_id = int(tok["injection_right_neighbor_id"])

    # Live-tokenizer round trip: the char must encode to exactly the pinned id.
    ids = tokenizer.encode(char, add_special_tokens=False)
    assert ids == [inj_id], (
        f"tokenizer drift: {char!r} encodes to {ids}, sidecar pins [{inj_id}]. "
        f"The parquet prompt_ids were rendered under a different tokenizer."
    )
    unk = getattr(tokenizer, "unk_token_id", None)
    assert unk is None or inj_id != unk, f"injection id {inj_id} is UNK"

    # The canonical actor template must render with the marker mid-sequence and
    # the pinned neighbors (same check prep ran across every trained layer).
    template = raw["prompt_templates"]["actor"]
    probe = template.replace("{injection_char}", char)
    probe_ids = tokenizer.encode(probe, add_special_tokens=False)
    slots = [i for i, t in enumerate(probe_ids) if t == inj_id]
    assert len(slots) == 1 and 0 < slots[0] < len(probe_ids) - 1, (
        f"canonical template renders {len(slots)} marker slots (want exactly 1, mid-sequence)"
    )
    s = slots[0]
    assert probe_ids[s - 1] == left_id and probe_ids[s + 1] == right_id, (
        f"neighbor drift: template renders neighbors "
        f"({probe_ids[s - 1]}, {probe_ids[s + 1]}), sidecar pins ({left_id}, {right_id})"
    )

    return SidecarConfig(
        injection_char=char,
        injection_token_id=inj_id,
        left_neighbor_id=left_id,
        right_neighbor_id=right_id,
        d_model=int(raw["d_model"]),
        base_checkpoint=raw["base_checkpoint"],
        actor_template=template,
        raw=raw,
    )
