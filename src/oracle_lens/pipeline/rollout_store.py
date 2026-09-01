"""Rollout shards for the iolens program: token ids + provenance, safetensors on disk.

One shard = one generation task's conversations. Layout (flat ragged, like the pairs stores):

- ``ids``        int32 ``[total_tokens]`` — prompt then output, per conversation, concatenated
- ``offsets``    int64 ``[n + 1]`` — conversation ``i`` is ``ids[offsets[i]:offsets[i+1]]``
- ``prompt_len`` int32 ``[n]`` — the first ``prompt_len[i]`` tokens of conversation ``i`` are the
  prompt; the rest are the model's own generation (the ONLY region spans may come from)
- ``seed_hash``  int64 ``[n]`` — blake2b-8 of the seed key (dedup/freshness ledger)
- ``split_id``   int8 ``[n]`` — the 4-way split, assigned BEFORE generation (see SPLITS)

Meta (json string under the ``meta`` key): model_id, mode (chat|pt), engine, engine_version,
tokenizer_sha, sampling params, exact token counts, git_commit. Counts are exact sums over this
shard — every stage reconciles against them (gate G9), never ``avg x n``.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

SPLITS: tuple[str, ...] = ("ar_train", "ao_train", "ao_val", "eval")
SPLIT_FRACTIONS: dict[str, float] = {"ar_train": 0.60, "ao_train": 0.30, "ao_val": 0.05,
                                     "eval": 0.05}


def seed_hash64(seed_key: str) -> int:
    """blake2b-8 of the seed key as a signed int64 (safetensors has no uint64)."""
    h = hashlib.blake2b(seed_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=True)


def split_of_key(seed_key: str) -> int:
    """4-way split id from the seed content hash — assigned BEFORE generation, so a requeued or
    duplicated generation task can never move a conversation across the AR/AO boundary."""
    h = hashlib.sha256(seed_key.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big") / 2**64
    acc = 0.0
    for i, name in enumerate(SPLITS):
        acc += SPLIT_FRACTIONS[name]
        if x < acc:
            return i
    return len(SPLITS) - 1


@dataclass
class RolloutShardMeta:
    model_id: str
    mode: str  # "chat" | "pt"
    engine: str
    engine_version: str
    tokenizer_sha: str
    temperature: float
    top_p: float
    max_new: int
    n_convs: int
    n_prompt_tokens: int
    n_output_tokens: int
    git_commit: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class RolloutShards:
    """Read-only view over one or more rollout shards (eager: ids are small, ~4 B/token)."""

    def __init__(
        self,
        ids: Tensor,
        offsets: Tensor,
        prompt_len: Tensor,
        seed_hash: Tensor,
        split_id: Tensor,
        metas: list[RolloutShardMeta],
    ) -> None:
        self.ids = ids
        self.offsets = offsets
        self.prompt_len = prompt_len
        self.seed_hash = seed_hash
        self.split_id = split_id
        self.metas = metas

    def __len__(self) -> int:
        return int(self.prompt_len.shape[0])

    def conv_ids(self, i: int) -> Tensor:
        return self.ids[int(self.offsets[i]) : int(self.offsets[i + 1])]

    def prompt_ids(self, i: int) -> Tensor:
        return self.conv_ids(i)[: int(self.prompt_len[i])]

    def output_ids(self, i: int) -> Tensor:
        return self.conv_ids(i)[int(self.prompt_len[i]) :]

    def output_lengths(self) -> Tensor:
        return (self.offsets[1:] - self.offsets[:-1]) - self.prompt_len.long()

    def counts(self) -> dict[str, int]:
        """Exact token counts (recomputed from the tensors, not the meta — G9 reconciles both)."""
        out_lens = self.output_lengths()
        return {
            "n_convs": len(self),
            "n_prompt_tokens": int(self.prompt_len.sum()),
            "n_output_tokens": int(out_lens.sum()),
        }


def save_rollout_shard(
    path: Path,
    *,
    conv_ids: list[list[int]],
    prompt_lens: list[int],
    seed_hashes: list[int],
    split_ids: list[int],
    meta: RolloutShardMeta,
) -> None:
    """Atomic write (unique tmp → replace). Counts in ``meta`` must match the tensors."""
    if not (len(conv_ids) == len(prompt_lens) == len(seed_hashes) == len(split_ids)):
        raise ValueError("ragged inputs: conv_ids/prompt_lens/seed_hashes/split_ids differ")
    flat: list[int] = []
    offsets = [0]
    for ids in conv_ids:
        flat.extend(ids)
        offsets.append(len(flat))
    n_prompt = int(sum(prompt_lens))
    n_out = len(flat) - n_prompt
    if meta.n_convs != len(conv_ids) or meta.n_prompt_tokens != n_prompt or (
        meta.n_output_tokens != n_out
    ):
        raise ValueError(
            f"meta counts drift: meta says ({meta.n_convs}, {meta.n_prompt_tokens}, "
            f"{meta.n_output_tokens}), tensors say ({len(conv_ids)}, {n_prompt}, {n_out})"
        )
    tensors = {
        "ids": torch.tensor(flat, dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "prompt_len": torch.tensor(prompt_lens, dtype=torch.int32),
        "seed_hash": torch.tensor(seed_hashes, dtype=torch.int64),
        "split_id": torch.tensor(split_ids, dtype=torch.int8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    save_file(tensors, str(tmp), metadata={"meta": meta.to_json()})
    tmp.replace(path)


def load_rollout_shards(paths: list[Path]) -> RolloutShards:
    """Concatenate shards into one view; refuses mixed modes or tokenizers."""
    if not paths:
        raise FileNotFoundError("no rollout shards given")
    ids_parts, off_parts, plen_parts, hash_parts, split_parts = [], [], [], [], []
    metas: list[RolloutShardMeta] = []
    base = 0
    for p in sorted(paths):
        with safe_open(str(p), framework="pt") as f:
            meta = RolloutShardMeta(**json.loads((f.metadata() or {})["meta"]))
            ids = f.get_tensor("ids")
            off = f.get_tensor("offsets")
            plen = f.get_tensor("prompt_len")
            sh = f.get_tensor("seed_hash")
            sp = f.get_tensor("split_id")
        if metas and (meta.mode != metas[0].mode or meta.tokenizer_sha != metas[0].tokenizer_sha):
            raise ValueError(f"shard {p} mixes mode/tokenizer with {paths[0]}")
        metas.append(meta)
        ids_parts.append(ids)
        off_parts.append(off[1:] + base if base else off)
        base += int(off[-1])
        plen_parts.append(plen)
        hash_parts.append(sh)
        split_parts.append(sp)
    offsets = torch.cat([off_parts[0], *off_parts[1:]])
    return RolloutShards(
        torch.cat(ids_parts),
        offsets,
        torch.cat(plen_parts),
        torch.cat(hash_parts),
        torch.cat(split_parts),
        metas,
    )
