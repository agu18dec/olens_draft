"""Round-2 distillation shards (schema ``distill_v1``): one row = one student training example.

Unlike ``multilayer_v1`` (span text + a 17-layer target stack, layer drawn at example-build
time), a distill row is fully materialized: the FORMATTED list-target token ids (k
``<explanation>`` blocks + EOS, exactly the CE target — the trainer re-wraps nothing), the
single injection vector at the row's ASSIGNED layer (17x smaller than the full stack), and the
layer as a per-row column (``layer_idx`` indexes ``multilayer.LAYERS``). ``src_row`` points back
into the resid-pairs pool for provenance.

Written by ``scripts/ola/assemble_distill.py``; consumed by ``gt_train.build_distill_examples``.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from oracle_lens.pipeline.multilayer import LAYERS

__all__ = [
    "DistillPairs",
    "DistillShardMeta",
    "load_distill_shards",
    "save_distill_shard",
]


@dataclass(frozen=True)
class DistillShardMeta:
    """Provenance for one distill shard (all values stringified for safetensors)."""

    model_id: str
    teacher_run: str  # round-1 run (plus /stepN when not the final checkpoint)
    variant: str  # k4n256, ...
    k: int
    n: int
    prompt_mode: str  # "continuation" (k1n1024) | "list" | "list_upto" (r2sf variable-k)
    pairs_dir: str  # resid-pairs provenance the vectors came from
    split: str
    seed: int
    git_commit: str
    n_rows: int
    schema: str = "distill_v1"

    def to_metadata(self) -> dict[str, str]:
        return {key: str(v) for key, v in asdict(self).items()}

    @classmethod
    def from_metadata(cls, meta: dict[str, str]) -> "DistillShardMeta":
        return cls(
            model_id=meta["model_id"],
            teacher_run=meta["teacher_run"],
            variant=meta["variant"],
            k=int(meta["k"]),
            n=int(meta["n"]),
            prompt_mode=meta["prompt_mode"],
            pairs_dir=meta["pairs_dir"],
            split=meta["split"],
            seed=int(meta["seed"]),
            git_commit=meta["git_commit"],
            n_rows=int(meta["n_rows"]),
            schema=meta.get("schema", "distill_v1"),
        )


@dataclass(frozen=True)
class DistillPairs:
    """Ragged formatted targets + per-row injection vector and layer assignment."""

    target_ids: Int[Tensor, "total"]
    offsets: Int[Tensor, "n_plus_1"]
    vec: Float[Tensor, "n d"]
    layer_idx: Int[Tensor, "n"]  # index into multilayer.LAYERS
    src_row: Int[Tensor, "n"]  # row in the source resid-pairs pool

    def __post_init__(self) -> None:
        if int(self.layer_idx.numel()) and int(self.layer_idx.max()) >= len(LAYERS):
            raise ValueError(f"layer_idx {int(self.layer_idx.max())} outside LAYERS")

    def __len__(self) -> int:
        return int(self.offsets.shape[0]) - 1

    @property
    def lengths(self) -> Int[Tensor, "n"]:
        return self.offsets[1:] - self.offsets[:-1]

    def row_target(self, i: int) -> Int[Tensor, "n_i"]:
        return self.target_ids[int(self.offsets[i]) : int(self.offsets[i + 1])].long()

    def layer_of(self, i: int) -> int:
        return LAYERS[int(self.layer_idx[i])]

    def share_memory_(self) -> "DistillPairs":
        for t in (self.target_ids, self.offsets, self.vec, self.layer_idx, self.src_row):
            t.share_memory_()  # type: ignore[no-untyped-call]
        return self


def save_distill_shard(path: Path, pairs: DistillPairs, meta: DistillShardMeta) -> None:
    if len(pairs) != meta.n_rows:
        raise ValueError(f"meta.n_rows {meta.n_rows} != rows {len(pairs)}")
    tmp = path.with_suffix(".tmp")
    save_file(
        {
            "target_ids": pairs.target_ids.to(torch.int32).cpu(),
            "offsets": pairs.offsets.cpu(),
            "vec": pairs.vec.to(torch.bfloat16).cpu(),
            "layer_idx": pairs.layer_idx.cpu(),
            "src_row": pairs.src_row.cpu(),
        },
        str(tmp),
        metadata=meta.to_metadata(),
    )
    tmp.replace(path)  # atomic against retries


def _read_meta(path: Path) -> DistillShardMeta:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta is None:
        raise ValueError(f"{path}: shard has no metadata")
    return DistillShardMeta.from_metadata(meta)


def load_distill_shards(paths: list[Path]) -> tuple[DistillPairs, DistillShardMeta]:
    """Concatenate shards (row order = path order), shifting offsets; identity keys must match.

    Eager on purpose: a 100k-row variant is ~1 GB of vecs + <0.5 GB of target ids — no lazy
    mmap machinery needed (contrast ``multilayer.load_multilayer_shards_lazy``).
    """
    if not paths:
        raise ValueError("no shard paths given")
    parts = [load_file(str(p), device="cpu") for p in paths]
    metas = [_read_meta(p) for p in paths]
    first = metas[0]
    for m in metas[1:]:
        if (m.model_id, m.teacher_run, m.variant, m.k, m.n, m.prompt_mode, m.schema, m.split) != (
            first.model_id,
            first.teacher_run,
            first.variant,
            first.k,
            first.n,
            first.prompt_mode,
            first.schema,
            first.split,
        ):
            raise ValueError(f"shard meta mismatch: {m} vs {first}")
    offsets_parts: list[Tensor] = []
    shift = 0
    for t in parts:
        off = t["offsets"]
        offsets_parts.append(off[1:] + shift if shift or offsets_parts else off)
        shift += int(off[-1])
    pairs = DistillPairs(
        target_ids=torch.cat([t["target_ids"] for t in parts]),
        offsets=torch.cat(offsets_parts),
        vec=torch.cat([t["vec"] for t in parts]),
        layer_idx=torch.cat([t["layer_idx"] for t in parts]),
        src_row=torch.cat([t["src_row"] for t in parts]),
    )
    total = sum(int(m.n_rows) for m in metas)
    if len(pairs) != total:
        raise ValueError(f"row count {len(pairs)} != sum of shard metas {total}")
    return pairs, first
