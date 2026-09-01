"""Pair-shard storage and capture helpers for the reconstructor context dump.

A shard is one safetensors file of row-aligned (phrase, target) pairs:

    phrase_ids        [n, max_n]  int64, right-padded with -1
    phrase_len        [n]         int64  (N)
    targets           [n, d]      bf16   (layer-L residual at prev_pos, raw space)
    conv_index        [n]         int64  (doc index in the streamed dataset)
    prev_pos          [n]         int64
    prev_is_assistant [n]         bool

plus string metadata (``PairShardMeta``). Phrase *text* is decode-on-demand from ids.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from safetensors.torch import load_file, save_file
from torch import Tensor

from jlens.hooks import ActivationRecorder
from oracle_lens.core.sampling import PhraseSpan

PAD_ID = -1


def conversation_layer_resid(
    backend: Any, rendered_ids: list[int], *, layer: int
) -> Float[Tensor, "pos d"]:
    """One conversation's layer-``layer`` residual stack (single forward, batch=1, fp32).

    Records only the one block (not all 64 — a 2048-pos full stack is ~2.7 GB of waste per
    conversation). Same lock/no-grad/recorder discipline as ``ModelBackend.run``.
    """
    recorder = ActivationRecorder(backend.layers, at=[layer])
    ids = torch.tensor([rendered_ids], dtype=torch.long, device=backend.device)
    with backend._model_lock, torch.no_grad(), recorder:
        backend.model(input_ids=ids, use_cache=False)
    resid: Tensor = recorder.activations[layer][0]
    return resid.float()


def conversation_layers_resid(
    backend: Any, rendered_ids: list[int], *, layers: list[int]
) -> dict[int, Float[Tensor, "pos d"]]:
    """Multi-layer variant of ``conversation_layer_resid``: ONE forward, all listed layers.

    The recorder accepts a layer list, so an 11-layer capture costs the same forward as one
    layer (memory: n_layers x [pos, d] fp32 — ~0.5 GB for 11 layers at 2048 positions).
    """
    recorder = ActivationRecorder(backend.layers, at=list(layers))
    ids = torch.tensor([rendered_ids], dtype=torch.long, device=backend.device)
    with backend._model_lock, torch.no_grad(), recorder:
        backend.model(input_ids=ids, use_cache=False)
    return {layer: recorder.activations[layer][0].float() for layer in layers}


def conversations_layers_resid_batched(
    backend: Any, batch_ids: list[list[int]], *, layers: list[int]
) -> dict[int, Float[Tensor, "b pos d"]]:
    """Batched multi-conversation capture: ONE right-padded forward, all listed layers.

    Rows are right-padded to the batch max with an attention mask, so positions past each row's
    real length are garbage — callers must only index positions ``< len(batch_ids[i])``.
    Batching is the capture-throughput lever: a lone ~1k-token forward leaves an H200 mostly
    idle, while a ~16k-token batch runs the same prefill compute-bound. Memory is the recorder's
    ``n_layers x [b, pos, d]`` fp32 stack (~0.33 GB per row at 2048 pos x 17 layers) — size the
    batch token budget accordingly.
    """
    b = len(batch_ids)
    maxlen = max(len(r) for r in batch_ids)
    ids = torch.zeros(b, maxlen, dtype=torch.long, device=backend.device)
    mask = torch.zeros(b, maxlen, dtype=torch.long, device=backend.device)
    for i, row in enumerate(batch_ids):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=backend.device)
        mask[i, : len(row)] = 1
    recorder = ActivationRecorder(backend.layers, at=list(layers))
    with backend._model_lock, torch.no_grad(), recorder:
        backend.model(input_ids=ids, attention_mask=mask, use_cache=False)
    return {layer: recorder.activations[layer].float() for layer in layers}


@dataclass(frozen=True)
class PairShardMeta:
    """Provenance carried on every shard (all values stringified for safetensors)."""

    model_id: str
    dataset_id: str
    layer: int
    t_max: int
    split: str
    seed: int
    git_commit: str
    n_pairs: int
    phrase_presentation: str = "bare_ids"

    def to_metadata(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_metadata(cls, meta: dict[str, str]) -> "PairShardMeta":
        return cls(
            model_id=meta["model_id"],
            dataset_id=meta["dataset_id"],
            layer=int(meta["layer"]),
            t_max=int(meta["t_max"]),
            split=meta["split"],
            seed=int(meta["seed"]),
            git_commit=meta["git_commit"],
            n_pairs=int(meta["n_pairs"]),
            phrase_presentation=meta.get("phrase_presentation", "bare_ids"),
        )


@dataclass(frozen=True)
class PairBatch:
    """Stacked pairs from one or more shards (targets kept in their stored dtype)."""

    phrase_ids: Int[Tensor, "n max_n"]
    phrase_len: Int[Tensor, "n"]
    targets: Float[Tensor, "n d"]
    conv_index: Int[Tensor, "n"]
    prev_pos: Int[Tensor, "n"]
    prev_is_assistant: Bool[Tensor, "n"]

    def __len__(self) -> int:
        return int(self.phrase_ids.shape[0])

    def select(self, idx: Int[Tensor, "m"]) -> "PairBatch":
        return PairBatch(
            phrase_ids=self.phrase_ids[idx],
            phrase_len=self.phrase_len[idx],
            targets=self.targets[idx],
            conv_index=self.conv_index[idx],
            prev_pos=self.prev_pos[idx],
            prev_is_assistant=self.prev_is_assistant[idx],
        )


def capture_targets(
    hidden: Float[Tensor, "pos d"], spans: list[PhraseSpan]
) -> Float[Tensor, "n d"]:
    """Gather the target activation at each span's ``prev_pos`` from one conversation's stack."""
    idx = torch.tensor([s.prev_pos for s in spans], dtype=torch.long, device=hidden.device)
    return hidden[idx]


def pack_phrase_ids(
    rendered_ids: list[int], spans: list[PhraseSpan], *, max_n: int
) -> tuple[Int[Tensor, "n max_n"], Int[Tensor, "n"]]:
    """Slice each span's phrase tokens out of the rendered conversation, right-pad with -1."""
    out = torch.full((len(spans), max_n), PAD_ID, dtype=torch.long)
    lens = torch.zeros(len(spans), dtype=torch.long)
    for row, span in enumerate(spans):
        ids = rendered_ids[span.start : span.start + span.n_tokens]
        out[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        lens[row] = len(ids)
    return out, lens


def left_pad_ids(
    id_rows: Int[Tensor, "b max_n"],
    lens: Int[Tensor, "b"],
    *,
    pad_id: int,
    device: str | torch.device,
    width: int | None = None,
) -> tuple[Int[Tensor, "b p"], Int[Tensor, "b p"]]:
    """LEFT-pad right-padded id rows so every row's last real token sits at index -1.

    The existing ``template_lens.capture.left_pad_batch`` takes *strings*; the reconstructor
    must feed the phrase's original token ids (re-tokenizing decoded text can fragment tokens —
    the column-zero gotcha), hence this id-level variant. Returns (input_ids, attention_mask).

    ``width`` pins a FIXED sequence length (pad-to-32) so every training batch has one static
    shape — required for ``torch.compile`` on the backbone blocks to compile once, not per
    batch-max-length. None keeps the dynamic batch-max behavior.
    """
    batch, _ = id_rows.shape
    width = width if width is not None else int(lens.max().item())
    if int(lens.max().item()) > width:
        raise ValueError(f"width {width} < longest row {int(lens.max().item())}")
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    attention = torch.zeros((batch, width), dtype=torch.long)
    for row in range(batch):
        n = int(lens[row].item())
        input_ids[row, width - n :] = id_rows[row, :n]
        attention[row, width - n :] = 1
    return input_ids.to(device), attention.to(device)


def save_pair_shard(
    path: Path,
    *,
    phrase_ids: Int[Tensor, "n max_n"],
    phrase_len: Int[Tensor, "n"],
    targets: Float[Tensor, "n d"],
    conv_index: Int[Tensor, "n"],
    prev_pos: Int[Tensor, "n"],
    prev_is_assistant: Bool[Tensor, "n"],
    meta: PairShardMeta,
) -> None:
    tmp = path.with_suffix(".tmp")
    save_file(
        {
            "phrase_ids": phrase_ids.cpu(),
            "phrase_len": phrase_len.cpu(),
            "targets": targets.to(torch.bfloat16).cpu(),
            "conv_index": conv_index.cpu(),
            "prev_pos": prev_pos.cpu(),
            "prev_is_assistant": prev_is_assistant.cpu(),
        },
        str(tmp),
        metadata=meta.to_metadata(),
    )
    tmp.replace(path)  # atomic against retries


def load_pair_shards(paths: list[Path]) -> tuple[PairBatch, PairShardMeta]:
    """Concatenate shards (row order = path order). Meta from the first shard; layer/model/t_max
    consistency across shards is asserted."""
    if not paths:
        raise ValueError("no shard paths given")
    parts = [load_file(str(p), device="cpu") for p in paths]
    metas = [_read_meta(p) for p in paths]
    first = metas[0]
    for m in metas[1:]:
        if (m.model_id, m.layer, m.t_max) != (first.model_id, first.layer, first.t_max):
            raise ValueError(f"shard meta mismatch: {m} vs {first}")
    max_n = max(int(t["phrase_ids"].shape[1]) for t in parts)
    padded = []
    for t in parts:
        ids = t["phrase_ids"]
        if ids.shape[1] < max_n:
            pad = torch.full((ids.shape[0], max_n - ids.shape[1]), PAD_ID, dtype=torch.long)
            ids = torch.cat([ids, pad], dim=1)
        padded.append(ids)
    batch = PairBatch(
        phrase_ids=torch.cat(padded),
        phrase_len=torch.cat([t["phrase_len"] for t in parts]),
        targets=torch.cat([t["targets"] for t in parts]),
        conv_index=torch.cat([t["conv_index"] for t in parts]),
        prev_pos=torch.cat([t["prev_pos"] for t in parts]),
        prev_is_assistant=torch.cat([t["prev_is_assistant"] for t in parts]),
    )
    return batch, first


def _read_meta(path: Path) -> PairShardMeta:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta is None:
        raise ValueError(f"{path}: shard has no metadata")
    return PairShardMeta.from_metadata(meta)
