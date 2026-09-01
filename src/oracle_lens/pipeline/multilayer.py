"""Multi-layer long-span pairs: one span -> the residual at prev_pos at MANY layers.

The single-layer AR (``shards.LongSpanPairs``) stores one target per span (the layer-44 residual
at the preceding position). The multi-layer reconstructor predicts the residual at every layer in
``LAYERS`` = {20,24,...,60}, so a row carries ``targets[n, n_layers, d]`` — captured in ONE forward
per conversation (``dump.conversation_layers_resid``). Schema is otherwise identical to the ragged
single-layer format (flat ``span_ids`` + ``offsets``), so the same span-length machinery applies.
"""

import os
import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.utils.data
from jaxtyping import Bool, Float, Int
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

LAYERS: tuple[int, ...] = (*range(0, 61, 4), 63)  # 0,4,..,60 + last block 63 (output residual) — 17


class LazyTargets:
    """Memory-mapped per-row ``[n_layers, d]`` targets, read from the safetensors shards on demand.

    The multi-layer target stack is ``174 KB/pair`` (17 layers x 5120 x bf16), so the full pool
    (44.7M-token RECON = ~250 GB, 100M = ~570 GB) cannot be held resident — a fully-materialized
    ``targets`` tensor overflows RAM/``/dev/shm`` and SIGSEGVs on ``share_memory_``. This reads one
    row at a time from the mmap'd shard files instead: nothing large is ever resident, so the ladder
    scales past RAM. Handles are (re)opened lazily per process so it survives DDP ``mp.spawn`` and
    DataLoader worker forks (pickled without live handles; each fork/process reopens on first read).

    Quacks like a Tensor's first axis: ``t[int]`` -> one row ``[n_layers, d]``; ``t[LongTensor]`` ->
    a new *view* (no copy) so ``MultiLayerPairs.select`` stays a pure index remap.
    """

    def __init__(
        self,
        paths: list[Path],
        shard_rows: list[int],
        n_layers: int,
        d: int,
        row_map: Tensor | None = None,
    ) -> None:
        self._paths = [str(p) for p in paths]
        cum = torch.tensor(shard_rows).cumsum(0).tolist()
        self._cum = torch.tensor([0, *cum], dtype=torch.int64)
        self._row_map = row_map  # view index -> global row (None = identity over the whole pool)
        self.n_layers, self.d = n_layers, d
        self._handles: list[Any] | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return int(self._row_map.shape[0]) if self._row_map is not None else int(self._cum[-1])

    def _open(self) -> list[Any]:
        pid = os.getpid()
        if self._handles is None or self._pid != pid:  # (re)open after fork / in a fresh process
            self._handles = [safe_open(p, framework="pt") for p in self._paths]
            self._pid = pid
        return self._handles

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, Tensor) and i.dim() > 0:  # advanced index -> a remapped view (select)
            base = self._row_map if self._row_map is not None else torch.arange(int(self._cum[-1]))
            return LazyTargets(
                [Path(p) for p in self._paths],
                (self._cum[1:] - self._cum[:-1]).tolist(),
                self.n_layers,
                self.d,
                row_map=base[i.long()],
            )
        g = int(self._row_map[int(i)]) if self._row_map is not None else int(i)
        s = int(torch.searchsorted(self._cum, torch.tensor(g), right=True)) - 1
        local = g - int(self._cum[s])
        return self._open()[s].get_slice("targets")[local]  # [n_layers, d], bf16, from mmap

    def __getstate__(self) -> dict[str, Any]:
        st = self.__dict__.copy()
        st["_handles"], st["_pid"] = None, None  # never pickle live mmap handles
        return st


@dataclass(frozen=True)
class MultiLayerShardMeta:
    """Provenance for one multi-layer shard (all values stringified for safetensors)."""

    model_id: str
    dataset_id: str
    layers: tuple[int, ...]
    t_max: int
    n_max: int
    n_min: int
    length_law: str
    region_start: int
    region_end: int
    split: str
    seed: int
    git_commit: str
    n_pairs: int
    schema: str = "multilayer_v1"
    # "resid" = true residuals from a teacher-forced capture; "recon" = frozen-AR predictions
    # (the long-oracle-lens precompute). ``ar_ckpt`` names the AR run for recon shards.
    targets_kind: str = "resid"
    ar_ckpt: str = ""

    def to_metadata(self) -> dict[str, str]:
        d = asdict(self)
        d["layers"] = ",".join(str(x) for x in self.layers)
        return {k: str(v) for k, v in d.items()}

    @classmethod
    def from_metadata(cls, meta: dict[str, str]) -> "MultiLayerShardMeta":
        return cls(
            model_id=meta["model_id"],
            dataset_id=meta["dataset_id"],
            layers=tuple(int(x) for x in meta["layers"].split(",")),
            t_max=int(meta["t_max"]),
            n_max=int(meta["n_max"]),
            n_min=int(meta["n_min"]),
            length_law=meta["length_law"],
            region_start=int(meta["region_start"]),
            region_end=int(meta["region_end"]),
            split=meta["split"],
            seed=int(meta["seed"]),
            git_commit=meta["git_commit"],
            n_pairs=int(meta["n_pairs"]),
            schema=meta.get("schema", "multilayer_v1"),
            targets_kind=meta.get("targets_kind", "resid"),  # pre-existing shards = residuals
            ar_ckpt=meta.get("ar_ckpt", ""),
        )


@dataclass(frozen=True)
class MultiLayerPairs:
    """Ragged spans + a per-layer target stack (targets[n, n_layers, d], raw residual)."""

    span_ids: Int[Tensor, "total"]
    offsets: Int[Tensor, "n_plus_1"]
    targets: Float[Tensor, "n n_layers d"]
    layers: tuple[int, ...]
    conv_index: Int[Tensor, "n"]
    prev_pos: Int[Tensor, "n"]
    prev_token_id: Int[Tensor, "n"]
    prev_is_assistant: Bool[Tensor, "n"]

    def __len__(self) -> int:
        return int(self.offsets.shape[0]) - 1

    @property
    def lengths(self) -> Int[Tensor, "n"]:
        return self.offsets[1:] - self.offsets[:-1]

    def row_ids(self, i: int) -> Int[Tensor, "n_i"]:
        return self.span_ids[int(self.offsets[i]) : int(self.offsets[i + 1])].long()

    def select(self, idx: Int[Tensor, "m"]) -> "MultiLayerPairs":
        """Row subset, vectorized (mirrors LongSpanPairs.select)."""
        idx = idx.long()
        lens = self.lengths[idx]
        new_offsets = torch.zeros(len(idx) + 1, dtype=torch.int64)
        torch.cumsum(lens, dim=0, out=new_offsets[1:])
        starts = self.offsets[idx]
        pos = torch.repeat_interleave(starts - new_offsets[:-1], lens) + torch.arange(
            int(lens.sum())
        )
        return MultiLayerPairs(
            span_ids=self.span_ids[pos],
            offsets=new_offsets,
            targets=self.targets[idx],
            layers=self.layers,
            conv_index=self.conv_index[idx],
            prev_pos=self.prev_pos[idx],
            prev_token_id=self.prev_token_id[idx],
            prev_is_assistant=self.prev_is_assistant[idx],
        )

    def share_memory_(self) -> "MultiLayerPairs":
        for t in (
            self.span_ids,
            self.offsets,
            self.targets,
            self.conv_index,
            self.prev_pos,
            self.prev_token_id,
            self.prev_is_assistant,
        ):
            if isinstance(t, Tensor):  # LazyTargets is mmap-backed (shared via the OS) — skip
                t.share_memory_()  # type: ignore[no-untyped-call]
        return self


def save_multilayer_shard(path: Path, pairs: MultiLayerPairs, meta: MultiLayerShardMeta) -> None:
    if len(pairs) != meta.n_pairs:
        raise ValueError(f"meta.n_pairs {meta.n_pairs} != rows {len(pairs)}")
    tmp = path.with_suffix(".tmp")
    save_file(
        {
            "span_ids": pairs.span_ids.to(torch.int32).cpu(),
            "offsets": pairs.offsets.cpu(),
            "targets": pairs.targets.to(torch.bfloat16).cpu(),
            "conv_index": pairs.conv_index.cpu(),
            "prev_pos": pairs.prev_pos.cpu(),
            "prev_token_id": pairs.prev_token_id.cpu(),
            "prev_is_assistant": pairs.prev_is_assistant.cpu(),
        },
        str(tmp),
        metadata=meta.to_metadata(),
    )
    tmp.replace(path)  # atomic against retries


def _read_meta(path: Path) -> MultiLayerShardMeta:
    with safe_open(str(path), framework="pt") as f:
        return MultiLayerShardMeta.from_metadata(dict(f.metadata()))


def load_multilayer_shards(paths: list[Path]) -> tuple[MultiLayerPairs, MultiLayerShardMeta]:
    """Concatenate shards (row order = path order), shifting offsets; layers/model must match."""
    if not paths:
        raise ValueError("no shard paths given")
    parts = [load_file(str(p), device="cpu") for p in paths]
    metas = [_read_meta(p) for p in paths]
    first = metas[0]
    for m in metas[1:]:
        if (m.model_id, m.layers, m.schema, m.targets_kind, m.ar_ckpt) != (
            first.model_id, first.layers, first.schema, first.targets_kind, first.ar_ckpt
        ):
            raise ValueError(f"shard meta mismatch: {m} vs {first}")
    offsets_parts: list[Tensor] = []
    shift = 0
    for t in parts:
        off = t["offsets"]
        offsets_parts.append(off[1:] + shift if shift or offsets_parts else off)
        shift += int(off[-1])
    pairs = MultiLayerPairs(
        span_ids=torch.cat([t["span_ids"] for t in parts]),
        offsets=torch.cat(offsets_parts),
        targets=torch.cat([t["targets"] for t in parts]),
        layers=first.layers,
        conv_index=torch.cat([t["conv_index"] for t in parts]),
        prev_pos=torch.cat([t["prev_pos"] for t in parts]),
        prev_token_id=torch.cat([t["prev_token_id"] for t in parts]),
        prev_is_assistant=torch.cat([t["prev_is_assistant"] for t in parts]),
    )
    total = sum(int(m.n_pairs) for m in metas)
    if len(pairs) != total:
        raise ValueError(f"row count {len(pairs)} != sum of shard metas {total}")
    return pairs, first


_SMALL_KEYS = (
    "span_ids", "offsets", "conv_index", "prev_pos", "prev_token_id", "prev_is_assistant",
)


def load_multilayer_shards_lazy(paths: list[Path]) -> tuple[MultiLayerPairs, MultiLayerShardMeta]:
    """Like :func:`load_multilayer_shards` but the big ``targets`` stack stays on disk (mmap).

    Only the small ragged/metadata tensors (~200 MB for the whole RECON pool) are read into RAM and
    concatenated; ``targets`` becomes a :class:`LazyTargets` reading rows from the shard mmaps on
    demand. This is what makes the >RAM rungs (all / 100M) trainable — see ``LazyTargets``.
    """
    if not paths:
        raise ValueError("no shard paths given")
    metas = [_read_meta(p) for p in paths]
    first = metas[0]
    for m in metas[1:]:
        if (m.model_id, m.layers, m.schema, m.targets_kind, m.ar_ckpt) != (
            first.model_id, first.layers, first.schema, first.targets_kind, first.ar_ckpt
        ):
            raise ValueError(f"shard meta mismatch: {m} vs {first}")
    small: dict[str, list[Tensor]] = {k: [] for k in _SMALL_KEYS}
    d = 0
    for p in paths:
        with safe_open(str(p), framework="pt") as f:
            for k in _SMALL_KEYS:
                small[k].append(f.get_tensor(k))
            d = int(f.get_slice("targets").get_shape()[-1])
    offsets_parts: list[Tensor] = []
    shift = 0
    for off in small["offsets"]:
        offsets_parts.append(off[1:] + shift if shift or offsets_parts else off)
        shift += int(off[-1])
    shard_rows = [int(m.n_pairs) for m in metas]
    targets = LazyTargets(paths, shard_rows, n_layers=len(first.layers), d=d)
    pairs = MultiLayerPairs(
        span_ids=torch.cat(small["span_ids"]),
        offsets=torch.cat(offsets_parts),
        targets=targets,  # type: ignore[arg-type]
        layers=first.layers,
        conv_index=torch.cat(small["conv_index"]),
        prev_pos=torch.cat(small["prev_pos"]),
        prev_token_id=torch.cat(small["prev_token_id"]),
        prev_is_assistant=torch.cat(small["prev_is_assistant"]),
    )
    if len(pairs) != sum(shard_rows):
        raise ValueError(f"row count {len(pairs)} != sum of shard metas {sum(shard_rows)}")
    return pairs, first


def _octave(length: int) -> int:
    """Octave bucket floor: lengths 1->0, 2-3->1, 4-7->2, ... (== length.bit_length()-1)."""
    return max(0, int(length).bit_length() - 1)


def length_hist_fractions(lengths: Int[Tensor, "n"]) -> dict[int, float]:
    """Fraction of spans in each octave length bucket (sums to 1). The empirical length dist."""
    counts: dict[int, int] = {}
    for x in lengths.tolist():
        b = _octave(x)
        counts[b] = counts.get(b, 0) + 1
    total = max(1, sum(counts.values()))
    return {b: n / total for b, n in counts.items()}


def length_matched_indices(
    lengths: Int[Tensor, "n"], target_frac: dict[int, float], n_total: int, seed: int = 1234
) -> Int[Tensor, "m"]:
    """Pick ``n_total`` row indices whose octave-length histogram matches ``target_frac``.

    Removes the span-length confound between data cells: subsample each cell to a common length
    distribution so any downstream FVE difference is content, not length. Per bucket keep
    ``round(n_total * target_frac[b])`` rows (sampled without replacement; with replacement only if
    a bucket is too scarce — rare given large pools). Buckets absent from ``target_frac`` are cut.
    """
    by_bucket: dict[int, list[int]] = {}
    for i, x in enumerate(lengths.tolist()):
        by_bucket.setdefault(_octave(x), []).append(i)
    gen = torch.Generator().manual_seed(seed)
    picks: list[Tensor] = []
    for b, frac in sorted(target_frac.items()):
        want = round(n_total * frac)
        avail = by_bucket.get(b, [])
        if want <= 0 or not avail:
            continue
        idx = torch.tensor(avail, dtype=torch.int64)
        if want <= len(avail):
            picks.append(idx[torch.randperm(len(avail), generator=gen)[:want]])
        else:  # scarce bucket — sample with replacement to hit the target fraction
            picks.append(idx[torch.randint(len(avail), (want,), generator=gen)])
    if not picks:
        return torch.zeros(0, dtype=torch.int64)
    out = torch.cat(picks)
    return out[torch.randperm(len(out), generator=gen)]  # shuffle so buckets interleave


class _CroppedTargets:
    """Row-indexed view over a source targets object (LazyTargets, FileTargets or Tensor)."""

    def __init__(self, src: Any, row_idx: Int[Tensor, "m"]) -> None:
        self._src = src
        self._row_idx = row_idx

    def __len__(self) -> int:
        return int(self._row_idx.shape[0])

    def __getitem__(self, i: int) -> Tensor:
        out: Tensor = self._src[int(self._row_idx[int(i)])]
        return out


class FileTargets:
    """Container-LOCAL raw-file mmap of ``[n, n_layers, d]`` bf16 targets (``torch.from_file``).

    The localized counterpart of LazyTargets: same per-pid lazy (re)open contract (survives
    mp.spawn + DataLoader forks; never pickles a live map), but backed by local NVMe where random
    reads are ~free — the fix for network-volume target I/O dominating step time at big rungs."""

    def __init__(self, path: str, n: int, n_layers: int, d: int) -> None:
        self.path, self.n, self.n_layers, self.d = path, n, n_layers, d
        self._map: Tensor | None = None
        self._pid: int | None = None

    def _open(self) -> Tensor:
        pid = os.getpid()
        if self._map is None or self._pid != pid:
            flat = torch.from_file(
                self.path, shared=True, size=self.n * self.n_layers * self.d, dtype=torch.bfloat16
            )
            self._map = flat.view(self.n, self.n_layers, self.d)
            self._pid = pid
        return self._map

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tensor:
        return self._open()[int(i)]

    def __getstate__(self) -> dict[str, Any]:
        st = self.__dict__.copy()
        st["_map"], st["_pid"] = None, None
        return st


class CroppedPairs:
    """Prefix-crop view of MultiLayerPairs: crop ``i`` = FIRST ``crop_len[i]`` tokens of source row
    ``row_idx[i]``, SAME target (the residual at start-1 precedes the span, so a prefix keeps the
    pair valid: "sample a position, take the next N tokens"). Zero-copy — no new capture/storage.
    Implements the surface MLReconDataset/TokenBudgetSampler/train loop touch: __len__, lengths,
    row_ids, targets, select. ``targets_override``/``target_idx`` (set by ``localize_crops``)
    redirect ONLY the target reads (ids still resolve through source+row_idx)."""

    def __init__(self, source: MultiLayerPairs, row_idx: Int[Tensor, "m"],
                 crop_len: Int[Tensor, "m"], targets_override: Any = None,
                 target_idx: Int[Tensor, "m"] | None = None) -> None:
        self.source = source
        self.row_idx = row_idx.long()
        self.crop_len = crop_len.long()
        self._targets_override = targets_override
        self._target_idx = target_idx.long() if target_idx is not None else None
        if targets_override is not None:
            assert self._target_idx is not None
            self.targets = _CroppedTargets(targets_override, self._target_idx)
        else:
            self.targets = _CroppedTargets(source.targets, self.row_idx)

    def __len__(self) -> int:
        return int(self.row_idx.shape[0])

    @property
    def lengths(self) -> Int[Tensor, "m"]:
        return self.crop_len

    def row_ids(self, i: int) -> Int[Tensor, "n_i"]:
        return self.source.row_ids(int(self.row_idx[i]))[: int(self.crop_len[i])]

    def select(self, idx: Int[Tensor, "k"]) -> "CroppedPairs":
        idx = idx.long()
        return CroppedPairs(
            self.source, self.row_idx[idx], self.crop_len[idx],
            targets_override=self._targets_override,
            target_idx=self._target_idx[idx] if self._target_idx is not None else None,
        )

    def share_memory_(self) -> "CroppedPairs":
        """DDP mp.spawn contract (mirrors MultiLayerPairs): share the source + crop tensors."""
        self.source.share_memory_()
        self.row_idx.share_memory_()  # type: ignore[no-untyped-call]
        self.crop_len.share_memory_()  # type: ignore[no-untyped-call]
        if self._target_idx is not None:
            self._target_idx.share_memory_()  # type: ignore[no-untyped-call]
        return self


def localize_crops(
    crops_list: list[CroppedPairs], out_path: str, *, chunk_rows: int = 65536
) -> list[CroppedPairs]:
    """Copy the union of the crops' needed target rows from the network-volume shard mmaps into ONE
    container-local raw file, then redirect the crops' target reads to it (ids untouched).

    Reads each source shard in big SEQUENTIAL chunks (the access pattern volumes serve at line
    rate) and keeps only wanted rows — predictable wall-clock (scan of all shards) instead of
    hours of random-read tax at training time. All CroppedPairs must share the same ``source``.
    Pass train+eval together so the union lands in one file."""
    src = crops_list[0].source
    assert all(c.source is src for c in crops_list)
    tgt = src.targets
    assert isinstance(tgt, LazyTargets), "localize_crops expects a LazyTargets-backed source"
    all_rows = torch.cat([c.row_idx for c in crops_list])
    uniq = torch.unique(all_rows)  # sorted ascending = shard-major order
    # source may itself be a select() view: resolve to GLOBAL rows for the scan, keeping slot order
    gmap = tgt._row_map if tgt._row_map is not None else None
    grows = gmap[uniq] if gmap is not None else uniq
    order = torch.argsort(grows)
    grows_sorted = grows[order]
    slot_of_uniqpos = torch.empty_like(order)
    slot_of_uniqpos[order] = torch.arange(len(order))
    n, nl, d = len(uniq), tgt.n_layers, tgt.d
    with open(out_path, "wb") as fh:  # pre-allocate (from_file maps an existing file)
        # posix_fallocate RESERVES the blocks now: insufficient space raises a clean OSError here
        # instead of a silent SIGBUS/SIGSEGV mid-scan when an mmap store hits the cap (2026-07-27
        # crash loops: /dev/shm tmpfs cap AND /tmp default quota both ~60 GB)
        os.posix_fallocate(fh.fileno(), 0, n * nl * d * 2)
    flat = torch.from_file(
        out_path, shared=True, size=n * nl * d, dtype=torch.bfloat16
    )
    local = flat.view(n, nl, d)
    handles = tgt._open()
    cum = tgt._cum
    t0 = time.time()
    done = 0
    for s in range(len(handles)):
        lo, hi = int(cum[s]), int(cum[s + 1])
        w_lo = int(torch.searchsorted(grows_sorted, lo))
        w_hi = int(torch.searchsorted(grows_sorted, hi))
        if w_lo == w_hi:
            continue
        sl = handles[s].get_slice("targets")
        for start in range(lo, hi, chunk_rows):
            end = min(start + chunk_rows, hi)
            c_lo = int(torch.searchsorted(grows_sorted, start))
            c_hi = int(torch.searchsorted(grows_sorted, end))
            if c_lo == c_hi:
                continue
            block = sl[start - lo : end - lo]  # sequential chunk read
            want = grows_sorted[c_lo:c_hi] - start
            local[c_lo:c_hi] = block[want]
            done += c_hi - c_lo
        print(
            f"[localize] shard {s}: {done}/{n} rows ({time.time() - t0:.0f}s)", flush=True
        )
    # local slot for each uniq position; then remap every crop's rows -> slots (vectorized:
    # uniq is sorted, so searchsorted(uniq, rows) gives each row's uniq-position)
    ft = FileTargets(out_path, n, nl, d)
    out = []
    for c in crops_list:
        tidx = slot_of_uniqpos[torch.searchsorted(uniq, c.row_idx)]
        out.append(CroppedPairs(src, c.row_idx, c.crop_len, targets_override=ft, target_idx=tidx))
    print(f"[localize] {n} rows -> {out_path} ({n * nl * d * 2 / 1e9:.1f} GB, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return out


def stratified_crop_pool(
    lengths: Int[Tensor, "n"], lo: int, hi: int, per_n: int, seed: int = 1234,
    exclude: tuple[Int[Tensor, "k"], Int[Tensor, "k"]] | None = None,
) -> tuple[Int[Tensor, "m"], Int[Tensor, "m"]]:
    """Stratified-uniform crop pool over every integer N in lo..hi — see
    ``stratified_crop_pool_set`` (this is the ns=range(lo, hi+1) special case)."""
    return stratified_crop_pool_set(
        lengths, tuple(range(lo, hi + 1)), per_n, seed=seed, exclude=exclude
    )


def stratified_crop_pool_set(
    lengths: Int[Tensor, "n"], ns: tuple[int, ...], per_n: int, seed: int = 1234,
    exclude: tuple[Int[Tensor, "k"], Int[Tensor, "k"]] | None = None,
) -> tuple[Int[Tensor, "m"], Int[Tensor, "m"]]:
    """Stratified crop pool over an EXPLICIT length set: for each N in ``ns`` draw ``per_n``
    rows WITHOUT replacement from rows with length >= N (a long row may serve several N — a few
    views per target, never the same (row, N) twice). Ordered ROUND-ROBIN over N (rank-within-N
    major): the first R crops of the pool are an exact-uniform nested rung, so prefix = scaling
    rung. Returns (row_idx, crop_len). Prints a loud warning when a length is scarce (< per_n).
    ``exclude`` = (rows, lens) of a PRIOR pool (reconstructible from its seed): those exact
    (row, N) pairs are removed from eligibility, so a second-view pool is 100% disjoint from
    what a previous run trained on (continuation experiments: every sample never-seen).

    ``ns=(1, 2, 4, 8, 16, 32)`` is the iolens AR law (Agam 2026-08-01): equal counts per
    power-of-2 length, nothing above 32 — long captured spans serve as crop sources only."""
    gen = torch.Generator().manual_seed(seed)
    excl_by_n: dict[int, Tensor] = {}
    if exclude is not None:
        ex_rows, ex_lens = exclude
        for n in ns:
            excl_by_n[n] = ex_rows[ex_lens == n]
    per_len_rows: list[Tensor] = []
    short: dict[int, int] = {}
    for n in ns:
        elig = torch.nonzero(lengths >= n).squeeze(-1)
        if n in excl_by_n and len(excl_by_n[n]):
            elig = elig[~torch.isin(elig, excl_by_n[n])]
        take = min(per_n, len(elig))
        if take < per_n:
            short[n] = len(elig)
        picked = elig[torch.randperm(len(elig), generator=gen)[:take]]
        per_len_rows.append(picked)
    if short:
        print(f"[crop-pool] WARNING scarce lengths (want {per_n}): {short}", flush=True)
    # vectorized round-robin (rank-major, N minor): pad ragged per-N lists to [n_lens, depth],
    # transpose -> [depth, n_lens], flatten, drop padding
    n_lens = len(ns)
    depth = max(len(r) for r in per_len_rows)
    mat = torch.full((n_lens, depth), -1, dtype=torch.int64)
    for j, rows in enumerate(per_len_rows):
        mat[j, : len(rows)] = rows
    lens_mat = torch.tensor(ns, dtype=torch.int64).unsqueeze(1).expand(n_lens, depth)
    flat_rows = mat.T.reshape(-1)
    flat_lens = lens_mat.T.reshape(-1)
    keep = flat_rows >= 0
    return flat_rows[keep], flat_lens[keep]


class StreamingCropDataset(torch.utils.data.IterableDataset):  # type: ignore[type-arg]
    """Volume-friendly streaming of a CroppedPairs pool: iterate crops in SOURCE STORAGE ORDER
    (so target reads are near-sequential over the network volume — the access pattern it serves at
    line rate; no local materialization, no capacity walls), through a shuffle buffer.

    Each rank takes the disjoint stride ``storage_order[rank::world]`` (balanced, DDP-clean).
    Batches are drawn uniformly from a ``buffer_rows`` buffer that the scan keeps full — when the
    buffer covers the whole per-rank stream (rungs <= world*buffer_rows) this is EXACTLY a global
    uniform shuffle (same semantics as the map-style path); above that it is a windowed shuffle
    (window = buffer/stream, stated in the run log). Epoch reshuffles via ``set_epoch``-free
    design: one pass = one epoch (epochs=1 everywhere here)."""

    def __init__(
        self, crops: CroppedPairs, *, rank: int, world: int, buffer_rows: int = 150_000,
        seed: int = 0, skip_batches: int = 0, micro_batch: int = 0,
    ) -> None:
        self.crops = crops
        self.rank, self.world = rank, max(1, world)
        self.buffer_rows = buffer_rows
        self.seed = seed
        # Resume fast-forward: replay the stream and DISCARD the first ``skip_batches``
        # micro-batches this rank already trained on (exact: the stream is a pure function of
        # (seed, rank, world, workers)). Requires ``micro_batch`` to convert batches -> yields.
        self.skip_batches = max(0, skip_batches)
        self.micro_batch = micro_batch
        if self.skip_batches and not micro_batch:
            raise ValueError("skip_batches needs micro_batch to convert batches to rows")
        # storage order: sort by the source row's GLOBAL row (through a select()'s row_map if any)
        tgt = crops.source.targets
        rmap = tgt._row_map if isinstance(tgt, LazyTargets) and tgt._row_map is not None else None
        key = rmap[crops.row_idx] if rmap is not None else crops.row_idx
        self._order = torch.argsort(key)

    def _mine(self) -> Tensor:
        return self._order[self.rank :: self.world]

    def __len__(self) -> int:
        return int(self._mine().shape[0])

    def _iter_worker(self, wid: int, wn: int) -> Iterator[dict[str, Tensor]]:
        """One DataLoader worker's sub-stride. Multiple worker PROCESSES are the parallelism that
        makes streaming feed the GPUs: a single reader does blocking ~tens-of-ms per-row volume
        reads (mmap page faults serialize on the GIL — threads don't help), which starved the
        first all-data attempt (loadavg 0, +12 GB/10 min). ``buffer_rows`` is the TOTAL per-rank
        buffer, split across workers; progress prints keep the fill phase visible to watchdogs."""
        mine = self._mine()[wid::wn]
        n = int(mine.shape[0])
        window = max(1, min(self.buffer_rows // max(1, wn), n))
        rng = random.Random(self.seed + 7919 * self.rank + 104_729 * wid)
        # Resume fast-forward: the DataLoader round-robins COMPLETE micro-batches over workers,
        # so worker ``wid`` owns batches {k : k % wn == wid} of the first ``skip_batches``. Those
        # yields are replayed and discarded WITHOUT reading targets (the expensive mmap part):
        # rows buffered during the skip window carry a deferred index instead of a target, and
        # any that survive into the live region are materialized at yield time (one-off
        # random-read cost bounded by the buffer window).
        skip_yields = 0
        if self.skip_batches:
            n_batches_w = self.skip_batches // wn + (1 if self.skip_batches % wn > wid else 0)
            skip_yields = n_batches_w * self.micro_batch
        yielded = 0
        buf: list[dict[str, Tensor]] = []
        t0 = time.time()

        def _materialize(row: dict[str, Tensor]) -> dict[str, Tensor]:
            if "_defer" in row:
                i = int(row.pop("_defer").item())
                row["target"] = self.crops.targets[i].float()
            return row

        for k, i in enumerate(mine.tolist()):
            if yielded < skip_yields:
                buf.append({"ids": self.crops.row_ids(i), "_defer": torch.tensor(i)})
            else:
                buf.append({"ids": self.crops.row_ids(i), "target": self.crops.targets[i].float()})
            if (k + 1) % 25_000 == 0:
                print(
                    f"[stream] rank{self.rank} w{wid}: {k + 1}/{n} rows "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if len(buf) >= window:
                row = buf.pop(rng.randrange(len(buf)))
                yielded += 1
                if yielded > skip_yields:
                    yield _materialize(row)
                elif yielded == skip_yields:
                    print(f"[stream] rank{self.rank} w{wid}: skipped {skip_yields} resumed rows",
                          flush=True)
        rng.shuffle(buf)
        for row in buf:
            yielded += 1
            if yielded > skip_yields:
                yield _materialize(row)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        info = torch.utils.data.get_worker_info()
        wid, wn = (int(info.id), int(info.num_workers)) if info is not None else (0, 1)
        return self._iter_worker(wid, wn)


def uniform_length_indices(
    lengths: Int[Tensor, "n"], lo: int, hi: int, n_total: int, seed: int = 1234
) -> Int[Tensor, "m"]:
    """Row indices approximating the paper span law N ~ uniform{lo..hi}: EQUAL counts per exact
    length. The dumps drew loguniform (density ~ 1/n, short-heavy), so uniform needs per-length
    rebalancing, bounded by the scarcest length (usually n=hi). Takes ``n_total // (hi-lo+1)`` rows
    per length (all available if scarce — prints a loud warning, since a short-fall skews the law).
    Returns a shuffled index tensor."""
    by_len: dict[int, list[int]] = {}
    for i, x in enumerate(lengths.tolist()):
        if lo <= x <= hi:
            by_len.setdefault(x, []).append(i)
    want = max(1, n_total // (hi - lo + 1))
    gen = torch.Generator().manual_seed(seed)
    picks: list[Tensor] = []
    short = {}
    for n in range(lo, hi + 1):
        avail = by_len.get(n, [])
        take = min(want, len(avail))
        if take < want:
            short[n] = len(avail)
        if not take:
            continue
        idx = torch.tensor(avail, dtype=torch.int64)
        picks.append(idx[torch.randperm(len(avail), generator=gen)[:take]])
    if short:
        print(f"[uniform-lengths] WARNING scarce lengths (want {want}/len): {short}", flush=True)
    if not picks:
        return torch.zeros(0, dtype=torch.int64)
    out = torch.cat(picks)
    return out[torch.randperm(len(out), generator=gen)]


def pack_token_budget(
    lengths: list[int], budget: int, max_rows: int, seed: int
) -> list[list[int]]:
    """Greedy length-sorted packing into variable-size batches with ``rows * max_len <= budget``.

    Sorting by length keeps each batch near-uniform in length, so pad-to-batch-max wastes little,
    and the per-batch padded-token count is bounded by ``budget`` regardless of span length. That
    bound is what makes grad-checkpointing OFF safe (a batch that catches a 1024-tok span holds only
    ~``budget/1024`` rows, not ``micro_batch`` of them). Spans longer than ``budget`` (never happens
    at budget 4096, max span 1024) form a lone over-budget batch rather than being dropped/split.
    A small random tie-break varies packing across seeds.
    """
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], rng.random()))
    batches: list[list[int]] = []
    cur: list[int] = []
    curmax = 0
    for i in order:
        length = lengths[i]
        newmax = length if length > curmax else curmax
        if cur and ((len(cur) + 1) * newmax > budget or len(cur) >= max_rows):
            batches.append(cur)
            cur, curmax = [i], length
        else:
            cur.append(i)
            curmax = newmax
    if cur:
        batches.append(cur)
    return batches


def token_budget_nbatches(
    lengths: list[int], budget: int, world: int, max_rows: int, seed: int = 0
) -> int:
    """Per-rank optimizer-batch count for a token-budget schedule (for ``max_steps``)."""
    return len(pack_token_budget(lengths, budget, max_rows, seed)) // max(1, world)


class TokenBudgetSampler:
    """DDP-aware token-budget batch sampler: variable-size, length-sorted batches with padded tokens
    (``rows * max_len``) bounded by ``budget``. Batches are packed ONCE (deterministic in ``seed``,
    so ``__len__`` is stable), then each epoch their ORDER is shuffled and dealt round-robin across
    ranks — every rank runs exactly ``len(all)//world`` batches (the ``< world`` remainder is
    dropped), so DDP step counts match and never deadlock. Ranks get disjoint batch sets."""

    def __init__(
        self, lengths: list[int], budget: int, *, world: int = 1, rank: int = 0,
        seed: int = 0, max_rows: int = 256,
    ) -> None:
        self._all = pack_token_budget(lengths, budget, max_rows, seed)
        self.world = max(1, world)
        self.rank = rank
        self.seed = seed
        self._nbatches = len(self._all) // self.world
        self._epoch = 0

    def __len__(self) -> int:
        return self._nbatches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        order = list(range(len(self._all)))
        rng.shuffle(order)
        picked = order[self.rank :: self.world][: self._nbatches]
        for k in picked:
            yield self._all[k]
