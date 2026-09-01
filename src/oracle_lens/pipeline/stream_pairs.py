"""Streaming pair sampler: train continuously off a directory that a producer keeps filling.

The Inverted OLens capture rate (~300 pairs/s/GPU) far exceeds the AR's consumption rate
(~70 samples/s/cell), and 17-layer targets are 174 KB each — so the pairs for a long run never
fit on disk at once. Instead of wave-chained re-launches, the trainer STREAMS: a throttled
producer writes fresh pair shards into a buffer directory, this dataset yields their rows, and
consumed shards are deleted to make room. Training never stops and no pair is seen twice.

Contract
--------
- **Row striding, not shard ownership**: EVERY rank opens every shard and reads a disjoint row
  stride — reader ``rank * n_workers + worker_id`` of ``world * n_workers`` takes rows
  ``k, k+step, …``, with each shard truncated to a whole multiple of the stride. Every reader
  therefore gets *exactly* the same number of rows from every shard.
  This exactness is load-bearing, not tidiness: shards were previously assigned whole to a rank
  by name hash, which is only statistically balanced (observed 3/6/1 across three ranks). The
  short rank ran dry, blocked waiting for the producer while its peers kept stepping, and the
  next ALLREDUCE timed out after 600 s and killed the run (2026-08-02).
- **Consumed markers**: a worker stamps ``<shard>.done.<rank>.w<id>`` when its slice is done; the
  rank marker ``<shard>.done.<rank>`` follows once all of that rank's workers have stamped, and
  the janitor (``sweep_consumed``) deletes the shard once every rank has marked it. The
  two levels matter — promoting on worker 0 alone let the janitor delete a shard out from under
  slower workers of the same rank.
- **Resume is shard-exact**: the consumed-marker files ARE the resume state. A restarted run
  skips any shard already marked by its rank; arrival order need not be reproducible (which is
  why batch-exact replay is neither possible nor needed here).
- **Freshness**: every row is intended to be a first-time sample. That is *enforced* upstream by
  the capture slice ledger and here by a per-worker ``seen`` set; it is *verified* by
  ``scripts/ola/iolens_reconcile_counts.py`` (G9), because when this was merely asserted in a
  docstring it was false twice over.
"""

import os
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.utils.data
from torch import Tensor

SHARD_GLOB = "pairs_train_*.safetensors"


def shard_marker(shard: Path, rank: int) -> Path:
    return shard.with_suffix(f".done.{rank}")


def worker_marker(shard: Path, rank: int, wid: int) -> Path:
    """Per-(rank, worker) completion marker; the rank marker is written once all of these exist."""
    return shard.with_suffix(f".done.{rank}.w{wid}")


def list_shards(buffer_dir: Path) -> list[Path]:
    """All present shards in stable global order (filename sort)."""
    return sorted(buffer_dir.glob(SHARD_GLOB))


def sweep_consumed(buffer_dir: Path, world: int, *, keep_markers: bool = True) -> int:
    """Delete shards EVERY rank has finished, returning bytes freed.

    All ranks read every shard (disjoint row strides), so a shard is only spent once all of them
    have marked it. This is safe precisely because reading is strided rather than partitioned by
    shard: under the old whole-shard ownership, requiring all ranks to mark deadlocked the buffer
    (nothing was ever deletable), which is why that rule was briefly relaxed to owner-only.
    """
    shards = list_shards(buffer_dir)
    if not shards:
        return 0
    ranks = range(max(1, world))
    consumed = [s for s in shards if all(shard_marker(s, r).exists() for r in ranks)]
    freed = 0
    for shard in consumed:
        freed += shard.stat().st_size
        shard.unlink()
        for stale in shard.parent.glob(f"{shard.stem}.done.*.w*"):
            stale.unlink(missing_ok=True)  # per-worker markers are only useful while the shard is
        if not keep_markers:
            for r in ranks:
                shard_marker(shard, r).unlink(missing_ok=True)
    return freed


def buffer_bytes(buffer_dir: Path) -> int:
    return sum(p.stat().st_size for p in list_shards(buffer_dir))


class StreamingPairDataset(torch.utils.data.IterableDataset):  # type: ignore[type-arg]
    """Iterate pair rows from a live buffer directory, newest shards included as they land.

    Yields ``{"ids": LongTensor[n], "target": FloatTensor[n_layers, d]}`` — the same row schema
    ``MLReconDataset`` produces, so the collate/trainer path is unchanged. ``layer_indices``
    selects a subset of the stored layer stack (the AR drops layer 0, whose activations are
    essentially "which token preceded this span" and score at chance).
    """

    def __init__(
        self,
        buffer_dir: Path,
        *,
        rank: int,
        world: int,
        layer_indices: tuple[int, ...] | None = None,
        shuffle_rows: int = 20_000,
        seed: int = 0,
        wait_s: float = 30.0,
        max_wait_s: float = 3600.0,
        max_rows: int = 0,
        min_len: int = 100_000,
    ) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.rank, self.world = rank, max(1, world)
        self.layer_indices = layer_indices
        self.shuffle_rows = max(1, shuffle_rows)
        self.seed = seed
        self.wait_s = wait_s
        self.max_wait_s = max_wait_s
        self.max_rows = max_rows
        # __len__ floor: the harness divides by batch size with drop_last, so a small length
        # rounds to 0 batches and it divides by zero. The stream's true horizon is unbounded
        # (the producer keeps appending), so a floor is honest as well as safe.
        self.min_len = min_len

    def __len__(self) -> int:
        """Rows this rank can still read, from shard metadata only (no tensor loads).

        The vendored harness computes ``global_step % len(dataloader)``, so an IterableDataset
        without a length crashes it. This is an estimate by nature — the buffer grows while the
        producer runs — and it is used only for progress accounting, never for correctness
        (``max_steps`` bounds the run; the stream itself decides when rows run out).
        """
        import json as _json

        from safetensors import safe_open as _so

        total = 0
        for shard in self._my_unconsumed():
            try:
                with _so(str(shard), framework="pt") as f:
                    meta = _json.loads((f.metadata() or {}).get("n_pairs", "0"))
                    total += int(meta) if meta else int(f.get_slice("offsets").get_shape()[0]) - 1
            except Exception:  # a shard mid-write; it will be counted on the next call
                continue
        # every rank sees every shard but reads only its 1/world stride of the rows
        return max(total // max(1, self.world), self.min_len)

    def _my_unconsumed(self) -> list[Path]:
        """Shards this rank has not finished. EVERY rank sees every shard.

        Whole-shard ownership starved DDP: shard counts per rank are hash-imbalanced (observed
        3/6/1 across three ranks), so the rank that ran dry blocked in the producer wait while
        its peers kept stepping, and the next ALLREDUCE timed out after 600 s and killed the run
        (2026-08-02). Rank balance has to be exact, not statistical — hence row striding.
        """
        return [
            shard for shard in list_shards(self.buffer_dir)
            if not shard_marker(shard, self.rank).exists()
        ]

    def _rows(self, shard: Path, wid: int, wn: int) -> Iterator[dict[str, Tensor]]:
        from oracle_lens.pipeline.multilayer import load_multilayer_shards_lazy

        try:
            pairs, _meta = load_multilayer_shards_lazy([shard])
        except FileNotFoundError:
            # The shard was swept between _my_unconsumed() listing it and this open. The janitor
            # only deletes shards their owner marked consumed, so in steady state this cannot
            # happen — but an out-of-band mark (e.g. quarantining duplicate shards) can race a
            # worker that already listed it, and that killed a 3-GPU run on 2026-08-02. A shard
            # that no longer exists has no rows; losing them is strictly better than losing the
            # run, and the stream is unbounded anyway.
            print(f"[stream] rank{self.rank} w{wid}: {shard.name} vanished — skipping", flush=True)
            return
        n = len(pairs)
        # Global stride over (rank, worker): reader k of (world * wn) takes rows k, k+step, ...
        # Every reader therefore gets within one row of the same count from every shard, which is
        # what keeps DDP ranks in lockstep — they cannot run out at different times.
        offset = self.rank * wn + wid
        step = self.world * wn
        # Truncate to a whole multiple of the stride so EVERY reader gets exactly n // step rows.
        # The remainder (< world * workers rows per shard, i.e. tens out of ~100k) is dropped;
        # paying that keeps rank balance exact rather than approximate, and exact is what stops
        # a rank running dry ahead of its peers and timing out the next collective.
        n -= n % step
        for i in range(offset, n, step):
            target = torch.as_tensor(pairs.targets[i]).float()
            if self.layer_indices is not None:
                target = target[list(self.layer_indices)]
            yield {"ids": pairs.row_ids(i), "target": target}

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        info = torch.utils.data.get_worker_info()
        wid, wn = (int(info.id), int(info.num_workers)) if info is not None else (0, 1)
        rng = random.Random(self.seed + 7919 * self.rank + 104_729 * wid)
        buf: list[dict[str, Tensor]] = []
        emitted = 0
        waited = 0.0
        window = max(1, self.shuffle_rows // max(1, wn))
        # Shards THIS worker has already read. The on-disk marker is written by worker 0 only
        # (one marker per rank), but every worker filters on it — so a worker that finishes its
        # snapshot before worker 0 has marked those shards re-lists them and re-reads its whole
        # stride, yielding byte-identical rows the trainer counts as fresh samples. Measured
        # 2026-08-02: ~497k repeated rows in the pt cell (21% of its recorded examples) with no
        # duplicate capture involved at all. A per-worker set is the fix; the marker stays
        # rank-level because that is what the janitor keys deletion on.
        seen: set[str] = set()
        while True:
            shards = [s for s in self._my_unconsumed() if s.name not in seen]
            if not shards:
                if waited >= self.max_wait_s:
                    break  # producer is gone / horizon reached — drain and stop
                time.sleep(self.wait_s)
                waited += self.wait_s
                continue
            waited = 0.0
            for shard in shards:
                seen.add(shard.name)
                for row in self._rows(shard, wid, wn):
                    buf.append(row)
                    if len(buf) >= window:
                        yield buf.pop(rng.randrange(len(buf)))
                        emitted += 1
                        if self.max_rows and emitted >= self.max_rows:
                            return
                # Mark this worker's slice, then promote to the rank marker only once EVERY
                # worker of this rank is done. Marking on worker 0 alone let the janitor delete a
                # shard while slower workers were still reading it (their rows are then lost, and
                # before the FileNotFoundError guard it crashed the run).
                worker_marker(shard, self.rank, wid).touch()
                if all(worker_marker(shard, self.rank, w).exists() for w in range(wn)):
                    shard_marker(shard, self.rank).touch()
                print(f"[stream] rank{self.rank} w{wid}: finished {shard.name}", flush=True)
        rng.shuffle(buf)
        for row in buf:
            yield row
            emitted += 1
            if self.max_rows and emitted >= self.max_rows:
                return


def stream_stats(buffer_dir: Path, world: int) -> dict[str, Any]:
    shards = list_shards(Path(buffer_dir))
    consumed = sum(
        1 for s in shards if all(shard_marker(s, r).exists() for r in range(world))
    )
    return {
        "shards_present": len(shards),
        "shards_fully_consumed": consumed,
        "buffer_gb": round(sum(p.stat().st_size for p in shards) / 2**30, 1),
        "pid": os.getpid(),
    }
