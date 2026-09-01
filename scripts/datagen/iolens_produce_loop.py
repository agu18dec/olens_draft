"""Throttled capture producer + janitor: keep the AR training buffer full, forever.

Capture (~300 pairs/s/GPU) outruns AR training (~75 samples/s/GPU), and pairs are 174 KB each,
so the producer must be *throttled by disk*, not run flat out. This loop:

1. sweeps shards every rank has marked consumed (``stream_pairs.sweep_consumed``) — frees space;
2. if the buffer is under ``--low-gb``, captures the next never-captured slice and appends it;
3. otherwise sleeps.

The slice cursor is persisted, so every pass takes conversations no pass has taken before: it
walks (rollout shard x wave) pairs, where wave ``w`` uses ``--train-frac-skip w*frac``. Restarts
resume from the cursor; nothing is ever captured twice.

    CUDA_VISIBLE_DEVICES=5 uv run --no-sync python scripts/datagen/iolens_produce_loop.py \
        --cells chat:6:0.14,pt:9:0.10 --low-gb 120 --high-gb 260 --world 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)


@dataclass
class Cell:
    mode: str          # chat | pt
    n_shards: int      # FLOOR on rollout shards; the real count is discovered from disk
    frac: float        # fraction of ar_train convs captured per slice
    world: int         # DDP world of the trainer consuming this buffer

    @property
    def buffer(self) -> str:
        return f"ml_pairs_iolens_{self.mode}"

    def shards_available(self, root: Path) -> int:
        """Rollout shards actually on disk, so newly GENERATED ones are picked up.

        The count used to come only from --cells. gen_handoff.sh generates fresh rollout shards
        when a cell runs dry, but a producer launched with `chat:6` kept reporting "nothing fresh
        left" after shards 6-8 appeared, and the trainer restarted into an empty buffer and
        exited. Counting the directory means a generation round is usable without restarting the
        producer.
        """
        n = len(list((root / f"rollouts_iolens/{self.mode}").glob("rollouts_*.safetensors")))
        return max(n, self.n_shards)


def ledger_done(buffer_dir: Path, cell: "Cell") -> set[tuple[int, float]]:
    """(shard, skip) slices recorded in the buffer's capture ledger.

    The ledger is appended by the capture process itself, so it is the authoritative record of
    what exists on disk — the cursor is only this loop's bookkeeping and cannot know about
    captures made outside it (e.g. the pre-producer manual wave, which is exactly how chat
    shards 0-2 and pt shard 0 came to be captured twice on 2026-08-02).
    """
    path = buffer_dir / "capture_ledger.json"
    if not path.exists():
        return set()
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError:  # mid-write; the cursor still covers this pass
        return set()
    return {(int(e["rollout_shard"]), round(float(e["lo"]), 4)) for e in entries}


def next_slice(
    cursor: dict[str, list[list[float]]], cell: Cell, buffer_dir: Path, n_shards: int | None = None
) -> tuple[int, float] | None:
    """The next (rollout_shard, skip) never captured for this cell, or None if exhausted.

    Returns None rather than raising: this used to be a ``SystemExit``, which propagates out of
    ``main()`` and kills the WHOLE producer — so the first cell to run out would also cut off the
    other cell, which may have far more seeds left (at time of writing chat is ~3 h from
    exhaustion with 6 of 13 rollout shards generated, while pt has 43 of 52 still ungenerated).
    One cell finishing its captured rollouts is a normal end state, not a fatal error.
    """
    done = {(int(s), round(float(k), 4)) for s, k in cursor.get(cell.mode, [])}
    done |= ledger_done(buffer_dir, cell)
    max_waves = max(1, int(1.0 / cell.frac))
    total = cell.n_shards if n_shards is None else n_shards
    for wave in range(max_waves):
        skip = round(wave * cell.frac, 4)
        for shard in range(total):
            if (shard, skip) not in done:
                return shard, skip
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="chat:6:0.14,pt:9:0.10",
                    help="csv of mode:n_rollout_shards:train_frac")
    ap.add_argument("--worlds", default="chat:3,pt:2", help="csv of mode:trainer_world_size")
    ap.add_argument("--low-gb", type=float, default=120.0, help="capture when a buffer is below")
    ap.add_argument("--high-gb", type=float, default=260.0, help="never exceed (per cell)")
    ap.add_argument("--min-free-gb", type=float, default=80.0, help="hard disk floor")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--cursor", default="produce_cursor.json")
    ap.add_argument("--once", action="store_true", help="one capture then exit (smoke test)")
    args = ap.parse_args()

    import shutil

    from oracle_lens.pipeline.stream_pairs import buffer_bytes, sweep_consumed

    root = ola_root()
    worlds = {k: int(v) for k, v in (x.split(":") for x in args.worlds.split(","))}
    cells = []
    for spec in args.cells.split(","):
        mode, n_shards, frac = spec.split(":")
        cells.append(Cell(mode, int(n_shards), float(frac), worlds.get(mode, 1)))
    cursor_path = root / args.cursor
    capture = Path(__file__).with_name("iolens_capture_pairs.py")

    while True:
        cursor: dict[str, list[list[float]]] = (
            json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
        )
        acted = False
        # Serve the NEEDIEST cell first, not the list order. One capture takes ~25 min and this
        # loop is serial, so whichever cell is checked first holds the GPU for that long. With a
        # fixed order chat always won, and pt drained to ZERO unconsumed shards — its trainer
        # blocked mid-run — while a chat capture ran with chat's buffer already at 140 GiB, well
        # above the low-water mark. Ordering by current buffer size costs nothing and makes
        # starving the hungrier cell impossible.
        for cell in sorted(cells, key=lambda c: buffer_bytes(root / c.buffer)):
            bdir = root / cell.buffer
            bdir.mkdir(parents=True, exist_ok=True)
            freed = sweep_consumed(bdir, cell.world)
            if freed:
                print(f"[produce] {cell.mode}: janitor freed {freed / 2**30:.1f} GiB", flush=True)
            gb = buffer_bytes(bdir) / 2**30
            free_gb = shutil.disk_usage(bdir).free / 2**30
            print(f"[produce] {cell.mode}: buffer {gb:.0f} GiB, disk free {free_gb:.0f} GiB",
                  flush=True)
            if gb >= args.low_gb or free_gb <= args.min_free_gb:
                continue
            avail = cell.shards_available(root)
            nxt = next_slice(cursor, cell, bdir, avail)
            if nxt is None:
                # Keep the exact wording: gen_handoff.sh greps for it to know this cell's GPUs
                # can be repurposed to rollout generation.
                print(f"[produce] {cell.mode}: whole ar_train split captured across "
                      f"{avail} rollout shards — nothing fresh left "
                      f"(other cells continue)", flush=True)
                continue
            shard, skip = nxt
            print(f"[produce] {cell.mode}: capturing shard {shard} skip {skip} "
                  f"(buffer {gb:.0f} < {args.low_gb:.0f} GiB)", flush=True)
            cmd = [
                sys.executable, str(capture), "--mode", cell.mode,
                "--rollout-shard", str(shard), "--out-dir", cell.buffer,
                "--train-frac", str(cell.frac), "--train-frac-skip", str(skip),
                "--max-eval-convs", "0",  # eval pairs were captured once, at wave 0
            ]
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"[produce] {cell.mode}: capture rc={rc} — retrying next poll", flush=True)
                continue
            cursor.setdefault(cell.mode, []).append([shard, skip])
            tmp = cursor_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cursor, indent=2))
            tmp.replace(cursor_path)
            acted = True
            if args.once:
                return
            # ONE capture per pass, then re-evaluate from the top. Without this the loop serves
            # every cell in the same pass: it correctly picked the neediest cell, captured it for
            # ~25 min, then immediately captured the OTHER cell for another ~25 min — and the
            # first cell drained to zero again in the meantime. Sorting by need only helps if the
            # need is re-measured after each capture, since a capture is long enough for the
            # ranking to invert.
            break
        if not acted:
            time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
