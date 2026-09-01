"""iolens S4: standalone SGLang rollout worker — one GPU, one seed-shard stride, no repo imports.

Runs in the ``sglang-olens`` venv (which has no ``global_workspace`` install), so everything it
needs is inline; ``tests/test_iolens_gen_constants.py`` asserts the inline seed-hash / shard
layout stay in sync with ``ola.rollout_store``. Flow per worker:

1. launch a stock ``sglang.launch_server`` on this GPU (no NLA patch needed — plain generation),
2. read the seed shards, take stride ``rows[shard::n_shards]`` (deterministic, balanced),
3. per --chunk seeds: POST one batched ``/generate`` with ``input_ids`` (chat template applied
   here for chat mode; raw prefix ids for pt), ``return_logprob`` on so the response carries the
   EXACT output token ids (never re-tokenized text),
4. write an atomic chunk part file (preemption/crash loses ≤1 chunk; done chunks are skipped),
5. when all chunks exist: pack the shard to ``rollouts_{shard:04d}.safetensors`` (rollout_store
   layout), write ``reports/rollouts_{shard:04d}.json`` with EXACT counts + length percentiles +
   degeneracy stats (G4/G5), delete the parts.

Degenerate outputs (< --min-out tokens = empty/EOS-only, or an immediately-repeating 20-gram
loop) are DROPPED and counted exactly; the report fails the shard (``"g5_pass": false``) above
--degen-fail-rate. Legitimate short answers are DATA (the short-span octaves) — don't filter
them (the 2026-07-31 pilot measured ~9% short-legit at a min-out of 16, zero true loops).

    CUDA_VISIBLE_DEVICES=0 <your-sglang-venv>/bin/python \
        scripts/datagen/iolens_rollout_gen.py --mode chat --shard 0 --n-shards 4 --port 30100
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
# Inline constants (parity-tested against oracle_lens.pipeline.rollout_store):
SPLITS = ("ar_train", "ao_train", "ao_val", "eval")


def seed_hash64(seed_key: str) -> int:
    h = hashlib.blake2b(seed_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=True)


def ola_root() -> Path:
    root = os.environ.get("OLA_ROOT")
    if not root:
        raise SystemExit("OLA_ROOT is unset — `source scripts/cluster/env.sh` first")
    return Path(root)


def tokenizer_sha(tok: Any) -> str:
    """Venv-independent tokenizer identity: sha256 over the sorted vocab (G1)."""
    blob = json.dumps(tok.get_vocab(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def launch_server(
    port: int, mem_frac: float, seed: int, log_path: Path, max_running: int = 0,
    extra_args: str = "",
) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_ID,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--mem-fraction-static", str(mem_frac),
        "--random-seed", str(seed),
    ]
    if max_running:
        cmd += ["--max-running-requests", str(max_running)]
    if extra_args:
        cmd += extra_args.split()
    # The serving env is exec'd unactivated; sglang JIT-compiles kernels and shells out to
    # `ninja`, which lives in the venv's bin — prepend it or the server dies on first request.
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    with open(log_path, "ab") as log:
        # own process group: sglang spawns scheduler/worker children that outlive a plain
        # terminate() of the launcher (observed: a 119 GiB orphan squatting the GPU after a
        # crash) — killpg on the group is the only reliable teardown.
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, start_new_session=True)
    deadline = time.time() + 1800
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"[iolens-gen] server died during startup — see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"[iolens-gen] server healthy on :{port}", flush=True)
                    return proc
        except Exception:
            time.sleep(5)
    raise SystemExit("[iolens-gen] server did not become healthy in 30 min")


def generate_batch(
    port: int, batch_input_ids: list[list[int]], *, temperature: float, top_p: float,
    max_new: int, with_logprobs: bool = True,
) -> list[tuple[list[int], list[float]]]:
    """One batched /generate; returns (EXACT output token ids, their engine logprobs) per row.

    With logprobs: ids come from ``output_token_logprobs`` (stored for gate G3's engine-vs-HF
    parity). Without: ids come from the response's ``output_ids`` (same exactness, no logprob
    compute) and the logprob list is empty."""
    body = json.dumps(
        {
            "input_ids": batch_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new,
            },
            "return_logprob": with_logprobs,
            "logprob_start_len": -1,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.loads(r.read())
    rows = out if isinstance(out, list) else [out]
    result = []
    for row in rows:
        if with_logprobs:
            triples = row["meta_info"]["output_token_logprobs"]  # [logprob, token_id, ...]
            result.append(([int(t[1]) for t in triples], [float(t[0]) for t in triples]))
        else:
            result.append(([int(t) for t in row["output_ids"]], []))
    return result


def is_degenerate(ids: list[int], min_out: int) -> bool:
    if len(ids) < min_out:
        return True
    return any(ids[i : i + 20] == ids[i + 20 : i + 40] for i in range(len(ids) - 40))


def pack_shard(
    out_path: Path, part_paths: list[Path], meta_base: dict[str, Any], max_new: int,
    degen_fail_rate: float, report_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    rows: list[dict[str, Any]] = []
    for p in part_paths:
        rows.extend(json.loads(p.read_text()))
    kept = [r for r in rows if not r["degen"]]
    n_degen = len(rows) - len(kept)
    flat: list[int] = []
    offsets = [0]
    prompt_lens, hashes, split_ids = [], [], []
    out_lens = []
    out_logprobs: list[float] = []
    for r in kept:
        flat.extend(r["prompt_ids"])
        flat.extend(r["output_ids"])
        offsets.append(len(flat))
        prompt_lens.append(len(r["prompt_ids"]))
        hashes.append(seed_hash64(r["key"]))
        split_ids.append(r["split"])
        out_lens.append(len(r["output_ids"]))
        out_logprobs.extend(r["output_logprobs"])
    n_prompt = sum(prompt_lens)
    n_out = len(flat) - n_prompt
    meta = dict(
        meta_base, n_convs=len(kept), n_prompt_tokens=n_prompt, n_output_tokens=n_out
    )
    tensors = {
        "ids": torch.tensor(flat, dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "prompt_len": torch.tensor(prompt_lens, dtype=torch.int32),
        "seed_hash": torch.tensor(hashes, dtype=torch.int64),
        "split_id": torch.tensor(split_ids, dtype=torch.int8),
        # engine logprobs of the output tokens (ragged by output length, conv order) — G3 input
        "out_logprob": torch.tensor(out_logprobs, dtype=torch.float16),
    }
    tmp = out_path.with_suffix(f".tmp.{os.getpid()}")
    save_file(tensors, str(tmp), metadata={"meta": json.dumps(meta)})
    tmp.replace(out_path)

    sl = sorted(out_lens) or [0]

    def pct(q: float) -> int:
        return int(sl[min(len(sl) - 1, int(q * len(sl)))])
    n_trunc = sum(1 for x in out_lens if x >= max_new)
    degen_rate = n_degen / max(1, len(rows))
    report = {
        "shard": out_path.name,
        "n_generated": len(rows),
        "n_kept": len(kept),
        "n_degenerate_dropped": n_degen,
        "degen_rate": round(degen_rate, 5),
        "g5_pass": degen_rate <= degen_fail_rate,
        "n_prompt_tokens": n_prompt,
        "n_output_tokens": n_out,
        "out_len_p50": pct(0.50),
        "out_len_p90": pct(0.90),
        "out_len_p99": pct(0.99),
        "truncated_at_max_new": n_trunc,
        "trunc_rate": round(n_trunc / max(1, len(out_lens)), 5),
        "split_counts": {
            SPLITS[s]: int(sum(1 for x in split_ids if x == s)) for s in range(len(SPLITS))
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rtmp = report_path.with_suffix(f".tmp.{os.getpid()}")
    rtmp.write_text(json.dumps(report, indent=2))
    rtmp.replace(report_path)
    # keep a decodable sample of the DROPPED rows — G5 calibration needs eyes on them
    degen_rows = [r for r in rows if r["degen"]][:20]
    if degen_rows:
        (report_path.parent / f"degen_sample_{out_path.stem[-4:]}.json").write_text(
            json.dumps(
                [{"key": r["key"], "output_ids": r["output_ids"]} for r in degen_rows]
            )
        )
    for p in part_paths:
        p.unlink()
    print(f"[iolens-gen] packed {out_path.name}: {json.dumps(report)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["chat", "pt"])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--seeds-dir", default="", help="default seeds_iolens_<mode>")
    ap.add_argument("--out-dir", default="", help="default rollouts_iolens/<mode>")
    ap.add_argument("--port", type=int, default=0, help="0 = 30100 + first visible GPU")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--sub-batch", type=int, default=512, help="rows per /generate POST")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument(
        "--min-out", type=int, default=2,
        help="drop outputs shorter than this. 2 = only empty/EOS-only rows (no span material)."
        " Calibrated 2026-07-31: at 16 the filter ate ~9%% of rollouts, ALL of them legitimate"
        " short answers (greetings/acks) — i.e. the short-span octaves, not degeneration.",
    )
    ap.add_argument("--degen-fail-rate", type=float, default=0.03)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--max-running-requests", type=int, default=0,
                    help="0 = sglang default; the S0 bench picks the knee")
    ap.add_argument("--no-logprobs", action="store_true",
                    help="skip engine logprobs (fleet speed knob; pilots keep them for G3)")
    ap.add_argument("--extra-server-args", default="",
                    help="extra sglang.launch_server args, e.g. '--disable-radix-cache' "
                    "(re-enables the overlap scheduler under mamba no_buffer; our seeds share "
                    "no prefixes, so the radix cache buys nothing)")
    ap.add_argument("--max-seeds", type=int, default=0, help="cap this worker's stride (pilot)")
    ap.add_argument("--no-server", action="store_true", help="server already running on --port")
    args = ap.parse_args()

    root = ola_root()
    seeds_dir = root / (args.seeds_dir or f"seeds_iolens_{args.mode}")
    out_dir = root / (args.out_dir or f"rollouts_iolens/{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rollouts_{args.shard:04d}.safetensors"
    report_path = out_dir / "reports" / f"rollouts_{args.shard:04d}.json"
    if out_path.exists():
        print(f"[iolens-gen] {out_path} exists — skipping (idempotent)", flush=True)
        return

    seed_rows: list[dict[str, Any]] = []
    for p in sorted(seeds_dir.glob("seeds_*.json")):
        seed_rows.extend(json.loads(p.read_text()))
    if not seed_rows:
        raise SystemExit(f"[iolens-gen] no seeds under {seeds_dir}")
    mine = seed_rows[args.shard :: args.n_shards]
    if args.max_seeds:
        mine = mine[: args.max_seeds]
    print(f"[iolens-gen] mode={args.mode} shard {args.shard}/{args.n_shards}: "
          f"{len(mine)} seeds", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tsha = tokenizer_sha(tok)

    first_gpu = int((os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]) or 0)
    port = args.port or (30100 + first_gpu)
    proc = None
    if not args.no_server:
        log_path = out_dir / "reports" / f"server_{args.shard:04d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc = launch_server(port, args.mem_frac, seed=args.shard, log_path=log_path,
                             max_running=args.max_running_requests,
                             extra_args=args.extra_server_args)

    try:
        import importlib.metadata

        engine_version = importlib.metadata.version("sglang")
        meta_base = {
            "model_id": MODEL_ID,
            "mode": args.mode,
            "engine": "sglang",
            "engine_version": engine_version,
            "tokenizer_sha": tsha,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new": args.max_new,
            "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        }
        part_paths = []
        t_start = time.time()
        done_tokens = 0
        for c0 in range(0, len(mine), args.chunk):
            chunk_rows = mine[c0 : c0 + args.chunk]
            part = out_dir / f"rollouts_{args.shard:04d}.part{c0 // args.chunk:04d}.json"
            part_paths.append(part)
            if part.exists():
                continue
            batch_inputs: list[list[int]] = []
            for r in chunk_rows:
                if args.mode == "chat":
                    ids = tok.apply_chat_template(
                        [{"role": "user", "content": r["text"]}],
                        add_generation_prompt=True,
                        enable_thinking=False,
                        tokenize=True,
                        return_dict=False,  # transformers 5.x defaults to a dict here
                    )
                else:
                    ids = list(r["prefix_ids"])
                batch_inputs.append([int(t) for t in list(ids)])
            outs: list[tuple[list[int], list[float]]] = []
            for b0 in range(0, len(batch_inputs), args.sub_batch):
                outs.extend(
                    generate_batch(
                        port, batch_inputs[b0 : b0 + args.sub_batch],
                        temperature=args.temperature, top_p=args.top_p, max_new=args.max_new,
                        with_logprobs=not args.no_logprobs,
                    )
                )
            rows_out = []
            for r, pids, (oids, olps) in zip(chunk_rows, batch_inputs, outs, strict=True):
                rows_out.append(
                    {
                        "key": r["key"],
                        "split": r["split"],
                        "prompt_ids": pids,
                        "output_ids": oids,
                        "output_logprobs": olps,
                        "degen": is_degenerate(oids, args.min_out),
                    }
                )
                done_tokens += len(oids)
            tmp = part.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(rows_out))
            tmp.replace(part)
            rate = done_tokens / max(1.0, time.time() - t_start)
            print(
                f"[iolens-gen] chunk {c0 // args.chunk}: {min(c0 + args.chunk, len(mine))}/"
                f"{len(mine)} seeds, {done_tokens:,} gen tokens this run, {rate:,.0f} tok/s",
                flush=True,
            )
        pack_shard(out_path, part_paths, meta_base, args.max_new, args.degen_fail_rate,
                   report_path)
    finally:
        if proc is not None:
            import signal

            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
            import contextlib

            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
