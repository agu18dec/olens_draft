"""Shared helpers for the GPU checks: server lifecycle, splice, HF reference.

Runs in the miles-rl venv. Every check takes --model-dir (+ optional --sidecar/--parquet from
mk_toy_assets.py), so the SAME scripts rerun on the 27B merged checkpoint unchanged.
Server pattern mirrors scripts/oracle_lens_evals/olens_sglang/worker.py (start_new_session so
kill_server can killpg the scheduler/detokenizer children; readiness on /get_model_info).
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_snapshot(model: str) -> str:
    """A model id or a local dir -> a local dir sglang/HF can load offline."""
    p = Path(model)
    if p.is_dir():
        return str(p)
    cache_name = "models--" + model.replace("/", "--")
    for cache in (Path(os.environ.get("HF_HOME", "")) / "hub",):
        snaps = cache / cache_name / "snapshots"
        if snaps.is_dir():
            newest = max(snaps.iterdir(), key=lambda d: d.stat().st_mtime)
            return str(newest)
    return model  # let HF resolve (may download)


def launch_server(
    model_dir: str,
    *,
    port: int = 30001,
    mem_fraction: float = 0.5,
    radix: bool = False,
    log_path: str = "/tmp/sglang_check.log",
    context_length: int = 2048,
) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_dir,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--mem-fraction-static", str(mem_fraction),
        "--context-length", str(context_length),
        # gate servers value robustness over throughput: the piecewise-CUDA-graph warmup
        # hits cudaErrorIllegalAddress on small dense Qwen3 at this pin (observed 1.7B,
        # node-28, 2026-08-08; the 27B qwen3_5 eval worker serves fine WITH graphs)
        "--disable-cuda-graph",
    ]
    if not radix:
        cmd.append("--disable-radix-cache")
    # the scheduler JIT shells out to `ninja`; the venv python runs unactivated, so its
    # bin/ must be prepended to PATH (setup_sglang_env.sh lesson)
    venv_bin = str(Path(sys.executable).parent)
    env = {**os.environ, "SGL_ENABLE_JIT_DEEPGEMM": "false",
           "PATH": venv_bin + os.pathsep + os.environ.get("PATH", "")}
    log = open(log_path, "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True, env=env)
    deadline = time.monotonic() + 1200  # first-ever launch JIT-compiles kernels
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = Path(log_path).read_text(errors="replace").splitlines()[-25:]
            raise RuntimeError("sglang server died at startup:\n" + "\n".join(tail))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/get_model_info", timeout=3):
                return proc
        except Exception:
            time.sleep(2.0)
    raise RuntimeError("sglang server not ready within deadline")


def kill_server(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    time.sleep(2.0)  # CUDA settle before the next server


def post_generate(port: int, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out: dict[str, Any] = json.loads(r.read())
    return out


def greedy_params(max_new: int) -> dict[str, Any]:
    return {"temperature": 0.0, "max_new_tokens": max_new, "top_p": 1.0, "top_k": 1}


def spliced_embeds(
    model: Any, ids: list[int], slot: int, vec: torch.Tensor | None
) -> torch.Tensor:
    """[1, T, d] fp32 input embeddings with vec REPLACING the slot row (None = no injection)."""
    embed = model.get_input_embeddings()
    with torch.no_grad():
        e = embed(torch.tensor([ids], device=embed.weight.device)).float()
        if vec is not None:
            e[0, slot] = vec.to(e.device, torch.float32)
    return e  # type: ignore[no-any-return]


def hf_greedy(
    model: Any, tokenizer: Any, e: torch.Tensor, max_new: int
) -> list[int]:
    """Greedy continuation ids from inputs_embeds (the reference arm of the splice gate)."""
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=e.to(model.dtype),
            attention_mask=torch.ones(e.shape[:2], dtype=torch.long, device=e.device),
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            pad_token_id=int(tokenizer.pad_token_id or 0),
        )
    return [int(x) for x in out[0]]


def load_toy(parquet: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    t = pq.read_table(parquet)  # type: ignore[no-untyped-call]
    cols = t.column_names
    for i in range(t.num_rows):
        rows.append({c: t.column(c)[i].as_py() for c in cols})
    return rows


def verdict(name: str, ok: bool, detail: dict[str, Any]) -> int:
    print(json.dumps(detail, indent=2, default=str))
    print(f"[{name}] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1
