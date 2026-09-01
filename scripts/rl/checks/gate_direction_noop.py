"""Gate 4 — full-loop direction + no-op probes (port of check_08/09, toy reward).

Runs train_rl_ao.py as a subprocess twice on the gate parquet:
  direction (--toy-reward token_x): reward = fraction of response tokens equal
    to --toy-token-id. Over --steps steps the mean reward must RISE
    (last > first) — the optimizer moves the policy the way the reward points.
  no-op (--toy-reward constant): all-equal group rewards ⇒ zero advantages, and
    default ≡ reference at init ⇒ exact-KL term is identically zero ⇒
    grad_norm must be ≈ 0 EVERY step.
Both: step-0 KL ≈ 0 (fresh start loads the same adapter twice).

1 GPU (toy mode never loads the AR).

Run: uv run python scripts/rl/checks/gate_direction_noop.py \
       --parquet $SC/rl/rl_gate_0.parquet --sidecar $SC/rl/merged/nla_meta.yaml \
       --ao-lora $SC/hf/ckpts/ao/chat/k4.L20plus.s2/step3002/lora_hf
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from _sc_lib import verdict

_STEP_RE = re.compile(
    r"step (\d+) \| r (-?[\d.]+) .*?\| kl ([\d.eE+-]+) \| ent [\d.]+ \| ext \d+% "
    r"\| gnorm ([\d.eE+-]+)")


def run_trainer(args, mode, token_id, save_dir):
    cmd = [
        sys.executable, str(Path(__file__).parents[1] / "train_rl_ao.py"),
        "--ao-lora", args.ao_lora,
        "--base-ckpt", args.base_ckpt,
        "--ar-ckpt", "/dev/null", "--whitener-dir", "/dev/null",  # toy mode: unused
        "--rl-parquet", args.parquet,
        "--sidecar", args.sidecar,
        "--save-dir", save_dir,
        "--actor-device", args.device,
        "--toy-reward", mode, "--toy-token-id", str(token_id),
        "--num-steps", str(args.steps),
        "--batch-prompts", "2", "--group-size", "8",
        "--max-new-tokens", "24",
        "--lr", str(args.lr),
        "--logp-micro-batch", "4",
        "--eval-every", "0", "--save-every", "100000",
        "--no-wandb", "--seed", "0",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(out.stdout[-2000:])
    if out.returncode != 0:
        print(out.stderr[-3000:])
        raise SystemExit(f"trainer ({mode}) failed rc={out.returncode}")
    steps = []
    for line in out.stdout.splitlines():
        m = _STEP_RE.search(line)
        if m:
            steps.append({"step": int(m.group(1)), "r": float(m.group(2)),
                          "kl": float(m.group(3)), "gnorm": float(m.group(4))})
    assert steps, f"no step lines parsed from {mode} run"
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--ao-lora", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4, help="hot lr so 6 toy steps move")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    ids = tok.encode(" the", add_special_tokens=False)
    token_id = ids[0]

    with tempfile.TemporaryDirectory() as td:
        dir_steps = run_trainer(args, "token_x", token_id, td + "/dir")
        noop_steps = run_trainer(args, "constant", token_id, td + "/noop")

    r_first, r_last = dir_steps[0]["r"], dir_steps[-1]["r"]
    direction_ok = r_last > r_first
    noop_max_gnorm = max(s["gnorm"] for s in noop_steps)
    noop_ok = noop_max_gnorm < 1e-3
    kl0_ok = dir_steps[0]["kl"] < 1e-4 and noop_steps[0]["kl"] < 1e-4
    ok = direction_ok and noop_ok and kl0_ok
    return verdict("gate_direction_noop", ok, {
        "direction": {"r_first": r_first, "r_last": r_last, "rising": direction_ok,
                      "trace": [s["r"] for s in dir_steps]},
        "noop": {"max_gnorm": noop_max_gnorm, "ok": noop_ok,
                 "trace": [s["gnorm"] for s in noop_steps]},
        "step0_kl": {"direction": dir_steps[0]["kl"], "noop": noop_steps[0]["kl"],
                     "ok": kl0_ok},
        "toy_token_id": token_id,
    })


if __name__ == "__main__":
    raise SystemExit(main())
