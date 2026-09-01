"""Gate 1 — injection triple + hook≡splice equivalence (port of check_03, engine-free).

Asserts, on the real actor through the trainer's embedding-replacement hook:
  triple (mechanism causality, greedy):
    repeat_identical     same vector twice        -> identical continuations
    different_acts_differ  two different vectors  -> different continuations
    zero_vec_differs     zero vector vs real      -> different continuations
  hook ≡ splice: hook-generate token-for-token equal to hf_greedy over
    spliced_embeds (the reference arm the old stack gated sglang against) on
    --n gate rows. Both arms are in-process bf16 -> exact match expected.

Run: uv run python scripts/rl/checks/gate_injection.py \
       --parquet $SC/rl/rl_gate_0.parquet --sidecar $SC/rl/merged/nla_meta.yaml \
       --ao-lora $SC/hf/ckpts/ao/chat/k4.L20plus.s2/step3002/lora_hf
"""

import argparse

import numpy as np
import torch

# _sc_lib FIRST: importing it inserts scripts/rl/checks on sys.path, which is the only
# thing that makes `_lib` importable when this gate runs as a bare script (run_gates.sh).
from _sc_lib import hook_greedy, load_actor, load_gate_rows, verdict  # isort: skip
from _lib import hf_greedy, spliced_embeds  # isort: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--ao-lora", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    actor, tok, _cfg, vectors_ref, eos_ids = load_actor(
        args.base_ckpt, args.ao_lora, args.sidecar, args.device)
    pad_id = tok.eos_token_id
    rows = load_gate_rows(args.parquet, args.n)
    assert len(rows) >= 2, "need >= 2 gate rows"

    def gen(prompt_ids, vec):
        return hook_greedy(actor, prompt_ids, vec, vectors_ref, eos_ids, pad_id,
                           args.max_new, args.device)

    r0, r1 = rows[0], rows[1]
    v0 = torch.tensor(np.asarray(r0["activation_vector"], dtype=np.float32))
    v1 = torch.tensor(np.asarray(r1["activation_vector"], dtype=np.float32))

    a = gen(r0["prompt_ids"], v0)
    b = gen(r0["prompt_ids"], v0)
    c = gen(r0["prompt_ids"], v1)
    z = gen(r0["prompt_ids"], torch.zeros_like(v0))
    triple = {
        "repeat_identical": a == b,
        "different_acts_differ": a != c,
        "zero_vec_differs": a != z,
    }

    # hook ≡ splice on n rows
    mismatches = []
    for r in rows:
        v = torch.tensor(np.asarray(r["activation_vector"], dtype=np.float32))
        hook_ids = gen(r["prompt_ids"], v)
        e = spliced_embeds(actor, r["prompt_ids"], int(r["slot"]), v)
        splice_ids = hf_greedy(actor, tok, e, args.max_new)
        # hf_greedy returns only NEW tokens when fed inputs_embeds; trim both at
        # the first stop id for a fair compare (splice arm has no marker-stop).
        n_real = next((i + 1 for i, t in enumerate(splice_ids) if t in eos_ids),
                      len(splice_ids))
        splice_ids = splice_ids[:n_real]
        if hook_ids != splice_ids:
            mismatches.append({
                "row_id": r["row_id"],
                "hook": tok.decode(hook_ids)[:80],
                "splice": tok.decode(splice_ids)[:80],
            })

    ok = all(triple.values()) and not mismatches
    return verdict("gate_injection", ok, {
        **triple,
        "n_rows": len(rows),
        "hook_vs_splice_mismatches": mismatches,
        "sample_text": tok.decode(a)[:120],
    })


if __name__ == "__main__":
    raise SystemExit(main())
