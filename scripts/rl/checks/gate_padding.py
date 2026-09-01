"""Gate 3 — padding floor: right-padding adds no logp error beyond batch numerics.

Freezes one greedy rollout batch (variable response lengths), then compares
per-token policy logp through the trainer's selective-logprob path (injection
active) at micro_batch=1 vs micro_batch=8 — twice:

  control: all rows truncated to one length -> mb8 has ZERO padding. Any mb1-vs-
    mb8 delta here is pure batch-size-dependent kernel numerics (the GDN torch-
    fallback's chunked reductions re-order across B; measured ~0.1 nats max /
    ~0.008 mean on this stack, deterministic on repeat).
  padded: natural variable lengths -> mb8 right-pads to the chunk max.

PASS iff the padded case is statistically indistinguishable from the control
(max ≤ 1.5x control max, mean ≤ 2x control mean) AND the absolute mean stays
≤ 0.02 nats/token (the old stack's engine-axis floor at 27B was 0.44-0.55 MEAN;
this is trainer-vs-trainer so the bar is 20x+ tighter). A genuine padding leak
(pads reaching real tokens' state) would blow the ratio, not the baseline.
NB: no IS ratio exists in this trainer, so batch-numeric logp offsets cannot
bias the gradient the way they would on the engine axis.

Run: uv run --no-sync python scripts/rl/checks/gate_padding.py \
       --parquet $SC/rl/rl_gate_0.parquet --sidecar $SC/rl/merged/nla_meta.yaml \
       --ao-lora $SC/hf/ckpts/ao/chat/k4.L20plus.s2/step3002/lora_hf
"""

import argparse

import numpy as np
import torch
from _sc_lib import hook_greedy, load_actor, load_gate_rows, verdict


@torch.no_grad()
def selective_logp(actor, full_ids_list, prompt_lens, inject_vecs, vectors_ref,
                   pad_id, device, micro_batch):
    """Per-sample per-token logp via the trainer's update-path math (no grad)."""
    out = [None] * len(full_ids_list)
    for cs in range(0, len(full_ids_list), micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, len(full_ids_list))))
        bs = len(idxs)
        max_len = max(full_ids_list[i].numel() for i in idxs)
        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for row, i in enumerate(idxs):
            length = full_ids_list[i].numel()
            batch_ids[row, :length] = full_ids_list[i].to(device)
            attn[row, :length] = 1
        vectors_ref[0] = torch.stack(
            [inject_vecs[i].to(device).float() for i in idxs], dim=0)
        try:
            logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            vectors_ref[0] = None
        for row, i in enumerate(idxs):
            length = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            target_ids = batch_ids[row, p_len:length]
            pred_idx = torch.arange(p_len - 1, length - 1, device=device)
            resp_logits = logits[row].index_select(0, pred_idx).float()
            lse = torch.logsumexp(resp_logits, dim=-1)
            out[i] = (resp_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                      - lse).cpu()
        del logits
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--ao-lora", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--mean-floor", type=float, default=0.02,
                    help="absolute ceiling on padded mean |Δlogp| nats/token")
    ap.add_argument("--max-ratio", type=float, default=1.5,
                    help="padded max must be ≤ this x the same-length control max")
    ap.add_argument("--mean-ratio", type=float, default=2.0,
                    help="padded mean must be ≤ this x the control mean")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    actor, tok, _cfg, vectors_ref, eos_ids = load_actor(
        args.base_ckpt, args.ao_lora, args.sidecar, args.device)
    pad_id = tok.eos_token_id
    rows = load_gate_rows(args.parquet, args.n)

    full_ids_list, prompt_lens, inject_vecs = [], [], []
    for r in rows:
        v = torch.tensor(np.asarray(r["activation_vector"], dtype=np.float32))
        resp = hook_greedy(actor, r["prompt_ids"], v, vectors_ref, eos_ids, pad_id,
                           args.max_new, args.device)
        full_ids_list.append(torch.tensor(r["prompt_ids"] + resp, dtype=torch.long))
        prompt_lens.append(len(r["prompt_ids"]))
        inject_vecs.append(v)

    def compare(ids_list):
        lp_1 = selective_logp(actor, ids_list, prompt_lens, inject_vecs, vectors_ref,
                              pad_id, args.device, micro_batch=1)
        lp_m = selective_logp(actor, ids_list, prompt_lens, inject_vecs, vectors_ref,
                              pad_id, args.device, micro_batch=args.micro_batch)
        max_d = max(float((a - b).abs().max()) for a, b in zip(lp_1, lp_m, strict=True))
        mean_d = float(np.mean(
            [float((a - b).abs().mean()) for a, b in zip(lp_1, lp_m, strict=True)]))
        return max_d, mean_d

    # control: same-length rows -> zero padding; deltas = pure batch numerics
    minlen = min(f.numel() for f in full_ids_list)
    ctl_max, ctl_mean = compare([f[:minlen] for f in full_ids_list])
    # padded: natural variable lengths -> real right-padding exercised
    pad_max, pad_mean = compare(full_ids_list)

    resp_lens = [int(f.numel() - p) for f, p in zip(full_ids_list, prompt_lens, strict=True)]
    ok = (
        pad_mean <= args.mean_floor
        and pad_max <= max(args.max_ratio * ctl_max, 0.02)
        and pad_mean <= max(args.mean_ratio * ctl_mean, 0.005)
    )
    return verdict("gate_padding", ok, {
        "padded": {"max": round(pad_max, 6), "mean": round(pad_mean, 6)},
        "control_same_length": {"max": round(ctl_max, 6), "mean": round(ctl_mean, 6)},
        "mean_floor": args.mean_floor, "max_ratio": args.max_ratio,
        "micro_batch": args.micro_batch,
        "resp_lens": resp_lens,  # variable lengths => real padding exercised
    })


if __name__ == "__main__":
    raise SystemExit(main())
