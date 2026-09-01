"""RAFT-2 SFT (Path B): supervised fine-tune the AO on its own best-of-16 readouts.

Reuses the VALIDATED train_rl_ao machinery (base + LoRA actor, embedding-replacement
injection hook, the exact unit-alpha16000 contract). For each harvested item we
inject the activation at the ㈜ marker and minimize cross-entropy on the WINNING
readout (best_rollout_text, joint-FVE-selected during harvest). Whole-readout target
=> the 4-bullet format is preserved by construction (no per-bullet reward to game).

Warm-starts the LoRA from iter_000600 (continue training the RL policy toward its
own best samples). Reward-hack guard drops content-free / too-short / <min-bullets
winners so RAFT doesn't distill the bug-mutate-style degenerate high-FVE outputs.

  python raft_sft.py --run-dir <ddp600> --harvest-glob 'raft2_harvest/harvest_*.json' \
      --init-lora <ddp600>/iter_000600 --out <dir> --epochs 2 --device cuda:0
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from data import load_ao_rl_dataset  # noqa: E402
from hooks import register_embed_injection_hook  # noqa: E402
from sidecar import load_sidecar  # noqa: E402

DEGEN = re.compile(r"^\s*-?\s*(this function (performs|is|'s)|option [ab]\b|[ab])\s*$", re.I)


def split_bullets(t):
    return [b.strip() for b in re.split(r"\n?\s*-\s+", t or "") if b.strip()]


def guard_ok(text, min_bullets=3, min_chars=40):
    b = split_bullets(text)
    if len(b) < min_bullets or len(text.strip()) < min_chars:
        return False
    uniq = {x.lower() for x in b}
    if len(uniq) < len(b):          # duplicate bullets
        return False
    if any(DEGEN.match(x) for x in b):   # content-free stems
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--harvest-glob", required=True)
    p.add_argument("--init-lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parquet", default="artifacts/sc/rl_distill/rl_train_0.parquet")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no-guard", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load((Path(args.run_dir) / "run_config.yaml").read_text())
    rargs = SimpleNamespace(**cfg)
    dev = args.device

    # ---- harvested winners: row_id -> best readout (guarded) ----
    winners = {}
    for f in glob.glob(args.harvest_glob):
        for rec in json.loads(Path(f).read_text()).get("records", []):
            txt = rec.get("best_rollout_text", "")
            if args.no_guard or guard_ok(txt):
                winners[int(rec["row_id"])] = txt
    print(f"[raft-sft] {len(winners)} guarded winners from {args.harvest_glob}", flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rargs.base_ckpt)
    scfg = load_sidecar(rargs.sidecar, tok)
    inj_id = scfg.injection_token_id

    rows = load_ao_rl_dataset(args.parquet, inj_id=inj_id, n_max=None)
    items = [(r, winners[r["row_id"]]) for r in rows if r["row_id"] in winners]
    print(f"[raft-sft] {len(items)} training items (joined to vecs)", flush=True)

    base = AutoModelForCausalLM.from_pretrained(rargs.base_ckpt, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(dev)
    actor = PeftModel.from_pretrained(base, args.init_lora, is_trainable=True)
    actor.train()
    vectors_ref = [None]
    register_embed_injection_hook(actor, vectors_ref, inj_id,
                                  scfg.left_neighbor_id, scfg.right_neighbor_id)
    opt = torch.optim.AdamW([p_ for p_ in actor.parameters() if p_.requires_grad], lr=args.lr)
    pad = tok.eos_token_id

    def batches():
        idx = np.random.RandomState(0).permutation(len(items))
        for i in range(0, len(idx), args.batch):
            yield [items[j] for j in idx[i:i + args.batch]]

    step = 0
    for ep in range(args.epochs):
        for batch in batches():
            # build input = prompt_ids + target_ids; label-mask the prompt
            seqs, labels, vecs = [], [], []
            for r, tgt in batch:
                pids = r["prompt_ids"]
                tids = tok(tgt, add_special_tokens=False).input_ids + [tok.eos_token_id]
                seqs.append(pids + tids)
                labels.append([-100] * len(pids) + tids)
                vecs.append(torch.from_numpy(r["inject"]).float())
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            lab = torch.full((len(seqs), L), -100, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for k, (s, lb) in enumerate(zip(seqs, labels)):
                ids[k, :len(s)] = torch.tensor(s); lab[k, :len(lb)] = torch.tensor(lb)
                att[k, :len(s)] = 1
            vectors_ref[0] = torch.stack(vecs).to(dev)
            try:
                out = actor(input_ids=ids.to(dev), attention_mask=att.to(dev),
                            labels=lab.to(dev))
            finally:
                vectors_ref[0] = None
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p_ for p_ in actor.parameters() if p_.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            if step % 20 == 0:
                print(f"[raft-sft] ep{ep} step{step} loss {out.loss.item():.4f}", flush=True)
            step += 1
    Path(args.out).mkdir(parents=True, exist_ok=True)
    actor.save_pretrained(args.out)
    print(f"[raft-sft] saved LoRA -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
