"""Eval-set FVE ceiling checks — is 0.15 low because the eval set is hard?

Two references, both scored through the *identical* frozen-AR joint-FVE scorer the
trainer uses (imported build_reward_fn -> score_ar), on the SAME eval slice:

  --mode gt      GT-text ceiling. Feed each eval row's `source_text` (the crop whose
                 preceding residual IS the gold activation) through the scorer. No
                 generation, no actor. Answers: does the TRUE text even reconstruct
                 these golds well? If GT-FVE ~= RL-FVE, the golds are just hard and
                 RL is at ceiling; if GT-FVE >> RL-FVE, there is headroom.

  --mode oracle  Policy-relative upper bound. Sample --n-rollouts completions/prompt
                 from a checkpoint, pool ALL their bullets, nnomp-select the best
                 subset onto the gold. Answers: how high could the current policy
                 reach if we oracle-selected among its own samples?

Reads the run's snapshotted run_config.yaml so the scorer contract (ar_ckpt,
whiteners, whiten/unit_norm, k_max) matches the run byte-for-byte.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # scripts/rl on path (data, sidecar, ...)

from data import load_ao_rl_dataset  # noqa: E402
from sidecar import load_sidecar  # noqa: E402


def _load_args(run_dir: str, reward_device: str) -> SimpleNamespace:
    cfg = yaml.safe_load((Path(run_dir) / "run_config.yaml").read_text())
    cfg["reward_device"] = reward_device  # run used cuda:7; the check picks a free GPU
    cfg.setdefault("reward_loo_lambda", 0.0)  # older run_config.yaml predates this flag
    cfg.setdefault("reward_loo", False)
    cfg.setdefault("reward_agg", "single")        # contrastive-run keys (snapshots predate them)
    cfg.setdefault("reward_contrastive_beta", 1.0)
    cfg.setdefault("reward_n_distractors", 8)
    cfg.setdefault("reward_bank_cap", 128)
    cfg.setdefault("reward_k_max", 4)
    cfg.setdefault("diversity_lambda", 0.0)  # det diversity bonus (newer trainer flag)
    cfg.setdefault("dr_grpo", False)
    return SimpleNamespace(**cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="RL run dir (has run_config.yaml)")
    p.add_argument("--mode", choices=["gt", "oracle"], default="gt")
    p.add_argument("--reward-device", default="cuda:0", help="AR + nnomp device")
    p.add_argument("--actor-device", default="cuda:1", help="actor (oracle mode) device")
    p.add_argument("--n-eval", type=int, default=None, help="override eval_n_prompts")
    p.add_argument("--per-layer", type=int, default=None,
                   help="balanced draw: N rows per layer (overrides --n-eval)")
    p.add_argument("--parquet", default=None,
                   help="override the eval parquet (e.g. the diagnostic-probe bank)")
    p.add_argument("--out", default=None, help="write per-item json here")
    # oracle mode
    p.add_argument("--lora", default=None, help="checkpoint iter_* dir (oracle mode)")
    p.add_argument("--n-rollouts", type=int, default=16)
    p.add_argument("--oracle-k", type=int, default=8, help="nnomp select budget")
    p.add_argument("--n-shards", type=int, default=1, help="shard the row list (RAFT harvest)")
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()

    rargs = _load_args(args.run_dir, args.reward_device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(rargs.base_ckpt)
    cfg = load_sidecar(rargs.sidecar, tokenizer)
    inj_id = cfg.injection_token_id

    parquet = args.parquet or rargs.eval_parquet
    all_rows = load_ao_rl_dataset(parquet, inj_id=inj_id, n_max=None)
    if args.per_layer:
        by = {}
        for r in all_rows:
            by.setdefault(int(r["layer"]), []).append(r)
        eval_rows = []
        for ly in sorted(by):
            eval_rows.extend(by[ly][:args.per_layer])
        print(f"[data] balanced {args.per_layer}/layer -> {len(eval_rows)} rows "
              f"over {len(by)} layers from {rargs.eval_parquet}", flush=True)
    else:
        n_eval = args.n_eval or rargs.eval_n_prompts
        eval_rows = all_rows[:n_eval]
        print(f"[data] {len(eval_rows)} eval rows from {parquet}", flush=True)
    if args.n_shards > 1:  # RAFT harvest: split the row list across GPU pairs
        eval_rows = eval_rows[args.shard::args.n_shards]
        print(f"[shard] {args.shard}/{args.n_shards} -> {len(eval_rows)} rows", flush=True)

    from train_rl_ao import build_reward_fn
    score_fn, _floor = build_reward_fn(rargs, tokenizer)

    golds = torch.from_numpy(np.stack([r["gold"] for r in eval_rows])).float()
    layers = torch.tensor([r["layer"] for r in eval_rows], dtype=torch.long)

    if args.mode == "gt":
        texts = [(r.get("source_text") or "") for r in eval_rows]
        res = score_fn(texts, golds, layers, [[] for _ in eval_rows])
        fve = res.fve.float().numpy()
        valid = res.valid.numpy()
        _report("GT-text (source_text through AR)", fve, valid, layers.numpy(), eval_rows,
                texts, args.out)
        empties = sum(1 for t in texts if not t.strip())
        print(f"[gt] empty source_text rows: {empties}/{len(texts)}")
        print("[gt] note: source_text is a raw crop (typically no '- ' bullets) -> "
              "scored as a single readout (k=1 fallback). This is the single-true-text "
              "ceiling; the RL model may exceed it via multi-bullet refit.")
    else:
        _oracle(args, rargs, tokenizer, cfg, inj_id, eval_rows, golds, layers, score_fn)


def _ar_unit_images(recon, tokenizer, phrases, layer, whitener, space, device, mb=32):
    """AR-map `phrases` at `layer`, whiten (+unit-norm if space.unit_norm) -> [m,d] and
    the kept-phrase indices. Mirrors score_texts_joint's mapping EXACTLY."""
    from oracle_lens.pipeline.multilayer_reconstructor import ml_collate
    from oracle_lens.pipeline.rl_reward import capture_block_outputs, reward_text_ids
    pos = {int(ly): j for j, ly in enumerate(recon.layers)}
    emb_w = recon.layer_emb.weight
    id_rows = [reward_text_ids(tokenizer, ph) for ph in phrases]
    keep = [j for j, ids in enumerate(id_rows) if ids]
    out = {}
    if keep:
        order = sorted(keep, key=lambda j: len(id_rows[j]))
        with torch.no_grad():
            for lo in range(0, len(order), mb):
                sel = order[lo:lo + mb]
                batch = ml_collate(
                    [{"ids": torch.tensor(id_rows[j]), "target": torch.zeros(1)} for j in sel],
                    pad_id=0)
                ids_dev = batch["input_ids"].to(device)
                mask_dev = batch["attention_mask"].to(device)
                cap = capture_block_outputs(recon.backbone, [layer], ids_dev, mask_dev)
                last = mask_dev.sum(dim=1) - 1
                for row, j in enumerate(sel):
                    h_l = cap[layer][row, int(last[row])].float()
                    out[j] = recon.head(h_l + emb_w[pos[layer]]).float().cpu()
    imgs, kept = [], []
    for j in keep:
        pv = out.get(j)
        if pv is None:
            continue
        pv = whitener.whiten(pv.unsqueeze(0))[0] if space.whiten else (pv - whitener.mu)
        if space.unit_norm:
            pv = pv / (pv.norm() + 1e-9)
        imgs.append(pv)
        kept.append(j)
    return (torch.stack(imgs) if imgs else torch.zeros(0, emb_w.shape[1])), kept


def _oracle(args, rargs, tokenizer, cfg, inj_id, eval_rows, golds, layers, score_fn):
    """Sample n_rollouts/prompt from --lora, pool bullets, nnomp-select onto the gold."""
    import time

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from oracle_lens.core.nnomp import nnls_refit, nnomp_batch
    from oracle_lens.pipeline.ar_loader import load_ladder_whiteners, load_lc_reconstructor
    from oracle_lens.pipeline.rl_reward import RewardSpace, split_bullets

    sys.path.insert(0, str(_HERE.parent))
    from hooks import register_embed_injection_hook
    from train_rl_ao import rollout_prompts

    adev = args.actor_device            # actor (27B) on its own GPU
    rdev = args.reward_device           # AR + nnomp on a separate GPU (both are large)
    lora = args.lora or str(Path(args.run_dir) / "iter_000600")
    print(f"[oracle] policy LoRA = {lora}  actor@{adev} AR@{rdev}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(rargs.base_ckpt, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(adev)
    actor = PeftModel.from_pretrained(base, lora, adapter_name="default", is_trainable=False)
    actor.eval()
    vectors_ref = [None]
    register_embed_injection_hook(actor, vectors_ref, inj_id, cfg.left_neighbor_id,
                                  cfg.right_neighbor_id)
    eos_ids = {tokenizer.eos_token_id, inj_id}
    eos_ids.discard(None)
    pad_id = tokenizer.eos_token_id
    mnt = int(getattr(rargs, "max_new_tokens", 128))

    recon = load_lc_reconstructor(Path(rargs.ar_ckpt), device=rdev, eager=True)
    whiteners = load_ladder_whiteners(Path(rargs.whitener_dir), prefix=rargs.whitener_prefix,
                                      ridge_c=rargs.ridge, layers=tuple(recon.layers))
    space = RewardSpace(whiten=rargs.reward_whiten, unit_norm=rargs.reward_unit_norm)

    n_roll = args.n_rollouts
    greedy_fve, oracle_fve, full_fve, ncand, gt_fve, row_layer = [], [], [], [], [], []
    best1_fve, bestn_fve = [], []   # mean single-sample; max over N (each its own 4 bullets)
    records = []  # per-item greedy text etc. for qualitative diffs
    # GT-text (source_text) baseline on the SAME items, single scorer pass (k=1 fallback)
    gt_res = score_fn([(r.get("source_text") or "") for r in eval_rows], golds, layers,
                      [[] for _ in eval_rows])
    gt_all = gt_res.fve.float().numpy()
    t0 = time.time()
    for i, r in enumerate(eval_rows):
        vec = torch.from_numpy(r["inject"]).float()
        ly = int(r["layer"])
        w = whiteners[ly]
        # 16 sampled rollouts (on-policy T=1) + 1 greedy
        per = rollout_prompts(actor, [r["prompt_ids"]], [vec], vectors_ref,
                              group_size=n_roll, max_new_tokens=mnt, temperature=1.0,
                              device=adev, eos_ids=eos_ids, pad_id=pad_id)[0]
        texts = [tokenizer.decode(x["resp_ids"], skip_special_tokens=True) for x in per]
        # pooled candidate bullets across all rollouts, then dedup on token-ids
        # (identical bullet strings collapse) — matches the R2S selection protocol:
        # 32 whole readouts -> ~128 raw bullets -> dedup -> ~110 candidates.
        raw = []
        for t in texts:
            raw.extend(split_bullets(t, k_max=99))
        if not raw:
            raw = [t for t in texts if t.strip()]
        seen = set()
        pool = []
        for b in raw:
            key = tuple(tokenizer.encode(b.strip()))  # dedup on token ids
            if len(key) and key not in seen:
                seen.add(key)
                pool.append(b)
        n_raw = len(raw)
        # best@1 / best@N: score EACH rollout as its own 4-bullet readout (no mixing)
        gi = golds[i:i+1]
        li = layers[i:i+1]
        rr = score_fn(texts, gi.repeat(len(texts), 1), li.repeat(len(texts)),
                      [[] for _ in texts])
        ps = rr.fve.float().numpy()
        best1_fve.append(float(ps.mean()))
        bestn_fve.append(float(ps.max()))
        imgs, kept = _ar_unit_images(recon, tokenizer, pool, ly, w, space, rdev,
                                     mb=rargs.rm_micro_batch)
        gw = (w.whiten(golds[i:i+1])[0] if space.whiten else (golds[i] - w.mu)).float()
        picked = []   # oracle-selected bullets (text + coeff), in selection order
        if imgs.shape[0] == 0:
            oracle_fve.append(0.0)
            full_fve.append(0.0)
            ncand.append(0)
        else:
            res = nnomp_batch(gw.unsqueeze(0).to(rdev), imgs.to(rdev), max_atoms=args.oracle_k)
            oracle_fve.append(float(res.fve[0].cpu()))
            for a, c in zip(res.atoms[0].tolist(), res.coeffs[0].tolist(), strict=True):
                if a >= 0 and abs(c) > 1e-6:
                    picked.append({"text": pool[kept[a]][:220], "coeff": round(float(c), 4)})
            # full NNLS over ALL candidates = loosest upper bound
            vv = imgs.unsqueeze(0)
            vm = torch.ones(1, imgs.shape[0], dtype=torch.bool)
            _, ff = nnls_refit(vv, gw.unsqueeze(0), vm)
            full_fve.append(float(ff[0]))
            ncand.append(imgs.shape[0])
        # single readout at T=1 (NOT greedy — evals match the on-policy sampler),
        # scored via the training scorer (k<=4), paired point
        vectors_ref[0] = vec.to(adev).unsqueeze(0)
        try:
            gout = actor.generate(
                input_ids=torch.tensor([r["prompt_ids"]], device=adev),
                attention_mask=torch.ones(1, len(r["prompt_ids"]), dtype=torch.long, device=adev),
                max_new_tokens=mnt, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, pad_token_id=pad_id,
                eos_token_id=sorted(eos_ids), return_dict_in_generate=True)
        finally:
            vectors_ref[0] = None
        gtxt = tokenizer.decode(gout.sequences[0, len(r["prompt_ids"]):].tolist(),
                                skip_special_tokens=True)
        gr = score_fn([gtxt], golds[i:i+1], layers[i:i+1], [[]])
        greedy_fve.append(float(gr.fve[0]))
        gt_fve.append(float(gt_all[i]))
        row_layer.append(int(r["layer"]))
        _bi = int(ps.argmax())
        records.append({
            "row_id": r.get("row_id"), "layer": int(r["layer"]),
            "sample1_fve": float(gr.fve[0]), "gt_fve": float(gt_all[i]),
            "best1_fve": float(ps.mean()), "bestN_fve": float(ps.max()),
            "oracle_fve": oracle_fve[-1], "n_cand": ncand[-1], "n_raw": n_raw,
            "source_text": r.get("source_text") or "",
            "prompt_text": tokenizer.decode(r["prompt_ids"], skip_special_tokens=False),
            "sample1_text": gtxt[:600],
            "best_rollout_idx": _bi, "best_rollout_text": texts[_bi][:600],
            "rollouts": [{"text": t[:260], "fve": round(float(f), 4)}
                         for t, f in zip(texts, ps.tolist(), strict=True)],
            "oracle_bullets": picked,
        })
        if (i + 1) % 16 == 0:
            print(f"  [{i+1}/{len(eval_rows)}] greedy={np.mean(greedy_fve):.4f} "
                  f"oracle{args.oracle_k}={np.mean(oracle_fve):.4f} "
                  f"fullNNLS={np.mean(full_fve):.4f} cand~{np.mean(ncand):.0f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    gf = np.array(greedy_fve)
    of = np.array(oracle_fve)
    ff = np.array(full_fve)
    b1 = np.array(best1_fve)
    bn = np.array(bestn_fve)
    gt = np.array(gt_fve)
    lyr = np.array(row_layer)
    print(f"\n=== best-of ladder over {n_roll} rollouts, temp=1.0 (n={len(gf)}) ===")
    print(f"  GT-text (source, k=1)          : {gt.mean():.4f}  median {np.median(gt):.4f}")
    print(f"  sample@1 (single T=1 draw)     : {gf.mean():.4f}  median {np.median(gf):.4f}")
    print(f"  best@1 (mean single T=1 sample): {b1.mean():.4f}  median {np.median(b1):.4f}")
    print(f"  best@{n_roll} (max per-rollout)       : {bn.mean():.4f}  median {np.median(bn):.4f}")
    print(f"  pooled nnomp@{n_roll} (mix, pick {args.oracle_k})  : {of.mean():.4f}  "
          f"median {np.median(of):.4f}")
    print(f"  full NNLS (all ~{np.mean(ncand):.0f} cand)      : {ff.mean():.4f}")
    print(f"\n  per-layer (n | GT | greedy | best@1 | best@{n_roll} | pooled@{n_roll}):")
    for ly in sorted(set(lyr.tolist())):
        m = lyr == ly
        print(f"    L{ly:>2}: n={int(m.sum()):>2}  gt={gt[m].mean():.3f}  greedy={gf[m].mean():.3f}"
              f"  best1={b1[m].mean():.3f}  best{n_roll}={bn[m].mean():.3f}  "
              f"pooled={of[m].mean():.3f}")
    if args.out:
        import json
        Path(args.out).write_text(json.dumps(
            {"greedy": gf.tolist(), "best1": b1.tolist(), "bestN": bn.tolist(),
             "oracle": of.tolist(), "full_nnls": ff.tolist(),
             "gt_text": gt.tolist(), "layer": lyr.tolist(),
             "n_cand": ncand, "n_rollouts": n_roll, "oracle_k": args.oracle_k,
             "lora": lora, "records": records}, indent=2))
        print(f"  wrote {args.out}")


def _report(label, fve, valid, layers, rows, texts, out_path):
    v = valid.astype(bool)
    vf = fve[v]
    print(f"\n=== {label} — eval-set ceiling ===")
    print(f"  valid: {v.sum()}/{len(v)}")
    print(f"  mean FVE (valid) : {vf.mean():.4f}")
    print(f"  median           : {np.median(vf):.4f}")
    print(f"  p10 / p90        : {np.percentile(vf,10):.4f} / {np.percentile(vf,90):.4f}")
    print(f"  min / max        : {vf.min():.4f} / {vf.max():.4f}")
    # per-layer breakdown (eval set spans layers 20..60)
    print("  by layer:")
    for ly in sorted(set(layers.tolist())):
        m = (layers == ly) & v
        if m.sum():
            print(f"    L{ly:>2}: n={int(m.sum()):>3}  meanFVE={fve[m].mean():.4f}")
    if out_path:
        import json
        rec = [{"row_id": rows[i].get("row_id"), "layer": int(layers[i]),
                "fve": float(fve[i]), "valid": bool(v[i]),
                "source_text": texts[i][:200]} for i in range(len(rows))]
        Path(out_path).write_text(json.dumps(rec, indent=2))
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
