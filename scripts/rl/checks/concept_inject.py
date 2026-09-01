"""Step B — single-concept faithfulness sanity check for the inverted-OLens.

For each known concept we build an "activation" three independent ways and inject it into
the OLens, then check whether the bullet readout names the concept:
  - ar     : AR reconstruction of the concept text  (recon.head(h+layer_emb)), the exact
             training injection distribution after inject_gt(unit, alpha).
  - steer  : the J-lens steering DIRECTION for the concept token, v = W_U[tok] @ J[layer]
             (jlens_vector), injected the same way. A second, AR-independent handle.
  - control: norm-matched Gaussian noise — the honesty floor (should ~never hit).

If `ar`/`steer` hit-rates >> `control` on some layers, the lens reads real signal for known
single concepts. Shardable across GPUs with --shard/--n-shards (one process per GPU).

Reuses: reward_text_ids + capture_block_outputs (text->AR), inject_gt (injection space),
register_embed_injection_hook + actor.generate (the eval_ceiling _oracle path),
jlens_vector + stacked_jacobians (steering).
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # scripts/rl on path
from data import load_ao_rl_dataset  # noqa: E402
from hooks import register_embed_injection_hook  # noqa: E402
from sidecar import load_sidecar  # noqa: E402

CONCEPTS = [
    "cheese", "mozzarella", "Paris", "France", "anger", "happiness", "recursion",
    "gravity", "bankruptcy", "photosynthesis", "ocean", "guitar", "democracy", "virus",
    "chess", "coffee", "mountain", "lawyer", "insulin", "earthquake",
]
LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]


def ar_activation(recon, tok, text, layer, dev):
    from oracle_lens.pipeline.multilayer_reconstructor import ml_collate
    from oracle_lens.pipeline.rl_reward import capture_block_outputs, reward_text_ids
    ids = reward_text_ids(tok, text)
    batch = ml_collate([{"ids": torch.tensor(ids), "target": torch.zeros(1)}], pad_id=0)
    idd = batch["input_ids"].to(dev)
    mask = batch["attention_mask"].to(dev)
    cap = capture_block_outputs(recon.backbone, [layer], idd, mask)
    last = int(mask.sum(1) - 1)
    pos = {int(ly): i for i, ly in enumerate(recon.layers)}
    h = cap[layer][0, last].float()
    return recon.head(h + recon.layer_emb.weight[pos[layer]]).float().cpu()


def readouts(actor, tok, prompt_ids, vec, vectors_ref, inj_id, dev, mnt, n):
    eos = sorted({tok.eos_token_id, inj_id} - {None})
    outs = []
    for _ in range(n):
        vectors_ref[0] = vec.to(dev).unsqueeze(0)
        try:
            o = actor.generate(
                input_ids=torch.tensor([prompt_ids], device=dev),
                attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long, device=dev),
                max_new_tokens=mnt, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, pad_token_id=tok.eos_token_id,
                eos_token_id=eos, return_dict_in_generate=True)
        finally:
            vectors_ref[0] = None
        outs.append(tok.decode(o.sequences[0, len(prompt_ids):].tolist(),
                               skip_special_tokens=True))
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--carrier-parquet", default="artifacts/sc/rl_distill/rl_gate_0.parquet")
    p.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    p.add_argument("--lens-file",
                   default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    p.add_argument("--alpha", type=float, default=16000.0)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = args.device
    cfg_run = yaml.safe_load((Path(args.run_dir) / "run_config.yaml").read_text())
    rargs = SimpleNamespace(**cfg_run)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.lens import JacobianLens
    from oracle_lens.jlens_readout import jlens_vector, stacked_jacobians
    from oracle_lens.pipeline.ar_loader import load_lc_reconstructor
    from oracle_lens.pipeline.inject import inject_gt

    tok = AutoTokenizer.from_pretrained(rargs.base_ckpt)
    cfg = load_sidecar(rargs.sidecar, tok)
    inj_id = cfg.injection_token_id
    print(f"[concept] loading actor {args.lora} on {dev}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(rargs.base_ckpt, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(dev)
    actor = PeftModel.from_pretrained(base, args.lora, is_trainable=False).eval()
    vectors_ref = [None]
    register_embed_injection_hook(actor, vectors_ref, inj_id,
                                  cfg.left_neighbor_id, cfg.right_neighbor_id)
    recon = load_lc_reconstructor(Path(rargs.ar_ckpt), device=dev, eager=True)
    w_u = actor.get_output_embeddings().weight.detach().float().cpu()  # [vocab, d]

    print("[concept] loading J-lens (steering directions)", flush=True)
    lens = JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    jstack = stacked_jacobians(lens, device="cpu", dtype=torch.float32)  # [covered, d, d]
    covered = jstack.shape[0]
    d = w_u.shape[1]

    # one carrier prompt per layer (carries the marker trigram + encodes the layer)
    rows = load_ao_rl_dataset(args.carrier_parquet, inj_id=inj_id)
    carrier = {}
    for r in rows:
        carrier.setdefault(int(r["layer"]), r["prompt_ids"])
    torch.manual_seed(0)

    concepts = [c for i, c in enumerate(CONCEPTS) if i % args.n_shards == args.shard]
    print(f"[concept] shard {args.shard}/{args.n_shards}: {concepts}", flush=True)
    recs = []
    for concept in concepts:
        ids = tok.encode(" " + concept, add_special_tokens=False)
        token_id = ids[-1] if ids else tok.encode(concept, add_special_tokens=False)[-1]
        for L in LAYERS:
            if L not in carrier or covered <= L:
                continue
            prompt = carrier[L]
            ar_raw = ar_activation(recon, tok, concept, L, dev)
            v_steer = jlens_vector(jstack, w_u, token_id, layer=L).float()
            g = torch.randn(d)
            arms = {
                "ar": inject_gt(ar_raw.unsqueeze(0), "unit", alpha=args.alpha, scale=1.0)[0],
                "steer": inject_gt(v_steer.unsqueeze(0), "unit", alpha=args.alpha, scale=1.0)[0],
                "control": inject_gt(g.unsqueeze(0), "unit", alpha=args.alpha, scale=1.0)[0],
            }
            for arm, vec in arms.items():
                outs = readouts(actor, tok, prompt, vec, vectors_ref, inj_id, dev,
                                args.max_new_tokens, args.n_samples)
                hit = any(concept.lower() in o.lower() for o in outs)
                recs.append({"concept": concept, "layer": L, "arm": arm,
                             "token_id": int(token_id), "hit": hit, "readouts": outs})
            print(f"  {concept} L{L}: "
                  + " ".join(f"{a}={'Y' if recs[-3 + i]['hit'] else '.'}"
                             for i, a in enumerate(['ar', 'steer', 'control'])), flush=True)
    Path(args.out).write_text(json.dumps({"records": recs}, ensure_ascii=False))
    print(f"[concept] wrote {len(recs)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
