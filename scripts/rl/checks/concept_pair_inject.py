"""2-concept test: sum two known single-token steering directions into ONE activation
(provably >1-D by construction) and check whether the OLens surfaces BOTH concepts.

v = alpha * unit( unit(W_U[c1]@J[L]) + unit(W_U[c2]@J[L]) )  -> inject -> readout.
BOTH concepts named  => decoder can decompose superposition (diversity extractable when
present). Only one / a blend => decoder can't unmix. Companion to concept_inject.py (single
concept), which already reads a single injected direction ~100%/95%.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from data import load_ao_rl_dataset  # noqa: E402
from hooks import register_embed_injection_hook  # noqa: E402
from sidecar import load_sidecar  # noqa: E402

PAIRS = [  # semantically unrelated single-ish-token concepts so "both surfaced" is unambiguous
    ("cheese", "Paris"), ("anger", "recursion"), ("ocean", "democracy"),
    ("guitar", "insulin"), ("volcano", "lawyer"), ("chess", "photosynthesis"),
    ("coffee", "gravity"), ("virus", "guitar"),
]
LAYERS = [24, 40, 52, 60]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="artifacts/sc/rl_runs/iolens-rl-final-ddp600")
    p.add_argument("--lora", default="artifacts/sc/rl_runs/iolens-rl-final-ddp600/iter_000600")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--carrier-parquet", default="artifacts/sc/rl_distill/rl_gate_0.parquet")
    p.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    p.add_argument("--lens-file",
                   default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    p.add_argument("--alpha", type=float, default=16000.0)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = args.device
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.lens import JacobianLens
    from oracle_lens.jlens_readout import jlens_vector, stacked_jacobians
    from oracle_lens.pipeline.ar_loader import load_lc_reconstructor  # noqa: F401 (parity)
    from oracle_lens.pipeline.inject import inject_gt

    cfg_run = yaml.safe_load((Path(args.run_dir) / "run_config.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg_run["base_ckpt"])
    cfg = load_sidecar(cfg_run["sidecar"], tok)
    inj_id = cfg.injection_token_id
    base = AutoModelForCausalLM.from_pretrained(cfg_run["base_ckpt"], dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(dev)
    actor = PeftModel.from_pretrained(base, args.lora, is_trainable=False).eval()
    vectors_ref = [None]
    register_embed_injection_hook(actor, vectors_ref, inj_id,
                                  cfg.left_neighbor_id, cfg.right_neighbor_id)
    w_u = actor.get_output_embeddings().weight.detach().float().cpu()
    lens = JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    jstack = stacked_jacobians(lens, device="cpu", dtype=torch.float32)

    rows = load_ao_rl_dataset(args.carrier_parquet, inj_id=inj_id)
    carrier = {}
    for r in rows:
        carrier.setdefault(int(r["layer"]), r["prompt_ids"])
    eos = sorted({tok.eos_token_id, inj_id} - {None})

    def tokid(word):
        ids = tok.encode(" " + word, add_special_tokens=False)
        return ids[-1] if ids else tok.encode(word, add_special_tokens=False)[-1]

    def readouts(prompt, vec):
        outs = []
        for _ in range(args.n_samples):
            vectors_ref[0] = vec.to(dev).unsqueeze(0)
            try:
                o = actor.generate(input_ids=torch.tensor([prompt], device=dev),
                                   attention_mask=torch.ones(1, len(prompt), dtype=torch.long,
                                                             device=dev),
                                   max_new_tokens=args.max_new_tokens, do_sample=True,
                                   temperature=1.0,
                                   top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id,
                                   eos_token_id=eos, return_dict_in_generate=True)
            finally:
                vectors_ref[0] = None
            outs.append(tok.decode(o.sequences[0, len(prompt):].tolist(), skip_special_tokens=True))
        return outs

    recs = []
    for c1, c2 in PAIRS:
        t1, t2 = tokid(c1), tokid(c2)
        for layer in LAYERS:
            if layer not in carrier or jstack.shape[0] <= layer:
                continue
            v1 = jlens_vector(jstack, w_u, t1, layer=layer).float()
            v2 = jlens_vector(jstack, w_u, t2, layer=layer).float()
            vs = v1 / (v1.norm() + 1e-9) + v2 / (v2.norm() + 1e-9)
            vec = inject_gt(vs.unsqueeze(0), "unit", alpha=args.alpha, scale=1.0)[0]
            outs = readouts(carrier[layer], vec)
            h1 = any(c1.lower() in o.lower() for o in outs)
            h2 = any(c2.lower() in o.lower() for o in outs)
            both = sum(1 for o in outs if c1.lower() in o.lower() and c2.lower() in o.lower())
            recs.append({"c1": c1, "c2": c2, "layer": layer, "c1_hit": h1, "c2_hit": h2,
                         "both_same_readout": both, "readouts": outs})
            print(f"  {c1}+{c2} L{layer}: c1={'Y' if h1 else '.'} c2={'Y' if h2 else '.'} "
                  f"both-in-one={both}/{args.n_samples}", flush=True)
    n = len(recs)
    bo = sum(1 for r in recs if r["c1_hit"] and r["c2_hit"])
    one = sum(1 for r in recs if (r["c1_hit"] ^ r["c2_hit"]))
    neither = sum(1 for r in recs if not r["c1_hit"] and not r["c2_hit"])
    print(f"\n=== 2-concept decomposition ({n} pairxlayer cells) ===")
    print(f"  BOTH concepts surface: {100*bo/n:.0f}%   ONE only: {100*one/n:.0f}%   "
          f"NEITHER: {100*neither/n:.0f}%")
    print(f"  both in the SAME readout (>=1 sample): "
          f"{100*sum(1 for r in recs if r['both_same_readout']>0)/n:.0f}%")
    Path(args.out).write_text(json.dumps({"records": recs}, ensure_ascii=False))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
