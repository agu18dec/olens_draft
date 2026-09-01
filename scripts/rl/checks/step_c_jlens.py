"""Step C — is the confabulated content even IN the activation? J-lens grounding.

For activations the OLens confabulated, run the (independent) Jacobian lens on the real
source text at the injection layer and check whether the TRUE distinctive concept surfaces
in the J-lens readout. Since Step B proved the OLens faithfully decodes J-lens directions,
this is the clean fork:
  - J-lens reads the true concept  -> the info is in the activation; the OLens/reward failed
    to use it (a supervision problem).
  - J-lens can't read it either     -> the distinctive content isn't cleanly there; the
    honest readout really is the skeleton.
"""
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# row_id -> (label, true-concept probe strings, is-faithful-control)
TARGETS = {
    56943: ("Midjourney concept (OLens: swapped 16/16)",
            ["business", "owner", "woman", "female", "Asian", "entrepreneur", "CEO"], False),
    48545: ("metaphor phrase (OLens: invented 16/16)",
            ["occupiers", "East", "South", "beating", "destroy", "enemy"], False),
    26944: ("coffee shop (OLens: swapped to other shops)",
            ["coffee", "咖啡", "cafe", "café", "barista"], False),
    8696: ("MAC address (CONTROL: OLens faithful)",
           ["MAC", "网卡", "address", "physical", "LAN"], True),
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="artifacts/sc/rl_runs/iolens-rl-final-ddp600/ladder_RL.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    p.add_argument("--lens-file",
                   default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--carrier-parquet", default="artifacts/sc/rl_distill/rl_gate_0.parquet",
                   help="parquet with the gold_vector for each row_id")
    args = p.parse_args()

    from jlens.lens import JacobianLens
    from oracle_lens.jlens_readout import cosine_readout, grid_top_k, stacked_jacobians
    from oracle_lens.model import ModelBackend

    recs = {r["row_id"]: r for r in json.load(open(args.records))["records"]}
    import yaml as _yaml
    from data import load_ao_rl_dataset
    from sidecar import load_sidecar
    from transformers import AutoTokenizer
    rd = str(Path(args.records).parent)
    _cfg = _yaml.safe_load((Path(rd) / "run_config.yaml").read_text())
    _tok = AutoTokenizer.from_pretrained(_cfg["base_ckpt"])
    _inj = load_sidecar(_cfg["sidecar"], _tok).injection_token_id
    _rows = load_ao_rl_dataset(args.carrier_parquet, inj_id=_inj)
    golds = {r["row_id"]: torch.from_numpy(r["gold"]).float() for r in _rows}
    backend = ModelBackend("Qwen/Qwen3.6-27B", device=args.device, dtype=torch.bfloat16)
    w_u = backend.W_U.float()
    lens = JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    jstack = stacked_jacobians(lens, device=args.device, dtype=torch.float32)
    covered = jstack.shape[0]
    tok = backend.tokenizer

    def probe_ids(words):
        ids = set()
        for w in words:
            for variant in (w, " " + w):
                for t in tok.encode(variant, add_special_tokens=False):
                    ids.add(t)
        return ids

    for rid, (label, probes, faithful) in TARGETS.items():
        r = recs.get(rid)
        if r is None:
            print(f"[skip] row {rid} not in records"); continue
        L = int(r["layer"]); src = r.get("source_text") or ""
        if covered <= L:
            print(f"[skip] row {rid} layer {L} >= covered {covered}"); continue
        gold = golds.get(rid)
        if gold is None:
            print(f"[skip] row {rid} gold not found in parquet"); continue
        # read J-lens on the EXACT gold activation (resid at L), no position confound
        h = gold.to(args.device).float().view(1, 1, -1)     # [1, 1, d]
        # cosine_readout = magnitude-free projection; suppresses high-norm glitch tokens
        # (newlines/EOS) that dominate the softmax readout of a raw activation.
        readout = cosine_readout(jstack[L:L + 1], h, w_u)   # [1,1,vocab] ranking scores
        order = readout[0, 0].argsort(descending=True)      # [vocab]
        ids, _ = grid_top_k(readout, args.topk)
        pset = probe_ids(probes)
        print("=" * 78)
        print(f"row {rid} · L{L} · {label}")
        print(f"  SOURCE: {src[:140].replace(chr(10),' ')!r}")
        toptoks = [tok.decode([int(t)]) for t in ids[0, 0].tolist()]
        print(f"  J-lens top-{args.topk} of the exact activation: {toptoks}")
        ranks = {int(t): int((order == t).nonzero()[0]) for t in pset if t < w_u.shape[0]}
        if ranks:
            mt = min(ranks, key=ranks.get)
            verdict = "FOUND (info IS in activation)" if ranks[mt] < 50 else "absent (not cleanly there)"
            print(f"  TRUE concept best rank: {tok.decode([mt])!r} rank {ranks[mt]}  -> {verdict}")
        else:
            print("  TRUE concept: no probe token in vocab")


if __name__ == "__main__":
    main()
