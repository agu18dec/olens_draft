"""How diverse are the rollouts, really? Separates SEMANTIC/lexical diversity (different
words) from FUNCTIONAL diversity (different reconstructive content through the AR), and
within-rollout (the 4 bullets) from across-rollout (the 32 samples).

Key question: is the model saying the SAME thing 32 ways, or genuinely different things?
And is any across-sample diversity FAITHFUL (reduces reconstruction residual) or just
lexical noise? Reads the enriched ladder json (rollouts + per-rollout fve + oracle_bullets).
"""
import json
import re
import sys


def toks(s):
    return set(re.findall(r"\w+", s.lower()))


def bullets(text):
    return [b.strip(" -").strip() for b in text.splitlines() if b.strip().startswith("-")] or [text.strip()]


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main(path):
    recs = json.load(open(path))["records"]
    across_j, within_j, distinct2 = [], [], []
    fve_spread, pool_gain, tail_share = [], [], []
    for r in recs:
        ros = [x["text"] for x in r["rollouts"]]
        fves = [x.get("fve", 0.0) for x in r["rollouts"]]
        # across-rollout lexical: mean pairwise Jaccard of whole-readout token sets
        ts = [toks(t) for t in ros]
        pj = [jaccard(ts[i], ts[j]) for i in range(len(ts)) for j in range(i + 1, len(ts))]
        across_j.append(sum(pj) / len(pj) if pj else 0.0)
        # distinct-2 across all rollouts (unique bigrams / total)
        allbg = []
        for t in ros:
            w = re.findall(r"\w+", t.lower())
            allbg += list(zip(w, w[1:]))
        distinct2.append(len(set(allbg)) / max(1, len(allbg)))
        # within-rollout: mean pairwise Jaccard of the bullets inside each rollout
        wj = []
        for t in ros:
            bs = [toks(b) for b in bullets(t)]
            pw = [jaccard(bs[i], bs[j]) for i in range(len(bs)) for j in range(i + 1, len(bs))]
            if pw:
                wj.append(sum(pw) / len(pw))
        within_j.append(sum(wj) / len(wj) if wj else 0.0)
        # functional: spread of per-rollout FVE, and pooled gain over best single
        if fves:
            fve_spread.append(max(fves) - min(fves))
        pool_gain.append(r.get("oracle_fve", 0) - r.get("best1_fve", 0))
        tail_share.append(r.get("bestN_fve", 0) - r.get("best1_fve", 0))

    n = len(recs)
    mean = lambda xs: sum(xs) / len(xs)
    print(f"=== diversity probe on {path.split('/')[-1]}  ({n} items, "
          f"{len(recs[0]['rollouts'])} rollouts each) ===")
    print(f"ACROSS-rollout lexical Jaccard (1=identical words): {mean(across_j):.2f}"
          f"   [>0.5 => rephrasings]")
    print(f"ACROSS-rollout distinct-2 (1=all-unique bigrams):   {mean(distinct2):.2f}")
    print(f"WITHIN-rollout bullet Jaccard (1=4 rephrasings):    {mean(within_j):.2f}")
    print(f"per-rollout FVE spread (max-min):                   {mean(fve_spread)*100:.1f}%")
    print(f"FUNCTIONAL: pooled@N - best@1 (faithful cross gain): {mean(pool_gain)*100:+.1f}%")
    print(f"FUNCTIONAL: best@N  - best@1 (best-of-N headroom):   {mean(tail_share)*100:+.1f}%")
    # bucket items by whether cross-rollout pooling helps a lot (real diversity) or not
    hi = [i for i, g in enumerate(pool_gain) if g > 0.05]
    print(f"items where pooling adds >5 FVE (real extractable diversity): {len(hi)}/{n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "artifacts/sc/rl_runs/iolens-rl-final-ddp600/ladder_RL.json")
