"""best-of-N debug plot: how much is extractable, and did RL shift it?

x = N (rollouts). Curves (RL solid, SFT dashed):
  best@N   = E[ max joint-FVE over N whole readouts ]  (no cross-sample mixing)
Anchors: best@1 (N=1), pooled-nnomp@32 (mix bullets across 32), GT-text ceiling.
Bootstraps the expected max over random N-subsets of the 32 saved rollouts.

Usage: python make_bestofn_plot.py ladder_RL.json ladder_SFT.json out.html
"""
import json
import sys
from pathlib import Path

import numpy as np


def curve(recs):
    F = [np.array([r["fve"] for r in rec["rollouts"]]) for rec in recs]
    Ns = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    rng = np.random.RandomState(0)
    out = {}
    for N in Ns:
        vals = []
        for f in F:
            if len(f) <= N:
                vals.append(f.max())
            else:
                vals.append(np.mean([f[rng.choice(len(f), N, False)].max() for _ in range(300)]))
        out[N] = float(np.mean(vals)) * 100
    return out, Ns


def main():
    rl = json.loads(Path(sys.argv[1]).read_text()); sft = json.loads(Path(sys.argv[2]).read_text())
    out = sys.argv[3] if len(sys.argv) > 3 else "bestofn_plot.html"
    # optional nnomp_analysis json -> pooled-nnomp@N curve (mix bullets across N rollouts)
    nnj = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
    R, S = rl["records"], sft["records"]
    crl, Ns = curve(R); csft, _ = curve(S)
    poolR = np.mean([r["oracle_fve"] for r in R]) * 100
    poolS = np.mean([r["oracle_fve"] for r in S]) * 100
    gt = np.mean([r["gt_fve"] for r in R]) * 100
    # pooled@N curve mapped onto the same Ns (nnomp json uses string keys 1,2,4,8,16,32)
    pooledN = None
    if nnj and "pooledN" in nnj:
        pk = {int(k): v * 100 for k, v in nnj["pooledN"].items()}
        pooledN = {N: pk.get(N) for N in Ns if N in pk}

    W, H = 860, 500; padL, padR, padB, padT = 62, 210, 52, 34
    x0, x1, y0, y1 = padL, W - padR, H - padB, padT
    ymax = max(max(crl.values()), max(csft.values()), poolR) * 1.12
    X = lambda i: x0 + (x1 - x0) * i / (len(Ns) - 1)
    Y = lambda v: y1 + (y0 - y1) * (1 - v / ymax)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="ui-sans-serif,system-ui,sans-serif">']
    for t in range(6):
        v = ymax * t / 5; yy = Y(v)
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" fill="var(--muted)" font-size="11">{v:.0f}%</text>')
    for i, N in enumerate(Ns):
        s.append(f'<text x="{X(i):.1f}" y="{y0+18}" text-anchor="middle" fill="var(--ink)" font-size="11">{N}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" text-anchor="middle" fill="var(--muted)" font-size="13">N (rollouts)</text>')
    s.append(f'<text x="18" y="{(y0+y1)/2:.0f}" text-anchor="middle" fill="var(--muted)" font-size="13" transform="rotate(-90 18 {(y0+y1)/2:.0f})">joint FVE (%)</text>')
    # GT ceiling rule
    gy = Y(gt)
    s.append(f'<line x1="{x0}" x2="{x1}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="#59a14f" stroke-width="1.5" stroke-dasharray="2 3"/>')
    s.append(f'<text x="{x1}" y="{gy-5:.1f}" text-anchor="end" fill="#59a14f" font-size="11">GT-text ceiling {gt:.1f}%</text>')
    def draw(c, col, dash):
        pts = [(X(i), Y(c[N])) for i, N in enumerate(Ns)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.6"{dash}/>')
        for (x, y), N in zip(pts, Ns):
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{col}"><title>best@{N}: {c[N]:.1f}%</title></circle>')
    draw(crl, "#d97a1a", ""); draw(csft, "#3f7fc4", ' stroke-dasharray="6 4"')
    # pooled-nnomp@N curve (mix bullets across N rollouts) — RL solid-ish, plus SFT curve if given
    sftPN = None
    if nnj and nnj.get("pooledN_sft"):
        sk = {int(k): v * 100 for k, v in nnj["pooledN_sft"].items()}
        sftPN = {N: sk.get(N) for N in Ns if N in sk}
    if pooledN:
        pts = [(X(Ns.index(N)), Y(v)) for N, v in pooledN.items() if v is not None]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        s.append(f'<path d="{d}" fill="none" stroke="#9350a8" stroke-width="2.6" stroke-dasharray="1 3"/>')
        for (x, y), (N, v) in zip(pts, [(N, v) for N, v in pooledN.items() if v is not None]):
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#9350a8"><title>RL pooled-nnomp@{N}: {v:.1f}%</title></circle>')
        if sftPN:  # full SFT pooled@N curve (once nnomp on SFT is available)
            sp = [(X(Ns.index(N)), Y(v)) for N, v in sftPN.items() if v is not None]
            ds = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in sp)
            s.append(f'<path d="{ds}" fill="none" stroke="#b98fca" stroke-width="2.2" stroke-dasharray="1 3"/>')
            for (x, y), (N, v) in zip(sp, [(N, v) for N, v in sftPN.items() if v is not None]):
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#b98fca"><title>SFT pooled-nnomp@{N}: {v:.1f}%</title></circle>')
        else:  # only the SFT pooled@32 anchor is available so far (no GPU for the curve yet)
            s.append(f'<circle cx="{X(len(Ns)-1):.1f}" cy="{Y(poolS):.1f}" r="4.5" fill="none" stroke="#b98fca" stroke-width="2"><title>SFT pooled-nnomp@32: {poolS:.1f}% (full curve pending GPU)</title></circle>')
            s.append(f'<text x="{X(len(Ns)-1):.1f}" y="{Y(poolS)+16:.1f}" text-anchor="middle" fill="#b98fca" font-size="9">SFT pooled@32</text>')
    else:
        for pv in [poolR, poolS]:
            s.append(f'<circle cx="{X(len(Ns)-1):.1f}" cy="{Y(pv):.1f}" r="5" fill="#9350a8" stroke="var(--card)" stroke-width="1.5"><title>pooled-nnomp@32: {pv:.1f}%</title></circle>')
    # legend
    leg = [("RL best@N", "#d97a1a", f"1→32: {crl[1]:.1f}→{crl[32]:.1f}%"),
           ("SFT best@N", "#3f7fc4", f"{csft[1]:.1f}→{csft[32]:.1f}%"),
           ("pooled-nnomp@32 (mix)", "#9350a8", f"RL {poolR:.1f} / SFT {poolS:.1f}%"),
           ("GT-text ceiling", "#59a14f", f"{gt:.1f}%")]
    for k, (lab, col, note) in enumerate(leg):
        yy = padT + 8 + k * 24
        s.append(f'<line x1="{x1+14}" x2="{x1+34}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="3"/>')
        s.append(f'<text x="{x1+40}" y="{yy+4}" fill="var(--ink)" font-size="11.5">{lab}</text>')
        s.append(f'<text x="{x1+40}" y="{yy+17}" fill="var(--muted)" font-size="10">{note}</text>')
    s.append("</svg>")
    doc = f"""<!doctype html><meta charset=utf-8><title>best-of-N — RL vs SFT</title>
<style>:root{{--bg:#f7f7f5;--card:#fff;--grid:#ececec;--ink:#1a1a1a;--muted:#6b6b6b;--line:#e5e3dd}}
body{{font-family:ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:30px}}
h1{{font-size:18px}}.sub{{color:var(--muted);font-size:13px;margin-bottom:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;max-width:900px}}</style>
<h1>best-of-N: how much is extractable, and did RL move it?</h1>
<div class="sub">n={len(R)} eval items · best@N = expected max joint-FVE over N whole readouts · pooled = nnomp mixing bullets across 32 · solid RL / dashed SFT</div>
<div class="card">{''.join(s)}</div>"""
    Path(out).write_text(doc); print(f"wrote {out}")
    print(f"RL  best@1={crl[1]:.1f} best@32={crl[32]:.1f} pooled@32={poolR:.1f}")
    print(f"SFT best@1={csft[1]:.1f} best@32={csft[32]:.1f} pooled@32={poolS:.1f}  GT={gt:.1f}")


if __name__ == "__main__":
    main()
