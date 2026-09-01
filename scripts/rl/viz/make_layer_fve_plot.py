"""Per-layer FVE plot + summary table for the four readout conditions.

Conditions (each a line, x=layer, y=FVE %):
  GT text single bullet                 (source crop through AR, k=1)
  SFT single bullet (before warmstart)  (SFT first-bullet only, k=1)
  SFT 4 bullet                          (SFT best@1, <=4 bullet readout, T=1)
  RL 4 bullet                           (RL  best@1, <=4 bullet readout, T=1)

Also reports median + range (the "other things", not just the mean) in a table.

Usage: python make_layer_fve_plot.py ladder_RL.json ladder_SFT.json \
       ladder_SFT_single1.json out.html
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def pl_records(recs, key):
    a = defaultdict(list)
    for r in recs:
        a[r["layer"]].append(r[key])
    return {ly: sum(v) / len(v) for ly, v in sorted(a.items())}


def pl_arrays(vals, layers):
    a = defaultdict(list)
    for v, ly in zip(vals, layers):
        a[int(ly)].append(v)
    return {ly: sum(v) / len(v) for ly, v in sorted(a.items())}


def main():
    rl = json.loads(Path(sys.argv[1]).read_text())
    sft = json.loads(Path(sys.argv[2]).read_text())
    s1 = json.loads(Path(sys.argv[3]).read_text())
    out = sys.argv[4] if len(sys.argv) > 4 else "layer_fve_plot.html"

    layers = sorted({r["layer"] for r in rl["records"]})
    ncS = sum(r["n_cand"] for r in sft["records"]) / len(sft["records"])
    ncR = sum(r["n_cand"] for r in rl["records"]) / len(rl["records"])
    # (name, per-item array, per-layer dict, color, dashed, unique-candidate-bullets)
    conds = [
        ("GT text single bullet", [r["gt_fve"] for r in rl["records"]],
         pl_records(rl["records"], "gt_fve"), "#59a14f", True, None),
        ("SFT single bullet (before warmstart)", s1["per_item"],
         pl_arrays(s1["per_item"], s1["layer"]), "#9350a8", False, None),
        ("SFT 4 bullet", [r["best1_fve"] for r in sft["records"]],
         pl_records(sft["records"], "best1_fve"), "#3f7fc4", False, ncS),
        ("RL 4 bullet", [r["best1_fve"] for r in rl["records"]],
         pl_records(rl["records"], "best1_fve"), "#d97a1a", False, ncR),
    ]

    W, H = 1080, 520
    padL, padR, padB, padT = 66, 300, 54, 36
    x0, x1, y0, y1 = padL, W - padR, H - padB, padT
    ymax = max(max(d.values()) for _, _, d, _, _, _ in conds) * 100 * 1.15
    X = lambda i: x0 + (x1 - x0) * i / max(1, len(layers) - 1)
    Y = lambda pct: y1 + (y0 - y1) * (1 - pct / ymax)

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         'font-family="ui-sans-serif,system-ui,sans-serif">']
    for t in range(6):
        pct = ymax * t / 5; yy = Y(pct)
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" fill="var(--muted)" '
                 f'font-size="12">{pct:.0f}%</text>')
    for i, ly in enumerate(layers):
        s.append(f'<text x="{X(i):.1f}" y="{y0+18}" text-anchor="middle" fill="var(--ink)" '
                 f'font-size="12">{ly}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" text-anchor="middle" fill="var(--muted)" '
             f'font-size="14">layer</text>')
    s.append(f'<text x="20" y="{(y0+y1)/2:.0f}" text-anchor="middle" fill="var(--muted)" '
             f'font-size="14" transform="rotate(-90 20 {(y0+y1)/2:.0f})">FVE (%)</text>')
    for name, _, d, col, dash, _nc in conds:
        pts = [(X(i), Y(d[ly] * 100)) for i, ly in enumerate(layers)]
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = ' stroke-dasharray="7 5"' if dash else ""
        # thick lines so every series (incl. the single-bullet one) is clearly visible
        s.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="3.6"{da}/>')
        for (x, y), ly in zip(pts, layers):
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="{col}" '
                     f'stroke="var(--card)" stroke-width="1.5">'
                     f'<title>{name} · L{ly}: {d[ly]*100:.1f}%</title></circle>')
    for k, (name, _, d, col, dash, _nc) in enumerate(conds):
        mean = sum(d.values()) / len(d) * 100
        yy = padT + 10 + k * 26
        da = ' stroke-dasharray="7 5"' if dash else ""
        s.append(f'<line x1="{x1+16}" x2="{x1+40}" y1="{yy}" y2="{yy}" stroke="{col}" '
                 f'stroke-width="4"{da}/>')
        s.append(f'<text x="{x1+46}" y="{yy+4}" fill="var(--ink)" font-size="12">{name} '
                 f'<tspan fill="var(--muted)">({mean:.1f}%)</tspan></text>')
    s.append('</svg>')

    # summary table — mean / median / range, as PERCENT (the "other things" beyond mean)
    def row(name, arr, d, col, nc):
        a = [v * 100 for v in arr]
        lm = {ly: v * 100 for ly, v in d.items()}          # per-layer means (%)
        lo, hi = min(lm, key=lm.get), max(lm, key=lm.get)   # layers of the extremes
        lrange = f"L{lo} {lm[lo]:.1f}% → L{hi} {lm[hi]:.1f}%"
        nccell = f"{nc:.0f}" if nc is not None else "—"
        return (f'<tr><td><span class="sw" style="background:{col}"></span>{name}</td>'
                f'<td>{st.mean(a):.1f}%</td><td>{st.median(a):.1f}%</td>'
                f'<td>{lrange}</td><td>{nccell}</td></tr>')
    rows = "".join(row(n, arr, d, col, nc) for n, arr, d, col, _, nc in conds)

    doc = f"""<!doctype html><meta charset=utf-8><title>iolens.final — FVE by layer</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --grid:#e7e7e4; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e5e3dd; }}
body {{ font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--ink); margin:0; padding:30px; }}
h1 {{ font-size:19px; }} .sub {{ color:var(--muted); font-size:13px; margin-bottom:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 22px; max-width:1120px; }}
table {{ border-collapse:collapse; font-size:13.5px; margin-top:16px; width:100%; }}
td,th {{ padding:7px 12px; text-align:right; border-bottom:1px solid var(--grid); }}
td:first-child,th:first-child {{ text-align:left; }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:8px; vertical-align:middle; }}
</style>
<h1>iolens.final — reconstruction FVE by layer, four readout conditions</h1>
<div class="sub">n={len(rl['records'])} activations · 4-bullet = best@1 (T=1) · single bullet = first-bullet only (k=1) · dashed = GT-text ceiling</div>
<div class="card">{''.join(s)}
<table><tr><th>condition</th><th>mean FVE</th><th>median</th><th>layer range</th><th>unique bullets/item</th><th>n</th></tr>{rows}</table>
<div class="sub" style="margin-top:10px">unique candidate bullets = distinct (token-deduped) bullets pooled over the 32 T=1 rollouts. RL's pool is ~2× SFT's (~{ncR:.0f} vs ~{ncS:.0f}) — the readouts are far more diverse, which is what gives the oracle more to select from. Single-readout conditions have no pool.</div>
"""
    Path(out).write_text(doc)
    print(f"wrote {out}")
    for n, arr, _, _, _, _ in conds:
        a = [v * 100 for v in arr]
        print(f"  {n:38} mean {st.mean(a):.1f}%  median {st.median(a):.1f}%")


if __name__ == "__main__":
    main()
