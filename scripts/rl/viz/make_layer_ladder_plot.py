"""Wide per-layer plot of ALL ladder metrics, SFT vs RL. x=layer, y=FVE(%).

Lines (color = rung, SOLID = RL, DASHED = SFT):
  GT-text ceiling (single line, checkpoint-independent) · sample@1 · best@1 ·
  best@32 · pooled-nnomp@32.
Subtitle carries the aggregate table incl. unique-candidate-bullet counts (a count,
not an FVE, so it is not plotted).

Usage: python make_layer_ladder_plot.py ladder_RL.json ladder_SFT.json out.html
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def pl(recs, key):
    a = defaultdict(list)
    for r in recs:
        a[r["layer"]].append(r[key])
    return {ly: sum(v) / len(v) for ly, v in sorted(a.items())}


def amean(recs, k):
    return sum(r[k] for r in recs) / len(recs)


def main():
    rl = json.loads(Path(sys.argv[1]).read_text())
    sft = json.loads(Path(sys.argv[2]).read_text())
    out = sys.argv[3] if len(sys.argv) > 3 else "layer_ladder_plot.html"
    R, S = rl["records"], sft["records"]
    layers = sorted({r["layer"] for r in R})

    # rung -> (record key, color)
    rungs = [("GT-text ceiling", "gt_fve", "#59a14f"),
             ("sample@1", "sample1_fve", "#9c755f"),
             ("best@1", "best1_fve", "#3f7fc4"),
             ("best@32", "bestN_fve", "#d97a1a"),
             ("pooled-nnomp@32", "oracle_fve", "#9350a8")]

    W, H = 1180, 540
    padL, padR, padB, padT = 66, 300, 56, 40
    x0, x1, y0, y1 = padL, W - padR, H - padB, padT
    allvals = []
    for _, k, _ in rungs:
        allvals += [pl(R, k)[ly] for ly in layers] + [pl(S, k)[ly] for ly in layers]
    ymax = max(allvals) * 100 * 1.12
    X = lambda i: x0 + (x1 - x0) * i / max(1, len(layers) - 1)
    Y = lambda pct: y1 + (y0 - y1) * (1 - pct / ymax)

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         'font-family="ui-sans-serif,system-ui,sans-serif">']
    for t in range(7):
        pct = ymax * t / 6; yy = Y(pct)
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" fill="var(--muted)" '
                 f'font-size="11">{pct:.0f}%</text>')
    for i, ly in enumerate(layers):
        s.append(f'<text x="{X(i):.1f}" y="{y0+18}" text-anchor="middle" fill="var(--ink)" '
                 f'font-size="11">{ly}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" text-anchor="middle" fill="var(--muted)" '
             f'font-size="13">layer</text>')
    s.append(f'<text x="20" y="{(y0+y1)/2:.0f}" text-anchor="middle" fill="var(--muted)" '
             f'font-size="13" transform="rotate(-90 20 {(y0+y1)/2:.0f})">FVE (%)</text>')

    def draw(sr, color, style, name):
        pts = [(X(i), Y(sr[ly] * 100)) for i, ly in enumerate(layers)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.3"{style}/>')
        for (x, y), ly in zip(pts, layers):
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}">'
                     f'<title>{name} · L{ly}: {sr[ly]*100:.1f}%</title></circle>')

    for name, k, col in rungs:
        if name == "GT-text ceiling":
            draw(pl(R, k), col, ' stroke-dasharray="2 3"', "GT-text (k=1)")
        else:
            draw(pl(R, k), col, "", f"RL {name}")
            draw(pl(S, k), col, ' stroke-dasharray="5 4"', f"SFT {name}")

    # legend: rung colours + style key
    lx = x1 + 18; ly0 = padT + 6
    s.append(f'<text x="{lx}" y="{ly0}" fill="var(--muted)" font-size="11" font-weight="600">rung (mean SFT→RL)</text>')
    for j, (name, k, col) in enumerate(rungs):
        yy = ly0 + 18 + j * 22
        s.append(f'<line x1="{lx}" x2="{lx+22}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="3"/>')
        if name == "GT-text ceiling":
            s.append(f'<text x="{lx+28}" y="{yy+4}" fill="var(--ink)" font-size="11.5">{name} '
                     f'<tspan fill="var(--muted)">{amean(R,k)*100:.1f}%</tspan></text>')
        else:
            s.append(f'<text x="{lx+28}" y="{yy+4}" fill="var(--ink)" font-size="11.5">{name} '
                     f'<tspan fill="var(--muted)">{amean(S,k)*100:.1f}→{amean(R,k)*100:.1f}%</tspan></text>')
    ky = ly0 + 18 + len(rungs) * 22 + 12
    s.append(f'<line x1="{lx}" x2="{lx+22}" y1="{ky}" y2="{ky}" stroke="var(--ink)" stroke-width="2.3"/>')
    s.append(f'<text x="{lx+28}" y="{ky+4}" fill="var(--ink)" font-size="11.5">solid = RL</text>')
    s.append(f'<line x1="{lx}" x2="{lx+22}" y1="{ky+20}" y2="{ky+20}" stroke="var(--ink)" stroke-width="2.3" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{lx+28}" y="{ky+24}" fill="var(--ink)" font-size="11.5">dashed = SFT</text>')
    s.append('</svg>')

    # aggregate table (incl candidate counts)
    ncR = sum(r["n_cand"] for r in R) / len(R); ncS = sum(r["n_cand"] for r in S) / len(S)
    trs = "".join(
        f"<tr><td>{n}</td><td>{amean(S,k):.3f}</td><td>{amean(R,k):.3f}</td>"
        f"<td class='up'>+{amean(R,k)-amean(S,k):.3f}</td></tr>"
        for n, k, _ in rungs)
    trs += (f"<tr><td>unique candidate bullets / item</td><td>{ncS:.0f}</td>"
            f"<td>{ncR:.0f}</td><td class='up'>~{ncR/ncS:.1f}×</td></tr>")

    doc = f"""<!doctype html><meta charset=utf-8><title>iolens.final — all ladder metrics by layer</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --grid:#ececec; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e5e3dd; --up:#3d8b40; }}
body {{ font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--ink); margin:0; padding:30px; }}
h1 {{ font-size:18px; }} .sub {{ color:var(--muted); font-size:13px; margin-bottom:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; max-width:1220px; }}
table {{ border-collapse:collapse; font-size:13px; margin-top:14px; }}
td,th {{ padding:5px 12px; text-align:right; border-bottom:1px solid var(--grid); }}
td:first-child,th:first-child {{ text-align:left; }} .up {{ color:var(--up); }}
</style>
<h1>iolens.final — all ladder metrics by layer (SFT → RL)</h1>
<div class="sub">n={len(R)} activations · all sampled at T=1 · solid = RL (iter 600), dashed = SFT (pre-RL) · GT-text is checkpoint-independent</div>
<div class="card">{''.join(s)}
<table><tr><th>metric</th><th>SFT</th><th>RL</th><th>Δ</th></tr>{trs}</table></div>
"""
    Path(out).write_text(doc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
