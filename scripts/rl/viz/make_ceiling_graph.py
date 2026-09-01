"""Render the before/after eval-FVE graph (SFT vs RL, per layer) + ceilings.

Consumes perlayer_RL.json + perlayer_SFT.json (from checks/eval_ceiling.py --mode oracle
--per-layer N). Emits a self-contained HTML: per-layer line chart of
  SFT greedy | RL greedy | GT-text ceiling | RL oracle-16 upper bound
plus an aggregate summary and a table view. Colorblind-safe categorical palette
(validated with the dataviz skill's validate_palette.js).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def by_layer(d, key):
    agg = defaultdict(list)
    for f, ly in zip(d[key], d["layer"]):
        agg[int(ly)].append(f)
    return {ly: sum(v) / len(v) for ly, v in sorted(agg.items())}


def main():
    rl = json.loads(Path(sys.argv[1]).read_text())
    sft = json.loads(Path(sys.argv[2]).read_text())
    out = sys.argv[3] if len(sys.argv) > 3 else "ceiling_graph.html"

    layers = sorted({int(x) for x in rl["layer"]})
    series = {
        "SFT greedy": by_layer(sft, "greedy"),
        "RL greedy": by_layer(rl, "greedy"),
        "GT-text ceiling": by_layer(rl, "gt_text"),
        "RL oracle-16": by_layer(rl, "oracle"),
    }
    # aggregate means
    def amean(d, k):
        return sum(d[k]) / len(d[k])
    agg = {
        "SFT greedy": amean(sft, "greedy"), "RL greedy": amean(rl, "greedy"),
        "GT-text ceiling": amean(rl, "gt_text"), "RL oracle-16": amean(rl, "oracle"),
    }
    # colorblind-safe (dataviz default categorical order): blue, orange, teal-green, red
    # CVD-safe: blue/orange/green/purple (no green-red pair); ceiling also dashed +
    # every series direct-labeled, so identity is never color-alone.
    colors = {"SFT greedy": "#4e79a7", "RL greedy": "#f28e2b",
              "GT-text ceiling": "#59a14f", "RL oracle-16": "#b07aa1"}

    W, H = 900, 460
    padL, padR, padB, padT = 60, 190, 50, 40
    ymax = max(max(s.values()) for s in series.values()) * 1.1
    x0, x1 = padL, W - padR
    y0, y1 = H - padB, padT

    def X(i):
        return x0 + (x1 - x0) * i / max(1, len(layers) - 1)

    def Y(v):
        return y1 + (y0 - y1) * (1 - v / ymax)

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
           'font-family="ui-sans-serif,system-ui,sans-serif">']
    svg.append(f'<rect width="{W}" height="{H}" fill="var(--bg)"/>')
    # y gridlines
    for t in range(0, 6):
        v = ymax * t / 5
        yy = Y(v)
        svg.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/>')
        svg.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" fill="var(--muted)" '
                   f'font-size="11">{v*100:.0f}%</text>')
    # x labels
    for i, ly in enumerate(layers):
        svg.append(f'<text x="{X(i):.1f}" y="{y0+18}" text-anchor="middle" fill="var(--muted)" '
                   f'font-size="11">{ly}</text>')
    svg.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-6}" text-anchor="middle" fill="var(--muted)" '
               f'font-size="12">layer</text>')
    svg.append(f'<text x="16" y="{(y0+y1)/2:.0f}" text-anchor="middle" fill="var(--muted)" '
               f'font-size="12" transform="rotate(-90 16 {(y0+y1)/2:.0f})">FVE (%)</text>')
    # lines + points
    for name, s in series.items():
        pts = [(X(i), Y(s[ly])) for i, ly in enumerate(layers)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if name == "GT-text ceiling" else ""
        svg.append(f'<path d="{d}" fill="none" stroke="{colors[name]}" stroke-width="2.2"{dash}/>')
        for (x, y), ly in zip(pts, layers):
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{colors[name]}">'
                       f'<title>{name} · L{ly}: {s[ly]*100:.1f}%</title></circle>')
    # legend (direct labels at right)
    ly0 = padT + 10
    for k, (name, s) in enumerate(series.items()):
        yy = ly0 + k * 22
        svg.append(f'<line x1="{x1+16}" x2="{x1+34}" y1="{yy}" y2="{yy}" '
                   f'stroke="{colors[name]}" stroke-width="3"/>')
        svg.append(f'<text x="{x1+40}" y="{yy+4}" fill="var(--ink)" font-size="12">{name} '
                   f'<tspan fill="var(--muted)">({agg[name]:.3f})</tspan></text>')
    svg.append('</svg>')

    # table
    rows = "".join(
        f"<tr><td>L{ly}</td>" + "".join(
            f"<td>{series[n][ly]:.3f}</td>" for n in series) + "</tr>"
        for ly in layers)
    hdr = "".join(f"<th>{n}</th>" for n in series)

    html = f"""<!doctype html><meta charset=utf-8>
<title>iolens.final RL — eval-FVE ceiling, before/after by layer</title>
<style>
:root {{ --bg:#f7f7f5; --grid:#ececec; --ink:#1a1a1a; --muted:#6b6b6b; --card:#ffffff; }}
body {{ background:var(--bg); font-family:ui-sans-serif,system-ui,sans-serif; color:var(--ink);
 max-width:960px; margin:32px auto; padding:0 16px; }}
.card {{ background:var(--card); border-radius:12px; padding:18px 22px; margin:14px 0; }}
h1 {{ font-size:19px; }} h2 {{ font-size:14px; color:var(--muted); font-weight:600; }}
table {{ border-collapse:collapse; font-size:12px; width:100%; }}
td,th {{ padding:4px 8px; text-align:right; border-bottom:1px solid var(--grid); }}
td:first-child,th:first-child {{ text-align:left; }}
.big {{ font-size:13px; line-height:1.7; }} .big b {{ color:var(--ink); }}
code {{ background:var(--grid); padding:1px 5px; border-radius:4px; }}
</style>
<h1>iolens.final — RL vs SFT joint-FVE, per layer (eval set, oracle over {rl['n_rollouts']} rollouts, temp 1.0)</h1>
<div class=card>{''.join(svg)}</div>
<div class="card big">
<b>Aggregate (mean over {len(rl['greedy'])} items):</b><br>
SFT greedy <b>{agg['SFT greedy']:.3f}</b> → RL greedy <b>{agg['RL greedy']:.3f}</b>
(+{agg['RL greedy']-agg['SFT greedy']:.3f}, {(agg['RL greedy']/agg['SFT greedy']-1)*100:.0f}%) &nbsp;·&nbsp;
GT-text ceiling <b>{agg['GT-text ceiling']:.3f}</b> &nbsp;·&nbsp;
RL oracle-16 upper bound <b>{agg['RL oracle-16']:.3f}</b><br>
<span style="color:var(--muted)">GT-text = the literal source crop scored through the same AR (k=1); oracle-16 = best nnomp
selection over the RL policy's own {rl['n_rollouts']} sampled rollouts. RL LoRA: <code>{Path(rl['lora']).name}</code></span>
</div>
<div class=card><h2>table</h2><table><tr><th>layer</th>{hdr}</tr>{rows}</table></div>
"""
    Path(out).write_text(html)
    print(f"wrote {out}")
    print("aggregate:", {k: round(v, 4) for k, v in agg.items()})


if __name__ == "__main__":
    main()
