"""Clean grouped-bar plot of ALL ladder metrics, SFT vs RL (aggregate means).

Usage: python make_metrics_plot.py ladder_RL.json ladder_SFT.json out.html
One vertical grouped bar per rung (sample@1 / best@1 / best@32 / pooled-nnomp@32),
plus the GT-text ceiling as a reference rule. SFT vs RL, direct value labels,
one axis, CVD-safe two-colour palette, dark+light.
"""
import json
import sys
from pathlib import Path


def amean(recs, k):
    return sum(r[k] for r in recs) / len(recs)


def main():
    rl = json.loads(Path(sys.argv[1]).read_text())
    sft = json.loads(Path(sys.argv[2]).read_text())
    out = sys.argv[3] if len(sys.argv) > 3 else "metrics_plot.html"
    R, S = rl["records"], sft["records"]
    rungs = [("sample@1", "sample1_fve"), ("best@1", "best1_fve"),
             ("best@32", "bestN_fve"), ("pooled-nnomp@32", "oracle_fve")]
    data = [(name, amean(S, k), amean(R, k)) for name, k in rungs]
    gt = amean(R, "gt_fve")  # checkpoint-independent reference

    W, H = 720, 420
    padL, padR, padB, padT = 56, 30, 64, 46
    x0, x1, y0, y1 = padL, W - padR, H - padB, padT
    ymax = max(max(s, r) for _, s, r in data) * 1.16
    Y = lambda v: y1 + (y0 - y1) * (1 - v / ymax)
    n = len(data); gw = (x1 - x0) / n; bw = gw * 0.30
    SFT, RL = "#6ba3e0", "#f28e2b"
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         'font-family="ui-sans-serif,system-ui,sans-serif">']
    # gridlines
    for t in range(6):
        v = ymax * t / 5; yy = Y(v)
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" fill="var(--muted)" '
                 f'font-size="11">{v:.2f}</text>')
    # GT reference rule
    gy = Y(gt)
    s.append(f'<line x1="{x0}" x2="{x1}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="#59a14f" '
             f'stroke-width="1.6" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{x1}" y="{gy-6:.1f}" text-anchor="end" fill="#59a14f" font-size="11">'
             f'GT-text ceiling {gt:.3f}</text>')
    # bars
    for i, (name, sv, rv) in enumerate(data):
        cx = x0 + gw * (i + 0.5)
        for j, (val, col) in enumerate([(sv, SFT), (rv, RL)]):
            bx = cx + (j - 1) * bw - bw * 0.08 + (bw * 0.16 if j else 0)
            bx = cx - bw - 2 + j * (bw + 4)
            by = Y(val)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y0-by:.1f}" '
                     f'rx="4" fill="{col}"><title>{name} {"RL" if j else "SFT"}: {val:.3f}</title></rect>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{by-5:.1f}" text-anchor="middle" '
                     f'fill="var(--ink)" font-size="11" font-weight="600">{val:.3f}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y0+18}" text-anchor="middle" fill="var(--ink)" '
                 f'font-size="12">{name}</text>')
        # delta below
        s.append(f'<text x="{cx:.1f}" y="{y0+34}" text-anchor="middle" fill="var(--up)" '
                 f'font-size="11">+{rv-sv:.3f}</text>')
    # legend
    for k, (lab, col) in enumerate([("SFT (pre-RL)", SFT), ("RL (iter 600)", RL)]):
        lx = x0 + k * 150
        s.append(f'<rect x="{lx}" y="18" width="12" height="12" rx="3" fill="{col}"/>')
        s.append(f'<text x="{lx+18}" y="28" fill="var(--ink)" font-size="12">{lab}</text>')
    s.append(f'<text x="{x0-44}" y="{(y0+y1)/2:.0f}" text-anchor="middle" fill="var(--muted)" '
             f'font-size="12" transform="rotate(-90 {x0-44} {(y0+y1)/2:.0f})">joint FVE (T=1)</text>')
    s.append('</svg>')

    tbl = "".join(f"<tr><td>{n}</td><td>{sv:.3f}</td><td>{rv:.3f}</td>"
                  f"<td class='up'>+{rv-sv:.3f}</td><td>{(rv/sv-1)*100:+.0f}%</td></tr>"
                  for n, sv, rv in data)
    doc = f"""<!doctype html><meta charset=utf-8><title>iolens.final — all ladder metrics, SFT vs RL</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --grid:#ececec;
 --line:#e5e3dd; --up:#3d8b40; }}
body {{ font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--ink); margin:0; padding:28px 30px; }}
h1 {{ font-size:18px; }} .sub {{ color:var(--muted); font-size:13px; margin-bottom:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; max-width:760px; }}
table {{ border-collapse:collapse; font-size:13px; margin-top:14px; width:100%; }}
td,th {{ padding:5px 10px; text-align:right; border-bottom:1px solid var(--grid); }}
td:first-child,th:first-child {{ text-align:left; }} .up {{ color:var(--up); }}
</style>
<h1>iolens.final — all ladder metrics, SFT → RL</h1>
<div class="sub">n={len(R)} activations, 11 layers balanced · all sampled at T=1 (32 rollouts) · RL beats SFT on every rung</div>
<div class="card">{''.join(s)}
<table><tr><th>rung</th><th>SFT</th><th>RL</th><th>Δ</th><th>rel</th></tr>{tbl}</table></div>
"""
    Path(out).write_text(doc)
    print(f"wrote {out}")
    for name, sv, rv in data:
        print(f"  {name:18} SFT {sv:.3f}  RL {rv:.3f}  (+{rv-sv:.3f})")


if __name__ == "__main__":
    main()
