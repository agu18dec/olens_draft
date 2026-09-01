"""Aesthetic before/after (SFT vs RL) comparison site for the iolens.final ladder.

Consumes ladder_RL.json + ladder_SFT.json (checks/eval_ceiling.py --mode oracle
--per-layer N, enriched records). Emits ONE self-contained HTML:
  - aggregate stat tiles + per-layer line graphs (best@1 SFT vs RL; the RL best-of ladder)
  - one card per activation: the injection prompt (㈜ marker highlighted), the source
    crop it was sampled from, and SFT-vs-RL side by side across the ladder rungs
    (sample@1 / best@1 / best@32 / pooled-nnomp@32 + the oracle-picked bullets), with
    every rollout + its FVE expandable. Sortable/filterable by layer and RL−SFT delta.

Usage: python make_comparison_site.py ladder_RL.json ladder_SFT.json out.html
"""
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def per_layer(recs, key):
    agg = defaultdict(list)
    for r in recs:
        agg[r["layer"]].append(r[key])
    return {ly: sum(v) / len(v) for ly, v in sorted(agg.items())}


def line_svg(layers, series, colors, dashed, W=780, H=360, title=""):
    padL, padR, padB, padT = 64, 150, 56, 30
    ymax = max((max(s.values()) for s in series.values()), default=0.3) * 1.12 or 0.3
    x0, x1, y0, y1 = padL, W - padR, H - padB, padT
    X = lambda i: x0 + (x1 - x0) * i / max(1, len(layers) - 1)
    Y = lambda v: y1 + (y0 - y1) * (1 - v / ymax)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">']
    if title:
        s.append(f'<text x="{padL}" y="18" fill="var(--muted)" font-size="12" '
                 f'font-weight="600">{title}</text>')
    for t in range(6):
        v = ymax * t / 5; yy = Y(v)
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" fill="var(--muted)" '
                 f'font-size="10">{v*100:.0f}%</text>')
    for i, ly in enumerate(layers):
        s.append(f'<text x="{X(i):.1f}" y="{y0+16}" text-anchor="middle" fill="var(--muted)" '
                 f'font-size="10">{ly}</text>')
    # axis titles
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-6}" text-anchor="middle" fill="var(--muted)" '
             f'font-size="12">layer</text>')
    s.append(f'<text x="16" y="{(y0+y1)/2:.0f}" text-anchor="middle" fill="var(--muted)" '
             f'font-size="12" transform="rotate(-90 16 {(y0+y1)/2:.0f})">FVE (%)</text>')
    for name, sr in series.items():
        pts = [(X(i), Y(sr[ly])) for i, ly in enumerate(layers)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = ' stroke-dasharray="5 4"' if name in dashed else ""
        s.append(f'<path d="{d}" fill="none" stroke="{colors[name]}" stroke-width="2.6"{da}/>')
        for (x, y), ly in zip(pts, layers):
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{colors[name]}">'
                     f'<title>{name} · L{ly}: {sr[ly]*100:.1f}%</title></circle>')
    for k, name in enumerate(series):
        yy = padT + 6 + k * 20
        da = ' stroke-dasharray="5 4"' if name in dashed else ""
        s.append(f'<line x1="{x1+14}" x2="{x1+32}" y1="{yy}" y2="{yy}" stroke="{colors[name]}" '
                 f'stroke-width="3"{da}/>')
        s.append(f'<text x="{x1+38}" y="{yy+4}" fill="var(--ink)" font-size="11">{name}</text>')
    s.append('</svg>')
    return "".join(s)


def esc(t):
    return html.escape(t or "")


def prompt_html(p):
    # highlight the CJK injection marker ㈜ (the activation slot)
    marker = "㈜"
    return esc(p).replace(esc(marker),
                          f'<span class="marker" title="activation injected here">{esc(marker)}</span>')


def bullets_html(txt, terms=None):
    lines = [l for l in (txt or "").split("\n") if l.strip()]
    return "".join(f'<div class="bul">{mark(l, terms)}</div>' for l in lines) or '<div class="bul dim">—</div>'


def mark(text, terms):
    """Escape, then <mark> any Opus-flagged interesting terms (case-insensitive)."""
    out = esc(text)
    for t in sorted({t for t in (terms or []) if t.strip()}, key=len, reverse=True):
        out = re.sub(re.escape(esc(t)), lambda m: f"<mark>{m.group(0)}</mark>", out, flags=re.I)
    return out


def side(rec, which, idx, hl):
    terms = (hl or {}).get("terms", [])
    note = (hl or {}).get("note", "")
    cid = f"c{idx}{which}"
    picks = "".join(
        f'<div class="pick"><span class="coef">{b["coeff"]:+.2f}</span> {mark(b["text"], terms)}</div>'
        for b in rec.get("oracle_bullets", [])) or '<div class="bul dim">—</div>'
    rolls = "".join(
        f'<div class="roll"><span class="rf">{r["fve"]:.3f}</span>{bullets_html(r["text"], terms)}</div>'
        for r in sorted(rec.get("rollouts", []), key=lambda x: -x["fve"]))
    bestfve = rec['rollouts'][rec['best_rollout_idx']]['fve']
    # panels keyed to each rung; clicking a rung shows its readout
    panels = {
        "sample@1": f'<div class="lbl">one T=1 draw</div><div class="readout">{bullets_html(rec.get("sample1_text",""), terms)}</div>',
        "best@1": f'<div class="lbl">best@1 = mean FVE over the 32 single T=1 samples (no single artifact); representative draw below</div><div class="readout">{bullets_html(rec.get("sample1_text",""), terms)}</div>',
        "best@32": f'<div class="lbl">best single rollout of 32 (fve {bestfve:.3f})</div><div class="readout">{bullets_html(rec["best_rollout_text"], terms)}</div>',
        "pooled@32": f'<div class="lbl">oracle-picked bullets — nnomp mixes across rollouts (→ {rec["oracle_fve"]:.3f})</div><div class="readout picks">{picks}</div>',
    }
    rungs = "".join(
        f'<button class="rung{" hi" if name=="pooled@32" else ""}{" active" if name=="pooled@32" else ""}" '
        f'onclick="pick(\'{cid}\',\'{name}\')" data-cid="{cid}">{name} <b>{rec[key]:.3f}</b></button>'
        for name, key in [("sample@1", "sample1_fve"), ("best@1", "best1_fve"),
                          ("best@32", "bestN_fve"), ("pooled@32", "oracle_fve")])
    panelhtml = "".join(
        f'<div class="panel" data-cid="{cid}" data-name="{name}" '
        f'style="{"" if name=="pooled@32" else "display:none"}">{html_}</div>'
        for name, html_ in panels.items())
    notehtml = f'<div class="opnote">🔎 {esc(note)}</div>' if note else ""
    return f"""
    <div class="col {which}">
      <div class="colhead">{'RL (iter 600)' if which=='rl' else 'SFT (pre-RL)'}</div>
      {notehtml}
      <div class="rungs">{rungs}</div>
      {panelhtml}
      <details><summary>all {len(rec['rollouts'])} rollouts (sorted by fve)</summary>
        <div class="rolls">{rolls}</div></details>
    </div>"""


def main():
    rl = json.loads(Path(sys.argv[1]).read_text())
    sft = json.loads(Path(sys.argv[2]).read_text())
    out = sys.argv[3] if len(sys.argv) > 3 else "comparison_site.html"
    # optional Opus highlights: {"<idx>": {"rl": {"terms":[...],"note":".."}, "sft": {...}}}
    HL = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else {}
    R, S = rl["records"], sft["records"]
    assert len(R) == len(S), "record count mismatch"
    layers = sorted({r["layer"] for r in R})

    # graphs
    c1 = {"RL best@1": "#f28e2b", "SFT best@1": "#4e79a7", "GT-text": "#59a14f"}
    g1 = line_svg(layers, {"RL best@1": per_layer(R, "best1_fve"),
                           "SFT best@1": per_layer(S, "best1_fve"),
                           "GT-text": per_layer(R, "gt_fve")},
                  c1, {"GT-text"}, title="before/after — best@1 (T=1) per layer")
    c2 = {"sample@1": "#9c755f", "best@1": "#4e79a7", "best@32": "#f28e2b",
          "pooled@32": "#b07aa1", "GT-text": "#59a14f"}
    g2 = line_svg(layers, {"sample@1": per_layer(R, "sample1_fve"),
                           "best@1": per_layer(R, "best1_fve"),
                           "best@32": per_layer(R, "bestN_fve"),
                           "pooled@32": per_layer(R, "oracle_fve"),
                           "GT-text": per_layer(R, "gt_fve")},
                  c2, {"GT-text"}, title="RL best-of ladder per layer")

    def amean(recs, k):
        return sum(r[k] for r in recs) / len(recs)
    tiles = [
        ("best@1 (T=1)", amean(S, "best1_fve"), amean(R, "best1_fve")),
        ("best@32", amean(S, "bestN_fve"), amean(R, "bestN_fve")),
        ("pooled-nnomp@32", amean(S, "oracle_fve"), amean(R, "oracle_fve")),
        ("GT-text ceiling", amean(S, "gt_fve"), amean(R, "gt_fve")),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tl">{n}</div>'
        f'<div class="tv"><span class="sft">{s:.3f}</span> → <span class="rl">{r:.3f}</span></div>'
        f'<div class="td {"up" if r>=s else "dn"}">{r-s:+.3f}</div></div>'
        for n, s, r in tiles)

    # cards data (matched by index)
    cards = []
    for i, (rr, sr) in enumerate(zip(R, S)):
        d = rr["best1_fve"] - sr["best1_fve"]
        cards.append({
            "i": i, "layer": rr["layer"], "row_id": rr["row_id"], "delta": d,
            "rl_b1": rr["best1_fve"], "sft_b1": sr["best1_fve"],
            "html": f"""
      <div class="card" data-layer="{rr['layer']}" data-delta="{d:.4f}" data-rl="{rr['best1_fve']:.4f}">
        <div class="chead" onclick="this.parentNode.classList.toggle('open')">
          <span class="ly">L{rr['layer']}</span>
          <span class="rid">#{rr['row_id']}</span>
          <span class="delta {'up' if d>=0 else 'dn'}">Δbest@1 {d:+.3f}</span>
          <span class="mini">SFT {sr['best1_fve']:.3f} → RL {rr['best1_fve']:.3f}</span>
          <span class="chev">▾</span>
        </div>
        <div class="cbody">
          <div class="ctx">
            <div class="lbl">activation prompt <span class="dim">(㈜ = injected activation)</span></div>
            <div class="prompt">{prompt_html(rr['prompt_text'])}</div>
            <div class="lbl">sampled from (the crop whose preceding residual IS this activation)</div>
            <div class="source">{esc(rr['source_text'])}</div>
          </div>
          <div class="cols">{side(sr,'sft',i,HL.get(str(i),{}).get('sft'))}{side(rr,'rl',i,HL.get(str(i),{}).get('rl'))}</div>
        </div>
      </div>"""})
    cards_html = "".join(c["html"] for c in cards)

    doc = f"""<!doctype html><meta charset=utf-8>
<title>iolens.final — RL vs SFT activation-readout comparison</title>
<style>
:root {{ --bg:#f7f7f5; --card:#ffffff; --card2:#f3f2ef; --ink:#1a1a1a; --muted:#6b6b6b;
 --grid:#ececec; --line:#e5e3dd; --rl:#d97a1a; --sft:#3f7fc4; --up:#3d8b40; --dn:#c0392b;
 --marker:#9350a8; }}
* {{ box-sizing:border-box; }}
body {{ font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; background:var(--bg);
 color:var(--ink); margin:0; padding:28px 30px 80px; line-height:1.5; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 16px; min-width:170px; }}
.tl {{ color:var(--muted); font-size:12px; }} .tv {{ font-size:17px; font-weight:600; margin-top:3px; }}
.tv .sft {{ color:var(--sft); }} .tv .rl {{ color:var(--rl); }}
.td {{ font-size:12px; margin-top:2px; }} .td.up {{ color:var(--up); }} .td.dn {{ color:var(--dn); }}
.graphs {{ display:flex; gap:18px; flex-wrap:wrap; margin-bottom:22px; }}
.gcard {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; flex:1; min-width:380px; }}
.controls {{ position:sticky; top:0; background:var(--bg); padding:10px 0; display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:5; border-bottom:1px solid var(--line); margin-bottom:8px; }}
.controls label {{ font-size:12px; color:var(--muted); }}
select, input {{ background:var(--card2); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:5px 9px; font-size:13px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; margin:9px 0; overflow:hidden; }}
.chead {{ display:flex; gap:12px; align-items:center; padding:11px 15px; cursor:pointer; user-select:none; }}
.chead:hover {{ background:var(--card2); }}
.ly {{ font-weight:700; color:var(--rl); }} .rid {{ color:var(--muted); font-size:12px; }}
.delta {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:20px; background:var(--card2); }}
.delta.up {{ color:var(--up); }} .delta.dn {{ color:var(--dn); }}
.mini {{ color:var(--muted); font-size:12px; margin-left:auto; }}
.chev {{ color:var(--muted); transition:transform .15s; }} .card.open .chev {{ transform:rotate(180deg); }}
.cbody {{ display:none; padding:4px 15px 16px; }} .card.open .cbody {{ display:block; }}
.ctx {{ margin-bottom:12px; }}
.lbl {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:12px 0 4px; }}
.dim {{ color:var(--muted); text-transform:none; letter-spacing:0; }}
.prompt {{ background:var(--card2); border-radius:8px; padding:10px 12px; font-size:12.5px; white-space:pre-wrap; max-height:150px; overflow:auto; font-family:ui-monospace,Menlo,monospace; }}
.source {{ background:var(--card2); border-radius:8px; padding:10px 12px; font-size:13px; white-space:pre-wrap; border-left:3px solid var(--marker); }}
.marker {{ background:var(--marker); color:#fff; padding:0 4px; border-radius:4px; font-weight:700; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.col {{ background:var(--card2); border-radius:10px; padding:12px; }}
.col.rl {{ outline:1px solid rgba(242,142,43,.35); }} .col.sft {{ outline:1px solid rgba(107,163,224,.28); }}
.colhead {{ font-weight:700; font-size:13px; margin-bottom:8px; }} .tag {{ font-weight:400; color:var(--muted); font-size:11px; }}
.rungs {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }}
.rung {{ font-size:11px; color:var(--muted); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:4px 8px; cursor:pointer; font-family:inherit; }}
.rung:hover {{ border-color:var(--marker); }}
.rung b {{ color:var(--ink); }} .rung.hi {{ border-color:var(--marker); }} .rung.hi b {{ color:var(--marker); }}
.rung.active {{ background:var(--marker); border-color:var(--marker); color:#fff; }} .rung.active b {{ color:#fff; }}
.opnote {{ font-size:12px; color:var(--marker); background:rgba(147,80,168,.08); border:1px solid rgba(147,80,168,.25); border-radius:6px; padding:5px 9px; margin:6px 0; }}
mark {{ background:#ffe58a; color:inherit; padding:0 2px; border-radius:3px; }}
.readout {{ background:var(--card); border-radius:8px; padding:8px 10px; font-size:13px; }}
.bul {{ padding:2px 0; }} .bul.dim {{ color:var(--muted); }}
.picks .pick {{ padding:3px 0; font-size:13px; }} .coef {{ color:var(--marker); font-variant-numeric:tabular-nums; font-size:11px; margin-right:6px; }}
details {{ margin-top:8px; }} summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
.rolls {{ max-height:280px; overflow:auto; margin-top:6px; }}
.roll {{ border-top:1px solid var(--line); padding:6px 0; font-size:12.5px; }}
.rf {{ display:inline-block; color:var(--rl); font-variant-numeric:tabular-nums; font-size:11px; margin-right:8px; }}
.col.rl {{ outline:1px solid rgba(217,122,26,.30); }} .col.sft {{ outline:1px solid rgba(63,127,196,.28); }}
</style>
<h1>iolens.final — activation readouts, SFT → RL</h1>
<div class="sub">{len(R)} activations, {len(layers)} layers · each: 32 rollouts @ T=1 · best-of ladder + oracle nnomp selection · RL LoRA <code>{esc(Path(rl['lora']).name)}</code></div>
<div class="tiles">{tile_html}</div>
<div class="graphs"><div class="gcard">{g1}</div><div class="gcard">{g2}</div></div>
<div class="controls">
  <label>layer <select id="fl" onchange="render()"><option value="">all</option>
   {''.join(f'<option>L{ly}</option>' for ly in layers)}</select></label>
  <label>sort <select id="so" onchange="render()">
   <option value="delta">RL−SFT best@1 (desc)</option>
   <option value="deltaA">RL−SFT best@1 (asc)</option>
   <option value="layer">layer</option>
   <option value="rl">RL best@1 (desc)</option></select></label>
  <span id="cnt" class="mini"></span>
</div>
<div id="cards">{cards_html}</div>
<script>
function pick(cid, name) {{
  document.querySelectorAll('.panel[data-cid="'+cid+'"]').forEach(p=>
    p.style.display = p.dataset.name===name ? '' : 'none');
  document.querySelectorAll('.rung[data-cid="'+cid+'"]').forEach(b=>
    b.classList.toggle('active', b.textContent.trim().startsWith(name)));
}}
function render() {{
  const fl=document.getElementById('fl').value, so=document.getElementById('so').value;
  const cards=[...document.querySelectorAll('.card')];
  let vis=cards.filter(c=>!fl||('L'+c.dataset.layer)===fl);
  const cmp={{delta:(a,b)=>b.dataset.delta-a.dataset.delta,deltaA:(a,b)=>a.dataset.delta-b.dataset.delta,
    layer:(a,b)=>a.dataset.layer-b.dataset.layer,rl:(a,b)=>b.dataset.rl-a.dataset.rl}}[so];
  vis.sort(cmp);
  const box=document.getElementById('cards'); cards.forEach(c=>c.style.display='none');
  vis.forEach(c=>{{c.style.display='';box.appendChild(c);}});
  document.getElementById('cnt').textContent=vis.length+' shown';
}}
render();
</script>
"""
    Path(out).write_text(doc)
    print(f"wrote {out}  ({len(doc)//1024} KB, {len(R)} cards)")


if __name__ == "__main__":
    main()
