"""Clean light-mode matplotlib figures for the joint-RL iolens checkpoint report.

Reads the oracle ladders (ladder_RL.json / ladder_SFT.json) for exact per-layer and
aggregate best@k numbers; pooled@N + marginal + AO-dict values are the documented
audit results. Outputs 5 PNGs to <run>/report_plots/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUN = Path("artifacts/sc/rl_runs/iolens-rl-final-ddp600")
OUT = RUN / "report_plots"
OUT.mkdir(exist_ok=True)

# ---- palette (light mode) -------------------------------------------------
INK, MUTED, GRID = "#1a1a2e", "#6b7280", "#e5e7eb"
BLUE, BLUE_L = "#2563eb", "#93c5fd"      # RL primary / light
GRAY = "#9ca3af"                          # SFT / reference
CEIL = "#dc2626"                          # ceiling
GREEN = "#059669"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def agg(path):
    recs = json.load(open(path))["records"]
    keys = ["sample1_fve", "best1_fve", "bestN_fve", "oracle_fve", "gt_fve"]
    a = {k: 100 * sum(x[k] for x in recs) / len(recs) for k in keys}
    byL = {}
    for x in recs:
        byL.setdefault(x["layer"], []).append(x)
    a["per_layer"] = {L: {k: 100 * sum(x[k] for x in xs) / len(xs) for k in keys}
                      for L, xs in sorted(byL.items())}
    return a


RL = agg(RUN / "ladder_RL.json")
SFT = agg(RUN / "ladder_SFT.json")


def finish(ax, ax_pct="y"):
    ax.grid(axis="y" if ax_pct == "y" else "x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    (ax.yaxis if ax_pct == "y" else ax.xaxis).set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


# ==== 1. reconstruction ladder (horizontal bars) ===========================
fig, ax = plt.subplots(figsize=(8.2, 4.4))
rows = [
    ("AO-dict, foreign-only", 1.4, GRAY),
    ("GT-text (source crop)", RL["gt_fve"], GRAY),
    ("sample@1 (one draw)", RL["sample1_fve"], BLUE_L),
    ("best@1 (shipped)", RL["best1_fve"], BLUE),
    ("best@32", RL["bestN_fve"], BLUE_L),
    ("pooled-nnomp@32", RL["oracle_fve"], GREEN),
    ("AO-dict superset", 22.2, GREEN),
]
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
cols = [r[2] for r in rows]
y = range(len(rows))
ax.barh(y, vals, color=cols, height=0.68, zorder=3)
ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.invert_yaxis()
for yi, v in zip(y, vals):
    ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", ha="left", fontsize=10, color=INK)
ax.axvline(22.2, color=CEIL, ls="--", lw=1.2, zorder=2)
ax.text(22.2, len(rows) - 0.35, " ~22% target/AR ceiling", color=CEIL, fontsize=9, va="top")
ax.set_xlim(0, 26)
finish(ax, "x")
ax.set_xlabel("whitened joint-FVE")
ax.set_title("Reconstruction ladder — joint-RL checkpoint (132-item eval, 32 rollouts)",
             fontsize=12, fontweight="bold", color=INK, pad=10)
fig.tight_layout()
fig.savefig(OUT / "1_ladder.png", dpi=160)
plt.close(fig)

# ==== 2. pooled@N curve ====================================================
N = [1, 2, 4, 8, 16, 32]
pooled = [15.9, 17.3, 18.6, 19.6, 20.8, 21.8]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(N, pooled, "-o", color=BLUE, lw=2.2, ms=7, zorder=3, label="pooled-nnomp@N (RL)")
for x, v in zip(N, pooled):
    ax.text(x, v + 0.35, f"{v:.1f}", ha="center", fontsize=9, color=INK)
ax.axhline(22.2, color=CEIL, ls="--", lw=1.2, label="AO-dict ceiling 22.2%")
ax.axhline(RL["best1_fve"], color=GRAY, ls=":", lw=1.4,
           label=f"best@1 {RL['best1_fve']:.1f}%")
ax.set_xscale("log", base=2)
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlabel("N (rollouts pooled)")
ax.set_ylabel("joint-FVE")
ax.set_ylim(14, 24)
finish(ax, "y")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Pooling more rollouts — log-shaped, no plateau below the ceiling",
             fontsize=12, fontweight="bold", color=INK, pad=10)
fig.tight_layout()
fig.savefig(OUT / "2_pooled_curve.png", dpi=160)
plt.close(fig)

# ==== 3. marginal FVE per oracle-picked bullet =============================
marg = [19.7, 1.1, 0.4, 0.2]
cum = [sum(marg[:i + 1]) for i in range(4)]
x = [1, 2, 3, 4]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.bar(x, marg, color=BLUE_L, width=0.6, zorder=3, label="marginal gain of bullet k")
ax.plot(x, cum, "-o", color=BLUE, lw=2, ms=6, zorder=4, label="cumulative joint-FVE")
for xi, m, c in zip(x, marg, cum):
    ax.text(xi, m + 0.5, f"+{m:.1f}", ha="center", fontsize=9, color=INK)
    ax.text(xi, c + 0.6, f"{c:.1f}%", ha="center", fontsize=9, color=BLUE, fontweight="bold")
ax.set_xticks(x)
ax.set_xlabel("oracle-picked bullet #")
ax.set_ylabel("joint-FVE")
ax.set_ylim(0, 24)
finish(ax, "y")
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.set_title("Bullet 1 carries ~93% — the target is ~1-dimensional",
             fontsize=12, fontweight="bold", color=INK, pad=10)
fig.tight_layout()
fig.savefig(OUT / "3_marginal_bullet.png", dpi=160)
plt.close(fig)

# ==== 4. per-layer best@1 (RL) with best@32 headroom =======================
layers = list(RL["per_layer"].keys())
b1 = [RL["per_layer"][L]["best1_fve"] for L in layers]
bN = [RL["per_layer"][L]["bestN_fve"] for L in layers]
gt = [RL["per_layer"][L]["gt_fve"] for L in layers]
fig, ax = plt.subplots(figsize=(9.2, 4.6))
xi = range(len(layers))
ax.bar([i - 0.2 for i in xi], b1, width=0.4, color=BLUE, zorder=3, label="best@1")
ax.bar([i + 0.2 for i in xi], bN, width=0.4, color=BLUE_L, zorder=3, label="best@32")
ax.plot(list(xi), gt, "D", color=GRAY, ms=5, zorder=4, label="GT-text")
ax.axhline(RL["best1_fve"], color=INK, ls=":", lw=1, alpha=0.5)
ax.text(len(layers) - 1, RL["best1_fve"] + 0.4, f"mean best@1 {RL['best1_fve']:.1f}%",
        ha="right", fontsize=8, color=INK)
ax.set_xticks(list(xi))
ax.set_xticklabels([f"L{L}" for L in layers])
ax.set_xlabel("injected layer")
ax.set_ylabel("joint-FVE")
finish(ax, "y")
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
ax.set_title("Per-layer reconstruction — deep layers (L52, L60) far above shallow",
             fontsize=12, fontweight="bold", color=INK, pad=10)
fig.tight_layout()
fig.savefig(OUT / "4_per_layer.png", dpi=160)
plt.close(fig)

# ==== 5. what RL bought — SFT vs RL =========================================
metrics = [("sample@1", "sample1_fve"), ("best@1", "best1_fve"),
           ("best@32", "bestN_fve"), ("pooled@32", "oracle_fve")]
sft_v = [SFT[k] for _, k in metrics]
rl_v = [RL[k] for _, k in metrics]
fig, ax = plt.subplots(figsize=(7.8, 4.6))
xi = range(len(metrics))
ax.bar([i - 0.2 for i in xi], sft_v, width=0.4, color=GRAY, zorder=3, label="SFT (warmstart)")
ax.bar([i + 0.2 for i in xi], rl_v, width=0.4, color=BLUE, zorder=3, label="joint-RL")
for i, (s, r) in enumerate(zip(sft_v, rl_v)):
    ax.text(i - 0.2, s + 0.3, f"{s:.1f}", ha="center", fontsize=8, color=INK)
    ax.text(i + 0.2, r + 0.3, f"{r:.1f}", ha="center", fontsize=8, color=BLUE, fontweight="bold")
ax.set_xticks(list(xi))
ax.set_xticklabels([m[0] for m in metrics])
ax.set_ylabel("joint-FVE")
ax.set_ylim(0, 24)
finish(ax, "y")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("What 600 steps of RL bought (best@1 11.9→15.6, +32%)",
             fontsize=12, fontweight="bold", color=INK, pad=10)
fig.tight_layout()
fig.savefig(OUT / "5_sft_vs_rl.png", dpi=160)
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
