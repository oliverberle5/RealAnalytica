"""One-off chart: distribution of current agents' likelihood-to-leave
percentages, with the 4 risk-tier bands overlaid. Reads the already-
scored current_agent_risk_scores.csv (run score_agents.py first if it's
stale) -- does not retrain anything itself. 4-tier system (Low/Medium/
High/Extra High), replacing the original 5-tier quintile system
2026-07-24.

Palette: the same pre-validated blue ordinal ramp used in
generate_report.py (dataviz skill references/palette.md, steps
250/450/550/700), light-to-dark = Low-to-Extra-High risk.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIER_COLORS = ["#86b6ef", "#5598e7", "#1c5cab", "#0d366b"]
TIER_LABELS = ["Low", "Medium", "High", "Extra High"]
NAVY = "#0f2540"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
PAGE_BG = "#f7fafd"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def main():
    df = pd.read_csv("current_agent_risk_scores.csv")
    pct_all = df["percent_chance"].to_numpy() * 100  # display in percentage points
    cut_points = sorted(df.groupby("risk_tier")["percent_chance"].min().to_numpy()[1:] * 100)
    snapshot_date = df["snapshot_date"].iloc[0]
    n_agents = len(df)

    # Long right tail: a handful of agents score far above the rest, which
    # would otherwise stretch the axis and leave most of the chart empty.
    # Cap the axis at 8% (covers ~99% of agents) and call out the rest
    # explicitly rather than silently dropping them.
    X_CAP = 8.0
    n_above_cap = int((pct_all > X_CAP).sum())
    pct = pct_all[pct_all <= X_CAP]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.patch.set_facecolor(PAGE_BG)

    bin_edges = np.linspace(0, X_CAP, 33)
    counts, edges = np.histogram(pct, bins=bin_edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    boundaries = [0] + cut_points + [X_CAP + 1]
    bar_colors = []
    for c in bin_centers:
        q = next(i for i in range(4) if boundaries[i] <= c < boundaries[i + 1])
        bar_colors.append(TIER_COLORS[q])

    ax.bar(bin_centers, counts, width=(edges[1] - edges[0]) * 0.95, color=bar_colors, zorder=3)

    for cp in cut_points:
        if cp <= X_CAP:
            ax.axvline(cp, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)

    ax.set_xlim(0, X_CAP)
    ax.set_ylim(top=counts.max() * 1.12)
    ax.set_xlabel("Predicted 3-month departure chance")
    ax.set_ylabel("Number of agents")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLORS]
    ax.legend(legend_handles, TIER_LABELS, title="Risk tier", loc="upper right",
               frameon=False, fontsize=9.5, title_fontsize=9.5)

    fig.suptitle("Current Agent Risk Score Distribution", fontsize=16, fontweight="bold",
                  color=NAVY, x=0.09, ha="left", y=0.975)
    fig.text(0.09, 0.925,
              f"{n_agents:,} currently active agents  |  snapshot {snapshot_date}",
              fontsize=10.5, color=INK_SECONDARY, ha="left")

    fig.text(0.09, 0.045,
              f"Axis capped at {X_CAP:.0f}% for readability — {n_above_cap} agents "
              f"({n_above_cap / n_agents:.1%}) score above that, up to {pct_all.max():.1f}%.",
              fontsize=8.5, color=INK_MUTED, ha="left")
    fig.text(0.09, 0.015,
              "Agent Likelihood-to-Leave Model  |  RIAR", fontsize=8.5, color=INK_MUTED, ha="left")

    fig.subplots_adjust(left=0.09, right=0.96, top=0.87, bottom=0.13)

    out_png = "risk_score_distribution.png"
    fig.savefig(out_png, dpi=200)
    print(f"[chart] saved {out_png}")


if __name__ == "__main__":
    main()
