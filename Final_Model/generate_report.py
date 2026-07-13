"""One-off report generator: builds a white/blue-themed PDF summarizing the
final CatBoost model's accuracy, features, and calibration for a
non-technical/business audience. Not part of the modeling pipeline --
regenerate whenever the model or numbers change meaningfully.

Palette: a single blue hue at three lightness steps (light/medium/dark),
taken from the dataviz skill's pre-validated ordinal ramp (steps 250/450/700
in references/palette.md) so the report is monochrome-blue-safe without
re-deriving contrast ratios by eye.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, average_precision_score

from data import load_clean, RAW_FEATURE_COLUMNS
from crossval import (
    MODELS, MONOTONE, TREE_FEATURES, _forward_folds, _lift_at_top_decile,
    _fit_catboost,
)
from calibrate import (
    _collect_oof, compute_recent_bias_correction, fit_production_calibrator,
    score_to_output, QUINTILE_LABELS,
)
from catboost import CatBoostClassifier

# ---------------------------------------------------------------- palette --
BLUE_LIGHT = "#86b6ef"   # ramp step 250 -- logistic (baseline, de-emphasized)
BLUE_MED = "#2a78d6"     # ramp step 450 -- XGBoost
BLUE_DARK = "#0d366b"    # ramp step 700 -- CatBoost (the winner)
NAVY = "#0f2540"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
PAGE_BG = "#f7fafd"
CARD_BG = "#eef4fb"

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

PAGE_SIZE = (8.5, 11)


def _new_page():
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor(PAGE_BG)
    return fig


def _header(fig, title, subtitle=None, page_num=None):
    fig.text(0.08, 0.955, title, fontsize=20, fontweight="bold", color=NAVY, ha="left")
    if subtitle:
        fig.text(0.08, 0.925, subtitle, fontsize=11, color=INK_SECONDARY, ha="left")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.905, 0.905], color=BLUE_MED, linewidth=2, transform=fig.transFigure))
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.055, 0.055], color=GRID, linewidth=0.8, transform=fig.transFigure))
    if page_num:
        fig.text(0.92, 0.025, f"{page_num}", fontsize=9, color=INK_MUTED, ha="right")
    fig.text(0.08, 0.025, "Agent Likelihood-to-Leave Model  |  RIAR", fontsize=9, color=INK_MUTED, ha="left")


def _fit_fontsize(fig, text, max_width_frac, start_size=9.5, min_size=6.5):
    """Shrink fontsize until `text` renders within max_width_frac of figure width."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    fs = start_size
    while fs > min_size:
        t = fig.text(0.5, 1.5, text, fontsize=fs)  # placed above the page, never rendered on the page itself
        fig.canvas.draw()
        width_px = t.get_window_extent(renderer=renderer).width
        t.remove()
        if width_px <= max_width_frac * fig_w_px:
            break
        fs -= 0.5
    return fs


def _stat_card(fig, x, y, w, h, value, label, accent=BLUE_MED):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                           transform=fig.transFigure, facecolor=CARD_BG, edgecolor=GRID, linewidth=1)
    fig.add_artist(card)
    fig.add_artist(plt.Rectangle((x, y), 0.012, h, transform=fig.transFigure, facecolor=accent, edgecolor="none"))
    fig.text(x + w / 2, y + h * 0.60, value, fontsize=19, fontweight="bold", color=NAVY, ha="center", va="center")
    label_fs = _fit_fontsize(fig, label, max_width_frac=w - 0.02)
    fig.text(x + w / 2, y + h * 0.24, label, fontsize=label_fs, color=INK_SECONDARY, ha="center", va="center")


def main():
    print("[report] loading data...")
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())

    n_agents = df["agent_profile_id"].nunique()
    n_rows = len(df)
    n_quarters = df["snapshot_date"].nunique()
    base_rate = df["label_left_3m"].mean()

    print("[report] running 5-fold comparison (logistic / xgboost / catboost)...")
    fold_results = {m: {"auc": [], "prauc": [], "lift": []} for m in MODELS}
    catboost_oof = []  # for calibration reuse
    for train_dates, test_dates in folds:
        train = df[df["snapshot_date"].isin(train_dates)]
        test = df[df["snapshot_date"].isin(test_dates)]
        y_test = test["label_left_3m"].to_numpy()
        for name, fitfn in MODELS.items():
            scores, _ = fitfn(train, test)
            fold_results[name]["auc"].append(roc_auc_score(y_test, scores))
            fold_results[name]["prauc"].append(average_precision_score(y_test, scores))
            fold_results[name]["lift"].append(_lift_at_top_decile(y_test, scores))
            if name == "catboost":
                catboost_oof.append({"scores": scores, "labels": y_test, "dates": test["snapshot_date"].to_numpy()})

    mean_auc = {m: np.mean(fold_results[m]["auc"]) for m in MODELS}
    mean_lift = {m: np.mean(fold_results[m]["lift"]) for m in MODELS}

    print("[report] fitting final CatBoost for feature importance...")
    X_full = df[TREE_FEATURES].astype(float)
    y_full = df["label_left_3m"].to_numpy()
    monotone = [MONOTONE.get(c, 0) for c in TREE_FEATURES]
    final_model = CatBoostClassifier(
        iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=3.0,
        auto_class_weights="Balanced", monotone_constraints=monotone,
        loss_function="Logloss", eval_metric="PRAUC", verbose=False, allow_writing_files=False,
    )
    final_model.fit(X_full, y_full)
    importance = pd.Series(final_model.get_feature_importance(), index=TREE_FEATURES).sort_values(ascending=False)

    print("[report] calibration...")
    platt, cut_points, prod_base_rate = fit_production_calibrator(catboost_oof)
    bias_correction = compute_recent_bias_correction(catboost_oof, recent_fold_idx=4)

    all_scores = np.concatenate([f["scores"] for f in catboost_oof])
    all_labels = np.concatenate([f["labels"] for f in catboost_oof])
    calibrated_pct = platt.predict_proba(all_scores.reshape(-1, 1))[:, 1] * bias_correction
    bucket_idx = np.searchsorted(cut_points, calibrated_pct)
    quintile_predicted, quintile_actual, quintile_n = [], [], []
    for i in range(5):
        mask = bucket_idx == i
        quintile_predicted.append(calibrated_pct[mask].mean())
        quintile_actual.append(all_labels[mask].mean())
        quintile_n.append(mask.sum())

    print("[report] rendering PDF...")
    out_path = "agent_likelihood_to_leave_report.pdf"
    with PdfPages(out_path) as pdf:

        # ---------------------------------------------------------- Page 1: cover
        fig = _new_page()
        fig.add_artist(plt.Rectangle((0, 0.72), 1, 0.28, transform=fig.transFigure, facecolor=NAVY, edgecolor="none"))
        fig.text(0.08, 0.90, "AGENT LIKELIHOOD-TO-LEAVE MODEL", fontsize=15, color="#cfe0f5", ha="left", fontweight="bold")
        fig.text(0.08, 0.85, "Model Performance & Feature Summary", fontsize=24, color="white", ha="left", fontweight="bold")
        fig.text(0.08, 0.79, "Rhode Island (RIAR) real estate agents  |  July 2026", fontsize=12, color="#cfe0f5", ha="left")

        fig.text(0.08, 0.60, "73.7%", fontsize=64, fontweight="bold", color=BLUE_DARK, ha="left")
        fig.text(0.08, 0.535, "AUC (ranking accuracy) — CatBoost, 5-fold time-based validation", fontsize=12, color=INK_SECONDARY, ha="left")

        cards = [
            (f"{n_rows:,}", "agent-quarter records"),
            (f"{n_agents:,}", "unique agents"),
            (f"{n_quarters}", "quarters (2021Q4–2026Q2)"),
            (f"{base_rate:.2%}", "base 3-month departure rate"),
        ]
        card_w, gap = 0.20, 0.017
        start_x = 0.08
        for i, (val, label) in enumerate(cards):
            _stat_card(fig, start_x + i * (card_w + gap), 0.38, card_w, 0.11, val, label)

        fig.text(0.08, 0.28, "What this model does", fontsize=13, fontweight="bold", color=NAVY, ha="left")
        body = (
            "Predicts each agent's percent chance of switching brokerages in the next 3 months,\n"
            "for rank-based (top-N) retention outreach. Output: a calibrated percent chance, a risk\n"
            "multiplier vs. the base rate, and a 5-tier risk label (Low → High)."
        )
        fig.text(0.08, 0.25, body, fontsize=11, color=INK_SECONDARY, ha="left", va="top", linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------- Page 2: model comparison
        fig = _new_page()
        _header(fig, "Model Comparison", "Three models compared on 5 forward-chaining (time-based) folds", page_num=2)

        ax1 = fig.add_axes([0.10, 0.56, 0.80, 0.28])
        names = ["Logistic\nRegression", "XGBoost", "CatBoost"]
        keys = ["logistic", "xgboost", "catboost"]
        colors = [BLUE_LIGHT, BLUE_MED, BLUE_DARK]
        vals = [mean_auc[k] for k in keys]
        bars = ax1.bar(names, vals, color=colors, width=0.55, zorder=3)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold", color=NAVY)
        ax1.set_ylim(0, 0.85)
        ax1.set_ylabel("Mean AUC (5 folds)")
        ax1.spines[["top", "right", "left"]].set_visible(False)
        ax1.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax1.set_axisbelow(True)
        ax1.tick_params(left=False)
        ax1.set_title("Mean ranking accuracy by model", fontsize=11, color=INK_SECONDARY, loc="left")

        ax2 = fig.add_axes([0.10, 0.24, 0.80, 0.24])
        fold_x = np.arange(1, 6)
        for k, c, lab in zip(keys, colors, ["Logistic", "XGBoost", "CatBoost"]):
            ax2.plot(fold_x, fold_results[k]["auc"], marker="o", markersize=5, linewidth=2, color=c, label=lab)
        ax2.set_xticks(fold_x)
        ax2.set_xlabel("Fold (each = 2 held-out quarters, never seen in training)")
        ax2.set_ylabel("AUC")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax2.set_axisbelow(True)
        ax2.legend(loc="lower right", frameon=False, fontsize=9)
        ax2.set_title("Per-fold stability — no single-fold collapse", fontsize=11, color=INK_SECONDARY, loc="left")

        fig.text(0.10, 0.145, "Why CatBoost:", fontsize=10.5, fontweight="bold", color=NAVY, ha="left")
        fig.text(0.10, 0.115,
                  "Wins 4 of 5 folds, best mean AUC and PR-AUC, and lift-at-top-decile of "
                  f"{mean_lift['catboost']:.1f}x the base rate\n(vs. {mean_lift['xgboost']:.1f}x for XGBoost). "
                  "The speed gap (~7s vs ~1.4s per fold) doesn't matter at a quarterly retrain cadence.\n"
                  "Logistic regression is kept only as an interpretability gut-check, not for production scoring.",
                  fontsize=9.5, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------ Page 3: feature importance
        fig = _new_page()
        _header(fig, "What Drives the Prediction", "Top 15 features by CatBoost importance (of ~95 total)", page_num=3)

        top15 = importance.head(15).iloc[::-1]
        pretty_names = {
            "office_exit_rate_12m": "Office peer exit rate (12mo)",
            "tenure_current_office_months": "Tenure at current office",
            "new_listings_12m": "New listings taken (12mo)",
            "months_since_last_closing": "Months since last closing",
            "career_months": "Total career length",
            "list_share_12m": "Share of business that's listings",
            "price_min_12m": "Lowest sale price handled (12mo)",
            "share_of_office_volume_12m": "Share of office's total volume",
            "units_prior_12m": "Prior-year unit count",
            "volume_prior_12m": "Prior-year dollar volume",
            "office_age_months": "Office age",
            "office_active_agents_asof": "Office headcount",
            "units_3m": "Units closed (last 3mo)",
            "office_volume_prior_12m": "Office's prior-year volume",
            "buy_sides_12m": "Buy-side transactions (12mo)",
            "office_volume_trend_vs_market_12m": "Office volume trend vs. market",
            "office_volume_12m_total": "Office's total dollar volume (12mo)",
            "company_exit_rate_12m": "Company-wide peer exit rate (12mo)",
            "office_brand_independent": "Independent (non-franchise) office",
        }
        labels = [pretty_names.get(f, f) for f in top15.index]
        ax = fig.add_axes([0.32, 0.16, 0.58, 0.72])
        bars = ax.barh(labels, top15.values, color=BLUE_MED, height=0.6, zorder=3)
        for b, v in zip(bars, top15.values):
            ax.text(v + max(top15.values) * 0.015, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9, color=INK_SECONDARY)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(left=False, labelsize=10)
        ax.set_xlabel("CatBoost feature importance")

        fig.text(0.08, 0.095,
                  "New this round: company-level exit rate and office franchise brand both rank in the\n"
                  "top 9 — unused signals that turned out to matter once wired in.",
                  fontsize=9.5, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # -------------------------------------------------- Page 4: calibration
        fig = _new_page()
        _header(fig, "Calibration: Do the Percentages Mean What They Say?", "Predicted vs. actual departure rate, by risk quintile (out-of-fold, bias-corrected)", page_num=4)

        ax = fig.add_axes([0.10, 0.52, 0.80, 0.32])
        x = np.arange(5)
        w = 0.35
        b1 = ax.bar(x - w / 2, np.array(quintile_predicted) * 100, w, color=BLUE_LIGHT, label="Predicted", zorder=3)
        b2 = ax.bar(x + w / 2, np.array(quintile_actual) * 100, w, color=BLUE_DARK, label="Actual", zorder=3)
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.15, f"{h:.1f}%", ha="center", fontsize=8.5, color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(QUINTILE_LABELS)
        ax.set_ylabel("3-month departure rate")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.set_title("Risk quintile: predicted vs. actual", fontsize=11, color=INK_SECONDARY, loc="left")

        fig.text(0.08, 0.44, "Risk output per agent", fontsize=12, fontweight="bold", color=NAVY, ha="left")
        table_y = 0.32
        col_x = [0.08, 0.28, 0.52, 0.70]
        headers = ["Quintile", "Cutpoint (calibrated %)", "Actual rate", "Risk multiplier"]
        for cx, h in zip(col_x, headers):
            fig.text(cx, table_y + 0.06, h, fontsize=9.5, fontweight="bold", color=INK_SECONDARY, ha="left")
        fig.add_artist(plt.Line2D([0.08, 0.92], [table_y + 0.045, table_y + 0.045], color=GRID, linewidth=1, transform=fig.transFigure))
        cut_labels = ["< " + f"{cut_points[0]:.2%}"] + [f"{cut_points[i-1]:.2%}–{cut_points[i]:.2%}" for i in range(1, 4)] + [f"≥ {cut_points[3]:.2%}"]
        for i, (ql, cl, act) in enumerate(zip(QUINTILE_LABELS, cut_labels, quintile_actual)):
            ry = table_y + 0.02 - i * 0.038
            fig.text(col_x[0], ry, ql, fontsize=9.5, color=INK_PRIMARY, ha="left")
            fig.text(col_x[1], ry, cl, fontsize=9.5, color=INK_PRIMARY, ha="left")
            fig.text(col_x[2], ry, f"{act:.2%}", fontsize=9.5, color=INK_PRIMARY, ha="left")
            fig.text(col_x[3], ry, f"{act / base_rate:.1f}x base rate", fontsize=9.5, color=INK_PRIMARY, ha="left")

        fig.text(0.08, 0.145, "Bias correction & honest uncertainty", fontsize=10.5, fontweight="bold", color=NAVY, ha="left")
        fig.text(0.08, 0.115,
                  f"A backtest found the calibrator over-predicting in recent quarters (traced to a spike in confirmed\n"
                  f"rebrand/M&A cleanup, not real behavior change). A bias correction of {bias_correction:.2f}x is applied to today's\n"
                  "output, refreshed every quarterly retrain, plus a ±15% plausible range shown alongside each percentage.",
                  fontsize=9, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # -------------------------------------------------- Page 5: methodology
        fig = _new_page()
        _header(fig, "Methodology at a Glance", "Key decisions behind the numbers", page_num=5)

        rows = [
            ("Prediction horizon", "3 months (not 12) — unlocks 94% of rows vs. 78%, matches “who's a flight risk right now.”"),
            ("Validation", "Forward-chaining time-based folds, never random k-fold — agents recur across quarters, so a\nrandom split would leak the future into the past."),
            ("Label cleanup", "335 rows recoded from “left” to “stayed” across 23 confirmed rebrand/M&A/mass-mover office\ntransitions (≥5 agents, same origin/destination, same quarter) — corporate events, not\nvoluntary departures."),
            ("Features", f"~95 features: agent, office, company, and market-level activity and trends, plus macro\nindicators (state unemployment, CPI) and market-relative performance ratios."),
            ("Class imbalance", "Handled via class-weighting (not synthetic oversampling) — only ~2.1% of rows are positive."),
            ("Calibration", "Platt scaling with a quarterly bias-correction step (see previous page), not a fixed\nhistorical calibration."),
        ]
        y = 0.86
        for label, text in rows:
            fig.text(0.08, y, label, fontsize=11, fontweight="bold", color=NAVY, ha="left")
            fig.text(0.08, y - 0.028, text, fontsize=9.7, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
            n_lines = text.count("\n") + 1
            y -= 0.028 + n_lines * 0.028 + 0.045

        fig.text(0.08, 0.15, "Data source", fontsize=11, fontweight="bold", color=NAVY, ha="left")
        fig.text(0.08, 0.12,
                  f"{n_rows:,} agent-quarter records, {n_agents:,} unique RIAR (Rhode Island) agents, "
                  f"{n_quarters} quarterly snapshots\n(2021 Q4 – 2026 Q2).",
                  fontsize=9.7, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[report] saved to {out_path}")


if __name__ == "__main__":
    main()
