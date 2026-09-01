"""Regenerate VERIFIED_TABLES.md T9b -- the raw (uncalibrated) score
distribution, its Platt coefficients, the tier cutpoints mapped back onto the
raw axis, and the histogram bin counts the pgfplots figure carries inline.

Exists because raw scores are not persisted anywhere: score_agents.py computes
them and discards them the moment they are Platt-scaled, and
current_agent_risk_scores.csv carries only percent_chance / risk_multiplier.
T9b's own "how to re-derive after a retrain" note describes doing this by hand;
this script is that note, executed, so the next refresh is one command rather
than a reconstruction.

Reproduces score_agents.py's path exactly (load_clean -> _collect_oof ->
fit_production_calibrator -> compute_recent_bias_correction ->
train_production_model -> predict_proba) rather than approximating it, and
asserts the calibrated percentiles it derives match the shipped CSV, so a drift
between this script and production surfaces here instead of in the paper.
"""
import numpy as np
import pandas as pd

from calibrate import (
    _collect_oof,
    compute_recent_bias_correction,
    compute_tier_cutpoints,
    fit_production_calibrator,
)
from crossval import _forward_folds
from data import load_clean, load_current_snapshot
from crossval import PRODUCTION_FEATURES
from score_agents import train_production_model

BINS = np.arange(0, 0.5625, 0.0125)  # figure bins, 0.0125 wide (T9b caption)


def main():
    print("[t9b] loading history, collecting OOF for the calibrator...")
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())
    per_fold = _collect_oof(df, folds)
    platt, _cut, base_rate = fit_production_calibrator(per_fold)
    bias = compute_recent_bias_correction(per_fold, recent_fold_idx=len(per_fold) - 1)
    corrected_base = base_rate * bias
    cut_points = compute_tier_cutpoints(corrected_base)

    print("[t9b] training production model, scoring current snapshot...")
    model = train_production_model(df)
    current = load_current_snapshot()
    raw = model.predict_proba(current[PRODUCTION_FEATURES].astype(float))[:, 1]
    pct = platt.predict_proba(raw.reshape(-1, 1))[:, 1] * bias

    a = float(platt.coef_[0][0]); b = float(platt.intercept_[0])
    print(f"\n[t9b] n = {len(raw):,}   Platt a = {a:.5f}, b = {b:.5f}   bias = {bias:.4f}")
    print(f"[t9b] base rate {base_rate:.4%}  ->  bias-corrected {corrected_base:.4%}")

    # Cross-check against the shipped spreadsheet -- same agents, same pipeline.
    shipped = pd.read_csv("current_agent_risk_scores.csv", dtype={"mls_agent_id": str})
    s = shipped.set_index("mls_agent_id")["percent_chance"]
    mine = pd.Series(pct, index=current["mls_agent_id"].astype(str))
    delta = (mine - s.reindex(mine.index)).abs().max()
    print(f"[t9b] max |pct - shipped percent_chance| = {delta:.3e}  "
          f"({'MATCHES production' if delta < 1e-9 else 'DIVERGED -- investigate'})")

    print("\n--- T9b table: raw score -> calibrated % ---")
    rows = [("Minimum", raw.min()), ("1st pct", np.percentile(raw, 1)),
            ("5th pct", np.percentile(raw, 5)), ("10th pct", np.percentile(raw, 10)),
            ("25th pct", np.percentile(raw, 25)), ("Median", np.percentile(raw, 50)),
            ("75th pct", np.percentile(raw, 75)), ("90th pct", np.percentile(raw, 90)),
            ("95th pct", np.percentile(raw, 95)), ("99th pct", np.percentile(raw, 99)),
            ("Maximum", raw.max())]
    for label, r in rows:
        p = float(platt.predict_proba(np.array([[r]]))[:, 1][0]) * bias
        print(f"  {label:<10} {r:.4f}   {p:.2%}")
    print(f"  {'Mean':<10} {raw.mean():.4f}   ---")
    print(f"  {'Std. dev.':<10} {raw.std(ddof=0):.4f}   ---")

    print("\n--- tier cutpoints, calibrated % -> raw axis (inverse Platt) ---")
    for name, c in zip(("Low|Medium", "Medium|High", "High|Extra High"), cut_points):
        raw_cut = (np.log(c / bias / (1 - c / bias)) - b) / a
        print(f"  {name:<16} {c:.4%}  ->  raw {raw_cut:.5f}")

    print("\n--- histogram bin counts for the pgfplots figure (bin width 0.0125) ---")
    counts, edges = np.histogram(raw, bins=BINS)
    coords = " ".join(f"({edges[i]:.5f},{counts[i]})" for i in range(len(counts)))
    print(coords)
    above = (raw > BINS[-1]).sum()
    print(f"\n  agents above axis cap {BINS[-1]:.2f}: {above} ({above / len(raw):.2%}), max {raw.max():.4f}")

    # Ordering check T9b asserts: raw-space tiers must reproduce calibrated-space tiers.
    raw_cuts = [(np.log(c / bias / (1 - c / bias)) - b) / a for c in cut_points]
    same = (np.searchsorted(raw_cuts, raw) == np.searchsorted(cut_points, pct)).sum()
    print(f"  tier assignment from raw cutpoints matches calibrated: {same} of {len(raw)}")


if __name__ == "__main__":
    main()
