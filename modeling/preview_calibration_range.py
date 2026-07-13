"""Quick check: before building the calibration layer, what range of
calibrated percentages would it actually produce? Pools out-of-fold
CatBoost predictions across the 5 forward-chaining folds (genuine
out-of-sample scores, same data the real calibration step would use),
fits a simple Platt scaling (1-D logistic) calibrator, and reports the
resulting percentage distribution -- so we know whether calibration is
worth building before investing in it properly.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from crossval import MONOTONE, TREE_FEATURES, _fit_catboost, _forward_folds, _tree_xy
from data import load_clean


def main():
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())

    all_scores, all_labels = [], []
    for train_dates, test_dates in folds:
        train = df[df["snapshot_date"].isin(train_dates)]
        test = df[df["snapshot_date"].isin(test_dates)]
        scores, _ = _fit_catboost(train, test)
        all_scores.append(scores)
        all_labels.append(test["label_left_3m"].to_numpy())

    raw = np.concatenate(all_scores)
    y = np.concatenate(all_labels)
    base_rate = y.mean()

    print(f"pooled out-of-fold rows: {len(raw)}, positives: {y.sum()}, base rate: {base_rate:.3%}\n")

    print("=== RAW model score distribution (before calibration) ===")
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 100]
    for q in qs:
        print(f"  p{q:5.1f}: {np.percentile(raw, q):.4f}")

    platt = LogisticRegression()
    platt.fit(raw.reshape(-1, 1), y)
    calibrated = platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    print("\n=== CALIBRATED percent-chance distribution (Platt scaling, out-of-fold) ===")
    for q in qs:
        print(f"  p{q:5.1f}: {np.percentile(calibrated, q):.3%}")

    print(f"\nmean calibrated (should ~= base rate): {calibrated.mean():.3%}  (actual base rate: {base_rate:.3%})")

    top_decile_cut = np.percentile(raw, 90)
    top_pct_cut = np.percentile(raw, 99)
    print(f"\ntop 10% of agents by raw score -> calibrated range: "
          f"{calibrated[raw>=top_decile_cut].min():.3%} to {calibrated[raw>=top_decile_cut].max():.3%}, "
          f"mean {calibrated[raw>=top_decile_cut].mean():.3%} "
          f"({calibrated[raw>=top_decile_cut].mean()/base_rate:.1f}x base rate)")
    print(f"top 1% of agents by raw score  -> calibrated range: "
          f"{calibrated[raw>=top_pct_cut].min():.3%} to {calibrated[raw>=top_pct_cut].max():.3%}, "
          f"mean {calibrated[raw>=top_pct_cut].mean():.3%} "
          f"({calibrated[raw>=top_pct_cut].mean()/base_rate:.1f}x base rate)")
    print(f"bottom 50% of agents by raw score -> calibrated range: "
          f"{calibrated[raw<=np.percentile(raw,50)].min():.3%} to {calibrated[raw<=np.percentile(raw,50)].max():.3%}")


if __name__ == "__main__":
    main()
