"""Preview what equal-population quintile buckets would actually look like,
using the same pooled out-of-fold CatBoost predictions as
preview_calibration_range.py, so the bucket design can be judged against
real numbers rather than in the abstract.
"""

import numpy as np
import pandas as pd

from crossval import _fit_catboost, _forward_folds
from data import load_clean

LABELS = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]


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

    quintile = pd.qcut(raw, 5, labels=LABELS)
    out = pd.DataFrame({"raw_score": raw, "label": y, "quintile": quintile})

    print(f"pooled rows: {len(raw)}, positives: {y.sum()}, base rate: {base_rate:.3%}\n")
    print("=== EQUAL-POPULATION quintiles (each bucket = 20% of agents) ===")
    summary = out.groupby("quintile", observed=True).agg(
        n=("label", "size"),
        n_positive=("label", "sum"),
        actual_hit_rate=("label", "mean"),
        raw_score_min=("raw_score", "min"),
        raw_score_max=("raw_score", "max"),
    )
    summary["lift_vs_base"] = summary["actual_hit_rate"] / base_rate
    print(summary.to_string())

    print()
    print("=== Internal spread WITHIN the top bucket (does 'High' hide real differences?) ===")
    high = out[out["quintile"] == "High"]
    for q in [0, 25, 50, 75, 90, 99, 100]:
        print(f"  p{q:3d} raw score within High bucket: {np.percentile(high['raw_score'], q):.4f}")
    top_half_of_high = high[high["raw_score"] >= high["raw_score"].median()]
    bottom_half_of_high = high[high["raw_score"] < high["raw_score"].median()]
    print(f"  hit rate, TOP half of 'High' bucket:    {top_half_of_high['label'].mean():.3%}")
    print(f"  hit rate, BOTTOM half of 'High' bucket: {bottom_half_of_high['label'].mean():.3%}")


if __name__ == "__main__":
    main()
