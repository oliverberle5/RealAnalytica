"""Diagnose the fold-5 collapse (test = 2025-12-31 + 2026-03-31).

Both trees dropped from ~0.72-0.73 AUC (folds 1-4) to ~0.61-0.62 on the most
recent fold, while logistic regression held steady at ~0.70. This script
checks: is the drop broad (something shifted for everyone) or concentrated
(a few offices/segments driving it), and which features moved the most
between the train era and the test era.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crossval import MONOTONE, TREE_FEATURES, _tree_xy
from data import load_clean

TRAIN_DATES_CUTOFF = pd.Timestamp("2025-09-30", tz="US/Eastern")


def fit_catboost(X_train, y_train):
    from catboost import CatBoostClassifier

    monotone = [MONOTONE.get(c, 0) for c in TREE_FEATURES]
    model = CatBoostClassifier(
        iterations=400,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        monotone_constraints=monotone,
        loss_function="Logloss",
        eval_metric="PRAUC",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return model


def safe_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 30:
        return np.nan, len(y), int(y.sum())
    return roc_auc_score(y, s), len(y), int(y.sum())


def main():
    df = load_clean()
    train = df[df["snapshot_date"] <= TRAIN_DATES_CUTOFF]
    test = df[df["snapshot_date"] > TRAIN_DATES_CUTOFF]

    X_train, y_train = _tree_xy(train)
    X_test, y_test = _tree_xy(test)
    model = fit_catboost(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    test = test.assign(pred=scores)

    print("=" * 72)
    print("1) OVERALL: confirm the collapse, and split by the two test quarters")
    print("=" * 72)
    auc, n, npos = safe_auc(y_test, scores)
    print(f"fold-5 combined: AUC {auc:.4f}  n={n}  positives={npos}")
    for d, g in test.groupby("snapshot_date"):
        auc, n, npos = safe_auc(g["label_left_3m"], g["pred"])
        print(f"  {d.date()}: AUC {auc:.4f}  n={n}  positives={npos}")

    print()
    print("=" * 72)
    print("2) TRAIN-ERA vs TEST-ERA AUC on identical model (sanity: is it just noise?)")
    print("=" * 72)
    train_scores = model.predict_proba(X_train)[:, 1]
    # Score the model on the LAST TWO quarters of the train era for an
    # apples-to-apples comparison of "recent-but-in-sample" vs "recent-and-OOS"
    last_train_dates = sorted(train["snapshot_date"].unique())[-2:]
    recent_train_mask = train["snapshot_date"].isin(last_train_dates)
    auc, n, npos = safe_auc(y_train[recent_train_mask.to_numpy()], train_scores[recent_train_mask.to_numpy()])
    print(f"last 2 in-sample train quarters {[d.date() for d in last_train_dates]}: AUC {auc:.4f}  n={n}  positives={npos}")

    print()
    print("=" * 72)
    print("3) IS THE DROP CONCENTRATED? AUC by segment, test period only")
    print("=" * 72)

    def by_segment(col, bins=None, labels=None):
        s = test[col]
        if bins is not None:
            s = pd.cut(s, bins=bins, labels=labels)
        print(f"\n-- by {col} --")
        for val, g in test.groupby(s, observed=True):
            auc, n, npos = safe_auc(g["label_left_3m"], g["pred"])
            auc_s = f"{auc:.4f}" if not np.isnan(auc) else "  n/a"
            print(f"  {str(val):15s} AUC {auc_s}  n={n:5d}  positives={npos:3d}  base_rate={npos/n:.3%}")

    by_segment("cohort_volume_quartile")
    by_segment("office_is_branch")
    by_segment(
        "tenure_current_office_months",
        bins=[-0.01, 6, 24, 60, 1e9],
        labels=["0-6mo", "6-24mo", "2-5y", "5y+"],
    )

    print("\n-- by office (top 15 offices by row count in test) --")
    office_counts = test["office_mls_id"].value_counts().head(15).index
    for off in office_counts:
        g = test[test["office_mls_id"] == off]
        auc, n, npos = safe_auc(g["label_left_3m"], g["pred"])
        auc_s = f"{auc:.4f}" if not np.isnan(auc) else "  n/a"
        print(f"  {off:10s} AUC {auc_s}  n={n:4d}  positives={npos:2d}  base_rate={npos/n:.3%}")

    print()
    print("=" * 72)
    print("4) CONCENTRATION OF LEAVERS: are test-period positives clustered in")
    print("   a few offices (an office-level shock) vs spread out (broad shift)?")
    print("=" * 72)
    for label, frame in [("train era", train), ("test era (fold 5)", test)]:
        pos = frame[frame["label_left_3m"] == 1]
        n_offices_with_leaver = pos["office_mls_id"].nunique()
        total_offices = frame["office_mls_id"].nunique()
        leaver_counts_by_office = pos.groupby("office_mls_id").size()
        hhi = ((leaver_counts_by_office / leaver_counts_by_office.sum()) ** 2).sum() if len(leaver_counts_by_office) else np.nan
        print(
            f"{label:20s} leavers={len(pos):4d}  offices_with_>=1_leaver={n_offices_with_leaver:3d} "
            f"/ {total_offices:3d} total offices  HHI_of_leavers_by_office={hhi:.4f} "
            f"(higher = more concentrated)"
        )

    print()
    print("=" * 72)
    print("5) FEATURE DRIFT: standardized mean shift, train era -> test era")
    print("   (top 15 by |shift|, using train-era std as the yardstick)")
    print("=" * 72)
    importances = pd.Series(model.get_feature_importance(), index=TREE_FEATURES).sort_values(ascending=False)
    drift = []
    for col in TREE_FEATURES:
        tr = train[col].astype(float)
        te = test[col].astype(float)
        std = tr.std()
        if not std or np.isnan(std):
            continue
        shift = (te.mean() - tr.mean()) / std
        drift.append((col, shift, tr.mean(), te.mean(), importances.get(col, 0.0)))
    drift_df = pd.DataFrame(drift, columns=["feature", "std_shift", "train_mean", "test_mean", "model_importance"])
    drift_df["abs_shift"] = drift_df["std_shift"].abs()
    print(drift_df.sort_values("abs_shift", ascending=False).head(15).to_string(index=False))

    print()
    print("=" * 72)
    print("6) BASE RATE trend, quarter by quarter, full history")
    print("=" * 72)
    print(df.groupby(df["snapshot_date"].dt.date)["label_left_3m"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
