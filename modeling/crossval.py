"""Forward-chaining cross-validation: Logistic vs XGBoost vs CatBoost.

Why forward-chaining (expanding window) rather than plain k-fold:
the data is a panel of the SAME agents observed quarter after quarter, and
the task is to forecast forward in time. A random k-fold would leak future
snapshots into training and let an agent's own later quarters inform its
earlier ones -- inflating every metric. Each fold here trains on all
snapshots strictly before the test block, exactly mirroring deployment
(retrain each quarter, score the next). Per-fold numbers are reported, not
just the mean, so we can see stability -- which matters a lot given only
~2.5% of rows are positive (~1,300 leavers in the whole training era).

Comparison metrics are RANKING metrics (ROC-AUC, PR-AUC, lift@top-decile),
because (a) the production use is top-N alerting, which only needs a correct
ordering, and (b) they are invariant to the probability miscalibration that
class-weighting deliberately introduces. Turning the winner's raw score into
an honest percentage is a separate downstream calibration step.

All three models are given the SAME numeric feature set and the SAME
imbalance handling (balanced class weights) so the comparison isolates the
algorithm, not the feature engineering. Notes on levers each model has that
are deliberately left unused here (CatBoost native categoricals, tree-native
missing market features) are in the conversation writeup.
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from data import RAW_FEATURE_COLUMNS, load_clean
from train_baseline import (
    MARKET_FEATURES_EXCLUDED,
    build_design_matrix,
)

# Tree feature set: same columns the linear model is allowed to use
# (MARKET_FEATURES_EXCLUDED is empty as of the 2026-07 data update -- every
# market/macro column now has full coverage -- kept as a filter for parity
# with build_design_matrix in case a future column reintroduces a gap).
# Trees get RAW continuous values -- including raw tenure -- and discover
# breakpoints themselves, instead of the hand-chosen bins the linear model
# needs.
TREE_FEATURES = [c for c in RAW_FEATURE_COLUMNS if c not in MARKET_FEATURES_EXCLUDED]

# Monotonic constraints only where the business-intuition prior is strong and
# single-directional. NOT on tenure (its risk curve is non-monotonic: highest
# for brand-new hires, lowest at 5-10y, rising again at 10y+). Acts as a
# regularizer given how few positives we have, and makes the model defensible.
MONOTONE = {
    "office_exit_rate_12m": 1,        # more peers leaving -> more likely to leave
    "months_since_last_closing": 1,   # longer disengaged -> more likely to leave
    "share_of_office_volume_12m": -1, # more embedded in office -> less likely
    "volume_trend_vs_market_12m": -1, # outperforming the market -> less likely to leave
    "company_exit_rate_12m": 1,       # same prior, one level up the org hierarchy
}


def _lift_at_top_decile(y_true, y_score):
    n = len(y_true)
    k = max(1, n // 10)
    order = np.argsort(y_score)[::-1]
    top = order[:k]
    top_rate = y_true[top].mean()
    base = y_true.mean()
    return top_rate / base if base > 0 else np.nan


def _forward_folds(snapshot_dates, start_train=9, test_block=2):
    dates = sorted(snapshot_dates)
    folds = []
    i = start_train
    while i + test_block <= len(dates):
        train_dates = set(dates[:i])
        test_dates = set(dates[i : i + test_block])
        folds.append((train_dates, test_dates))
        i += test_block
    return folds


def _tree_xy(df):
    X = df[TREE_FEATURES].astype(float)
    y = df["label_left_3m"].to_numpy()
    return X, y


def _fit_logistic(train, test):
    X_train, medians = build_design_matrix(train)
    X_test, _ = build_design_matrix(test, medians=medians)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    y_train = train["label_left_3m"].to_numpy()
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    fit_s = time.perf_counter() - t0
    return model.predict_proba(Xte)[:, 1], fit_s


def _fit_xgboost(train, test):
    from xgboost import XGBClassifier

    X_train, y_train = _tree_xy(train)
    X_test, _ = _tree_xy(test)
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)
    monotone = tuple(MONOTONE.get(c, 0) for c in TREE_FEATURES)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        monotone_constraints=monotone,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    return model.predict_proba(X_test)[:, 1], fit_s


def _fit_catboost(train, test):
    from catboost import CatBoostClassifier

    X_train, y_train = _tree_xy(train)
    X_test, _ = _tree_xy(test)
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
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    return model.predict_proba(X_test)[:, 1], fit_s


MODELS = {
    "logistic": _fit_logistic,
    "xgboost": _fit_xgboost,
    "catboost": _fit_catboost,
}


def main():
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())
    print(f"[cv] {len(folds)} forward-chaining folds (expanding window, 2 snapshots/test)\n")

    results = {m: {"auc": [], "prauc": [], "lift": [], "fit_s": []} for m in MODELS}

    for fi, (train_dates, test_dates) in enumerate(folds, 1):
        train = df[df["snapshot_date"].isin(train_dates)]
        test = df[df["snapshot_date"].isin(test_dates)]
        y_test = test["label_left_3m"].to_numpy()
        tr_lo, tr_hi = min(train_dates).date(), max(train_dates).date()
        te_lo, te_hi = min(test_dates).date(), max(test_dates).date()
        print(
            f"[fold {fi}] train {tr_lo}..{tr_hi} ({len(train)} rows, {y_test.mean():.2%} test pos) "
            f"-> test {te_lo}..{te_hi} ({len(test)} rows)"
        )
        for name, fitfn in MODELS.items():
            scores, fit_s = fitfn(train, test)
            auc = roc_auc_score(y_test, scores)
            prauc = average_precision_score(y_test, scores)
            lift = _lift_at_top_decile(y_test, scores)
            results[name]["auc"].append(auc)
            results[name]["prauc"].append(prauc)
            results[name]["lift"].append(lift)
            results[name]["fit_s"].append(fit_s)
            print(
                f"          {name:9s}  AUC {auc:.4f}  PR-AUC {prauc:.4f}  "
                f"lift@10% {lift:.2f}x  fit {fit_s:.2f}s"
            )
        print()

    base = df["label_left_3m"].mean()
    print("=" * 72)
    print(f"MEAN ACROSS FOLDS (base positive rate {base:.3%})")
    print("=" * 72)
    print(f"{'model':10s} {'AUC':>8s} {'PR-AUC':>8s} {'PR/base':>8s} {'lift@10%':>9s} {'fit_s':>7s}")
    for name in MODELS:
        r = results[name]
        auc = np.mean(r["auc"])
        prauc = np.mean(r["prauc"])
        lift = np.mean(r["lift"])
        fit_s = np.mean(r["fit_s"])
        print(
            f"{name:10s} {auc:8.4f} {prauc:8.4f} {prauc / base:8.2f} "
            f"{lift:9.2f} {fit_s:7.2f}"
        )
    print()
    print("Per-fold AUC (stability check):")
    for name in MODELS:
        aucs = " ".join(f"{a:.3f}" for a in results[name]["auc"])
        print(f"  {name:10s} {aucs}")


if __name__ == "__main__":
    main()
