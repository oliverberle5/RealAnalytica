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

from data import AUTORESEARCH_CHAMPION_COLUMNS, RAW_FEATURE_COLUMNS, load_clean
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
#
# 2026-08-07: this dict declared five constraints but the deployed model only
# ever carried four -- same as RI, since MA inherits the RI feature set.
# `volume_trend_vs_market_12m` (-1) is not in PRODUCTION_FEATURES (the 94->54
# pruning kept `units_trend_vs_market_12m` instead), and MONOTONE.get(c, 0)
# silently resolved the missing key to "unconstrained" rather than erroring.
# Resolved in favour of the model: declared and deployed now agree at four.
# See RI Final_Model/crossval.py for the measured cost of also dropping it
# from TREE_FEATURES (inside the noise band).
MONOTONE = {
    "office_exit_rate_12m": 1,        # more peers leaving -> more likely to leave
    "months_since_last_closing": 1,   # longer disengaged -> more likely to leave
    "share_of_office_volume_12m": -1, # more embedded in office -> less likely
    "company_exit_rate_12m": 1,       # same prior, one level up the org hierarchy
}

# 2026-07-28: appended the 5 engineered features from the GEPA autoresearch
# champion (round r003_20260726_204147, CONFIRMED by confirm.py 2026-07-27 --
# see data.py::AUTORESEARCH_CHAMPION_COLUMNS / add_autoresearch_champion_features
# and AutoResearch/gepa_loop/best/champion.json). Same production hyperparameters
# below, unchanged -- only the feature list grew from 49 to 54.

# PRODUCTION model config (2026-07-25): promoted from AutoResearch's confirmed-best
# commit `ff0a573` (branch autoresearch/jul13, AutoResearch/results.tsv, mean AUC
# 0.746821 / worst-fold 0.721336 as originally measured) -- six rounds of
# hyperparameter/weighting/pruning search (see AutoResearch/results.tsv,
# CHANGELOG.md) converged on this and found nothing that beat it by more than
# statistical noise. This had never actually been wired into score_agents.py /
# train_multi_horizon.py / calibrate.py before now -- production was still running
# the plain depth=4, full-feature, Balanced-weighted reference config the whole
# time this sat validated in AutoResearch/. Freshly re-derived against the current
# data file (modeling/derive_winning_config.py) rather than trusted blindly from the
# original commit, since file pulls have quietly shifted values before (see data.py
# docstring): reproduces mean 0.744235 / worst-fold 0.721177 -- close to but not
# identical to the original 0.746821/0.721336 (expected drift, not a bug), still a
# clear, real improvement over the 0.7384 current-production baseline on this file.
#
# PRODUCTION_FEATURES: bottom-45-importance features dropped from TREE_FEATURES
# (49 of 94 kept), where importance is ranked by fitting a depth=3,
# auto_class_weights="SqrtBalanced" CatBoost model on FOLD 1's TRAINING WINDOW
# ONLY (never re-derived per fold -- a fixed, reviewable list, not a live
# recomputation on every run). See derive_winning_config.py for the exact
# reproduction. Notably drops units_12m/volume_12m (the agent's own raw
# production level) -- for THIS target (who leaves), tenure/office-embeddedness
# context outranks raw production, unlike the Sales project's target where
# production level dominates; not a bug, a genuine difference between the two
# prediction problems.
PRODUCTION_FEATURES = [
    "tenure_current_office_months",
    "career_months",
    "moves_5y_asof",
    "months_since_last_move",
    "units_prior_12m",
    "volume_prior_12m",
    "units_3m",
    "units_prior_3m",
    "trend_3m",
    "buy_sides_12m",
    "list_share_12m",
    "new_listings_12m",
    "rentals_12m",
    "colist_sides_12m",
    "avg_sale_price_12m",
    "price_min_12m",
    "months_since_last_closing",
    "pct_units_single_family_12m",
    "pct_units_multi_family_12m",
    "pct_units_rental_12m",
    "volume_delta_vs_cohort_median_pts",
    "share_of_office_volume_12m",
    "rank_in_office_volume_12m",
    "price_gap_vs_office",
    "office_active_agents_asof",
    "office_exits_12m",
    "office_net_flow_12m",
    "office_exit_rate_12m",
    "office_units_12m",
    "office_volume_prior_12m",
    "office_avg_sale_price_12m",
    "office_is_branch",
    "market_total_volume_12m",
    "company_volume_12m",
    "company_exits_12m",
    "company_net_flow_12m",
    "company_exit_rate_12m",
    "office_exit_concentration_12m",
    "prev_stint_months",
    "avg_prior_stint_months",
    "max_prior_stint_months",
    "office_brand_independent",
    "office_brand_century21",
    "office_brand_compass",
    "office_headcount_trend_12m",
    "units_trend_12m",
    "units_trend_vs_market_12m",
    "company_headcount_trend_12m",
    "company_arrival_rate_12m",
] + AUTORESEARCH_CHAMPION_COLUMNS

PRODUCTION_CATBOOST_PARAMS = dict(
    iterations=400,
    depth=3,
    learning_rate=0.04,
    l2_leaf_reg=3.5,
    auto_class_weights="SqrtBalanced",
    loss_function="Logloss",
    eval_metric="PRAUC",
    verbose=False,
    allow_writing_files=False,
)


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
    """CatBoost, PRODUCTION config (2026-07-25) -- see PRODUCTION_FEATURES /
    PRODUCTION_CATBOOST_PARAMS above. This is what calibrate.py, score_agents.py,
    and train_multi_horizon.py all actually deploy, so this function deliberately
    diverges from _fit_logistic/_fit_xgboost's shared TREE_FEATURES (breaking the
    module docstring's original "same features for every model" fairness
    principle, on purpose) -- the catboost entry in main()'s bake-off below now
    reflects the real deployed recipe, not a same-features-forced comparison.
    """
    from catboost import CatBoostClassifier

    X_train = train[PRODUCTION_FEATURES].astype(float)
    y_train = train["label_left_3m"].to_numpy()
    X_test = test[PRODUCTION_FEATURES].astype(float)
    monotone = [MONOTONE.get(c, 0) for c in PRODUCTION_FEATURES]
    model = CatBoostClassifier(
        monotone_constraints=monotone,
        **PRODUCTION_CATBOOST_PARAMS,
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
