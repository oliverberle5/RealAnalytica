"""
XGBoost vs CatBoost comparison for the Forecasted Sales Algorithm.

Same targets, same time split, same metrics as train_baseline.py (Ridge) --
this is a like-for-like upgrade comparison, not a new experiment design.
Leave-risk score is still NOT included as a feature (decision #7 in
HANDOFF.md -- that's a separate, later experiment).

Unlike Ridge, both tree models handle missing values natively, so no
median-imputation/standardization step is needed -- raw features (with
NaNs left in) go in directly. This also removes one of Ridge's structural
weaknesses (imputing structurally-missing columns like `prev_stint_months`
with a median, which invents a value for agents where "missing" itself is
meaningful -- e.g. num_offices_career==1).
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from data import build_model_frame, get_feature_columns
from train_baseline import (
    RANDOM_STATE,
    TARGET_UNITS,
    TARGET_VOLUME,
    TRAIN_QUARTERS,
    evaluate,
    time_split,
)


def run_target(df, target_col, feature_cols, train_quarters, test_quarters):
    train = df[df["snapshot_date"].isin(train_quarters)]
    test = df[df["snapshot_date"].isin(test_quarters)]

    Xtr, Xte = train[feature_cols], test[feature_cols]
    ytr, yte = train[target_col].values, test[target_col].values

    print(f"\n=== {target_col} ===")
    print(f"  train rows={len(train):,}  test rows={len(test):,}")

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(Xtr, ytr)
    xgb_pred = np.clip(xgb.predict(Xte), 0, None)
    xgb_metrics = evaluate(yte, xgb_pred, "XGBoost")

    cat = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    cat.fit(Xtr, ytr)
    cat_pred = np.clip(cat.predict(Xte), 0, None)
    cat_metrics = evaluate(yte, cat_pred, "CatBoost")

    return xgb, cat, xgb_metrics, cat_metrics


def top_importances(model, feature_cols, model_name, n=10):
    if model_name == "XGBoost":
        imp = model.feature_importances_
    else:
        imp = model.get_feature_importance()
    order = np.argsort(imp)[::-1][:n]
    print(f"\n  Top {n} features ({model_name}):")
    for i in order:
        print(f"    {feature_cols[i]:40s} {imp[i]:.4f}")


def main():
    df = build_model_frame()
    feature_cols = get_feature_columns(df)
    train, test, train_quarters, test_quarters = time_split(df)
    print(f"Usable quarters: {len(sorted(df['snapshot_date'].unique()))}")
    print(f"Train: {train_quarters[0].date()} .. {train_quarters[-1].date()} "
          f"({len(train_quarters)} quarters, {len(train):,} rows)")
    print(f"Test:  {test_quarters[0].date()} .. {test_quarters[-1].date()} "
          f"({len(test_quarters)} quarters, {len(test):,} rows)")
    print(f"Feature count: {len(feature_cols)}")

    for target_col in (TARGET_VOLUME, TARGET_UNITS):
        xgb, cat, xgb_m, cat_m = run_target(df, target_col, feature_cols, train_quarters, test_quarters)
        top_importances(xgb, feature_cols, "XGBoost")
        top_importances(cat, feature_cols, "CatBoost")


if __name__ == "__main__":
    main()
