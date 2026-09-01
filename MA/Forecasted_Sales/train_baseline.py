"""
Simple baseline gut-check for the Forecasted Sales Algorithm.

Mirrors the role `train_baseline.py` played in the Leave model: not the
production model, just a fast, easy-to-audit first pass to answer "is
there real signal in this data at all," and to give later, fancier models
something concrete to beat. Ridge regression here plays the same role
logistic regression played there -- the simplest model that still respects
the time-based split.

Two targets, evaluated completely separately per the user's instruction:
target_volume_next_12m (dollars) and target_units_next_12m (count).
Leave-risk score is NOT included as a feature in this pass -- that's a
separate, explicit follow-up experiment (see HANDOFF.md).

A single forward time split is used here (not full forward-chaining
cross-validation) -- consistent with how the Leave project's own
`split.py` was used for a first gut-check before `crossval.py` came in
for real model comparison. Full CV will matter once we're comparing
candidate production models, not for "does this data have signal."
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import build_model_frame, get_feature_columns, CATEGORICAL_FEATURE_COLUMNS

TARGET_VOLUME = "target_volume_next_12m"
TARGET_UNITS = "target_units_next_12m"

TRAIN_QUARTERS = 12  # of 16 usable feature-snapshot quarters
RANDOM_STATE = 0


def time_split(df: pd.DataFrame):
    quarters = sorted(df["snapshot_date"].unique())
    train_quarters = quarters[:TRAIN_QUARTERS]
    test_quarters = quarters[TRAIN_QUARTERS:]
    train = df[df["snapshot_date"].isin(train_quarters)].copy()
    test = df[df["snapshot_date"].isin(test_quarters)].copy()
    return train, test, train_quarters, test_quarters


def prep_features(train, test, feature_cols):
    Xtr = train[feature_cols].astype(float).copy()
    Xte = test[feature_cols].astype(float).copy()
    medians = Xtr.median()
    Xtr = Xtr.fillna(medians)
    Xte = Xte.fillna(medians)
    # standardize using train stats only
    mu, sigma = Xtr.mean(), Xtr.std().replace(0, 1)
    Xtr = (Xtr - mu) / sigma
    Xte = (Xte - mu) / sigma
    return Xtr, Xte


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    # log1p RMSE -- clip negative predictions at 0 first (money/units can't be negative)
    y_pred_clipped = np.clip(y_pred, 0, None)
    log_rmse = mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped)) ** 0.5
    print(f"  {label:32s}  MAE={mae:>14,.0f}  RMSE={rmse:>14,.0f}  "
          f"R2={r2:>7.4f}  Spearman={rho:>7.4f}  log1p-RMSE={log_rmse:>7.4f}")
    return dict(mae=mae, rmse=rmse, r2=r2, spearman=rho, log_rmse=log_rmse)


def run_target(df, target_col, source_col, feature_cols, train_quarters, test_quarters):
    train = df[df["snapshot_date"].isin(train_quarters)]
    test = df[df["snapshot_date"].isin(test_quarters)]

    Xtr, Xte = prep_features(train, test, feature_cols)
    ytr, yte = train[target_col].values, test[target_col].values

    model = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    print(f"\n=== {target_col} ===")
    print(f"  train rows={len(train):,}  test rows={len(test):,}")
    ridge_metrics = evaluate(yte, pred, "Ridge regression")

    # naive persistence baseline: "next 12m = current trailing 12m" (no model at all)
    naive_pred = test[source_col].values
    naive_metrics = evaluate(yte, naive_pred, "Naive persistence (t+0=t+12m)")

    # naive mean baseline: predict the train-set mean for everyone (zero-signal floor)
    mean_pred = np.full_like(yte, fill_value=ytr.mean(), dtype=float)
    evaluate(yte, mean_pred, "Naive mean (train avg)")

    return ridge_metrics, naive_metrics


def main():
    df = build_model_frame()
    feature_cols = get_feature_columns(df)  # already includes one-hot brand_* dummies (bool dtype is numeric)

    train, test, train_quarters, test_quarters = time_split(df)
    print(f"Usable quarters: {len(sorted(df['snapshot_date'].unique()))}")
    print(f"Train: {train_quarters[0].date()} .. {train_quarters[-1].date()} "
          f"({len(train_quarters)} quarters, {len(train):,} rows)")
    print(f"Test:  {test_quarters[0].date()} .. {test_quarters[-1].date()} "
          f"({len(test_quarters)} quarters, {len(test):,} rows)")
    print(f"Feature count: {len(feature_cols)}")

    run_target(df, TARGET_VOLUME, "volume_12m", feature_cols, train_quarters, test_quarters)
    run_target(df, TARGET_UNITS, "units_12m", feature_cols, train_quarters, test_quarters)


if __name__ == "__main__":
    main()
