"""
How wide does the range actually need to be to hit real 95% coverage?

Uses a proper 3-way split so nothing is measured on the same data it was
tuned on:
  - TRAIN quarters: fit the XGBoost point-estimate model (same model as
    train_trees.py, just fewer quarters -- 2 are carved out for calibration).
  - CALIBRATION quarters: never used for training. We measure the model's
    residuals here to build candidate ranges. This is what a real
    conformal-prediction-style range-builder is allowed to look at.
  - TEST quarters: never used for training OR calibration. Coverage is
    only trustworthy if measured here -- checking coverage on the same
    data used to build the range would be circular (of course a range
    built to contain 95% of a dataset contains ~95% of that dataset).

Two candidate range constructions, both fit on CALIBRATION, both measured
on TEST:
  1. "Naive" symmetric normal-approximation range: pred +/- 1.96*RMSE.
     This is the textbook formula for a 95% CI, IF residuals were
     normally distributed. Real estate sales are not -- this is the
     thing we're testing.
  2. "Empirical" range: pred + [2.5th pctile residual, 97.5th pctile
     residual], measured directly off the calibration residuals'
     distribution -- no normality assumption, just "how wrong were we,
     historically, at each end."

Also checks whether error size scales with agent size (heteroscedasticity)
-- if a top producer's typical miss is much bigger in absolute dollars
than a small agent's, a single flat width for everyone is the wrong shape
of range, even once its overall coverage is correct.
"""
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

from data import build_model_frame, get_feature_columns
from train_baseline import RANDOM_STATE, TARGET_UNITS, TARGET_VOLUME

TRAIN_N = 10
CALIB_N = 2
# remaining quarters -> test


def three_way_split(df):
    quarters = sorted(df["snapshot_date"].unique())
    train_q = quarters[:TRAIN_N]
    calib_q = quarters[TRAIN_N:TRAIN_N + CALIB_N]
    test_q = quarters[TRAIN_N + CALIB_N:]
    return train_q, calib_q, test_q


def fit_model(train, feature_cols, target_col):
    model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(train[feature_cols], train[target_col].values)
    return model


def coverage_and_width(y_true, lo, hi):
    lo = np.clip(lo, 0, None)
    hi = np.clip(hi, 0, None)
    inside = (y_true >= lo) & (y_true <= hi)
    coverage = inside.mean()
    width = (hi - lo)
    return coverage, width.mean(), width


def analyze_target(df, target_col, feature_cols, train_q, calib_q, test_q, label, unit_fmt):
    train = df[df["snapshot_date"].isin(train_q)]
    calib = df[df["snapshot_date"].isin(calib_q)]
    test = df[df["snapshot_date"].isin(test_q)]

    model = fit_model(train, feature_cols, target_col)

    pred_calib = model.predict(calib[feature_cols])
    y_calib = calib[target_col].values
    resid_calib = y_calib - pred_calib

    pred_test = model.predict(test[feature_cols])
    y_test = test[target_col].values

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"train={len(train):,} rows ({train_q[0].date()}..{train_q[-1].date()})  "
          f"calib={len(calib):,} rows ({calib_q[0].date()}..{calib_q[-1].date()})  "
          f"test={len(test):,} rows ({test_q[0].date()}..{test_q[-1].date()})")

    # --- heteroscedasticity check: does |residual| scale with predicted size? ---
    abs_resid_calib = np.abs(resid_calib)
    pred_calib_safe = np.clip(pred_calib, 1, None)
    corr, _ = stats.pearsonr(pred_calib_safe, abs_resid_calib)
    print(f"\nHeteroscedasticity check (calibration set): "
          f"correlation(predicted size, |miss|) = {corr:.3f}")
    print("  (near 0 = flat/uniform error size regardless of agent size;"
          " closer to 1 = bigger producers have bigger absolute misses)")

    # --- range 1: naive normal approximation ---
    rmse_calib = np.sqrt(np.mean(resid_calib ** 2))
    naive_lo = pred_test - 1.96 * rmse_calib
    naive_hi = pred_test + 1.96 * rmse_calib
    naive_cov, naive_width, _ = coverage_and_width(y_test, naive_lo, naive_hi)

    # --- range 2: empirical residual quantiles ---
    q_lo, q_hi = np.percentile(resid_calib, [2.5, 97.5])
    emp_lo = pred_test + q_lo
    emp_hi = pred_test + q_hi
    emp_cov, emp_width, _ = coverage_and_width(y_test, emp_lo, emp_hi)

    # --- range 3: relative/multiplicative band (scales with predicted size) ---
    rel_resid_calib = resid_calib / pred_calib_safe
    rq_lo, rq_hi = np.percentile(rel_resid_calib, [2.5, 97.5])
    pred_test_safe = np.clip(pred_test, 1, None)
    rel_lo = pred_test + rq_lo * pred_test_safe
    rel_hi = pred_test + rq_hi * pred_test_safe
    rel_cov, rel_width, rel_width_arr = coverage_and_width(y_test, rel_lo, rel_hi)

    print(f"\n{'range type':32s} {'coverage':>10s} {'avg width':>16s} {'vs naive width':>16s}")
    print(f"{'naive normal (+/-1.96*RMSE)':32s} {naive_cov:>9.1%} {unit_fmt(naive_width):>16s} {'--':>16s}")
    print(f"{'empirical (calib residual q.)':32s} {emp_cov:>9.1%} {unit_fmt(emp_width):>16s} "
          f"{emp_width / naive_width - 1:>+15.1%}")
    print(f"{'relative (scales w/ agent size)':32s} {rel_cov:>9.1%} {unit_fmt(rel_width):>16s} "
          f"{rel_width / naive_width - 1:>+15.1%}")

    # how does the relative band's width vary for small vs large predicted agents?
    order = np.argsort(pred_test)
    third = len(order) // 3
    lo_idx, mid_idx, hi_idx = order[:third], order[third:2 * third], order[2 * third:]

    print(f"\n  Coverage broken down by agent size tercile (this is the check that matters --")
    print(f"  a flat band can hit ~95% OVERALL while being wrong for most individual agents):")
    print(f"  {'tercile':22s} {'naive coverage':>16s} {'relative coverage':>18s} "
          f"{'naive width':>14s} {'relative width':>16s}")
    for name, idx in [("smallest-predicted 1/3", lo_idx), ("middle 1/3", mid_idx), ("largest-predicted 1/3", hi_idx)]:
        naive_lo_i = np.clip(pred_test[idx] - 1.96 * rmse_calib, 0, None)
        naive_hi_i = np.clip(pred_test[idx] + 1.96 * rmse_calib, 0, None)
        naive_cov_i = ((y_test[idx] >= naive_lo_i) & (y_test[idx] <= naive_hi_i)).mean()
        rel_cov_i = ((y_test[idx] >= np.clip(rel_lo[idx], 0, None)) & (y_test[idx] <= np.clip(rel_hi[idx], 0, None))).mean()
        print(f"  {name:22s} {naive_cov_i:>15.1%} {rel_cov_i:>18.1%} "
              f"{unit_fmt(naive_hi_i.mean() - naive_lo_i.mean()):>14s} {unit_fmt(rel_width_arr[idx].mean()):>16s}")


def main():
    df = build_model_frame()
    feature_cols = get_feature_columns(df)
    train_q, calib_q, test_q = three_way_split(df)

    analyze_target(
        df, TARGET_VOLUME, feature_cols, train_q, calib_q, test_q,
        "VOLUME ($)", lambda w: f"${w:,.0f}",
    )
    analyze_target(
        df, TARGET_UNITS, feature_cols, train_q, calib_q, test_q,
        "UNITS (#)", lambda w: f"{w:.1f}",
    )


if __name__ == "__main__":
    main()
