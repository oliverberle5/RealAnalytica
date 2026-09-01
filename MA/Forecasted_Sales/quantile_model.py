"""
Real quantile regression + conformal calibration for the 95% range.

Where quantify_range.py took ONE point-estimate model and bolted a
residual-based band onto it (flat, then size-scaled as a proxy), this
trains the actual 2.5th/50th/97.5th percentile as three separate XGBoost
models (`objective="reg:quantileerror"`), so the band's shape can depend
on ANY feature the model finds predictive of volatility -- not just
predicted size. That directly addresses the round-3 finding: a size-only
proxy fixes most of the miscalibration but isn't the real fix.

Then applies split-conformal calibration (CQR -- conformalized quantile
regression) on top: the raw quantile model's coverage is not guaranteed
to be exactly 95% (it's only trained to minimize pinball loss, which is a
different objective than "hit exactly 95% coverage"). CQR fixes this by
measuring, on a held-out CALIBRATION set the quantile model never trained
on, how far outside the raw [q_lo, q_hi] band the true values actually
fell, and shifting both edges by that amount -- same conformal idea used
in quantify_range.py's split, just applied to a real quantile model
instead of a flat/relative residual band. This is what gives an actual
coverage GUARANTEE (under the standard conformal exchangeability
assumption) rather than a hope.

Same train/calibration/test split as quantify_range.py, same reasoning
for why (no coverage number is trustworthy unless measured on data the
band-building process never touched).
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data import build_model_frame, get_feature_columns
from quantify_range import three_way_split, coverage_and_width
from train_baseline import RANDOM_STATE, TARGET_UNITS, TARGET_VOLUME

QUANTILES = {"lo": 0.025, "med": 0.5, "hi": 0.975}
TARGET_ALPHA = 0.05  # 1 - 0.95


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))


def fit_quantile_model(train, feature_cols, target_col, alpha):
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(train[feature_cols], train[target_col].values)
    return model


def conformal_adjustment(y_calib, lo_calib, hi_calib):
    """Split-conformal (CQR) adjustment: how far outside the raw band did
    the truth fall, at the calibration set's 95th percentile of miss."""
    nonconformity = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    n = len(nonconformity)
    # finite-sample-corrected quantile level, standard CQR formula
    level = min(1.0, np.ceil((n + 1) * (1 - TARGET_ALPHA)) / n)
    return np.quantile(nonconformity, level)


def tercile_report(y_test, pred_med, lo, hi, unit_fmt, label):
    order = np.argsort(pred_med)
    third = len(order) // 3
    terciles = [
        ("smallest-predicted 1/3", order[:third]),
        ("middle 1/3", order[third:2 * third]),
        ("largest-predicted 1/3", order[2 * third:]),
    ]
    print(f"\n  {label} -- coverage by predicted-size tercile:")
    print(f"  {'tercile':24s} {'coverage':>10s} {'avg width':>16s}")
    for name, idx in terciles:
        lo_i = np.clip(lo[idx], 0, None)
        hi_i = np.clip(hi[idx], 0, None)
        cov = ((y_test[idx] >= lo_i) & (y_test[idx] <= hi_i)).mean()
        print(f"  {name:24s} {cov:>9.1%} {unit_fmt((hi_i - lo_i).mean()):>16s}")


def analyze_target(df, target_col, feature_cols, train_q, calib_q, test_q, label, unit_fmt):
    train = df[df["snapshot_date"].isin(train_q)]
    calib = df[df["snapshot_date"].isin(calib_q)]
    test = df[df["snapshot_date"].isin(test_q)]

    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")

    models = {name: fit_quantile_model(train, feature_cols, target_col, a)
              for name, a in QUANTILES.items()}

    calib_preds = {name: m.predict(calib[feature_cols]) for name, m in models.items()}
    test_preds = {name: m.predict(test[feature_cols]) for name, m in models.items()}
    y_calib = calib[target_col].values
    y_test = test[target_col].values

    # --- raw quantile model, no conformal adjustment ---
    raw_lo, raw_hi = test_preds["lo"], test_preds["hi"]
    raw_cov, raw_width, _ = coverage_and_width(y_test, raw_lo, raw_hi)

    # --- conformal (CQR) adjustment, measured on calibration, applied to test ---
    adj = conformal_adjustment(y_calib, calib_preds["lo"], calib_preds["hi"])
    cqr_lo, cqr_hi = test_preds["lo"] - adj, test_preds["hi"] + adj
    cqr_cov, cqr_width, _ = coverage_and_width(y_test, cqr_lo, cqr_hi)

    print(f"train={len(train):,}  calib={len(calib):,}  test={len(test):,}")
    print(f"conformal adjustment (shift applied to both edges): {unit_fmt(adj)}")

    print(f"\n{'band':32s} {'coverage':>10s} {'avg width':>16s}")
    print(f"{'raw quantile model (uncalibrated)':32s} {raw_cov:>9.1%} {unit_fmt(raw_width):>16s}")
    print(f"{'CQR (conformal-calibrated)':32s} {cqr_cov:>9.1%} {unit_fmt(cqr_width):>16s}")

    tercile_report(y_test, test_preds["med"], raw_lo, raw_hi, unit_fmt, "raw quantile model")
    tercile_report(y_test, test_preds["med"], cqr_lo, cqr_hi, unit_fmt, "CQR (conformal-calibrated)")

    # --- pinball loss: the actual scoring metric for interval quality ---
    pb_lo = pinball_loss(y_test, raw_lo, 0.025)
    pb_hi = pinball_loss(y_test, raw_hi, 0.975)
    print(f"\n  pinball loss (raw quantile model): lower={unit_fmt(pb_lo)}  upper={unit_fmt(pb_hi)}")

    # --- bonus: how does the median quantile model compare to the squared-error point model? ---
    from sklearn.metrics import mean_absolute_error
    mae_med = mean_absolute_error(y_test, test_preds["med"])
    print(f"  median-quantile-model MAE: {unit_fmt(mae_med)} (compare to round-2 XGBoost squared-error MAE in HANDOFF.md)")

    return dict(raw_cov=raw_cov, raw_width=raw_width, cqr_cov=cqr_cov, cqr_width=cqr_width, adj=adj)


def main():
    df = build_model_frame()
    feature_cols = get_feature_columns(df)
    train_q, calib_q, test_q = three_way_split(df)
    print(f"Train: {train_q[0].date()}..{train_q[-1].date()}  "
          f"Calib: {calib_q[0].date()}..{calib_q[-1].date()}  "
          f"Test: {test_q[0].date()}..{test_q[-1].date()}")

    analyze_target(df, TARGET_VOLUME, feature_cols, train_q, calib_q, test_q,
                    "VOLUME ($) -- quantile regression + CQR", lambda w: f"${w:,.0f}")
    analyze_target(df, TARGET_UNITS, feature_cols, train_q, calib_q, test_q,
                    "UNITS (#) -- quantile regression + CQR", lambda w: f"{w:.1f}")


if __name__ == "__main__":
    main()
