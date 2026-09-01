"""
Decision #7 in HANDOFF.md: is the leave-risk score worth adding to the
sales quantile model? Two clean passes, identical in every way except the
feature set, so any difference is attributable to the feature itself:
  (A) baseline -- exactly round 4's quantile model, features only.
  (B) + leave_risk_score -- same features plus one column.

Same TRAIN/CALIB/TEST split, same hyperparameters, same CQR procedure, same
metrics as quantile_model.py. Both passes are re-run fresh in this script
(not reusing round 4's saved numbers) so the comparison is apples-to-apples
under identical randomness, not compared across separate runs.
"""
import numpy as np
import pandas as pd

from data import build_model_frame, get_feature_columns
from leave_risk_feature import build_leave_risk_scores
from quantify_range import three_way_split, coverage_and_width
from quantile_model import (
    QUANTILES, fit_quantile_model, conformal_adjustment, pinball_loss,
)
from train_baseline import TARGET_UNITS, TARGET_VOLUME


def tercile_spread(y_test, pred_med, lo, hi):
    order = np.argsort(pred_med)
    third = len(order) // 3
    covs = []
    for idx in (order[:third], order[third:2 * third], order[2 * third:]):
        lo_i = np.clip(lo[idx], 0, None)
        hi_i = np.clip(hi[idx], 0, None)
        covs.append(((y_test[idx] >= lo_i) & (y_test[idx] <= hi_i)).mean())
    return covs  # [small, mid, large]


def run_pass(df, target_col, feature_cols, train_q, calib_q, test_q, label, unit_fmt):
    train = df[df["snapshot_date"].isin(train_q)]
    calib = df[df["snapshot_date"].isin(calib_q)]
    test = df[df["snapshot_date"].isin(test_q)]

    models = {name: fit_quantile_model(train, feature_cols, target_col, a)
              for name, a in QUANTILES.items()}
    calib_preds = {name: m.predict(calib[feature_cols]) for name, m in models.items()}
    test_preds = {name: m.predict(test[feature_cols]) for name, m in models.items()}
    y_calib, y_test = calib[target_col].values, test[target_col].values

    adj = conformal_adjustment(y_calib, calib_preds["lo"], calib_preds["hi"])
    cqr_lo, cqr_hi = test_preds["lo"] - adj, test_preds["hi"] + adj
    cqr_cov, cqr_width, _ = coverage_and_width(y_test, cqr_lo, cqr_hi)

    covs = tercile_spread(y_test, test_preds["med"], cqr_lo, cqr_hi)
    pb_lo = pinball_loss(y_test, test_preds["lo"], 0.025)
    pb_hi = pinball_loss(y_test, test_preds["hi"], 0.975)
    pb_med = pinball_loss(y_test, test_preds["med"], 0.5)

    print(f"  {label:32s}  coverage={cqr_cov:>7.1%}  width={unit_fmt(cqr_width):>14s}  "
          f"tercile_spread=[{covs[0]:.1%},{covs[1]:.1%},{covs[2]:.1%}]  "
          f"pinball(lo/med/hi)=[{unit_fmt(pb_lo)},{unit_fmt(pb_med)},{unit_fmt(pb_hi)}]")

    return dict(coverage=cqr_cov, width=cqr_width, tercile_spread=covs,
                pb_lo=pb_lo, pb_med=pb_med, pb_hi=pb_hi)


def compare_target(df, target_col, base_features, train_q, calib_q, test_q, label, unit_fmt):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    a = run_pass(df, target_col, base_features, train_q, calib_q, test_q,
                 "(A) baseline, no leave-risk", unit_fmt)
    b = run_pass(df, target_col, base_features + ["leave_risk_score"], train_q, calib_q, test_q,
                 "(B) + leave_risk_score", unit_fmt)

    width_delta = b["width"] / a["width"] - 1
    cov_delta = b["coverage"] - a["coverage"]
    spread_a = max(a["tercile_spread"]) - min(a["tercile_spread"])
    spread_b = max(b["tercile_spread"]) - min(b["tercile_spread"])
    pb_delta = (b["pb_lo"] + b["pb_med"] + b["pb_hi"]) / (a["pb_lo"] + a["pb_med"] + a["pb_hi"]) - 1

    print(f"\n  Delta (B vs A): width {width_delta:+.2%}  |  coverage {cov_delta:+.2%} pts  |  "
          f"tercile-coverage spread {spread_a:.1%} -> {spread_b:.1%}  |  total pinball loss {pb_delta:+.2%}")
    verdict = "KEEP" if (pb_delta < -0.01 and cov_delta > -0.02) else "DISCARD (no clear win)"
    print(f"  Verdict at this single-split check: {verdict}")


def main():
    print("Building sales feature frame...")
    df = build_model_frame()
    base_features = get_feature_columns(df)

    print("Building leave-risk scores (classifier trained through 2024-07-01 only)...")
    risk = build_leave_risk_scores()

    before = len(df)
    df = df.merge(risk, on=["mls_agent_id", "snapshot_date"], how="left")
    matched = df["leave_risk_score"].notna().sum()
    print(f"\nMerge coverage: {matched:,}/{before:,} rows matched ({matched / before:.1%})")
    df = df.dropna(subset=["leave_risk_score"]).reset_index(drop=True)

    train_q, calib_q, test_q = three_way_split(df)
    print(f"Train: {train_q[0].date()}..{train_q[-1].date()}  "
          f"Calib: {calib_q[0].date()}..{calib_q[-1].date()}  "
          f"Test: {test_q[0].date()}..{test_q[-1].date()}")

    compare_target(df, TARGET_VOLUME, base_features, train_q, calib_q, test_q,
                    "VOLUME ($)", lambda w: f"${w:,.0f}")
    compare_target(df, TARGET_UNITS, base_features, train_q, calib_q, test_q,
                    "UNITS (#)", lambda w: f"{w:.2f}")


if __name__ == "__main__":
    main()
