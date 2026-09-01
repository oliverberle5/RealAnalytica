"""
Fine sweep of range width vs. confidence level, 50%-99%, for the current
production model (log1p CatBoost d3, round 11). Extends
compare_precision_levels.py's 3-point comparison (60/75/95) into a full
curve so the width-vs-confidence trade-off can be seen continuously,
not just at 3 samples.

For confidence level c, the quantile pair is ((1-c)/2, (1+c)/2) -- e.g.
c=0.60 -> (0.20, 0.80), same convention as compare_precision_levels.py.
Each level needs its own pair of quantile models (CatBoost's quantile
loss is fit per-alpha, not derivable from a single fit), so this is
2 x len(LEVELS) model fits per target -- genuinely expensive, run once
and cache the result to JSON so the chart-building step doesn't need to
refit.

Same train/calib windows, same conformal calibration, same current
snapshot as score_agents.py -- every number here is directly comparable
to (and a superset of) what's already in HANDOFF.md's 60/75/95 analysis.
"""
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from data import get_feature_columns, TARGET_COLUMNS
from score_agents import (
    build_full_frame, CALIB_QUARTERS, MODEL_PARAMS,
    compute_realistic_floor, snap_unrealistic_floor,
)
from train_baseline import TARGET_UNITS, TARGET_VOLUME

LEVELS_PCT = [50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "confidence_sweep_results.json"


def fit_quantile_model(train, feature_cols, target_col, alpha):
    m = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", **MODEL_PARAMS)
    m.fit(train[feature_cols], np.log1p(train[target_col].values))
    return m


def conformal_adjustment(y_calib, lo_calib, hi_calib, target_alpha):
    nonconformity = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    n = len(nonconformity)
    level = min(1.0, np.ceil((n + 1) * (1 - target_alpha)) / n)
    return np.quantile(nonconformity, level)


def main():
    print("[sweep] loading data...")
    full = build_full_frame()
    resolved = full.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    feature_cols = get_feature_columns(full)
    current_snapshot_date = full["snapshot_date"].max()
    current = full[full["snapshot_date"] == current_snapshot_date].reset_index(drop=True)
    print(f"[sweep] current snapshot: {current_snapshot_date.date()}  ({len(current):,} agents)")

    quarters = sorted(resolved["snapshot_date"].unique())
    calib_q = quarters[-CALIB_QUARTERS:]
    train_q = quarters[:-CALIB_QUARTERS]
    train = resolved[resolved["snapshot_date"].isin(train_q)]
    calib = resolved[resolved["snapshot_date"].isin(calib_q)]
    print(f"[sweep] train={len(train):,} rows ({train_q[0].date()}..{train_q[-1].date()})  "
          f"calib={len(calib):,} rows ({calib_q[0].date()}..{calib_q[-1].date()})")
    print(f"[sweep] {len(LEVELS_PCT)} confidence levels x 2 quantiles x 2 targets = "
          f"{len(LEVELS_PCT) * 2 * 2} model fits\n")

    results = {"snapshot_date": str(current_snapshot_date.date()), "n_agents": len(current), "levels": []}

    for target_col, label in [(TARGET_VOLUME, "volume"), (TARGET_UNITS, "units")]:
        y_calib = calib[target_col].values
        # Only applied to volume -- a $131 lower bound looks broken (no home
        # costs that little), a 0.2-unit lower bound is a normal way to
        # express "usually zero, occasionally one." See score_agents.py.
        floor = compute_realistic_floor(resolved, target_col) if target_col == TARGET_VOLUME else None
        if floor is not None:
            print(f"  [{label}] realistic-floor snap: any min between $0 and ${floor:,.0f} is treated as $0")
        per_level = {}
        for pct in LEVELS_PCT:
            c = pct / 100.0
            lo_alpha, hi_alpha = (1 - c) / 2, (1 + c) / 2
            target_alpha = 1 - c

            t0 = time.perf_counter()
            m_lo = fit_quantile_model(train, feature_cols, target_col, lo_alpha)
            m_hi = fit_quantile_model(train, feature_cols, target_col, hi_alpha)
            elapsed = time.perf_counter() - t0

            lo_calib = np.expm1(m_lo.predict(calib[feature_cols]))
            hi_calib = np.expm1(m_hi.predict(calib[feature_cols]))
            adj = conformal_adjustment(y_calib, lo_calib, hi_calib, target_alpha)

            lo_cur = np.clip(np.expm1(m_lo.predict(current[feature_cols])) - adj, 0, None)
            hi_cur = np.clip(np.expm1(m_hi.predict(current[feature_cols])) + adj, 0, None)
            if floor is not None:
                lo_cur = snap_unrealistic_floor(lo_cur, floor)
            width = hi_cur - lo_cur

            per_level[pct] = dict(
                mean_width=float(width.mean()),
                median_width=float(np.median(width)),
                p25_width=float(np.percentile(width, 25)),
                p75_width=float(np.percentile(width, 75)),
                zero_floor_count=int((lo_cur <= 0).sum()),
                zero_floor_pct=float((lo_cur <= 0).mean()),
            )
            print(f"  [{label}] {pct}%  mean_width={width.mean():,.0f}  "
                  f"median_width={np.median(width):,.0f}  zero_floor={per_level[pct]['zero_floor_pct']:.1%}  "
                  f"[{elapsed:.1f}s]")

        results["levels"].append({"target": label, "by_pct": per_level, "realistic_floor": floor})

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[sweep] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
