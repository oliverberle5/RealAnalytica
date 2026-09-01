"""
What does the range look like at 75% confidence instead of 95%?

Same production model (CatBoost depth=4, round 7 winner), same train/calib
windows as score_agents.py, same CQR procedure -- just two different
quantile pairs: (2.5%, 97.5%) for a 95% interval vs. (12.5%, 87.5%) for a
75% interval. Both get their own conformal calibration (a 75% interval
needs a smaller conformal correction than a 95% one -- the target
miscoverage rate changes from 5% to 25%).

Answers two concrete questions: how much narrower is a 75% range in
practice, and how many fewer agents have a $0 lower bound (the model
admitting "we truly can't rule out zero closings this year" at 95%
confidence) once we stop insisting on covering the extreme tail.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from data import get_feature_columns, TARGET_COLUMNS
from score_agents import build_full_frame, CALIB_QUARTERS, MODEL_PARAMS
from train_baseline import TARGET_UNITS, TARGET_VOLUME

QUANTILE_SETS = {
    "95%": dict(lo=0.025, hi=0.975, target_alpha=0.05),
    "75%": dict(lo=0.125, hi=0.875, target_alpha=0.25),
    "60%": dict(lo=0.20, hi=0.80, target_alpha=0.40),
}


def fit_quantile_model(train, feature_cols, target_col, alpha):
    """log1p(target), expm1'd on predict -- matches score_agents.py's round-11
    production config. Caller must expm1() this model's raw .predict() output."""
    m = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", **MODEL_PARAMS)
    m.fit(train[feature_cols], np.log1p(train[target_col].values))
    return m


def conformal_adjustment(y_calib, lo_calib, hi_calib, target_alpha):
    nonconformity = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    n = len(nonconformity)
    level = min(1.0, np.ceil((n + 1) * (1 - target_alpha)) / n)
    return np.quantile(nonconformity, level)


def forecast_at_level(resolved, current, feature_cols, target_col, level_cfg):
    quarters = sorted(resolved["snapshot_date"].unique())
    calib_q = quarters[-CALIB_QUARTERS:]
    train_q = quarters[:-CALIB_QUARTERS]
    train = resolved[resolved["snapshot_date"].isin(train_q)]
    calib = resolved[resolved["snapshot_date"].isin(calib_q)]

    m_lo = fit_quantile_model(train, feature_cols, target_col, level_cfg["lo"])
    m_hi = fit_quantile_model(train, feature_cols, target_col, level_cfg["hi"])

    lo_calib = np.expm1(m_lo.predict(calib[feature_cols]))
    hi_calib = np.expm1(m_hi.predict(calib[feature_cols]))
    y_calib = calib[target_col].values
    adj = conformal_adjustment(y_calib, lo_calib, hi_calib, level_cfg["target_alpha"])

    lo_cur = np.clip(np.expm1(m_lo.predict(current[feature_cols])) - adj, 0, None)
    hi_cur = np.clip(np.expm1(m_hi.predict(current[feature_cols])) + adj, 0, None)
    return lo_cur, hi_cur, adj


def main():
    print("[compare] loading data...")
    full = build_full_frame()
    resolved = full.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    feature_cols = get_feature_columns(full)
    current_snapshot_date = full["snapshot_date"].max()
    current = full[full["snapshot_date"] == current_snapshot_date].reset_index(drop=True)
    print(f"[compare] current snapshot: {current_snapshot_date.date()}  ({len(current):,} agents)\n")

    results = {}
    for target_col, label, fmt in [
        (TARGET_VOLUME, "volume", lambda x: f"${x:,.0f}"),
        (TARGET_UNITS, "units", lambda x: f"{x:.1f}"),
    ]:
        print(f"{'=' * 70}\n{label.upper()}\n{'=' * 70}")
        level_results = {}
        for level_name, cfg in QUANTILE_SETS.items():
            lo, hi, adj = forecast_at_level(resolved, current, feature_cols, target_col, cfg)
            width = hi - lo
            zero_lo_count = int((lo <= 0).sum())
            level_results[level_name] = dict(lo=lo, hi=hi, width=width)

            print(f"\n  {level_name} interval (quantiles {cfg['lo']}-{cfg['hi']}, conformal adj={fmt(adj)}):")
            print(f"    mean width: {fmt(width.mean())}   median width: {fmt(np.median(width))}")
            print(f"    agents with $0/0-unit lower bound: {zero_lo_count:,} / {len(current):,} "
                  f"({zero_lo_count / len(current):.1%})")

        # narrowing + zero-bound comparison, each level vs. the 95% baseline
        w95 = level_results["95%"]["width"]
        zero95 = int((level_results["95%"]["lo"] <= 0).sum())
        for level_name in level_results:
            if level_name == "95%":
                continue
            w = level_results[level_name]["width"]
            narrowing = 1 - (w.mean() / w95.mean())
            zero_n = int((level_results[level_name]["lo"] <= 0).sum())
            print(f"\n  --> going from 95% to {level_name}: range narrows by {narrowing:.1%} on average; "
                  f"agents with a $0/0-unit floor drops from {zero95:,} to {zero_n:,} "
                  f"({zero95 - zero_n:,} fewer, a {(zero95 - zero_n) / max(zero95,1):.1%} reduction)")

        results[target_col] = level_results

    # side-by-side example table using volume rank as the reference ordering for both targets
    vol = results[TARGET_VOLUME]
    units = results[TARGET_UNITS]
    point_rank = (vol["95%"]["lo"] + vol["95%"]["hi"]) / 2  # rough point-ish rank for picking representative rows
    order = np.argsort(-point_rank)
    print(f"\n{'=' * 70}\nSIDE-BY-SIDE EXAMPLES\n{'=' * 70}")
    for pct in (0.25, 0.5, 0.75):
        i = order[int(len(order) * pct)]
        print(f"\n  ~{int(pct*100)}th percentile agent:")
        for level_name in QUANTILE_SETS:
            print(f"    volume  {level_name}: ${vol[level_name]['lo'][i]:,.0f} - ${vol[level_name]['hi'][i]:,.0f}   "
                  f"units {level_name}: {units[level_name]['lo'][i]:.1f} - {units[level_name]['hi'][i]:.1f}")


if __name__ == "__main__":
    main()
