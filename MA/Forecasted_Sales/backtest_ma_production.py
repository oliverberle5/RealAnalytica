"""Held-out backtest of the production hurdle+CQR pipeline, MA and RI side by side.

METHODOLOGY mirrors the RI project's rounds 3-4 (Forecasted_Sales_Algorithm/
HANDOFF.md) rather than inventing a new one, so the two states' numbers mean the
same thing:

  TRAIN  -- all resolved quarters except the last (CALIB_QUARTERS + N_TEST)
  CALIB  -- the next 2 quarters, used ONLY for the conformal adjustment
  TEST   -- the final 4 resolved quarters, never touched until scoring

Measuring coverage on the data used to build the range would be circular, which
is why TEST is carved out beyond the calibration window.

WHY THE TERCILE BREAKDOWN IS REPORTED, NOT JUST AVERAGE COVERAGE: HANDOFF.md's
round 3 established that overall coverage near the nominal level can hide a
badly-behaved model -- a range can land at ~95% on average while massively
over-covering small producers (99%+, a uselessly wide range) and under-covering
large ones. Coverage is therefore broken out by predicted-size tercile, and the
SPREAD across terciles is reported as a first-class number. A model with even
coverage across terciles is better than one with a prettier average.

Run:  python backtest_ma_production.py            # MA (mlspin)
      python backtest_ma_production.py --mls riar # RI, for comparison
      python backtest_ma_production.py --both     # both, side by side
"""
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import data as data_module
from data import CATEGORICAL_FEATURE_COLUMNS, TARGET_COLUMNS, get_feature_columns
from score_agents import CALIB_QUARTERS, MODEL_PARAMS
from train_baseline import TARGET_UNITS, TARGET_VOLUME

N_TEST = 4  # final resolved quarters held out entirely

# Two of the production menu's levels -- the widest and a mid one people
# actually use. Running all six would triple the runtime for no extra insight
# into whether the port is sound.
BANDS = {
    "80%": dict(lo=0.10, hi=0.90, target_alpha=0.20),
    "95%": dict(lo=0.025, hi=0.975, target_alpha=0.05),
}


def build_frame():
    """Same t+12mo self-join as score_agents.build_full_frame, but local so the
    mls override below is picked up."""
    df = data_module.load_clean()
    df = df.sort_values(["mls_agent_id", "snapshot_date"]).reset_index(drop=True)
    future = df[["mls_agent_id", "snapshot_date", "volume_12m", "units_12m"]].copy()
    future = future.rename(columns={
        "volume_12m": "target_volume_next_12m",
        "units_12m": "target_units_next_12m",
    })
    future["snapshot_date"] = future["snapshot_date"] - pd.DateOffset(months=12)
    merged = df.merge(future, on=["mls_agent_id", "snapshot_date"], how="left")
    return pd.get_dummies(merged, columns=CATEGORICAL_FEATURE_COLUMNS, prefix="brand")


def fit_q(train, feature_cols, target_col, alpha):
    m = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", **MODEL_PARAMS)
    m.fit(train[feature_cols], np.log1p(train[target_col].values))
    return m


def conformal_adjustment(y_calib, lo_calib, hi_calib, target_alpha):
    nonconformity = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    n = len(nonconformity)
    level = min(1.0, np.ceil((n + 1) * (1 - target_alpha)) / n)
    return np.quantile(nonconformity, level)


def evaluate(mls_code: str) -> list[dict]:
    data_module.MLS_CODE = mls_code
    full = build_frame()
    resolved = full.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    feature_cols = get_feature_columns(full)

    quarters = sorted(resolved["snapshot_date"].unique())
    test_q = quarters[-N_TEST:]
    calib_q = quarters[-(N_TEST + CALIB_QUARTERS):-N_TEST]
    train_q = quarters[:-(N_TEST + CALIB_QUARTERS)]

    train = resolved[resolved["snapshot_date"].isin(train_q)]
    calib = resolved[resolved["snapshot_date"].isin(calib_q)]
    test = resolved[resolved["snapshot_date"].isin(test_q)]

    print(f"\n{'#' * 78}\n# {mls_code.upper()}  ({'MA' if mls_code == 'mlspin' else 'RI'})\n{'#' * 78}")
    print(f"train {train_q[0].date()}..{train_q[-1].date()} ({len(train):,} rows) | "
          f"calib {calib_q[0].date()}..{calib_q[-1].date()} ({len(calib):,}) | "
          f"test {test_q[0].date()}..{test_q[-1].date()} ({len(test):,})")

    rows = []
    for target_col, label, fmt in [
        (TARGET_VOLUME, "volume", lambda x: f"${x:,.0f}"),
        (TARGET_UNITS, "units", lambda x: f"{x:,.2f}"),
    ]:
        y_test = test[target_col].values

        m_med = fit_q(train, feature_cols, target_col, 0.5)
        pred_med = np.clip(np.expm1(m_med.predict(test[feature_cols])), 0, None)
        mae = np.mean(np.abs(y_test - pred_med))
        # Median APE over nonzero actuals only -- APE is undefined at y=0, and
        # the zero population is what the hurdle's P(zero) half exists to describe.
        nz = y_test > 0
        mdape = np.median(np.abs(y_test[nz] - pred_med[nz]) / y_test[nz]) if nz.any() else np.nan

        print(f"\n  {label.upper()}   n_test={len(test):,}  zero-rate={100*(y_test==0).mean():.1f}%")
        print(f"    point (median q):  MAE {fmt(mae)}   MdAPE {mdape:.1%}   "
              f"naive-persistence MAE {fmt(np.mean(np.abs(y_test - test['volume_12m' if label=='volume' else 'units_12m'].values)))}")

        # Terciles by PREDICTED size (what you'd know at forecast time).
        terc = pd.qcut(pred_med, 3, labels=["small", "mid", "large"], duplicates="drop")

        for band_name, cfg in BANDS.items():
            m_lo = fit_q(train, feature_cols, target_col, cfg["lo"])
            m_hi = fit_q(train, feature_cols, target_col, cfg["hi"])
            lo_c = np.expm1(m_lo.predict(calib[feature_cols]))
            hi_c = np.expm1(m_hi.predict(calib[feature_cols]))
            adj = conformal_adjustment(calib[target_col].values, lo_c, hi_c, cfg["target_alpha"])

            lo = np.clip(np.expm1(m_lo.predict(test[feature_cols])) - adj, 0, None)
            hi = np.clip(np.expm1(m_hi.predict(test[feature_cols])) + adj, 0, None)
            covered = (y_test >= lo) & (y_test <= hi)
            width = hi - lo

            by_terc = pd.DataFrame({"t": terc, "c": covered}).groupby("t", observed=True)["c"].mean()
            spread = (by_terc.max() - by_terc.min()) * 100

            print(f"    {band_name} band: coverage {covered.mean():.1%} (nominal {band_name})  "
                  f"mean width {fmt(width.mean())}  conformal adj {fmt(adj)}")
            print(f"              by tercile: " + "  ".join(f"{k} {v:.1%}" for k, v in by_terc.items())
                  + f"   spread {spread:.1f}pts")

            rows.append(dict(mls=mls_code, target=label, band=band_name,
                             coverage=covered.mean(), nominal=int(band_name[:-1]) / 100,
                             mean_width=width.mean(), tercile_spread_pts=spread,
                             mae=mae, mdape=mdape, n_test=len(test)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mls", default="mlspin")
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()

    codes = ["mlspin", "riar"] if args.both else [args.mls]
    allrows = []
    for c in codes:
        allrows.extend(evaluate(c))

    out = pd.DataFrame(allrows)
    out.to_csv("backtest_ma_production_results.csv", index=False)
    print(f"\nSaved backtest_ma_production_results.csv ({len(out)} rows)")

    if args.both:
        print("\n" + "=" * 78 + "\nMA vs RI -- coverage should land near nominal for BOTH; "
              "\nwidth is not comparable across states (different market sizes).\n" + "=" * 78)
        piv = out.pivot_table(index=["target", "band"], columns="mls",
                              values=["coverage", "tercile_spread_pts"])
        print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
