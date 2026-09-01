"""Production quarterly-refresh pipeline: forecast next-12m dollar volume
and unit count for every currently active RIAR agent, at a fixed menu of
confidence levels (decision, HANDOFF.md): 50/65/70/75/80/95%.

Model: CatBoost quantile regression, depth=3, fit on log1p(target) and
expm1-transformed back -- the round-11 autoresearch winner (HANDOFF.md),
confirmed across 4 independent forward-chaining folds, superseding round
7's depth=4/raw-target config. No leave-risk feature (round 5: tested,
discarded, doesn't help).

MULTI-LEVEL OUTPUT DESIGN: the point estimate (median) doesn't depend on
the confidence level at all -- only the low/high bounds do -- so it's
fit ONCE per target and reused across every bucket, not refit per level.
Each bucket still needs its own lo/hi quantile pair (CatBoost's quantile
loss is fit per-alpha, no way to derive one from another), so the total
cost is `1 (median) + len(CONFIDENCE_LEVELS) * 2` fits per target, not
`len(CONFIDENCE_LEVELS) * 3` -- e.g. for the 6 levels below, 13 fits per
target instead of 18. The output is ONE wide spreadsheet (one row per
agent, a min/max column pair per bucket), not 6 separate files -- the
only thing that varies per bucket is 2 columns, not the whole dataset.

QUARTERLY REFRESH: every date window below (train quarters, calibration
quarters, "current" scoring snapshot) is computed relative to whatever
the newest data in the CSV is -- nothing is hardcoded to a calendar
date. Re-running this script after a new quarter of data lands
automatically: (1) trains on one more quarter of resolved history,
(2) recalibrates the conformal adjustment PER BUCKET using the most
recently COMPLETED quarter (not a frozen historical window) -- directly
addresses the calibration-drift problem the Leave model diagnosed in its
own project (a stale calibration window silently goes stale as market
conditions shift; refreshing it every quarter is the fix, same lesson,
applied here to CQR instead of Platt scaling), and (3) scores whichever
snapshot is now the newest ("current") one.

Retrains from scratch every run rather than loading a saved model
artifact -- same design choice as the Leave model's score_agents.py, for
the same reason: always reflects whatever data.py/quantile logic
currently does, no model-versioning/drift-between-runs risk to manage,
at the cost of a few minutes of retraining time per run.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from data import (
    CATEGORICAL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    get_feature_columns,
    load_clean,
)
from train_baseline import TARGET_UNITS, TARGET_VOLUME

CALIB_QUARTERS = 2  # most-recently-completed resolved quarters reserved for conformal calibration
MODEL_PARAMS = dict(iterations=500, depth=3, learning_rate=0.03, verbose=False)  # catboost_d3, round 9 production config (see HANDOFF.md)

CONFIDENCE_LEVELS = [50, 65, 70, 75, 80, 95]  # user-selected menu, not evenly spaced by design --
# denser in the 50-80% "cheap zone" the sweep analysis identified, capped at 95% rather than
# offering 96-99%, which the same analysis showed is a poor trade (see confidence_level_analysis.html)

OUTPUT_PATH = "current_agent_sales_forecast.csv"


def build_full_frame() -> pd.DataFrame:
    """Same t+12mo self-join as data.py::build_targets, but WITHOUT
    dropping the unresolved rows -- those unresolved rows (the newest
    snapshot, which by construction can never have a future outcome yet)
    are exactly the agents this script needs to score. build_model_frame()
    intentionally drops them for training-only use cases; this is the one
    place that also needs them, so the merge logic is duplicated here
    rather than changing what build_model_frame() returns for every other
    caller."""
    df = load_clean()
    df = df.sort_values(["mls_agent_id", "snapshot_date"]).reset_index(drop=True)

    future = df[["mls_agent_id", "snapshot_date", "volume_12m", "units_12m"]].copy()
    future = future.rename(columns={
        "volume_12m": "target_volume_next_12m",
        "units_12m": "target_units_next_12m",
    })
    future["snapshot_date"] = future["snapshot_date"] - pd.DateOffset(months=12)

    merged = df.merge(future, on=["mls_agent_id", "snapshot_date"], how="left")
    merged = pd.get_dummies(merged, columns=CATEGORICAL_FEATURE_COLUMNS, prefix="brand")
    return merged


def conformal_adjustment(y_calib, lo_calib, hi_calib, target_alpha):
    """Split-conformal (CQR) adjustment for an arbitrary target miscoverage
    rate -- generalizes quantile_model.py's version (hardcoded to the 95%
    case, target_alpha=0.05) to the multi-level menu used here."""
    nonconformity = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    n = len(nonconformity)
    level = min(1.0, np.ceil((n + 1) * (1 - target_alpha)) / n)
    return np.quantile(nonconformity, level)


def fit_quantile_model(train, feature_cols, target_col, alpha):
    """Fits on log1p(target), not the raw target -- round 11's confirmed
    winner (see HANDOFF.md). Valid because quantiles commute with any
    monotonic transform (unlike a mean/squared-error target, where this
    trick would NOT be valid). Callers must expm1() the predictions."""
    m = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", **MODEL_PARAMS)
    y_log = np.log1p(train[target_col].values)
    m.fit(train[feature_cols], y_log)
    return m


REALISTIC_FLOOR_PCTILE = 1  # see compute_realistic_floor() -- only applied to TARGET_VOLUME


def compute_realistic_floor(resolved, target_col, pctile=REALISTIC_FLOOR_PCTILE):
    """The Nth percentile of historically observed NON-ZERO outcomes for
    this target -- e.g. for volume, "if an agent sold anything at all,
    99% of the time they sold at least this much." A raw quantile
    prediction strictly between $0 and this floor doesn't describe a real
    possible outcome (there's no such thing as a $131 home) -- it's
    continuous-regression noise where the true answer is "essentially
    zero." Computed fresh from the current resolved training data every
    run, same self-updating spirit as everything else in this pipeline,
    not a hardcoded number that could go stale.
    Empirically: 1st percentile of non-zero target_volume_next_12m ~
    $165,000 (2026-07 data) -- deliberately NOT derived from the raw
    per-transaction price_min_12m column (a different, noisier
    granularity than the annual-volume target this model actually
    predicts)."""
    nonzero = resolved.loc[resolved[target_col] > 0, target_col]
    return float(nonzero.quantile(pctile / 100))


def snap_unrealistic_floor(lo, floor, direction="down"):
    """lo values strictly between 0 and `floor` are snapped to one of the
    two ends -- never left at a nonsense in-between value like $131.

    direction="down" (default, used by the UNCONDITIONAL forecast): snap
    to exactly 0, since the honest interpretation of a tiny-but-nonzero
    prediction, when 0 is itself a real possible outcome, is "most likely
    no sales at all," not "guaranteed at least the floor amount."

    direction="up" (used by hurdle_model.py's CONDITIONAL-on-nonzero
    range): snap UP to the floor instead -- 0 is not a valid answer there
    by construction (the whole point of conditioning on "given they sell
    something" is that the range never collapses back to 0), so the
    honest floor for "smallest realistic nonzero sale" is the floor
    itself, not 0.

    Values already at 0 or at/above the floor are left untouched either
    way."""
    if direction == "down":
        return np.where((lo > 0) & (lo < floor), 0.0, lo)
    return np.where((lo > 0) & (lo < floor), floor, lo)


def forecast_target_multi(resolved, current, feature_cols, target_col, levels, zero_floor="down"):
    """Returns (point_estimate, {level: (lo, hi)}, meta). Fits the median
    model ONCE (confidence-independent) and one lo/hi quantile pair PER
    level -- see module docstring for why this is 2N+1 fits, not 3N.

    `zero_floor` controls snap_unrealistic_floor()'s direction (only
    matters for TARGET_VOLUME, see compute_realistic_floor); pass None to
    skip the floor snap entirely."""
    quarters = sorted(resolved["snapshot_date"].unique())
    calib_q = quarters[-CALIB_QUARTERS:]
    train_q = quarters[:-CALIB_QUARTERS]

    train = resolved[resolved["snapshot_date"].isin(train_q)]
    calib = resolved[resolved["snapshot_date"].isin(calib_q)]
    y_calib = calib[target_col].values

    med_model = fit_quantile_model(train, feature_cols, target_col, 0.5)
    point = np.clip(np.expm1(med_model.predict(current[feature_cols])), 0, None)

    # Only applied to volume: a $131 lower bound looks broken (no home
    # costs that little); a 0.2-unit lower bound is a normal, already-
    # accepted way to express "usually zero, occasionally one" for a count.
    floor = compute_realistic_floor(resolved, target_col) if target_col == TARGET_VOLUME else None

    by_level = {}
    adjustments = {}
    for level in levels:
        c = level / 100.0
        lo_alpha, hi_alpha = (1 - c) / 2, (1 + c) / 2
        target_alpha = 1 - c

        m_lo = fit_quantile_model(train, feature_cols, target_col, lo_alpha)
        m_hi = fit_quantile_model(train, feature_cols, target_col, hi_alpha)

        # expm1 immediately after predicting -- everything downstream
        # (conformal adjustment, clipping) operates in the original
        # dollar/unit scale, same as every non-transformed round before this.
        lo_calib = np.expm1(m_lo.predict(calib[feature_cols]))
        hi_calib = np.expm1(m_hi.predict(calib[feature_cols]))
        adj = conformal_adjustment(y_calib, lo_calib, hi_calib, target_alpha)

        lo_cur = np.clip(np.expm1(m_lo.predict(current[feature_cols])) - adj, 0, None)
        hi_cur = np.clip(np.expm1(m_hi.predict(current[feature_cols])) + adj, 0, None)
        if floor is not None and zero_floor is not None:
            lo_cur = snap_unrealistic_floor(lo_cur, floor, direction=zero_floor)
        by_level[level] = (lo_cur, hi_cur)
        adjustments[level] = adj

    meta = dict(train_quarters=train_q, calib_quarters=calib_q, adjustments=adjustments, floor=floor)
    return point, by_level, meta


def score_current_agents() -> pd.DataFrame:
    """Hurdle output (round 15/16, HANDOFF.md): every target is reported as
    TWO honest numbers rather than one range that has to do both jobs --
    `{label}_pct_zero` (calibrated probability of $0 volume / 0 units) and
    a conditional-on-nonzero point estimate + range (`{label}_point_estimate`,
    `{label}_min_{level}`/`{label}_max_{level}` -- "if they sell anything,
    expect $X-$Y"). See hurdle_model.py."""
    from hurdle_model import forecast_conditional_nonzero, forecast_pzero

    print("[score] loading data and building t+12mo targets...")
    full = build_full_frame()
    resolved = full.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    feature_cols = get_feature_columns(full)

    current_snapshot_date = full["snapshot_date"].max()
    current = full[full["snapshot_date"] == current_snapshot_date].reset_index(drop=True)
    print(f"[score] current snapshot: {current_snapshot_date.date()}  "
          f"({len(current):,} active agents to forecast)")
    print(f"[score] resolved training history: {len(resolved):,} agent-quarters, "
          f"{resolved['snapshot_date'].nunique()} quarters")

    out = current[["mls_agent_id", "office_name", "agent_city", "agent_state"]].copy()
    out["snapshot_date"] = current_snapshot_date

    for target_col, label in [(TARGET_VOLUME, "volume"), (TARGET_UNITS, "units")]:
        pct_zero, pz_meta = forecast_pzero(resolved, current, feature_cols, target_col)
        out[f"{label}_pct_zero"] = pct_zero
        print(f"[score] {label} P(zero): trained on {len(pz_meta['train_quarters'])} quarters, "
              f"isotonic-calibrated on {pz_meta['calib_quarters'][0].date()}.."
              f"{pz_meta['calib_quarters'][-1].date()}, base rate {pz_meta['base_rate']:.1%}")

        point, by_level, meta = forecast_conditional_nonzero(
            resolved, current, feature_cols, target_col, CONFIDENCE_LEVELS)
        out[f"{label}_point_estimate"] = point
        for level in CONFIDENCE_LEVELS:
            lo, hi = by_level[level]
            out[f"{label}_min_{level}"] = lo
            out[f"{label}_max_{level}"] = hi

        adj_str = ", ".join(f"{lvl}%={meta['adjustments'][lvl]:.3g}" for lvl in CONFIDENCE_LEVELS)
        print(f"[score] {label} (conditional on nonzero): trained on {len(meta['train_quarters'])} quarters "
              f"({meta['train_quarters'][0].date()}..{meta['train_quarters'][-1].date()}), "
              f"conformal-calibrated on {meta['calib_quarters'][0].date()}..{meta['calib_quarters'][-1].date()} "
              f"(the most recently completed history at run time)")
        print(f"[score] {label} conformal adjustments -> {adj_str}")
        if meta["floor"] is not None:
            print(f"[score] {label} realistic-floor snap: any min between $0 and "
                  f"${meta['floor']:,.0f} is raised to ${meta['floor']:,.0f} "
                  f"(never 0 -- this range is conditional on nonzero)")

    out = out.sort_values("volume_point_estimate", ascending=False).reset_index(drop=True)
    out = round_output_columns(out)
    return out


def round_output_columns(out: pd.DataFrame) -> pd.DataFrame:
    """The raw CQR/isotonic pipeline produces full float64 precision (e.g.
    $126,241,846.06585617) -- meaningless past the nearest dollar/tenth-
    unit, and it's exactly the kind of long decimal that spreadsheet tools
    render in scientific notation or garble the column width on. Round
    each column to the precision the number actually means: whole dollars
    for volume, one decimal place for units (a fractional unit is already
    the normal way this pipeline expresses "usually 0, occasionally 1" --
    round 14 -- so keep one decimal, don't round to an integer), and 4
    decimal places for the pct_zero probabilities."""
    out = out.copy()
    for col in out.columns:
        if col.startswith("volume_") and col != "volume_pct_zero":
            out[col] = out[col].round(0).astype("int64")  # whole dollars -- no ".0" suffix in the CSV
        elif col.startswith("units_") and col != "units_pct_zero":
            out[col] = out[col].round(1)
        elif col.endswith("_pct_zero"):
            out[col] = out[col].round(4)
    return out


def main():
    out = score_current_agents()
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[score] scored {len(out):,} agents -> {OUTPUT_PATH}  ({len(out.columns)} columns)")
    print(f"[score] columns: {list(out.columns)}")

    print("\nTop 5 agents by predicted volume (95% bucket shown, conditional on nonzero):")
    cols = ["mls_agent_id", "office_name", "volume_pct_zero", "volume_min_95",
            "volume_point_estimate", "volume_max_95"]
    with pd.option_context("display.max_colwidth", 22, "display.width", 160):
        print(out.head(5)[cols].to_string(index=False))

    print("\nA representative agent (median by predicted volume), both halves of the hurdle:")
    row = out.iloc[len(out) // 2]
    print(f"  P(zero): {row.volume_pct_zero:.1%} volume  |  {row.units_pct_zero:.1%} units")
    print(f"  if selling anything -- point estimate: ${row.volume_point_estimate:,.0f}  "
          f"({row.units_point_estimate:.1f} units)")
    for level in CONFIDENCE_LEVELS:
        print(f"  {level:>3d}%:  ${row[f'volume_min_{level}']:,.0f} - ${row[f'volume_max_{level}']:,.0f}   "
              f"|  {row[f'units_min_{level}']:.1f} - {row[f'units_max_{level}']:.1f} units")


if __name__ == "__main__":
    main()
