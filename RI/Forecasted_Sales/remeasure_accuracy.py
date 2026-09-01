"""Re-measure every headline accuracy figure for the production pipeline,
on a genuinely held-out test window, under the current population.

WHY THIS EXISTS: `final_accuracy_check.log` (R2 0.7966 volume / 0.8593
units, Spearman 0.7243/0.8017) was a one-off run on 2026-07-21 -- BEFORE
the team-account exclusion (2026-07-28, MAE -9%), before "00001"
(2026-08-11) and before "0014" (2026-08-22). No script for it was kept.
Every figure it reports describes a model that no longer exists.
HANDOFF.md's STOP box says plainly: "before restating any headline figure,
run the backtest under several seeds / row orders and report the spread."
This is that run.

SPLIT. The production pipeline (score_agents.py) internally reserves its
last CALIB_QUARTERS resolved quarters for conformal / isotonic calibration
and trains on everything earlier. To test it, the whole thing is shifted
back by TEST_QUARTERS: the last 2 resolved quarters are held out entirely
and never passed to any fit, and the production functions are handed the
remainder as their `resolved` history -- so they carve their own
train/calib split out of it exactly as they do in production. Nothing here
re-implements the model; `forecast_pzero` and `forecast_conditional_nonzero`
are imported and called unmodified. Same test window (2025-04-01..2025-07-01)
the team-exclusion backtest used on 2026-07-28, so the numbers are
comparable to the ones already in HANDOFF.md.

WHAT IT REPORTS, and why more than one number:
 - P(zero) half: Brier + AUC. This half IS a classifier, so AUC is defined
   here even though it is meaningless for the forecast as a whole.
 - Conditional-on-nonzero point accuracy: the number the spreadsheet's
   `*_point_estimate` column actually is. R2 / MAE / median APE.
 - Expected-value point accuracy: (1 - P(zero)) x conditional point, i.e.
   both halves of the hurdle recombined into one unconditional number,
   scored on ALL test rows including the true zeros. This is the honest
   whole-model figure; the conditional one is scored on a subpopulation
   the model itself picked out.
 - Ranking: Spearman, and top-decile / top-quintile capture -- "of the
   agents who really were the top 10% of producers, how many did we have
   in our predicted top 10%". This is the closest true analog to what AUC
   measured for the Leave model, and unlike R2 it is not inflatable by
   agent size being persistent.
 - Interval coverage AND width at all six production levels.
 - Naive persistence ("they sell next year what they sold last year") on
   the identical test rows, for every point metric. R2 without a baseline
   is not interpretable here: backtest_long_horizons_results.csv already
   showed persistence scoring R2 ~0.807 on volume at this horizon, so a
   model R2 of ~0.80 is not by itself evidence of skill.

SEEDS. Both learners subsample by row order, and HANDOFF.md item 3
established a ~2% median per-agent run-to-run noise floor from the seed
alone. Every metric is therefore run at 3 seeds and reported as a spread,
not a point value. A delta smaller than the seed spread is not a result.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score

import hurdle_model
import score_agents
from data import TARGET_COLUMNS, get_feature_columns
from score_agents import CONFIDENCE_LEVELS, build_full_frame
from train_baseline import TARGET_UNITS, TARGET_VOLUME

TEST_QUARTERS = 2
SEEDS = [0, 1, 2]

# The agent's own current trailing-12m production, i.e. the naive
# "same as last year" forecast. Kept as a feature by the real model too.
PERSISTENCE_COLUMN = {TARGET_VOLUME: "volume_12m", TARGET_UNITS: "units_12m"}


def set_seed(seed):
    """Both halves of the hurdle, patched at the module global the
    production code reads at call time -- so the imported production
    functions run unmodified but with a different bootstrap draw."""
    score_agents.MODEL_PARAMS = dict(score_agents.MODEL_PARAMS, random_seed=seed)
    hurdle_model.ZERO_CLASSIFIER_PARAMS = dict(
        hurdle_model.ZERO_CLASSIFIER_PARAMS, random_state=seed)


def capture_at(y_true, y_pred, frac):
    """Of the true top `frac` of producers, what share did the model also
    rank in its own top `frac`. Ties in y_true at 0 are irrelevant here --
    the top decile is far above zero for both targets."""
    k = max(1, int(round(len(y_true) * frac)))
    true_top = set(np.argsort(-np.asarray(y_true), kind="stable")[:k].tolist())
    pred_top = set(np.argsort(-np.asarray(y_pred), kind="stable")[:k].tolist())
    return len(true_top & pred_top) / k


def point_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true > 0
    return dict(
        n=len(y_true),
        r2=r2_score(y_true, y_pred),
        mae=float(np.mean(np.abs(y_true - y_pred))),
        # median, not mean: a handful of agents whose true production is a
        # few thousand dollars make a mean APE meaningless.
        median_ape=float(np.median(np.abs(y_true[nz] - y_pred[nz]) / y_true[nz])) if nz.any() else np.nan,
        spearman=float(stats.spearmanr(y_true, y_pred).correlation),
        capture_top10=capture_at(y_true, y_pred, 0.10),
        capture_top20=capture_at(y_true, y_pred, 0.20),
    )


def run_one(resolved_hist, test, feature_cols, seed):
    set_seed(seed)
    out = {}
    for target_col, label in [(TARGET_VOLUME, "volume"), (TARGET_UNITS, "units")]:
        y_test = test[target_col].values.astype(float)

        pct_zero, _ = hurdle_model.forecast_pzero(
            resolved_hist, test, feature_cols, target_col)
        point, by_level, _ = hurdle_model.forecast_conditional_nonzero(
            resolved_hist, test, feature_cols, target_col, CONFIDENCE_LEVELS)

        is_zero = (y_test == 0).astype(int)
        res = dict(
            pzero_brier=brier_score_loss(is_zero, np.clip(pct_zero, 0, 1)),
            pzero_auc=roc_auc_score(is_zero, pct_zero),
            zero_rate=float(is_zero.mean()),
        )

        # Whole model, unconditional: both hurdle halves recombined.
        ev = (1.0 - pct_zero) * point
        res["ev"] = point_metrics(y_test, ev)
        res["ev_naive"] = point_metrics(y_test, test[PERSISTENCE_COLUMN[target_col]].values)

        # Conditional-on-nonzero: the spreadsheet's point_estimate column,
        # scored only where the outcome really was nonzero (what the
        # 2026-07-28 team-exclusion backtest reported).
        nz = y_test > 0
        res["cond"] = point_metrics(y_test[nz], point[nz])
        res["cond_naive"] = point_metrics(
            y_test[nz], test[PERSISTENCE_COLUMN[target_col]].values[nz])

        # Bands are conditional on nonzero by construction, so coverage is
        # measured on the nonzero rows -- measuring it on true zeros would
        # score the band for missing outcomes it explicitly does not claim.
        res["bands"] = {}
        for lvl in CONFIDENCE_LEVELS:
            lo, hi = by_level[lvl]
            inside = (y_test[nz] >= lo[nz]) & (y_test[nz] <= hi[nz])
            res["bands"][lvl] = dict(
                coverage=float(inside.mean()),
                mean_width=float(np.mean(hi[nz] - lo[nz])),
                median_width=float(np.median(hi[nz] - lo[nz])),
            )
        out[label] = res
    return out


def spread(vals, fmt):
    v = np.array(vals, float)
    return f"{fmt(v.mean())}  [{fmt(v.min())}..{fmt(v.max())}]"


def main():
    exclude_non_member = "--no-exclusions" not in sys.argv
    # Isolates the 2026-08-22 adoption: everything as production, except
    # "0014" is put back. The --no-exclusions arm cannot do this job on its
    # own -- it also puts back "12345" and "00001", two records large enough
    # to move R2 by ~0.05 all by themselves, which would swamp a 10-row
    # training change.
    legacy_only = "--legacy-exclusions" in sys.argv
    if legacy_only:
        import data as data_mod
        data_mod.NON_MLS_MEMBER_AGENT_IDS = {"12345", "00001"}
        print("[remeasure] exclusion set forced to {12345, 00001} -- 0014 PUT BACK")
    if not exclude_non_member:
        # Population A/B: run the whole thing with the placeholder
        # exclusions OFF, so 0014's effect can be read against the seed
        # spread instead of against zero.
        import data as data_mod
        orig = data_mod.load_clean
        patched = lambda *a, **k: orig(*a, **{**k, "exclude_non_member": False})
        data_mod.load_clean = patched
        score_agents.load_clean = patched

    full = build_full_frame()
    resolved = full.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    feature_cols = get_feature_columns(full)

    quarters = sorted(resolved["snapshot_date"].unique())
    test_q = quarters[-TEST_QUARTERS:]
    hist_q = quarters[:-TEST_QUARTERS]
    resolved_hist = resolved[resolved["snapshot_date"].isin(hist_q)].reset_index(drop=True)
    test = resolved[resolved["snapshot_date"].isin(test_q)].reset_index(drop=True)

    def d(x):
        return pd.Timestamp(x).date()

    print(f"[remeasure] exclusions {'ON' if exclude_non_member else 'OFF'}")
    print(f"[remeasure] resolved panel: {len(resolved):,} agent-quarters, "
          f"{len(quarters)} quarters, {resolved['mls_agent_id'].nunique():,} agents")
    print(f"[remeasure] history given to the model: {len(resolved_hist):,} rows "
          f"({d(hist_q[0])}..{d(hist_q[-1])}) -- it splits its own calib window off the end")
    print(f"[remeasure] HELD-OUT TEST: {len(test):,} rows ({d(test_q[0])}..{d(test_q[-1])}), "
          f"{test['mls_agent_id'].nunique():,} agents, never seen by any fit\n")

    runs = []
    for seed in SEEDS:
        print(f"[remeasure] seed {seed} ...", flush=True)
        runs.append(run_one(resolved_hist, test, feature_cols, seed))

    for label in ["volume", "units"]:
        money = label == "volume"
        f_amt = (lambda x: f"${x:,.0f}") if money else (lambda x: f"{x:.2f}")

        def f_r(x):
            return f"{x:.4f}"

        def f_pct(x):
            return f"{x:.1%}"

        R = [r[label] for r in runs]

        print(f"\n{'=' * 78}\n{label.upper()}   (3 seeds: mean [min..max])\n{'=' * 78}")
        print(f"  true zero-rate in test window     {R[0]['zero_rate']:.1%}")
        print(f"  P(zero) Brier                     {spread([r['pzero_brier'] for r in R], f_r)}")
        print(f"  P(zero) AUC                       {spread([r['pzero_auc'] for r in R], f_r)}")

        for key, title in [("ev", "WHOLE MODEL, unconditional  (1-P(zero)) x point, all test rows"),
                           ("cond", "CONDITIONAL ON NONZERO  (the spreadsheet's point_estimate)")]:
            nkey = key + "_naive"
            print(f"\n  -- {title}")
            print(f"     n                              {R[0][key]['n']:,}")
            for m, f in [("r2", f_r), ("mae", f_amt), ("median_ape", f_pct),
                         ("spearman", f_r), ("capture_top10", f_pct), ("capture_top20", f_pct)]:
                model = spread([r[key][m] for r in R], f)
                naive = f(R[0][nkey][m])
                print(f"     {m:<28s}   {model:<36s} naive-persistence: {naive}")

        print("\n  -- INTERVAL COVERAGE / WIDTH (nonzero test rows)")
        print(f"     {'level':<8}{'coverage':<28}{'mean width':<32}{'median width'}")
        for lvl in CONFIDENCE_LEVELS:
            cov = spread([r["bands"][lvl]["coverage"] for r in R], f_pct)
            mw = spread([r["bands"][lvl]["mean_width"] for r in R], f_amt)
            mdw = f_amt(np.mean([r["bands"][lvl]["median_width"] for r in R]))
            print(f"     {lvl:<8}{cov:<28}{mw:<32}{mdw}")

    rows = []
    for seed, r in zip(SEEDS, runs):
        for label in ["volume", "units"]:
            dd = r[label]
            row = dict(seed=seed, target=label, exclusions=exclude_non_member,
                       zero_rate=dd["zero_rate"], pzero_brier=dd["pzero_brier"],
                       pzero_auc=dd["pzero_auc"])
            for key in ["ev", "cond", "ev_naive", "cond_naive"]:
                for m, v in dd[key].items():
                    row[f"{key}_{m}"] = v
            for lvl in CONFIDENCE_LEVELS:
                row[f"cov_{lvl}"] = dd["bands"][lvl]["coverage"]
                row[f"width_{lvl}"] = dd["bands"][lvl]["mean_width"]
                row[f"medwidth_{lvl}"] = dd["bands"][lvl]["median_width"]
            rows.append(row)
    suffix = "_legacy" if legacy_only else ("" if exclude_non_member else "_no_exclusions")
    out_path = f"remeasure_accuracy_results{suffix}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[remeasure] per-seed metrics -> {out_path}")


if __name__ == "__main__":
    main()
