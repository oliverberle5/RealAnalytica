"""Calibration layer: raw CatBoost score -> (percent chance, risk tier).

Two distinct things happen here, deliberately kept separate:

1. BACKTEST (honest validation): Platt scaling is fit ONLY on folds 1-3's
   out-of-fold predictions, then reliability is checked on folds 4-5's
   out-of-fold predictions -- data the calibrator never touched. This is
   the real test of whether the calibrated percentages can be trusted, as
   opposed to fitting and evaluating on the same pooled set (which would
   just show the curve fits the data it was fit to).

2. PRODUCTION calibrator: fit on ALL pooled out-of-fold predictions (folds
   1-5), since more data is better once we're done validating the approach
   and are ready to ship it. This is what score_to_output() uses.

Risk tier cutpoints (2026-07-24, revised same day): NOT equal-population.
Equal-population (25% each) quartiles were tried first and rejected --
user feedback: "we do not want to label someone as high when in reality
they have a less than random chance of leaving," which a population-
quantile scheme cannot guarantee (the top 25% by rank is still the top
25% even if raw risk is fairly flat/low that quarter). Replaced with
MULTIPLIER-OF-BASE-RATE cutpoints (`MULT_CUT_POINTS`, self-calibrating --
shifts automatically if the base rate moves, never hardcoded as a raw
percentage): Low = below 1x base rate (literally worse-than-random odds,
can never be mislabeled upward), Medium = 1-3x, High = 3-6x, Extra High
= 6x+. The 6x upper anchor was chosen empirically, not asserted: swept
multiplier thresholds from 1.5x to 20x on the honest out-of-sample split
(fit on folds 1-3, tested on folds 4-5) and picked the point where
precision (actual 3-month-departure rate within the bucket) peaks while
the bucket is still large enough to trust (6x: 12.3% precision, n=391,
2.1% of roster; 10x+: precision keeps LOOKING higher but n collapses to
tens of people with a 95% CI wide enough to be meaningless). See
`tier_backtest.py` for the full sweep and the resulting 4-bucket
backtest. Per the "top bucket hides a spread" finding, each output still
includes the calibrated percentage AND the risk multiplier alongside the
tier label, not the label alone.

ROOT CAUSE OF THE PER-QUARTER DRIFT (diagnosed 2026-07 #5): the backtest
found the calibrator systematically OVER-predicting in the most recent
quarters (2025-07 through 2026-04). Root-caused by comparing the RAW
(pre-mass-mover-recode) positive rate per quarter against the recode rate
per quarter: the raw rate shows NO decline (2025-10 and 2026-04 are among
the highest in the whole panel) -- agents are not genuinely switching
offices less. But the mass-mover recode rate (share of raw "moves" that
get correctly zeroed out as a confirmed rebrand/M&A/cluster event) jumps
from ~3-8% of raw positives in the calibrator's fit window (2024-01 to
2025-04) to 26-48% in the test window (2025-07 to 2026-04) -- driven by
real events (the SMRT->RMREV rebrand rollout, Lamacchia's acquisitions,
etc.) landing heavily in this specific calendar window. The calibrator
learned "what a raw score means" from a period with much less of this
legitimate cleanup happening, so it overpredicts once cleanup activity
spikes. This is NOT smooth concept drift and NOT random per-quarter noise
-- it's the lumpy, one-off timing of M&A/rebrand events.

RECENCY-WEIGHTED PLATT SCALING -- TESTED, DID NOT HELP (kept as
_recency_weights/backtest(recency_weighted=True) for reference, not used
in production). Exponential down-weighting of older OOF rows was the
first idea tried (half-life 4 quarters). Result: ECE 0.6821% (unweighted)
vs 0.7050% (recency-weighted) -- no improvement, slightly worse. Why it
can't work here: the fit window (2024-01 to 2025-04) has a FLAT, low
recode rate the entire way through (3-8%) -- there's no gradual ramp for
"weight recent rows more" to detect and amplify. The regime jump to
26-48% happens entirely AFTER the fit window ends. No reweighting of
historical rows can reveal a regime that hadn't started yet when those
rows were recorded.

BIAS-CORRECTION PATCH -- TESTED, WORKS BETTER (this is the production
approach). Instead of reshaping old training rows, track the ACTUAL
miscalibration on the most recently COMPLETED quarter (once its true
3-month outcome is known) and apply that as a multiplicative correction
to predictions for the quarter after. Since 2026-08-11 this happens on a
real cadence rather than whenever someone remembers: see
`quarterly_recalibration.py`, which refits Platt and re-measures this ratio
whenever a new quarter lands, and logs both per run. Backtest: a calibrator fit on
2021-2024 data alone overpredicted 2025-07/2025-10 (2.48% predicted vs.
2.10% actual, bias_ratio=0.85). Applying that 0.85 correction to the
NEXT quarter's predictions (2026-01/2026-04) cut the gap roughly in half
(2.61% uncorrected -> 2.21% corrected, vs. 1.70% actual) and improved the
Brier score. This directly tracks whatever the CURRENT cleanup-rate
regime looks like, refreshed every quarter as part of the retrain cycle
-- see compute_recent_bias_correction() / apply_bias_correction().
Caveat: it's a one-quarter-lagged correction, so a sudden NEW regime
shift (calendar quarter after the bias was measured) would still catch
it briefly off guard -- but it self-corrects every quarter rather than
staying stuck on a stale multi-year-old calibration.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from crossval import _fit_catboost, _forward_folds
from data import load_clean

RISK_TIER_LABELS = ["Low", "Medium", "High", "Extra High"]
RECENCY_HALF_LIFE_QUARTERS = 4

# Multiples of the base rate marking tier boundaries -- empirically chosen,
# see module docstring. Self-calibrating: multiply by whatever base_rate
# actually is this quarter, never a hardcoded raw percentage.
MULT_CUT_POINTS = [1.0, 3.0, 6.0]

# Per-horizon anchors (2026-08-25). MULT_CUT_POINTS above is a 3-MONTH
# anchor and does not transfer: a multiplier threshold is applied to a base
# rate, so at the 12-month base rate (7.7%) "6x" means a predicted 46%
# chance of leaving and the top bucket collapses to a handful of agents.
# HANDOFF.md's 2026-08-13 entry established this; these are the re-anchored
# equivalents, swept out-of-sample in horizon_precision.py::reanchor_multipliers
# (fit on early folds, applied unchanged to the last).
#
# The criterion is the one that actually binds -- BUCKET SIZE, not peak
# precision. Precision keeps climbing past every one of these thresholds at
# all three horizons; what stops you going higher is that n collapses to a
# few dozen agents with a CI wide enough to be meaningless. So each horizon's
# anchors are the multipliers that reproduce the 3-month tiers' share of the
# roster (~1% Extra High, ~3.5% High), which is what makes the tier LABELS
# mean the same operational thing at any horizon:
#
#   horizon   Extra High anchor   % roster   precision      High anchor   % roster
#      3m           6x              0.99%      15.7%            3x          3.55%
#      6m           8x              1.05%      19.4%            4x          3.89%
#     12m           3.5x            1.11%      48.4%            2.5x        3.73%
HORIZON_MULT_CUT_POINTS = {
    3: [1.0, 3.0, 6.0],
    6: [1.0, 4.0, 8.0],
    12: [1.0, 2.5, 3.5],
}


def compute_tier_cutpoints(base_rate: float, mult_cut_points=None) -> np.ndarray:
    """mult_cut_points defaults to the 3-month production anchors. Pass a
    horizon's entry from HORIZON_MULT_CUT_POINTS when tiering any other
    horizon -- reusing [1, 3, 6] there silently empties the top bucket."""
    return np.array(MULT_CUT_POINTS if mult_cut_points is None else mult_cut_points) * base_rate


def _wilson_interval(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _collect_oof(df, folds):
    """Per-fold out-of-fold raw scores, labels, and snapshot dates (kept
    separate by fold so backtest/production splits and per-quarter checks
    are both possible downstream)."""
    per_fold = []
    for train_dates, test_dates in folds:
        train = df[df["snapshot_date"].isin(train_dates)]
        test = df[df["snapshot_date"].isin(test_dates)]
        scores, _ = _fit_catboost(train, test)
        per_fold.append(
            {
                "scores": scores,
                "labels": test["label_left_3m"].to_numpy(),
                "dates": test["snapshot_date"].to_numpy(),
            }
        )
    return per_fold


def _recency_weights(dates: np.ndarray, reference_date, half_life_quarters: float = RECENCY_HALF_LIFE_QUARTERS) -> np.ndarray:
    """Exponential recency weights -- tested, did not fix the drift (see
    module docstring). Kept for reference / in case a future diagnosis
    finds a scenario where a gradual (not discontinuous) drift responds
    better to this than the current bias-correction approach does.
    """
    quarters_back = (pd.Timestamp(reference_date) - pd.to_datetime(dates)).days / 91.0
    return 0.5 ** (quarters_back / half_life_quarters)


def backtest(per_fold, fit_folds=(0, 1, 2), test_folds=(3, 4), recency_weighted=False):
    fit_scores = np.concatenate([per_fold[i]["scores"] for i in fit_folds])
    fit_labels = np.concatenate([per_fold[i]["labels"] for i in fit_folds])
    fit_dates = np.concatenate([per_fold[i]["dates"] for i in fit_folds])

    platt = LogisticRegression()
    if recency_weighted:
        weights = _recency_weights(fit_dates, reference_date=fit_dates.max())
        platt.fit(fit_scores.reshape(-1, 1), fit_labels, sample_weight=weights)
    else:
        platt.fit(fit_scores.reshape(-1, 1), fit_labels)

    test_scores = np.concatenate([per_fold[i]["scores"] for i in test_folds])
    test_labels = np.concatenate([per_fold[i]["labels"] for i in test_folds])
    test_dates = np.concatenate([per_fold[i]["dates"] for i in test_folds])
    test_pred = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    brier = np.mean((test_pred - test_labels) ** 2)
    tag = "recency-weighted" if recency_weighted else "unweighted"
    print(f"[backtest:{tag}] calibrator fit on folds {fit_folds}, tested on folds {test_folds} (never seen by the fit)")
    print(f"[backtest:{tag}] test set: {len(test_labels)} rows, {test_labels.sum()} positives, Brier score {brier:.5f}\n")

    print(f"[backtest:{tag}] RELIABILITY DIAGRAM (pooled): predicted vs actual, by predicted-probability decile")
    bins = pd.qcut(test_pred, 10, labels=False, duplicates="drop")
    ece = 0.0
    n_outside_ci = 0
    for b in sorted(pd.unique(bins)):
        mask = bins == b
        n = mask.sum()
        k = test_labels[mask].sum()
        actual = k / n
        predicted = test_pred[mask].mean()
        lo, hi = _wilson_interval(k, n)
        gap = abs(actual - predicted)
        ece += (n / len(test_pred)) * gap
        outside = not (lo <= predicted <= hi)
        n_outside_ci += outside
        flag = "  <-- outside 95% CI" if outside else ""
        print(
            f"  bin {b}: n={n:4d}  predicted={predicted:.3%}  actual={actual:.3%}  "
            f"95% CI=[{lo:.3%}, {hi:.3%}]{flag}"
        )
    print(f"\n[backtest:{tag}] Expected Calibration Error (pooled): {ece:.4%}  ({n_outside_ci}/10 deciles outside 95% CI)")

    print(f"\n[backtest:{tag}] PER-QUARTER check (does pooled reliability hide quarter-level drift?)")
    for d in sorted(pd.unique(test_dates)):
        mask = test_dates == d
        n = mask.sum()
        k = test_labels[mask].sum()
        actual = k / n if n else np.nan
        predicted = test_pred[mask].mean() if n else np.nan
        lo, hi = _wilson_interval(k, n)
        flag = "  <-- outside 95% CI" if n and not (lo <= predicted <= hi) else ""
        print(
            f"  {pd.Timestamp(d).date()}: n={n:4d}  predicted={predicted:.3%}  actual={actual:.3%}  "
            f"95% CI=[{lo:.3%}, {hi:.3%}]{flag}"
        )
    print()

    return platt, ece


def compute_recent_bias_correction(per_fold, recent_fold_idx: int) -> float:
    """The production bias-tracking step: how much did a calibrator fit on
    everything BEFORE `recent_fold_idx` miscalibrate on that fold's actual
    outcome (now known)? Returns actual/predicted -- multiply future
    predictions by this to correct for whatever regime the most recently
    completed quarter(s) revealed. Re-run this every quarter as part of the
    retrain cycle -- `quarterly_recalibration.py` now does that for you, and
    passes the last fold, i.e. whichever one has just become fully known.
    """
    prior_folds = range(recent_fold_idx)
    fit_scores = np.concatenate([per_fold[i]["scores"] for i in prior_folds])
    fit_labels = np.concatenate([per_fold[i]["labels"] for i in prior_folds])
    platt = LogisticRegression()
    platt.fit(fit_scores.reshape(-1, 1), fit_labels)

    recent_scores = per_fold[recent_fold_idx]["scores"]
    recent_labels = per_fold[recent_fold_idx]["labels"]
    recent_pred = platt.predict_proba(recent_scores.reshape(-1, 1))[:, 1]

    actual_rate = recent_labels.mean()
    predicted_rate = recent_pred.mean()
    bias_ratio = actual_rate / predicted_rate
    print(
        f"[bias-correction] fold {recent_fold_idx}: actual={actual_rate:.4%}, "
        f"predicted-by-prior-calibrator={predicted_rate:.4%}, bias_ratio={bias_ratio:.4f}"
    )
    return bias_ratio


def fit_production_calibrator(per_fold):
    all_scores = np.concatenate([f["scores"] for f in per_fold])
    all_labels = np.concatenate([f["labels"] for f in per_fold])

    platt = LogisticRegression()
    platt.fit(all_scores.reshape(-1, 1), all_labels)

    base_rate = all_labels.mean()
    cut_points = compute_tier_cutpoints(base_rate)
    return platt, cut_points, base_rate


def score_to_output(raw_score: float, platt: LogisticRegression, cut_points, base_rate: float,
                     bias_correction: float = 1.0, range_half_width_pct: float = 0.15) -> dict:
    """bias_correction: multiplicative factor from compute_recent_bias_correction(),
    applied on top of the Platt-calibrated percentage (see module docstring
    -- this is the production fix for the per-quarter drift). Defaults to
    1.0 (no correction) so this function still works standalone/in tests.

    range_half_width_pct: short-term uncertainty-communication patch --
    +/- this fraction of the calibrated percentage, e.g. 0.15 means an 8%
    point estimate displays as "6.8%-9.2%". Narrower than the raw
    pre-bias-correction miscalibration (which ran +/-20-30% before this
    quarter's correction) since the bias_correction step now removes most
    of the recent systematic error -- this remaining band is for ordinary
    per-quarter noise, not the corrected-for regime shift. NOT a formal
    per-row confidence interval (would need a heavier bootstrap/
    cluster-robust approach, flagged as not-yet-done in HANDOFF.md).
    """
    pct = platt.predict_proba([[raw_score]])[0, 1] * bias_correction
    bucket_idx = int(np.searchsorted(cut_points, pct))
    return {
        "percent_chance": pct,
        "plausible_range": (pct * (1 - range_half_width_pct), pct * (1 + range_half_width_pct)),
        "risk_tier": RISK_TIER_LABELS[bucket_idx],
        "risk_multiplier": pct / base_rate,
    }


def main():
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())
    per_fold = _collect_oof(df, folds)

    print("=" * 72)
    print("BACKTEST: unweighted Platt scaling (the honest validation)")
    print("=" * 72)
    backtest(per_fold, fit_folds=(0, 1, 2), test_folds=(3, 4), recency_weighted=False)

    print("=" * 72)
    print("REJECTED APPROACH: recency-weighted Platt scaling (tested, did not help -- see module docstring)")
    print("=" * 72)
    backtest(per_fold, fit_folds=(0, 1, 2), test_folds=(3, 4), recency_weighted=True)

    print("=" * 72)
    print("BIAS-CORRECTION DEMONSTRATION: fold 4's measured bias applied to the")
    print("INDEPENDENT, later fold 5 (not the same fold the bias was measured on)")
    print("=" * 72)
    # Measure bias on fold 4 (index 3) using only folds 1-3 (indices 0-2) --
    # exactly what compute_recent_bias_correction(recent_fold_idx=3) does.
    demo_bias = compute_recent_bias_correction(per_fold, recent_fold_idx=3)
    # Now apply that fold-4-derived bias to fold 5 (index 4) -- a genuinely
    # independent, later fold the bias was never measured on.
    platt_1to4 = LogisticRegression()
    s = np.concatenate([per_fold[i]["scores"] for i in range(4)])
    l = np.concatenate([per_fold[i]["labels"] for i in range(4)])
    platt_1to4.fit(s.reshape(-1, 1), l)
    fold5_pred = platt_1to4.predict_proba(per_fold[4]["scores"].reshape(-1, 1))[:, 1]
    fold5_labels = per_fold[4]["labels"]
    brier_uncorrected = np.mean((fold5_pred - fold5_labels) ** 2)
    brier_corrected = np.mean((fold5_pred * demo_bias - fold5_labels) ** 2)
    print(f"  fold5 actual: {fold5_labels.mean():.4%}")
    print(f"  fold5 predicted (uncorrected, using folds 1-4 calibrator): {fold5_pred.mean():.4%}  (Brier {brier_uncorrected:.5f})")
    print(f"  fold5 predicted (corrected by fold4's bias_ratio={demo_bias:.4f}): {(fold5_pred * demo_bias).mean():.4%}  (Brier {brier_corrected:.5f})")

    print("\n" + "=" * 72)
    print("PRODUCTION CALIBRATOR (fit on all 5 folds pooled)")
    print("=" * 72)
    platt, cut_points, base_rate = fit_production_calibrator(per_fold)
    print(f"base rate: {base_rate:.3%}")
    print(f"risk tier cut points (calibrated %): {[f'{c:.3%}' for c in cut_points]}")

    print("\nCurrent bias correction to apply going forward (measured on fold 5, the most recently completed quarter):")
    current_bias_correction = compute_recent_bias_correction(per_fold, recent_fold_idx=4)

    print("\nExample outputs across the raw score range (WITH current bias correction applied):")
    for raw in [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]:
        out = score_to_output(raw, platt, cut_points, base_rate, bias_correction=current_bias_correction)
        lo, hi = out["plausible_range"]
        print(
            f"  raw={raw:.2f} -> {out['percent_chance']:.2%} chance ({lo:.2%}-{hi:.2%}), "
            f"{out['risk_multiplier']:.1f}x base rate, bucket={out['risk_tier']}"
        )


if __name__ == "__main__":
    main()
