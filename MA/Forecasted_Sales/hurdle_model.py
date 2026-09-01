"""Hurdle model: splits each target's next-12m forecast into two honest
numbers instead of one range that has to do both jobs (round 15,
HANDOFF.md):

1. P(zero) -- the probability an agent has $0 volume / 0 units at all,
   reported directly as its own percentage.
2. A conditional nonzero range -- "if they sell anything, expect
   $X-$Y" -- fit ONLY on historical rows where the target was actually
   nonzero, so it can never collapse back to 0 the way a single
   unconditional range does at low/mid confidence levels (round 14: a
   large, genuine, individually-varying zero-sales population was making
   the unconditional range look broken even though the model was right).

Volume and units are modeled as two SEPARATE hurdles, not one shared
"is this agent inactive" flag -- confirmed empirically that they're
correlated but not identical: every unit-zero row is also volume-zero,
but 3,579 of 61,100 resolved rows (~5.9%) have volume==0 with units>0
(likely rentals/assignments or other $0-recorded transactions), so a
single combined classifier would misdescribe that slice for whichever
target it wasn't built for. Same "two independent targets" discipline as
decision #1 in HANDOFF.md.

P(zero) calibration mirrors the quantile pipeline's own train/calib
split exactly (score_agents.py): fit the classifier on the earlier
quarters, then isotonic-calibrate its raw probabilities against actual
outcomes on the held-out most-recently-completed quarters, same
train-on-early/correct-on-recent pattern CQR already uses for the
quantile bounds -- a raw classifier probability is not trustworthy on
its own (class-imbalance rebalancing during training deliberately
distorts the raw output; isotonic regression undoes that distortion
using held-out ground truth).

MODEL: XGBoost depth=4, `scale_pos_weight`-balanced -- the
`AutoResearch/autoloop_pzero.py` round-17 winner (HANDOFF.md), beating
the original CatBoost-d4 gut-check config on mean Brier score across the
same 4 forward-chaining folds used for every quantile-model search
(composite 0.9940, i.e. ~0.6% lower average Brier score, with AUC
unchanged-to-better on both targets -- not a calibration trick that
traded away discrimination). CatBoost, not XGBoost, is still the winner
for the quantile RANGE (round 11) -- the two hurdle halves are tuned,
and won, independently; there's no reason they should end up in the same
model family.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from score_agents import CALIB_QUARTERS, TARGET_VOLUME, forecast_target_multi
from train_baseline import RANDOM_STATE

ZERO_CLASSIFIER_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric="auc",
)


def fit_zero_classifier(train, feature_cols, target_col):
    y = (train[target_col].values == 0).astype(int)
    # scale_pos_weight balances the classes (round-17 winner) -- computed
    # fresh per fit since the zero-rate isn't fixed (round 14: it's climbed
    # from 16.4% to 24.7% across the panel's history), not a hardcoded ratio.
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    m = XGBClassifier(scale_pos_weight=scale_pos_weight, **ZERO_CLASSIFIER_PARAMS)
    m.fit(train[feature_cols], y)
    return m


def forecast_pzero(resolved, current, feature_cols, target_col, calib_quarters=CALIB_QUARTERS):
    """Returns (pct_zero_current, meta). Same train/calib quarter split
    as forecast_target_multi(), so both halves of the hurdle forecast are
    built on identically-defined windows."""
    quarters = sorted(resolved["snapshot_date"].unique())
    calib_q = quarters[-calib_quarters:]
    train_q = quarters[:-calib_quarters]

    train = resolved[resolved["snapshot_date"].isin(train_q)]
    calib = resolved[resolved["snapshot_date"].isin(calib_q)]

    model = fit_zero_classifier(train, feature_cols, target_col)

    raw_calib = model.predict_proba(calib[feature_cols])[:, 1]
    y_calib = (calib[target_col].values == 0).astype(int)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_calib, y_calib)

    raw_current = model.predict_proba(current[feature_cols])[:, 1]
    pct_zero = np.maximum(calibrator.predict(raw_current), 0.01)

    meta = dict(train_quarters=train_q, calib_quarters=calib_q, base_rate=float(y_calib.mean()))
    return pct_zero, meta


def forecast_conditional_nonzero(resolved, current, feature_cols, target_col, levels):
    """Conditional range given the agent has ANY sales at all -- reuses
    the exact same quantile+CQR machinery as the unconditional forecast
    (forecast_target_multi), restricted to historical rows where this
    target was actually nonzero. `zero_floor="up"` for volume so the
    realistic-floor snap (round 14) raises a too-small prediction up to
    the smallest plausible nonzero sale instead of dropping it to 0 --
    0 is not a valid answer here by construction."""
    nonzero = resolved.loc[resolved[target_col] > 0].reset_index(drop=True)
    zero_floor = "up" if target_col == TARGET_VOLUME else None
    return forecast_target_multi(nonzero, current, feature_cols, target_col, levels, zero_floor=zero_floor)
