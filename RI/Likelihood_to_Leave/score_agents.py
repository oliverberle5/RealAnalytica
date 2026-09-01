"""Production scoring: publish, for each currently-active agent, a
likelihood-to-leave percent chance, a 1-4 risk tier, and a risk
multiplier (percentage / base rate).

Tier convention: 1 = Low, 4 = Extra High (matches RISK_TIER_LABELS order).
4-bucket system (Low/Medium/High/Extra High) replaced the original 5-tier
quintile system 2026-07-24 -- see HANDOFF.md.

Normally invoked by `quarterly_recalibration.py`, not by hand: that wrapper
runs the mass-mover guard first, runs train_multi_horizon.py off the same
panel afterwards, checks the output before letting it reach any other
directory, and records what the calibration did this quarter. Running this
file directly still works and produces the same scores -- it just skips all
of that. Retrains the model and calibrator fresh each run rather than
loading a saved artifact, so it always reflects whatever data.py /
crossval.py / calibrate.py currently do -- there is no model-versioning/
drift-between-runs risk to manage yet, at the cost of a few minutes of
retraining time per run.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from calibrate import (
    RISK_TIER_LABELS,
    _collect_oof,
    compute_recent_bias_correction,
    compute_tier_cutpoints,
    fit_production_calibrator,
)
from crossval import MONOTONE, PRODUCTION_CATBOOST_PARAMS, PRODUCTION_FEATURES, _forward_folds
from data import load_clean, load_current_snapshot
from explain_agents import generate_explanations

OUTPUT_PATH = "current_agent_risk_scores.csv"


def train_production_model(df: pd.DataFrame) -> CatBoostClassifier:
    # AutoResearch-winning config (2026-07-25 promotion) -- see crossval.py's
    # PRODUCTION_FEATURES / PRODUCTION_CATBOOST_PARAMS docstring for provenance.
    X = df[PRODUCTION_FEATURES].astype(float)
    y = df["label_left_3m"].to_numpy()
    monotone = [MONOTONE.get(c, 0) for c in PRODUCTION_FEATURES]
    model = CatBoostClassifier(monotone_constraints=monotone, **PRODUCTION_CATBOOST_PARAMS)
    model.fit(X, y)
    return model


def score_current_agents() -> tuple[pd.DataFrame, dict]:
    """Returns (scores, meta). `meta` carries the three numbers that define
    this quarter's calibration -- base_rate, bias_correction, and the
    corrected base rate the tiers and multipliers are anchored to -- so the
    quarterly wrapper can check and log them without re-deriving them from
    console output.
    """
    print("[score] loading training history and cross-validating for calibration...")
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())
    per_fold = _collect_oof(df, folds)

    platt, _unused_cut_points, base_rate = fit_production_calibrator(per_fold)
    # Bias correction measured on the most recently completed fold -- see
    # calibrate.py module docstring for why this beats recency-weighting.
    bias_correction = compute_recent_bias_correction(per_fold, recent_fold_idx=len(per_fold) - 1)
    # Tier cutpoints and risk_multiplier must be anchored to the SAME
    # bias-corrected base rate that pct itself is scaled by below -- using
    # the raw pre-correction base_rate here would compare a corrected
    # score against an uncorrected threshold, silently shrinking every
    # tier (caught 2026-07-24: production showed 1 Extra High agent
    # against a backtested expectation of ~1-2%, traced to exactly this
    # mismatch).
    corrected_base_rate = base_rate * bias_correction
    cut_points = compute_tier_cutpoints(corrected_base_rate)

    print("[score] training production CatBoost on full history...")
    model = train_production_model(df)

    print("[score] scoring current agents...")
    current = load_current_snapshot()
    X_current = current[PRODUCTION_FEATURES].astype(float)
    raw_scores = model.predict_proba(X_current)[:, 1]
    pct = platt.predict_proba(raw_scores.reshape(-1, 1))[:, 1] * bias_correction

    print("[score] generating per-agent 'why this score' explanations...")
    why = generate_explanations(model, X_current, raw=current)

    bucket_idx = np.searchsorted(cut_points, pct)  # 0..3
    out = current[["mls_agent_id", "office_name", "agent_city", "agent_state", "snapshot_date"]].copy()
    out["percent_chance"] = pct
    out["risk_tier"] = bucket_idx + 1  # 1 = Low, 4 = Extra High
    out["risk_tier_label"] = [RISK_TIER_LABELS[i] for i in bucket_idx]
    out["risk_multiplier"] = pct / corrected_base_rate
    out = pd.concat([out.reset_index(drop=True), why.reset_index(drop=True)], axis=1)
    out = out.sort_values("percent_chance", ascending=False).reset_index(drop=True)

    print(f"[score] base rate: {base_rate:.3%}  |  bias correction applied: {bias_correction:.4f}x  "
          f"|  bias-corrected base rate used for tiers/multiplier: {corrected_base_rate:.3%}")
    meta = {
        "base_rate": float(base_rate),
        "bias_correction": float(bias_correction),
        "corrected_base_rate": float(corrected_base_rate),
        "cut_points": [float(c) for c in cut_points],
        "snapshot_date": str(pd.Timestamp(current["snapshot_date"].iloc[0]).date()),
    }
    return out, meta


def main():
    out, _meta = score_current_agents()
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[score] scored {len(out)} agents -> {OUTPUT_PATH}")

    print("\nDistribution across risk tiers:")
    print(out["risk_tier"].value_counts().sort_index().rename("n_agents"))

    print("\nTop 10 highest-risk agents:")
    cols = ["mls_agent_id", "office_name", "percent_chance", "risk_tier", "risk_multiplier"]
    with pd.option_context("display.max_colwidth", 28, "display.width", 140):
        print(out.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
