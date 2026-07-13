"""Production scoring: publish, for each currently-active agent, a
likelihood-to-leave percent chance, a 1-5 risk quintile, and a risk
multiplier (percentage / base rate).

Quintile convention: 1 = Low, 5 = High (matches QUINTILE_LABELS order).

Not yet wired into a scheduled pipeline (see HANDOFF.md "Next up" #3, the
not-yet-built quarterly retrain/rescore cycle) -- run this manually each
quarter after new data arrives. Retrains the model and calibrator fresh
each run rather than loading a saved artifact, so it always reflects
whatever data.py / crossval.py / calibrate.py currently do -- there is no
model-versioning/drift-between-runs risk to manage yet, at the cost of a
few minutes of retraining time per run.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from calibrate import (
    QUINTILE_LABELS,
    _collect_oof,
    compute_recent_bias_correction,
    fit_production_calibrator,
)
from crossval import MONOTONE, TREE_FEATURES, _forward_folds
from data import load_clean, load_current_snapshot

OUTPUT_PATH = "current_agent_risk_scores.csv"


def train_production_model(df: pd.DataFrame) -> CatBoostClassifier:
    X = df[TREE_FEATURES].astype(float)
    y = df["label_left_3m"].to_numpy()
    monotone = [MONOTONE.get(c, 0) for c in TREE_FEATURES]
    model = CatBoostClassifier(
        iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=3.0,
        auto_class_weights="Balanced", monotone_constraints=monotone,
        loss_function="Logloss", eval_metric="PRAUC", verbose=False, allow_writing_files=False,
    )
    model.fit(X, y)
    return model


def score_current_agents() -> pd.DataFrame:
    print("[score] loading training history and cross-validating for calibration...")
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())
    per_fold = _collect_oof(df, folds)

    platt, cut_points, base_rate = fit_production_calibrator(per_fold)
    # Bias correction measured on the most recently completed fold -- see
    # calibrate.py module docstring for why this beats recency-weighting.
    bias_correction = compute_recent_bias_correction(per_fold, recent_fold_idx=len(per_fold) - 1)

    print("[score] training production CatBoost on full history...")
    model = train_production_model(df)

    print("[score] scoring current agents...")
    current = load_current_snapshot()
    X_current = current[TREE_FEATURES].astype(float)
    raw_scores = model.predict_proba(X_current)[:, 1]
    pct = platt.predict_proba(raw_scores.reshape(-1, 1))[:, 1] * bias_correction

    bucket_idx = np.searchsorted(cut_points, pct)  # 0..4
    out = current[["agent_profile_id", "office_name", "agent_city", "agent_state", "snapshot_date"]].copy()
    out["percent_chance"] = pct
    out["risk_quintile"] = bucket_idx + 1  # 1 = Low, 5 = High
    out["risk_quintile_label"] = [QUINTILE_LABELS[i] for i in bucket_idx]
    out["risk_multiplier"] = pct / base_rate
    out = out.sort_values("percent_chance", ascending=False).reset_index(drop=True)

    print(f"[score] base rate: {base_rate:.3%}  |  bias correction applied: {bias_correction:.4f}x")
    return out


def main():
    out = score_current_agents()
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[score] scored {len(out)} agents -> {OUTPUT_PATH}")

    print("\nDistribution across risk quintiles:")
    print(out["risk_quintile"].value_counts().sort_index().rename("n_agents"))

    print("\nTop 10 highest-risk agents:")
    cols = ["agent_profile_id", "office_name", "percent_chance", "risk_quintile", "risk_multiplier"]
    with pd.option_context("display.max_colwidth", 28, "display.width", 140):
        print(out.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
