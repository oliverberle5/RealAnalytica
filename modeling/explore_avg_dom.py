"""One-off exploratory test: does per-agent avg_dom_12m carry real signal
once restricted to the subset where it's not the fabricated zero/null?

Context: avg_dom_12m is 38.9% null and, among non-null rows, mostly a
fabricated 0 (59% of ALL rows, null or not) -- confirmed broken, excluded
from the production feature set (see data.py RAW_FEATURE_COLUMNS comment).
User's request: before writing this column off permanently, check whether
it has real explanatory power on just the rows where it looks like a
plausible days-on-market value (> 0), including a quarter-over-quarter
TREND (this quarter's DOM vs the agent's own prior quarter) -- the
hypothesis being that a rising DOM (listings sitting longer) might signal
a frustrated agent about to leave, distinct from just the raw level.

Caveat up front: the genuinely-nonzero subset is only 1,779 of 84,417 RIAR
rows (2.1%) -- much smaller than "59% zeros" suggested (that stat was
among non-null rows only). Any finding here is necessarily low-powered.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from crossval import MONOTONE, TREE_FEATURES
from data import DATA_PATH, MLS_CODE, _parse_snapshot_date, load_clean
from split import time_based_split

raw = pd.read_csv(DATA_PATH, low_memory=False)
raw = raw[raw["mls_code"] == MLS_CODE].copy()
raw["snapshot_date"] = _parse_snapshot_date(raw["snapshot_date"])

# Panel-lag: agent's own avg_dom_12m one quarter prior (not a same-row ratio
# -- needs each agent's own earlier row, same engineering pattern as
# _build_moves's office lag).
raw = raw.sort_values(["agent_profile_id", "snapshot_date"])
raw["avg_dom_12m_prior"] = raw.groupby("agent_profile_id")["avg_dom_12m"].shift(1)
dom_lookup = raw.set_index(["agent_profile_id", "snapshot_date"])[["avg_dom_12m", "avg_dom_12m_prior"]]

df = load_clean()
df = df.join(dom_lookup, on=["agent_profile_id", "snapshot_date"], rsuffix="_dom")
df["avg_dom_trend_12m"] = np.where(
    (df["avg_dom_12m"] > 0) & (df["avg_dom_12m_prior"] > 0),
    df["avg_dom_12m"] / df["avg_dom_12m_prior"],
    np.nan,
)

nonzero = df[df["avg_dom_12m"] > 0].copy()
trend_ok = nonzero[nonzero["avg_dom_trend_12m"].notna()].copy()
print(f"[explore] full panel: {len(df)} rows, {df['label_left_3m'].mean():.3%} positive")
print(f"[explore] avg_dom_12m > 0 subset: {len(nonzero)} rows, {nonzero['label_left_3m'].mean():.3%} positive")
print(f"[explore] + prior-quarter also > 0 (trend computable): {len(trend_ok)} rows, {trend_ok['label_left_3m'].mean():.3%} positive")
print()


def _fit_eval(data: pd.DataFrame, features: list[str], label: str):
    if len(data) < 200 or data["label_left_3m"].sum() < 10:
        print(f"[{label}] too few rows/positives to fit reliably ({len(data)} rows, {data['label_left_3m'].sum()} positives) -- skipped")
        return
    train, test = time_based_split(data)
    if test["label_left_3m"].sum() == 0 or train["label_left_3m"].sum() == 0:
        print(f"[{label}] zero positives in train or test split -- skipped")
        return
    Xtr, ytr = train[features].astype(float), train["label_left_3m"].to_numpy()
    Xte, yte = test[features].astype(float), test["label_left_3m"].to_numpy()
    monotone = [MONOTONE.get(c, 0) for c in features]
    model = CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=3.0,
        auto_class_weights="Balanced", monotone_constraints=monotone,
        loss_function="Logloss", eval_metric="PRAUC", verbose=False, allow_writing_files=False,
    )
    model.fit(Xtr, ytr)
    pred = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, pred)
    imp = pd.Series(model.get_feature_importance(), index=features).sort_values(ascending=False)
    print(f"[{label}] test AUC: {auc:.4f} (n_train={len(train)}, n_test={len(test)})")
    for f in features:
        if f in ("avg_dom_12m", "avg_dom_trend_12m"):
            rank = list(imp.index).index(f) + 1
            print(f"    {f}: importance={imp[f]:.3f}, rank {rank}/{len(features)}")
    print()


print("=== On the avg_dom_12m > 0 subset ===")
_fit_eval(nonzero, TREE_FEATURES, "baseline (no avg_dom)")
_fit_eval(nonzero, TREE_FEATURES + ["avg_dom_12m"], "+ avg_dom_12m raw level")

print("=== On the trend-computable subset (both quarters > 0) ===")
_fit_eval(trend_ok, TREE_FEATURES, "baseline (no avg_dom)")
_fit_eval(trend_ok, TREE_FEATURES + ["avg_dom_12m"], "+ avg_dom_12m raw level")
_fit_eval(trend_ok, TREE_FEATURES + ["avg_dom_trend_12m"], "+ avg_dom_trend_12m only")
_fit_eval(trend_ok, TREE_FEATURES + ["avg_dom_12m", "avg_dom_trend_12m"], "+ both")
