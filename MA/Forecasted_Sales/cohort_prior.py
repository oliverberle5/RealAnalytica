"""
Cold-start cohort-prior blending for new agents (decision #5, HANDOFF.md).

Problem: an agent with a short career (<12 months) hasn't accumulated a
full trailing-12m production history to "trend from" -- their own
volume_12m/units_12m (the model's dominant features, ~36-39% importance
per round 2) are necessarily small/incomplete for these agents, not
because they're bad agents but because there simply hasn't been time
yet. Rather than trust the raw quantile model's output uncritically for
this population, blend it toward a COHORT PRIOR -- what similarly-new
agents at a similar office brand historically went on to produce in
their own next 12 months -- with the blend weight shifting toward the
agent's own model-based prediction as their actual career_months
accumulates toward 12. At career_months=0 the forecast is almost pure
cohort prior; at career_months=11 it's almost pure model; at 12+, no
blending at all.

Cohort = office_brand, reconstructed from the one-hot brand_* dummies
(the modeling frames don't keep the raw string column). Falls back to a
pooled (all-brands) new-agent prior when a specific brand has too few
historical new-agent comps to trust (MIN_COHORT_N) -- standard
hierarchical-fallback practice for a prior estimated from few examples.
"""
import numpy as np
import pandas as pd

COLD_START_MONTHS = 12
MIN_COHORT_N = 30
CAREER_COLUMN = "career_months"


def brand_label(df: pd.DataFrame) -> pd.Series:
    brand_cols = [c for c in df.columns if c.startswith("brand_")]
    sub = df[brand_cols]
    has_any = sub.any(axis=1)
    label = pd.Series("unknown", index=df.index)
    label[has_any] = sub.loc[has_any].idxmax(axis=1)
    return label


def compute_cohort_priors(train_df: pd.DataFrame, target_col: str, quantiles: dict):
    """quantiles: dict name->alpha, e.g. {'lo':0.025,'med':0.5,'hi':0.975}.
    Returns (priors_by_brand, pooled_fallback)."""
    new_agents = train_df[train_df[CAREER_COLUMN] < COLD_START_MONTHS].copy()
    new_agents["_brand"] = brand_label(new_agents)

    pooled = {name: float(np.quantile(new_agents[target_col], a)) for name, a in quantiles.items()}

    priors = {}
    for brand, grp in new_agents.groupby("_brand"):
        if len(grp) >= MIN_COHORT_N:
            priors[brand] = {name: float(np.quantile(grp[target_col], a)) for name, a in quantiles.items()}
    return priors, pooled


def blend_with_cohort_prior(raw_preds: dict, df: pd.DataFrame, priors: dict, pooled: dict, quantiles: dict):
    """raw_preds: dict name -> np.array, same row order as df.
    Returns a NEW dict (does not mutate raw_preds) of blended predictions.
    Rows with career_months >= COLD_START_MONTHS are returned unchanged."""
    career = df[CAREER_COLUMN].to_numpy()
    weight = np.clip(career / COLD_START_MONTHS, 0, 1)  # 1.0 = fully trust own model
    brand = brand_label(df).to_numpy()

    blended = {name: raw_preds[name].copy() for name in quantiles}
    cold_idx = np.where(weight < 1.0)[0]
    for i in cold_idx:
        prior = priors.get(brand[i], pooled)
        w = weight[i]
        for name in quantiles:
            blended[name][i] = w * raw_preds[name][i] + (1 - w) * prior[name]
    return blended
