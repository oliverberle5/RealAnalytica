"""
Per-agent "Why this score" explanations -- the SHAP-based explainability
layer flagged as "planned, never built" in HANDOFF.md ("Not yet done").
Upgrades the stale, static example in the old UI screenshot into a real,
per-agent-computed feature: for each agent, rank the features that
actually moved THEIR score (via CatBoost's native per-row SHAP values,
not just global feature importance), then fill in a human-readable
template with that agent's real numbers.

Two-part design, deliberately kept separate:
1. TEMPLATES (below) -- editable copy, one "risk" (this pushed the score
   UP) and one "safe" (this pushed the score DOWN) variant per feature.
   Edit these directly; nothing else needs to change to update wording.
2. generate_explanations() -- the mechanics: computes real per-agent SHAP
   contributions, picks the top N by |contribution|, selects the correct
   template variant by the SIGN of that agent's own SHAP value (not the
   feature's global monotonic direction -- a feature can push risk UP for
   one agent and DOWN for another even under a monotonic constraint,
   since the net contribution depends on where this agent sits on the
   curve), and fills in real values from that agent's own row.

Covers the 10 features that dominate CatBoost's global importance
ranking (see generate_report.py's top-15 list) and have a clean,
one-sentence real-world story: office/company peer exit rate, tenure,
inactivity, embeddedness, production trend (own and vs. market), career
mobility, listing activity, franchise context. Deliberately does NOT
template every one of the ~95 features -- the long tail doesn't move
scores enough to be worth writing a sentence for (confirmed by every
"bottom-N pruning didn't hurt" result across this project's AutoResearch
rounds), and a wall of bullets isn't more explainable than a chosen four.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from crossval import MONOTONE, PRODUCTION_CATBOOST_PARAMS, PRODUCTION_FEATURES
from data import load_clean, load_current_snapshot

TOP_N_BULLETS = 5  # 1 top driver (own highlighted box) + 4 supporting insights below it

# Some feature PAIRS tell functionally the same story and shouldn't both be
# shown -- e.g. "6 yrs at current office" + "0 office moves in 5 years" is
# not two insights, it's one restated (long current-office tenure literally
# forces moves_5y_asof to 0 once it clears 5 years). Verified empirically
# 2026-07-28, not just by inspection: tenure_current_office_months vs.
# moves_5y_asof correlate at -0.33 (and are deterministically linked once
# tenure clears 60 months); office_exit_rate_12m vs. company_exit_rate_12m
# correlate at 0.77 (same "peers are leaving" story, office- vs.
# company-level). Checked the rest of TEMPLATES for the same pattern
# (months_since_last_closing, volume_trend_12m, new_listings_12m,
# share_of_office_volume_12m, office_volume_trend_vs_market_12m) and found
# them pairwise near-zero correlated -- genuinely independent signals, not
# grouped. Only ONE feature per group may appear across top_driver +
# supporting for a given agent; features outside any group are unaffected
# (implicitly their own singleton group).
CONCEPT_GROUPS = {
    "tenure_current_office_months": "career_stability",
    "moves_5y_asof": "career_stability",
    "office_exit_rate_12m": "peer_turnover",
    "company_exit_rate_12m": "peer_turnover",
}

# Minimum share of an agent's TOTAL |SHAP| (across all ~95 features, not
# just the templated ones) a SUPPORTING bullet must carry to be shown --
# empirically chosen 2026-07-24 (not asserted): swept 1%-8% floors against
# the current roster. 3% trims the genuinely-negligible tail (~27% of
# agents lose >=1 weak bullet) without gutting most agents' insight count
# (73% still keep all 5). Only applies to supporting bullets, never the
# top driver -- the top driver's share never drops below ~10% for any
# agent in this roster, so it's never at risk of being filtered anyway.
MIN_SUPPORTING_SHARE = 0.03

# new_listings_12m carries no monotonic constraint (see crossval.py::MONOTONE),
# so CatBoost's learned SHAP direction for it can point either way regardless
# of the agent's actual count -- unlike the 4 monotonically-constrained
# features, where the sign is guaranteed to track the real value. (Was
# written as 5; corrected 2026-08-07 -- the fifth declared constraint,
# volume_trend_vs_market_12m, never bound because it isn't in
# PRODUCTION_FEATURES, and has since been removed from MONOTONE. The sign
# guarantee this comment relies on was only ever true of four.) Caught
# live 2026-07-28: the "safe" template fired at a raw count of 0 for 1,768
# agents ("active in generating new business" when they took literally zero),
# and the "risk" template fired at counts as high as 448 ("only" when that's
# far above the roster's ~8-count 90th percentile) -- self-contradictory
# either way. Bounds below are grounded in the roster's own distribution
# (median 1, 90th pct ~8.3), not guessed: at/below LOW_MAX reads as genuinely
# low, at/above HIGH_MIN reads as genuinely active; the 6-7 gap between them is
# deliberately ambiguous territory where NEITHER wording is trustworthy, so
# it's skipped rather than forced.
#
# 2026-08-11: the two values were TRANSPOSED at authoring -- LOW_MAX was 8 and
# HIGH_MIN was 5, self-contradictory on its face ("low" topping out ABOVE where
# "active" begins), and producing an OVERLAP instead of the gap described above:
# at counts 5-8 BOTH wordings were permitted, so two agents with the identical
# listing count could be told "only {n} ... below-average" and "{n} new
# listings ... active" -- the same class of contradiction the guardrail was
# added to prevent, just narrower. CHANGELOG's own 2026-07-28 entry describes
# "the ambiguous 6-7 gap", which is only reachable with these values the other
# way round, so this restores the documented intent rather than changing it.
NEW_LISTINGS_LOW_MAX = 5
NEW_LISTINGS_HIGH_MIN = 8

# ---------------------------------------------------------------- TEMPLATES
# feature -> {"risk": "...", "safe": "..."} using {named} placeholders.
# "risk" = this agent's SHAP contribution for this feature was POSITIVE
# (pushed their score up). "safe" = negative (pushed it down).
# tenure_current_office_months has a 3-way split (see _tenure_variant)
# since risk-increasing tenure means two very different real stories
# (brand new vs. very long-tenured) that need different wording.
TEMPLATES = {
    "office_exit_rate_12m": {
        "risk": "{pct} of the agents at their office left in the last 12 months.",
        "safe": "Only {pct} of the agents at their office left in the last 12 months - a stable office.",
    },
    "company_exit_rate_12m": {
        "risk": "{pct} of agents company-wide left in the last 12 months.",
        "safe": "Just {pct} of agents company-wide left in the last 12 months - low company-wide turnover.",
    },
    "tenure_current_office_new": {
        "risk": "Only {tenure} at their current office - new hires demonstrate increased risk.",
    },
    "tenure_current_office_early": {
        "risk": "{tenure} at their current office - still in the early period where risk is historically higher, before it bottoms out around year 5-10.",
    },
    "tenure_current_office_veteran": {
        "risk": "{tenure} at their current office - extended tenure introduces modest risk.",
    },
    "tenure_current_office_stable": {
        "safe": "{tenure} at their current office - past the highest-risk early period, in the historically stablest range.",
    },
    "months_since_last_closing": {
        "risk": "Hasn't closed a sale in {months} - extended inactivity is one of the strongest signals in this model.",
        "safe": "Closed a sale within the last {months} - actively engaged.",
    },
    "share_of_office_volume_12m": {
        "risk": "Accounts for only {pct} of their office's total volume - less embedded than a typical agent there.",
        "safe": "Accounts for {pct} of their office's total volume - a key producer, more embedded in the office.",
    },
    "volume_trend_12m": {
        "risk": "Sales volume down {pct_change} vs. the prior year.",
        "safe": "Sales volume up {pct_change} vs. the prior year.",
    },
    "office_volume_trend_vs_market_12m": {
        "risk": "Their office's volume is trailing the overall market trend by {pct_change}.",
        "safe": "Their office is outperforming the overall market trend by {pct_change}.",
    },
    "moves_5y_asof": {
        "risk": "Changed offices {n} time(s) in the last 5 years - a track record of moving.",
        "safe": "Has not changed offices in the last 5 years - a stable career pattern.",
    },
    "new_listings_12m": {
        "risk": "Took only {n} new listing(s) in the last 12 months - below-average listing activity.",
        "safe": "Took {n} new listings in the last 12 months - active in generating new business.",
    },
    "office_brand_independent": {
        "risk": "Works at an independent (non-franchise) office - historically higher turnover than franchise brands.",
        "safe": "Works at a franchise-affiliated office, not an independent - historically more stable.",
    },
}

def _fmt_months_years(months: float) -> str:
    y, m = divmod(round(months), 12)
    if y and m:
        return f"{y} yr {m} mo"
    if y:
        return f"{y} yr"
    return f"{m} mo"


def _tenure_variant(row: pd.Series, shap_val: float) -> str:
    tenure = row["tenure_current_office_months"]
    if shap_val <= 0:
        return "tenure_current_office_stable"
    if tenure < 12:
        return "tenure_current_office_new"
    if tenure >= 120:
        return "tenure_current_office_veteran"
    return "tenure_current_office_early"


def _display_exit_rate(row: pd.Series, level: str) -> float:
    """DISPLAY-ONLY recomputation of office/company exit rate against the
    PRIOR (12mo-ago) headcount instead of the raw feature's CURRENT-headcount
    denominator -- the model's actual PRODUCTION_FEATURES input
    (office_exit_rate_12m / company_exit_rate_12m) is untouched, unretrained.
    Current-headcount denominator can mathematically exceed 100% for a
    small, fast-shrinking org (e.g. 21 exits against a 14-agent CURRENT
    headcount reads as "150% left," confusing even though valid) -- prior
    headcount is the standard convention for a rate and reads sanely (that
    same case: 21 exits / 23 prior-headcount = 91%). Falls back to the raw
    feature value if prior headcount isn't positive (a real edge case, not
    just defensive: a net_flow_12m larger in magnitude than active_agents_asof
    is possible for a tiny org)."""
    exits = row[f"{level}_exits_12m"]
    prior_headcount = row[f"{level}_active_agents_asof"] - row[f"{level}_net_flow_12m"]
    if prior_headcount > 0:
        return exits / prior_headcount
    return row[f"{level}_exit_rate_12m"]


def _format_args(feature: str, row: pd.Series) -> dict:
    if feature == "office_exit_rate_12m":
        return {"pct": f"{_display_exit_rate(row, 'office'):.0%}"}
    if feature == "company_exit_rate_12m":
        return {"pct": f"{_display_exit_rate(row, 'company'):.0%}"}
    if feature == "share_of_office_volume_12m":
        return {"pct": f"{row[feature]:.0%}"}
    if feature.startswith("tenure_current_office"):
        return {"tenure": _fmt_months_years(row["tenure_current_office_months"])}
    if feature == "months_since_last_closing":
        return {"months": _fmt_months_years(row[feature])}
    if feature in ("volume_trend_12m", "office_volume_trend_vs_market_12m"):
        pct_change = abs(row[feature] - 1.0)
        return {"pct_change": f"{pct_change:.0%}"}
    if feature in ("moves_5y_asof", "new_listings_12m"):
        return {"n": int(row[feature])}
    if feature == "office_brand_independent":
        return {}
    return {}


def _resolve_template_key(feature: str, row: pd.Series, shap_val: float) -> str:
    if feature == "tenure_current_office_months":
        return _tenure_variant(row, shap_val)
    return feature


def explain_agent(row: pd.Series, shap_row: np.ndarray, feature_names: list[str], top_n: int = TOP_N_BULLETS) -> dict:
    """Returns {"top_driver": str, "supporting": [str, ...]} for one agent --
    kept structurally separate (not just a flat list) so the top driver can
    be rendered in its own highlighted box, supporting insights in a plain
    list below it -- no text marks it as "the top one," the box styling
    does that visually. Supporting bullets below MIN_SUPPORTING_SHARE of
    this agent's total |SHAP| are dropped rather than padded in -- an
    agent with thin/uninformative data may end up with fewer than 4
    supporting bullets, which is the intended behavior, not a bug."""
    total_shap = np.abs(shap_row).sum()
    templatable = [f for f in feature_names if f in TEMPLATES or f == "tenure_current_office_months"]
    idx = [feature_names.index(f) for f in templatable]
    shap_subset = shap_row[idx]
    # Over-fetch beyond top_n: the concept-group cap below can skip a
    # feature entirely, and without extra candidates in reserve an agent
    # would lose a bullet instead of backfilling from the next-highest one.
    order = np.argsort(-np.abs(shap_subset))

    bullets = []
    used_groups = set()
    rank = 0
    for pos in order:
        if len(bullets) >= top_n:
            break
        feature = templatable[pos]
        shap_val = shap_subset[pos]
        if rank > 0 and total_shap > 0 and abs(shap_val) / total_shap < MIN_SUPPORTING_SHARE:
            break  # ranked by |SHAP| descending, so nothing further qualifies either
        group = CONCEPT_GROUPS.get(feature, feature)
        if group in used_groups:
            continue  # same story as an already-selected bullet -- skip, don't consume a slot
        template_key = _resolve_template_key(feature, row, shap_val)
        variants = TEMPLATES[template_key]
        direction = "risk" if shap_val > 0 else "safe"
        if direction not in variants:
            continue  # e.g. tenure_current_office_new/veteran only have "risk"
        if feature == "new_listings_12m":
            n = row[feature]
            if direction == "risk" and n > NEW_LISTINGS_LOW_MAX:
                continue  # e.g. "only 448" -- not actually low, don't claim it is
            if direction == "safe" and n < NEW_LISTINGS_HIGH_MIN:
                continue  # e.g. "0 listings, active" -- not actually active
        bullets.append(variants[direction].format(**_format_args(feature, row)))
        used_groups.add(group)
        rank += 1

    if not bullets:
        return {"top_driver": None, "supporting": []}
    return {"top_driver": bullets[0], "supporting": bullets[1:5]}


def generate_explanations(model: CatBoostClassifier, X: pd.DataFrame, raw: pd.DataFrame = None) -> pd.DataFrame:
    """One row in, structured explanation columns out, per agent in X (same
    row order): why_top_driver + why_reason_2..5. Computes real per-agent
    SHAP contributions in one batched CatBoost call -- not one model call
    per agent.

    `raw`, if given (same row order/index as X, e.g. the full current
    snapshot before it's sliced down to PRODUCTION_FEATURES), is used ONLY
    for building each agent's display row -- lets _format_args reach columns
    like company_active_agents_asof that aren't themselves a model feature
    (only their derived trend ratio is) without adding anything to the
    model's actual SHAP input. Falls back to X alone if omitted."""
    pool = Pool(X, cat_features=[])
    shap_values = model.get_feature_importance(pool, type="ShapValues")  # (n, n_features+1); last col = bias
    feature_shap = shap_values[:, :-1]
    feature_names = list(X.columns)
    display_source = raw if raw is not None else X

    rows = []
    for i in range(len(X)):
        result = explain_agent(display_source.iloc[i], feature_shap[i], feature_names)
        supporting = result["supporting"] + [None] * (4 - len(result["supporting"]))
        rows.append({
            "why_top_driver": result["top_driver"],
            "why_reason_2": supporting[0],
            "why_reason_3": supporting[1],
            "why_reason_4": supporting[2],
            "why_reason_5": supporting[3],
        })
    return pd.DataFrame(rows, index=X.index)


def _train_production_model(df: pd.DataFrame) -> CatBoostClassifier:
    # AutoResearch-winning config (2026-07-25 promotion) -- see crossval.py's
    # PRODUCTION_FEATURES / PRODUCTION_CATBOOST_PARAMS docstring for provenance.
    X = df[PRODUCTION_FEATURES].astype(float)
    y = df["label_left_3m"].to_numpy()
    monotone = [MONOTONE.get(c, 0) for c in PRODUCTION_FEATURES]
    model = CatBoostClassifier(monotone_constraints=monotone, **PRODUCTION_CATBOOST_PARAMS)
    model.fit(X, y)
    return model


def main():
    print("[explain] loading + training production model...")
    df = load_clean()
    model = _train_production_model(df)

    print("[explain] scoring + explaining current agents...")
    current = load_current_snapshot()
    X_current = current[PRODUCTION_FEATURES].astype(float)
    why = generate_explanations(model, X_current, raw=current)

    demo = current[["mls_agent_id", "office_name"]].copy()
    demo = pd.concat([demo.reset_index(drop=True), why.reset_index(drop=True)], axis=1)

    # Demo on agents already discussed at length: the original "0 sales,
    # 9yr tenure" agent that kicked off the team investigation, plus the
    # then-current top-5 Extra-High-tier agents. Re-pointed 2026-08-10 from
    # the retired agent_profile_id UUIDs to mls_agent_id via
    # agent_id_crosswalk.csv -- same five people, readable IDs. This list is
    # a fixed historical sample, so it drifts out of the live top-5 as
    # scores move; the loop below just skips any ID no longer present.
    highlight_ids = [
        "38002",  # Curtis Lopes, Century 21 Limitless -- the original mystery agent
        "34684",  # Mark Sullivan
        "45328",  # Michele Lombardi
        "44863",  # Noel Abi-Kharma
        "49350",  # David Pedro
    ]
    print("\n" + "=" * 100)
    print("DEMO: explanations for previously-discussed / current top-risk agents")
    print("=" * 100)
    for aid in highlight_ids:
        row = demo[demo["mls_agent_id"] == aid]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"\n{r['mls_agent_id']}  |  {r['office_name']}")
        print(f"  {r['why_top_driver']}")
        for col in ["why_reason_2", "why_reason_3", "why_reason_4", "why_reason_5"]:
            if pd.notna(r[col]):
                print(f"    - {r[col]}")

    demo.to_csv("agent_explanations_demo.csv", index=False)
    print(f"\n[explain] wrote {len(demo)} explanations -> agent_explanations_demo.csv (demo only, not yet wired into score_agents.py)")


if __name__ == "__main__":
    main()
