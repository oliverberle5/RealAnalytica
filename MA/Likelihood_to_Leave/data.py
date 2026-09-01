"""Shared data loading for the MASSACHUSETTS (MLSPIN) likelihood-to-leave model.

PROVENANCE (2026-07-28): ported from the RI project
(`RI/Likelihood_to_Leave/data.py`) by changing MLS_CODE from
"riar" to "mlspin" and nothing else about the modeling logic. The source CSV is
the SAME file, read from the RI project's directory -- MLSPIN and RIAR are two
MLS panels bundled in one export, so "the MA model" is the same pipeline over
`mls_code == "mlspin"`. Everything below this paragraph that describes the
label reconstruction, censoring rule, team exclusion and feature engineering
applies identically to both states and is left unedited.

FOUR THINGS DIFFER FROM THE RI FILE, all data-driven and each marked inline:
  1. MLS_CODE / DATA_PATH (below).
  2. NON_MLS_MEMBER_AGENT_IDS -- MLSPIN has its own synthetic placeholder
     record with a different mls_agent_id; the RI id appears nowhere in
     MLSPIN, so inheriting it would have silently excluded nothing.
  3. OFFICE_BRAND_CATEGORIES -- MLSPIN contains `bhgre`, RIAR does not.
  4. rebrand_classifications -- RI's office-pair list is keyed on RIAR
     office_mls_ids that do not exist in MLSPIN. Regenerated for MA by
     `derive_ma_office_pairs.py`; see that script for what is and isn't
     verified. This is the one place where the MA model is currently WEAKER
     than RI: no MA pair has been web-verified, so MA gets no equivalent of
     RI's confirmed-M&A trickle handling.

MA is a substantially larger panel than RI: 612,358 rows / 45,186 agents /
20 raw quarters, versus RI's 84,417 / 5,625 / 20 -- roughly 7x the rows and
8x the agents. Post-load_clean (re-measured 2026-08-05, superseding the
earlier ~11,354 / ~2,168, which predate the current label cleanup): MA
556,365 usable rows / 9,737 positives / 1.7501%, versus RI's 77,422 / 1,519
/ 1.9620%, both over 19 usable quarters.

HORIZON: 3 months. This replaced the original 12-month framing entirely
(2026-07 decision) because a 3-month window closes so much faster that far
more of the panel becomes usable and it reaches three extra, more-recent
quarters, and it better matches "who's a live flight risk right now" than a
12-month outlook does. Recomputed on mlspin 2026-08-05 (the figures here
were previously RI's, inherited with the fork): of 612,358 mlspin rows,
583,908 (95.4%) are usable over 19 quarters at 3 months, versus 497,253
(81.2%) over 16 quarters at 12 months. Trade-off, kept deliberately visible
rather than hidden: the 3-month positive rate is only ~2.25% (vs ~8.54% for
12mo), and the absolute count of confirmed leavers actually drops (13,133 vs
42,445) despite more total rows, because 3x fewer of the ever-observed moves
fall inside a 91-day window than a 365-day one.

Label reconstruction: the source CSV only ships label_left_within_12m /
label_days_to_move / label_censored, computed for a 12-month horizon. A
3-month label is reconstructed from label_days_to_move here rather than
shipped directly. This is sound because the source data captures a move as
soon as it's observed regardless of whether the 12-month window has fully
closed (confirmed empirically: 841 rows have label_censored==1 AND
label_left_within_12m==1 -- i.e. a move was recorded even in a
not-yet-elapsed window). So label_days_to_move <= 91 is a reliable 3-month
positive wherever it's known. A row is usable, though, only once its own
3-month window is confirmed to have closed as of the data's true max
observed date (which isn't shipped directly, so a conservative lower-bound
estimate is used -- see _infer_max_observed_date_lower_bound). An observed
move does NOT on its own make a row usable, even though it makes the label
knowable -- see the comment in load_clean.

Hard rule carried over unchanged from the 12-month version: a row's outcome
must be fully resolvable within its own horizon before it's usable. No
"we already know this one, so keep it" special-casing that would bias the
recent edge of the data.

SOURCE DATA (2026-07 update): switched from `leave-dataset-riar.csv` to
`leave-dataset_most_recent_updated.csv`, which bundles RIAR (RI) together
with MLSPIN (MA) rows and adds macro/company columns. THIS project takes the
MLSPIN half, so `load_clean` filters `mls_code == "mlspin"` immediately after
reading (the RI project applies the same line with "riar"). Note that every
market_* and state_* column is keyed to the MLS, not to the agent's home
address -- MLSPIN rows carry Massachusetts market and macro series regardless
of where the individual agent lives. The RIAR subset of the new file is the same
84,417 rows as before, minus 55 exact-duplicate rows the old file had
(verified byte-identical across every column) -- no rows were lost.
`market_median_price_12m`/`_trend` were renamed `market_avg_price_12m`/
`_trend` upstream and no longer exist under the old name. New market
totals (market_total_volume_12m/_trend, market_total_sides_12m) and macro
indicators (state_unemployment_rate, state_unemployment_change_12m_pts,
cpi_yoy_pct) now have 100% coverage across all 20 quarters -- the old
pre-2024-06 market-column gap is closed except for market_avg_dom_12m,
which still has it (see RAW_FEATURE_COLUMNS comment, unchanged/excluded
per-agent avg_dom_12m issue is separate and also still unresolved).

SOURCE DATA (2026-07-15 update): switched again, to `leave-dataset (1).csv`
(filename kept as delivered -- see the file for why it wasn't renamed).
Same 83 raw columns, same RIAR row count (84,417), same 20 quarters
(2021-10-01 through 2026-07-01), zero duplicate rows or duplicate
(agent_profile_id, snapshot_date) keys -- verified before switching.
NOT a pure re-export, though: per-agent `avg_dom_12m`'s data-quality bug
(see RAW_FEATURE_COLUMNS comment) appears fixed upstream, and re-running
the existing full-feature reference config (crossval.py, unchanged feature
set) on the new file shows CatBoost mean AUC moving from 0.7368 to 0.7384
and XGBoost 0.7313 to 0.7328 -- both up slightly, suggesting some other
column values were quietly refreshed/corrected too, not just avg_dom_12m.
Because of this, autoresearch results measured against the OLD file are
NOT directly comparable to results against this one; the production
pruned-40 CatBoost config, freshly re-fit on this file, lands at mean
0.739950 / worst-fold 0.718956 -- treated as a new baseline, not a
regression from the old file's 0.741780 / 0.718821.

SOURCE DATA (2026-07-25 update): switched to
`leave-dataset-with-team-distinction.csv` -- same 696,775 rows, same 83
original columns, plus two new ones: `agent_name` and `is_team` (int 0/1).
Verified before switching: identical (agent_profile_id, snapshot_date,
mls_code) keys and identical volume_12m sum to the old file -- a pure
column addition, not a re-pull. `is_team` is STATIC per agent_profile_id
(confirmed: 0 of 50,811 IDs show more than one distinct value across their
observed quarters), so no (agent, quarter) grain concern despite the
roster refreshing quarterly going forward -- re-check this each quarter
though, since a future refresh could in principle relabel an existing ID
rather than only minting new ones (only one snapshot of the flag has been
observed so far). Confirms the "team-account contamination" issue this
project's own HANDOFF.md flagged as diagnosed-but-blocked: 155 of 5,625
RIAR agent_profile_ids (2.76%) are team accounts, e.g. "Nathan Clark Team"
($146.2M/543 units) sits on a SEPARATE agent_profile_id from "Nathan
Clark" the individual ($0/1 unit, 87mo tenure) -- team production credit
and the real person are genuinely split across two IDs, exactly the
mechanism diagnosed. `load_clean` now drops team-flagged rows entirely
(see the is_team filter below) so the individual churn model trains and
tests only on individual agents; team accounts are excluded, not scored,
until a dedicated team-level model exists (not yet built).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rebrand_classifications import CLUSTER_GATED_PAIRS, CONFIRMED_NOT_CHURN_PAIRS

# All four models (RI/MA x leave/sales) read this ONE file and differ only
# by their mls_code filter. It lives at <repo_root>/data/ and is NOT in
# git -- see README.md 'Getting the data'.
DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "leave-dataset-with-team-distinction-mls-id.csv"
)
MLS_CODE = "mlspin"

# The agent identifier column. Named here rather than inlined so the next
# identifier change is a one-line edit, and so the read_csv dtype override
# cannot drift apart from the column it protects. MUST stay str: three RIAR
# ids carry meaningful leading zeros ("00001", "0012", "0014") and every
# MLSPIN id is alphanumeric ("A9500308", "CN235473"), so dtype inference
# corrupts the first group and yields a mixed int/str column across the
# second -- which would make the exclusion sets below silently stop matching.
AGENT_ID_COLUMN = "mls_agent_id"

HORIZON_DAYS = 91
HORIZON_MONTHS = 3

# Snapshots with too few rows to be a real cohort (data-edge artifact, e.g.
# the 10-row 2026-06-30 snapshot) rather than a real quarter of the panel.
MIN_SNAPSHOT_ROWS = 500

# MLSPIN's synthetic placeholder record, not a real agent -- the MA analog of
# the RI project's "Non-Mls Member" bucket, holding buyer-side transactions
# where the cooperating agent isn't a board member. Identified by matching the
# RI record's exact structural signature rather than by name alone: present in
# all 20 snapshots, is_team==0 (so the team filter doesn't catch it), ZERO
# list-sides ever (1,678 units in the latest quarter, all buy-side),
# monotonically incrementing career_months, and -- coincidentally identical to
# the RI placeholder -- 1.16% of market volume in the latest quarter.
#
# 2026-08-10: re-pointed from the retired UUID
# "0d2b79f7-a100-45a2-9760-4c0ad3bb3849" to MLS ID "H1111111" alongside the
# mls_agent_id switch. Verified the same record is matched under the new ID:
# agent_name "Non Member", office_name "Non Member Office", 20 snapshots,
# 23,127 units all buy-side, 0 list-sides ever. Left on the UUID it would
# have matched nothing and silently un-excluded the placeholder.
#
# RI's second placeholder "00001" (Assisted Sale) is deliberately NOT added
# here: it is a riar id, MLS_CODE below filters to mlspin, so it could only
# ever be dead weight. MLSPIN was scanned for its own equivalent when the
# readable IDs landed (non-team, all 20 snapshots, single office, >=2,000
# units, degenerate list/buy split) and "H1111111" is the only hit -- the
# two other low-entropy MLSPIN ids, "H9999999" and "Z1111111", are ordinary
# low-volume agents with real names, not buckets.
NON_MLS_MEMBER_AGENT_IDS = {"H1111111"}

# Minimum simultaneous same-origin-to-same-destination movers, in the SAME
# quarter, to treat a transition as a corporate/structural event rather than
# individual attrition (see _recode_mass_mover_moves). Shared with
# detect_transitions.py so the threshold used to FIND candidates and the
# threshold used to actually recode labels never drift apart.
MIN_CLUSTER_MOVERS = 5

ID_COLUMNS = [
    "mls_agent_id",
    "mls_code",
    "snapshot_date",
    "office_mls_id",
    "office_name",
    "agent_city",
    "agent_state",
]

LABEL_COLUMN = "label_left_3m"

# heuristic_recruitability_score is dropped entirely per instruction: it is
# not a benchmark, not a feature, not used anywhere in this pipeline.
EXCLUDED_COLUMNS = ["heuristic_recruitability_score"]

RAW_FEATURE_COLUMNS = [
    "tenure_current_office_months",
    "career_months",
    "num_offices_career",
    "moves_5y_asof",
    "months_since_last_move",
    "units_12m",
    "volume_12m",
    "units_prior_12m",
    "volume_prior_12m",
    "volume_trend_12m",
    "units_3m",
    "units_prior_3m",
    "trend_3m",
    "list_sides_12m",
    "buy_sides_12m",
    "list_share_12m",
    "new_listings_12m",
    "rentals_12m",
    "colist_sides_12m",
    "avg_sale_price_12m",
    "price_min_12m",
    "price_max_12m",
    # avg_dom_12m (per-agent) intentionally excluded -- but the reason
    # changed 2026-07-15. Originally excluded for being BROKEN (59% zeros,
    # median 0, max 378, flagged via diagnose_fold5.py's drift check as
    # noise, not signal). The source data was fixed upstream in the
    # 2026-07-15 file: null rate now 36.4%, zero rate among non-null only
    # 0.8% (was 59%), median 27 days, sane IQR (13.8-46 days), consistent
    # with market_avg_dom_12m's 35.7-day median. Re-tested with this much
    # larger, clean sample (~53,653 usable rows, vs. ~1,600-1,800 in the
    # prior inconclusive test -- see explore_avg_dom.py): CatBoost's own
    # importance ranking DOES rank it above the pruning cutoff (it looks
    # useful in-sample), but including it makes held-out performance WORSE,
    # not better, on both the full reference config (crossval.py: 0.7384 ->
    # 0.7353 mean AUC) and the production pruned-40 config (0.739950 ->
    # 0.737228 mean, 0.718956 -> 0.714003 worst-fold). This resolves the
    # open question conclusively: fixed or not, it doesn't help. Still
    # excluded, now for a settled reason rather than a data-quality one.
    # Use market_avg_dom_12m (state-level, sanity-checked as legitimate:
    # 52-70 day range) for days-on-market / market-heat signal instead.
    "months_active_12m",
    "months_since_last_closing",
    "pct_units_single_family_12m",
    "pct_units_condo_12m",
    "pct_units_multi_family_12m",
    "pct_units_rental_12m",
    "cohort_volume_quartile",
    "volume_delta_vs_cohort_median_pts",
    "share_of_office_volume_12m",
    "rank_in_office_volume_12m",
    "office_roster_pct_rank",
    "price_gap_vs_office",
    "office_active_agents_asof",
    "office_arrivals_12m",
    "office_exits_12m",
    "office_net_flow_12m",
    "office_exit_rate_12m",
    "office_units_12m",
    "office_volume_12m_total",
    "office_volume_prior_12m",
    "office_volume_trend_12m",
    "office_avg_sale_price_12m",
    "office_is_branch",
    "office_age_months",
    "market_avg_price_12m",
    "market_avg_price_trend_12m",
    "market_closed_sides_trend_12m",
    "market_total_volume_12m",
    "market_total_volume_trend_12m",
    "market_total_sides_12m",
    "market_avg_dom_12m",
    # Macro indicators (2026-07 addition): vary by MLS-market state (RI vs
    # MA), not by the agent's personal mailing-address state -- confirmed
    # correct, not a bug. cpi_yoy_pct is national (CPI has no state grain),
    # shared across markets. 100% coverage across all 20 quarters.
    "state_unemployment_rate",
    "state_unemployment_change_12m_pts",
    "cpi_yoy_pct",
    # Company-level rollup (2026-07 #4 addition): parallel to the office_*
    # set above, one level up the org hierarchy (a company can own several
    # offices -- company_offices_count). Was shipped alongside the 2026-07
    # macro update but not wired in until this pass. 100% coverage except
    # company_volume_12m (0.15% null) and company_volume_trend_12m (4.1%
    # null, presumably 0/0-undefined trend like other trend columns).
    "company_offices_count",
    "company_active_agents_asof",
    "company_volume_12m",
    "company_volume_trend_12m",
    "pct_rank_volume_in_company_12m",
    "company_arrivals_12m",
    "company_exits_12m",
    "company_net_flow_12m",
    "company_exit_rate_12m",
    # office_exit_concentration_12m: HHI-style concentration of WHICH office
    # leavers went where (was previously only computed ad hoc as a one-off
    # diagnostic in diagnose_fold5.py -- it's actually shipped directly in
    # the source data). Null exactly when office_exits_12m==0 (undefined
    # with no exits to concentrate) -- confirmed structural, not a bug.
    "office_exit_concentration_12m",
    # Agent's own history of prior-office stint lengths -- a history of
    # short stints may predict another one. Null exactly when
    # num_offices_career==1 (agent has never had a prior office) --
    # confirmed structural, not a bug. Trees handle NaN natively; the
    # linear model's existing missing-indicator + median-fill handles it.
    "prev_stint_months",
    "avg_prior_stint_months",
    "max_prior_stint_months",
    # office_brand one-hot dummies (2026-07 #4 addition): franchise brand
    # of the CURRENT office (independent vs. RE/MAX vs. Keller Williams,
    # etc.) -- categorical, 15 known values as of this data pull. Included
    # as plain 0/1 columns (not drop-first) since trees don't need a
    # reference level and it keeps the linear model's CONTINUOUS_FEATURES
    # handling (median/missing-indicator machinery) irrelevant here -- a
    # 0/1 dummy has no missing values to begin with.
    "office_brand_independent",
    "office_brand_remax",
    "office_brand_keller_williams",
    "office_brand_century21",
    "office_brand_coldwell_banker",
    "office_brand_compass",
    "office_brand_sothebys",
    "office_brand_exp",
    "office_brand_raveis",
    "office_brand_era",
    "office_brand_redfin",
    "office_brand_laer",
    "office_brand_corcoran",
    "office_brand_elliman",
    "office_brand_berkshire_hathaway",
    "office_brand_bhgre",  # MA-only brand, absent from RIAR
]

# MA note: `bhgre` (Better Homes & Gardens Real Estate) is appended relative to
# the RI list -- it is present in MLSPIN (0.17% of rows) and absent from RIAR
# entirely. Everything else matches. Two brands carry very different weight in
# MA than RI and are worth remembering when comparing feature importances
# across the two models: berkshire_hathaway is 3.40% of MA rows vs 0.01% of RI
# (effectively an RI-absent brand), and remax is 4.32% in MA vs 10.50% in RI.
OFFICE_BRAND_CATEGORIES = [
    "independent", "remax", "keller_williams", "century21", "coldwell_banker",
    "compass", "sothebys", "exp", "raveis", "era", "redfin", "laer",
    "corcoran", "elliman", "berkshire_hathaway", "bhgre",
]


def add_office_brand_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for cat in OFFICE_BRAND_CATEGORIES:
        df[f"office_brand_{cat}"] = (df["office_brand"] == cat).astype(int)
    return df

# Derived ratio/trend features. The source data ships some level features
# with no accompanying trend (e.g. office headcount, agent unit count) even
# though the sibling "prior period" or "denominator" column needed to build
# one is already present elsewhere in the row. A raw level (e.g. "8 agents
# left this office") means something different for a 5-person office than a
# 50-person one; a normalized ratio corrects for that. Each one here is
# derived purely from columns already in the source file -- no new data.
#
# Also audited but NOT addable from current columns (would need new source
# data, not just arithmetic):
#   - office-level unit-count trend: office_units_prior_12m doesn't exist
#     (only office_volume_prior_12m, i.e. dollars, not counts)
#   - agent-level price trend: no avg_sale_price_prior_12m column
#   - rank/percentile trend (rank_in_office_volume_12m, office_roster_pct_rank):
#     would need each agent's OWN prior-quarter row (a panel lag-join), not a
#     same-row ratio -- different engineering pattern, flagged for later
DERIVED_RATIO_COLUMNS = [
    "office_headcount_trend_12m",
    "units_trend_12m",
    "office_arrival_rate_12m",
    "share_of_office_volume_trend_12m",
    "volume_trend_vs_market_12m",
    "units_trend_vs_market_12m",
    "office_volume_trend_vs_market_12m",
    # 2026-07 #4 additions: same treatment one level up the org hierarchy,
    # now that company_* columns are wired in.
    "company_headcount_trend_12m",
    "company_arrival_rate_12m",
    "company_volume_trend_vs_market_12m",
]

RAW_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + DERIVED_RATIO_COLUMNS


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator > 0)
    return numerator / denom


def add_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio/trend features derivable from columns already in the row.

    office_headcount_trend_12m: current office headcount vs. its estimated
    headcount 12mo ago (active_agents_asof - net_flow_12m). >1 = growing
    office, <1 = shrinking -- normalizes office_net_flow_12m (an absolute
    count) by office size, the same way volume_trend_12m already normalizes
    dollar volume. NaN when the estimated prior headcount is <= 0.

    units_trend_12m: agent's own unit-count trend (units_12m /
    units_prior_12m), the unit-count sibling of the existing dollar-based
    volume_trend_12m -- units_prior_12m was already a column but had no
    trend built from it.

    office_arrival_rate_12m: office_arrivals_12m normalized by office size,
    symmetric to the existing office_exit_rate_12m -- growth/recruiting
    pressure alongside attrition pressure.

    share_of_office_volume_trend_12m: is this agent becoming MORE or LESS
    central to their office's book of business over time? Ratio of current
    share_of_office_volume_12m to the same ratio computed on the prior
    12-month period.

    volume_trend_vs_market_12m / units_trend_vs_market_12m (2026-07
    addition): the agent's OWN dollar-volume / unit-count trend divided by
    the overall market's trend over the same 12-month window -- relative,
    not absolute, performance. An agent whose volume fell 12% in a market
    that fell 25% has a ratio > 1 (relatively strong, despite the raw
    decline); an agent flat in a rising market has a ratio < 1 (relatively
    weak, despite no raw decline). volume_trend_12m and
    market_total_volume_trend_12m are both already same-quarter/prior-quarter
    ratios at the same 12m grain, so dividing one by the other is a
    like-for-like comparison, not a unit mismatch.

    office_volume_trend_vs_market_12m: same idea one level up -- is this
    agent's OWN OFFICE outperforming or lagging the broader market? Lets the
    model separate "this agent is underperforming their own healthy office"
    from "this agent's whole office is swept up in a market downturn," which
    plausibly carry different flight-risk implications.

    company_headcount_trend_12m / company_arrival_rate_12m /
    company_volume_trend_vs_market_12m (2026-07 #4 addition): identical
    constructions to the office-level versions above, one level further up
    the org hierarchy (a company can span several offices). Lets the model
    separate "this agent's specific office is struggling" from "the whole
    parent company is struggling" from "the whole market is struggling."
    """
    df = df.copy()

    prior_headcount = df["office_active_agents_asof"] - df["office_net_flow_12m"]
    df["office_headcount_trend_12m"] = _safe_ratio(df["office_active_agents_asof"], prior_headcount)

    df["units_trend_12m"] = _safe_ratio(df["units_12m"], df["units_prior_12m"])

    # 2026-07-28 fix: was normalizing arrivals by CURRENT headcount while
    # office_exit_rate_12m (source column) normalizes exits by PRIOR
    # headcount -- an inconsistent base for two rates meant to be symmetric
    # "agents coming vs. going" measures. Now both share prior_headcount.
    # Verified via crossval on a cloned dataset before applying here: costs
    # production CatBoost ~0.3pp mean AUC (0.7434 -> 0.7402), adopted anyway
    # for correctness/defensibility, not for an AUC gain.
    df["office_arrival_rate_12m"] = _safe_ratio(df["office_arrivals_12m"], prior_headcount)

    prior_share = _safe_ratio(df["volume_prior_12m"], df["office_volume_prior_12m"])
    current_share = df["share_of_office_volume_12m"]
    df["share_of_office_volume_trend_12m"] = _safe_ratio(current_share, prior_share)

    df["volume_trend_vs_market_12m"] = _safe_ratio(df["volume_trend_12m"], df["market_total_volume_trend_12m"])
    df["units_trend_vs_market_12m"] = _safe_ratio(df["units_trend_12m"], df["market_closed_sides_trend_12m"])
    df["office_volume_trend_vs_market_12m"] = _safe_ratio(
        df["office_volume_trend_12m"], df["market_total_volume_trend_12m"]
    )

    company_prior_headcount = df["company_active_agents_asof"] - df["company_net_flow_12m"]
    df["company_headcount_trend_12m"] = _safe_ratio(df["company_active_agents_asof"], company_prior_headcount)
    # 2026-07-28 fix: same denominator-consistency fix as
    # office_arrival_rate_12m above, one level up the org hierarchy.
    df["company_arrival_rate_12m"] = _safe_ratio(df["company_arrivals_12m"], company_prior_headcount)
    df["company_volume_trend_vs_market_12m"] = _safe_ratio(
        df["company_volume_trend_12m"], df["market_total_volume_trend_12m"]
    )

    # Ratio-of-ratios features (e.g. share_of_office_volume_trend_12m) blow
    # up toward infinity when a tiny prior-period denominator is involved
    # (99.9th pctile ~38x, but max observed ~247,819x from one near-zero
    # denominator). Cap at the 99.9th-percentile-ish level so a handful of
    # pathological rows don't dominate the linear model's scaling; trees are
    # unaffected by this either way since they split on order, not magnitude.
    clip_caps = {
        "office_headcount_trend_12m": 30,
        "units_trend_12m": 20,
        "office_arrival_rate_12m": 5,
        "company_headcount_trend_12m": 30,
        "company_arrival_rate_12m": 5,
        "company_volume_trend_vs_market_12m": 20,
        "share_of_office_volume_trend_12m": 50,
        "volume_trend_vs_market_12m": 20,
        "units_trend_vs_market_12m": 20,
        "office_volume_trend_vs_market_12m": 20,
    }
    for col, cap in clip_caps.items():
        df[col] = df[col].clip(upper=cap)

    return df


# GEPA autoresearch champion (round r003_20260726_204147, CONFIRMED by
# confirm.py 2026-07-27 19:02:22 -- see AutoResearch/gepa_loop/best/champion.json).
# Seed-averaged gain of +0.0011 AUC over the 2026-07-25 production baseline,
# holding on both shifted fold structures (2/2) and improving 3/5 production
# folds -- a real improvement, not window-fitting. Promoted into production
# 2026-07-28.
AUTORESEARCH_CHAMPION_COLUMNS = [
    "tenure_to_stint_ratio",
    "closing_recency_vs_move",
    "pipeline_conversion_ratio",
    "dom_adjusted_closing_recency",
    "trend_deceleration_gap",
]


def add_autoresearch_champion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Five row-wise engineered features from the confirmed GEPA champion.

    tenure_to_stint_ratio: current stint length vs. the agent's own
    historical norm -- a stint that has run past the agent's typical prior
    stint is a "due for a move" signal relative to the agent, not absolute.

    closing_recency_vs_move: gap between "months since last closing" and
    "months since last move" -- flags disengagement independent of a normal
    post-move ramp-up dip.

    pipeline_conversion_ratio: new listings taken on per unit actually
    closed -- a high ratio alongside soft production suggests an agent
    building listings but failing to convert.

    dom_adjusted_closing_recency: closing recency normalized by current
    market speed, since a 6-month closing gap means something different in
    a fast market vs. a slow one.

    trend_deceleration_gap: short-term (3m) trend minus the agent's own
    longer-run (12m) trend -- the GAP is the inflection-point signal that
    neither trend column alone encodes.
    """
    df = df.copy()

    df["tenure_to_stint_ratio"] = df["tenure_current_office_months"] / (
        df["avg_prior_stint_months"].fillna(0.0) + 1.0
    )
    df["closing_recency_vs_move"] = (
        df["months_since_last_closing"] - df["months_since_last_move"]
    )
    df["pipeline_conversion_ratio"] = df["new_listings_12m"] / (df["units_12m"] + 1.0)
    df["dom_adjusted_closing_recency"] = df["months_since_last_closing"] / (
        df["market_avg_dom_12m"].fillna(0.0) / 30.0 + 1.0
    )
    df["trend_deceleration_gap"] = df["units_trend_12m"].fillna(0.0) - df["trend_3m"].fillna(0.0)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df


def _parse_snapshot_date(series: pd.Series) -> pd.Series:
    # New source ships plain ISO dates ("2021-10-01"), quarter-START not
    # quarter-END like the old file's "Thu Sep 30 2021 ... GMT-0400" strings
    # -- an offset-by-one-day convention shift, harmless here since all date
    # math below (HORIZON_MONTHS offsets, fold boundaries) is relative to
    # whatever snapshot_date means, not tied to a specific day-of-quarter.
    return pd.to_datetime(series)


def _infer_max_observed_date_lower_bound(df: pd.DataFrame) -> pd.Timestamp:
    """Conservative lower bound on the data's true max observed date.

    The source 12-month censoring flag tells us: the last snapshot with
    label_censored==0 had its full 12-month window observed, so the true
    max observed date is AT LEAST that snapshot's date + 12 months. Using
    this as a lower bound (rather than trying to guess the true value) errs
    toward excluding borderline-usable rows rather than falsely including
    ambiguous ones.
    """
    last_uncensored = df.loc[df["label_censored"] == 0, "snapshot_date"].max()
    return last_uncensored + pd.DateOffset(months=12)


def _build_moves(raw: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct agent office-to-office moves from consecutive snapshots.

    Independent of the label/censoring logic -- tracks each agent's
    office_mls_id across their own observed snapshots in order, so it can
    identify the DESTINATION office for a move even though the per-row
    label machinery elsewhere only knows THAT a move happened, not where.
    """
    m = raw[["mls_agent_id", "office_mls_id", "snapshot_date"]].sort_values(
        ["mls_agent_id", "snapshot_date"]
    )
    m["prev_office"] = m.groupby("mls_agent_id")["office_mls_id"].shift(1)
    moved = m[m["office_mls_id"] != m["prev_office"]].dropna(subset=["prev_office"])
    return moved.rename(columns={"office_mls_id": "dest_office", "prev_office": "origin_office", "snapshot_date": "move_quarter"})[
        ["mls_agent_id", "origin_office", "dest_office", "move_quarter"]
    ]


def _recode_mass_mover_moves(clean: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Zero out label_left_3m for rows that aren't real individual attrition.

    Per the 2026-07 #3 policy (see rebrand_classifications.py docstring),
    two different gates apply depending on how confident we are about the
    (origin, dest) office pair:

    - CONFIRMED_NOT_CHURN_PAIRS (web-verified rebrand/M&A/roll-up): EVERY
      move on that exact pair recodes to 0, in ANY quarter, at ANY volume.
      Independent evidence already tells us it's the same non-event,
      regardless of how many agents moved that particular quarter.

    - CLUSTER_GATED_PAIRS (confirmed-unrelated-competitor or unconfirmed --
      no independent evidence either way): a move only recodes to 0 if THIS
      SPECIFIC QUARTER independently saw >= MIN_CLUSTER_MOVERS agents make
      that exact same-origin-to-same-destination move together -- the only
      evidence we have for these is the mass-mover pattern itself, so only
      the actual flagged cluster instance(s) count, not a blanket amnesty
      for the pair. A lone agent making the same move outside a genuine
      cluster quarter still counts as real churn.

    Matches each label_left_3m==1 row to the destination office its agent
    landed at ~3 months later, then applies whichever gate the pair falls
    under.
    """
    moves = _build_moves(raw)
    cluster_sizes = moves.groupby(["origin_office", "dest_office", "move_quarter"]).size()

    candidates = clean[clean["label_left_3m"] == 1][["mls_agent_id", "office_mls_id", "snapshot_date"]].copy()
    candidates["move_quarter"] = candidates["snapshot_date"] + pd.DateOffset(months=HORIZON_MONTHS)
    matched = candidates.merge(
        moves,
        left_on=["mls_agent_id", "office_mls_id", "move_quarter"],
        right_on=["mls_agent_id", "origin_office", "move_quarter"],
        how="left",
    )

    def _is_not_churn(r: pd.Series) -> bool:
        if pd.isna(r["dest_office"]):
            return False
        pair = (r["origin_office"], r["dest_office"])
        if pair in CONFIRMED_NOT_CHURN_PAIRS:
            return True
        if pair in CLUSTER_GATED_PAIRS:
            n_movers = cluster_sizes.get((r["origin_office"], r["dest_office"], r["move_quarter"]), 0)
            return n_movers >= MIN_CLUSTER_MOVERS
        return False

    is_not_churn = matched.apply(_is_not_churn, axis=1)
    recode_index = candidates.index[is_not_churn.to_numpy()]

    clean = clean.copy()
    clean.loc[recode_index, "label_left_3m"] = 0
    print(
        f"[data] recoded {len(recode_index)} rows from label_left_3m=1 -> 0: confirmed rebrand/M&A/"
        f"roll-up pairs (any quarter) or unrelated-competitor mass-mover clusters "
        f"(>= {MIN_CLUSTER_MOVERS} agents, same origin/dest/quarter) -- not individual attrition"
    )
    return clean


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared by both the training path (load_clean) and
    the production scoring path (load_current_snapshot) -- office-brand
    one-hot dummies + derived ratio/trend features. Touches no label column,
    so it's safe to run on rows with no resolvable outcome yet.
    """
    df = add_office_brand_dummies(df)
    df = add_derived_ratios(df)
    df = add_autoresearch_champion_features(df)
    return df


def load_current_snapshot(path: Path = DATA_PATH, exclude_team: bool = True, exclude_non_member: bool = True) -> pd.DataFrame:
    """Load the most recent available quarter's rows, feature-engineered
    the same way as training, for production scoring of currently-active
    agents.

    Deliberately does NOT apply load_clean's label-resolvability filter or
    degenerate-snapshot drop -- those exist to guarantee a TRAINABLE label
    (we need to already know what happened to score a row as a training
    example). That's irrelevant here: scoring is inherently about the
    future, so the freshest quarter -- even one where the 3-month outcome
    can't be confirmed yet because it only just started -- is exactly the
    one we want to score.

    exclude_team: STANDARD as of 2026-07-25 (defaults True). A team account's
    mls_agent_id doesn't represent one person's personal churn risk --
    scoring it with the individual model would produce a number that isn't
    meaningful (see the "team-account contamination" HANDOFF.md section).
    Team-flagged agents simply don't appear in current_agent_risk_scores.csv
    until a dedicated team-level model exists (not yet built) -- a known,
    documented gap, not silent data loss.
    """
    df = pd.read_csv(path, low_memory=False, dtype={AGENT_ID_COLUMN: str})
    if "mls_code" in df.columns:
        df = df[df["mls_code"] == MLS_CODE].copy()
    df["snapshot_date"] = _parse_snapshot_date(df["snapshot_date"])

    latest_date = df["snapshot_date"].max()
    current = df[df["snapshot_date"] == latest_date].copy()

    if exclude_team:
        n_before = len(current)
        current = current[current["is_team"] == 0].copy()
        print(f"[data] current snapshot: excluded {n_before - len(current)} team-account rows")

    if exclude_non_member:
        n_before_nm = len(current)
        current = current[~current["mls_agent_id"].isin(NON_MLS_MEMBER_AGENT_IDS)].copy()
        print(f"[data] current snapshot: excluded {n_before_nm - len(current)} non-MLS-member placeholder rows")

    current = _engineer_features(current)
    print(f"[data] current snapshot: {latest_date.date()}, {len(current)} active agents")
    return current


def load_clean(path: Path = DATA_PATH, exclude_team: bool = True, exclude_non_member: bool = True) -> pd.DataFrame:
    """Load the dataset and return only rows with a resolvable 3-month outcome.

    exclude_team: STANDARD as of 2026-07-25 (defaults True). Validated in
    compare_team_exclusion.py -- a real, consistent AUC improvement (mean
    +0.0047, worst-fold +0.0114 on the current production config, clean
    sweep across all 5 folds) plus the structural correctness argument
    (team production credit and the real individual are split across
    separate mls_agent_ids -- see module docstring). Every production
    caller (score_agents.py, calibrate.py) calls load_clean() with no
    arguments, so this is now the real, live population the production
    model trains and tests on. Exposed as a parameter (rather than hardcoded)
    so compare_team_exclusion.py can still run the SAME code path with and
    without the filter for any future re-validation, instead of
    hand-reconstructing the "included" case separately and risking exactly
    the kind of mismatch (recoding skipped) that a first attempt at this hit.
    """
    df = pd.read_csv(path, low_memory=False, dtype={AGENT_ID_COLUMN: str})
    if "mls_code" in df.columns:
        df = df[df["mls_code"] == MLS_CODE].copy()
    df["snapshot_date"] = _parse_snapshot_date(df["snapshot_date"])
    n_before = len(df)

    max_date_lower_bound = _infer_max_observed_date_lower_bound(df)
    snapshot_plus_horizon = df["snapshot_date"] + pd.DateOffset(months=HORIZON_MONTHS)

    # A row is resolvable ONLY when its own horizon is confirmed closed.
    # Until 2026-08-05 this also admitted any row with an observed move
    # (`move_observed | window_confirmed_elapsed`, where move_observed =
    # label_days_to_move.notna()) -- exactly the "we already know this one,
    # so keep it" special-casing the module docstring forbids. Past the
    # observable edge a mover's move is on record but a stayer's non-move is
    # not, so that branch can only ever admit POSITIVES.
    #
    # Measured before removal (mlspin): it admitted 6 rows, all
    # label_left_3m == 1, all in the 2026-07-01 snapshot -- which
    # MIN_SNAPSHOT_ROWS below then dropped anyway, so this changes no fold
    # and no metric. It was harmless only by accident: max_date_lower_bound
    # currently lands exactly on the max snapshot date, so every earlier
    # snapshot clears the window test on its own and the OR-branch was
    # load-bearing at the final snapshot only. A future drop whose
    # label_censored==0 tail runs several quarters behind the last snapshot
    # would have it admit positives-only rows across multiple recent
    # quarters instead. Fixed in the RI parent at the same time.
    window_confirmed_elapsed = (df["label_censored"] == 0) | (snapshot_plus_horizon <= max_date_lower_bound)
    resolvable = window_confirmed_elapsed

    df["label_left_3m"] = (df["label_days_to_move"] <= HORIZON_DAYS).astype(int)

    clean = df[resolvable].copy()

    # Drop degenerate/artifact snapshots (too few rows to be a real cohort).
    snapshot_counts = clean["snapshot_date"].value_counts()
    degenerate_snapshots = snapshot_counts[snapshot_counts < MIN_SNAPSHOT_ROWS].index
    if len(degenerate_snapshots):
        print(
            f"[data] dropping {len(degenerate_snapshots)} degenerate snapshot(s) with "
            f"< {MIN_SNAPSHOT_ROWS} rows (data-edge artifact): "
            f"{sorted(d.date() for d in degenerate_snapshots)}"
        )
        clean = clean[~clean["snapshot_date"].isin(degenerate_snapshots)]

    clean = _recode_mass_mover_moves(clean, df)

    # Team-account exclusion (2026-07-25): a team's production credit lands
    # on one mls_agent_id while real teammates show near-$0 production
    # on their own separate IDs (see module docstring) -- training the
    # individual churn model on team rows would teach it "this account's
    # future production/tenure dynamics" using one person's features to
    # explain a team's collective behavior. Drop team-flagged rows from
    # BOTH training and test, not just scoring output, per the diagnosed fix.
    if exclude_team:
        n_before_team_filter = len(clean)
        n_team_agents = clean.loc[clean["is_team"] == 1, "mls_agent_id"].nunique()
        clean = clean[clean["is_team"] == 0].copy()
        print(
            f"[data] excluded {n_before_team_filter - len(clean)} team-account rows "
            f"({n_team_agents} distinct team mls_agent_ids) from the individual model"
        )

    if exclude_non_member:
        n_before_nm = len(clean)
        clean = clean[~clean["mls_agent_id"].isin(NON_MLS_MEMBER_AGENT_IDS)].copy()
        print(f"[data] excluded {n_before_nm - len(clean)} non-MLS-member placeholder rows")

    clean = clean.drop(columns=["label_censored", "label_left_within_12m", "label_days_to_move", "is_team", "agent_name"])
    clean = clean.drop(columns=[c for c in EXCLUDED_COLUMNS if c in clean.columns])
    clean = _engineer_features(clean)

    n_after = len(clean)
    print(
        f"[data] loaded {n_before} rows -> {n_after} usable for the {HORIZON_MONTHS}-month "
        f"horizon ({n_before - n_after} dropped: unresolvable or artifact rows). "
        f"3-month positive rate: {clean['label_left_3m'].mean():.4%}"
    )
    return clean


if __name__ == "__main__":
    df = load_clean()
    print(df.groupby(df["snapshot_date"].dt.date)["label_left_3m"].agg(["count", "mean"]))
