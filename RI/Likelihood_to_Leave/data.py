"""Shared data loading for the likelihood-to-leave models.

HORIZON: 3 months. This replaced the original 12-month framing entirely
(2026-07 decision) because a 3-month window closes so much faster that far
more of the panel becomes usable and it reaches three extra, more-recent
quarters, and it better matches "who's a live flight risk right now" than a
12-month outlook does. Recomputed 2026-08-05: of 84,417 riar rows, 79,663
(94.4%) are usable over 19 quarters at 3 months, versus 65,797 (77.9%) over
16 quarters at 12 months. The old wording said "four extra quarters"; that
counted the 2026-07-01 snapshot, which MIN_SNAPSHOT_ROWS drops and which
only reached the count at all via the resolvability bug removed below --
the honest figure is three. Trade-off, kept deliberately visible rather
than hidden: the 3-month positive rate is only ~2.54% (vs ~9.61% for 12mo),
and the absolute count of confirmed leavers actually drops (2,024 vs 6,321)
despite more total rows, because 3x fewer of the ever-observed moves fall
inside a 91-day window than a 365-day one.

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
with MLSPIN (MA) rows and adds macro/company columns. This project stays
RIAR-only by user decision, so `load_clean` filters `mls_code == "riar"`
immediately after reading. The RIAR subset of the new file is the same
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

SOURCE DATA (2026-08-10 update): switched to
`leave-dataset-with-team-distinction-mls-id.csv`. The ONLY change is the
agent identifier: the opaque per-row UUID `agent_profile_id` is replaced
by `mls_agent_id`, the agent's real ID in their own MLS. Every other
column is byte-identical. Verified before switching: same 696,775 rows,
same 82 shared columns row-for-row in the SAME order (checked positionally
on agent_name/mls_code/snapshot_date/office_mls_id/is_team/volume_12m/
units_12m/label_days_to_move/label_left_within_12m -- all identical), same
50,811 distinct agents, zero nulls, zero duplicate (mls_agent_id,
snapshot_date, mls_code) keys, and a strict 1:1 old<->new ID mapping (0
UUIDs splitting across IDs, 0 IDs collapsing multiple UUIDs). RIAR subset
unchanged at 84,417 rows / 5,625 agents / 155 team accounts. A full
UUID->MLS crosswalk is saved at `agent_id_crosswalk.csv` for joining
outputs produced before this switch. Because no feature value moved, this
is score-neutral by construction -- confirmed empirically, not assumed:
rescoring produced percent_chance identical to the pre-switch
current_agent_risk_scores.csv for all 4,607 agents (max abs diff 0.0).

`mls_agent_id` MUST be read as a string, never allowed to infer as int
(see the dtype= on both read_csv calls below). RIAR IDs are numeric but
three carry meaningful leading zeros ("00001", "0012", "0014"), and the
MLSPIN block that arrives with the MA expansion is alphanumeric outright
("A9500308", "CN235473", "CT005436"). Inferred dtype would corrupt the
first group and produce a mixed int/str column across the second.

ID SCOPE: mls_agent_id is guaranteed unique only WITHIN an MLS. It happens
to be globally unique across the current file (0 of 50,811 IDs appear
under more than one mls_code -- RIAR is 4-5 digits, MLSPIN 8 alphanumeric),
but that is a property of these two boards, not a guarantee. Any future
multi-MLS work should key on (mls_code, mls_agent_id), not mls_agent_id
alone. This project stays RIAR-only, so the bare ID is safe here today.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rebrand_classifications import CLUSTER_GATED_PAIRS, CONFIRMED_NOT_CHURN_PAIRS

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "leave-dataset-with-team-distinction-mls-id.csv"
)
MLS_CODE = "riar"

# The agent identifier column. Named here rather than inlined so the next
# identifier change is a one-line edit, and so the read_csv dtype override
# below can't drift apart from the column it protects.
AGENT_ID_COLUMN = "mls_agent_id"
HORIZON_DAYS = 91
HORIZON_MONTHS = 3

# Snapshots with too few rows to be a real cohort (data-edge artifact, e.g.
# the 10-row 2026-06-30 snapshot) rather than a real quarter of the panel.
MIN_SNAPSHOT_ROWS = 500

# 2026-07-28: a synthetic MLS placeholder record, not a real agent -- the
# "Non-Mls Member" bucket for buyer-side transactions where the cooperating
# agent isn't a board member (present in all 20 snapshots, is_team==0 so
# the team filter above doesn't catch it, 0 list-sides, static
# ever-incrementing tenure). Tested non-committally (cloned dataset+code,
# crossval before/after) before adopting: mean CatBoost AUC 0.7402 ->
# 0.7411 (+0.0009, small but a real improvement on the stated bar). See
# CHANGELOG.md for the full comparison. NB that +0.0009 is a single-seed,
# single-fold-structure measurement predating confirm.py (2026-07-26); it
# would not clear today's bar on its own. It has never been re-measured
# under the confirmation protocol -- the correctness argument is what
# carries it, as with "00001" below.
#
# 2026-08-10: re-pointed from the retired UUID
# "1a2504a4-2f87-42dd-ae86-09a6bc021f69" to MLS ID "12345" alongside the
# mls_agent_id switch. Verified the same 20 rows / same agent_name
# ("Non-Mls Member") are matched under the new ID, so the exclusion still
# bites -- had this constant been left on the UUID it would have matched
# nothing and silently reverted the 2026-07-28 change.
#
# 2026-08-10, second entry: "00001" ("Assisted Sale") added. The readable
# MLS IDs made three more synthetic records visible that UUIDs had hidden.
# Two need nothing: "0012" (MASS ALLIANCE PARTNER) is already caught by the
# is_team filter, and "0014" (ALLIANCE MEMBER) stops appearing after
# 2025-01-01. "00001" did: is_team==0, present in all 20 snapshots
# including the scored one, so it was being scored as though it were a real
# agent. Same synthetic signature as "12345" -- office_mls_id "ASST", 100%
# list sides, ~35-69 units/yr credited, tenure incrementing exactly 3.0
# months per quarter (211.1, 214.1, 217.1, ...) i.e. a counter, not a
# career, and it never once changes office across 19 usable rows.
#
# Measured before adopting via AutoResearch/gepa_loop/confirm_population.py
# (the confirm.py protocol adapted to a population change -- config fixed,
# panel varies), 3 seeds x 3 fold structures, 90 fits:
#   production 9/2: 0.741511 -> 0.742121 (+0.000610), worst-fold
#                   0.715368 -> 0.716269, 3/5 folds improved
#   shifted   7/2: 0.738277 -> 0.739596 (+0.001319)
#   shifted  11/2: 0.747655 -> 0.748003 (+0.000348)
# Verdict CONFIRMED: positive on all three structures under seed averaging.
#
# Read that honestly: +0.00061 is about 0.4 of a PAIRED standard error
# (SE(delta) = sigma*sqrt(2(1-rho)) ~= 0.0014 at rho=0.99; see
# Write_Up/AUTHORS_GUIDE.md 5.2b), and 2 of 5 folds got worse. What
# CONFIRMED buys is consistency of DIRECTION across nine independent
# measurements, not magnitude. NOTE (2026-08-14): this line previously read
# "~1/30th of the +/-0.019 CI" -- the wrong comparison, since +/-0.019 is
# the CI on the LEVEL of AUC and this is a paired difference on identical
# folds. The corrected figure is larger but the conclusion is unchanged:
# still well under one SE, still direction-only. The load-bearing argument
# for the exclusion is
# correctness -- "Assisted Sale" is not a person and has no churn risk to
# predict -- exactly as the magnet/trickle recodes were adopted at a
# measured AUC COST. The AUC here merely fails to argue against it.
#
# 2026-08-10, third entry: "0014" ("ALLIANCE MEMBER") added, on the user's
# call that the evidence already establishes what it is. It is a
# cross-board bucket, not an individual: office_mls_id "CTMLS", office_name
# "CONNECTICUT MLS", 21 units across 14 snapshots, 100% list-side, never
# changes office, and it stops appearing entirely after 2025-01-01. Note
# the geography -- Connecticut is NOT otherwise in this file, so this row
# is the RIAR feed's holding pen for CT-MLS cross-board activity, which is
# precisely why no individual agent stands behind it.
#
# Because it stops in 2025-01-01 it never reached the SCORED snapshot -- it
# was only ever contaminating TRAINING. That is the less visible of the two
# failure modes and the reason it survived unnoticed: nothing in the
# production output ever looked wrong.
#
# Measured the same way, chained onto the "00001" panel as the new baseline
# (confirm_population.py 0014, 3 seeds x 3 fold structures):
#   production 9/2: 0.742121 -> 0.743050 (+0.000929), worst-fold
#                   0.716269 -> 0.717216, 4/5 folds improved
#   shifted   7/2: 0.739596 -> 0.739696 (+0.000100)
#   shifted  11/2: 0.748003 -> 0.748894 (+0.000892)
# Verdict CONFIRMED. Same honesty caveat as above about magnitude -- but
# note this one improved 4/5 folds rather than 3/5, and its baseline arm
# reproduced the previous run's candidate arm to six decimals, so the two
# measurements chain cleanly rather than each drifting from its own
# reference.
#
# Cumulative effect of the two 2026-08-10 exclusions: panel 77,422 ->
# 77,389 usable rows (33 rows, all negatives; positives unchanged at
# 1,519), mean AUC 0.741511 -> 0.743050.
NON_MLS_MEMBER_AGENT_IDS = {"12345", "00001", "0014"}

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
]

OFFICE_BRAND_CATEGORIES = [
    "independent", "remax", "keller_williams", "century21", "coldwell_banker",
    "compass", "sothebys", "exp", "raveis", "era", "redfin", "laer",
    "corcoran", "elliman", "berkshire_hathaway",
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


def _first_move_in_window(candidates: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    """For each candidate row, the agent's first office move inside
    (snapshot_date, window_end] that departs from the office they were sitting
    in at the snapshot.

    Returns one row per candidate, in the SAME order/length as `candidates`,
    with dest_office/origin_office/move_quarter NaN where no such move exists
    (mirroring the left-merge the caller used to do, so the caller can keep
    treating a NaN destination as "no evidence, leave the label alone").
    """
    cand = candidates.reset_index(drop=True).reset_index(names="_cand_row")
    m = cand.merge(moves, on="mls_agent_id", how="left")

    in_window = (
        (m["move_quarter"] > m["snapshot_date"])
        & (m["move_quarter"] <= m["window_end"])
        & (m["origin_office"] == m["office_mls_id"])
    )
    m = m[in_window.fillna(False)]

    # Keep only the earliest qualifying move per candidate: that is the move
    # `label_days_to_move` is measuring, and the one whose destination decides
    # whether this label is real attrition.
    m = m.sort_values(["_cand_row", "move_quarter"]).drop_duplicates("_cand_row", keep="first")

    out = cand.merge(m[["_cand_row", "origin_office", "dest_office", "move_quarter"]], on="_cand_row", how="left")
    assert len(out) == len(candidates)
    return out


def _recode_mass_mover_moves(
    clean: pd.DataFrame,
    raw: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
    horizon_months: int = HORIZON_MONTHS,
) -> pd.DataFrame:
    """Zero out `label_col` for rows that aren't real individual attrition.

    Scope reviewed 2026-08-03 and left unchanged: narrowing this to
    detect_transitions.py's geometry alone (dropping the 30 magnet/trickle
    pairs) was proposed, applied, measured, and reversed. All 44 pairs
    recode as before. See the docstring of rebrand_classifications.py for
    the argument on both sides and the measured cost.

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

    Matches each `label_col`==1 row to the destination office its agent
    landed at, then applies whichever gate the pair falls under.

    Horizon generalization (2026-08-25): the match is against the agent's
    FIRST move inside the open-closed window (snapshot, snapshot +
    horizon_months], which is the move the label is reporting on
    (`label_days_to_move` is days to the next move after the snapshot).
    This previously pinned the destination to the single quarter
    `snapshot + horizon_months`. At the 3-month production horizon those
    are the same thing -- quarterly snapshots leave exactly one candidate
    quarter in the window -- so this changes no production label. At 6m/12m
    the old form silently matched only moves landing in the FINAL quarter of
    the window and missed every earlier one, which would have left most
    rebrand/M&A moves inside a longer window scored as real attrition.
    """
    moves = _build_moves(raw)
    cluster_sizes = moves.groupby(["origin_office", "dest_office", "move_quarter"]).size()

    candidates = clean[clean[label_col] == 1][["mls_agent_id", "office_mls_id", "snapshot_date"]].copy()
    candidates["window_end"] = candidates["snapshot_date"] + pd.DateOffset(months=horizon_months)
    matched = _first_move_in_window(candidates, moves)

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
    clean.loc[recode_index, label_col] = 0
    print(
        f"[data] recoded {len(recode_index)} rows from {label_col}=1 -> 0: confirmed rebrand/M&A/"
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


def load_clean(
    path: Path = DATA_PATH,
    exclude_team: bool = True,
    exclude_non_member: bool = True,
    horizon_months: int = HORIZON_MONTHS,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Load the dataset and return only rows with a resolvable outcome at
    `horizon_months`, labelled in a column named `label_left_{horizon_months}m`.

    horizon_months/horizon_days default to the production 3-month horizon and
    the defaults are the ONLY thing production ever uses -- `data.py`'s module
    docstring ("3 months, don't relitigate") is unchanged. They exist so the
    6m/12m horizon work runs through THIS function instead of a hand-copied
    twin. `horizon_generalization.build_clean_for_horizon` used to be that
    twin, and had drifted: it predated the 2026-07-25 team filter, the
    2026-07-28 non-MLS-member filter, the 2026-08-05 resolvability tightening
    and the mls_agent_id-as-str read, so it built a population production no
    longer trains on (documented as a known trap in HANDOFF.md 2026-08-13 and
    now fixed at the root rather than re-patched in the copy).

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

    label_col = f"label_left_{horizon_months}m"
    max_date_lower_bound = _infer_max_observed_date_lower_bound(df)
    snapshot_plus_horizon = df["snapshot_date"] + pd.DateOffset(months=horizon_months)

    # A row is resolvable ONLY when its own horizon is confirmed closed.
    # Until 2026-08-05 this also admitted any row with an observed move
    # (`move_observed | window_confirmed_elapsed`, where move_observed =
    # label_days_to_move.notna()) -- exactly the "we already know this one,
    # so keep it" special-casing the module docstring forbids. Past the
    # observable edge a mover's move is on record but a stayer's non-move is
    # not, so that branch can only ever admit POSITIVES.
    #
    # Measured before removal (riar, non-team): it admitted 12 rows, all
    # label_left_3m == 1, all in the 2026-07-01 snapshot -- which
    # MIN_SNAPSHOT_ROWS below then dropped anyway. So this changes no fold
    # and no metric (77,453 -> 77,441 rows, positive rate 2.536% ->
    # 2.521%). It was harmless only by accident: max_date_lower_bound
    # currently lands exactly on the max snapshot date, so every earlier
    # snapshot clears the window test on its own and the OR-branch was
    # load-bearing at the final snapshot only. A future drop whose
    # label_censored==0 tail runs several quarters behind the last snapshot
    # would have it admit positives-only rows across multiple recent
    # quarters instead.
    window_confirmed_elapsed = (df["label_censored"] == 0) | (snapshot_plus_horizon <= max_date_lower_bound)
    resolvable = window_confirmed_elapsed

    df[label_col] = (df["label_days_to_move"] <= horizon_days).astype(int)

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

    clean = _recode_mass_mover_moves(clean, df, label_col=label_col, horizon_months=horizon_months)

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
        f"[data] loaded {n_before} rows -> {n_after} usable for the {horizon_months}-month "
        f"horizon ({n_before - n_after} dropped: unresolvable or artifact rows). "
        f"{horizon_months}-month positive rate: {clean[label_col].mean():.4%}"
    )
    return clean


if __name__ == "__main__":
    df = load_clean()
    print(df.groupby(df["snapshot_date"].dt.date)["label_left_3m"].agg(["count", "mean"]))
