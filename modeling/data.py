"""Shared data loading for the likelihood-to-leave models.

HORIZON: 3 months. This replaced the original 12-month framing entirely
(2026-07 decision) because a 3-month window closes so much faster that far
more of the panel becomes usable (94% of rows vs. 78%), reaches four extra,
more-recent quarters, and better matches "who's a live flight risk right
now" than a 12-month outlook does. Trade-off, kept deliberately visible
rather than hidden: the 3-month positive rate is only ~2.55% (vs ~9.6% for
12mo), and the absolute count of confirmed leavers actually drops (~2,000 vs
~6,300) despite more total rows, because 3x fewer of the ever-observed moves
fall inside a 91-day window than a 365-day one.

Label reconstruction: the source CSV only ships label_left_within_12m /
label_days_to_move / label_censored, computed for a 12-month horizon. A
3-month label is reconstructed from label_days_to_move here rather than
shipped directly. This is sound because the source data captures a move as
soon as it's observed regardless of whether the 12-month window has fully
closed (confirmed empirically: 841 rows have label_censored==1 AND
label_left_within_12m==1 -- i.e. a move was recorded even in a
not-yet-elapsed window). So label_days_to_move <= 91 is a reliable 3-month
positive wherever it's known, and the only rows that are genuinely
ambiguous for a 3-month question are ones where (a) the 12-month window
never closed AND (b) no move was ever captured AND (c) we can't otherwise
confirm 3 months of runway had elapsed as of the data's true max observed
date (which isn't shipped directly, so a conservative lower-bound estimate
is used -- see _infer_max_observed_date_lower_bound).

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
"""

from pathlib import Path

import pandas as pd

from rebrand_classifications import CLUSTER_GATED_PAIRS, CONFIRMED_NOT_CHURN_PAIRS

DATA_PATH = Path(__file__).resolve().parent.parent / "leave-dataset_most_recent_updated.csv"
MLS_CODE = "riar"
HORIZON_DAYS = 91
HORIZON_MONTHS = 3

# Snapshots with too few rows to be a real cohort (data-edge artifact, e.g.
# the 10-row 2026-06-30 snapshot) rather than a real quarter of the panel.
MIN_SNAPSHOT_ROWS = 500

# Minimum simultaneous same-origin-to-same-destination movers, in the SAME
# quarter, to treat a transition as a corporate/structural event rather than
# individual attrition (see _recode_mass_mover_moves). Shared with
# detect_transitions.py so the threshold used to FIND candidates and the
# threshold used to actually recode labels never drift apart.
MIN_CLUSTER_MOVERS = 5

ID_COLUMNS = [
    "agent_profile_id",
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
    # avg_dom_12m (per-agent) intentionally excluded: 59% zeros, median 0,
    # max 378 -- not a plausible days-on-market distribution for active
    # agents. Confirmed broken via diagnose_fold5.py's drift check (flagged
    # as the top-importance feature with the largest train/test shift, which
    # turned out to be this column's noise, not a real signal). Use
    # market_avg_dom_12m (state-level, sanity-checked as legitimate: 52-70
    # day range) for days-on-market / market-heat signal instead.
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

    df["office_arrival_rate_12m"] = _safe_ratio(df["office_arrivals_12m"], df["office_active_agents_asof"])

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
    df["company_arrival_rate_12m"] = _safe_ratio(df["company_arrivals_12m"], df["company_active_agents_asof"])
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
    m = raw[["agent_profile_id", "office_mls_id", "snapshot_date"]].sort_values(
        ["agent_profile_id", "snapshot_date"]
    )
    m["prev_office"] = m.groupby("agent_profile_id")["office_mls_id"].shift(1)
    moved = m[m["office_mls_id"] != m["prev_office"]].dropna(subset=["prev_office"])
    return moved.rename(columns={"office_mls_id": "dest_office", "prev_office": "origin_office", "snapshot_date": "move_quarter"})[
        ["agent_profile_id", "origin_office", "dest_office", "move_quarter"]
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

    candidates = clean[clean["label_left_3m"] == 1][["agent_profile_id", "office_mls_id", "snapshot_date"]].copy()
    candidates["move_quarter"] = candidates["snapshot_date"] + pd.DateOffset(months=HORIZON_MONTHS)
    matched = candidates.merge(
        moves,
        left_on=["agent_profile_id", "office_mls_id", "move_quarter"],
        right_on=["agent_profile_id", "origin_office", "move_quarter"],
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
    return df


def load_current_snapshot(path: Path = DATA_PATH) -> pd.DataFrame:
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
    """
    df = pd.read_csv(path, low_memory=False)
    if "mls_code" in df.columns:
        df = df[df["mls_code"] == MLS_CODE].copy()
    df["snapshot_date"] = _parse_snapshot_date(df["snapshot_date"])

    latest_date = df["snapshot_date"].max()
    current = df[df["snapshot_date"] == latest_date].copy()
    current = _engineer_features(current)
    print(f"[data] current snapshot: {latest_date.date()}, {len(current)} active agents")
    return current


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and return only rows with a resolvable 3-month outcome."""
    df = pd.read_csv(path, low_memory=False)
    if "mls_code" in df.columns:
        df = df[df["mls_code"] == MLS_CODE].copy()
    df["snapshot_date"] = _parse_snapshot_date(df["snapshot_date"])
    n_before = len(df)

    max_date_lower_bound = _infer_max_observed_date_lower_bound(df)
    snapshot_plus_horizon = df["snapshot_date"] + pd.DateOffset(months=HORIZON_MONTHS)

    move_observed = df["label_days_to_move"].notna()
    window_confirmed_elapsed = (df["label_censored"] == 0) | (snapshot_plus_horizon <= max_date_lower_bound)
    resolvable = move_observed | window_confirmed_elapsed

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

    clean = clean.drop(columns=["label_censored", "label_left_within_12m", "label_days_to_move"])
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
