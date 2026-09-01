"""
Forecasted Sales Algorithm -- shared data loader.

Target construction: next-12m dollar volume and unit count, per agent,
RIAR-only. Built by joining each agent-quarter row to that SAME agent's row
exactly 12 months later -- volume_12m/units_12m at t+12mo already IS the
next-12m outcome as of time t, since those columns are trailing-12m sums
computed as of their own snapshot date. No new aggregation needed, just a
self-join on (mls_agent_id, snapshot_date + 12mo).

Rows with no resolvable t+12mo row are dropped (conservative censoring).
We deliberately do not distinguish "agent genuinely exited real estate"
from "agent moved to a brokerage outside the RIAR/MLSPIN-RIAR footprint
we can see" -- there is no field that tells them apart, and guessing
either way would corrupt the target for the ambiguous ~13-14% of agents
who disappear from the panel. See HANDOFF.md for the empirical breakdown.

Confirmed empirically (see HANDOFF.md): volume_12m/units_12m are agent-wide,
NOT reset at office switches -- corr(pre-switch, post-switch trailing
volume_12m) = 0.98 across 2,140 switch-quarters. So no special handling is
needed at office-switch quarters for target construction -- an agent's
target is their total production wherever they end up, which is what the
raw columns already give us.

SOURCE DATA (2026-07-28 update): switched to
`leave-dataset-with-team-distinction.csv` -- verified beforehand: identical
84,417 riar rows and identical volume_12m sum to the old `leave-dataset
(1).csv`, plus two new columns (`agent_name`, `is_team`). Team accounts
(155 distinct mls_agent_ids, 2,368 rows) are now excluded by default
from both training and scoring -- see `load_clean`'s `exclude_team`
docstring for the mechanism and the before/after backtest numbers, and
HANDOFF.md for the full writeup.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# All four models (RI/MA x leave/sales) read this ONE file and differ only
# by their mls_code filter. It lives at <repo_root>/data/ and is NOT in
# git -- see README.md 'Getting the data'.
DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "leave-dataset-with-team-distinction-mls-id.csv"
)

# The agent identifier column. Named here rather than inlined so the next
# identifier change is a one-line edit, and so the read_csv dtype override
# cannot drift apart from the column it protects. MUST stay str: three RIAR
# ids carry meaningful leading zeros ("00001", "0012", "0014") and every
# MLSPIN id is alphanumeric ("A9500308", "CN235473"), so dtype inference
# corrupts the first group and yields a mixed int/str column across the
# second -- which would make the exclusion sets below silently stop matching.
AGENT_ID_COLUMN = "mls_agent_id"


# Leave-model-specific labels -- not relevant to a sales-volume/units target.
LEAVE_MODEL_LABEL_COLUMNS = [
    "label_left_within_12m",
    "label_days_to_move",
    "label_censored",
]

# Confirmed broken/discarded per the Leave model's HANDOFF.md -- same data,
# same defects, no reason to re-litigate here.
DROPPED_COLUMNS = [
    "heuristic_recruitability_score",  # user: "bullshit, delete it entirely"
    "avg_dom_12m",  # 59% zeros, implausible distribution; use market_avg_dom_12m instead
]

ID_COLUMNS = [
    "mls_agent_id", "mls_code", "snapshot_date", "office_mls_id",
    "office_name", "agent_city", "agent_state", "company_name",
]

CATEGORICAL_FEATURE_COLUMNS = ["office_brand"]

TARGET_COLUMNS = ["target_volume_next_12m", "target_units_next_12m"]

# The two source columns the targets are pulled from -- kept as features too,
# since "current trailing-12m production" is a legitimate predictor of
# next-12m production (it's the agent's own current pace, not a future leak).
SOURCE_TARGET_COLUMNS = ["volume_12m", "units_12m"]

# 2026-07-28: a synthetic MLS placeholder record, not a real agent -- the
# "Non-Mls Member" bucket for buyer-side transactions where the cooperating
# agent isn't a board member (present in all 20 snapshots, is_team==0 so
# the team filter above doesn't catch it, 0 list-sides, static
# ever-incrementing tenure, ~1.2% of market volume in the latest quarter).
# Tested non-committally (cloned dataset+code, held-out backtest) before
# adopting -- small, consistent gains: volume MAE -0.2%, width -2.1% both
# bands, coverage unchanged; units roughly flat. See HANDOFF.md.
#
# 2026-08-10: re-pointed from the retired UUID
# "1a2504a4-2f87-42dd-ae86-09a6bc021f69" to MLS ID "12345" alongside the
# mls_agent_id switch; verified it still matches the same "Non-Mls Member"
# record (office_mls_id "NMLS", 7,004 units, 98.8% buy-side). Left on the
# UUID it would have matched nothing and silently reverted the 2026-07-28
# adoption above, with no error and no log line.
#
# "00001" ("Assisted Sale") added in the same pass -- office_mls_id "ASST",
# office_name "Broker Assisted Sale", 1,062 units 99.7% list-side, tenure
# incrementing exactly 3.0 months per quarter, never changes office. Same
# class of synthetic bucket as "12345", surfaced only when the agent key
# became readable.
#
# EVIDENCE, and the asymmetry that used to sit here: in the
# Likelihood_to_Leave project this exclusion was measured before adoption
# (confirm_population.py, 3 seeds x 3 fold structures: +0.000610 mean AUC,
# CONFIRMED), but for a while it had NOT been measured against THIS model,
# unlike the two adoptions documented above. Measured 2026-08-11 via
# compare_exclusion_forecasts.py (4 production passes: exclusion ON/OFF at
# seed 0, plus ON at seeds 1-2 for the noise band). Result: per-agent volume
# forecasts move a median of 1.47% (p95 5.12%, book +0.52%) -- SMALLER than
# the 1.79-2.02% median that re-rolling the seed alone produces, so the
# exclusion is immaterial to aggregate accuracy and no MAE win should be
# claimed for it. The honest expectation in the line that used to be here
# ("immaterial to MAE either way") was right.
#
# It stands on the correctness argument, which is sufficient: a
# broker-assisted-sale bucket is not a person and has no next-12m production
# to forecast. Concretely, because the shipped spreadsheet had been re-keyed
# rather than refit, "00001" was still being forecast in the delivered file
# at $51,688,219 -- rank 11 of 4,607, 46x the roster median. Its median
# training target ($94.9M / 176 units) sits at the 99.94th percentile of real
# agents and its max exceeds the largest real agent in the panel. None of
# that is visible in an MAE.
#
# "0014" ("ALLIANCE MEMBER") added 2026-08-22, closing the panel divergence
# that HANDOFF.md's STOP box item 2 had left open since 2026-08-10. Same
# class of record as the two above: office_mls_id "CTMLS", office_name
# "CONNECTICUT MLS", 21 units across 14 snapshots, 100% list-side,
# is_team == 0, never changes office. Connecticut is not otherwise in this
# file, so the row is the RIAR feed's holding pen for CT-MLS cross-board
# activity -- no individual agent stands behind it. Verified directly in
# this panel before adopting (14 riar rows, 2021-10-01..2025-01-01).
#
# It stops appearing after 2025-01-01, so it was never in the scored
# snapshot: this exclusion removes training rows only and changes no
# output row's existence -- exactly the failure mode where nothing in the
# production spreadsheet ever looks wrong. Measured in the Leave project
# (confirm_population.py, 3 seeds x 3 fold structures: +0.000929 mean AUC,
# 4/5 folds improved, CONFIRMED) and, per this project's rule that a
# leave-model AUC does not transfer to a sales forecast, re-measured HERE
# on the production hurdle+CQR pipeline before adoption
# (remeasure_accuracy.py, held-out 2025-04-01..2025-07-01) -- see
# HANDOFF.md. As with "00001", the effect on aggregate accuracy is inside
# this pipeline's own seed-to-seed noise band; the justification is
# correctness, and the two projects now train on identical row sets.
NON_MLS_MEMBER_AGENT_IDS = {"12345", "00001", "0014"}


def load_clean(path: Path = DATA_PATH, exclude_team: bool = True, exclude_non_member: bool = True) -> pd.DataFrame:
    """exclude_team: mirrors the Likelihood_to_Leave_Algorithm project's
    is_team filter (validated there first per HANDOFF.md's "roster arrives
    in the Leave project first" ordering). Team accounts carry a whole
    team's production credit on one mls_agent_id -- keeping them in the
    individual forecast trains/scores the model on production levels no
    single person actually produces, and the silent teammates whose rows
    sit near-$0 add fake low-volatility signal that distorts the quantile-
    range calibration. Excluded, not scored, until a dedicated team-level
    forecast exists (not yet built -- same gap as the Leave project).

    2026-07-28: tested non-committally (cloned dataset + code, held-out
    backtest of the actual production hurdle+CQR pipeline) before adopting
    -- see HANDOFF.md. Result: MAE -9.1% (volume) / -8.3% (units), 80%/95%
    band width -7 to -10%, coverage unchanged at target. A real accuracy
    and precision win, not just a correctness fix.
    """
    df = pd.read_csv(path, parse_dates=["snapshot_date"], dtype={AGENT_ID_COLUMN: str})
    df = df[df["mls_code"] == "riar"].copy()
    if exclude_team:
        df = df[df["is_team"] == 0].copy()
    if exclude_non_member:
        n_before_nm = len(df)
        df = df[~df["mls_agent_id"].isin(NON_MLS_MEMBER_AGENT_IDS)].copy()
        print(f"[data] excluded {n_before_nm - len(df)} non-MLS-member placeholder rows")
    df = df.drop(columns=LEAVE_MODEL_LABEL_COLUMNS + DROPPED_COLUMNS + ["is_team", "agent_name"])
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its own t+12mo row to create next-12m targets.

    Drops any row where the agent has no observed snapshot exactly 12
    months later (censored -- see module docstring).
    """
    df = df.sort_values(["mls_agent_id", "snapshot_date"]).reset_index(drop=True)

    future = df[["mls_agent_id", "snapshot_date", "volume_12m", "units_12m"]].copy()
    future = future.rename(columns={
        "volume_12m": "target_volume_next_12m",
        "units_12m": "target_units_next_12m",
    })
    # Shift the future table's date back by 12mo so a merge on snapshot_date==t
    # pulls in the value actually recorded at t+12mo.
    future["snapshot_date"] = future["snapshot_date"] - pd.DateOffset(months=12)

    merged = df.merge(future, on=["mls_agent_id", "snapshot_date"], how="left")
    resolved = merged.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    return resolved


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(ID_COLUMNS) | set(TARGET_COLUMNS) | set(CATEGORICAL_FEATURE_COLUMNS)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def build_model_frame(path: Path = DATA_PATH, exclude_team: bool = True, exclude_non_member: bool = True) -> pd.DataFrame:
    """Full pipeline: load, filter, build targets, one-hot the categorical."""
    df = load_clean(path, exclude_team=exclude_team, exclude_non_member=exclude_non_member)
    df = build_targets(df)
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURE_COLUMNS, prefix="brand")
    return df


if __name__ == "__main__":
    frame = build_model_frame()
    print("resolved (agent, quarter) rows with a clean next-12m target:", len(frame))
    print("date range of feature snapshots:", frame["snapshot_date"].min(), "to", frame["snapshot_date"].max())
    print("unique agents represented:", frame["mls_agent_id"].nunique())
    feats = get_feature_columns(frame)
    print("numeric feature count:", len(feats))
