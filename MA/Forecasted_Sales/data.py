"""
Forecasted Sales Algorithm (MASSACHUSETTS / MLSPIN) -- shared data loader.

PROVENANCE (2026-07-28): ported from `RI/Forecasted_Sales/data.py`
(RI) by changing the mls_code filter from "riar" to "mlspin" and swapping the
synthetic-placeholder agent id for MLSPIN's equivalent. No modeling logic
changed: the hurdle + conformalized-quantile pipeline, the target self-join, the
team exclusion and the feature set are all identical, so RI and MA results are
directly comparable methodology-wise.

Note that every market_* and state_* column in the source is keyed to the MLS,
not to the agent's home address -- MLSPIN rows carry Massachusetts market and
macro series regardless of where an individual agent happens to live.

Target construction: next-12m dollar volume and unit count, per agent,
MLSPIN-only. Built by joining each agent-quarter row to that SAME agent's row
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

MLS_CODE = "mlspin"

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

# MLSPIN's own synthetic placeholder record -- the MA analog of the RI
# project's "Non-Mls Member" bucket, holding buyer-side transactions where the
# cooperating agent isn't a board member. Different mls_agent_id from the
# RI one (the RI id does not appear in MLSPIN at all, so inheriting it would
# have excluded nothing); identified by the same structural signature: all 20
# snapshots, is_team==0, ZERO list-sides ever, 1,678 units all buy-side in the
# latest quarter, monotonically incrementing career_months, 1.16% of market
# volume. Excluded for the same reason as in RI -- it is not a person, and its
# ~$1.1B of trailing volume would otherwise sit in the training distribution as
# if it were one extraordinarily productive agent.
#
# 2026-08-10: re-pointed from the retired UUID
# "0d2b79f7-a100-45a2-9760-4c0ad3bb3849" to MLS ID "H1111111" alongside the
# mls_agent_id switch (agent_name "Non Member", office "Non Member Office",
# 23,127 units all buy-side). Left on the UUID it would have matched nothing
# and silently un-excluded the placeholder. RI's "00001" is a riar id and
# cannot appear under MLS_CODE "mlspin" -- see MA/Likelihood_to_Leave/data.py
# for the MLSPIN placeholder scan that found no other candidate.
NON_MLS_MEMBER_AGENT_IDS = {"H1111111"}


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
    df = df[df["mls_code"] == MLS_CODE].copy()
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
