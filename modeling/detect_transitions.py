"""Detect candidate rebrand/M&A events directly from agent movement patterns.

No web search needed to FIND candidates -- only to CLASSIFY them afterward.
A formal brand conversion or acquisition looks like a specific fingerprint
in the data: many agents from the same origin office landing at the SAME
destination office in the SAME quarter. An organic mass exodus (e.g. bad
management driving a wave of individual departures) looks different: agents
scatter to MANY different destinations even if concentrated in time. This
script reconstructs actual origin->destination moves from the raw panel
(tracking each agent's office_mls_id across consecutive observed snapshots,
not relying on the label/censoring logic used elsewhere) and ranks
candidates by concentration, so only a short list needs manual web
verification.
"""

import pandas as pd

from data import DATA_PATH, MIN_CLUSTER_MOVERS, MLS_CODE, _parse_snapshot_date

# Shared with data.py's _recode_mass_mover_moves so the threshold used to
# FIND candidates and the threshold used to actually recode labels can't
# drift apart.
MIN_MOVERS = MIN_CLUSTER_MOVERS


def build_moves(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["agent_profile_id", "mls_code", "office_mls_id", "office_name", "snapshot_date"])
    df = df[df["mls_code"] == MLS_CODE].copy()
    df["snapshot_date"] = _parse_snapshot_date(df["snapshot_date"])
    df = df.sort_values(["agent_profile_id", "snapshot_date"])

    df["prev_office"] = df.groupby("agent_profile_id")["office_mls_id"].shift(1)
    df["prev_office_name"] = df.groupby("agent_profile_id")["office_name"].shift(1)
    df["prev_date"] = df.groupby("agent_profile_id")["snapshot_date"].shift(1)

    moved = df[df["office_mls_id"] != df["prev_office"]].dropna(subset=["prev_office"]).copy()
    moved = moved.rename(
        columns={
            "office_mls_id": "dest_office",
            "office_name": "dest_office_name",
            "prev_office": "origin_office",
            "prev_office_name": "origin_office_name",
            "snapshot_date": "move_quarter",
        }
    )
    return moved[
        ["agent_profile_id", "origin_office", "origin_office_name", "dest_office", "dest_office_name", "move_quarter"]
    ]


def rank_candidates(moves: pd.DataFrame, roster_by_office_quarter: pd.Series) -> pd.DataFrame:
    grouped = (
        moves.groupby(["origin_office", "origin_office_name", "dest_office", "dest_office_name", "move_quarter"])
        .size()
        .reset_index(name="n_movers")
    )
    grouped = grouped[grouped["n_movers"] >= MIN_MOVERS]

    # roster size of the ORIGIN office as of the quarter before the move,
    # to normalize "8 movers" by whether that's 8-of-10 or 8-of-400
    prior_quarter = grouped["move_quarter"] - pd.DateOffset(months=3)
    grouped["origin_prior_roster"] = [
        roster_by_office_quarter.get((off, q), pd.NA)
        for off, q in zip(grouped["origin_office"], prior_quarter)
    ]
    grouped["pct_of_origin_roster"] = grouped["n_movers"] / grouped["origin_prior_roster"]

    # scatter check: of all movers OUT of this origin office in this quarter
    # (to ANY destination), what fraction went to this specific destination?
    movers_out_total = moves.groupby(["origin_office", "move_quarter"]).size()
    grouped["total_movers_out_of_origin_this_q"] = [
        movers_out_total.get((off, q), 0) for off, q in zip(grouped["origin_office"], grouped["move_quarter"])
    ]
    grouped["concentration_to_this_dest"] = grouped["n_movers"] / grouped["total_movers_out_of_origin_this_q"]

    return grouped.sort_values("n_movers", ascending=False)


def main():
    moves = build_moves()

    raw_dates = pd.read_csv(DATA_PATH, usecols=["mls_code", "office_mls_id", "snapshot_date"])
    raw_dates = raw_dates[raw_dates["mls_code"] == MLS_CODE].copy()
    raw_dates["snapshot_date"] = _parse_snapshot_date(raw_dates["snapshot_date"])
    roster_by_office_quarter = raw_dates.groupby(["office_mls_id", "snapshot_date"]).size()

    candidates = rank_candidates(moves, roster_by_office_quarter)

    print(f"[detect] {len(moves)} total agent office-changes observed across the panel")
    print(f"[detect] {len(candidates)} candidate transitions with >= {MIN_MOVERS} simultaneous movers "
          f"from the same origin to the same destination in the same quarter\n")

    cols = [
        "origin_office", "origin_office_name", "dest_office", "dest_office_name",
        "move_quarter", "n_movers", "origin_prior_roster", "pct_of_origin_roster",
        "concentration_to_this_dest",
    ]
    with pd.option_context("display.max_colwidth", 28, "display.width", 200):
        print(candidates[cols].to_string(index=False))


if __name__ == "__main__":
    main()
