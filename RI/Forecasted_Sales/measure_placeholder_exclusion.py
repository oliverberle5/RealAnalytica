"""What the two synthetic-placeholder exclusions actually do to this panel.

Answers the question HANDOFF.md's item 1 leaves OPEN: "12345" (Non-Mls Member)
and "00001" (Broker Assisted Sale) are excluded in `data.py::NON_MLS_MEMBER_AGENT_IDS`,
but the footprint of that exclusion was never measured on THIS project's data --
only on the Leave model's AUC, which does not transfer to a sales-forecast MAE.

Reports three things, in increasing order of how much they cost to compute:
  1. PANEL footprint  -- rows, snapshots, units and dollar volume the two records
     carry, and what share of the RIAR panel that is. Cheap, exact, no model.
  2. TRAINING footprint -- how many resolved (t+12mo) training rows they supply,
     which is the only channel through which they can affect a forecast at all,
     since neither record survives into the scored snapshot as a real agent.
  3. TARGET distortion -- what the two records would contribute as forecast
     targets if left in: their own next-12m volume/units, against the roster
     median. This is the "why it is wrong on correctness grounds" number.

Deliberately does NOT retrain anything -- the A/B on forecasts is a separate
script (compare_exclusion_forecasts.py) because a refit costs minutes and this
costs seconds.
"""
import pandas as pd

import data
from data import DATA_PATH, NON_MLS_MEMBER_AGENT_IDS, AGENT_ID_COLUMN

pd.set_option("display.width", 160)


def main():
    print(f"[measure] reading {DATA_PATH.name} ...")
    raw = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"], dtype={AGENT_ID_COLUMN: str})
    riar = raw[raw["mls_code"] == "riar"].copy()
    print(f"[measure] RIAR panel: {len(riar):,} rows, {riar[AGENT_ID_COLUMN].nunique():,} agents, "
          f"{riar['snapshot_date'].nunique()} snapshots\n")

    ph = riar[riar[AGENT_ID_COLUMN].isin(NON_MLS_MEMBER_AGENT_IDS)]
    print("=" * 96)
    print("1. PANEL FOOTPRINT")
    print("=" * 96)
    for aid in sorted(NON_MLS_MEMBER_AGENT_IDS):
        r = riar[riar[AGENT_ID_COLUMN] == aid]
        if r.empty:
            print(f"  {aid}: NOT PRESENT in panel")
            continue
        print(f"  {aid}  office_mls_id={r['office_mls_id'].iloc[0]!r}  "
              f"office_name={r['office_name'].iloc[0]!r}  is_team={sorted(r['is_team'].unique())}")
        print(f"      rows={len(r)}  snapshots={r['snapshot_date'].min().date()}..{r['snapshot_date'].max().date()}"
              f"  units_12m: min={r['units_12m'].min():.0f} max={r['units_12m'].max():.0f}"
              f"  volume_12m max=${r['volume_12m'].max():,.0f}")

    latest = riar["snapshot_date"].max()
    cur = riar[riar["snapshot_date"] == latest]
    cur_ph = cur[cur[AGENT_ID_COLUMN].isin(NON_MLS_MEMBER_AGENT_IDS)]
    print(f"\n  Latest snapshot ({latest.date()}), trailing-12m totals across the panel:")
    print(f"      volume: placeholders ${cur_ph['volume_12m'].sum():,.0f} of ${cur['volume_12m'].sum():,.0f} "
          f"= {cur_ph['volume_12m'].sum() / cur['volume_12m'].sum():.2%}")
    print(f"      units : placeholders {cur_ph['units_12m'].sum():,.0f} of {cur['units_12m'].sum():,.0f} "
          f"= {cur_ph['units_12m'].sum() / cur['units_12m'].sum():.2%}")
    print(f"      rows  : {len(cur_ph)} of {len(cur):,} = {len(cur_ph) / len(cur):.3%}")

    print("\n" + "=" * 96)
    print("2. TRAINING FOOTPRINT (resolved t+12mo rows -- the only channel to a forecast)")
    print("=" * 96)
    for exclude in (False, True):
        df = data.load_clean(exclude_non_member=exclude)
        resolved = data.build_targets(df)
        tag = "WITH exclusion" if exclude else "WITHOUT exclusion"
        print(f"  {tag:<18} clean rows={len(df):,}  resolved training rows={len(resolved):,}  "
              f"agents={df[AGENT_ID_COLUMN].nunique():,}")
        if not exclude:
            keep = resolved[resolved[AGENT_ID_COLUMN].isin(NON_MLS_MEMBER_AGENT_IDS)]
            print(f"  {'':18} of which placeholder rows: {len(keep)} "
                  f"({len(keep) / len(resolved):.4%} of training)")

    print("\n" + "=" * 96)
    print("3. TARGET DISTORTION (what the model would be asked to learn from them)")
    print("=" * 96)
    df_no = data.load_clean(exclude_non_member=False)
    res_no = data.build_targets(df_no)
    ph_res = res_no[res_no[AGENT_ID_COLUMN].isin(NON_MLS_MEMBER_AGENT_IDS)]
    real = res_no[~res_no[AGENT_ID_COLUMN].isin(NON_MLS_MEMBER_AGENT_IDS)]
    for col in ("target_volume_next_12m", "target_units_next_12m"):
        print(f"  {col}")
        print(f"      placeholders : n={len(ph_res)}  median={ph_res[col].median():,.0f}  max={ph_res[col].max():,.0f}")
        print(f"      real agents  : n={len(real):,}  median={real[col].median():,.0f}  "
              f"p99={real[col].quantile(0.99):,.0f}  max={real[col].max():,.0f}")
        if len(ph_res):
            pctile = (real[col] < ph_res[col].median()).mean()
            print(f"      -> a placeholder's median target sits at the {pctile:.2%} percentile of real agents")


if __name__ == "__main__":
    main()
