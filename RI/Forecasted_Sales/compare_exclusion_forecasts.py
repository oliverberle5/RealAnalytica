"""A/B the two synthetic-placeholder exclusions on THIS model's own output,
and put the result next to this pipeline's run-to-run noise band.

HANDOFF.md item 1 (OPEN): "12345"/"00001" were adopted into
`data.py::NON_MLS_MEMBER_AGENT_IDS` on the correctness argument plus the LEAVE
model's AUC, never on this project's own pipeline. Item 3 (NEW) says why a bare
A/B would not have settled it anyway: this pipeline is not invariant to training-
row order, so ~1.5% median per-agent movement shows up between runs that differ
by nothing meaningful at all. A delta reported against zero is therefore
uninterpretable here; it has to be reported against that band.

So this script runs the production scoring path four times:

  A  exclusion ON,  seed 0   <- production config, the shipped spreadsheet
  B  exclusion OFF, seed 0   <- the only difference from A is the two records
  C  exclusion ON,  seed 1   |  same config as A, different bootstrap draw:
  D  exclusion ON,  seed 2   |  this pair IS the noise band

A-vs-B is the exclusion effect. A-vs-C and A-vs-D are pure noise, measured on
the identical statistics. If the first is not clearly larger than the second,
the honest conclusion is "immaterial at this pipeline's resolution" -- which is
a result, not a failure, and is what the correctness argument for excluding a
non-person predicted.

Seed variation rather than key re-ordering: HANDOFF item 3 traces the
instability to both learners subsampling by row order (CatBoost MVS bootstrap,
XGBoost subsample=0.8/colsample=0.8). Varying the seed perturbs the same draw
directly, with the row order held fixed, so it isolates that variance source
without needing a second agent-key column to sort by.
"""
import numpy as np
import pandas as pd

import data
import hurdle_model
import score_agents

pd.set_option("display.width", 200)

AGENT_ID = data.AGENT_ID_COLUMN
KEY_COLS = ["volume_point_estimate", "units_point_estimate", "volume_pct_zero", "units_pct_zero"]


def run_once(exclude_non_member: bool, seed: int) -> pd.DataFrame:
    """One full production scoring pass under a given population + seed.

    Patches module attributes rather than editing score_agents.py: the
    production path must stay exactly what ships, and both knobs here
    (which rows load, which bootstrap draw) are things the shipped code
    deliberately does not expose."""
    orig_load = score_agents.load_clean
    orig_cb = dict(score_agents.MODEL_PARAMS)
    orig_xgb = dict(hurdle_model.ZERO_CLASSIFIER_PARAMS)
    try:
        score_agents.load_clean = lambda: orig_load(exclude_non_member=exclude_non_member)
        score_agents.MODEL_PARAMS.update(random_seed=seed)
        hurdle_model.ZERO_CLASSIFIER_PARAMS.update(random_state=seed)
        out = score_agents.score_current_agents()
    finally:
        score_agents.load_clean = orig_load
        score_agents.MODEL_PARAMS.clear(); score_agents.MODEL_PARAMS.update(orig_cb)
        hurdle_model.ZERO_CLASSIFIER_PARAMS.clear(); hurdle_model.ZERO_CLASSIFIER_PARAMS.update(orig_xgb)
    return out.set_index(AGENT_ID)


def compare(base: pd.DataFrame, other: pd.DataFrame, label: str) -> dict:
    """Per-agent movement on the intersection of the two scored rosters.

    Intersection, not union: run B additionally scores the two placeholder
    records themselves, and 'the forecast for a record that should not be
    forecast' is not a difference in anyone's forecast -- it is counted
    separately, as rows that appear at all."""
    common = base.index.intersection(other.index)
    row = {"comparison": label, "agents_compared": len(common),
           "rows_only_in_other": len(other.index.difference(base.index))}
    b, o = base.loc[common], other.loc[common]

    v_b, v_o = b["volume_point_estimate"].astype(float), o["volume_point_estimate"].astype(float)
    changed = (v_b != v_o)
    denom = v_b.replace(0, np.nan)
    pct = ((v_o - v_b).abs() / denom).dropna()
    row["pct_agents_moved"] = changed.mean()
    row["median_abs_pct"] = pct.median()
    row["p95_abs_pct"] = pct.quantile(0.95)
    row["max_abs_dollars"] = (v_o - v_b).abs().max()
    row["book_total_base"] = v_b.sum()
    row["book_total_other"] = v_o.sum()
    row["book_total_pct"] = (v_o.sum() - v_b.sum()) / v_b.sum()

    u_b, u_o = b["units_point_estimate"].astype(float), o["units_point_estimate"].astype(float)
    u_pct = ((u_o - u_b).abs() / u_b.replace(0, np.nan)).dropna()
    row["units_median_abs_pct"] = u_pct.median()
    row["units_book_pct"] = (u_o.sum() - u_b.sum()) / u_b.sum()

    row["pzero_median_abs_delta"] = (o["volume_pct_zero"] - b["volume_pct_zero"]).abs().median()
    return row


def main():
    runs = {}
    for label, (excl, seed) in {
        "A: exclusion ON,  seed 0 (production)": (True, 0),
        "B: exclusion OFF, seed 0": (False, 0),
        "C: exclusion ON,  seed 1": (True, 1),
        "D: exclusion ON,  seed 2": (True, 2),
    }.items():
        print("\n" + "=" * 96)
        print(f"RUN {label}")
        print("=" * 96)
        runs[label] = run_once(excl, seed)

    keys = list(runs)
    base = runs[keys[0]]
    rows = [compare(base, runs[k], f"A vs {k.split(':')[0]}") for k in keys[1:]]
    summary = pd.DataFrame(rows).set_index("comparison")

    print("\n" + "=" * 96)
    print("RESULT -- exclusion effect (A vs B) against the seed-noise band (A vs C, A vs D)")
    print("=" * 96)
    fmt = summary.copy()
    for c in ["pct_agents_moved", "median_abs_pct", "p95_abs_pct", "book_total_pct",
              "units_median_abs_pct", "units_book_pct", "pzero_median_abs_delta"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.4%}")
    for c in ["max_abs_dollars", "book_total_base", "book_total_other"]:
        fmt[c] = fmt[c].map(lambda v: f"${v:,.0f}")
    print(fmt.T.to_string())

    summary.to_csv("exclusion_impact_summary.csv")
    print("\n[compare] wrote exclusion_impact_summary.csv")

    base.reset_index().to_csv("current_agent_sales_forecast.csv", index=False)
    print(f"[compare] run A is the production config -> current_agent_sales_forecast.csv "
          f"({len(base):,} agents)")


if __name__ == "__main__":
    main()
