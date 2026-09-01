"""Derive a new state's StateSpec from the data. Run this FIRST.

WHY THIS EXISTS. Onboarding MA required finding MLSPIN's synthetic placeholder
record by hand. It was found by matching the RI placeholder's structural
signature -- present in every snapshot, is_team==0, zero list-sides ever,
single office, tenure incrementing forever. That reasoning was written down in
a docstring and then had to be re-performed from scratch, from prose, for the
next state. This module is that reasoning as code.

It matters because getting it wrong is SILENT. A placeholder left in the panel
is a non-person with ~$1.1B of trailing volume sitting in the training
distribution as one extraordinarily productive agent, and it will be scored and
shipped -- RI's "00001" was delivered at $51,688,219, rank 11 of 4,607, and
nothing in the spreadsheet looked wrong. Inheriting another panel's id excludes
nothing and also looks fine.

TWO INDEPENDENT CHANNELS, because either alone misses a known case:

  STRUCTURAL -- persistent, non-team, single-office, side-degenerate. This is
  what found MA's "H1111111" (23,127 units, all buy-side). It ranks by size, so
  it would rank RI's "0014" (21 units) near the bottom.

  NOMENCLATURE -- agent_name / office_name matching the bucket vocabulary
  ("Non Member", "Assisted Sale", "ALLIANCE MEMBER", "CONNECTICUT MLS"). This
  catches small buckets the structural channel buries.

Reported as a UNION with every piece of evidence shown, and NEVER auto-adopted.
A false positive here silently deletes a real, possibly large, agent from both
models -- the mirror image of the bug this is preventing, and just as invisible.
A human or agent confirms; this tool does the looking.

    python -m states.discover --mls-code <code>
    python -m states.discover --self-test
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from . import registry as R

# The vocabulary MLS feeds use for non-person buckets. Deliberately broad: this
# channel only nominates candidates for review, so a false positive costs one
# line of output, while a miss costs a shipped non-agent.
NAME_PATTERNS = re.compile(
    r"non[\s\-_]*mls|non[\s\-_]*member|assisted|alliance\s+member|"
    r"\bmls\b|cross[\s\-_]*board|\bunknown\b|placeholder",
    re.IGNORECASE,
)

NEEDED = ["mls_code", "mls_agent_id", "snapshot_date", "office_mls_id",
          "office_name", "office_brand", "agent_name", "is_team", "units_12m",
          "list_sides_12m", "volume_12m", "career_months"]

# Structural gates.
#
# CALIBRATED AGAINST THE KNOWN ANSWERS (2026-09-01), not guessed. The first
# version of this gate required `n_offices == 1` and only a low absolute unit
# floor. It nominated 970 MLSPIN agents and still MISSED "H1111111" -- the one
# record it was written to find -- because that bucket is booked under two
# office codes. Worst of both: unusable noise AND a false negative.
#
# What actually separates a bucket from a real agent is not office count, it is
# SIZE EXTREMITY combined with side degeneracy. "H1111111" sits at the 99.998th
# percentile of its panel by units. Plenty of small real agents are 100%
# buy-side; essentially none of them are also the largest record in the market.
# So the size gate is a panel-relative quantile, not an absolute number -- an
# absolute floor cannot transfer between a 5,625-agent panel and a 45,186-agent
# one. `n_offices` is still REPORTED in the table as evidence, just not gated on.
MIN_COVERAGE = 0.50        # share of the panel's quarters the record appears in
MIN_DEGENERACY = 0.98      # max(list_share, buy_share); a bucket is ~1.0
SIZE_QUANTILE = 0.999      # panel-relative size gate for the structural channel
MIN_UNITS = 15             # absolute floor, guards a tiny or degenerate panel


_RAW: pd.DataFrame | None = None


def _read_source() -> pd.DataFrame:
    """Read the shared CSV once per process. It is ~312MB, so the self-test
    (which profiles every registered state) would otherwise re-read it per
    state and dominate the runtime."""
    global _RAW
    if _RAW is None:
        _RAW = pd.read_csv(
            R.DATA_PATH, usecols=NEEDED,
            dtype={"mls_agent_id": str, "office_mls_id": str},
        )
    return _RAW


def load_panel(mls_code: str) -> pd.DataFrame:
    df = _read_source()
    codes = sorted(df["mls_code"].dropna().unique())
    if mls_code not in codes:
        raise SystemExit(
            f"\nmls_code {mls_code!r} is NOT in the source file.\n"
            f"Panels present: {codes}\n\n"
            f"If you expected this state, the data has not landed yet -- the file\n"
            f"bundles panels, and a state absent here cannot be modelled at all.\n"
            f"Note that filtering on agent_state is NOT a substitute: every market_*\n"
            f"and state_* column is keyed to the MLS, not the agent's address, so an\n"
            f"agent_state split yields a subset carrying another market's macro data.\n"
            f"See CT_EXPANSION_ASSESSMENT.md for how this was handled for CT."
        )
    return df[df["mls_code"] == mls_code].copy()


def profile_agents(panel: pd.DataFrame) -> pd.DataFrame:
    n_quarters = panel["snapshot_date"].nunique()
    panel = panel.copy()
    panel["list_sides_12m"] = panel["list_sides_12m"].fillna(0)
    panel["units_12m"] = panel["units_12m"].fillna(0)
    # Sorted once here so the vectorised monotonicity check below is meaningful
    # (a per-agent .apply over 45k groups is the slowest thing in this module).
    panel = panel.sort_values(["mls_agent_id", "snapshot_date"])

    g = panel.groupby("mls_agent_id")
    prof = pd.DataFrame({
        "n_snapshots": g["snapshot_date"].nunique(),
        "n_offices": g["office_mls_id"].nunique(),
        "ever_team": g["is_team"].max(),
        "units_sum": g["units_12m"].sum(),
        "units_max": g["units_12m"].max(),
        "list_sum": g["list_sides_12m"].sum(),
        "volume_sum": g["volume_12m"].sum(),
        "agent_name": g["agent_name"].last(),
        "office_name": g["office_name"].last(),
    })
    prof["coverage"] = prof["n_snapshots"] / n_quarters
    # Side degeneracy: a real agent lists AND sells; a bucket does exactly one.
    share = (prof["list_sum"] / prof["units_sum"].replace(0, np.nan)).astype(float)
    prof["list_share"] = share
    prof["degeneracy"] = pd.concat([share, 1 - share], axis=1).max(axis=1)
    # career_months should never decrease for a synthetic record -- it is a
    # clock, not a career. Weak evidence on its own, so it is reported in the
    # table but never gated on. Vectorised: within each agent (already sorted
    # above), no backward step in career_months.
    step = panel.groupby("mls_agent_id")["career_months"].diff()
    prof["career_monotonic"] = (
        step.fillna(0).ge(0).groupby(panel["mls_agent_id"]).all())
    return prof


def find_candidates(prof: pd.DataFrame) -> pd.DataFrame:
    size_gate = max(prof["units_sum"].quantile(SIZE_QUANTILE), MIN_UNITS)
    structural = (
        (prof["ever_team"] == 0)
        & (prof["coverage"] >= MIN_COVERAGE)
        & (prof["degeneracy"].fillna(0) >= MIN_DEGENERACY)
        & (prof["units_sum"] >= size_gate)
    )
    text = prof["agent_name"].fillna("") + " || " + prof["office_name"].fillna("")
    nomenclature = text.str.contains(NAME_PATTERNS) & (prof["ever_team"] == 0)

    hits = prof[structural | nomenclature].copy()
    hits["channel"] = (
        structural[hits.index].map({True: "S", False: "-"})
        + nomenclature[hits.index].map({True: "N", False: "-"})
    )
    return hits.sort_values("units_sum", ascending=False)


def report(mls_code: str, panel: pd.DataFrame, hits: pd.DataFrame, limit: int) -> None:
    print("=" * 96)
    print(f"PLACEHOLDER DISCOVERY -- mls_code = {mls_code!r}")
    print("=" * 96)
    print(f"  panel: {len(panel):,} rows | {panel.mls_agent_id.nunique():,} agents | "
          f"{panel.snapshot_date.nunique()} quarters "
          f"({panel.snapshot_date.min()} .. {panel.snapshot_date.max()})")
    print("  channels: S=structural (persistent + non-team + side-degenerate + "
          "top-0.1% by size), N=nomenclature")
    print(f"  {len(hits)} candidate(s); showing up to {limit}\n")
    if hits.empty:
        print("  NONE. Do not conclude the panel is clean -- loosen MIN_* and re-run,")
        print("  and inspect the largest agents by hand before registering an empty set.")
        return
    hdr = (f"  {'agent_id':<14}{'ch':<4}{'units':>9}{'cov':>7}{'deg':>7}{'off':>5}"
           f"{'mono':>6}  {'agent_name':<28}{'office_name'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for aid, r in hits.head(limit).iterrows():
        deg = "n/a" if pd.isna(r["degeneracy"]) else f"{r['degeneracy']:.2f}"
        print(f"  {str(aid):<14}{r['channel']:<4}{int(r['units_sum']):>9,}"
              f"{r['coverage']:>7.2f}{deg:>7}{int(r['n_offices']):>5}"
              f"{str(bool(r['career_monotonic']))[:5]:>6}  "
              f"{str(r['agent_name'])[:27]:<28}{str(r['office_name'])[:34]}")
    print()
    print("  READ THIS BEFORE ADOPTING. A candidate is a nomination, not a finding.")
    print("  Confirm each one is a BUCKET, not a person, on the evidence above:")
    print("    - degeneracy ~1.00 (never both lists and sells) is the strongest signal")
    print("    - coverage ~1.00 with a single office = a fixture, not a career")
    print("    - a real top producer will FAIL degeneracy; if a large candidate has")
    print("      deg < 0.98 it is almost certainly a real agent -- leave it in")
    print("  Excluding a real agent removes them from BOTH models, silently.")


def self_test() -> int:
    """Prove the detector rediscovers every placeholder already adopted.

    This is the only evidence that the tool works. A detector that has never
    been shown to find a known answer is a guess with a progress bar.
    """
    print("=" * 96)
    print("SELF-TEST -- can the detector rediscover the placeholders already in production?")
    print("=" * 96)
    ok = True
    for spec in R.STATES.values():
        panel = load_panel(spec.mls_code)
        hits = find_candidates(profile_agents(panel))
        found = set(hits.index.astype(str))
        want = set(spec.placeholder_agent_ids)
        missed = want - found
        status = "PASS" if not missed else "FAIL"
        if missed:
            ok = False
        print(f"\n  {status}  {spec.key} ({spec.mls_code}): registry has {sorted(want)}")
        print(f"        detector nominated {len(found)} candidate(s); "
              f"recovered {sorted(want & found)}")
        if missed:
            print(f"        *** MISSED {sorted(missed)} -- the gates are too strict ***")
        extra = found - want
        if extra:
            print(f"        {len(extra)} additional nomination(s) not in the registry "
                  f"(expected -- these are for review): {sorted(extra)[:8]}")
    print()
    print("PASS means the gates are loose enough to surface a known bucket.")
    print("It does NOT mean every nomination is a bucket -- that is the reviewer's job.")
    return 0 if ok else 1


def emit_spec(mls_code: str, panel: pd.DataFrame, hits: pd.DataFrame) -> None:
    brands = sorted(panel["office_brand"].dropna().unique())
    known = set(R.RI.office_brand_categories)
    extra = tuple(b for b in brands if b not in known)
    # Deliberately pre-fills NOTHING. An earlier version seeded this set with the
    # top nominations by size; on MLSPIN that would have proposed excluding four
    # visibly real people (a named agent with 2,312 units among them) alongside
    # the one true bucket. Excluding a real agent is silent in both models, so
    # the default must be the safe one: every candidate is a commented line the
    # reviewer uncomments after confirming it.
    cand_lines = []
    for aid, r in hits.head(12).iterrows():
        deg = "n/a" if pd.isna(r["degeneracy"]) else f"{r['degeneracy']:.2f}"
        cand_lines.append(
            f"    #   {str(aid):<12} ch={r['channel']}  units={int(r['units_sum']):>7,}  "
            f"cov={r['coverage']:.2f}  deg={deg}  "
            f"{str(r['agent_name'])[:24]!r} @ {str(r['office_name'])[:28]!r}")
    cands = "\n".join(cand_lines) if cand_lines else "    #   (none nominated)"
    print("\n" + "=" * 96)
    print("DRAFT StateSpec -- review every field, then paste into states/registry.py")
    print("=" * 96)
    print(f"""
NEW = StateSpec(
    key="<short handle, e.g. 'ct'>",
    mls_code={mls_code!r},
    display_name="<Full State Name>",
    mls_name="<MLS brand name>",
    dir_name="<REPO SUBDIR, e.g. 'CT'>",
    # CANDIDATES -- uncomment ONLY the ones you confirmed are buckets, not people.
    # Starts empty on purpose: a wrong entry here deletes a real agent from both
    # models with no error and no visible symptom.
{cands}
    placeholder_agent_ids=frozenset(),
    office_brand_categories=RI.office_brand_categories + {extra!r},
    office_pairs_web_verified=False,
    provisional=True,
    registered_rows={len(panel)},
    registered_agents={panel.mls_agent_id.nunique()},
    notes="<what you confirmed, and what you did not>",
)""")
    if extra:
        print(f"  NOTE: brands present here but absent from RI's list: {list(extra)}")
        print("  These MUST be added, or the Leave model one-hots them to all-zero and")
        print("  every agent at those brands looks brand-less. The Sales model builds")
        print("  its brand dummies dynamically and needs no change.")
    else:
        print("  No new office brands -- RI's list covers this panel.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive a new state's StateSpec from the data.")
    ap.add_argument("--mls-code", help="panel to profile, e.g. 'smartmls'")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--self-test", action="store_true",
                    help="prove the detector recovers already-adopted placeholders")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.mls_code:
        ap.error("--mls-code is required (or use --self-test)")

    panel = load_panel(a.mls_code)
    hits = find_candidates(profile_agents(panel))
    report(a.mls_code, panel, hits, a.limit)
    emit_spec(a.mls_code, panel, hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
