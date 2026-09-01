"""Derive MLSPIN (MA) office-pair churn classifications from the data.

WHY THIS EXISTS: the RI project's `rebrand_classifications.py` is a hand-curated
list of RIAR office_mls_id pairs (SMRT, CBRB23, KELW06, ...). Those codes do not
exist in MLSPIN, so importing that file into the MA model would silently
classify nothing -- every structural move (rebrand, internal office-code change,
M&A roll-up) would count as real individual churn and inflate the positive rate.

Rather than leave MA with no recoding at all, this script reproduces the two
gates the RI project defined, using only mechanisms that need no web research:

  TIER A -> CONFIRMED_NOT_CHURN_PAIRS. Origin and destination office_mls_id
  differ but their office_name is IDENTICAL after normalization. This is the
  exact zero-cost reasoning the RI project already used and documented for
  LVGP->LVGP02, CBRB23->CBRB24, CBRB15->CBRB06, CBHN->CBRB18, KELW06->KELW02:
  two office codes carrying the same office name are an internal code change,
  not an agent changing firms. No web search performed or needed.

  TIER B -> CLUSTER_GATED_PAIRS. Pairs that moved >= MIN_CLUSTER_MOVERS agents
  from the same origin to the same destination in the SAME quarter, and are NOT
  Tier A. These have NO independent evidence either way -- which is precisely
  the population CLUSTER_GATED_PAIRS was designed for (see the RI file's
  docstring: the mass-mover pattern is the only evidence, so only the actual
  flagged cluster quarters recode, never a blanket amnesty for the pair). A lone
  agent making the same move outside a cluster quarter still counts as churn.

WHAT THIS DELIBERATELY DOES NOT DO: it does not web-verify any MA pair, so no MA
pair gets the "confirmed M&A / same-ownership rebrand" treatment that in RI lets
a 3-mover trickle in an adjacent quarter recode. MA Tier B is therefore
STRICTER than RI's equivalent -- it will under-catch trickle moves around real
MA acquisitions. That is a known, documented gap (see MA/README.md), not an
oversight: it errs toward counting a structural move as churn rather than
excusing real attrition, which is the conservative direction for a churn model.

Writes `rebrand_classifications.py` in this directory. Re-run when new quarters
arrive; Tier B candidates are also dumped to `ma_office_pair_candidates.csv` for
optional human/web review, which is what would promote one to Tier A.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "leave-dataset-with-team-distinction-mls-id.csv"
)
MLS_CODE = "mlspin"
MIN_CLUSTER_MOVERS = 5  # matches data.py / the RI project's threshold
OUT_PY = Path(__file__).resolve().parent / "rebrand_classifications.py"
OUT_CSV = Path(__file__).resolve().parent / "ma_office_pair_candidates.csv"


def _normalize_name(s: pd.Series) -> pd.Series:
    """Lowercase, strip punctuation/legal suffixes and collapse whitespace, so
    'Lamacchia Realty, Inc.' and 'Lamacchia Realty Inc' compare equal. Kept
    deliberately conservative -- it only removes noise, never distinguishing
    words like a branch/town name, so two genuinely different offices of the
    same brand ('Compass - Newton' vs 'Compass - Brookline') do NOT collapse.
    """
    out = s.fillna("").str.lower()
    out = out.str.replace(r"[.,]", "", regex=True)
    out = out.str.replace(r"\b(inc|llc|llp|ltd|co|corp|company)\b", "", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def build_moves(raw: pd.DataFrame) -> pd.DataFrame:
    """Same reconstruction as data.py::_build_moves -- consecutive-snapshot
    office_mls_id changes per agent."""
    m = raw[["mls_agent_id", "office_mls_id", "snapshot_date"]].sort_values(
        ["mls_agent_id", "snapshot_date"]
    )
    m["prev_office"] = m.groupby("mls_agent_id")["office_mls_id"].shift(1)
    moved = m[m["office_mls_id"] != m["prev_office"]].dropna(subset=["prev_office"])
    return moved.rename(
        columns={
            "office_mls_id": "dest_office",
            "prev_office": "origin_office",
            "snapshot_date": "move_quarter",
        }
    )[["mls_agent_id", "origin_office", "dest_office", "move_quarter"]]


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "mls_code",
            "mls_agent_id",
            "office_mls_id",
            "office_name",
            "snapshot_date",
            "is_team",
        ],
        parse_dates=["snapshot_date"],
        low_memory=False,
    )
    df = df[df["mls_code"] == MLS_CODE].copy()
    print(f"[derive] {MLS_CODE}: {len(df)} rows, {df.mls_agent_id.nunique()} agents")

    # Most common name per office code (names occasionally vary row to row).
    name_by_code = (
        df.dropna(subset=["office_mls_id"])
        .groupby("office_mls_id")["office_name"]
        .agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else "")
    )
    norm_by_code = _normalize_name(name_by_code)

    moves = build_moves(df)
    print(f"[derive] {len(moves)} office-to-office moves reconstructed")

    pair_counts = moves.groupby(["origin_office", "dest_office"]).size()
    cluster_sizes = moves.groupby(["origin_office", "dest_office", "move_quarter"]).size()

    # ---- Tier A: identical normalized office name across the pair ----
    tier_a = {}
    for (o, d), n_total in pair_counts.items():
        no, nd = norm_by_code.get(o, ""), norm_by_code.get(d, "")
        if no and nd and no == nd:
            tier_a[(o, d)] = (
                f"{name_by_code.get(o)} -> {name_by_code.get(d)}: identical office name, "
                f"internal office-code change ({n_total} moves)"
            )

    # ---- Tier B: mass-mover clusters not already Tier A ----
    big = cluster_sizes[cluster_sizes >= MIN_CLUSTER_MOVERS]
    tier_b = {}
    rows = []
    for (o, d, q), n in big.items():
        if (o, d) in tier_a:
            continue
        # Containment flag: one office name wholly contains the other after
        # normalization ("Bentley's" -> "RE/MAX Bentley's", "Custom Home Realty"
        # -> "Century 21 Custom Home Realty"). Strong prima-facie evidence of a
        # franchise-flag change or rebrand rather than agents changing firms.
        # NOT auto-promoted to Tier A -- the RI project's convention is that
        # anything past exact-name-match is a human call -- but flagged here so
        # the review queue is ordered by how likely a promotion is.
        no, nd = norm_by_code.get(o, ""), norm_by_code.get(d, "")
        containment = bool(no and nd and (no in nd or nd in no))
        rows.append(
            {
                "origin_office": o,
                "dest_office": d,
                "origin_name": name_by_code.get(o),
                "dest_name": name_by_code.get(d),
                "move_quarter": q.date(),
                "n_movers": n,
                "pair_total_moves": pair_counts.get((o, d), 0),
                "name_containment": containment,
            }
        )
        prev = tier_b.get((o, d))
        desc = (
            f"{name_by_code.get(o)} -> {name_by_code.get(d)}: mass-mover cluster, "
            f"UNVERIFIED (no web research performed)"
        )
        if prev is None:
            tier_b[(o, d)] = desc

    cand = pd.DataFrame(rows).sort_values(
        ["name_containment", "n_movers"], ascending=[False, False]
    )
    cand.to_csv(OUT_CSV, index=False)

    print(f"[derive] Tier A (identical name, confirmed not-churn): {len(tier_a)} pairs")
    print(f"[derive] Tier B (unverified mass-mover clusters):      {len(tier_b)} pairs "
          f"across {len(cand)} (pair, quarter) instances")
    if len(cand):
        print("\n[derive] top Tier B candidates for review:")
        print(cand.head(15).to_string(index=False))

    _write_module(tier_a, tier_b, len(cand))
    print(f"\n[derive] wrote {OUT_PY}")
    print(f"[derive] wrote {OUT_CSV}")


def _write_module(tier_a: dict, tier_b: dict, n_instances: int) -> None:
    def fmt(d: dict) -> str:
        if not d:
            return "{}"
        lines = ["{"]
        for (o, dd), desc in sorted(d.items()):
            safe = str(desc).replace('"', "'")
            lines.append(f'    ("{o}", "{dd}"): "{safe}",')
        lines.append("}")
        return "\n".join(lines)

    body = f'''"""MLSPIN (MA) office-pair churn classifications -- GENERATED FILE.

Do not hand-edit: regenerate with `python derive_ma_office_pairs.py`. See that
script's docstring for the full rationale and for what this deliberately does
NOT do (no web verification of MA pairs, so no MA equivalent of RI's
confirmed-M&A trickle handling).

Mirrors the RI project's two-gate structure so `data.py::_recode_mass_mover_moves`
works unchanged:

  CONFIRMED_NOT_CHURN_PAIRS -- Tier A, origin and destination office codes carry
  an IDENTICAL office name. Every move on the pair recodes to not-churn in any
  quarter at any volume, exactly as in RI. Derived mechanically; the RI project
  used the same zero-cost reasoning for its own internal office-code pairs.

  CLUSTER_GATED_PAIRS -- Tier B, pairs that moved >= 5 agents together in some
  quarter and are NOT name-identical. UNVERIFIED: no independent evidence either
  way, so only the specific quarter(s) that independently clear the 5-mover
  threshold recode. A lone mover on this pair in another quarter stays churn.

Counts at generation time: {len(tier_a)} Tier A pairs, {len(tier_b)} Tier B pairs
({n_instances} (pair, quarter) cluster instances). Tier B candidates are listed
in ma_office_pair_candidates.csv for optional human/web review -- confirming one
would promote it into CONFIRMED_NOT_CHURN_PAIRS by hand in the derive script.
"""

# Tier A: identical office name across the pair -> internal code change.
CONFIRMED_NOT_CHURN_PAIRS = {fmt(tier_a)}

# Tier B: unverified mass-mover clusters. Recoded ONLY in the quarter(s) that
# independently clear MIN_CLUSTER_MOVERS (data.py checks the cluster size).
CLUSTER_GATED_PAIRS = {fmt(tier_b)}

ALL_KNOWN_PAIRS = {{**CONFIRMED_NOT_CHURN_PAIRS, **CLUSTER_GATED_PAIRS}}
'''
    OUT_PY.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
