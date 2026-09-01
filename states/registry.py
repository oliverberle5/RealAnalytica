"""THE registry: one declaration per state, for BOTH models.

WHY THIS EXISTS. Onboarding Massachusetts took a careful human port and still
shipped three silent failures, every one of them the same shape: a constant
that was supposed to be state-specific stayed pointed at another state's value,
matched nothing, and raised no error.

  * MA inherited RI's placeholder id -> excluded nothing, silently.
  * `backtest_ma_production.py --mls riar` overrode MLS_CODE but not
    NON_MLS_MEMBER_AGENT_IDS -> measured RI with all three buckets left in,
    and reported the inflated number as RI's accuracy for three weeks.
  * The 2026-08-10 identifier swap landed in data.py but nothing re-scored ->
    two shipped spreadsheets sat on a retired key while every number in them
    looked fine.

None of those is a modelling mistake. All three are one bug: **state-specific
values living in more than one place, with nothing checking they agree.** This
file is the one place. Everything else derives from it or is checked against it.

THE INVARIANT, worth stating once. All states read the SAME source CSV and
differ only by `mls_code`. There is no per-state copy of the data, and adding
one would recreate the drift this file exists to remove.

TO ADD A STATE: do not hand-write a StateSpec. Run
`python -m states.discover --mls-code <code>`, which derives every field below
from the data and prints a ready-to-paste block. See ONBOARDING_PROTOCOL.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The single shared source CSV. Every state reads this and filters by mls_code.
DATA_PATH = REPO_ROOT / "data" / "leave-dataset-with-team-distinction-mls-id.csv"

# The agent key. MUST be read as str: RIAR ids carry meaningful leading zeros
# ("00001", "0014") and MLSPIN ids are alphanumeric ("A9500308"), so dtype
# inference corrupts the first group and mixes types across the second --
# which makes every id-based exclusion below silently stop matching.
AGENT_ID_COLUMN = "mls_agent_id"

# Shared across states. Changing one of these changes every state, which is the
# intent -- they are properties of the method, not of a market.
HORIZON_DAYS = 91
HORIZON_MONTHS = 3
MIN_SNAPSHOT_ROWS = 500
MIN_CLUSTER_MOVERS = 5


@dataclass(frozen=True)
class StateSpec:
    """Everything that differs between one state and another, in both models."""

    key: str                      # short handle, e.g. "ma"
    mls_code: str                 # the panel filter -- THE state switch
    display_name: str             # "Massachusetts"
    mls_name: str                 # "MLSPIN"
    dir_name: str                 # repo subdirectory, e.g. "MA"

    # Synthetic non-agent records: buckets for cross-board / assisted / non-member
    # activity. Not people, so no departure and no next-12m production is defined
    # for them. STATE-SPECIFIC -- an id from another panel matches nothing.
    placeholder_agent_ids: frozenset[str]

    # Leave model only. The Sales model builds brand one-hots dynamically via
    # get_dummies, so it picks new brands up automatically; the Leave model's
    # list is hardcoded and must be extended per state.
    office_brand_categories: tuple[str, ...]

    # Whether this state's office-pair catalogue has had human/web verification.
    # False means Tier B is cluster-gated only -- stricter than RI, and it will
    # under-catch trickle moves around real acquisitions. A known gap, surfaced
    # rather than hidden.
    office_pairs_web_verified: bool = False

    # Set when the state is scaffolded but not yet validated end-to-end.
    provisional: bool = False

    notes: str = ""
    # Panel shape at registration, for drift detection. Informational.
    registered_rows: int | None = None
    registered_agents: int | None = None

    @property
    def leave_dir(self) -> Path:
        return REPO_ROOT / self.dir_name / "Likelihood_to_Leave"

    @property
    def sales_dir(self) -> Path:
        return REPO_ROOT / self.dir_name / "Forecasted_Sales"


# --------------------------------------------------------------------------
# Registered states.
#
# RI and MA are transcribed from the values live in their data.py files as of
# 2026-09-01, NOT re-derived -- the registry must describe what production
# actually does before anything is allowed to generate from it.
# `python -m states.verify` proves these still match the shipped files.
# --------------------------------------------------------------------------

RI = StateSpec(
    key="ri",
    mls_code="riar",
    display_name="Rhode Island",
    mls_name="RIAR",
    dir_name="RI",
    placeholder_agent_ids=frozenset({"12345", "00001", "0014"}),
    office_brand_categories=(
        "independent", "remax", "keller_williams", "century21", "coldwell_banker",
        "compass", "sothebys", "exp", "raveis", "era", "redfin", "laer",
        "corcoran", "elliman", "berkshire_hathaway",
    ),
    office_pairs_web_verified=True,
    registered_rows=84_417,
    registered_agents=5_625,
    notes=(
        "The reference implementation. Office pairs are hand-curated and "
        "web-verified, so RI gets confirmed-M&A trickle handling no generated "
        "catalogue can reproduce."
    ),
)

MA = StateSpec(
    key="ma",
    mls_code="mlspin",
    display_name="Massachusetts",
    mls_name="MLSPIN",
    dir_name="MA",
    placeholder_agent_ids=frozenset({"H1111111"}),
    office_brand_categories=RI.office_brand_categories + ("bhgre",),
    office_pairs_web_verified=False,
    registered_rows=612_358,
    registered_agents=45_186,
    notes=(
        "First port. Proves the pipeline is state-agnostic: every modelling "
        "module is byte-identical to RI and the switch lives entirely in "
        "data.py. Office pairs are GENERATED, never web-verified -- Tier A is "
        "735 name-identical pairs, Tier B is 111 cluster-gated pairs."
    ),
)

STATES: dict[str, StateSpec] = {s.key: s for s in (RI, MA)}
BY_MLS_CODE: dict[str, StateSpec] = {s.mls_code: s for s in STATES.values()}


def get(key: str) -> StateSpec:
    if key not in STATES:
        raise KeyError(
            f"unknown state {key!r}. Registered: {sorted(STATES)}. "
            f"To add one, run: python -m states.discover --mls-code <code>"
        )
    return STATES[key]


# Every state now has the identical shape: <STATE>/Likelihood_to_Leave and
# <STATE>/Forecasted_Sales. These wrappers used to special-case RI, whose two
# models lived at the repo root under different names. The 2026-09-01
# reorganisation removed that exception rather than keep coding around it.
def leave_dir(spec: StateSpec) -> Path:
    return spec.leave_dir


def sales_dir(spec: StateSpec) -> Path:
    return spec.sales_dir
