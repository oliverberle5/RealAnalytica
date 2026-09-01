"""Fault injection for `quarterly_recalibration.py`'s output checks.

WHY THIS EXISTS. A gate that has only ever seen good input is not a gate --
it is a line of code that has never once been observed to do its job. Every
check in the wrapper passed on the first run, which proves nothing except that
correct output is correct. This corrupts the currently shipped outputs one
defect at a time, in memory, and asserts the matching check fires.

It earned its place immediately: the first run found that `check_sales` raised
KeyError on a missing column instead of reporting a failure, which would have
killed the wrapper with a traceback rather than taking the orderly "checks
failed, nothing promoted" path -- and would have skipped every remaining check,
including the cross-model one. Fixed 2026-08-30.

The first two cases are not hypothetical. They are the defect that actually
shipped: on 2026-08-10 the agent identifier changed, nothing re-ran the
scoring, and both MA outputs sat on the retired `agent_profile_id` for three
weeks while every single-file check would have passed. Case 1 is that exact
state; case 2 is the half-migrated state you pass through while fixing it.

Reads the shipped files read-only and mutates copies in memory -- it never
writes anything. Run it after touching any check:

    ..\\Likelihood_to_Leave_Algorithm\\.venv\\Scripts\\python.exe test_recalibration_gates.py

Exit 0 = every injected defect was caught and clean output raised nothing.
"""

import json
import sys
from pathlib import Path

import pandas as pd

MA_DIR = Path(__file__).resolve().parent
if str(MA_DIR) not in sys.path:
    sys.path.insert(0, str(MA_DIR))

import quarterly_recalibration as qr

SCORES = qr.LEAVE_DIR / qr.SCORES_FILENAME
FORECAST = qr.SALES_DIR / qr.FORECAST_FILENAME
META = qr.LEAVE_DIR / qr.META_FILENAME

for path in (SCORES, FORECAST, META):
    if not path.exists():
        print(f"[test] {path} is missing -- score the models (or run --adopt-current) first.", file=sys.stderr)
        sys.exit(3)

scores0 = pd.read_csv(SCORES, dtype={qr.AGENT_KEY: str})
forecast0 = pd.read_csv(FORECAST, dtype={qr.AGENT_KEY: str})
meta0 = json.loads(META.read_text(encoding="utf-8"))

results: list[tuple[str, bool, str]] = []


def case(name, fn):
    """Record whether the check under test produced at least one failure."""
    failures = fn()
    results.append((name, bool(failures), failures[0][:100] if failures else "NO FAILURE RAISED"))


def leave_failures(scores=None, meta=None, state=None):
    failures, _warnings = qr.check_leave(scores if scores is not None else scores0,
                                         meta if meta is not None else meta0,
                                         state or {})
    return failures


def sales_failures(forecast):
    failures, _warnings = qr.check_sales(forecast)
    return failures


# --- the defect that actually shipped, 2026-08-10 to 2026-08-30 ------------
case("THE 2026-08 DEFECT: both files still on agent_profile_id",
     lambda: qr.check_cross_model(scores0.rename(columns={qr.AGENT_KEY: "agent_profile_id"}),
                                  forecast0.rename(columns={qr.AGENT_KEY: "agent_profile_id"})))

case("half-migrated: Leave re-keyed, Sales not",
     lambda: qr.check_cross_model(scores0, forecast0.rename(columns={qr.AGENT_KEY: "agent_profile_id"})))

case("roster mismatch: Sales missing 50 agents",
     lambda: qr.check_cross_model(scores0, forecast0.iloc[:-50]))


def _stale_snapshot():
    forecast = forecast0.copy()
    forecast["snapshot_date"] = "2026-04-01"
    return qr.check_cross_model(scores0, forecast)


case("snapshot mismatch: Sales a quarter behind", _stale_snapshot)


# --- RI's own shipped bug, 2026-07-24: cutpoints and the multiplier
# --- denominator computed on different bases
def _broken_invariant():
    scores = scores0.copy()
    scores["risk_multiplier"] = scores["risk_multiplier"] * 3.0
    return leave_failures(scores=scores)


case("cutpoints/multiplier on different bases (RI's 2026-07-24 bug)", _broken_invariant)


def _collapsed_top_tier():
    scores = scores0.copy()
    scores.loc[scores["risk_tier"] == 4, "risk_tier"] = 3
    scores.loc[scores.index[0], "risk_tier"] = 4
    return [f for f in leave_failures(scores=scores) if "Extra High" in f]


case("Extra High collapses to 1 agent", _collapsed_top_tier)

case("bias correction 9.0x -- calibrator unusable",
     lambda: [f for f in leave_failures(meta=dict(meta0, bias_correction=9.0)) if "bias correction" in f])


# --- Sales structural defects ---------------------------------------------
def _crossed_interval():
    forecast = forecast0.copy()
    forecast.loc[forecast.index[:20], "volume_min_80"] = forecast.loc[forecast.index[:20], "volume_max_80"] + 1
    return [f for f in sales_failures(forecast) if "crossed" in f]


case("crossed interval: conformal adjustment with the wrong sign", _crossed_interval)

case("forecast loses a column (schema parity broken)",
     lambda: [f for f in sales_failures(forecast0.drop(columns=["units_max_95"])) if "column" in f])


def _negative_point():
    forecast = forecast0.copy()
    forecast.loc[forecast.index[0], "volume_point_estimate"] = -1
    return [f for f in sales_failures(forecast) if "negative" in f]


case("negative volume point estimate", _negative_point)


def _bad_probability():
    forecast = forecast0.copy()
    forecast.loc[forecast.index[0], "units_pct_zero"] = 1.4
    return [f for f in sales_failures(forecast) if "probability" in f]


case("units_pct_zero above 1 -- not a probability", _bad_probability)


# --- the control: unmodified shipped output must raise nothing ------------
clean = leave_failures() + sales_failures(forecast0) + qr.check_cross_model(scores0, forecast0)

print()
print("=" * 100)
print("FAULT INJECTION RESULTS")
print("=" * 100)
width = max(len(name) for name, _, _ in results)
for name, caught, detail in results:
    print(f"  [{'CAUGHT' if caught else 'MISSED'}]  {name:<{width}}  {detail}")
print()
print(f"  [{'PASS' if not clean else 'FAIL'}]    control: unmodified shipped output raises "
      f"{len(clean)} failure(s), expected 0")
print()
n_caught = sum(1 for _, caught, _ in results if caught)
print(f"{n_caught}/{len(results)} injected defects caught; control clean: {not clean}")
sys.exit(0 if n_caught == len(results) and not clean else 1)
