"""Generate a new state's two model directories from the MA reference port.

WHY MA IS THE TEMPLATE, not RI. RI is the original and its layout is
irregular: its two models live at the repo root in differently-named folders,
its office-pair catalogue is hand-curated prose, and its Sales data.py inlines
the panel filter instead of naming it. MA is what RI looks like AFTER being
made portable -- every modelling module byte-identical, the state switch
isolated in data.py, the catalogue generated, and one wrapper driving both
models. Copying MA copies the portable shape.

WHAT THIS DOES NOT DO, on purpose:

  It does not fit, score, or promote anything. It writes source files and
  stops. Fitting is `quarterly_recalibration.py --dry-run`, and it must be a
  separate, deliberate step -- see ONBOARDING_PROTOCOL.md.

  It does not rewrite the prose in the docstrings it copies. Those paragraphs
  record findings measured on MASSACHUSETTS data -- base rates, AUCs, which
  brand is 3.40% of rows. Machine-substituting a state name into them would
  produce confident sentences that were never measured anywhere, which is
  worse than leaving them visibly about MA. Every generated file therefore
  gets a PROVENANCE banner at the top saying exactly that.

EVERY PATCH IS ASSERTED. If a constant this script expects to rewrite is not
found -- because the template changed -- it raises instead of writing a file
that silently kept Massachusetts' values. That failure mode (a state-specific
constant left pointing at another state, matching nothing, raising nothing) is
the one that has bitten this project three separate times.

    python -m states.scaffold --state ct [--force]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from . import registry as R

TEMPLATE = R.MA

# Files copied verbatim -- no state-specific content. Verified byte-identical
# between RI and MA, which is what makes them safe to copy blind.
LEAVE_VERBATIM = ["calibrate.py", "crossval.py", "explain_agents.py",
                  "generate_distribution_chart.py", "generate_report.py",
                  "score_agents.py", "split.py", "train_baseline.py"]
SALES_VERBATIM = ["cohort_prior.py", "compare_precision_levels.py",
                  "hurdle_model.py", "quantify_range.py", "quantile_model.py",
                  "remeasure_accuracy.py", "score_agents.py",
                  "sweep_confidence_levels.py", "train_baseline.py",
                  "train_trees.py", "backtest_long_horizons.py"]

# Renamed on the way out so the "ma_" prefix does not propagate into every
# future state. The template's own name is an accident of being first.
RENAMES = {
    "derive_ma_office_pairs.py": "derive_office_pairs.py",
    "derive_ma_config.py": "derive_config.py",
}

# Deliberately NOT copied.
#   backtest_ma_production.py -- superseded 2026-08-22 by remeasure_accuracy.py
#     (no baseline, one seed, R2 read as skill) and its cross-state arm is
#     contaminated. Carrying it forward would seed every new state with a
#     retired measurement.
#   rebrand_classifications.py -- GENERATED per state by derive_office_pairs.py.
#     Copying MA's would key the new state's churn recoding to MLSPIN office
#     codes that do not exist in its panel, classifying nothing while appearing
#     to work. This is the single most dangerous file to copy.
SKIP = {"backtest_ma_production.py", "rebrand_classifications.py"}

BANNER = '''# ==========================================================================
# GENERATED FOR {display} ({mls_name}) by states/scaffold.py on {date}.
# Template: MA/{sub}/{name} (Massachusetts / MLSPIN).
#
# The CONSTANTS below were rewritten for this state from states/registry.py.
# The PROSE was not. Any figure quoted in a docstring here -- base rates, AUCs,
# row counts, brand shares -- was measured on MASSACHUSETTS and has NOT been
# re-measured for {display}. Treat every one as unverified until you have run
# the steps in ONBOARDING_PROTOCOL.md and replaced it.
# ==========================================================================
'''


def _fmt_frozenset(ids) -> str:
    if not ids:
        return "frozenset()"
    return "{" + ", ".join(repr(i) for i in sorted(ids)) + "}"


def _patch(text: str, rules: list[tuple[str, str]], path: Path) -> str:
    """Apply each (pattern, replacement) and ASSERT it matched exactly once.

    The assertion is the entire point. A regex that silently matches nothing
    leaves the template's Massachusetts value in place, and nothing downstream
    can tell the difference between "correctly MA" and "wrongly still MA".
    """
    for rule in rules:
        pattern, repl = rule[0], rule[1]
        # A third element "all" means "replace every occurrence, expect >= 1" --
        # used for filenames, which appear in prose as well as in code. Default
        # is the strict form: exactly one match, or abort.
        all_occurrences = len(rule) > 2 and rule[2] == "all"
        new, n = re.subn(pattern, repl, text,
                         count=0 if all_occurrences else 1, flags=re.MULTILINE)
        if (n < 1) if all_occurrences else (n != 1):
            raise SystemExit(
                f"\nSCAFFOLD ABORTED -- pattern did not match "
                f"{'at least once' if all_occurrences else 'exactly once'} in {path.name}:\n"
                f"    {pattern}\n"
                f"  matched {n} time(s). The template has changed shape.\n"
                f"  Fix states/scaffold.py rather than hand-editing the output: a\n"
                f"  hand-edit here is exactly how a state-specific constant gets\n"
                f"  left pointing at another state's value."
            )
        text = new
    return text


def scaffold(spec: R.StateSpec, force: bool) -> int:
    from datetime import date
    if spec.mls_code == TEMPLATE.mls_code:
        raise SystemExit(f"{spec.key} IS the template state; nothing to generate.")

    t_leave, t_sales = R.leave_dir(TEMPLATE), R.sales_dir(TEMPLATE)
    o_root = R.REPO_ROOT / spec.dir_name
    o_leave, o_sales = o_root / "Likelihood_to_Leave", o_root / "Forecasted_Sales"

    if o_root.exists() and not force:
        raise SystemExit(
            f"{o_root} already exists. Refusing to overwrite a state that may hold\n"
            f"shipped scores. Pass --force only if you are certain, and check for\n"
            f"current_agent_*.csv first."
        )
    o_leave.mkdir(parents=True, exist_ok=True)
    o_sales.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    written: list[str] = []

    def emit(src: Path, dst: Path, sub: str, rules: list[tuple[str, str]] | None = None):
        text = src.read_text(encoding="utf-8")
        if rules:
            text = _patch(text, rules, src)
        banner = BANNER.format(display=spec.display_name, mls_name=spec.mls_name,
                               date=today, sub=sub, name=src.name)
        # Banner goes after the module docstring so it stays importable and the
        # docstring stays the first statement.
        m = re.match(r'^\s*(?:"""|\'\'\')', text)
        if m:
            q = text.lstrip()[:3]
            end = text.index(q, text.index(q) + 3) + 3
            text = text[:end] + "\n\n" + banner + text[end:]
        else:
            text = banner + text
        dst.write_text(text, encoding="utf-8")
        written.append(str(dst.relative_to(R.REPO_ROOT)))

    ph = _fmt_frozenset(spec.placeholder_agent_ids)
    brands = ",\n    ".join(
        ", ".join(f'"{b}"' for b in spec.office_brand_categories[i:i + 5])
        for i in range(0, len(spec.office_brand_categories), 5))

    # ---- Leave model -------------------------------------------------------
    for f in LEAVE_VERBATIM:
        p = t_leave / f
        if p.exists():
            emit(p, o_leave / f, "Likelihood_to_Leave")

    emit(t_leave / "data.py", o_leave / "data.py", "Likelihood_to_Leave", rules=[
        (r'^MLS_CODE = "mlspin"$', f'MLS_CODE = "{spec.mls_code}"'),
        (r'^NON_MLS_MEMBER_AGENT_IDS = \{[^}]*\}$',
         f'NON_MLS_MEMBER_AGENT_IDS = {ph}'),
        (r'^OFFICE_BRAND_CATEGORIES = \[[^\]]*\]$',
         f'OFFICE_BRAND_CATEGORIES = [\n    {brands},\n]'),
        # The header docstring tells the reader which script regenerates the
        # office-pair catalogue. Point it at the one that exists here.
        (r'derive_ma_office_pairs\.py', 'derive_office_pairs.py', "all"),
    ])

    emit(t_leave / "derive_ma_office_pairs.py",
         o_leave / RENAMES["derive_ma_office_pairs.py"], "Likelihood_to_Leave", rules=[
             (r'^MLS_CODE = "mlspin"$', f'MLS_CODE = "{spec.mls_code}"'),
             # "all": this filename appears in the module's own code AND inside
             # the docstring it writes into the generated rebrand_classifications.py,
             # which would otherwise point the next reader at a file that does
             # not exist in this state.
             (r'ma_office_pair_candidates\.csv', 'office_pair_candidates.csv', "all"),
             # Same reasoning: this module writes "regenerate with `python
             # derive_ma_office_pairs.py`" into the catalogue it generates. That
             # is an instruction, not a finding, and the named script does not
             # exist here. Prose figures stay; runnable commands get fixed.
             (r'derive_ma_office_pairs\.py', 'derive_office_pairs.py', "all"),
         ])

    if (t_leave / "derive_ma_config.py").exists():
        emit(t_leave / "derive_ma_config.py", o_leave / RENAMES["derive_ma_config.py"],
             "Likelihood_to_Leave")

    # ---- Sales model -------------------------------------------------------
    for f in SALES_VERBATIM:
        p = t_sales / f
        if p.exists():
            emit(p, o_sales / f, "Forecasted_Sales")

    emit(t_sales / "data.py", o_sales / "data.py", "Forecasted_Sales", rules=[
        (r'^MLS_CODE = "mlspin"$', f'MLS_CODE = "{spec.mls_code}"'),
        (r'^NON_MLS_MEMBER_AGENT_IDS = \{[^}]*\}$',
         f'NON_MLS_MEMBER_AGENT_IDS = {ph}'),
    ])

    # ---- Wrapper + its tests ----------------------------------------------
    emit(R.REPO_ROOT / TEMPLATE.dir_name / "quarterly_recalibration.py",
         o_root / "quarterly_recalibration.py", ".", rules=[
             (r'^MA_DIR = Path\(__file__\)\.resolve\(\)\.parent$',
              'STATE_DIR = Path(__file__).resolve().parent'),
             (r'^CANDIDATES_FILENAME = "ma_office_pair_candidates\.csv"$',
              'CANDIDATES_FILENAME = "office_pair_candidates.csv"'),
         ])
    # MA_DIR -> STATE_DIR everywhere else in that file, and the script rename.
    qp = o_root / "quarterly_recalibration.py"
    qt = qp.read_text(encoding="utf-8")
    qt = qt.replace("MA_DIR", "STATE_DIR").replace(
        "derive_ma_office_pairs.py", "derive_office_pairs.py")
    qp.write_text(qt, encoding="utf-8")

    tg = R.REPO_ROOT / TEMPLATE.dir_name / "test_recalibration_gates.py"
    if tg.exists():
        emit(tg, o_root / "test_recalibration_gates.py", ".")
        tp = o_root / "test_recalibration_gates.py"
        tp.write_text(tp.read_text(encoding="utf-8").replace("MA_DIR", "STATE_DIR"),
                      encoding="utf-8")

    print("=" * 78)
    print(f"SCAFFOLDED {spec.display_name} ({spec.mls_name}) -> {spec.dir_name}/")
    print("=" * 78)
    for w in written:
        print(f"  + {w}")
    print(f"\n  {len(written)} files written.")
    print(f"  MLS_CODE                 = {spec.mls_code!r}")
    print(f"  NON_MLS_MEMBER_AGENT_IDS = {sorted(spec.placeholder_agent_ids)}")
    print(f"  OFFICE_BRAND_CATEGORIES  = {len(spec.office_brand_categories)} brands")
    print("\nNOT generated (correctly):")
    print("  - rebrand_classifications.py  -> run derive_office_pairs.py; copying")
    print("    the template's would key churn recoding to MLSPIN office codes")
    print("    that do not exist in this panel.")
    print("  - backtest_ma_production.py   -> superseded by remeasure_accuracy.py")
    print("\nNEXT: python -m states.verify   then follow ONBOARDING_PROTOCOL.md step 5.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a new state's model directories.")
    ap.add_argument("--state", required=True, help="registry key, e.g. 'ct'")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing state directory")
    a = ap.parse_args()
    try:
        spec = R.get(a.state)
    except KeyError as e:
        # A traceback here is noise: the caller is following the protocol and
        # has simply not registered the state yet (protocol step 3).
        raise SystemExit(f"\n{e.args[0]}\n\nSee ONBOARDING_PROTOCOL.md steps 1-3.")
    return scaffold(spec, a.force)


if __name__ == "__main__":
    sys.exit(main())
