"""Prove the registry still matches the constants live in every shipped data.py.

WHY. A registry that drifts from production is WORSE than no registry: it is a
confident-looking second source of truth. This module is what stops that. It
re-reads each state's actual `data.py` files and asserts the values there equal
the StateSpec. If someone edits a constant in one place and not the other, this
fails loudly instead of the two quietly disagreeing.

Run it after ANY change to a data.py or to registry.py:
    python -m states.verify
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from . import registry as R


def _literal(path: Path, name: str):
    """Read a module-level constant WITHOUT importing the module.

    Importing would execute the file, which for these data.py modules pulls in
    pandas and (for the Leave model) a generated rebrand_classifications.py that
    may not exist yet in a half-scaffolded state. Parsing the AST reads the
    declared value with no side effects and no import order problems.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        return None  # non-literal (e.g. a computed Path)
    raise KeyError(f"{name} not found in {path}")


def _inline_mls_code(path: Path) -> str | None:
    """Fallback for a data.py that inlines the panel filter as a literal
    (`df[df["mls_code"] == "riar"]`) instead of naming a MLS_CODE constant.

    RI's Forecasted_Sales data.py does exactly this, so the value that decides
    which market the production model trains on is a bare string in the middle
    of a function. That is legal and it works -- but it is the one constant
    whose whole job is to be the state switch, and it is the only copy of it
    that no tool can see. Reported as a WARNING below rather than a failure:
    it is not currently wrong, it is merely unreadable to anything but a human.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Subscript)
                and isinstance(node.left.slice, ast.Constant)
                and node.left.slice.value == "mls_code"
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)):
            return node.comparators[0].value
    return None


def check_state(spec: R.StateSpec, warnings: list[str]) -> list[str]:
    problems: list[str] = []
    leave = R.leave_dir(spec) / "data.py"
    sales = R.sales_dir(spec) / "data.py"

    for label, p in (("leave", leave), ("sales", sales)):
        if not p.exists():
            problems.append(f"[{spec.key}/{label}] MISSING {p}")
            continue

        try:
            got = _literal(p, "MLS_CODE")
        except KeyError:
            got = _inline_mls_code(p)
            if got is not None:
                warnings.append(
                    f"[{spec.key}/{label}] no MLS_CODE constant -- panel filter is "
                    f"inlined as the literal {got!r}. Correct today, but invisible to "
                    f"tooling; naming it would make the state switch greppable.")
        if got is None:
            problems.append(f"[{spec.key}/{label}] cannot determine the panel filter at all")
        elif got != spec.mls_code:
            problems.append(
                f"[{spec.key}/{label}] panel filter is {got!r}, registry says {spec.mls_code!r}")

        ph = _literal(p, "NON_MLS_MEMBER_AGENT_IDS")
        if ph is not None and set(ph) != set(spec.placeholder_agent_ids):
            problems.append(
                f"[{spec.key}/{label}] placeholder ids {sorted(ph)} != "
                f"registry {sorted(spec.placeholder_agent_ids)}")

    # Brand list: Leave model only -- Sales builds one-hots dynamically.
    if leave.exists():
        try:
            brands = _literal(leave, "OFFICE_BRAND_CATEGORIES")
        except KeyError:
            brands = None
        if brands is not None and tuple(brands) != tuple(spec.office_brand_categories):
            extra = set(brands) - set(spec.office_brand_categories)
            miss = set(spec.office_brand_categories) - set(brands)
            problems.append(
                f"[{spec.key}/leave] OFFICE_BRAND_CATEGORIES differs "
                f"(in file not registry: {sorted(extra)}; in registry not file: {sorted(miss)})")
    return problems


def main() -> int:
    print("=" * 74)
    print("REGISTRY <-> PRODUCTION CONSISTENCY")
    print("=" * 74)
    if not R.DATA_PATH.exists():
        print(f"  FAIL  shared source CSV missing: {R.DATA_PATH}")
        return 1
    print(f"  shared source: {R.DATA_PATH.name}")
    print(f"  ({R.DATA_PATH.stat().st_size:,} bytes)\n")

    all_problems: list[str] = []
    warnings: list[str] = []
    for key, spec in R.STATES.items():
        probs = check_state(spec, warnings)
        flag = "OK  " if not probs else "FAIL"
        tag = "  [PROVISIONAL]" if spec.provisional else ""
        print(f"  {flag}  {spec.key:<4} {spec.display_name:<16} "
              f"mls_code={spec.mls_code:<8} placeholders={sorted(spec.placeholder_agent_ids)}{tag}")
        all_problems += probs

    # Cross-state check: no two states may claim the same panel, and no
    # placeholder id may be shared -- an id from another panel matches nothing
    # and is the exact bug that made MA exclude zero rows.
    seen: dict[str, str] = {}
    for key, spec in R.STATES.items():
        if spec.mls_code in seen:
            all_problems.append(
                f"[cross] {key} and {seen[spec.mls_code]} both claim mls_code {spec.mls_code!r}")
        seen[spec.mls_code] = key
    for a in R.STATES.values():
        for b in R.STATES.values():
            if a.key < b.key:
                shared = set(a.placeholder_agent_ids) & set(b.placeholder_agent_ids)
                if shared:
                    all_problems.append(
                        f"[cross] {a.key} and {b.key} share placeholder id(s) {sorted(shared)} "
                        f"-- ids are panel-specific; one of these matches nothing")

    if warnings:
        print()
        print(f"{len(warnings)} WARNING(S) -- not failures, but worth closing:")
        for w in warnings:
            print(f"  ~ {w}")

    print()
    if all_problems:
        print(f"{len(all_problems)} PROBLEM(S):")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print(f"All {len(R.STATES)} registered states agree with their shipped data.py files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
