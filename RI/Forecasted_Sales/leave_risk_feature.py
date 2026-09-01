"""
Score every RIAR agent-quarter with a leave-risk probability, for use as a
CANDIDATE feature in the sales quantile model -- decision #7 in HANDOFF.md:
test it, don't assume it helps.

Reuses the Leave model's own pipeline (`../../RI/Likelihood_to_Leave
/modeling/data.py` + `crossval.py`) directly rather than reimplementing
feature engineering/label recoding -- same source file
(`leave-dataset (1).csv`), same RIAR filter, same mass-mover recoding.

Runs as a SEPARATE SUBPROCESS from inside the Leave project's own modeling
directory (temp script, deleted after) rather than importing it in-process
-- both projects have a module literally named `data` (and `train_baseline`,
which `crossval.py` also imports), so an in-process import would either
silently grab the wrong same-named module or require fragile sys.modules
aliasing. Running it as its own process with its own sys.path sidesteps the
collision entirely; the only interface between the two projects is the CSV
this writes out.

Leakage guard: the leave-classifier is trained ONLY on quarters inside the
sales model's TRAIN+CALIB window (through 2024-07-01). If we trained it on
the full history (through 2026-07-01) and then used it to score the sales
model's TEST window (2024-10-01..2025-07-01), the leave-risk feature for
TEST rows would carry information from AFTER the sales TEST window closes
-- invalidating the comparison. Training the classifier on a strict
subset of what the sales model itself is allowed to see keeps the
leave-risk feature honest for this experiment.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

LEAVE_MODELING_DIR = Path(__file__).resolve().parents[2] / "RI" / "Likelihood_to_Leave"
OUTPUT_PATH = Path(__file__).resolve().parent / "leave_risk_scores.csv"
TRAIN_CALIB_CUTOFF = "2024-07-01"  # inclusive -- matches sales model's TRAIN+CALIB end

_SCORER_SCRIPT = '''
import pandas as pd
from catboost import CatBoostClassifier
from data import load_clean
from crossval import TREE_FEATURES, MONOTONE

CUTOFF = pd.Timestamp("{cutoff}")
OUT = r"{out_path}"

df = load_clean()
fit_rows = df[df["snapshot_date"] <= CUTOFF]
X_fit = fit_rows[TREE_FEATURES].astype(float)
y_fit = fit_rows["label_left_3m"].to_numpy()
monotone = [MONOTONE.get(c, 0) for c in TREE_FEATURES]
model = CatBoostClassifier(
    iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=3.0,
    auto_class_weights="Balanced", monotone_constraints=monotone,
    loss_function="Logloss", eval_metric="PRAUC", verbose=False, allow_writing_files=False,
)
model.fit(X_fit, y_fit)
print(f"[leave_risk subprocess] classifier fit on {{len(fit_rows):,}} rows "
      f"({{fit_rows['snapshot_date'].min().date()}}..{{fit_rows['snapshot_date'].max().date()}}), "
      f"positive rate {{y_fit.mean():.3%}}")

X_all = df[TREE_FEATURES].astype(float)
df["leave_risk_score"] = model.predict_proba(X_all)[:, 1]
out = df[["mls_agent_id", "snapshot_date", "leave_risk_score"]]
out.to_csv(OUT, index=False)
print(f"[leave_risk subprocess] scored {{len(out):,}} agent-quarters -> {{OUT}}")
'''


def build_leave_risk_scores(python_exe: str = None) -> pd.DataFrame:
    if python_exe is None:
        python_exe = sys.executable

    script_path = LEAVE_MODELING_DIR / "_tmp_score_for_sales_model.py"
    script_path.write_text(
        _SCORER_SCRIPT.format(cutoff=TRAIN_CALIB_CUTOFF, out_path=str(OUTPUT_PATH).replace("\\", "\\\\")),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [python_exe, script_path.name],
            cwd=str(LEAVE_MODELING_DIR),
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("leave-risk scoring subprocess failed")
    finally:
        script_path.unlink(missing_ok=True)

    scores = pd.read_csv(OUTPUT_PATH, parse_dates=["snapshot_date"])
    return scores


if __name__ == "__main__":
    scores = build_leave_risk_scores()
    print(scores.describe())
