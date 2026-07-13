# autoresearch: agent likelihood-to-leave model

Adapted from Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern
(`original/` in this folder — kept for reference, not touched). That repo's actual
mechanics (GPU pretraining, `val_bpb`, 5-minute time budget) don't apply here — this is
tabular gradient-boosting on ~84k rows, not LLM pretraining. What's borrowed is the
*operating loop*: one file the agent freely edits, a fixed ground-truth layer it can't
touch, one commit per experiment, a results log, keep-if-better/revert-if-worse, run
until interrupted.

**Read `../modeling/HANDOFF.md` in full before touching anything.** It documents years
of label-hygiene work (censoring rule, mass-mover/M&A recoding, horizon choice) that is
the actual reason AUC went from ~0.63 to ~0.74 — much more load-bearing than any model
choice. This program is about the *last mile* on top of that: which model family, which
hyperparameters, which additional derived features, squeezed as far as they'll go
without undoing that work.

## Goal

Maximize held-out AUC (mean across the 5 forward-chaining folds used in
`../modeling/crossval.py`), **without regressing fold stability** (see "Objective
function" below) and without violating any of the fixed assumptions in `../modeling/
HANDOFF.md`. Current production baseline: **CatBoost, 0.7368 mean AUC** (round 4).

Explicitly in scope, per the user: **try a lot of model families**, not just
tune CatBoost further. Logistic/regularized-regression variants, LightGBM, sklearn's
`HistGradientBoostingClassifier`, ExtraTrees, CatBoost, XGBoost, and blends/stacks of
the above are all fair game. So is new feature engineering, as long as every new
feature is derivable from columns that already exist in the row as of the snapshot
date (no future information, no re-deriving something `data.py` already excludes for a
documented reason).

## Setup

1. **Agree on a run tag** with the user: propose one from today's date (e.g. `jul11`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch** from current `master`/main working state of the
   `Likelihood to Leave Algorithm` repo (check what VCS root actually applies here first
   — this folder may not be its own git repo; confirm before assuming `git checkout -b`
   works as-is).
3. **Read the in-scope files** fully:
   - `../modeling/HANDOFF.md` — full project history and rationale (see above).
   - `../modeling/data.py` — the one shared loader. **Do not modify.**
   - `../modeling/rebrand_classifications.py` — the M&A/mass-mover lookup table.
     **Do not modify** (extending it requires new web-research verification, which is
     out of scope for an autonomous loop).
   - `../modeling/crossval.py` — read for the fold construction (`_forward_folds`),
     feature list (`TREE_FEATURES`), and monotonic constraints (`MONOTONE`). You may
     import from this file, but treat its existing model-fitting functions as a
     reference implementation, not something to hack in place — do your experimentation
     in the new file below instead.
   - `../modeling/train_baseline.py` — logistic design-matrix builder (tenure/career
     binning), if you want to experiment with regression variants.
4. **Verify environment**: `../.venv/Scripts/python.exe` should already have pandas,
   numpy, scikit-learn, xgboost, catboost. LightGBM is NOT currently installed — if you
   want to try it, `../.venv/Scripts/pip.exe install lightgbm` once during setup (this
   is the one exception to "no new dependencies," since testing a light/fast GBM
   alternative is an explicit goal here — confirm it installs cleanly on this machine
   before relying on it).
5. **Initialize `results.tsv`** in this folder with just the header row (see format
   below). Do not commit this file — add it to this folder's `.gitignore` or otherwise
   leave it untracked, same as the original repo's convention.
6. **Create `experiment.py`** in this folder (NOT inside `../modeling/`) as the one file
   you'll iterate on. It should import `load_clean`, `RAW_FEATURE_COLUMNS` from
   `../modeling/data.py` (add `../modeling` to `sys.path`, don't copy-paste the loader)
   and reimplement the fold loop / model registry so you can swap in new model classes
   and features freely without disturbing `crossval.py`, which stays as the
   human-owned reference comparison.
7. **Confirm setup with the user before the first run.**

## Fixed ground truth — do NOT modify or work around

These mirror the "critical decisions already made" section of `HANDOFF.md`. If an idea
requires changing one of these to show a gain, the gain is fake — discard the idea, do
not weaken the guardrail.

- **3-month horizon** (`label_left_3m`, `HORIZON_DAYS = 91`). Do not re-derive a
  different horizon.
- **Censoring rule**: rows without a resolvable outcome are already dropped by
  `load_clean()`. Do not include unresolvable rows to get more training data.
- **Mass-mover / M&A recoding** (`_recode_mass_mover_moves`, `CONFIRMED_NOT_CHURN_PAIRS`,
  `CLUSTER_GATED_PAIRS`): already applied inside `load_clean()`. Do not bypass it by
  reading the raw CSV directly.
- **`heuristic_recruitability_score`**: stays excluded. Do not add it back as a
  feature or benchmark.
- **`avg_dom_12m` (per-agent, raw)**: stays excluded (confirmed broken — 59% zeros).
  `market_avg_dom_12m` (state-level) is the legitimate substitute and is already in
  `RAW_FEATURE_COLUMNS`. You MAY experiment with a lagged quarter-over-quarter *trend*
  of `avg_dom_12m` on the non-zero subset if curious (see `explore_avg_dom.py` for
  prior art — it found suggestive but inconclusive signal on only ~1,600 usable rows),
  but do not put the raw broken column back into the main feature set.
- **Time-based (forward-chaining) splits only.** NEVER random k-fold, NEVER
  `train_test_split(shuffle=True)`. Same agents recur across quarters; a random split
  leaks future information into the past and will silently inflate every metric you
  report. Reuse `_forward_folds` from `crossval.py` (or an equivalent expanding-window
  scheme) for every single model you test.
- **Fold structure itself is fixed — not a tunable lever.** Use `crossval.py`'s exact
  `_forward_folds(dates, start_train=9, test_block=2)` (5 folds) for every experiment.
  Do not change `start_train`, `test_block`, or the number of folds to chase a better
  AUC number. This isn't a modeling decision, it's a measurement decision — changing it
  changes what's being measured, not how good the model actually is, and with only
  ~800-2,000 positives per fold already, smaller test blocks make individual fold AUCs
  noisier (more folds does not mean a more trustworthy mean), while larger ones lose
  the stability checkpoints that caught the fold-5 rebrand-contamination bug in the
  first place. It also breaks comparability with every number already in
  `HANDOFF.md` (0.7368 CatBoost, etc.) — every experiment here needs to be measured on
  the identical fold structure those numbers came from, or "improved AUC" is meaningless.
- **Class imbalance via class-weighting only** (`scale_pos_weight`,
  `class_weight="balanced"`, `auto_class_weights="Balanced"`). Do NOT use SMOTE or any
  synthetic oversampling — rejected previously because it interpolates between
  unrelated agent-quarters, producing physically meaningless synthetic rows. This
  applies to every new model you try, including LightGBM/HistGBM.
- **No leakage from labels or post-outcome columns.** If you engineer a new feature,
  trace it back to source columns and confirm it uses only data knowable as of the
  snapshot date. `data.py`'s module docstring explains the exact reasoning to reuse for
  this check.
- Comparisons should isolate one change at a time where practical (algorithm vs.
  feature engineering vs. hyperparameters) — mirrors how `crossval.py` gives every model
  the same feature set today, so gains are attributable.

## What you CAN do (the actual research surface)

- **Try different model families** on the same fold structure and feature set as a
  starting point, then branch out:
  - LightGBM (`LGBMClassifier`) — previously ruled out by reasoning alone (leaf-wise
    growth considered overfitting-prone on a ~1-2k-positive minority class), never
    actually benchmarked. Worth actually running now.
  - `sklearn.ensemble.HistGradientBoostingClassifier` — free (already installed),
    natively handles NaNs and monotonic constraints (`monotonic_cst` param), worth a
    direct comparison to CatBoost/XGBoost.
  - Regularized regression variants beyond the plain logistic baseline: elastic-net
    penalty (`penalty="elasticnet", solver="saga"`), or feeding it the same continuous
    tenure/career treatment but with interaction terms.
  - CatBoost/XGBoost with different depth/learning-rate/regularization combinations
    beyond what's already in `crossval.py`.
  - Blends/stacks: e.g. average of CatBoost + XGBoost rank scores, or a small logistic
    meta-model on top of the tree models' out-of-fold scores.
- **New engineered features** derived only from existing raw columns (same spirit as
  `add_derived_ratios` in `data.py`) — e.g. additional ratio/trend constructions,
  interaction terms, or target/frequency encodings of `office_brand` /
  `cohort_volume_quartile` as alternatives to one-hot.
- **Feature pruning** — CatBoost feature importance already ranks ~94 features; test
  whether dropping the bottom N changes AUC (regularization-by-subtraction), consistent
  with the project's stated "simpler is better, all else equal" ethos.
- **Hyperparameter search** — grid/random search is fine as long as each individual
  fold-fit stays fast (see budget below) and you're not fitting on the test fold to
  pick hyperparameters (use an inner time-based split within the training window, or a
  held-out validation slice, never the actual test fold).

## What you CANNOT do

- Modify `../modeling/data.py`, `../modeling/rebrand_classifications.py`, or
  `../modeling/detect_transitions.py`.
- Modify `../modeling/crossval.py` in place — it's the human-owned reference
  comparison; do your work in `experiment.py` instead. (You may read it freely.)
- Introduce a data source other than
  `../leave-dataset_most_recent_updated.csv`.
- Use random k-fold / shuffled splits anywhere.
- Use synthetic oversampling (SMOTE etc.).
- Touch `../modeling/calibrate.py` or the production calibration/quintile logic — this
  program is scoped to ranking quality (AUC) only; calibration is a separate,
  already-solved concern (bias-correction patch, see HANDOFF.md) that this loop should
  not disturb.
- Install dependencies beyond what's already present, except LightGBM (explicitly
  allowed above, one-time, during setup).

## Objective function: what counts as "improved"

Mean AUC alone is not enough — the handoff is explicit that fold *stability* matters as
much as the mean, since one bad fold historically meant a real bug (the fold-5 rebrand
contamination), not noise. So:

- Report **mean AUC across all 5 folds AND the worst single fold's AUC**, every run.
- **Keep** a change only if mean AUC improves by more than **+0.001** (not a rounding
  blip) AND the worst fold does not drop by more than **0.01** versus the current
  best-kept commit. A change that raises the mean but craters one fold is exactly the
  "fold-5 collapse" pattern the project already burned time diagnosing once — treat a
  new instance of it as a red flag to investigate (is there a new label-contamination
  issue?), not something to shrug off as "noisy fold."
- **Discard** (git reset) anything that doesn't clear both bars.
- If a model choice is a genuine tie with production CatBoost on AUC but is meaningfully
  simpler or faster to retrain, that's a legitimate "keep" too — same simplicity
  criterion as the original autoresearch program.md ("equal or better results with less
  complexity is a great outcome").

## Hardware note (GPU)

This machine has a GTX 1650 Ti. It doesn't matter here, and no setup or adjustment is
needed — this is a completely different regime from Karpathy's original repo, which
*requires* a GPU because it's training a neural net on hundreds of millions of tokens.
CatBoost/XGBoost/LightGBM/HistGBM all default to CPU in this codebase (no
`task_type="GPU"` / `tree_method="gpu_hist"` set anywhere), and at 84k rows x ~94
features the dataset is small enough that GPU kernel-launch and data-transfer overhead
typically isn't repaid — it's exactly why fits are already fast on CPU (1.4-7s/fold).
Stay on CPU for every experiment. If you're curious enough to try
`task_type="GPU"` (CatBoost) or `tree_method="gpu_hist"` / `device="cuda"` (XGBoost) as
one specific one-off experiment, treat it as a minor curiosity with the 1650 Ti's 4GB
VRAM as a hard ceiling, and expect no AUC change and no reliable speedup — do not spend
more than one experiment slot confirming this.

## Budget: hard 5-minute cap per experiment

Current models fit in ~1.4s (XGBoost) to ~7s (CatBoost) per fold, so a full 5-fold
experiment should normally finish in well under a minute. There's no reason to allow a
long-running process here, unlike GPU pretraining where a fixed 5-minute budget is the
whole point of the design — here, a run taking anywhere close to 5 minutes almost
certainly means something is wrong (an accidental full grid search, a pathological
hyperparameter, a hung fit), not a legitimately expensive experiment.

- If a single experiment (all 5 folds, one model config) exceeds **5 minutes** wall
  clock, kill it, log it as a `crash` in `results.tsv` with a description noting the
  timeout, and move on. Do not raise this cap to accommodate a slow idea — scope the
  idea down instead (smaller grid, fewer boosting rounds, etc.) so it fits comfortably
  inside the cap.
- If you want to grid/random-search hyperparameters, keep each individual point on the
  grid fast and bound the total grid size so the whole sweep still clears the 5-minute
  cap — do not queue a sweep that takes multiple experiment-loop iterations to finish
  as if it were one experiment.

## Logging results

`results.tsv` (tab-separated, NOT comma — commas break in descriptions), header + 5
columns:

```
commit	mean_auc	worst_fold_auc	status	description
```

1. git commit hash (short, 7 chars)
2. mean AUC across all 5 folds (e.g. `0.736800`) — `0.000000` for crashes
3. worst single fold AUC (e.g. `0.702000`) — `0.000000` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried (model family + the specific change)

Example:

```
commit	mean_auc	worst_fold_auc	status	description
a1b2c3d	0.736800	0.702000	keep	baseline: catboost round 4 (production config, ported to experiment.py)
b2c3d4e	0.741200	0.708000	keep	lightgbm, dart boosting, depth 4, same monotone constraints
c3d4e5f	0.742000	0.671000	discard	histgbm, deeper trees (depth 8) -- worst fold dropped >0.01, investigate before retrying
d4e5f6g	0.000000	0.000000	crash	catboost + elasticnet stacking meta-model (shape mismatch in OOF assembly)
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit).
2. Modify `experiment.py` with one experimental idea — a model family swap, a
   hyperparameter change, a new engineered feature, a feature-pruning pass, or a
   blend/stack. Prefer one isolated change per experiment so results are attributable,
   per the "no relitigating fixed ground truth" and "isolate one change" notes above.
3. `git commit` (in whatever repo root actually applies to this folder — confirm during
   setup).
4. Run it: `../.venv/Scripts/python.exe experiment.py > run.log 2>&1` (redirect
   everything — do not let output flood your context; grep it out afterward).
5. Read results: `grep "^mean_auc:\|^worst_fold_auc:" run.log` (make `experiment.py`
   print a summary block in this format, mirroring the original repo's `train.py`
   output convention, so this grep works).
6. If the grep is empty, the run crashed — `tail -n 50 run.log` for the traceback. Fix
   it if it's a quick/obvious bug; if the underlying idea is fundamentally broken, log
   `crash` and move on.
7. Record to `results.tsv` (per the objective function above).
8. If kept: the branch advances, keep the commit. If discarded: `git reset` back to the
   prior kept commit.
9. Never stop to ask "should I keep going?" once the loop has started — the user may be
   away from the computer. Keep iterating until manually interrupted. If you run out of
   ideas, re-read `HANDOFF.md`'s "Not yet done" section, revisit near-misses with a
   variation, or try combining two previously-kept ideas.

## Suggested starting order (not mandatory, just a reasonable sequence)

1. Port the current production CatBoost config into `experiment.py` verbatim as commit
   0 / baseline — confirms your harness reproduces 0.7368 before changing anything.
2. LightGBM, same feature set, same monotonic constraints, same class-weighting —
   directly resolves the "never actually benchmarked" gap flagged in HANDOFF.md.
3. `HistGradientBoostingClassifier` — same idea, zero new dependencies.
4. Hyperparameter sweeps on whichever of the above is currently winning.
5. Feature-set experiments (pruning, new ratios, encoding changes) on the
   current-best model.
6. Blends/stacks across the best 2-3 models once each has been individually tuned.

## Not in scope for this loop (flagged in HANDOFF.md, deliberately excluded here)

- SHAP explainability, the quarterly retrain/rebias pipeline, and segment-level
  calibration checks are separate, already-scoped workstreams — don't let the loop
  wander into them.
