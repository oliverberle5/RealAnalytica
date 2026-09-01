# Onboarding a new state — protocol

**Audience: an AI agent or engineer adding a state to both production models
(Likelihood to Leave and Forecasted Sales) when new MLS data lands.**

Follow the steps in order. Each has a command, a pass condition, and a stop
condition. Do not skip a step because the previous one looked fine — three of
the four historical failures in this project passed every check anyone ran by
eye.

Python for every command below:

```powershell
$py = "C:\Users\olive\RA_Algorithms_White\Likelihood_to_Leave_Algorithm\.venv\Scripts\python.exe"
cd C:\Users\olive\RA_Algorithms_White
```

---

## The one thing to understand first

**A state is not a folder. A state is a value of `mls_code`.**

All states read the *same* source CSV
(`data/leave-dataset-with-team-distinction-mls-id.csv`)
and differ only by that filter. There is no per-state data file, and creating
one is the drift bug this infrastructure exists to prevent.

Two consequences that are not obvious and have both caused real damage:

1. **`agent_state` is not a state filter.** Every `market_*` and `state_*`
   column is keyed to the **MLS**, not the agent's home address. MLSPIN is only
   91.2% Massachusetts agents; RIAR is 82.2% Rhode Island, with 14.9% MA and
   2.8% CT. Splitting on `agent_state` yields a subset carrying *another
   market's* macro series. Split on `mls_code`, always.

2. **An identifier from one panel matches nothing in another, silently.** RIAR
   ids (`12345`, `00001`, `0014`) cannot appear under `mlspin`. Inheriting them
   excludes zero rows and raises zero errors.

---

## Known traps — all four have actually happened here

| # | What happened | Why nothing caught it | Which step below prevents it |
|---|---|---|---|
| 1 | MA inherited RI's placeholder id → excluded nothing | The constant was present and looked correct | 2, 4 |
| 2 | `--mls riar` overrode `MLS_CODE` but not the exclusion set → measured RI with all three buckets in, reported it as RI's accuracy for 3 weeks | One flag, two constants, no check that they agreed | 4 |
| 3 | Identifier swap landed in `data.py`, nothing re-scored → two shipped spreadsheets sat on a retired join key for 3 weeks | Every single-file check passed; the break existed only *between* files | 11, 12 |
| 4 | MA's office pairs, if copied from RI, would have classified nothing while appearing to work | A generated catalogue with zero matches looks identical to one with nothing to match | 6 |

Every one is the same bug: **a state-specific value living in more than one
place with nothing checking they agree.** `states/registry.py` is the one place;
`states/verify.py` is the check.

---

## Step 0 — Is the data actually here?

```powershell
& $py -m states.discover --mls-code <new_code>
```

**PASS:** the panel prints with row/agent/quarter counts.

**STOP** if it reports the code is absent. The state cannot be modelled. Do not
substitute an `agent_state` filter — see above. Record the finding the way
`CT_EXPANSION_ASSESSMENT.md` does and stop. *(As of 2026-09-01 the file holds
exactly two panels: `riar` and `mlspin`. Connecticut is not in it.)*

---

## Step 1 — Read the discovery report

The same command prints a placeholder-candidate table and a draft `StateSpec`.

Sanity-check the panel before going further:

- **< ~5,000 agents or < 8 quarters?** Smaller than RI, the smaller production
  panel. Forward-chaining validation may not have enough folds. Flag it.
- **Quarter range much shorter than 2021-10-01..2026-07-01?** The Sales model
  needs resolved 12-month-forward targets; a short panel loses most of them.

---

## Step 2 — Confirm the placeholders **by hand**

The discovery table nominates; it never decides. Two channels: `S` structural
(persistent, non-team, side-degenerate, top-0.1% by size), `N` nomenclature
(name matches the bucket vocabulary).

For each candidate, decide **bucket or person**:

| Evidence | Bucket | Person |
|---|---|---|
| `deg` (never both lists and sells) | ~1.00 | < 0.98 |
| `cov` with large size | ~1.00 | varies |
| `agent_name` | "Non Member", "Assisted Sale", "ALLIANCE MEMBER" | a human name |

A large candidate with `deg < 0.98` is almost certainly a **real top producer**
— leave it in.

> **This cuts both ways.** Leaving a bucket in puts a non-person with ~$1.1B of
> trailing volume into training as one extraordinary agent, and it gets scored
> and shipped (RI's `00001` was delivered at $51,688,219, rank 11 of 4,607).
> Excluding a real agent deletes them from **both** models. Neither error
> raises anything. Confirm each id individually.

**PASS:** you can state, per id, why it is not a person.

---

## Step 3 — Register the state

Paste the draft into `states/registry.py`, then fill in by hand:

- `key`, `display_name`, `mls_name`, `dir_name`
- `placeholder_agent_ids` — **only the ids you confirmed in Step 2.** The draft
  ships this empty on purpose.
- `office_brand_categories` — the draft lists brands present in this panel but
  absent from RI's list. **These must be added**, or the Leave model one-hots
  them to all-zero and every agent at those brands looks brand-less. *(The
  Sales model builds its brand dummies dynamically and needs no change — this
  asymmetry is real and is why the Leave model is the one that breaks.)*
- `provisional=True` — leave it until Step 13.
- `notes` — **what you confirmed, and what you did not.**

---

## Step 4 — Verify the registry

```powershell
& $py -m states.verify
```

Checks every registered state's `data.py` against the registry, and cross-checks
that no two states share an `mls_code` or a placeholder id.

**PASS:** exit 0. **STOP** on any FAIL — a disagreement here is trap #1 or #2
in progress. *(A known WARNING is expected: RI's Sales `data.py` inlines its
panel filter as a literal instead of naming `MLS_CODE`. Correct today, invisible
to tooling.)*

---

## Step 5 — Scaffold both models

```powershell
& $py -m states.scaffold --state <key>
```

Generates `<DIR>/Likelihood_to_Leave/` and `<DIR>/Forecasted_Sales/` from the
**MA** reference port (MA, not RI: MA is what RI looks like after being made
portable), rewriting constants from the registry. Every patch is asserted — if
the template changed shape, it aborts rather than writing a file that silently
kept Massachusetts' values.

Two files are deliberately **not** generated:

- `rebrand_classifications.py` — generated per state in Step 6. Copying the
  template's is trap #4.
- `backtest_ma_production.py` — superseded 2026-08-22 by `remeasure_accuracy.py`.

**Every generated file carries a banner** saying its constants were rewritten
but its **prose was not**. Docstring figures (base rates, AUCs, brand shares)
were measured on Massachusetts. Treat every one as unverified until you replace
it. Do not quote them as this state's numbers.

Re-run `& $py -m states.verify` after scaffolding. **PASS:** exit 0.

---

## Step 6 — Generate the office-pair catalogue (Leave model only)

```powershell
cd <DIR>\Likelihood_to_Leave
& $py derive_office_pairs.py
```

Not every office switch is a resignation — rebrands, acquisitions and internal
code consolidations sweep whole teams and are not individual attrition.

- **Tier A** — origin and destination carry an identical office name after
  normalization. An internal code change. Recodes in any quarter.
- **Tier B** — ≥5 agents moved together in one quarter, names not identical. No
  independent evidence, so **only those specific quarters** recode.

**PASS:** `rebrand_classifications.py` and `office_pair_candidates.csv` exist,
and Tier A is non-trivial.

**INVESTIGATE if Tier A is 0.** That likely means office names do not repeat
across codes in this panel — plausible, but it is also exactly what trap #4
looks like. Confirm by checking that `office_name` is populated at all.

> **Known gap, inherited by every generated state:** no pair is web-verified, so
> this state gets no equivalent of RI's confirmed-M&A handling, where a 3-mover
> trickle in an adjacent quarter still recodes. Generated states are **stricter**
> and will under-catch trickle moves around real acquisitions. That errs toward
> counting a structural move as churn — conservative for a churn model — but it
> is a genuine gap. Say so in the README; do not claim parity with RI.

---

## Step 7 — Sanity-check both loaders

```powershell
cd <DIR>\Likelihood_to_Leave  ; & $py data.py
cd ..\Forecasted_Sales        ; & $py data.py
```

**PASS:** both print row counts and a plausible base rate. Confirm the
placeholder exclusion actually fired — look for a line like
`[data] excluded N non-MLS-member placeholder rows` with **N > 0**.

**STOP if N == 0** while `placeholder_agent_ids` is non-empty. That is trap #1,
caught. The ids do not match this panel.

---

## Step 8 — Leave model: measure

```powershell
cd <DIR>\Likelihood_to_Leave
& $py crossval.py
```

Forward-chaining (expanding-window) validation, logistic / XGBoost / CatBoost.

**PASS:** CatBoost mean AUC in the same neighbourhood as RI (0.7440) and MA
(0.7529). Report **per-fold**, not just the mean.

**Interpretation rules — these are not optional:**

- The documented 95% CI on 5-fold mean AUC is **≈ ±0.019**, and that understates
  it because folds share training windows and agents. **A gap smaller than that
  is not a result.** MA's XGBoost "beat" CatBoost by 0.0013 and CatBoost was
  correctly kept.
- Keep **CatBoost** as production for cross-state consistency unless another
  model wins by more than the noise band.
- Do **not** re-derive the feature list. MA tried; its own importance ranking
  lost on 5 of 5 folds. One shared list means future wins transfer between
  states. CatBoost feature importance is not predictive contribution.

---

## Step 9 — Leave model: score

```powershell
& $py score_agents.py
```

**PASS:** `current_agent_risk_scores.csv` written. Check tier shares are
plausible (MA: 84.5/13.3/1.5/0.7). Cutpoints are multiples of the base rate, so
they move with the state automatically — do not hand-tune them.

Expect a **bias correction** ≠ 1.0 if recent churn differs from the panel
average (MA's was 0.726x). That is the calibrator working, not a bug.

---

## Step 10 — Sales model: measure

```powershell
cd ..\Forecasted_Sales
& $py remeasure_accuracy.py
```

3 seeds, held-out final 2 quarters, naive-persistence baseline on every point
metric. Slow (~15 min on a 600k-row panel).

**PASS:** it completes and beats naive persistence on **MdAPE, MAE, Spearman
and capture** on both targets.

**Interpretation rules:**

- **R² may lose to persistence. That is expected and is not a defect.** MA loses
  on volume R² (0.6408 vs 0.7324); RI loses on units R² (0.7568 vs 0.7907). R²
  is squared error dominated by the largest producers, and the pipeline shrinks.
  Each state loses one target.
- **Never quote R² without its baseline.** Alone it invites the opposite of the
  correct conclusion.
- **Absolute MAE is not comparable across states** (MA average sale price $794K
  vs RI $625K). Use MdAPE, coverage, capture, and the model-vs-persistence ratio.
- Check interval coverage at all six levels. Volume bands should land within
  ~1pt of nominal. **Units bands over-cover at low levels in both states**
  (integer discreteness) — expected, but do not describe a 50% units band as a
  50% band.

---

## Step 11 — Sales model: score

```powershell
& $py score_agents.py
```

**PASS:** `current_agent_sales_forecast.csv` written, **33-column schema
identical to RI's** — downstream consumers rely on it.

> **Critical asymmetry between the two models.** `score_agents.py::build_full_frame`
> **sorts on the agent key**, and both learners subsample by row order, so
> **changing the key reorders every training row and fits a different model.**
> A fixed `random_state` does not protect you — what the seed indexes into moved.
> The Leave model does *not* have this property (`load_clean` never sorts on the
> key), so it can simply be rescored.
>
> **Therefore: if the identifier ever changes, re-key the Sales file by
> crosswalk join — do not refit.** RI measured a pure re-key moving 100% of
> agents by a median 1.48% and the book total by −1.23%, and concluded re-keying
> is not a reason to refit.

---

## Step 12 — Establish the recalibration baseline

```powershell
cd <DIR>
& $py quarterly_recalibration.py --adopt-current
& $py test_recalibration_gates.py
```

`--adopt-current` runs every output check against the files you just shipped and
records them as the baseline **without refitting** — "establish a baseline" and
"refit" are different requests, and the Sales model is not invariant to a rerun.
The row is marked `adopted: true` so the log can never imply a fit that did not
happen.

`test_recalibration_gates.py` fault-injects each defect and asserts the matching
check fires. **PASS:** all cases caught, clean output raises nothing. A check
that has only ever seen good input is a line of code nobody has watched do its job.

---

## Step 13 — Write the README, then clear `provisional`

Write `<DIR>/README.md` covering, at minimum:

- Where the data comes from, and the `mls_code` filter
- **What differs from the template**, each item data-driven
- Leave results: per-fold AUC vs RI/MA, base rate, tier distribution
- Sales results: the **whole-model** table with persistence on every line,
  and interval coverage at all six levels
- **What is weaker than RI** — at minimum the un-web-verified office pairs
- What was *not* measured

Then set `provisional=False` in the registry and re-run `states.verify`.

**Honesty rules for the write-up:**

- Report per-fold, not just means. State the noise band.
- Never state a cross-state comparison as "better" when the gap is inside the
  band — MA leads RI by +0.0089 mean AUC but *loses folds 3 and 5*.
- A metric measured under a superseded framing is **unrestated**, not "roughly
  the same". Say so plainly, as MA's README did for three weeks.

---

## Ongoing operation — every quarter

**Use the wrapper. Never the individual scoring scripts.**

```powershell
cd <DIR>
& $py quarterly_recalibration.py --check     # exit 0 = due, 1 = not
& $py quarterly_recalibration.py --dry-run   # full rehearsal, nothing promoted
& $py quarterly_recalibration.py             # the real thing
```

A run is due when the data holds a newer quarter **or the source file changed**.
That second trigger is what would have caught trap #3 — verified directly by
rewinding the state file to the pre-swap source with the quarter unchanged,
which correctly reports `DUE — source file changed`.

The wrapper drives **both** models, gates their output, and runs a **cross-model
consistency check** (same key, roster and snapshot). That check exists because
in trap #3 every single-file check passed the whole time; what was wrong existed
only *between* the two files.

Re-run `& $py -m states.verify` after touching any `data.py` or the registry.

---

## Quick reference

| Command | Purpose |
|---|---|
| `python -m states.discover --mls-code <code>` | Profile a panel, nominate placeholders, draft a StateSpec |
| `python -m states.discover --self-test` | Prove the detector still recovers known placeholders |
| `python -m states.verify` | Registry ↔ every shipped `data.py` agree |
| `python -m states.scaffold --state <key>` | Generate both model directories |
| `<DIR>/quarterly_recalibration.py --check` | Is a run due? |
| `<DIR>/quarterly_recalibration.py --adopt-current` | Baseline the shipped files, score nothing |

**Registered states:** `ri` (riar, Rhode Island — the original, irregular
layout), `ma` (mlspin, Massachusetts — the portable template).
