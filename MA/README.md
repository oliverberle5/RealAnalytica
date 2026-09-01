# Massachusetts (MLSPIN) models

Massachusetts ports of the two production models that previously ran only on
Rhode Island:

| model | folder | what it predicts |
|---|---|---|
| Likelihood to Leave | `Likelihood_to_Leave/` | each agent's chance of switching brokerages in the next 3 months |
| Forecasted Sales | `Forecasted_Sales/` | each agent's next-12-month dollar volume and unit count, as a calibrated range |

**When new data lands, run `quarterly_recalibration.py` — not the individual
scoring scripts.** It drives both models, gates their output, checks that the
two files still agree with each other, and logs what the calibration did. See
[Running it](#running-it). It exists because on 2026-08-10 an identifier change
landed in both `data.py` files, nothing re-ran the scoring, and both shipped
spreadsheets sat on a retired key for three weeks while every number in them
looked fine.

Connecticut was assessed and deferred — the dataset contains no CT market, and
this was re-verified against the current file on 2026-08-30 with no change. See
`../CT_EXPANSION_ASSESSMENT.md` for the full finding and what data would unblock it.

---

## Where the data comes from

Both models read the **same** file the RI models read:

```
Likelihood_to_Leave_Algorithm/leave-dataset-with-team-distinction.csv
```

That file is not copied into `MA/`. It is 332 MB and already exists twice in the
workspace; a third copy would add nothing but drift risk. Both MA `data.py`
files reference it by relative path.

The file bundles two MLS panels in one export, distinguished by `mls_code`:

| `mls_code` | market | rows | agents |
|---|---|---|---|
| `mlspin` | **Massachusetts** | 612,358 | 45,186 |
| `riar` | Rhode Island | 84,417 | 5,625 |

So "the MA model" is the same pipeline over `mls_code == "mlspin"`. MA is roughly
**7x the rows and 8x the agents** of RI, with ~11,400 three-month leave events
versus RI's ~2,200.

> **Important:** every `market_*` and `state_*` column is keyed to the **MLS**,
> not to the agent's home address. An agent living in NH or CT but belonging to
> MLSPIN carries Massachusetts market and macro figures. This is why a
> `agent_state`-based split does not produce a state model — see the CT
> assessment.

---

## What actually differs from the RI code

The port deliberately changed as little as possible, so RI and MA results are
methodologically comparable. Modeling logic — the label construction, censoring
rule, forward-chaining validation, team exclusion, hurdle + conformal quantile
machinery — is **byte-identical** to RI.

Five things differ, each data-driven and marked inline in the source:

1. **`MLS_CODE`** — `"riar"` → `"mlspin"`, in both `data.py` files.
2. **`DATA_PATH`** — points at the RI project's CSV instead of a local copy.
3. **`NON_MLS_MEMBER_AGENT_IDS`** — MLSPIN has its own synthetic placeholder
   record ("Non Member", `0d2b79f7-…`), a buyer-side catch-all bucket that is not
   a person. The RI placeholder's id appears nowhere in MLSPIN, so inheriting it
   would have silently excluded nothing. The MA record was identified by matching
   the RI one's structural signature: present in all 20 snapshots, `is_team == 0`,
   **zero** list-sides ever, 1,678 units all buy-side in the latest quarter,
   monotonically incrementing `career_months`, and — coincidentally identical to
   RI — 1.16% of market volume.
4. **`OFFICE_BRAND_CATEGORIES`** — MLSPIN contains `bhgre` (Better Homes &
   Gardens), absent from RIAR. Two brands also carry very different weight and
   are worth remembering when comparing feature importances across states:
   `berkshire_hathaway` is 3.40% of MA rows vs 0.01% of RI, and `remax` is 4.32%
   vs 10.50%. (The Sales model builds its brand one-hots dynamically via
   `get_dummies`, so it picked `bhgre` up automatically; the Leave model's list is
   hardcoded and was extended by hand.)
5. **`rebrand_classifications.py`** — see below. This is the one place the MA
   model is currently *weaker* than RI.

---

## The office-pair classification gap (known, documented)

Not every office switch is a resignation. Rebrands, acquisitions and internal
office-code consolidations sweep whole teams along and are not individual
attrition. The RI project handles this with a hand-curated, web-verified list of
`(origin_office, dest_office)` pairs.

**Those pairs are RIAR office codes and do not exist in MLSPIN.** Importing the
RI file into the MA model would have classified nothing while appearing to work
— every structural MA move would have counted as real churn.

`derive_ma_office_pairs.py` regenerates the list for MA using only mechanisms
that need no web research, reproducing the RI project's own two-gate structure:

- **Tier A → `CONFIRMED_NOT_CHURN_PAIRS`** (735 pairs). Origin and destination
  office codes carry an **identical** office name after normalization. This is
  the exact zero-cost reasoning the RI project already documented for its own
  internal code changes (`LVGP`→`LVGP02`, `CBRB23`→`CBRB24`, …). Every move on
  the pair recodes to not-churn, in any quarter.
- **Tier B → `CLUSTER_GATED_PAIRS`** (111 pairs, 146 pair-quarter instances).
  Pairs that moved ≥5 agents together in one quarter and are *not* name-identical.
  No independent evidence either way, which is precisely the population this gate
  was designed for: only the specific quarters that independently clear the
  5-mover threshold recode. A lone mover on the pair in another quarter stays churn.

This recoded **2,810 rows** from "left" to "stayed".

**What MA does not have:** no MA pair has been web-verified, so MA gets no
equivalent of RI's confirmed-M&A handling, where a 3-mover trickle in an adjacent
quarter still recodes because the pair is independently known to be one event. MA
is therefore *stricter*, and will under-catch trickle moves around real MA
acquisitions. This errs toward counting a structural move as churn rather than
excusing real attrition — the conservative direction for a churn model — but it
is a genuine gap, not a finished job.

`ma_office_pair_candidates.csv` is the review queue, sorted so the most likely
promotions come first. A `name_containment` flag marks pairs where one office
name wholly contains the other; these are strong prima-facie rebrands and are the
obvious next thing to confirm:

| origin | destination | movers |
|---|---|---|
| Custom Home Realty, Inc. | Century 21 Custom Home Realty | 20 |
| Bentley's | RE/MAX Bentley's | 13 |
| Boston City Properties | Amo Realty - Boston City Properties | 12 |
| Home And Key Real Estate, LLC | LPT Realty - Home & Key Group | 12 |
| Coldwell Banker Realty - Andover | Coldwell Banker Realty - Andovers/Readings Regional | 20 |

Confirming these by hand and promoting them into Tier A is the highest-value
follow-up available on the MA leave model.

---

## Results — Likelihood to Leave

Forward-chaining (expanding-window) validation, 5 folds, 2 test quarters each —
the same `crossval.py` used for RI. **RI was re-run on current code for this
comparison** rather than quoting its README, so both columns come from identical
code, identical fold windows and the same source file. The only difference is the
`mls_code` filter.

| model | MA mean AUC | RI mean AUC |
|---|---|---|
| Logistic regression | 0.7015 | 0.6910 |
| XGBoost | **0.7542** | 0.7279 |
| CatBoost *(production)* | 0.7529 | **0.7440** |

MA CatBoost per-fold: `0.7542  0.7755  0.7253  0.7534  0.7563` (mean 0.7529, worst 0.7253)
RI CatBoost per-fold: `0.7218  0.7369  0.7745  0.7167  0.7701` (mean 0.7440, worst 0.7167)

The RI column was re-measured 2026-08-11 after three synthetic placeholder records
(`12345`, `00001`, `0014`) were excluded from that panel; the MA column is from the
original paired run. Those IDs are RIAR records and were never in the MA panel, but MA has
not been re-checked for placeholders of its own — see the note at the end of this file.

Three things worth stating plainly:

- **MA is not uniformly better.** It leads on mean AUC by +0.0089 and wins 3 of
  the 5 folds, but RI wins folds 3 and 5 — and fold 3 is RI's best (0.775) and
  MA's worst (0.725). These are different markets with different quarter-to-quarter
  dynamics, so this is *not* an apples-to-apples "the MA model is better" claim.
  The honest reading is that the MA port achieves comparable-to-better
  discrimination on its own market, with a materially more stable estimate behind
  it (7x the rows, ~4.5x the leave events).
- **XGBoost and CatBoost are tied in MA** (0.7542 vs 0.7529, a 0.0013 gap). The RI
  project documents a 95% CI of roughly ±0.019 on 5-fold mean AUC, and notes that
  understates it because folds share training windows and agents. A 0.0013
  difference is far inside that band and is **not** evidence XGBoost is better.
  CatBoost is kept as the production model for consistency with RI — not because
  it won here, because it didn't meaningfully. Note this *is* a change from RI,
  where CatBoost led XGBoost by a clearer 0.0161.
- **RI's 0.7440 differs from the 0.7434 in its `champion.json`.** Two deliberate
  changes sit between them, in opposite directions: the 2026-07-28
  `office_arrival_rate_12m` / `company_arrival_rate_12m` correctness fix cost
  ~0.3pp and was adopted anyway for defensibility, taking it to 0.7411; the
  2026-08-10 exclusion of three synthetic placeholder records (not people, so no
  departure is defined for them) took it to 0.7440. 0.7440 is the current-code
  number and the right reference.

MA base positive rate after label cleanup is **1.75%** (9,737 leave events in
556,365 usable agent-quarters), versus RI's 1.96%.

### Feature list: re-derived for MA, then rejected — a logged negative result

RI's production feature list is the bottom-importance features pruned away
(**54 of 99 kept** — verified by import 2026-08-30; the "49 of 94" written here
originally was a stale count from an earlier feature pool), ranked by a CatBoost
fit on fold 1's training window. That
ranking is a property of *RIAR* data, so it was not assumed to transfer.
`derive_ma_config.py` re-ran the identical pruning procedure on MA's own fold-1
window and measured both lists head-to-head on the same folds.

The two lists diverge a lot — only **34 features overlap**. MA's own ranking
prunes all five GEPA-champion engineered features and all three office-brand
dummies, while keeping raw production columns (`units_12m`, `volume_12m`,
`list_sides_12m`) that RI pruned.

| variant | mean AUC | worst fold | fold spread | mean PR-AUC |
|---|---|---|---|---|
| **transferred (RI list)** | **0.752935** | 0.725296 | 0.050205 | 0.076625 |
| re-derived (MA list) | 0.751677 | 0.724965 | 0.048795 | 0.075983 |

Per-fold delta (MA − RI): `−0.0033  −0.0017  −0.0003  −0.0000  −0.0009` —
**the re-derived list loses 0 of 5 folds**, mean −0.0013.

**Decision: keep RI's list.** The gap is small enough to sit inside the noise
band, but it is consistently negative across every fold, so there is no evidence
to justify diverging — and keeping one shared feature list means future
autoresearch wins transfer between states without re-derivation. `PRODUCTION_FEATURES`
in `crossval.py` is therefore unchanged from RI.

Worth noting for anyone tempted to prune by importance later: MA's importance
ranking discarded the five GEPA champion features, yet keeping them scored
*better* on all five folds. CatBoost feature importance is not the same thing as
predictive contribution.

### Production scoring output

`score_agents.py` wrote `current_agent_risk_scores.csv` — **27,019 active MA
agents** scored off the 2026-07-01 snapshot (team accounts and the synthetic
placeholder excluded, as designed).

| tier | MA agents | MA share | RI share |
|---|---|---|---|
| 1 — Low | 22,822 | 84.47% | 73.11% |
| 2 — Medium | 3,591 | 13.29% | 22.03% |
| 3 — High | 405 | 1.50% | 2.95% |
| 4 — Extra High | 201 | 0.74% | 1.91% |

Mean predicted chance is 1.064% in MA vs 1.706% in RI, and MA flags 2.24% of
agents into tiers 3–4 versus RI's 4.86%. This is consistent rather than
suspicious: MA's underlying 3-month churn rate is genuinely lower and falling
(1.75% across the panel, but only 1.19% in the most recent fold, vs RI's 1.96%).
The calibrator picked that up — the recency bias correction came out at
**0.726x**, meaning the raw model over-predicted current MA churn by ~38% and was
scaled down accordingly, against a bias-corrected base rate of 1.190%.

Because tier cutpoints are multiples of the base rate (`MULT_CUT_POINTS =
[1.0, 3.0, 6.0]`) rather than fixed percentages, they moved with MA's lower base
rate automatically — no hand-tuning, and the same code path RI uses.

**Regenerated 2026-08-30, keyed on `mls_agent_id`.** The 2026-08-10 identifier
switch landed in `data.py` but nothing re-ran the scoring, so for three weeks the
shipped file still carried the retired `agent_profile_id` UUID in column 1 while
every RI output carried the MLS id — any join between them matched zero rows.
Two things about the rerun are worth recording:

- **It was score-neutral, and provably so.** `percent_chance` is identical to the
  July file for all **27,019** agents (max abs diff **0.0**), zero tier changes,
  identical tier distribution, identical base rate (1.640%) and bias correction
  (0.7260x). This is structural rather than lucky: `load_clean` never sorts on
  the agent key, so row order is file order and changing the key cannot reorder
  training rows. The Sales model does not share this property — see its section.
- **The rerun was still necessary**, because explanations can only be
  regenerated, not re-keyed. The 2026-08-11 `new_listings_12m` guardrail fix
  (`LOW_MAX`/`HIGH_MIN` had been transposed) was applied to the source but never
  reflected in the output: **1,084 agents** were being told they had
  "below-average listing activity" on 6, 7 or 8 new listings. Now 0.

Backup of the pre-rerun file: `current_agent_risk_scores.csv.uuid_keyed_backup`.
Schema re-verified identical to `Final_Model/current_agent_risk_scores.csv`.

> **Lesson worth keeping:** the model was fine the whole time and every check
> anyone would have run by eye passed — 27,019 agents, sensible tiers, plausible
> top-10. What was broken was the join key and 1,084 sentences. MA has no
> equivalent of RI's `quarterly_recalibration.py`, which is exactly the wrapper
> whose output gates and mirror step would have caught this. Building one is the
> highest-value operational follow-up on this project.

---

## Results — Forecasted Sales

**Restated 2026-08-31** by `remeasure_accuracy.py`, ported from RI. Held-out
test window `2025-04-01..2025-07-01` — the last 2 resolved quarters, never
passed to any fit. The production functions (`forecast_pzero`,
`forecast_conditional_nonzero`) are imported and called unmodified; the model
carves its own train/calib split out of the 359,263-row history it is handed.
3 seeds, reported as `mean [min..max]`, because both learners subsample by row
order and a single seed is not a point value.

This supersedes `backtest_ma_production.py`, whose framing RI retired on
2026-08-22 and whose RI comparison column was contaminated (see the note at the
end of this section). **RI and MA are now measured by byte-identical code on the
same test window**, so the columns below are directly comparable.

### The headline, stated honestly

MA beats naive persistence — "they sell next year what they sold last year" —
on **every scale-free metric, on both targets**. It loses to persistence on
**volume R²**, by a lot, and that is not noise.

Whole-model, unconditional: `(1 − P(zero)) × conditional point`, scored on all
47,889 test rows including true zeros. This is the honest whole-model figure.

| volume | MA | MA persistence | RI | RI persistence |
|---|---|---|---|---|
| R² | 0.6408 | **0.7324** | **0.7146** | 0.7020 |
| MAE | **$1,502,939** | $1,778,205 | **$1,447,541** | $1,649,261 |
| MdAPE | **46.6%** | 58.0% | **45.7%** | 57.1% |
| Spearman | **0.7385** | 0.6082 | **0.7160** | 0.6106 |
| capture top-10% | **70.7%** | 67.1% | **67.2%** | 65.3% |
| capture top-20% | **72.6%** | 68.9% | **73.7%** | 70.2% |
| P(zero) AUC | 0.8567 | — | 0.8239 | — |
| true zero-rate | 31.2% | — | 24.8% | — |

| units | MA | MA persistence | RI | RI persistence |
|---|---|---|---|---|
| R² | **0.7382** | 0.7199 | 0.7568 | **0.7907** |
| MAE | **2.70** | 3.11 | **2.86** | 3.08 |
| MdAPE | **41.0%** | 50.0% | **37.7%** | 42.9% |
| Spearman | **0.7674** | 0.6718 | **0.7927** | 0.7315 |
| capture top-10% | **71.0%** | 69.0% | 67.9% | **68.6%** |
| capture top-20% | **75.0%** | 73.1% | **73.9%** | 69.4% |
| P(zero) AUC | 0.8705 | — | 0.8636 | — |
| true zero-rate | 23.7% | — | 16.7% | — |

Seed spread is tiny — volume R² ranges 0.6367..0.6441 across three seeds, MAE
moves ~$3K, every other metric moves in the third decimal. **The −0.0916 volume
R² gap to persistence is ~12x the seed spread.** It is a real finding.

### Why volume R² loses to persistence, and why it is not a defect

**This happens in both states, on different targets.** MA loses on volume R²
(−0.0916); **RI loses on units R²** (0.7568 vs 0.7907, −0.0339) and on units
top-10% capture. Neither state loses on any scale-free metric anywhere. So this
is a property of the hurdle design, not of the MA port.

The mechanism: R² is squared error, so it is dominated almost entirely by the
largest producers. The pipeline shrinks — CatBoost at `depth=3, lr=0.03` is
heavily regularised, and the `(1 − P(zero))` multiplier shrinks every prediction
further. Shrinkage is the right trade for the typical agent (it wins MAE by
15.5% and MdAPE by 11.4 points) and for ranking (Spearman +0.13). But on a very
large producer who does repeat next year, predicting 0.8x their last year costs a
squared error that persistence simply does not pay.

MA amplifies this for two reasons that are both real properties of the market,
not modelling choices: MA's volume zero-rate is **31.2% vs RI's 24.8%**, so the
`(1 − P(zero))` multiplier is doing materially more shrinking on average; and MA's
tail is much longer (45,186 agents, $794K average sale price vs RI's $625K).

**This is exactly why RI demoted R² on 2026-08-22** and why this script prints a
baseline on every line. Read MdAPE, Spearman, capture and the model-vs-persistence
ratio. An R² of 0.64 that beats persistence on everything else is a better
forecast than an R² of 0.73 that is just a copy of last year — and a reader
shown only "R² 0.64" would conclude the opposite.

> **Do not quote MA volume R² without its baseline.** On its own it invites
> exactly the wrong conclusion. If one number is needed, use MdAPE (46.6% vs
> persistence 58.0%) or the MAE ratio (−15.5%).

### Interval coverage

Nominal vs actual on nonzero test rows, all six production levels:

| level | MA volume | RI volume | MA units | RI units |
|---|---|---|---|---|
| 50% | **51.1%** | **50.6%** | 56.8% | **51.7%** |
| 65% | **66.1%** | **65.5%** | 72.0% | 72.5% |
| 70% | **71.1%** | **70.6%** | 76.5% | 77.2% |
| 75% | **76.0%** | **75.5%** | 79.9% | 82.2% |
| 80% | **81.1%** | **80.8%** | 83.1% | 85.4% |
| 95% | **95.5%** | **94.8%** | **96.2%** | 96.1% |

- **MA volume bands are excellent** — every level within 1.1 points of nominal,
  the best calibration anywhere in either state. The conformal step sees ~6x RI's
  rows, which is the likely cause.
- **Units bands over-cover at low levels in both states** — MA is +6.8 points at
  the 50% level, RI +1.7, and both run ~+7 points at 65%. Units are small
  integers, so a conformal band cannot land between 3 and 4; discreteness forces
  it outward. MA is *better* than RI at 70/75/80 and worse at 50. Over-covering is
  the safe direction for a range shown to a user, but the 50% units band should
  not be described as a 50% band. **This is the one genuinely open calibration
  item on the MA sales model.**

### What the superseded numbers said, and what changed

The retired `backtest_ma_production.py` reported MA volume MAE $1,575,872 and
units MAE 2.86 on a different (4-quarter) split, with no baseline, one seed, and
R² not reported at all. The restated whole-model figures are $1,502,939 and 2.70.
These are not directly comparable — different window, different framing — and the
change should not be read as an improvement.

Its **RI comparison column was contaminated** and is now retired entirely:
`--mls riar` overrode `MLS_CODE` but not `NON_MLS_MEMBER_AGENT_IDS`, which stayed
`{"H1111111"}` — an MLSPIN id matching nothing in RIAR — so all three RIAR
placeholder buckets were left in, inflating RI's accuracy. The RI columns above
come from RI's own `remeasure_accuracy.py` run with its correct exclusion set
(`{"12345", "00001", "0014"}`), so that defect is gone rather than papered over.

Per-seed metrics: `remeasure_accuracy_results.csv`. Full run: `remeasure_accuracy_ma.log`.


### Production scoring output

`score_agents.py` wrote `current_agent_sales_forecast.csv` — the same **27,019
active MA agents**, off the same 2026-07-01 snapshot as the leave model, with
**identical 33-column schema to RI's output** (re-verified programmatically
2026-08-30), so anything consuming the RI file works on the MA one unchanged.

**Keyed on `mls_agent_id` as of 2026-08-30 — re-keyed by crosswalk join, not
refit.** The file had gone stale on the retired `agent_profile_id` UUID: the
2026-08-10 identifier switch landed in `data.py` but nothing re-ran the scoring,
so for three weeks column 1 disagreed with every RI output and any join between
them matched zero rows. Fixed by joining
`../../Likelihood_to_Leave_Algorithm/agent_id_crosswalk.csv` on
`agent_profile_id` and replacing column 1 — all 27,019 rows resolved, none
dropped, and **every other field is byte-identical to the reviewed July file**
(verified row-for-row: 0 rows with any changed value).

A refit was deliberately NOT done, following RI's rule for this exact
situation. `score_agents.py::build_full_frame` sorts on the agent key and both
learners subsample by row order (XGBoost `subsample`/`colsample_bytree`,
CatBoost's default MVS bootstrap), so **changing the key reorders every training
row and fits a different model** — fixed `random_state` does not protect against
this, because what the seed indexes into moved. RI measured a pure re-key moving
100% of agents by a median of 1.48% (p95 5.94%) and the book total by −1.23%,
and concluded: *re-keying is not a reason to refit.* MA has never quantified its
own noise floor, so a refit here would have moved every forecast for no reason a
reader could see. The backup of the pre-re-key file is
`current_agent_sales_forecast.csv.uuid_keyed_backup`.

Both halves of the hurdle are reported per agent: `volume_pct_zero` /
`units_pct_zero` as their own numbers, then a conditional point estimate and
min/max pair at each of the six confidence levels (50/65/70/75/80/95%). A
representative agent — the median by predicted volume:

```
P(zero):  39.3% volume  |  18.0% units
if selling anything -- point estimate: $1,251,302  (3.4 units)
  50%:  $725,555 - $2,050,898   |  1.9 -  5.4 units
  80%:  $468,251 - $3,131,235   |  0.9 -  8.1 units
  95%:  $296,040 - $4,917,472   |  1.0 - 12.2 units
```

Note the 39.3% P(zero) on volume for a *median* agent. That is not a model
failure — across the backtest's held-out window, 31.2% of MA agents had $0
next-12m volume, genuinely higher than RI's 24.3% over the same quarters.
Reporting it as its own number rather than folding it into a
range that collapses to $0 is exactly what the hurdle design exists for.

---

## Running it

### The quarterly run — use this, not the individual scripts

`quarterly_recalibration.py` is the single entry point when new data lands. It
drives **both** models, gates their output, and records what the calibration
did. Everything below it is for development.

```powershell
$py = "C:\Users\olive\RA_Algorithms_White\Likelihood_to_Leave_Algorithm\.venv\Scripts\python.exe"

cd MA
& $py quarterly_recalibration.py --check           # is a run due? exit 0 = due, 1 = not
& $py quarterly_recalibration.py --dry-run         # full rehearsal, nothing promoted
& $py quarterly_recalibration.py                   # the real thing
& $py quarterly_recalibration.py --adopt-current   # check the SHIPPED files, record as baseline, score nothing
& $py test_recalibration_gates.py                  # fault-inject every check; run after touching one
```

| Step | What it does |
|---|---|
| 0 | **Due check.** A run is due when the data holds a newer quarter *or the source file changed*. That second trigger is the one that would have caught the 2026-08-10 identifier swap. On an unchanged panel it exits cleanly, so it is safe to schedule monthly. |
| 1 | **Regenerate the office-pair catalogue** and report every *new* unverified Tier B mass-mover cluster — these recode moves from "left" to "stayed" with no web verification behind them, so they are surfaced and logged rather than absorbed silently. |
| 2 | **Score both models into a staging directory**, each in its own interpreter. |
| 3 | **Output checks.** Failures block promotion; the previously shipped files stay untouched. |
| 4 | **Promote and record** — `recalibration_state.json` plus one appended row in `recalibration_log.csv`. |

Three divergences from RI's wrapper, each deliberate:

- **It runs both models.** In MA they share one snapshot and one roster, and
  they went stale together. Driving them from one place is what makes the
  cross-model check possible.
- **There is no mirror step** — MA has one copy of each model, nothing to
  mirror to. RI's step 4 is replaced by a **cross-model consistency check**:
  the two outputs must agree on key column, roster and snapshot. That is not a
  like-for-like substitute, it is the check that would have caught the actual
  failure. Every single-file check passed the whole time the files were broken;
  what was wrong existed only *between* them.
- **It re-derives the catalogue instead of stopping for a human.** RI's pairs
  are hand-curated and web-verified, so an unknown cluster there is a real open
  question. MA's are generated and none has ever been verified, so a stop would
  be waiting on a verification step that does not exist. Use
  `--strict-clusters` if you would rather review new clusters before they
  recode anything.

What it deliberately does **not** do: re-key anything. If the identifier
changes again, the Leave model can be rescored (score-neutral — its pipeline
never sorts on the agent key) but the Sales model must be re-keyed by crosswalk
join instead, for the row-order reason documented in its section above.

`--adopt-current` exists for the same reason. "Establish a baseline" and
"refit" are different requests, and conflating them is expensive here: the
Sales model is not invariant to a rerun, so refitting it purely to seed a state
file would move 27,019 reviewed forecasts to record something already known.
That mode runs every check against the files already shipped and records them
if they pass, marking the row `adopted: true` so the log can never imply a fit
that did not happen. It is how the 2026-07-01 baseline was established.

### The gates are tested

`test_recalibration_gates.py` corrupts the shipped outputs in memory, one
defect at a time, and asserts the matching check fires — **11/11 caught, clean
output raises nothing.** A check that has only ever seen good input is a line
of code nobody has watched do its job. It earned its place on the first run by
finding that `check_sales` raised `KeyError` on a missing column instead of
reporting a failure, which would have killed the wrapper with a traceback
rather than taking the orderly "nothing promoted" path — and skipped every
later check, including the cross-model one.

The first two cases are the defect that actually shipped, not hypotheticals.

**The counterfactual was verified directly.** Rewinding the state file to the
pre-swap source file with the quarter left unchanged — exactly the situation on
2026-08-10 — makes the due check report `DUE — source file changed`. Had this
wrapper existed then, the run would have fired despite no new quarter, and the
three-week staleness would not have happened.

### Individual scripts, for development

```powershell
# --- Likelihood to Leave ---
cd MA\Likelihood_to_Leave
& $py derive_ma_office_pairs.py   # regenerate office-pair classifications (re-run when new quarters land)
& $py data.py                     # sanity-check the loader
& $py crossval.py                 # logistic / XGBoost / CatBoost bake-off
& $py derive_ma_config.py         # transferred-RI vs re-derived-MA feature list
& $py score_agents.py             # writes current_agent_risk_scores.csv

# --- Forecasted Sales ---
cd ..\Forecasted_Sales
& $py data.py
& $py remeasure_accuracy.py              # RESTATE ACCURACY -- 3 seeds, held-out, vs naive persistence
& $py remeasure_accuracy.py --no-exclusions   # population A/B: put placeholder "H1111111" back
# backtest_ma_production.py is SUPERSEDED -- kept for provenance only. Its framing was
# retired (no baseline, one seed, R2 read as skill) and its --mls riar arm is contaminated;
# see "What the superseded numbers said" above. Do not restate accuracy from it.
& $py score_agents.py                    # writes current_agent_sales_forecast.csv
```

`derive_ma_office_pairs.py` must run before the Leave model — `data.py` imports
the file it generates. The quarterly wrapper handles this ordering for you,
which is the point: running it by hand is how the catalogue drifts behind the
panel.
