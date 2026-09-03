# Real Analytica — Agent Intelligence Models

Two production models that score every active real estate agent in a market,
built to run across states from one shared codebase.

| Model | What it predicts | Output |
|---|---|---|
| **Likelihood to Leave** | Probability an agent switches brokerage in the next **3 months** | A calibrated percentage, a 1–4 risk tier, and up to five plain-language reasons (2.9 on average) |
| **Forecasted Sales** | An agent's **next-12-month** dollar volume and unit count | A point estimate plus a calibrated range at six confidence levels, and the probability of zero production |

Both currently run on **Rhode Island** (RIAR) and **Massachusetts** (MLSPIN).
Adding a state is a documented, tooled procedure — see
**[docs/ONBOARDING_PROTOCOL.md](docs/ONBOARDING_PROTOCOL.md)**.

---

## Where things stand

**Likelihood to Leave** — forward-chaining (expanding-window) validation, 5 folds,
CatBoost in production:

| | mean AUC | worst fold | base churn rate | agents scored |
|---|---|---|---|---|
| **Massachusetts** | **0.7529** | 0.7253 | 1.75% | 27,019 |
| **Rhode Island** | **0.7440** | 0.7167 | 1.96% | 4,606 |

**What those columns mean**

- **AUC** — given one agent who left and one who stayed, how often the model
  ranks the leaver as riskier. 0.50 is a coin flip; **0.75 is strong for a
  3-month window** where fewer than 2 agents in 100 actually move.
- **Worst fold** — the weakest of the 5 time-ordered validation folds. Quoted
  because a mean can hide one bad quarter.
- **Base churn rate** — how often the event actually happens. It is low, which is
  why tier cut points are multiples of the base rate rather than fixed
  percentages: they move with the market on their own.

> MA leads on the mean by +0.0089 but **loses folds 3 and 5**, and the documented
> 95% CI on 5-fold mean AUC is ≈ ±0.019. These are different markets; this is not
> an apples-to-apples "MA is better" claim.

**Forecasted Sales** — held-out final 2 quarters (never seen by any fit), 3 seeds,
scored on every test row. Each figure is shown against **naive persistence** —
"this agent sells next year exactly what they sold last year" — because a sales
forecast with no baseline cannot be judged.

| | volume error (MdAPE) | units error (MAE) | ranking (Spearman) | 95% range holds | agents scored |
|---|---|---|---|---|---|
| **Massachusetts** | **46.6%** vs 58.0% naive | **2.70** vs 3.11 naive | **0.739** vs 0.608 | 95.5% | 27,019 |
| **Rhode Island** | **45.7%** vs 57.1% naive | **2.86** vs 3.08 naive | **0.716** vs 0.611 | 94.8% | 4,606 |

*Test window: 47,889 MA agent-quarters, 7,989 RI. Bold = the model; it beats the
baseline in every column that has one.*

**What those columns mean**

- **MdAPE** — the typical agent's percentage miss. Scale-free, so it is the one
  error figure that *is* comparable across states.
- **MAE** — the average absolute miss. **Not comparable across states**: MA's
  average sale price is $794K vs RI's $625K, so MA dollar errors are naturally
  larger. On volume, MA beats its baseline by 15.5% and RI by 12.2%.
- **Spearman** — ranking quality: does the model put the right agents at the top?
  This is the closest analogue to AUC for a continuous forecast, and it is where
  the model beats persistence most decisively in both states.
- **95% range holds** — of the agents who did sell, how often the true value
  landed inside the 95% band. Nominal is 95%, so both states are within ~0.5pt.

**The other half of the model.** Each forecast is a *hurdle*: first "will this
agent sell anything at all", then "how much, if they do". The first half is a
real classifier and it is strong — **AUC 0.857 (MA) / 0.824 (RI)** on volume.
That half matters more in MA, where **31.2%** of agents post zero next-12-month
volume versus RI's **24.8%**. It is reported as its own column rather than folded
into a range that collapses to $0.

> **Volume R² tells the opposite story, and that is expected — not a defect.**
> MA scores 0.641 against persistence's 0.732; RI *wins* on volume (0.715 vs
> 0.702) but **loses on units** (0.757 vs 0.791). Each state loses one target.
> R² is squared error, so it is dominated almost entirely by the largest
> producers, and the pipeline deliberately shrinks its predictions. Shrinking is
> right for the typical agent — it wins MdAPE by 11.4 points — but on a very
> large producer who does repeat, predicting 0.8× last year costs a squared
> error that persistence never pays.
> **Never quote R² without its baseline.** Alone it invites the opposite of the
> correct conclusion.

Full detail: [MA/README.md](MA/README.md) ·
[docs/RI_Forecasted_Sales_HANDOFF.md](docs/RI_Forecasted_Sales_HANDOFF.md) ·
[docs/RI_Likelihood_to_Leave_CHANGELOG.md](docs/RI_Likelihood_to_Leave_CHANGELOG.md)

---

## Repository map

```
RI/                          Rhode Island (mls_code "riar")
  Likelihood_to_Leave/         churn model — data.py, crossval.py, score_agents.py
  Forecasted_Sales/            sales model — hurdle_model.py, score_agents.py
MA/                          Massachusetts (mls_code "mlspin")
  Likelihood_to_Leave/
  Forecasted_Sales/
  quarterly_recalibration.py   THE entry point when new data lands (drives both models)
  test_recalibration_gates.py  fault-injects each output check; 11/11 caught
  README.md                    MA results, methodology, and known gaps in full

states/                      State-onboarding infrastructure
  registry.py                  ONE declaration per state, for both models
  verify.py                    proves the registry matches every shipped data.py
  discover.py                  profiles a new panel, finds placeholder records
  scaffold.py                  generates a new state's two model directories

docs/
  ONBOARDING_PROTOCOL.md       13-step runbook for adding a state
  CT_EXPANSION_ASSESSMENT.md   why Connecticut is deferred
  schema/                      output contracts + synthetic samples

data/                        the shared source CSV (NOT in git — see below)
```

**Every state has the identical shape.** RI's two models used to sit at the repo
root under different names; that exception was removed on 2026-09-01 rather than
coded around.

---

## The one thing to understand

**A state is not a folder. A state is a value of `mls_code`.**

All four models read the *same* CSV and differ only by that filter. There is no
per-state data file, and creating one is the drift bug the tooling exists to
prevent.

Two non-obvious consequences, both of which have caused real incidents here:

1. **`agent_state` is not a state filter.** Every `market_*` and `state_*` column
   is keyed to the **MLS**, not the agent's home address. MLSPIN is only 91.2%
   Massachusetts agents. Splitting on `agent_state` yields a subset carrying
   another market's macro series.
2. **An identifier from one panel matches nothing in another, silently.** RIAR
   ids (`12345`, `00001`, `0014`) cannot appear under `mlspin`. Inheriting them
   excludes zero rows and raises zero errors.

---

## Getting the data

The source panel is **not in this repository**: it is ~298MB (over GitHub's
100MB limit) and holds personal data on ~50,000 named agents. The scored outputs
are excluded for the same reason.

Place the file here:

```
data/leave-dataset-with-team-distinction-mls-id.csv
```

It bundles both panels in one export:

| `mls_code` | market | rows | agents |
|---|---|---|---|
| `mlspin` | Massachusetts | 612,358 | 45,186 |
| `riar` | Rhode Island | 84,417 | 5,625 |

Output contracts are documented without the data in
**[docs/schema/](docs/schema/)** — column lists, dtypes, and fabricated sample
rows. No real agent appears anywhere in this repo.

---

## Quick start

```powershell
$py = "<path to python 3.14 with pandas, catboost, xgboost, scikit-learn, scipy>"
cd <repo root>

& $py -m states.verify                  # registry agrees with every shipped data.py
& $py -m states.discover --self-test    # placeholder detector still finds known answers
& $py MA\quarterly_recalibration.py --check   # is a recalibration due? exit 0 = due
```

If those three pass, the checkout is sound.

### Re-scoring a state when new data lands

**Use the wrapper, never the individual scripts.**

```powershell
cd MA
& $py quarterly_recalibration.py --check     # exit 0 = due, 1 = not
& $py quarterly_recalibration.py --dry-run   # full rehearsal, nothing promoted
& $py quarterly_recalibration.py             # the real thing
```

It drives **both** models, gates their output, runs a cross-model consistency
check (same key, roster and snapshot), and appends one row to
`recalibration_log.csv`. A run is due when the data holds a newer quarter **or
the source file changed** — that second trigger exists because an identifier
swap once landed in `data.py` with nothing re-scoring, and two shipped
spreadsheets sat on a retired join key for three weeks while every number in
them looked fine.

### Adding a state

Follow **[docs/ONBOARDING_PROTOCOL.md](docs/ONBOARDING_PROTOCOL.md)** — 13 steps,
each with a command, a pass condition and a stop condition.

```powershell
& $py -m states.discover --mls-code <new_code>   # profile the panel, draft a StateSpec
# ... confirm placeholders by hand, register the state ...
& $py -m states.verify
& $py -m states.scaffold --state <key>           # generate both model directories
```

---

## For an AI agent picking this up

Read in this order:

1. **This file** — what the models are and how the repo is laid out.
2. **[docs/ONBOARDING_PROTOCOL.md](docs/ONBOARDING_PROTOCOL.md)** — the operating
   procedure, including a table of the four failures that have actually happened
   here and which step now catches each.
3. **[MA/README.md](MA/README.md)** — the fullest worked example: results,
   methodology, and an explicit account of what is weaker than RI.
4. **`states/registry.py`** — the single source of truth for anything
   state-specific.

**Rules that are not optional here:**

- **Report per-fold, not just means.** State the noise band. A gap smaller than
  it is not a result — MA's XGBoost "beat" CatBoost by 0.0013 and CatBoost was
  correctly kept.
- **Never quote R² without a baseline**, and never compare absolute MAE across
  states (MA's average sale price is $794K vs RI's $625K).
- **A metric measured under a superseded framing is _unrestated_**, not "roughly
  the same".
- **The two models are not symmetric about re-keying.** The Sales model's
  `build_full_frame` sorts on the agent key and both learners subsample by row
  order, so changing the key refits a different model — a fixed seed does not
  protect you. If the identifier changes, **re-key the Sales file by crosswalk
  join, do not refit.** The Leave model never sorts on the key and can simply be
  rescored.

---

## Known gaps

- **Generated office-pair catalogues are not web-verified.** RI's list of
  (origin, destination) office pairs that represent rebrands and acquisitions
  rather than real churn is hand-curated. MA's is generated, so MA gets no
  equivalent of RI's confirmed-M&A trickle handling and will under-catch trickle
  moves around real acquisitions. Conservative direction, but a genuine gap.
- **Units confidence bands over-cover at low levels** in both states (56.8% at
  the nominal 50% level in MA). Integer discreteness forces the conformal band
  outward. Safe direction for a user-facing range, but a 50% units band should
  not be described as a 50% band.
- **36 agents (0.13%) in the shipped MA file carry an impossible explanation** —
  "300% of agents company-wide left in the last 12 months." The source feature
  `company_exit_rate_12m` reaches 4.0 because departures are divided by *current*
  headcount, so a company that shrank can exceed 1.0. The rate is a legitimate
  risk signal at that magnitude; rendering it as "300% of agents" is not a
  sentence that can be true. Needs a clamp in the explanation layer, not a model
  change. Same family as the guardrail defect that once told 1,084 agents they
  had "below-average listing activity" on 6–8 listings.
- **RI's Sales `data.py` has no `MLS_CODE` constant** — it inlines `== "riar"`
  mid-function. Correct today; `states/verify.py` reports it as a warning
  because the one value whose job is to be the state switch should be greppable.
- **Connecticut is deferred** — the dataset contains no CT panel. See
  [docs/CT_EXPANSION_ASSESSMENT.md](docs/CT_EXPANSION_ASSESSMENT.md).
