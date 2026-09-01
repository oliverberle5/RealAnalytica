# Agent Likelihood-to-Leave Model

Predicts each Rhode Island (RIAR) real estate agent's percent chance of
switching brokerages in the next 3 months, so retention outreach can be
prioritized by risk rather than guesswork.

**See `agent_likelihood_to_leave_report.pdf` for the full visual summary**
(accuracy, feature drivers, calibration) — this README covers the same
ground in text, plus how to reproduce or re-run everything.

## Headline result

**74.4% AUC** (CatBoost, 5-fold time-based validation) — meaning that if you
picked one agent who left and one who didn't, the model ranks the one who
left as higher-risk about 74% of the time. For comparison, a coin flip is
50%.

| Model | Mean AUC | Notes |
|---|---|---|
| Logistic regression | 69.1% | Interpretable gut-check only, not used for production scoring |
| XGBoost | 72.8% | Strong runner-up |
| **CatBoost** | **74.4%** | **Production model** — wins 5 of 5 validation folds |

**AUC is not accuracy and not precision** — this is the single easiest thing
to get wrong when quoting this model. Accuracy is a useless metric here: a
model that says "nobody ever leaves" scores 98.0% and is worth nothing. Two
figures translate better for a business reader:

- **PR-AUC 0.0771** — at a 1.9628% base rate, that is **3.93× better than
  chance** at concentrating real departures near the top of the list.
- **Lift @ top 10% = 3.59×** — the 10% highest-scored agents contain 3.59×
  as many actual departures as a random 10% would. Since the product drives
  ranked outreach lists, this is arguably the number to quote.

The defensible claim about model choice is the **5-of-5 fold sweep**, not
the 0.0123 mean margin over XGBoost — that margin sits inside the noise band
(±0.019 parametric, ±0.024 empirical, on the mean).

**2026-08-04 label policy, stated per geometry:** a **magnet** destination
(many distinct origins converging on one office in one quarter) **counts as
churn** — that shape is individual recruitment, and an agent who takes a
better offer left voluntarily. **Trickles** (one office pair bleeding 2–4
agents a quarter over many quarters) and **documented rebrands/acquisitions**
are **not churn**. Geometry alone never suppresses a label; evidence decides.
Labels and all metrics are unchanged by this restatement — 44 suppressed
pairs, 459 recoded rows, 1.963%, 74.4% AUC.

**2026-07-28 correctness fix:** `office_arrival_rate_12m` /
`company_arrival_rate_12m` were normalizing arrivals by *current* headcount
while the sibling `office_exit_rate_12m` / `company_exit_rate_12m` normalize
exits by *prior* headcount — an inconsistent base for two rates meant to be
symmetric "agents coming vs. going" measures. Both now divide by
`prior_headcount`. Verified on a cloned dataset before applying: cost
production CatBoost ~0.3pp mean AUC on its own, adopted anyway for
correctness/defensibility, not for an AUC gain.

**2026-07-28 data-quality fix:** a synthetic MLS placeholder record
("Non-Mls Member" — a buyer-side catch-all bucket for cooperating agents
outside the board, not a real person) was being trained and scored like any
other agent. Excluded (`NON_MLS_MEMBER_AGENT_IDS` in `data.py`), tested
non-committally first: mean AUC 0.7402 → 0.7411, small but real, and
restored CatBoost to winning all 5 folds. See `CHANGELOG.md` for both
fixes' full before/after numbers.

**2026-08-10 follow-up:** switching the agent key to the real MLS ID made two
more synthetic records readable that opaque UUIDs had hidden — `"00001"`
("Assisted Sale") and `"0014"` ("ALLIANCE MEMBER", the RIAR feed's holding pen
for Connecticut cross-board activity). Both were excluded on the same
correctness argument and both cleared `confirm_population.py` (3 seeds × 3 fold
structures). `"0014"` had stopped appearing after 2025-01-01, so it never
reached a scored snapshot — it contaminated training only, which is why nothing
in the production output ever looked wrong. Panel 77,422 → **77,389** usable
rows (33 rows, all negatives), mean AUC 0.7411 → **0.7440**.

## How it works, in plain terms

**The data.** 84,417 quarterly snapshots of 5,625 agents in the Rhode
Island MLS (RIAR), spanning late 2021 through mid-2026, of which **77,389
are usable** for the 3-month horizon after censoring and exclusions. Note
that "RIAR agents" means agents *in the RI MLS* — market and macro columns
key to the MLS, not to the agent's mailing address, and roughly 12,600 rows
belong to Massachusetts residents. Each snapshot records an
agent's activity (listings, sales volume, transactions), their office and
company's activity, and broader market/economic conditions at that point
in time.

**The question asked.** "Will this agent switch brokerages in the next 3
months?" A 3-month window was chosen over the more obvious 12-month one
because it closes fast enough that far more of the data becomes usable,
and it answers the more useful business question — who's a flight risk
*right now* — rather than a slower-moving one.

**Cleaning the label.** Not every office switch is a resignation. A big
chunk of what looks like "agents leaving" is actually corporate
housekeeping — a brokerage rebrand, an acquisition, or an internal office
consolidation that swept a whole team along with it, not any individual
agent's choice. **459 such rows** (across 44 confirmed office pairs,
verified individually via public records) were recoded from "left" to
"stayed" so the model learns from real voluntary attrition, not corporate
reshuffling. Before the cleanup **2.5223%** of agent-quarters are labeled a
departure; after it, **1.9628%** are a genuine one.

Worth being precise about how this works: the detectors find a *shape*
(many agents moving the same way at the same time), but evidence decides
the *cause*. Nothing is recoded on geometry alone — each candidate is
resolved against outside evidence, and plenty are deliberately left labeled
as churn when the evidence says they were real departures.

**What the model looks at.** Roughly 95 signals per agent-quarter,
spanning five levels:
- **The agent**: tenure, career length, sales volume and its trend, listing
  activity, how long since their last closing.
- **Their office**: headcount, growth/shrinkage, how many peers are
  leaving, how central this agent is to the office's business.
- **Their parent company**: the same kind of signals one level up, for
  companies that span multiple offices.
- **The broader market**: total transaction volume and its trend, days on
  market, for the state as a whole.
- **The macro backdrop**: state unemployment and its trend, national CPI.

A few of these (company-level exit rates, whether the office is an
independent or a franchise brand) turned out to be genuinely useful signals
that simply hadn't been wired into the model before this round.

**The model itself.** CatBoost, a gradient-boosted decision tree model,
chosen over a comparable XGBoost model and a simple logistic-regression
baseline after a proper apples-to-apples comparison. Validation is
*time-based* — the model is always tested on quarters it has never seen,
in chronological order after its training data — rather than a random
shuffle, because a random split would let future information leak into
the past and produce an unrealistically optimistic score.

**Turning a raw score into a percentage.** The model's raw output isn't a
percentage by itself — it's calibrated (via Platt scaling) against
historical outcomes so that "predicted 8%" really does mean "8 times out
of 100 in similar situations, this happened." A backtest found this
calibration drifting slightly in recent quarters — traced to a wave of
confirmed rebrand/M&A cleanup landing unevenly across the calendar, not a
real change in agent behavior — so a small correction factor is now
re-measured and applied every time the model is refreshed.

**The output.** For each currently active agent:
- A **percent chance** of leaving in the next 3 months.
- A **4-tier risk label** — Low / Medium / High / Extra High.
- A **risk multiplier** — how many times more (or less) likely this agent
  is to leave than the average agent.

The tiers are cut at **multiples of the base rate** (Low `<1×`, Medium
`1–3×`, High `3–6×`, Extra High `≥6×`), not at population quantiles. This
matters: a quantile scheme always puts a fixed share of agents in the top
bucket, even in a quarter when nobody is especially at risk, so it can label
someone "High" whose real odds are worse than random. Multiplier cutpoints
cannot do that by construction, and they self-calibrate as the base rate
moves. Out-of-sample backtest (cutpoints fit on 2021-10..2024-04, applied
unchanged thereafter): Low 0.53× / Medium 1.18× / High 4.18× / Extra High
6.00× — monotone, with Low correctly sitting *below* random.

As of the most recent data (July 2026), 4,606 agents are actively scored;
see `current_agent_risk_scores.csv` for the full list.

## Files in this folder

| File | What it does |
|---|---|
| `data.py` | Loads and cleans the source data: applies the 3-month label, the mass-mover cleanup, and builds all ~95 features. Everything else depends on this. |
| `rebrand_classifications.py` | The verified list of confirmed rebrand/M&A/mass-mover office transitions used in the label cleanup. |
| `split.py` | A simple time-based train/test split (used for quick checks). |
| `train_baseline.py` | The logistic-regression gut-check model. |
| `crossval.py` | The 5-fold time-based comparison between logistic regression, XGBoost, and CatBoost — this is what produced the headline accuracy numbers. |
| `calibrate.py` | Turns the model's raw score into a trustworthy percentage, a 4-tier risk label, and a risk multiplier — includes the quarterly bias-correction step. |
| `score_agents.py` | **The production script** — retrains the model, recalibrates, and scores every currently active agent. Produces `current_agent_risk_scores.csv`. Normally invoked by the quarterly wrapper below rather than by hand. |
| `../modeling/quarterly_recalibration.py` | **The quarterly entry point** (lives in `modeling/`, not here, because it needs the detector and the classification catalogue). One command per quarter: checks whether new data has landed, refuses to run if an unreviewed mass-mover cluster is present, rescores, checks the output, copies the result here and to the repo root, and logs what the calibration did. |
| `generate_report.py` | Builds `agent_likelihood_to_leave_report.pdf` from the current model. |
| `agent_likelihood_to_leave_report.pdf` | The visual summary report. |
| `current_agent_risk_scores.csv` | The current output: every active agent's percent chance, risk tier, risk multiplier, and up to five plain-language "why this score" reasons. Agents are identified by `mls_agent_id` — their real MLS ID — as of 2026-08-10; previously an opaque `agent_profile_id` UUID. Use `../agent_id_crosswalk.csv` to join against anything produced before that date. |
| `generate_distribution_chart.py` | Builds `risk_score_distribution.png` from the current scores. |
| `risk_score_distribution.png` | Histogram of all active agents' risk percentages, shaded by risk tier — shows most agents cluster under 1%, with a long thin tail of higher-risk agents (axis capped at 8% for readability; the excluded ~1% of agents scoring higher are called out on the chart itself). |

## Reproducing / refreshing the numbers

Requires the same Python environment as the main project (`pandas`,
`numpy`, `scikit-learn`, `xgboost`, `catboost`, `matplotlib`) and the
source data file one folder up
(`../leave-dataset-with-team-distinction-mls-id.csv`). That filename was
stale here through two source-file switches; `data.py::DATA_PATH` is the
authoritative pointer, and this line is now correct as of 2026-08-10.

**The normal quarterly refresh is one command**, run from `../modeling/`:

```
python quarterly_recalibration.py           # runs only if new data has landed
python quarterly_recalibration.py --check   # is a refresh due? (nothing is written)
```

It refits the calibration to the newest quarter, verifies the result, and
copies the scored CSV here and to the repo root. The individual scripts
below are still the right tool for re-running one piece in isolation:

```
python crossval.py          # re-run the model comparison
python calibrate.py         # re-check calibration and the bias correction
python score_agents.py      # rescore all current agents -> current_agent_risk_scores.csv
python generate_report.py   # rebuild the PDF report
python generate_distribution_chart.py   # rebuild the risk score distribution chart
```

Each of these retrains from scratch (about a minute for `score_agents.py`,
a few for `crossval.py`) rather than loading a saved model — there's no
versioned model artifact yet, so re-running is always working from the
latest data and the latest code.

## Known limitations (kept visible on purpose, not hidden)

- **Tiers are fixed risk levels, not population quotas.** Because the
  cutpoints are multiples of the base rate, bucket sizes float from quarter
  to quarter — that's intentional, but it means "Extra High" won't always
  hold the same share of the roster. **Currently worth watching:** Extra
  High sits at **1.8%** of the roster (82 of 4,606, re-measured 2026-08-11)
  against this project's own documented ~1.0–1.5% expectation, having risen
  across successive runs and then held (1.1% → 1.5% → 1.9% → 1.8%). Flagged OPEN
  in `HANDOFF.md`; not yet attributed. The quarterly wrapper now records
  this share every run, so the next few points come from a log rather than
  from whoever happened to look.
- **Team accounts are excluded, which is a mitigation and not a solution.**
  No field distinguishes a team roster from an individual, and production
  credit concentrates on one member's row, leaving active teammates showing
  ~$0. The model scored measurably worse on these rows (pooled AUC 0.7024
  vs. 0.7400 on the rest), so **153 team accounts (2,222 rows) are dropped**
  from the individual model. Those people still switch brokerages and are
  no longer scored at all. A statistical size proxy was tested as an
  identification method and rejected on principle — it cannot separate "team
  with concentrated credit" from "elite solo producer."
- **The mass-mover catalogue is not exhaustive.** Only clusters clearing the
  detectors' thresholds have been reviewed; undetected consolidations are
  presumed present.
- **Per-agent explanations are template-based, not SHAP.** The scored CSV
  carries up to five plain-language reasons per agent, but 2 of the ~10
  templates reference features that were pruned from the model and
  therefore never fire — not a crash, just quietly unused.
- **The quarterly refresh is automated, but nothing schedules it for you.**
  `../modeling/quarterly_recalibration.py` runs the whole cycle as one
  command and refuses to ship output that fails its checks, but a human (or
  a Task Scheduler entry — see the Quarterly Reset Protocol in `HANDOFF.md`)
  still has to trigger it. It is safe to trigger more often than quarterly:
  it exits without doing anything when no new data has landed.
- **Per-agent day-on-market data is unused — but no longer because it's
  broken.** The upstream DOM bug was fixed as of the 2026-07-15 data file
  (null rate 36.4%, median 27 days, sane IQR). It stays excluded for a
  different and more interesting reason: CatBoost's own importance ranking
  places it *above* the pruning cutoff, yet including it makes held-out AUC
  **worse** (0.7384 → 0.7353). In-sample importance is not out-of-sample
  usefulness.

For the full engineering history behind every decision above — including
approaches that were tried and rejected, and why — see `HANDOFF.md` in the
`modeling/` folder.
