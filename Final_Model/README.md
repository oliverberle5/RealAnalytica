# Agent Likelihood-to-Leave Model

Predicts each Rhode Island (RIAR) real estate agent's percent chance of
switching brokerages in the next 3 months, so retention outreach can be
prioritized by risk rather than guesswork.

**See `agent_likelihood_to_leave_report.pdf` for the full visual summary**
(accuracy, feature drivers, calibration) — this README covers the same
ground in text, plus how to reproduce or re-run everything.

## Headline result

**73.7% AUC** (CatBoost, 5-fold time-based validation) — meaning that if you
picked one agent who left and one who didn't, the model ranks the one who
left as higher-risk about 74% of the time. For comparison, a coin flip is
50%.

| Model | Mean AUC | Notes |
|---|---|---|
| Logistic regression | 68.1% | Interpretable gut-check only, not used for production scoring |
| XGBoost | 73.1% | Strong runner-up |
| **CatBoost** | **73.7%** | **Production model** — wins 4 of 5 validation folds |

## How it works, in plain terms

**The data.** 84,417 quarterly snapshots of 5,625 Rhode Island real estate
agents, spanning late 2021 through mid-2026 — each snapshot records an
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
agent's choice. 335 such rows (across 23 confirmed events, verified
individually via public records) were recoded from "left" to "stayed" so
the model learns from real voluntary attrition, not corporate reshuffling.
After this cleanup, about 2.1% of agent-quarters are a genuine departure.

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
- A **1–5 risk quintile** (1 = Low, 5 = High).
- A **risk multiplier** — how many times more (or less) likely this agent
  is to leave than the average agent.

As of the most recent data (July 2026), 4,754 agents are actively scored;
see `current_agent_risk_scores.csv` for the full list.

## Files in this folder

| File | What it does |
|---|---|
| `data.py` | Loads and cleans the source data: applies the 3-month label, the mass-mover cleanup, and builds all ~95 features. Everything else depends on this. |
| `rebrand_classifications.py` | The verified list of confirmed rebrand/M&A/mass-mover office transitions used in the label cleanup. |
| `split.py` | A simple time-based train/test split (used for quick checks). |
| `train_baseline.py` | The logistic-regression gut-check model. |
| `crossval.py` | The 5-fold time-based comparison between logistic regression, XGBoost, and CatBoost — this is what produced the headline accuracy numbers. |
| `calibrate.py` | Turns the model's raw score into a trustworthy percentage, a risk quintile, and a risk multiplier — includes the quarterly bias-correction step. |
| `score_agents.py` | **The production script** — retrains the model, recalibrates, and scores every currently active agent. Produces `current_agent_risk_scores.csv`. |
| `generate_report.py` | Builds `agent_likelihood_to_leave_report.pdf` from the current model. |
| `agent_likelihood_to_leave_report.pdf` | The visual summary report. |
| `current_agent_risk_scores.csv` | The current output: every active agent's percent chance, risk quintile, and risk multiplier. |
| `generate_distribution_chart.py` | Builds `risk_score_distribution.png` from the current scores. |
| `risk_score_distribution.png` | Histogram of all active agents' risk percentages, shaded by quintile — shows most agents cluster under 1%, with a long thin tail of higher-risk agents (axis capped at 8% for readability; the excluded ~1% of agents scoring higher are called out on the chart itself). |

## Reproducing / refreshing the numbers

Requires the same Python environment as the main project (`pandas`,
`numpy`, `scikit-learn`, `xgboost`, `catboost`, `matplotlib`) and the
source data file one folder up (`../leave-dataset_most_recent_updated.csv`).

```
python crossval.py          # re-run the model comparison
python calibrate.py         # re-check calibration and the bias correction
python score_agents.py      # rescore all current agents -> current_agent_risk_scores.csv
python generate_report.py   # rebuild the PDF report
python generate_distribution_chart.py   # rebuild the risk score distribution chart
```

Each of these retrains from scratch (a few minutes) rather than loading a
saved model — there's no versioned model artifact yet, so re-running is
always working from the latest data and the latest code.

## Known limitations (kept visible on purpose, not hidden)

- **Quintiles are fixed risk tiers, not a forced 20%-per-bucket split.**
  The bias-correction step can shift a given quarter's population away
  from an even 20/20/20/20/20 distribution — that's intentional (the
  tiers represent a fixed risk level over time), but it means "quintile 5"
  won't always contain exactly a fifth of agents.
- **A per-agent explainability layer (e.g. SHAP, "why is this agent
  flagged") doesn't exist yet.**
- **No automated quarterly refresh pipeline yet** — each script above is
  run manually.
- **Per-agent day-on-market data is currently unused** — the existing
  column is mostly broken (fabricated zero values); a fix would need to
  happen at the data source, not in this model.

For the full engineering history behind every decision above — including
approaches that were tried and rejected, and why — see `HANDOFF.md` in the
`modeling/` folder.
