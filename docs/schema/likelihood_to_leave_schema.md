# `current_agent_risk_scores.csv` — schema

**Model:** Likelihood to Leave  
**Grain:** one row per active agent  
**Columns:** 14  
**Rows in the current MA run:** 27,019

One row per active agent: the calibrated probability that the agent switches brokerage in the next 3 months, the 1-4 risk tier, and up to five plain-language reasons shown to the user.

> The real file is **not in git** — it contains named agents with per-person predictions. This schema and the synthetic sample beside it are the contract. RI and MA share it exactly, so anything consuming one works on the other unchanged.

## Reading it

- `risk_tier` is 1-4 (`Low` / `Medium` / `High` / `Extra High`). Cut points are multiples of the state's base rate, so they move with the market rather than being fixed percentages.
- `percent_chance` is a percentage (0-100), not a 0-1 probability.
- `risk_multiplier` is the agent's risk relative to the state base rate.
- `why_*` are generated sentences, ordered most to least important; an agent may have fewer than five.

## Columns

| # | column | dtype | example |
|---|---|---|---|
| 1 | `mls_agent_id` | `str` | EXAMPLE002 |
| 2 | `office_name` | `str` | Example Realty Harbor |
| 3 | `agent_city` | `str` | Rivertown |
| 4 | `agent_state` | `str` | XX |
| 5 | `snapshot_date` | `str` | 2026-07-01 |
| 6 | `percent_chance` | `float64` | 3.874 |
| 7 | `risk_tier` | `int64` | 3 |
| 8 | `risk_tier_label` | `str` | High |
| 9 | `risk_multiplier` | `float64` | 3.25 |
| 10 | `why_top_driver` | `str` | 18% of the agents at their office left in the last 12 months. |
| 11 | `why_reason_2` | `str` | Hasn't closed a sale in 10 mo - extended inactivity is one of the s... |
| 12 | `why_reason_3` | `float64` | Only 7 mo at their current office - new hires demonstrate increased... |
| 13 | `why_reason_4` | `float64` | Took only 2 new listing(s) in the last 12 months - below-average li... |
| 14 | `why_reason_5` | `float64` | 2 office changes in the last 5 years. |

## Synthetic sample

Three fabricated agents spanning the range — see [`likelihood_to_leave_SAMPLE_synthetic.csv`](likelihood_to_leave_SAMPLE_synthetic.csv). **No real agent, office, or prediction appears in this repository.**
