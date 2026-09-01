# `current_agent_risk_scores.csv` — schema

**Model:** Likelihood to Leave  
**Grain:** one row per active agent  
**Columns:** 14  
**Rows in the current MA run:** 27,019

One row per active agent. Probability that agent switches brokerage in the next 3 months, plus the tier and the plain-language explanation shown to users.

> The real file is **not in git** — it contains named agents with per-person predictions. This schema and the synthetic sample beside it define the contract; regenerate the real file with `score_agents.py`. RI and MA share this schema exactly, so anything consuming one works on the other unchanged.

| # | column | dtype | example |
|---|---|---|---|
| 1 | `mls_agent_id` | `str` | `EXAMPLE001` |
| 2 | `office_name` | `str` | `Example Realty C` |
| 3 | `agent_city` | `str` | `Springfield` |
| 4 | `agent_state` | `str` | `XX` |
| 5 | `snapshot_date` | `str` | `2026-07-01` |
| 6 | `percent_chance` | `float64` | `25.66` |
| 7 | `risk_tier` | `int64` | `7` |
| 8 | `risk_tier_label` | `str` | `example_8` |
| 9 | `risk_multiplier` | `float64` | `6.066` |
| 10 | `why_top_driver` | `str` | `example_10` |
| 11 | `why_reason_2` | `str` | `example_11` |
| 12 | `why_reason_3` | `str` | `example_12` |
| 13 | `why_reason_4` | `str` | `example_13` |
| 14 | `why_reason_5` | `str` | `example_14` |
