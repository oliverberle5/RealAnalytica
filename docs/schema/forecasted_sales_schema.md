# `current_agent_sales_forecast.csv` — schema

**Model:** Forecasted Sales  
**Grain:** one row per active agent  
**Columns:** 33  
**Rows in the current MA run:** 27,019

One row per active agent: next-12-month dollar volume and unit count as a calibrated range at six confidence levels, with both halves of the hurdle model reported separately.

> The real file is **not in git** — it contains named agents with per-person predictions. This schema and the synthetic sample beside it are the contract. RI and MA share it exactly, so anything consuming one works on the other unchanged.

## Reading it

- **Both halves of the hurdle are separate columns.** `*_pct_zero` is the probability of *zero* production; the point estimate and bands are **conditional on the agent selling anything at all**. Do not read the point estimate as an unconditional forecast -- multiply by `(1 - pct_zero/100)` for that.
- Bands nest: `min_95 <= min_80 <= ... <= min_50 <= point <= max_50 <= ... <= max_95`.
- Levels are 50/65/70/75/80/95 -- a deliberately uneven, user-chosen menu.
- Volume columns are whole dollars (`int64`); units carry one decimal (`float64`).

## Columns

| # | column | dtype | example |
|---|---|---|---|
| 1 | `mls_agent_id` | `str` | EXAMPLE002 |
| 2 | `office_name` | `str` | Example Realty Harbor |
| 3 | `agent_city` | `str` | Rivertown |
| 4 | `agent_state` | `str` | XX |
| 5 | `snapshot_date` | `str` | 2026-07-01 |
| 6 | `volume_pct_zero` | `float64` | 39.3 |
| 7 | `volume_point_estimate` | `int64` | 1251000 |
| 8 | `volume_min_50` | `int64` | 726000 |
| 9 | `volume_max_50` | `int64` | 2092000 |
| 10 | `volume_min_65` | `int64` | 475000 |
| 11 | `volume_max_65` | `int64` | 2492000 |
| 12 | `volume_min_70` | `int64` | 375000 |
| 13 | `volume_max_70` | `int64` | 2652000 |
| 14 | `volume_min_75` | `int64` | 250000 |
| 15 | `volume_max_75` | `int64` | 2852000 |
| 16 | `volume_min_80` | `int64` | 75000 |
| 17 | `volume_max_80` | `int64` | 3133000 |
| 18 | `volume_min_95` | `int64` | 63000 |
| 19 | `volume_max_95` | `int64` | 4353000 |
| 20 | `units_pct_zero` | `float64` | 18.0 |
| 21 | `units_point_estimate` | `float64` | 3.4 |
| 22 | `units_min_50` | `float64` | 2.0 |
| 23 | `units_max_50` | `float64` | 5.7 |
| 24 | `units_min_65` | `float64` | 1.3 |
| 25 | `units_max_65` | `float64` | 6.8 |
| 26 | `units_min_70` | `float64` | 1.0 |
| 27 | `units_max_70` | `float64` | 7.2 |
| 28 | `units_min_75` | `float64` | 0.7 |
| 29 | `units_max_75` | `float64` | 7.8 |
| 30 | `units_min_80` | `float64` | 0.2 |
| 31 | `units_max_80` | `float64` | 8.5 |
| 32 | `units_min_95` | `float64` | 0.0 |
| 33 | `units_max_95` | `float64` | 11.8 |

## Synthetic sample

Three fabricated agents spanning the range — see [`forecasted_sales_SAMPLE_synthetic.csv`](forecasted_sales_SAMPLE_synthetic.csv). **No real agent, office, or prediction appears in this repository.**
