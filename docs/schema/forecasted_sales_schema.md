# `current_agent_sales_forecast.csv` — schema

**Model:** Forecasted Sales  
**Grain:** one row per active agent  
**Columns:** 33  
**Rows in the current MA run:** 27,019

One row per active agent. Next-12-month dollar volume and unit count as a calibrated range, with both halves of the hurdle (P(zero) and the conditional estimate) reported separately.

> The real file is **not in git** — it contains named agents with per-person predictions. This schema and the synthetic sample beside it define the contract; regenerate the real file with `score_agents.py`. RI and MA share this schema exactly, so anything consuming one works on the other unchanged.

| # | column | dtype | example |
|---|---|---|---|
| 1 | `mls_agent_id` | `str` | `EXAMPLE001` |
| 2 | `office_name` | `str` | `Example Realty C` |
| 3 | `agent_city` | `str` | `Springfield` |
| 4 | `agent_state` | `str` | `XX` |
| 5 | `snapshot_date` | `str` | `2026-07-01` |
| 6 | `volume_pct_zero` | `float64` | `32.726` |
| 7 | `volume_point_estimate` | `int64` | `30` |
| 8 | `volume_min_50` | `int64` | `4` |
| 9 | `volume_max_50` | `int64` | `4` |
| 10 | `volume_min_65` | `int64` | `16` |
| 11 | `volume_max_65` | `int64` | `1` |
| 12 | `volume_min_70` | `int64` | `11` |
| 13 | `volume_max_70` | `int64` | `18` |
| 14 | `volume_min_75` | `int64` | `15` |
| 15 | `volume_max_75` | `int64` | `33` |
| 16 | `volume_min_80` | `int64` | `35` |
| 17 | `volume_max_80` | `int64` | `33` |
| 18 | `volume_min_95` | `int64` | `17` |
| 19 | `volume_max_95` | `int64` | `3` |
| 20 | `units_pct_zero` | `float64` | `23.075` |
| 21 | `units_point_estimate` | `float64` | `1688152.0` |
| 22 | `units_min_50` | `float64` | `519258.0` |
| 23 | `units_max_50` | `float64` | `3530640.0` |
| 24 | `units_min_65` | `float64` | `1911290.0` |
| 25 | `units_max_65` | `float64` | `1737297.0` |
| 26 | `units_min_70` | `float64` | `1335045.0` |
| 27 | `units_max_70` | `float64` | `1587419.0` |
| 28 | `units_min_75` | `float64` | `1873433.0` |
| 29 | `units_max_75` | `float64` | `2556811.0` |
| 30 | `units_min_80` | `float64` | `3079370.0` |
| 31 | `units_max_80` | `float64` | `1775692.0` |
| 32 | `units_min_95` | `float64` | `636744.0` |
| 33 | `units_max_95` | `float64` | `255884.0` |
