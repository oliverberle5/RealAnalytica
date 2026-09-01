# Output schemas

Both production models write one CSV per state. The real files are gitignored (named agents + per-person predictions); these documents are the contract.

| model | file | columns | rows (MA, 2026-07-01) | schema |
|---|---|---|---|---|
| Likelihood to Leave | `current_agent_risk_scores.csv` | 14 | 27,019 | [schema](likelihood_to_leave_schema.md) · [sample](likelihood_to_leave_SAMPLE_synthetic.csv) |
| Forecasted Sales | `current_agent_sales_forecast.csv` | 33 | 27,019 | [schema](forecasted_sales_schema.md) · [sample](forecasted_sales_SAMPLE_synthetic.csv) |

Every `*_SAMPLE_synthetic.csv` is **fabricated**. No real agent, office, or prediction appears in this repository.
