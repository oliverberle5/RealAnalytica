# Connecticut expansion — assessment and deferral

**Date:** 2026-07-28
**Decision:** CT deferred. MA built (see `MA/`). No CT folder was created.
**Reason:** the current dataset contains no Connecticut market. Not "too little
CT data" — no CT market data at all.

---

## What was checked

Source: `Likelihood_to_Leave_Algorithm/leave-dataset-with-team-distinction.csv`
(696,775 rows, the same file both production models use).

The file contains exactly two MLS panels:

| `mls_code` | rows    | market it represents |
|------------|---------|----------------------|
| `mlspin`   | 612,358 | Massachusetts (MLS Property Information Network) |
| `riar`     |  84,417 | Rhode Island (RI Association of Realtors) |

There is no third MLS. There is no Connecticut MLS, SmartMLS, or any CT board.

## Why `agent_state == "CT"` is not a substitute

9,928 rows do carry `agent_state == "CT"` (839 distinct agents), which is what
makes a CT model look superficially possible. It isn't, for three compounding
reasons.

### 1. Those agents are members of the MA and RI MLSs, not a CT one

Of the 839 CT-resident agents, 651 belong to `mlspin` and 188 to `riar`. They are
already inside the MA and RI training populations. They are not a separate
market — they are a slice of the two markets already modeled.

### 2. Every market and macro feature is keyed to the MLS, not the agent

This is the decisive finding. Verified on the 2026-04-01 snapshot: within
`mlspin`, `state_unemployment_rate` has exactly **one** distinct value (4.8) across
every `agent_state` — CT, NH, FL and MA agents all carry it. Same for
`cpi_yoy_pct`, `market_avg_price_12m` ($794,443), `market_total_volume_12m`,
`market_avg_dom_12m` and the rest of the `market_*` family.

So a CT-filtered model would train on rows whose market and macro context is
**Massachusetts'**. There is no CT unemployment rate, no CT price level, no CT
market volume, no CT days-on-market anywhere in the file. A "CT model" built this
way would not be measuring Connecticut; it would be measuring Massachusetts with
a smaller, geographically biased sample.

### 3. The sample is a border-commuter artifact, not a state

The CT agent population is concentrated in exactly the towns on the MA and RI
lines:

| town | rows | border |
|---|---|---|
| Enfield | 1,161 | MA (Springfield) |
| Mystic | 862 | RI (Westerly) |
| Somers | 514 | MA |
| Suffield | 513 | MA |
| Dayville | 373 | NE corner |
| Woodstock | 304 | MA/RI corner |
| West Hartford | 299 | interior |
| Danielson / Putnam / Brooklyn / Thompson | 940 | NE corner |

This is a commuter fringe: agents who live in Connecticut and sell into the
Massachusetts and Rhode Island markets. Hartford, Stamford, New Haven,
Bridgeport, Waterbury — the actual Connecticut market — are essentially absent.
Modeling this population and labeling the result "Connecticut" would misdescribe
what it is.

## Sample size, for completeness

Even setting aside points 1–3, the leave model is not viable at this size:

| | RI (shipped) | CT (hypothetical) |
|---|---|---|
| agents | 5,625 | 839 |
| rows | 84,417 | 9,928 |
| 3-month positives | ~2,168 | **157** |
| positives per test fold (5-fold) | ~430 | **~31** |

The RI project already documents a 95% CI of about ±0.019 on 5-fold mean AUC at
its own sample size, and notes that figure understates the true uncertainty
because folds share training windows and agents. At roughly 1/14th the positive
count, a CT AUC would carry a confidence interval wide enough to be
uninterpretable — any number it produced, good or bad, would be noise.

The forecasted-sales model is less starved (6,479 of 9,928 CT rows resolve to a
clean next-12m target, since that target needs no rare event) but is defeated by
points 1–3 regardless: it would be forecasting CT-resident agents' production
using Massachusetts market covariates.

---

## What would actually unblock CT

A CT data pull with the **same 85-column schema**, containing:

1. **A real CT MLS panel** — SmartMLS is the statewide MLS covering Hartford,
   New Haven, Fairfield and the rest of the state. This needs to arrive as its
   own `mls_code` value (e.g. `smartmls`), with its own agent roster, offices and
   companies.
2. **CT-keyed market columns** — `market_total_volume_12m`,
   `market_avg_price_12m`, `market_avg_dom_12m`, `market_total_sides_12m` and
   their `_trend` siblings, computed over the CT market rather than inherited
   from MA.
3. **CT-keyed macro columns** — `state_unemployment_rate`,
   `state_unemployment_change_12m_pts` for Connecticut. (`cpi_yoy_pct` is
   national and can stay as-is.)
4. **The same 20 quarters** (2021-10-01 through 2026-07-01) so the forward-chaining
   fold structure and the 12-month target self-join work unchanged.
5. **The `is_team` flag and `agent_name`**, so the team-account exclusion that
   both production models rely on carries over.

With that in hand, standing CT up is mechanical and cheap — the MA port done
today changed four things in `data.py` (the `mls_code` filter, the data path, the
synthetic-placeholder agent id, and one office-brand category) plus a regenerated
office-pair file. CT would follow the identical recipe. The blocker is purely the
data pull, not the modeling.

## Interim option, if CT visibility is needed before that pull

The 839 CT-resident agents are already scored today — 651 of them by the MA
model and 188 by the RI model, since they sit inside those MLS panels. A CT view
could be produced right now by filtering the existing
`current_agent_risk_scores.csv` / `current_agent_sales_forecast.csv` outputs on
`agent_state == "CT"`, presented honestly as *"CT-resident agents operating in
the MA and RI markets"* rather than as a Connecticut model. That was offered and
not selected; noting it here so the option is on the record.
