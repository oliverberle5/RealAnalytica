# Handoff: Agent Likelihood-to-Leave Model

Read this fully before touching code. This project has accumulated a lot of
non-obvious decisions that are easy to accidentally undo. When in doubt,
check the rationale below before "fixing" something that looks wrong.

## Goal

Predict each real estate agent's percent chance of switching brokerages,
for a quarterly-refreshing dashboard. Output per agent: a calibrated
percent chance, a risk multiplier vs. base rate, and a 5-tier quintile
label (Low/Medium-Low/Medium/Medium-High/High). Consumers use it for
rank-based (top-N/decile) retention outreach prioritization, not hard
probability thresholds.

## Data

- **`../leave-dataset_most_recent_updated.csv` is now the live source**
  (2026-07 update, replaces `leave-dataset-riar.csv`). It bundles RIAR (RI)
  together with MLSPIN (MA) rows (696,764 total) plus new macro/company
  columns — `data.py::load_clean` filters `mls_code == "riar"` immediately,
  per user decision to stay RI-only for now. The RIAR subset is 84,417
  rows — same as the old file minus 55 rows that were verified
  byte-identical exact duplicates in the old file (confirmed: 0 duplicates
  of any kind — full-row or (agent, snapshot) key — in the new file, both
  RIAR-only and combined). No data was lost; old file kept on disk,
  untouched, not read by anything anymore.
- New columns of note: `state_unemployment_rate` (varies by MLS-market
  state, i.e. RI vs MA — NOT by the agent's personal mailing-address
  state, which is a distinct field and often out-of-state; confirmed
  correct, not a bug), `state_unemployment_change_12m_pts`, `cpi_yoy_pct`
  (national, no state grain), `market_total_volume_12m/_trend`,
  `market_total_sides_12m`. All have 100% coverage across all 20 quarters
  — the old pre-2024-06 market-column gap is closed. `market_median_price_12m`
  no longer exists — renamed `market_avg_price_12m` upstream (average, not
  median — a metric change, not just a rename).
  `market_avg_dom_12m` is the one column that STILL has a real coverage gap
  (now pre-2024-01, was pre-2024-06) — unchanged issue, see decision #4.
  Company-level rollup columns (`company_volume_12m`, `company_exit_rate_12m`,
  etc., parallel to the existing `office_*` set) also shipped but are NOT
  yet wired into the feature set — flagged as a future addition, not done
  this session (out of the explicit scope of "market data + M&A policy").
- `../likelihood-to-leave-columns (1).csv` — source data dictionary (may be
  stale against the new file's added columns — not yet re-checked).
- Explicitly out of scope, no columns exist for these: spouse relocation,
  benefits, commission split (deferred, no data source yet — note for
  later). Leadership/M&A/rebrand/acquisition changes are now handled
  directly via label recoding — see decision #5 (rewritten this session).
  Inflation/unemployment macro data has arrived and is wired in (see above).

## Environment

`../.venv` (Windows venv). Activate via `../.venv/Scripts/python.exe`.
Installed: pandas, numpy, scikit-learn, matplotlib, xgboost 3.3.0,
catboost 1.2.10. NOT installed: lightgbm, lifelines, shap (were in the
original plan, never actually used — see "Models" below for why LightGBM
was dropped in favor of XGBoost).

## Critical decisions already made (don't relitigate without reason)

1. **3-month horizon, not 12-month.** Replaced the original 12-month
   framing entirely. Unlocks ~94% of rows vs. ~78%, reaches quarters closer
   to present. Base rate is ~2.4% (not ~9.6%) — this is expected and
   correct, not a bug. See `data.py` module docstring.
2. **Censoring rule: drop ALL `label_censored==1` rows uniformly**,
   regardless of label value. Do not special-case "but we already know
   this one left" — that biases the recent edge of the data. See
   `data.py` docstring, empirical proof (841 rows).
   **Verified 2026-07: no over-censoring.** Traced the resolvability math
   against raw data — the conservative "max observed date" lower bound the
   code computes (last `label_censored==0` snapshot + 12mo) lands exactly
   on the true last snapshot date (2026-06-30 in the old file), i.e. it
   isn't underestimating anything. Every quarter from 2021-09-30 through
   2026-03-31 comes out 100% resolvable under the 3-month reconstruction.
   Only the very last snapshot is (correctly) ~99.8% unresolvable, since
   fewer than 3 months have elapsed since it — the genuine edge of what's
   knowable, not a bug.
3. **`heuristic_recruitability_score` is fully discarded** — not a
   feature, not a benchmark. User called it "bullshit, delete it
   entirely."
4. **`avg_dom_12m` (per-agent) is EXCLUDED — confirmed broken**, not just
   noisy: 59% zeros, median 0, implausible days-on-market distribution.
   Confirmed still broken in the new 2026-07 data file too (same pattern).
   Replaced by `market_avg_dom_12m` (state-level, sanity-checked as
   legitimate: 52-70 day range) for the days-on-market/market-heat signal.
   See `data.py` RAW_FEATURE_COLUMNS comment. **User's stated future plan
   (not yet done): once we have the best model, re-run it on the subset
   where `avg_dom_12m != 0` to test whether it carries real signal despite
   the zero-inflation, before deciding whether it's worth fixing at the
   source.**
5. **Rebrand/M&A label correction is live** (`rebrand_classifications.py`,
   wired into `data.py::_recode_rebrand_moves`). Detected via
   `detect_transitions.py` (mass-simultaneous same-origin-to-same-
   destination office transitions — no web search needed to find
   candidates, only to classify them).
   **Policy changed 2026-07 (user-directed):** originally only
   same-ownership rebrands recoded to 0 (label unaffected by genuine M&A).
   Now ANY transition where the destination is a continuation of the same
   underlying business — rebrand, M&A/acquisition (even where ownership
   changed), or a same-brand-family roll-up/consolidation — recodes to 0.
   Rationale: user considers rebrand/M&A/acquisition/leadership-change all
   "everyday part of the business," not a voluntary departure. This moved
   6 pairs from `GENUINE_PAIRS` into `REBRAND_PAIRS` (Lamacchia's KSPG and
   TIRR acquisitions, plus 4 Century-21 roll-up pairs), taking the total
   recoded from 120 rows (8 pairs) to 228 rows (14 pairs).
   **Superseded again, same day (2026-07 #2, user-directed):** the
   REBRAND-vs-GENUINE distinction above is gone. User's call: ANY
   mass-simultaneous same-origin-to-same-destination cluster (>=5 agents,
   same quarter, as detected by `detect_transitions.py`) reflects a
   team-lead's or office's collective/corporate decision, not individual
   agent free will — regardless of whether the destination brokerage has
   any confirmed relationship to the origin. This now also recodes the 4
   previously-"genuine" pairs (RESI→CMPM, CMWT10→EXPW03, CMWT03→LVGP,
   KELW03→SMRT) and the 5 unconfirmed/ambiguous pairs — all 23 distinct
   detected pairs (27 quarter-instances) are now excluded from churn.
   Mechanically: `rebrand_classifications.py` still keeps the original
   `REBRAND_PAIRS` / `GENUINE_PAIRS` / `AMBIGUOUS_UNCONFIRMED_PAIRS`
   buckets for historical audit trail (the underlying research into which
   pairs are true M&A vs. unrelated-competitor moves is still true and
   still documented), but a new `NOT_CHURN_PAIRS = {**REBRAND_PAIRS,
   **GENUINE_PAIRS, **AMBIGUOUS_UNCONFIRMED_PAIRS}` is what `data.py`
   actually uses now (renamed `_recode_rebrand_moves` →
   `_recode_mass_mover_moves` to match). Verified: all 23 pairs
   `detect_transitions.py` currently surfaces are covered, and all 27
   quarter-instances already have prior documented web research — nothing
   new to verify this round.

   **Bug caught and fixed same day**: the first implementation matched on
   the (origin, dest) office pair ALONE, no per-quarter check — once a
   pair was in `NOT_CHURN_PAIRS`, ANY agent move between those two
   offices, in ANY quarter, got waved through, including a lone unrelated
   departure with no connection to the actual flagged cluster event. Fixed
   by requiring BOTH (a) the pair is in `NOT_CHURN_PAIRS` AND (b) THIS
   SPECIFIC quarter independently had >= `MIN_CLUSTER_MOVERS` (5, a
   constant now shared with `detect_transitions.py` so the two thresholds
   can't drift apart) simultaneous movers on that exact pair — see
   `data.py::_recode_mass_mover_moves`. A lone agent independently making
   the same RESI→CMPM move outside of a genuine cluster quarter still
   counts as real churn.
   Recoded row count: 228 (14 pairs, round 2) → 368 rows under the buggy
   blanket-pair version → 278 rows under the (then-)corrected, quarter-gated
   version (confirmed ~90 rows were being wrongly recoded by the bug).

   **Refined again, same day (2026-07 #3, user-directed diligence check):**
   user asked to justify `MIN_CLUSTER_MOVERS=5` and, while checking that,
   we found the 278-row quarter-gated version was itself too strict for
   pairs we'd already confirmed via web research: `KSPG→MGLM`, `TIRR→MGLM`,
   the `CBRB` internal office-code pairs, and `KELW03→SMRT` all had
   additional 3-4-mover "trickle" instances in adjacent quarters that were
   still being counted as real churn purely because that specific quarter
   didn't independently clear 5 movers — even though we already have
   independent evidence the pair itself is a non-event. Split the policy
   into two gates (`rebrand_classifications.py`):
   - `CONFIRMED_NOT_CHURN_PAIRS` (the 14 web-verified rebrand/M&A/roll-up
     pairs, was `REBRAND_PAIRS`): recode EVERY move on that pair, ANY
     quarter, ANY volume — independent evidence already settles it.
   - `CLUSTER_GATED_PAIRS` (the 9 confirmed-unrelated-competitor +
     unconfirmed pairs, was `GENUINE_PAIRS` + `AMBIGUOUS_UNCONFIRMED_PAIRS`):
     still requires the specific quarter to independently clear
     `MIN_CLUSTER_MOVERS` — for these, the mass-mover pattern IS the only
     evidence we have, so only the actual flagged instance(s) count.
   `data.py::_recode_mass_mover_moves` now checks both gates.
   Recoded row count: 278 (round-3, over-strict) → **335 rows (round 4,
   final)**.

   **MIN_CLUSTER_MOVERS=5 justification (the question that surfaced the
   above):** among all 1,650 same-origin/same-dest/same-quarter clusters
   in the panel, 88.3% are singleton moves, 98.4% are <=4 movers, and only
   1.6% (27 instances / 23 pairs) clear 5+. Mean/median/max among the 27
   flagged instances: mean 10.3, median 7, max 56 (SMRT→RMREV). Honestly:
   there is NO sharp statistical elbow exactly at 5 — the decay from 2 to
   3 to 4 to 5 movers (110, 38, 18, 7) is a smooth roughly-halving curve,
   not a cliff, so 4 or 6 would be almost as defensible on distribution
   shape alone. 5 is justified as "top 1.6%, comfortably past where
   organic coincidence dominates," not as a provably optimal cutoff. This
   matters less than it used to, now that confirmed pairs (round 4) don't
   depend on the threshold at all — it only gates discovery and the
   cluster-only pairs.

   **HomeSmart-family pattern checked and confirmed genuine** (surfaced
   while investigating the trickle-instance issue, never actually a
   `detect_transitions.py` candidate since it never cleared 5 movers):
   `SMRT` (HomeSmart Professionals, owner Dean deTonnancourt) → `HSFC`
   (HomeSmart First Class Realty, owner Ryan Cook, MA) and `HSHR`
   (HomeSmart Heritage Realty, owner Jason Araujo, MA) → `SMRT` are BOTH
   different independent franchisees under the same international
   "HomeSmart" brand, not the same underlying business as Dean
   deTonnancourt's RI operation — correctly left as real churn, no
   action needed.

   The original (narrower) fix took XGBoost/CatBoost fold-5 AUC from ~0.63
   (looked like model failure) to ~0.74 (best fold) — see "Results" below.
   **This candidate list is NOT exhaustive** — only investigated the top 27
   candidates with ≥5 simultaneous movers (re-checked against the new data
   file 2026-07: still exactly 27, nothing new). Re-run
   `detect_transitions.py` whenever new quarters of data arrive; new
   candidates need manual web verification before being added to either
   `REBRAND_PAIRS` or `GENUINE_PAIRS`.
6. **Ratio/trend features added** (`data.py::add_derived_ratios`):
   `office_headcount_trend_12m`, `units_trend_12m`,
   `office_arrival_rate_12m`, `share_of_office_volume_trend_12m` — all
   derived from existing columns, no new data needed. Gave modest gains.
   Audited-but-not-addable-without-new-data: office-level unit-count trend,
   agent-level price trend, rank/percentile trend (would need a panel-lag
   join, different mechanism than a same-row ratio).
   **2026-07 addition — market-relative performance ratios:**
   `volume_trend_vs_market_12m` (agent's own volume_trend_12m /
   market_total_volume_trend_12m), `units_trend_vs_market_12m` (same idea,
   unit counts), `office_volume_trend_vs_market_12m` (the agent's OWN
   OFFICE's trend vs. the market's, to separate "underperforming a healthy
   office" from "whole office swept up in a market downturn"). Directly
   targets the "agent lost 12% but the market lost 25%, so relatively they
   had a good year" framing. **Result: modest but real signal, not a game
   changer.** `office_volume_trend_vs_market_12m` lands around #17 by
   CatBoost importance (more useful than the raw office/market trend
   columns individually) — the office-vs-market framing carries real
   information. The agent-level `volume_trend_vs_market_12m` and
   `units_trend_vs_market_12m` rank near the bottom of all ~65 features —
   apparently redundant with `office_exit_rate_12m`, tenure, and
   `months_since_last_closing`, which already dominate. Overall 5-fold mean
   AUC is flat versus the pre-2026-07 numbers (CatBoost 0.7255 vs. 0.736,
   XGBoost 0.7251 vs. 0.732) — the macro/market integration plus the
   broadened M&A recoding did not net an improvement, despite the plausible
   business rationale. Not a regression either — still a stable 0.70-0.76
   band across folds, no fold-5-style collapse. Documented honestly rather
   than oversold.
   **2026-07 #4 addition — company-level rollup, office_brand,
   office_exit_concentration_12m, prior-stint history.** An audit of all 83
   source columns found 16 unused ones. User asked to wire in company-level
   columns (parallel to the existing `office_*` set, one level up the org
   hierarchy) plus, after review, 5 more: `office_brand` (categorical
   franchise/independent, one-hot encoded into 15 dummy columns),
   `office_exit_concentration_12m` (HHI-style leaver concentration —
   already shipped in source data; previously only computed ad hoc as a
   one-off in `diagnose_fold5.py`), and `prev_stint_months` /
   `avg_prior_stint_months` / `max_prior_stint_months` (agent's own history
   of how long they stayed at PRIOR offices). Both `office_exit_concentration_12m`
   (null iff `office_exits_12m==0`) and the stint columns (null iff
   `num_offices_career==1`) have confirmed STRUCTURAL nulls, not data bugs.
   Added company-level derived ratios mirroring the office-level ones:
   `company_headcount_trend_12m`, `company_arrival_rate_12m`,
   `company_volume_trend_vs_market_12m` (`data.py::add_derived_ratios`), and
   a `company_exit_rate_12m` monotonic constraint (+1) mirroring the
   office-level one. **Result: a real, non-trivial lift** — CatBoost 0.733 →
   **0.7368**, XGBoost 0.727 → **0.7313**. Feature importance confirms both
   were worth adding: `company_exit_rate_12m` ranks **#7 of 94** features,
   `office_brand_independent` ranks **#9**. Small brand categories (Redfin,
   Corcoran, Sotheby's, Elliman, Berkshire Hathaway — all under ~300 rows)
   show ~0 importance, as expected from sample size.
7. **Time-based (forward-chaining) splits only** — never random k-fold.
   Same agents recur across quarters (repeated panel structure); a random
   split would leak future information into the past. See `split.py` and
   `crossval.py::_forward_folds`.

   **`start_train=9, test_block=2` (5 folds) justified 2026-07 #6** —
   previously an unexamined default with no rationale anywhere, unlike e.g.
   `MIN_CLUSTER_MOVERS=5` which had one. Computed properly this round using
   the Hanley-McNeil AUC standard error formula (`SE(AUC)` as a function of
   positive/negative counts in a fold), on the actual 19 usable quarters
   (2021-10-01 through 2026-04-01; 2026-07-01 is dropped as a degenerate
   <500-row snapshot).

   The real tradeoff: a larger `test_block` gives each fold more positives
   (tighter per-fold AUC estimate) but fewer folds (fewer independent
   stability checkpoints — the fold-5 rebrand-contamination bug was only
   caught *because* there were enough folds to notice one collapse against
   a stable background):

   | test_block | folds (start_train=9) | min positives/fold | per-fold AUC SE (worst fold) | per-fold 95% CI half-width |
   |---|---|---|---|---|
   | 1 quarter | 10 | 75 | 0.0333 | ±0.065 |
   | **2 quarters (current)** | **5** | **158** | **0.0229** | **±0.045** |
   | 3 quarters | 3 | 266 | 0.0177 | ±0.035 |
   | 4 quarters | 2 | 365 | 0.0151 | ±0.030 |

   (`start_train=7` or `11` instead of `9` barely moves these numbers — the
   minimum positive count per fold is set by which calendar quarters happen
   to be leanest, not by exactly where the training window starts.)

   The actual 5 production folds, with real counts:

   | fold | test window | rows | positives | AUC SE | 95% CI half-width |
   |---|---|---|---|---|---|
   | 1 | 2024-01-01..2024-04-01 | 8,343 | 195 | 0.0207 | ±0.041 |
   | 2 | 2024-07-01..2024-10-01 | 8,600 | 170 | 0.0221 | ±0.043 |
   | 3 | 2025-01-01..2025-04-01 | 8,799 | 209 | 0.0200 | ±0.039 |
   | 4 | 2025-07-01..2025-10-01 | 9,099 | 191 | 0.0209 | ±0.041 |
   | 5 | 2026-01-01..2026-04-01 | 9,282 | 158 | 0.0229 | ±0.045 |

   **Verdict: `test_block=2` is a defensible middle ground, not an
   optimum** — `1` is noticeably noisier per fold (±6.5 points) for twice
   as many checkpoints that individually say less; `3` or `4` tighten the
   per-fold estimate somewhat but cut the number of independent stability
   checkpoints to 3 or 2, which is uncomfortably close to "just a single
   train/test split" and would have made the fold-5 collapse harder to
   distinguish from ordinary noise. `2` keeps 5 checkpoints (enough to see
   a pattern) without a min-positives count so low it makes every fold's
   AUC nearly meaningless. **Not changing it** — this also preserves
   comparability with every AUC number already in this document, all
   measured on this exact fold structure.

   **Important caveat surfaced by this exercise, relevant to any future
   model comparison (including the autoresearch loop in
   `../AutoResearch/program.md`):** even pretending the 5 folds were
   statistically independent (they are not — training windows overlap and
   the same agents recur across folds, so this UNDERSTATES the true
   uncertainty), the naive SE of the 5-fold MEAN AUC is ~0.0095, i.e. a 95%
   CI of roughly **±0.019** around any reported mean AUC. Every AUC
   difference discussed in this document so far (CatBoost 0.7368 vs.
   XGBoost 0.7313, or round-to-round deltas of 0.001-0.01) is smaller than
   that confidence interval. None of these comparisons are necessarily
   "real" in a formal statistical sense — they're directionally
   suggestive, repeated across rounds, and consistent with
   feature-importance evidence, which is why they've been trusted, but
   nobody should treat a 0.002 AUC gain as proven superior without also
   checking it holds up across several independent changes, not just one
   comparison. This is the statistical justification behind the
   autoresearch program's "+0.001 mean AUC and no worse than -0.01 on the
   worst fold" keep/discard bar being a practical noise filter, not a
   rigorous significance test.
8. **Monotonic constraints applied** on `office_exit_rate_12m` (+),
   `months_since_last_closing` (+), `share_of_office_volume_12m` (−) for
   both trees — NOT on tenure (its risk curve is non-monotonic, see #9).
   Acts as a regularizer given how few positives exist (~800-2,000/fold).
9. **Tenure is binned, not linear, in the logistic baseline.** Real finding
   (held up across horizon changes and the rebrand fix): risk is highest
   for brand-new hires (0-6mo), bottoms out at 5-10y office tenure, ticks
   back up at 10y+. NOT the "peaks in the middle" bell curve originally
   hypothesized. Trees see raw tenure and discover this themselves.
10. **Class imbalance handled via class-weighting** (`scale_pos_weight` /
    `auto_class_weights="Balanced"`), NOT synthetic oversampling (SMOTE
    rejected — would interpolate between unrelated agent-quarters,
    producing physically meaningless synthetic agents).

## Models: current standings

| model | pre-2026-07 | round 1 (market data + widened M&A, buggy blanket-pair) | round 2 (quarter-gated fix, later found over-strict) | round 3 (confirmed-pairs-any-quarter) | **round 4 / CURRENT (+ company rollup, brand, concentration, stint)** | notes |
|---|---|---|---|---|---|---|
| logistic regression | 0.698 | 0.675 (fold 2: 0.587) | 0.688 (fold 2: 0.607) | 0.679 (fold 2: 0.602) | 0.681 (fold 2: 0.603) | interpretable gut-check only, binned tenure/career, not for production |
| XGBoost | 0.732 | 0.725 | 0.735 | 0.727 | **0.731** | monotonic constraints, class-weighted, fast (~1.4s/fold) |
| CatBoost | 0.736 | 0.7255 | 0.738 | 0.733 | **0.737** | ordered boosting, marginal winner, ~5x slower (~7s/fold) than XGBoost |

Recoded row counts across the mass-mover-policy rounds: 228 → 368 (bug) →
278 (over-strict) → 335 (final, unchanged since round 3 — round 4 only
added features, no further label changes). **CatBoost (round 4) is the
production choice** — 0.7368 mean AUC, back above the original
pre-2026-07 baseline (0.736), consistent 4-of-5-fold winner over XGBoost,
and the speed gap (~7s vs ~1.4s per fold) doesn't matter at a quarterly
retrain cadence. Every fold across both tree models sits in a stable
0.70-0.77 band, no collapse, across all four rounds this session.

**Logistic regression fold-2 anomaly — root-caused, not fixed (user
doesn't plan to use this model in production, diagnosis is for
awareness only):** fold 2 (train 2021-10-01..2024-04-01, test
2024-07-01..2024-10-01) drops to AUC ~0.60, versus ~0.68-0.74 for every
other fold. Cause: the new market/macro columns
(`market_avg_price_12m`, `market_total_volume_12m`, `state_unemployment_rate`,
`cpi_yoy_pct`, etc.) take a SINGLE value shared by every agent within a
given quarter — they're quarter-level, not agent-level. With only 11
distinct training quarters in fold 2, a linear model can fit these
columns as a near-unique "quarter ID," absorbing spurious quarter-level
effects that don't extrapolate to the two unseen test quarters. Confirmed
by ablation: dropping all market/macro columns for fold 2 alone recovers
AUC from 0.602 to 0.673 (not all the way back — some of the drop has
another cause, not investigated further). The largest fold-2 logistic
coefficients by magnitude are literally the market/macro columns
(`market_closed_sides_trend_12m`, `market_avg_price_12m`,
`market_total_volume_12m`, ...), confirming the mechanism directly.
Tree models (XGBoost/CatBoost) are naturally immune to this — they split
on order/thresholds with monotonic constraints and depth/regularization,
not raw magnitude, which is exactly why only the linear baseline showed
the dip. The old code coincidentally excluded these same market columns
from the linear model already (`MARKET_FEATURES_EXCLUDED`, for an
unrelated reason — a coverage gap that's since closed) — removing that
exclusion this session incidentally exposed the collinearity issue.
Not fixed, since the logistic model is a gut-check only; if it's ever
promoted, re-exclude quarter-constant market/macro columns from it
specifically (trees can keep them).

**LightGBM was never built** — ruled out early via reasoning (leaf-wise
growth is the most overfitting-prone of the GBM family on a ~1-2k-positive
minority class) in favor of XGBoost (level-wise, more conservative) and
CatBoost (ordered boosting, built specifically to resist overfitting on
smaller/noisier data). **User's message for this session says "testing
catboost and light models" — clarify whether this means (a) actually
building LightGBM now for a 3-way tree comparison, or (b) shorthand for
the existing XGBoost+CatBoost comparison ("light" = lightweight/fast
models generally). Do not assume — ask.**

Fold-5 collapse (was AUC ~0.63, looked like a fundamental generalization
failure) was root-caused to the HomeSmart Professionals → REMAX Revolution
brokerage rebrand (confirmed via web search, see `rebrand_classifications.py`)
contaminating 120 labels. After the fix, all 5 folds sit in a stable
0.72-0.75 AUC band. This is the single most important result of the
project so far — don't let a future "the model looks unstable in recent
quarters" moment go uninvestigated; check `detect_transitions.py` first.

## Calibration layer (`calibrate.py`)

- Platt scaling (1-D logistic), NOT isotonic or fine deciles — too few
  positives (~800-2k/fold) for those to be stable.
- **Backtest done properly**: fit on folds 1-3's out-of-fold predictions,
  tested on folds 4-5's (never seen by the fit) — not fit-and-eval on the
  same data, which would be circular.
- **Rebuilt 2026-07 #4** against the final feature set (company rollup,
  office_brand, exit concentration, stint history — see decision #6 below)
  and final label definition (335-row mass-mover recode). No code changes
  needed — `calibrate.py` calls into `crossval.py`/`data.py` dynamically,
  so it picked up everything automatically on re-run.
- **New backtest result: reliability is WORSE than before, and the
  per-quarter drift issue is confirmed to persist despite full macro/
  market/company data integration** (matches the expectation set with the
  user when the macro data first arrived — recalibration is inherently
  lagging, and macro features fix rank-order more than aggregate level).
  ECE 0.68% (was 0.51%), and now **4 of 10 deciles are outside their 95%
  CI** (bins 6-9, all overpredicting) instead of just the top bin before.
  Per-quarter: 3 of 4 backtest quarters (2025-07-01, 2026-01-01,
  2026-04-01) show statistically-significant over-prediction; only
  2025-10-01 is within CI. 2025-07-01 is a NEWLY-drifting quarter that
  wasn't flagged in the pre-2026-07 version. Base rate for the production
  calibrator: 2.092% (down from ~2.4% at project start, reflecting all the
  label-diligence work this session — fewer confirmed-non-event rows
  counted as churn). New quintile cut points: 0.722% / 1.144% / 1.787% /
  2.898%.

  **ROOT-CAUSED 2026-07 #5.** Compared the RAW (pre-mass-mover-recode)
  positive rate per quarter against the recode rate (share of raw "moves"
  correctly zeroed out as a confirmed rebrand/M&A/cluster event) per
  quarter. Finding: the raw rate shows NO decline at all — 2025-10 (3.05%)
  and 2026-04 (3.43%) are among the HIGHEST raw rates in the entire panel.
  Agents are not genuinely switching offices less. But the recode rate
  jumps from ~3-8% of raw positives in the calibrator's fit window
  (2024-01 to 2025-04) to **26-48%** in the test window (2025-07 to
  2026-04) — driven by real events (the SMRT→RMREV rebrand rollout,
  Lamacchia's KSPG/TIRR acquisitions, etc.) landing heavily in this
  specific calendar window. The calibrator learned "what a raw score
  means" from a period with much less of this legitimate cleanup
  happening, so it now overpredicts. **This is NOT smooth concept drift
  and NOT random per-quarter noise — it's the lumpy, one-time timing of
  M&A/rebrand events**, not a property of the market or the model.

  **Recency-weighted Platt scaling (user's preferred "professional" fix)
  — TESTED, DID NOT HELP.** Exponential down-weighting of older OOF rows
  (half-life 4 quarters) made things marginally WORSE: ECE 0.6821%
  (unweighted) → 0.7050% (recency-weighted), same 4/10 deciles still
  outside CI. Why it structurally can't work here: the fit window
  (2024-01 to 2025-04) has a FLAT, low recode rate the entire way through
  (3-8%) — there's no gradual ramp for "weight recent rows more" to detect
  and amplify. The regime jump happens entirely AFTER the fit window
  ends. No amount of reweighting historical rows can reveal a regime that
  hadn't started yet when those rows were recorded.

  **Bias-correction patch — TESTED, WORKS, now the production approach.**
  Instead of reshaping old training rows, track the actual miscalibration
  on the most recently COMPLETED quarter and apply it as a multiplicative
  correction to the NEXT quarter's predictions, refreshed every retrain
  cycle. Genuinely held-out test: a calibrator fit on folds 1-4
  (2021-10 to 2025-10) predicted 2.4906% for fold 5 (2026-01/2026-04)
  against an actual of 1.7022% — Brier 0.01627. Applying the bias ratio
  measured on the PRIOR fold (fold 4: predicted 2.4799%, actual 2.0991%,
  ratio 0.8464 — a fold the fold-5 prediction never touched) corrected
  fold 5's prediction down to 2.1082%, improving Brier to 0.01622. Real,
  if partial, improvement — roughly halves the gap. Caveat: it's a
  one-quarter-lagged correction, so a BRAND NEW regime shift (starting the
  quarter right after the bias was measured) would still catch it briefly
  off guard — but it self-corrects every quarter rather than staying stuck
  on a stale multi-year calibration. See `calibrate.py::compute_recent_bias_correction`
  / `score_to_output(bias_correction=...)`. Current correction to apply
  going forward (measured on fold 5, the most recently completed quarter):
  **bias_ratio = 0.6835** — i.e. today's production calibrator is
  overpredicting by roughly 46% in relative terms, apply a ~0.68x
  multiplier to new scores until the next quarterly refresh.
  **Range/uncertainty display added as a complementary, cheaper patch**:
  `score_to_output()` now also returns a `plausible_range` (±15% of the
  bias-corrected percentage) alongside the point estimate — narrower than
  the pre-correction miscalibration (~±20-30%) since the bias correction
  now removes most of the systematic error; this remaining band
  communicates ordinary per-quarter noise, not a formal statistical CI.
- Production calibrator fits on ALL 5 folds pooled (more current than the
  backtest-only version). Quintile cut points computed via equal-population
  quantiles (20% each) on the calibrated percentage.
- **Known gap in the top quintile**: the "High" bucket (top 20%) spans a
  real spread internally — solved via outputting percentage + risk
  multiplier ALONGSIDE the quintile label (`score_to_output()`), not the
  label alone. Not re-measured this round; re-check if precision here
  matters before shipping.
- Confidence intervals in the backtest use Wilson score intervals but
  assume independent rows — not strictly true given repeated agents across
  quarters; true uncertainty is somewhat wider than shown. Documented, not
  yet fixed (would need a cluster-robust approach).

## Next up (user's stated priorities, in the order given)

1. ~~New global housing market/economic data~~ **DONE 2026-07.** Arrived,
   integrated (`state_unemployment_rate/_change`, `cpi_yoy_pct`,
   `market_total_volume_12m/_trend`, `market_total_sides_12m`,
   `market_avg_price_12m/_trend`), lookahead-bias spot-checked against a
   real BLS release (passed). Full market-relative "vs market" ratio
   features also built (decision #6). Net effect on AUC across this
   session's first attempt: flat, not the hoped-for lift on its own — but
   see decision #6's company/brand/stint addition, which DID produce a
   real lift, and calibration below.
   **DONE 2026-07 #4: `calibrate.py` re-run** against the final model.
   Result: the per-quarter drift is CONFIRMED TO PERSIST (worse, actually
   — see Calibration section above) even with full macro/market/company
   integration. Matches the expectation set with the user when the macro
   data first arrived (recalibration is inherently lagging; macro features
   fix rank-order more than aggregate level) — not a surprise, but not
   solved either. Root cause of the drift itself still not identified.
2. ~~Assess whether `avg_dom_12m` (per-agent) is truly broken/unrecoverable~~
   **DONE 2026-07 #4.** Built `explore_avg_dom.py`: tested both the raw
   level and a new quarter-over-quarter trend (lagged panel join, agent's
   own prior-quarter value) on the subset where the value isn't the
   fabricated 0/null. Finding: the genuinely-usable subset is only 1,596
   of 84,417 rows (1.9%, not the ~40% "59% zeros among non-null" framing
   suggested) — too few positives (~25-30 total) for a confident read.
   Neither the raw level nor the trend improved held-out AUC on this
   subset (both came in below a no-avg_dom baseline), but the trend
   consistently ranked far higher in feature importance (#15-18 of ~95)
   than the raw level (#41-74) — suggestive that IF there's real signal
   here, it's in the change over time, not the level, matching the user's
   original hypothesis, but not confirmable with this little data.
   **Recommendation: keep excluding both from production** until the
   source system's DOM computation bug is fixed and restores real
   coverage — revisit then, not before.
3. **Quarterly reset/recalibration pipeline** — formalize what's currently
   ad hoc scripts (`calibrate.py`) into a repeatable "new quarter arrives →
   retrain trees → refit Platt scaling → recompute quintile cutpoints"
   process. Doesn't exist as a single runnable pipeline yet.
4. LightGBM ambiguity resolved: user confirmed "light models" meant the
   existing XGBoost+CatBoost comparison, not a new LightGBM build. No
   action needed here.
5. ~~Company-level rollup columns~~ **DONE 2026-07 #4** — see decision #6.
6. **New, not yet investigated:** logistic baseline's fold 2 AUC dropped
   to ~0.60 (was more stable, ~0.65-0.70 range, before this session's
   changes) — root-caused to quarter-level market/macro columns acting as
   a near-unique quarter ID with only 11 training quarters (see Models
   section), but not fixed since the linear model isn't used in
   production. If it's ever promoted, exclude quarter-constant market/
   macro columns from it specifically.
7. ~~Calibration drift's root cause~~ **DONE 2026-07 #5** — root-caused to
   the lumpy timing of confirmed mass-mover recode events (NOT genuine
   behavior change, NOT noise), recency-weighting tested and rejected,
   bias-correction patch built and adopted as the production approach. See
   Calibration section above. **Not yet done: build the actual quarterly
   retrain/rebias cycle** that would apply this automatically (currently a
   manual `compute_recent_bias_correction()` call) — folds into "Next up"
   item #3, the not-yet-built quarterly pipeline.

## Not yet done (mentioned in conversation, no code exists)

- SHAP-based per-agent explainability (`explain.py` was planned, never
  built).
- A single end-to-end "score new agents" production script — everything
  today is research/comparison scripts (`crossval.py`, `calibrate.py`,
  etc.), not a deployable scoring pipeline.
- Segment-level calibration reliability check (flagged as worth doing,
  not built — global calibration only so far).

## File guide

- `data.py` — the one shared loader everything else imports. Read the
  module docstring fully; it documents the horizon, censoring rule, and
  ratio features.
- `split.py` — time-based train/test split (single split, for quick
  checks).
- `crossval.py` — the 5-fold forward-chaining comparison harness
  (logistic/XGBoost/CatBoost). This is the source of truth for model
  comparison.
- `detect_transitions.py` — finds candidate mass-office-transition events
  from raw agent movement data. Run this whenever new quarters arrive.
- `rebrand_classifications.py` — manually verified (origin, dest) →
  rebrand/genuine/ambiguous lookup table. Extend this after verifying new
  candidates via web search.
- `diagnose_fold5.py` — the diagnostic that found the rebrand issue
  (segment-level AUC breakdown, feature drift, leaver concentration/HHI).
  Reusable template for diagnosing any future "one fold looks bad" moment.
- `train_baseline.py` — logistic regression with binned tenure/career,
  the interpretability gut-check.
- `calibrate.py` — Platt scaling backtest + production calibrator +
  `score_to_output()`.
- `preview_calibration_range.py`, `preview_quintiles.py`,
  `explore_avg_dom.py` — one-off exploratory scripts used to sanity-check
  outputs/hypotheses before committing to an approach; not part of the
  pipeline, safe to ignore/delete. `explore_avg_dom.py` specifically tests
  whether per-agent `avg_dom_12m` (raw and trend) has signal on the
  non-broken subset — see "Next up" item #2's finding (no, not with
  current data coverage).
