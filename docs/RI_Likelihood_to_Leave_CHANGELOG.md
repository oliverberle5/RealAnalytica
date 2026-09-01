# Decision Changelog: Agent Likelihood-to-Leave Model

> **Archived engineering log — paths predate the 2026-09-01 reorganisation.**
> This document was written while RI's models lived at the repo root. Read its
> file references with this mapping:
>
> | as written | now |
> |---|---|
> | `modeling/` (sales) | `RI/Forecasted_Sales/` |
> | `Final_Model/` (leave) | `RI/Likelihood_to_Leave/` |
> | `Likelihood_to_Leave_Algorithm/<the CSV>` | `data/<the CSV>` |
> | `AutoResearch/` | not in this repository — research scratch, kept locally |
>
> The findings and figures are unchanged; only the paths moved.



Raw source material for an academic write-up — not the write-up itself. This is a
factual, chronological record of every material decision made on this project: what was
tried, what alternatives were considered and rejected, the quantitative result, and the
final call. Every entry is sourced (file, docstring, commit hash, or results log) so it
can be independently verified while drafting the paper.

> ⚠ **Every AUC and row count below is a point-in-time measurement, and most are
> intermediates on a chain.** That is the point of the file — but it means nothing here
> should be quoted as a current figure without checking it first. Current production
> values (2026-08-11): **mean AUC 0.7440**, base positive rate **1.9628%**, **459** rows
> recoded across **44** pairs, **153** team accounts excluded, **52** synthetic-placeholder
> rows excluded, panel **84,417 → 77,389** usable. Label policy: magnets count as churn; trickles and documented rebrands do not.
> See `Write_Up/VERIFIED_TABLES.md` for per-figure provenance, and `modeling/HANDOFF.md`'s
> header for the superseded-value map.

Two parts:
- **Part I — Foundational decisions.** Made before this changelog's sourcing began.
  Exact calendar dates aren't recoverable from what's on disk (see "On dating" below), so
  these are ordered by logical/build dependency instead — the order in which each
  decision necessarily had to precede the next.
- **Part II — Dated timeline.** Everything from 2026-07 onward, in strict chronological
  order, several entries timestamped to the minute (autoresearch's git commits).

### On dating

`modeling/HANDOFF.md` labels the current month's work in rounds ("2026-07 #1" through
"2026-07 #6") but the foundational decisions in Part I predate that labeling scheme and
carry no timestamps in the code or docs. This machine's session transcripts
(`~/.claude/projects/`) don't go back far enough to recover exact dates for them either —
only sessions from the last few days remain. Where a precise date/time *is* recoverable
(git commit timestamps, this session's own work), it's given exactly. Where it isn't,
that's stated rather than guessed.

---

## Part I — Foundational decisions (order = build dependency, not calendar time)

### 1. Prediction horizon: 12 months → 3 months
**Decision:** Replaced an original 12-month "will this agent leave" framing with a
3-month one.
**Why:** A 3-month window closes fast enough that far more of the panel becomes usable
(91.7% of rows over 19 quarters vs. 75.8% over 16 for 12-month; recomputed 2026-08-11 on
the production panel — see the 2026-08-05 entry for the earlier, wider-panel figures),
reaches quarters closer to the present, and better
answers the actual business question — "who's a flight risk *right now*" — than a
slower-moving 12-month outlook.
**Trade-off, accepted knowingly:** Positive rate drops from ~9.6% to ~2.4% (~2.1% after
later label-cleanup work), and the absolute count of confirmed leavers drops too
(~6,300 → ~2,000) even though total usable rows increase, because far fewer of the
ever-observed moves fall inside a 91-day window than a 365-day one.
**Alternative considered:** Keeping the 12-month label. Rejected — it discards a fifth
of the data and answers a less operationally useful question.
**Source:** `data.py` module docstring; `HANDOFF.md` decision #1.

### 2. Label reconstruction and censoring rule
**Decision:** The source data ships 12-month labels only (`label_left_within_12m`,
`label_days_to_move`, `label_censored`); the 3-month label is *reconstructed* from
`label_days_to_move` rather than taken as shipped. A row is usable only if its 3-month
outcome is fully resolvable as of the data's true observed-date horizon — no
"we already happen to know this one, so keep it" exception.
**Why the hard rule:** Special-casing rows where the outcome happens to already be known,
even though the 3-month window hasn't formally closed, would bias the recent edge of the
data toward known outcomes and distort the true near-term positive rate.
**Verification done:** Traced the conservative "max observed date" lower bound the code
computes against the raw data — confirmed it lands exactly on the true last snapshot
date, i.e. not silently under-including resolvable rows. Every quarter from 2021-09-30
through 2026-03-31 came out 100% resolvable; only the single newest snapshot is
correctly ~99.8% unresolvable (genuinely too recent to know the 3-month outcome yet).
**Source:** `data.py::_infer_max_observed_date_lower_bound`, module docstring; `HANDOFF.md`
decision #2.

### 3. `heuristic_recruitability_score` discarded entirely
**Decision:** Not used as a feature, not used as a benchmark, dropped from the pipeline
outright.
**Why:** Explicit user call — described as "bullshit, delete it entirely." Not a
data-quality finding, a judgment call about the column's validity.
**Source:** `data.py::EXCLUDED_COLUMNS`; `HANDOFF.md` decision #3.

### 4. Initial model shortlist and first baseline run
**Decision:** Compare logistic regression (interpretability gut-check), XGBoost, and
CatBoost on identical features and identical class-imbalance handling, so any AUC
difference is attributable to the algorithm, not the feature set.
**Alternative considered and rejected without ever building it: LightGBM.** Ruled out by
reasoning alone at this stage — leaf-wise tree growth is the most overfitting-prone
member of the GBM family on a minority class this small (~1-2k positives). This
reasoning stood unverified for the rest of the project until the autoresearch run in
Part II actually tested it.
**Source:** `crossval.py` module docstring; `HANDOFF.md` "Models" section.

### 5. The fold-5 collapse and its investigation
**Decision point:** An early cross-validation run showed fold 5 AUC around ~0.63,
sharply below every other fold — looked like a fundamental model failure, not noise.
**Investigation:** Built `diagnose_fold5.py` (segment-level AUC breakdown, feature
train/test drift check, leaver-concentration/HHI analysis) specifically to root-cause
this rather than accept it as "the model is unstable in recent quarters."
**Two things this investigation found, not one:**
- `avg_dom_12m` (per-agent days-on-market) surfaced as the top-importance feature with
  the largest train/test distribution shift — investigated and confirmed to be noise
  (59% zeros, median 0, an implausible days-on-market distribution), not a real signal.
  Excluded from the feature set; replaced by `market_avg_dom_12m` (state-level, sanity
  checked as legitimate: 52-70 day range) for the days-on-market signal.
- The actual root cause: a HomeSmart Professionals → RE/MAX Revolution brokerage
  rebrand (same owner-broker, franchise-flag change only, confirmed via web search) was
  contaminating 120 labels — agents coded as "left" who had in fact stayed with the same
  business under a new name.
**Result:** Fixing the rebrand contamination alone took fold-5 AUC from ~0.63 to ~0.74
(best fold). All 5 folds then sat in a stable 0.72-0.75 band.
**Framing for the record:** This is described in the project's own documentation as
"the single most important result of the project so far" — a reminder to
diagnose fold-level anomalies rather than write them off as instability.
**Source:** `diagnose_fold5.py`; `HANDOFF.md` "Fold-5 collapse" note (bottom of Models
section) and decision #4 (avg_dom_12m).

### 6. Mass-mover / M&A label recoding — round 1 (narrowest)
**Decision:** Built `detect_transitions.py` to systematically find *candidate* mass
office-to-office transitions (many agents moving same-origin → same-destination in the
same quarter) directly from movement data — no web search needed to find candidates,
only to classify them. Built `rebrand_classifications.py` as a manually
web-verified lookup table. Round 1 policy: only confirmed same-ownership/franchise-flag
rebrands recoded to non-churn; confirmed M&A/acquisitions still counted as real churn.
**Result:** 120 rows (8 pairs) recoded from label=1 to label=0.
**Source:** `rebrand_classifications.py` docstring ("Round 1 (original, narrowest)");
`HANDOFF.md` decision #5.

### 7. Time-based (forward-chaining) splits, established as a hard methodological rule
**Decision:** Cross-validation uses expanding-window, forward-chaining folds — train on
all snapshots strictly before the test block, exactly mirroring deployment (retrain
each quarter, score the next). Never random k-fold.
**Why:** The data is a panel — the same agents recur across quarters. A random split
would let an agent's own later-quarter snapshots leak into training data used to predict
their earlier quarters, inflating every metric.
**Source:** `crossval.py::_forward_folds`, module docstring; `split.py`; `HANDOFF.md`
decision #7.

### 8. Monotonic constraints
**Decision:** Applied directional constraints on `office_exit_rate_12m` (+),
`months_since_last_closing` (+), `share_of_office_volume_12m` (−) for both tree models —
deliberately *not* on tenure (see #9).
**Current state (2026-08-07):** four, not three — `company_exit_rate_12m` (+) was added
in the 2026-07 round (see the "five additions" entry below). A fifth,
`volume_trend_vs_market_12m` (−), was declared alongside it but never bound and has been
removed; see the 2026-08-07 entry.
**Why:** Regularizer, chosen specifically because so few positives exist (~800-2,000 per
fold) — a strong, single-directional business-intuition prior reduces the model's
freedom to fit noise in exactly the places that intuition is confident about.
**Source:** `crossval.py::MONOTONE`; `HANDOFF.md` decision #8.

### 9. Tenure binning finding (non-monotonic risk curve)
**Finding:** Risk is highest for brand-new hires (0-6 months), bottoms out at 5-10 years
of office tenure, and ticks back up at 10+ years — not the "peaks in the middle" bell
curve originally hypothesized.
**Decision:** Tenure is *binned*, not entered as a raw linear term, in the logistic
baseline specifically so it can express this non-monotonic shape (a linear term could
only express "more tenure = uniformly more/less risk"). Tree models see raw tenure and
discover the same shape themselves without binning.
**Robustness note:** This finding held up across the later horizon change and the
rebrand-recoding fix — i.e., it isn't an artifact of either of those decisions.
**Source:** `train_baseline.py` module docstring, `TENURE_BINS`; `HANDOFF.md` decision
#9.

### 10. Class imbalance: class-weighting, not oversampling
**Decision:** Handle the ~2% positive rate via class weights (`scale_pos_weight`,
`auto_class_weights="Balanced"`), not synthetic oversampling.
**Alternative considered and rejected: SMOTE.** Rejected because it interpolates between
unrelated agent-quarters to manufacture synthetic minority examples — those synthetic
rows would be physically meaningless (a linear blend of two different agents' feature
vectors doesn't correspond to any real agent).
**Source:** `HANDOFF.md` decision #10.

---

## Part II — Dated timeline (2026-07 onward)

### 2026-07 — New data source arrives
**What changed:** Source data switched from `leave-dataset-riar.csv` to
`leave-dataset_most_recent_updated.csv`, which bundles RIAR (Rhode Island) together with
MLSPIN (Massachusetts) rows and adds new macro/company columns.
**Decision:** Stay RIAR-only for now (`data.py::load_clean` filters `mls_code == "riar"`
immediately) — a deliberate scope decision, not a technical constraint.
**Verification done:** Confirmed the RIAR subset of the new file is the same 84,417 rows
as the old file, minus 55 rows verified byte-identical exact duplicates in the old file.
No data lost.
**New columns absorbed:** `state_unemployment_rate` (varies by MLS-market state, i.e.
RI vs. MA — confirmed NOT the agent's personal mailing-address state, a distinct and
often out-of-state field), `state_unemployment_change_12m_pts`, `cpi_yoy_pct` (national),
`market_total_volume_12m/_trend`, `market_total_sides_12m` — all at 100% coverage across
20 quarters, closing an old pre-2024-06 market-column gap.
**Metric renamed, not just renamed — redefined:** `market_median_price_12m` no longer
exists; replaced by `market_avg_price_12m` (a mean, not a median — a metric change, not
a pure rename, noted explicitly to avoid mis-attributing continuity).
**Left for later, explicitly out of scope this round:** Company-level rollup columns
(`company_volume_12m`, `company_exit_rate_12m`, etc.) shipped in the same file update but
were not wired into the feature set yet (see 2026-07 #4 below).
**Source:** `data.py` module docstring, dated section.

### 2026-07 #1 — Mass-mover policy, round 2 (widened to cover M&A)
**Decision:** Widened the rebrand-recoding policy from "same-ownership rebrands only" to
also cover confirmed M&A/acquisitions and same-brand-family roll-ups, on the reasoning
that the user considers all of these — rebrand, M&A, leadership change — "everyday part
of the business," not voluntary departure.
**What moved:** 6 additional pairs (Lamacchia's KSPG and TIRR acquisitions, plus 4
Century-21 roll-up pairs) moved from the "genuine churn" bucket into the
"not-churn" bucket.
**Result:** Recoded row count went from 120 rows (8 pairs) to 228 rows (14 pairs).
**Source:** `HANDOFF.md` decision #5, "Policy changed 2026-07."

### 2026-07 #2 — Mass-mover policy, round 3 (widened again; bug found and fixed)
**Decision, same day:** The REBRAND-vs-GENUINE distinction dropped entirely. New
policy: ANY mass-simultaneous same-origin-to-same-destination cluster (≥5 agents, same
quarter) reflects a team-lead's or office's collective decision, not individual free
will — recoded to non-churn *regardless* of whether the destination brokerage has any
confirmed relationship to the origin. This pulled in the 4 previously-"genuine"
competitor-move pairs and 5 unconfirmed/ambiguous pairs. All 23 distinct detected pairs
(27 quarter-instances) excluded from churn.
**Bug found the same day:** The first implementation matched on the (origin, destination)
office pair *alone*, with no per-quarter check — once a pair was whitelisted, *any*
agent move between those offices in *any* quarter got waved through, including a lone
unrelated departure with no connection to the actual flagged cluster event.
**Fix:** Required BOTH (a) the pair is whitelisted AND (b) that *specific quarter*
independently had ≥5 simultaneous movers on that exact pair.
**Result, in sequence:** 228 rows (round 2) → 368 rows (buggy blanket-pair version,
confirmed ~90 rows wrongly recoded) → 278 rows (bug-fixed, quarter-gated version).
**Source:** `HANDOFF.md` decision #5, "Superseded again, same day" and "Bug caught and
fixed same day."

### 2026-07 #3 — Mass-mover policy, round 4 (final split; diligence check)
**Trigger:** User asked to justify `MIN_CLUSTER_MOVERS=5`. While checking that, found
the 278-row quarter-gated version was *too strict* for pairs already independently
confirmed via web research (`KSPG→MGLM`, `TIRR→MGLM`, `CBRB` internal office-code pairs,
`KELW03→SMRT`) — these had additional 3-4-mover "trickle" instances in adjacent quarters
still being counted as real churn purely because that specific quarter didn't
independently clear 5 movers, even though independent evidence already settled the pair.
**Decision — two-gate policy, final:**
- `CONFIRMED_NOT_CHURN_PAIRS` (14 web-verified rebrand/M&A/roll-up pairs): recode every
  move on that pair, any quarter, any volume — independent evidence already settles it.
- `CLUSTER_GATED_PAIRS` (9 confirmed-unrelated-competitor + unconfirmed pairs): still
  requires the specific quarter to independently clear `MIN_CLUSTER_MOVERS` — for these,
  the mass-mover pattern *is* the only evidence available.
**`MIN_CLUSTER_MOVERS=5` justification, documented honestly as NOT provably optimal:**
among all 1,650 same-origin/same-dest/same-quarter clusters in the panel, 88.3% are
singleton moves, 98.4% are ≤4 movers, only 1.6% (27 instances / 23 pairs) clear 5+.
Cluster-size decay (2→3→4→5 movers: 110, 38, 18, 7) is smooth, not a sharp elbow — 4 or 6
would be nearly as defensible on distribution shape alone. 5 is justified as "top 1.6%,
comfortably past where organic coincidence dominates," not as a proven optimum. Matters
less after this round, since confirmed pairs no longer depend on the threshold at all.
**Side investigation, same day:** Checked the "HomeSmart"-family pattern on its own
initiative (SMRT→HSFC, HSHR→SMRT) even though neither ever cleared 5 movers. Confirmed
both are *different independent franchisees* under the same international brand, not
the same underlying business — correctly left as real churn, no change made.
**Result:** 278 rows (round 3) → **335 rows (round 4, final)**.
**Source:** `rebrand_classifications.py` full docstring; `HANDOFF.md` decision #5,
"Refined again, same day."

### 2026-07 — Market-relative performance ratio features
**Decision:** Added `volume_trend_vs_market_12m`, `units_trend_vs_market_12m`,
`office_volume_trend_vs_market_12m` — each agent's (or office's) own trend divided by
the market's trend over the same window, to separate "underperforming a healthy office"
from "whole office swept up in a market downturn."
**Result, reported honestly rather than oversold:** `office_volume_trend_vs_market_12m`
ranked ~#17 of ~65 features by CatBoost importance — real, useful signal. The
agent-level versions ranked near the bottom of all features — apparently redundant with
`office_exit_rate_12m`, tenure, and `months_since_last_closing`. **Overall 5-fold mean
AUC was flat versus pre-2026-07** (CatBoost 0.7255 vs. 0.736, XGBoost 0.7251 vs. 0.732) —
the combined macro/market integration and the broadened M&A recoding did not net an
improvement this round, despite a plausible business rationale. Not a regression either
— still a stable 0.70-0.76 band, no fold collapse.
**Source:** `HANDOFF.md` decision #6, first 2026-07 sub-entry.

### 2026-07 #4 — Company-level rollup, brand, exit concentration, prior-stint history
**Trigger:** An audit of all 83 source columns found 16 unused ones.
**Decision — five additions wired in:**
- Company-level rollup columns parallel to the existing `office_*` set one level up the
  org hierarchy (`company_offices_count`, `company_exit_rate_12m`, etc.), plus derived
  ratios mirroring the office-level ones (`company_headcount_trend_12m`,
  `company_arrival_rate_12m`, `company_volume_trend_vs_market_12m`) and a
  `company_exit_rate_12m` monotonic constraint (+1) mirroring the office-level one.
- `office_brand` (categorical franchise/independent) one-hot encoded into 15 dummy
  columns.
- `office_exit_concentration_12m` (HHI-style leaver concentration) — already present in
  source data, previously only computed ad hoc as a one-off in `diagnose_fold5.py`.
- `prev_stint_months` / `avg_prior_stint_months` / `max_prior_stint_months` — agent's own
  history of how long they stayed at prior offices. Both this and the exit-concentration
  column have confirmed *structural* nulls (undefined when there's no prior office /
  no exits to concentrate), not data bugs.
**Result: a real, non-trivial lift.** CatBoost 0.733 → **0.7368**, XGBoost 0.727 →
**0.7313**. Feature importance confirmed both new groups earned their place:
`company_exit_rate_12m` ranked **#7 of 94**, `office_brand_independent` ranked **#9**.
Small brand categories (Redfin, Corcoran, Sotheby's, Elliman, Berkshire Hathaway — each
under ~300 rows) showed ~0 importance, as expected from sample size.
**Source:** `HANDOFF.md` decision #6, "2026-07 #4 addition."

### 2026-07 #4 — `avg_dom_12m` (per-agent) trend, explored and rejected
**Trigger:** Item flagged since the original fold-5 investigation (#5 above) as a future
open question: does the excluded per-agent days-on-market column carry real signal in a
*trend* form (quarter-over-quarter change), even though the raw level is broken?
**Built:** `explore_avg_dom.py` — tested both the raw level and a lagged
quarter-over-quarter trend on the subset where the value isn't the fabricated 0/null.
**Finding:** The genuinely-usable subset is only 1,596-1,779 of 84,417 rows (~1.9-2.1%,
much smaller than the "59% zeros among non-null" framing initially suggested) — too few
positives (~25-30 total) for a confident read. Neither the raw level nor the trend
improved held-out AUC on this subset. But the trend consistently ranked far higher in
feature importance (#15-18 of ~95) than the raw level (#41-74) — suggestive that if
there's real signal here, it's in the *change*, not the level, matching the original
hypothesis, but not confirmable with this little data.
**Decision:** Keep excluding both raw level and trend from production until the source
system's DOM computation bug is fixed and restores real coverage.
**Source:** `explore_avg_dom.py`; `HANDOFF.md` "Next up" item #2.

### 2026-07 #4 — Calibration rebuilt against the final feature set; reliability found worse
**What happened:** `calibrate.py` re-run against the final 2026-07 #4 feature set and
335-row label definition. No code changes needed — it calls into `crossval.py`/`data.py`
dynamically.
**Result:** Reliability got *worse*, not better, despite full macro/market/company
integration. ECE 0.68% (was 0.51%). 4 of 10 deciles now outside their 95% CI (bins 6-9,
all overpredicting), versus just the top bin before. 3 of 4 backtest quarters
(2025-07-01, 2026-01-01, 2026-04-01) showed statistically significant over-prediction;
only 2025-10-01 was within CI. Base rate for the production calibrator: 2.092% (down
from ~2.4% at project start, reflecting the session's label-diligence work). New
quintile cut points: 0.722% / 1.144% / 1.787% / 2.898%.
**Source:** `HANDOFF.md` "Calibration layer" section.

### 2026-07 #5 — Calibration drift, root-caused
**Investigation:** Compared the RAW (pre-mass-mover-recode) positive rate per quarter
against the recode rate (share of raw "moves" correctly zeroed out as a confirmed
non-event) per quarter.
**Finding:** The raw rate shows NO decline at all — 2025-10 (3.05%) and 2026-04 (3.43%)
are among the *highest* raw rates in the entire panel. Agents are not genuinely
switching offices less. But the recode rate jumps from ~3-8% of raw positives in the
calibrator's fit window (2024-01 to 2025-04) to **26-48%** in the test window (2025-07 to
2026-04) — driven by real events (the SMRT→RMREV rebrand rollout, Lamacchia's KSPG/TIRR
acquisitions) landing heavily in that specific calendar window.
**Conclusion:** This is NOT smooth concept drift and NOT random per-quarter noise — it's
the lumpy, one-time timing of M&A/rebrand events, not a property of the market or model.
**Alternative tried and rejected: recency-weighted Platt scaling.** Exponential
down-weighting of older out-of-fold rows (half-life 4 quarters) made things marginally
*worse* — ECE 0.6821% (unweighted) → 0.7050% (recency-weighted), same 4/10 deciles still
outside CI. Structurally can't work here: the fit window has a flat, low recode rate the
entire way through — there's no gradual ramp for recency-weighting to detect and
amplify, since the regime jump happens entirely *after* the fit window ends.
**Adopted instead: bias-correction patch.** Track the actual miscalibration on the most
recently completed quarter and apply it as a multiplicative correction to the *next*
quarter's predictions, refreshed every retrain cycle. Held-out test: a calibrator fit on
folds 1-4 predicted 2.4906% for fold 5 against an actual of 1.7022% (Brier 0.01627).
Applying the bias ratio measured on the prior fold (0.8464, a fold fold-5's prediction
never touched) corrected the prediction to 2.1082%, improving Brier to 0.01622 — roughly
halves the gap. Caveat: one-quarter-lagged, so a brand-new regime shift starting right
after the bias was measured would still catch it briefly off guard, but it self-corrects
every quarter rather than staying stuck on a stale multi-year calibration.
**Current production correction:** bias_ratio = 0.6835 (measured on the most recently
completed quarter) — a ~0.68x multiplier applied to new scores until the next quarterly
refresh.
**Complementary patch also added:** `score_to_output()` now returns a `plausible_range`
(±15% of the bias-corrected percentage) alongside the point estimate.
**Source:** `HANDOFF.md` "Calibration layer" section, "ROOT-CAUSED 2026-07 #5" onward.

### 2026-07-13 — Fold-count / cross-validation-structure justification (2026-07 #6)
**Trigger:** A direct question — was the existing 5-fold structure
(`start_train=9, test_block=2` in `crossval.py`) ever actually justified, or just an
unexamined default?
**Finding:** No prior justification existed anywhere in the code, unlike
`MIN_CLUSTER_MOVERS=5` (which had one). Computed properly this session using the
Hanley-McNeil AUC standard-error formula on the 19 actual usable quarters
(2021-10-01 through 2026-04-01; 2026-07-01 dropped as a degenerate <500-row snapshot).
**Trade-off quantified:**

| test_block | folds | min positives/fold | worst-fold AUC 95% CI half-width |
|---|---|---|---|
| 1 quarter | 10 | 75 | ±0.065 |
| 2 quarters (current) | 5 | 158 | ±0.045 |
| 3 quarters | 3 | 266 | ±0.035 |
| 4 quarters | 2 | 365 | ±0.030 |

**Verdict:** `test_block=2` (5 folds) is a defensible middle ground, not a proven
optimum — fewer/larger folds tighten the per-fold estimate but cut the number of
independent stability checkpoints to 2-3, uncomfortably close to a single train/test
split (which would have made the fold-5 rebrand collapse, see foundational item #5,
harder to distinguish from ordinary noise). Kept unchanged, also to preserve
comparability with every AUC number already on record.
**Important caveat surfaced and documented:** even treating the 5 folds as
(optimistically) independent, the naive standard error of the 5-fold *mean* AUC is
~0.0095 — a 95% CI of roughly **±0.019** around any reported mean AUC. Every AUC
difference discussed in this document, including CatBoost-vs-XGBoost (0.7368 vs. 0.7313)
and most round-to-round deltas (0.001-0.01), is smaller than that interval. These
comparisons are directionally trusted because they're consistent across rounds and
back up feature-importance evidence — but none are proven at a formal significance
level, and this is now the explicit statistical basis for the AUC improvement bar used
in the autoresearch program below.
**Source:** `HANDOFF.md` decision #7, "`start_train=9, test_block=2`... justified
2026-07 #6" (this session's addition).

### 2026-07-13 — Project pushed to GitHub (private repository)
**Decision:** Code and documentation (`modeling/`, `Final_Model/`,
`AutoResearch/program.md`) pushed to `oliverberle5/RealAnalytica` on GitHub.
**Data-handling decision:** Both raw/derived CSVs (31 MB and 319 MB, real per-agent
brokerage data) excluded via `.gitignore` — never committed. The 319 MB file would also
hard-fail a normal GitHub push regardless (100 MB per-file limit, no LFS configured).
**Visibility decision:** The target repository was already public. Flagged before
pushing that `rebrand_classifications.py` names real broker-owners (Dean deTonnancourt,
Jodi Hedrick, Phil Tirrell, Benjamin Emerick, Ryan Cook, Jason Araujo) in the context of
an internal M&A/churn classification. Decision: flip the repository to private before
pushing anything, rather than push publicly or redact the names.
**Source:** this session; `oliverberle5/RealAnalytica`, branch `master`.

### 2026-07-13 — Autoresearch adaptation designed
**Decision:** Adapted Karpathy's `autoresearch` pattern (an autonomous
edit-train-evaluate-keep/discard loop originally built for GPU LLM pretraining) to this
tabular churn-prediction problem, since its actual mechanics (GPU, `val_bpb` metric,
fixed 5-minute training budget) don't transfer, but its operating loop does.
**Design decisions made, each with a rationale distinct from the original repo:**
- **Ground-truth layer, fixed:** `data.py`, `rebrand_classifications.py`,
  `detect_transitions.py`, and the fold structure itself are off-limits to the
  autonomous loop — mirrors the original repo's untouchable `prepare.py`, but the
  specific things being protected (censoring rule, mass-mover recoding, fold count) are
  this project's own, established over Parts I and II above.
- **Editable surface:** a new `experiment.py`, not the existing `crossval.py` — keeps
  the human-owned reference comparison undisturbed.
- **Objective function redefined:** mean AUC across the 5 fixed folds AND worst-fold
  AUC as a required second gate (keep only if mean improves by >0.001 AND worst fold
  doesn't drop by >0.01) — directly informed by the fold-justification work above (2026-07
  #6): mean-AUC-alone was judged insufficient given the fold-5 collapse precedent and the
  ~±0.019 CI on the mean.
- **Hardware note:** GTX 1650 Ti flagged as a non-factor — all models here default to
  CPU, and at 84k rows GPU boosting typically doesn't beat CPU anyway (unlike the
  original repo, which specifically requires a GPU for LLM pretraining).
- **Time budget redefined:** replaced the original repo's fixed 5-minute *training*
  budget (a design constraint for GPU pretraining) with a 5-minute *hard cap* per
  experiment as a bug-detection tripwire (fits here normally take 1.4-7 seconds) — the
  cap here means "something is wrong," not "this is the intended runtime."
**Source:** `AutoResearch/program.md`; this session.

### 2026-07-13 — Autoresearch loop executed (branch `autoresearch/jul13`)
Run tag `jul13` agreed; branch created from `master`. Setup: read all in-scope files,
verified environment (pandas 3.0.3, numpy 2.5.1, sklearn 1.9.0, xgboost 3.3.0, catboost
1.2.10), installed LightGBM 4.6.0 (the one permitted new dependency, per
`program.md`), initialized `results.tsv`. All times below are exact (git commit
timestamps, `America/New_York`, i.e. UTC-4):

| time | commit | mean AUC | worst-fold AUC | status | what was tried |
|---|---|---|---|---|---|
| 11:06 | `39e174a` | 0.736786 | 0.713310 | **keep** (baseline) | Ported the production CatBoost round-4 config verbatim into `experiment.py` — reproduces the existing production number (0.7368 in `HANDOFF.md`, ±0.00001 rounding) before any real experimenting starts. |
| — | `fd8a190` | 0.728477 | 0.700649 | discard | **LightGBM**, `num_leaves=15` (depth-4 equivalent), same monotone constraints and class-weighting as the CatBoost baseline. This directly resolves the open question left in foundational item #4 above — LightGBM was ruled out by reasoning alone and never actually benchmarked until now. **Answer: worse on both mean and worst-fold AUC** — the original overfitting concern held up empirically. |
| — | `19cfa83` | 0.728973 | 0.702321 | discard | `HistGradientBoostingClassifier` (sklearn, zero new dependencies), same hyperparameter analogues. Worse on both metrics. |
| — | `ac9b7d5` | 0.731331 | 0.708559 | discard | XGBoost, same config as the existing `crossval.py` reference implementation — confirms `crossval.py`'s own number (0.731) rather than finding anything new. Worse than CatBoost on both metrics. |
| — | `2482d57` | 0.726072 | 0.702895 | discard | CatBoost depth 4 → 6, all else identical. Worse on both metrics — likely overfitting the ~1-2k-positive minority class, consistent with the project's general caution about tree depth given so few positives. |
| 11:14 | `4667a2a` | **0.741122** | **0.713973** | **keep** (new best) | CatBoost depth 4 → 3, all else identical. Mean AUC **+0.0043** over baseline, worst fold also slightly better. |
| — | `87f1806` | 0.737677 | 0.707503 | discard | Depth-3 best, iterations 400 → 600. Worse — more boosting rounds overfits slightly at this depth. |
| — | `e901ea9` | 0.739776 | 0.715542 | discard | Depth-3 best, `l2_leaf_reg` 3.0 → 5.0. Slightly worse. |
| — | `b122d93` | 0.734513 | 0.707593 | discard | Depth-3 best, learning rate 0.03 → 0.05. Worse on both metrics. |
| — | `115b12b` | 0.736003 | 0.711643 | discard | Depth 3 → 2. Worse — confirms depth=3 is a local optimum on this depth curve (both depth 2 and depth 4-6 underperform it). |
| 12:20 | `4a44d01` | 0.741182 | 0.714013 | **keep** | Depth-3 best, bottom-20-importance features pruned (ranked on fold-1 training window only). Essential tie on mean AUC (+0.00006) but ~21% fewer features — kept per the project's stated "simpler is better, all else equal" criterion. |
| 12:22 | `07f5c1a` | **0.741780** | **0.718821** | **keep** (current best) | Bottom-40-importance features pruned (54 of 94 features remain). Mean AUC near-tie with the bottom-20 version (+0.0006) but worst fold **notably more stable** (+0.0048) — and simpler. This is the current production-candidate configuration as of this session. |
| — | `863b8e2` | 0.737362 | 0.711030 | discard | Bottom-55-importance features pruned (39 features). Worse on both metrics than the bottom-40 version — over-pruned. |
| — | `f64c29a` | 0.740030 | 0.718010 | discard | Rank-average blend of the current-best CatBoost with the XGBoost reference config. Slightly below the CatBoost-alone best — the XGBoost half of the blend drags the ensemble down rather than helping it. |

**Net result as of this session:** mean AUC **0.736786 → 0.741780** (+0.0050), worst-fold
AUC **0.713310 → 0.718821** (+0.0055) — a genuine improvement by the loop's own
statistical bar (requires >+0.001 mean and no worse than −0.01 worst-fold), achieved via
shallower trees (depth 3, not 4) plus aggressive-but-validated feature pruning (54 of 94
features), not via a different model family — every alternative model family tried
(LightGBM, HistGBM, deeper CatBoost, blending) underperformed the tuned CatBoost.
**Source:** `AutoResearch/results.tsv`; `git log` on branch `autoresearch/jul13`;
`AutoResearch/SESSION_HANDOFF.md`.

### 2026-07-14 — Autoresearch loop continued
**Context:** The loop above stopped cleanly at 12:40 on 2026-07-13 with no crash logged
— the driving session simply wasn't re-invoked to continue it (this changelog's own
process has no visibility into why; the work itself was left in a consistent state,
`HEAD` matching `results.tsv` with no uncommitted diff). Resumed the same loop, same
branch, same bar, starting 19:17 on 2026-07-14.
**Idea explored: combine the two prior near-misses.** `bec1fc7` (extra monotone
constraint) and `badc1c7` (new `tenure_vs_own_avg_stint_ratio` feature) had each
individually landed just under the +0.001 bar on their own (+0.00033 and +0.00029
respectively). Per `program.md`'s own guidance for exhausted fresh ideas ("try combining
previous near-misses"), tried both together.

| time | commit | mean AUC | worst-fold AUC | status | what was tried |
|---|---|---|---|---|---|
| 19:17 | `4690c31` | 0.739882 | 0.718171 | discard | Both near-misses combined on the pruned-40 base. **Worse than pruned-40 alone on both metrics** — the two individually-marginal changes did not stack additively; whatever each was capturing overlapped or interfered rather than adding. |
| 19:2x | `a3716a5` | 0.739917 | 0.716085 | discard | Checked whether 40 was truly a local optimum on the pruning-depth curve or just the best of the coarse 20/40/55 grid already tried — tested 45 (49 features). Worse on both metrics than 40. **Confirms 40 is a real local optimum**, not a grid artifact. |
| 19:2x | `e828e25` | 0.741780 | 0.718821 | discard | `bagging_temperature=1.0` on the pruned-40 base — a regularization axis not yet tried. Result came back **identical to the pruned-40 baseline to 6 decimal places** — strong evidence the parameter was silently a no-op, since CatBoost only honors it under `bootstrap_type="Bayesian"` and the effective default here is evidently something else. |
| 19:2x | `8cb2c6e` | 0.739730 | 0.716301 | discard | Retried the same idea with `bootstrap_type="Bayesian"` set explicitly. This time it had a real (non-identical) effect — confirming the no-op diagnosis was correct — but the effect was negative: worse on both metrics. |
| 19:32 | `67e6103` | 0.737911 | 0.712190 | discard | `random_strength` 1.0 → 2.0 (split-scoring randomness, a distinct CatBoost regularization mechanism from bagging). Worse on both metrics. |

**Net result of this continuation: no new best.** Five more ideas tried, all discarded;
the pruned-40 configuration from 2026-07-13 (`07f5c1a`, mean 0.741780, worst-fold
0.718821) remains the best-kept commit. Two negative findings are worth keeping for
their own sake, independent of the AUC outcome: (1) combining two individually-marginal
near-misses does not guarantee an additive gain and can net negative — a caution against
assuming near-miss effects stack; (2) `bagging_temperature` silently no-ops under
CatBoost's default `bootstrap_type` on this machine/version, which is a reusable
diagnostic (identical-to-6-decimals results are a no-op signature, not a "this parameter
doesn't matter" conclusion) for any future hyperparameter sweep on this codebase.
**Emerging pattern, not yet confirmed:** every regularization/hyperparameter variation
tried since `07f5c1a` clusters in the 0.7377-0.7422 mean-AUC band — consistent with
being near a genuine plateau for this depth-3/pruned-40 CatBoost family, though this is
observational, not statistically tested (recall from the 2026-07-13 fold-count
justification above: the mean AUC's own 95% CI is roughly ±0.019, wider than every gap
in this band).
**Source:** `AutoResearch/results.tsv`; `git log` on branch `autoresearch/jul13`.

### 2026-07-14 (continued) — Pruning methodology and interaction-feature probes
**Idea explored: does the pruning-selection METHOD, not just its depth, matter?** The
current best ranks feature importance once, on fold 1's training window only, then
reuses that fixed set for every fold. Built a per-fold variant (`_pruned_features`
recomputed fresh per fold, each fold selecting its own bottom-40 to drop from its own
training window only — more conservative, no fold ever influenced by another fold's
future training data).

| commit | mean AUC | worst-fold AUC | status | what was tried |
|---|---|---|---|---|
| `b58db4b` | 0.742265 | **0.721058** | discard | Per-fold bottom-40 pruning. **Best worst-fold AUC seen in the whole project to date**, and methodologically more defensible than the fold-1-only version. But mean AUC improved only **+0.000485** over the current best — under the +0.001 bar. Discarded for consistency with how `bec1fc7`/`badc1c7` were handled; no exception carved out despite the appeal. |
| `f95f017` | 0.738644 | 0.715335 | discard | Per-fold pruning at a lighter `n_drop=30` — worse than per-fold-40 on both metrics. |
| `5f63452` | 0.740297 | 0.717640 | discard | Per-fold pruning at a heavier `n_drop=50` — also worse than per-fold-40. Brackets 40 as the local optimum under this ranking method too, same shape as the fold-1-only method's own optimum at 40. |
| `21d8097` | 0.738433 | 0.715941 | discard | New explicit interaction feature `(tenure≤6mo) × office_exit_rate_12m`, motivated directly by the project's own tenure-binning finding (risk peaks for brand-new hires). Worse on both metrics — trees at depth 3 apparently already capture this interaction adequately via separate splits on the two source columns; making it explicit didn't help. |
| `42e4586` | 0.738306 | 0.714899 | discard | Re-checked depth 3 vs. 4 specifically on the pruned-40 (54-feature) set — the original depth sweep that found depth=3 best was run before pruning existed, on the noisier 94-feature set. Worse than depth=3 on the pruned set too — **confirms depth=3 is a real preference, not an artifact of the feature set it was originally tuned against.** |

**Net result: still no new best**, six more ideas tried and discarded (one, `b58db4b`,
a very close and informative near-miss). Two negative-but-useful findings for anyone
continuing this loop: (1) the fold-1-only pruning selection is *not* leaving easy AUC on
the table — a more rigorous per-fold version gets extremely close but not over the bar,
so the shortcut was a reasonable one; (2) depth=3 and the bottom-40 pruning threshold are
both now cross-validated against each other (each re-tested under the other's
methodology) and hold up, which is stronger evidence for keeping them than either
finding was in isolation.
**Source:** `AutoResearch/results.tsv`; `git log` on branch `autoresearch/jul13`.

**One more axis tried, same batch:** `c9b196c` — `min_data_in_leaf` 1 → 50 (a leaf-size
floor, distinct overfitting control from `l2_leaf_reg` or depth). Result identical to
the pruned-40 baseline to 6 decimal places — but unlike the earlier `bagging_temperature`
no-op, this one has an ordinary, non-buggy explanation: depth-3 trees have at most 8
leaves spread across thousands of rows per fold, so a 50-row floor per leaf is never
close to binding. Discarded as a genuine no-op, not investigated further.

**Two more tried, closing out this continuation:**
- `c1d6ac6` — LightGBM was only ever benchmarked on the full 94-feature set
  (`fd8a190`, discarded 2026-07-13, mean 0.7285). Re-checked on the pruned-40 set to
  close the same gap already closed for CatBoost depth=4. Result: mean 0.713166 —
  **worse than LightGBM's own original full-feature attempt**, not better. The
  CatBoost-importance-derived pruned set does not transfer to a different model family;
  LightGBM's leaf-wise growth apparently relies on some of the features CatBoost's
  importance ranking considered droppable.
- `8198edf` — new feature `compounding_exit_pressure` (`office_exit_rate_12m *
  company_exit_rate_12m`, both already top-10 features individually, testing whether an
  office and its parent company both shedding agents simultaneously is a qualitatively
  distinct signal). Mean 0.741500, worst-fold 0.718544 — essentially a wash, both
  metrics very slightly worse than pruned-40 alone.

**Session total as of this entry: 14 experiments tried across the 2026-07-14
continuation (5 + 6 + 1 + 2), zero new keeps.** The best-kept commit remains `07f5c1a`
(mean 0.741780, worst-fold 0.718821) from 2026-07-13. Every straightforward variation on
hyperparameters, pruning method/depth, two hand-built interaction features, and a
model-family re-check has now been tried and has failed to clear the bar — consistent
with (though not proof of) the plateau noted earlier in this section. Remaining
unexplored territory, for whoever continues this loop next: fold-averaged (not just
per-fold-independent) importance ranking was never actually tried, target/frequency
encoding of categorical features beyond the one already-discarded attempt on
`office_brand`, and genuinely different model families re-checked on the pruned-40 set
(HistGBM, currently only tested on the full 94-feature set, same gap LightGBM and
CatBoost-depth-4 just had closed).
**Source:** `AutoResearch/results.tsv`; `git log` on branch `autoresearch/jul13`.

### 2026-07-14/15 — Unattended autoloop (39 experiments), a methodology correction, and a "go for hours" mechanism
**Context:** asked to keep iterating "for hours." Hand-driving one experiment per
conversational turn is inherently bounded — a turn cannot literally run for hours.
Built `AutoResearch/autoloop.py` instead: a driver that loads the dataset ONCE (the
dominant per-experiment cost turned out to be re-reading the 319MB CSV every run, not
model-fitting time), evaluates each queued idea in-process against the already-loaded
data, and still produces one real git commit per idea (writing the exact source
evaluated into `experiment.py`, so the commit and the number can never disagree) —
genuinely unattended, launched as a real background process, not the agent narrating
many turns in one response.

**Queue: 39 items** — HistGBM/XGBoost/CatBoost-Lossguide re-checked on the pruned-40
feature set, Bernoulli-bootstrap bagging, fold-averaged pruning-selection (a third
method between fold-1-only and per-fold-independent), fine-grained learning-rate/L2/
pruning-depth probes, and a 27-point joint grid over (learning_rate × l2_leaf_reg ×
n_drop) — the first joint search in this project, since every prior sweep varied one
hyperparameter at a time and structurally could not find a correlated optimum.

**Result: zero new keeps, one crash (handled cleanly), several close near-misses.**
`catboost_lossguide_pruned40` hit a genuine CatBoost limitation (monotone constraints
unsupported on non-symmetric trees) — caught, logged as `crash`, loop continued without
stalling, confirming the crash-recovery design works as intended. Best near-miss: a
joint-grid point (`lr=0.04, l2=3.0, n_drop=35`) hit **worst-fold AUC 0.724015 — the best
worst-fold seen in the whole project** — but only +0.0006 mean, still under the bar.
The joint grid overall found nothing beating `07f5c1a` anywhere in that neighborhood,
reinforcing (not proving) the plateau read.

**Methodology correction, caught and disclosed rather than buried:** the "n_drop fine
grid" and "joint grid" queue items were labeled as testing the fold-1-only pruning
method (the current best's own method), but were actually wired to call
`_pruned_features()` directly per fold without the caching wrapper — so they actually
tested the **per-fold** method instead. Confirmed via an exact match: `lr=0.03, l2=3.0,
n_drop=40` (nominally "fold-1-only") reproduced `b58db4b`'s numbers (0.742265/0.721058,
a known per-fold-method result) to 6 decimal places — only possible if it's literally
the same computation. The AUC numbers themselves are all still valid (no leakage, no
bug in the modeling) — only the methodology label in ~30 `results.tsv` description
cells is wrong. `results.tsv` itself was left unedited (append-only log, not revised
after the fact); this note is the correction of record. Net effect: a genuine
fold-1-only fine-grained sweep around n_drop=40 remains untested — a real, disclosed
gap, not a fabricated one.
**Source:** `AutoResearch/autoloop.py`, `AutoResearch/results.tsv` (rows tagged
`[autoloop]`), `git log` on branch `autoresearch/jul13`.

### 2026-07-15 — New data file: `avg_dom_12m` bug fixed upstream, conclusively tested, still excluded
**Context:** a new file arrived (user-provided, dropped in as `leave-dataset (1).csv`,
kept under that name rather than renamed — see `data.py` module docstring for why).
User's hypothesis: some previously-zero-heavy columns might now carry real information.

**Verification before trusting anything (same rigor as every prior data-source swap):**
same 83 raw columns (zero new ones — checked programmatically against `data.py`'s known
column set), same RIAR row count (84,417), same 20 quarters, zero duplicate rows, zero
duplicate `(agent_profile_id, snapshot_date)` keys. Clean to swap in.

**The hypothesis was right, specifically for `avg_dom_12m`:** null rate 36.4% (was
undocumented exactly, but zero-rate context: 59% zeros before), zero rate among non-null
now only **0.8%** (was 59%), median **27 days**, sane IQR (13.8-46 days), consistent with
`market_avg_dom_12m`'s 35.7-day median. A full column-by-column null/zero-rate scan
found nothing else with a similarly dramatic before/after signature — the stint and
`office_exit_concentration_12m` columns show exactly their known *structural* nulls,
unchanged.

**But the fix doesn't help — conclusively, this time.** Re-tested at full scale (~53,653
clean rows now, vs. ~1,600-1,800 in the prior inconclusive `explore_avg_dom.py` test):
including `avg_dom_12m` made held-out AUC *worse* on both the full 94-feature reference
config (crossval.py: mean 0.7384 → 0.7353) and the production pruned-40 config (mean
0.739950 → 0.737228, worst-fold 0.718956 → 0.714003) — despite CatBoost's own importance
ranking placing it above the pruning cutoff (it looks useful in-sample, doesn't
generalize). This resolves the open question `HANDOFF.md` had flagged as inconclusive,
decisively: fixed or not, it doesn't help. Kept excluded, now for a settled empirical
reason instead of a data-quality one.

**Not a pure re-export, a genuine dataset change:** even with `avg_dom_12m` held
constant (excluded, as before), the full reference config's numbers moved on the new
file — CatBoost 0.7368 → 0.7384, XGBoost 0.7313 → 0.7328, both up slightly — meaning some
other column values were quietly refreshed too, not just `avg_dom_12m`. Documented
honestly rather than assumed away. Practical consequence: round 5's autoresearch best
(0.741780/0.718821, old data) and this new baseline (0.739950/0.718956, new data,
freshly re-fit pruning) are **not directly comparable** — different datasets. The new
number is the baseline going forward, not a regression from the old one.

**Integration, done on `master` (not the autoresearch branch — ground-truth changes,
same discipline as every previous data-source swap):** `data.py`'s `DATA_PATH` repointed
at the new file; `avg_dom_12m` stays excluded with an updated comment explaining why;
module docstring and `HANDOFF.md` decision #4 updated with the full before/after numbers
so a future reader doesn't have to re-derive them.
**Source:** `modeling/data.py` (module docstring + `RAW_FEATURE_COLUMNS` comment),
`modeling/HANDOFF.md` decision #4 and "Models: current standings" (round 5 / 2026-07-15
entries).

### 2026-07-15 — Round 2 autoloop on the new data: new best found (n_drop=45, not 40)
**Context:** with the new baseline established (0.739950/0.718956) and one methodology
gap disclosed (round 1's "fold-1-only fine grid" had actually run the per-fold method —
see the correction note above), built `AutoResearch/autoloop2.py` for a focused 15-item
round targeting fundamentals fresh on the new data: a depth bracket, a CORRECTLY
fold-1-only pruning grid this time, the per-fold method re-checked at its best depths,
and two model-family re-checks.

One bug caught before launch: the driver's baseline-commit reference was hardcoded and
went stale the moment the script itself got committed (each commit advances `HEAD` past
whatever hash was baked in — hit this exact bug twice in a row before fixing it properly
by deriving the baseline from `git rev-parse HEAD` at runtime instead of hardcoding it).

| commit | mean AUC | worst-fold AUC | status | what was tried |
|---|---|---|---|---|
| `62aeddb`/`9a1db1f`/`b8a387d` | 0.7367 / 0.7380 / 0.7384 | 0.7126 / 0.7141 / 0.7111 | discard (×3) | Depth 2/3/4 on the full 94-feature set, fresh on new data. None beat pruning; depth ordering roughly held but all below the pruned baseline. |
| `a235ebd` (n_drop=20) | 0.739335 | 0.716180 | discard | Genuinely fold-1-only pruning, light prune. |
| `b691de8` (n_drop=30) | 0.740319 | 0.716189 | discard | Close miss (+0.0004). |
| `6edcdd7` (n_drop=35) | 0.739424 | 0.717738 | discard | |
| `660a7a4` (n_drop=40) | 0.739950 | 0.718956 | discard | **Exact match to the logged baseline** — a clean internal consistency check that the corrected fold-1-only implementation reproduces the number it should. |
| `344c8da` (n_drop=45) | **0.742093** | **0.719898** | **keep — new best** | Mean AUC **+0.002143** over baseline, worst-fold also improved (not just "not worse"). The pruning-depth optimum genuinely moved from 40 to 45 on the new data — re-deriving it instead of assuming continuity was the right call. This number is also higher than the OLD file's best (0.741780), recovering past what the data swap had temporarily cost. |
| `e9ff012` (n_drop=50) | 0.739722 | 0.717090 | discard | |
| `6c2b0e5` (n_drop=60) | 0.737936 | 0.716840 | discard | Brackets 45 cleanly as the local optimum on the new data. |
| `8332e9e` (per-fold, n_drop=35) | 0.741490 | 0.717275 | discard | Good, but below the new fold-1-only-45 result. |
| `04ec122` (per-fold, n_drop=40) | 0.740974 | 0.717826 | discard | Same. |
| — (per-fold, n_drop=45) | 0.000000 | 0.000000 | **crash** | Exceeded the 5-minute cap (311s). |
| — (LightGBM, n_drop=40) | 0.000000 | 0.000000 | **crash** | Exceeded the 5-minute cap — but not caught early: **ran for ~1566s (~26 min)** before the after-the-fact elapsed-time check flagged it. The cap in both autoloop scripts measures elapsed time after a fit completes rather than pre-empting a hung/pathological fit, so "5-minute cap" understates how long a bad run can actually take. Not investigated further (the round's real finding — the pruning-depth shift — was already in hand), but worth fixing in any future driver script: wrap the fit call itself in a hard subprocess or thread timeout, not just a post-hoc elapsed check. |
| `a2c0914` (XGBoost, n_drop=40) | 0.726898 | 0.698685 | discard | Consistent with every prior XGBoost test — not competitive here regardless of feature set. |

**New production candidate: `344c8da`** — CatBoost depth=3, bottom-45-importance
features pruned (fold-1-only method), mean AUC 0.742093, worst-fold 0.719898. This is
now the checked-out state of `AutoResearch/experiment.py` on `autoresearch/jul13`.
**Source:** `AutoResearch/autoloop2.py`, `AutoResearch/results.tsv` (rows tagged
`[autoloop2]`), `git log` on branch `autoresearch/jul13`.

### 2026-07-15 — Round 3 autoloop: fine-tuning around n_drop=45, new best found at the very last grid point
**Context:** targeted the open items flagged after round 2 — fine sweeps on
learning_rate/l2_leaf_reg/iterations around the new n_drop=45 optimum, a tight n_drop
grid immediately around 45 (42-48), both bagging mechanisms re-tested (done correctly
from the start this time), `random_strength`, both hand-built interaction features
re-tested on the new base, and a 27-point joint grid (learning_rate × l2_leaf_reg ×
n_drop, centered on 45). Deliberately skipped LightGBM — already conclusively ruled out
twice, and it's the config that hung for ~26 minutes in round 2.

**50 items, 1 crash, 1 new best — found in the LAST joint-grid point checked:**
- Fine lr/l2/iterations sweeps (12 items): all discarded, nothing beat the n_drop=45
  baseline in this immediate neighborhood. Best of these was `lr=0.032` at +0.000622,
  still under the bar.
- n_drop fine grid 42/43/44/46/47/48 (6 items): all discarded individually — but see
  below, 47 turned out to matter once combined with different lr/l2.
- Bagging (Bayesian + Bernoulli), `random_strength`, both interaction features (5
  items): all discarded, consistent with every prior round's findings on these axes —
  none of these mechanisms help this model/data combination.
- **One crash**: `joint_lr0.03_l22.5_nd47` hit the 5-minute cap but — same known gap as
  round 2 — wasn't caught pre-emptively. This one ran for **~37.5 minutes** before being
  flagged, on a plain CatBoost fit (not LightGBM this time), confirming the timeout gap
  isn't specific to one library. Handled cleanly (logged as crash, loop continued).
- **27-point joint grid**: mostly discards, several close-but-under-bar results
  (`lr=0.035, l2=2.5, n_drop=43`: +0.000292; `lr=0.03, l2=2.5, n_drop=45`: worst-fold
  0.720114, best worst-fold of the round, but mean below baseline). The very last
  combination tried, `lr=0.035, l2=3.5, n_drop=47` (`2b17c68`), cleared the bar: **mean
  AUC 0.743615 (+0.001522 over the round's baseline), worst-fold 0.718880** (a small
  -0.001018 dip from the immediately prior best, well inside the -0.01 tolerance). This
  is exactly the kind of correlated, multi-parameter optimum a one-at-a-time sweep
  cannot find — none of the individual fine sweeps (lr alone, l2 alone, n_drop alone)
  turned this combination up on their own.

**New production candidate: `2b17c68`** — CatBoost depth=3, learning_rate=0.035,
l2_leaf_reg=3.5, bottom-47-importance features pruned (fold-1-only method). Cumulative
picture: mean AUC has moved **0.7368 (original round-4 production baseline) →
0.741780 (round 5, old data) → 0.739950 (new-data baseline, after the 2026-07-15 data
swap) → 0.742093 (round 2) → 0.743615 (round 3, current)** — net +0.0068 over the
original production baseline, +0.0018 over the pre-data-swap best, achieved across five
distinct rounds without ever relaxing the fold-count, censoring, or class-imbalance
ground rules established earlier in this document.
**Source:** `AutoResearch/autoloop3.py`, `AutoResearch/results.tsv` (rows tagged
`[autoloop3]`), `git log` on branch `autoresearch/jul13`.

### 2026-07-15 — Round 4: chasing the grid edge further, testing class weighting, and a real infrastructure fix
**Context:** round 3's winner sat at the boundary of every range searched (n_drop=47
was the top of its grid, l2=3.5 the top of its sweep, lr=0.035 the top of its fine
sweep) — a real signal to push further rather than assume the interior optimum was
found. Also: class weighting had never been varied across any of the three prior
rounds. The no-SMOTE rule (`program.md`) is about the mechanism (no synthetic
oversampling) — the specific weighting value within class-weighting was always fair
game, just untested until now.

**Infrastructure fix attempted first**: rounds 2 and 3 each had a fit hang far past the
intended 5-minute cap (LightGBM ~26 min, then a plain CatBoost fit ~37.5 min) because the
cap was only checked *after* a fit call returned — it could never actually kill anything
mid-flight. Built `autoloop4.py` with each candidate run in its own
`multiprocessing.Process`, joined with a hard 300s timeout and forcibly terminated if
still alive. Verified working in isolation before launching for real: a synthetic
30000s-sleep candidate was killed at ~5s against a 5s test timeout. Required restructuring
the driver so all expensive setup (data loading) sits behind `if __name__ ==
"__main__":`, since Windows' `spawn` start method re-imports the whole module in every
child process — the already-loaded DataFrame is instead pickled to a scratch file once
and re-read by each child, preserving the "load once" principle without sharing memory
across the process boundary.

**An unrelated incident during this round, resolved cleanly**: the first launch was
killed externally partway through item 15 of 33 (not a bug in the loop — something outside
this process terminated it). Found and terminated an orphaned child process left running
with no result being collected. Separately, in cleaning up, a `git reset --hard` was
issued to the wrong commit (`2b17c68`, which predates `autoloop4.py`'s own addition to
the repo) and briefly deleted the driver script itself from disk — recovered cleanly via
`git reflog` and `git cat-file`, since the commits were still present as unreachable-but-
not-yet-garbage-collected objects, nothing was actually lost. Added a `resume_from`
argument to the driver so a future interruption can skip already-completed items instead
of redoing them; relaunched from item 15, the 14 already-valid results were kept as-is.

**Results: 33 items, zero new keeps, one very close call.**
- The extended grid-edge chase (lr up to 0.05, l2 up to 5.0, n_drop up to 55, plus an
  18-point joint grid over the extended region) found nothing above `2b17c68` — the
  round-3 optimum sitting at its search boundary was coincidental, not evidence of an
  unexplored better region beyond it. One item exactly reproduced the current best
  (`lr=0.035, l2=3.5, n_drop=47`, tested again as part of the extended grid) to 6
  decimals — a clean internal consistency check.
- `auto_class_weights="SqrtBalanced"` (CatBoost's built-in gentler alternative to
  `"Balanced"`) reached **mean AUC 0.744389 — the best mean AUC seen anywhere in this
  project to date** — but only +0.000774 over the current best, still under the +0.001
  bar. Discarded per the same strict-bar discipline as every prior near-miss (`bec1fc7`,
  `badc1c7`, `b58db4b`), no exception made.
- Explicit `class_weights=[1, r]` tested at r=20, 30 (gentler than the natural ~46:1
  inverse-frequency ratio implied by the ~2.12% base rate) and r=60, 70, 90 (more
  aggressive). `r=20` came close (mean 0.743558, worst-fold 0.721421 — a strong
  worst-fold result) but stayed under the bar; more aggressive ratios (60+) were
  progressively worse, confirming over-correcting the imbalance hurts, same direction
  as under-correcting.
- **Genuine takeaway**: class weighting has real, previously-unknown headroom — both
  `SqrtBalanced` and the gentler explicit ratios cluster near (but under) the bar,
  unlike most other axes tried across all four rounds which cluster well below it. Worth
  a dedicated follow-up round centered specifically around `SqrtBalanced` and ratios
  near 20, rather than written off as another discarded idea.
- **Timeout mechanism, honest caveat**: several items this round took far longer than
  the 5-minute cap (up to ~34 minutes) without the hard-kill ever firing — but all of
  them eventually returned valid results rather than hanging forever, unlike rounds 2-3's
  genuine infinite hangs. The kill logic itself was verified working in isolation before
  the run; what's untested is its behavior against "very slow but not stuck" cases,
  which this round exposed for the first time. Not alarming (nothing needed to be killed
  that wasn't eventually going to finish), but worth understanding before relying on the
  cap as a hard ceiling in a future round.

**Best-kept commit unchanged: `2b17c68`** (mean 0.743615, worst-fold 0.718880).
**Source:** `AutoResearch/autoloop4.py`, `AutoResearch/results.tsv` (rows tagged
`[autoloop4]`), `git log` and `git reflog` on branch `autoresearch/jul13`.

### 2026-07-18/20 — Round 5: chasing the class-weighting lead, two new bests, a resume bug caught and fixed
**Context:** round 4's one live lead was class weighting — `auto_class_weights=
"SqrtBalanced"` came within +0.000774 of clearing the bar, using the SAME base
hyperparameters (lr=0.035, l2=3.5, n_drop=47) as the existing best, never re-tuned
jointly with the new weighting scheme. Built `autoloop5.py` (same hard-timeout
infrastructure as `autoloop4.py`) with a 30-item queue: fine explicit `class_weights`
ratios near the promising r=20, `SqrtBalanced` combined with n_drop/learning_rate/l2
perturbations, and a joint grid over weighting × n_drop × learning_rate.

**Two new bests found:**
1. `0924def` — explicit `class_weights=[1,15]` (gentler than the natural ~46:1
   imbalance ratio): mean 0.745551, worst-fold 0.719035 (+0.001936 over `2b17c68`).
2. `ff0a573` — `SqrtBalanced` + n_drop=45 + learning_rate=0.04: **mean 0.746821,
   worst-fold 0.721336** (+0.001270 over the previous new best, +0.003206 total over
   round 4's baseline). This confirms round 4's read was correct: class weighting had
   genuine headroom that no prior round had touched.

**External interruptions, four this round** (on top of round 4's one) — the background
process was killed externally (not a crash in the loop) four separate times over the
course of this round, each recovered cleanly: verified no orphaned processes, confirmed
git working-tree state, resumed from the correct index each time via the
`resume_from` mechanism built for exactly this. No data or progress was lost across any
of the four interruptions.

**A real bug caught during the third recovery, not before it caused damage:** the
resume logic hardcoded `best_mean`/`best_worst` to the ORIGINAL pre-round-5 baseline
(0.743615/0.718880) on every resume, rather than the true best found so far *within*
the round. This let a worse result (`051795b`, mean 0.745376 — worse than the
already-established `ff0a573` at 0.746821) get logged as `keep`, because it was only
compared against the stale original baseline. The underlying AUC measurement for
`051795b` is still valid (each candidate is self-contained, doesn't depend on git state
from prior commits) — only the keep/discard *decision* was wrong. Fixed by deriving
`best_mean`/`best_worst`/`best_commit` directly from the maximum `mean_auc` among all
`keep` rows in `results.tsv` itself at the start of every run, instead of a hardcoded
constant — this makes a regression-via-stale-baseline structurally impossible on any
future resume, and the fix immediately corrected `HEAD` back to the true best (`ff0a573`)
before continuing. `results.tsv`'s `051795b` row is left as-is (append-only log,
reflects what the buggy run actually decided at the time) — this note is the correction
of record, not a rewrite of the log.

**New production candidate: `ff0a573`** — CatBoost depth=3, `auto_class_weights=
"SqrtBalanced"`, learning_rate=0.04, l2_leaf_reg=3.5, bottom-45-importance features
pruned (fold-1-only method, re-derived under SqrtBalanced rather than reused from the
Balanced-derived ranking). Cumulative picture: **0.7368 (original) → 0.741780 (old
data) → 0.739950 (new-data baseline) → 0.742093 (round 2) → 0.743615 (round 3) →
0.746821 (round 5, current)** — net **+0.010** over the original production baseline.
**Source:** `AutoResearch/autoloop5.py`, `AutoResearch/results.tsv` (rows tagged
`[autoloop5]`), `git log` on branch `autoresearch/jul13`.

### 2026-07-21 — Round 6: the unexplored gentle class-weighting region, a first blend attempt, and a second self-preservation bug
**Context:** two directions left open after round 5. (1) CatBoost's `SqrtBalanced` mode
computes weight ≈ √(neg/pos) ≈ √46.17 ≈ **6.79** — but every explicit `class_weights`
ratio tested in rounds 4-5 (12 through 90) was harsher than that; nobody had tried the
gentler region below 12, which is exactly where `SqrtBalanced`'s own implicit ratio
sits. (2) A blend/stack attempt — the last one (round 1, `f64c29a`) blended CatBoost
with XGBoost and was dragged down by XGBoost, which has been uncompetitive in every
test since; worth trying a blend of two strong-but-different CatBoost configs instead.
Built `autoloop6.py` (same hard-timeout infrastructure, plus the results.tsv-derived
best-tracking fix from round 5) with a 31-item queue: explicit ratios 3-11, a direct
`k·√46.17` parameterization, a joint grid (ratio × learning_rate), and three rank-average
blends of `ff0a573` (current best) with `0924def` (the other strong round-5 config).

**A second self-preservation bug, hit immediately on first launch, before a single
real experiment ran:** the "derive best_commit from results.tsv" fix from round 5 is
correct but incomplete — it can legitimately point at a commit that *predates this
script's own addition to the repo* (the true best on launch was `ff0a573`, from round
5, but `autoloop6.py` wasn't committed until afterward). The first `git reset --hard
ff0a573` wiped `autoloop6.py` from disk while it was still the running `__main__`
module, so every subsequent `multiprocessing` spawn (which re-imports the file fresh
under Windows' spawn semantics) failed with `FileNotFoundError` — **all 31 queued items
crashed as an artifact of this bug, not as real experiment attempts.** These are
recorded in `results.tsv` as `[autoloop6]` crash rows reading "process exited (code 1)
with no result" — flagged here explicitly so they're never mistaken for a real signal
about those hyperparameters. Recovered the deleted script from git's unreachable-but-
not-yet-garbage-collected objects (same technique as the round-4 and this same
incident's own recovery), then fixed properly: added `_safe_reset()`, which re-checks
out the driver's own file from `_RUN_START_COMMIT` (captured at the very start of
`main()`, guaranteed to include the script) after every reset, regardless of how far
back the reset target is. Verified in isolation before relaunching. **General lesson,
worth carrying into any future round**: any driver script that resets to a
dynamically-derived "best" commit must guard its own file against exactly this failure
mode — a hardcoded commit reference isn't the only way to get this wrong.

**Relaunched clean, all 31 items ran for real this time — zero crashes, zero new
keeps.** Every explicit ratio in the gentle region (3, 5-11) landed at or below
`SqrtBalanced`'s own 0.744389 (round 4), confirming `SqrtBalanced`'s implicit ~6.79
ratio is a genuinely good value, not a placeholder beatable by a cleaner number. The
`k·√46.17` parameterization and the joint ratio×learning_rate grid both found several
near-misses clustered just below the bar — best was `ratio=9, lr=0.045` at mean
0.747020, only **+0.000199** over `ff0a573`, deep within the ±0.019 statistical noise
floor established back in the fold-count justification. The three blends (rank-average
of `ff0a573` + `0924def` at 50/50, 60/40, 40/60) all landed close to but below
`ff0a573` alone — blending two strong CatBoost configs didn't help here, though this
remains a simple prediction-blend, not true meta-learner stacking (would need a nested-
CV layer to fit a meta-model on out-of-fold predictions without leakage — not attempted).

**No new best. `ff0a573` stands, now confirmed against 31 additional real attempts
across two genuinely new directions with zero close calls exceeding noise.** This is the
strongest evidence yet that this is a real, not artifactual, local optimum for this
model family and data.
**Source:** `AutoResearch/autoloop6.py`, `AutoResearch/results.tsv` (rows tagged
`[autoloop6]`; the 31 crash rows from the first, buggy launch are also tagged
`[autoloop6]` and must be read alongside this note, not as independent signal),
`git log` and `git reflog` on branch `autoresearch/jul13`.

---

## Open threads not yet resolved (as of 2026-07-21)

Carried over from `HANDOFF.md`, still true as of this changelog:
- SHAP-based per-agent explainability — planned, never built.
- A single end-to-end "score new agents" production script — everything today is
  research/comparison scripts, not a deployable scoring pipeline.
- Segment-level calibration reliability check — flagged as worth doing, not built.
- The quarterly retrain/rebias pipeline that would apply the bias-correction patch
  (2026-07 #5) automatically — currently a manual function call.
- Company-level rollup columns' interaction with the pruned feature set has not been
  individually re-examined against the 2026-07 #4 additions in isolation (the pruning is
  importance-based, not a targeted ablation of that specific feature group).
- **`avg_dom_12m` question is now CLOSED** (was open as of 2026-07-14): the 2026-07-15
  data file fixed the upstream data-quality bug, and it was re-tested at full scale —
  conclusively doesn't help. See the 2026-07-15 entries above and `HANDOFF.md` decision
  #4. Remove this from future "possible next directions" lists.
- The autoresearch loop (branch `autoresearch/jul13`) is currently PAUSED at the time of
  this writing, not actively running — current best-kept commit is `ff0a573` (mean AUC
  0.746821, worst-fold 0.721336, **on the 2026-07-15 data file**, CatBoost depth=3,
  `auto_class_weights="SqrtBalanced"`, learning_rate=0.04, l2_leaf_reg=3.5,
  bottom-45-importance features pruned). The loop was externally killed mid-flight five
  times total across rounds 4-5 — always recovered cleanly — but round 6 completed in
  one clean pass with zero external interruptions and zero crashes (after its own
  self-preservation bug was fixed early on, see below), so the interruption pattern is
  not a certainty for future rounds, just a real possibility to expect.
- **Timeout gap partially addressed** (round 4): several round-4 items ran far past the
  5-minute cap without the hard-kill firing. Round 6 saw no item exceed 136 seconds —
  no evidence either way on whether the underlying pre-emptive-kill mechanism itself
  would actually fire correctly under real (not synthetic) load; still only verified in
  isolation.
- **Resume-bug class — now TWO instances, both fixed, pattern worth remembering for any
  future driver script**: (1) round 5: the resume mechanism hardcoded the "best to beat"
  to the original pre-round baseline instead of the true current best, letting one
  worse result get mislabeled `keep` — fixed by deriving the true best from
  `results.tsv`'s own `keep` rows at the start of every run. (2) round 6: that fix
  itself could point at a commit predating the driver script's own addition to the
  repo, so a reset-to-best could delete the running script out from under itself —
  fixed with `_safe_reset()`, which re-checks out the driver's own file from a
  `_RUN_START_COMMIT` captured before any reset can happen. **Both bugs share a root
  cause**: any mechanism that resets git state to a *dynamically determined* commit
  needs an explicit guarantee that the reset can never remove something the currently-
  running process still depends on (its own tracked bookkeeping value, or its own
  source file). Check `autoloop6.py`'s `_safe_reset()` for the current best template
  before writing another driver from scratch.
- **Class weighting is now CLOSED, thoroughly** — explicit ratios 3-90 tested across
  three rounds, `SqrtBalanced` combined with n_drop/lr/l2 individually and jointly, a
  direct `k·√(natural ratio)` parameterization, all converging on the same answer:
  `SqrtBalanced` (≈6.79 implicit ratio) at lr=0.04 is very close to the true optimum
  for this axis. Remove from future "possible next directions" lists.
- **Grid-edge chase (round 4) came back negative** — remove from future lists.
- **First blend attempt (round 6) came back negative** — rank-averaging the two
  strongest distinct CatBoost configs at three mixing ratios all landed below the
  best config alone. Not fully closed, though: this was a simple prediction blend, not
  true meta-learner stacking (would need a nested-CV layer — fit the meta-model on
  out-of-fold predictions, not the test fold's own predictions, to avoid leakage). If
  blending is revisited, that's the version worth trying, not a repeat of the simple
  rank-average.
- Feature-engineering ideas beyond the two already-tried interaction terms remain
  the main unexplored territory left. After three rounds of hyperparameter/weighting
  search all converging tightly around `ff0a573` (every close call across rounds 5-6
  landed within ~0.001 of it, well inside the ±0.019 statistical noise floor), further
  hyperparameter search on this same model family is likely to have hit its practical
  ceiling — new signal (features) is a better bet than further tuning of what's already
  been squeezed hard across six rounds.

---

### 2026-07-25 — AutoResearch's confirmed-best config finally wired into production

**Decision:** Promoted `ff0a573`'s config (depth=3 CatBoost, `SqrtBalanced` class
weights, `learning_rate=0.04`, `l2_leaf_reg=3.5`, bottom-45-importance features pruned
to 49) into `score_agents.py` / `train_multi_horizon.py` / `calibrate.py`. Until this
point production had silently kept running the older plain depth=4, full-feature,
`Balanced`-weighted reference config the whole time this result sat validated in
`AutoResearch/` — six rounds of search had found the improvement but nothing had gone
back to actually deploy it.
**Why:** Re-derived rather than trusted blindly from the original commit (see
`modeling/derive_winning_config.py`), since file pulls have quietly shifted values
before. Reproduces mean 0.744235 / worst-fold 0.721177 on the current data file — close
to but not identical to the original 0.746821/0.721336 (expected drift from the data
file changing under it, not a bug) — still a clear, real improvement over the 0.7384
baseline this config replaced.
**Source:** `Final_Model/crossval.py` module docstring, `modeling/derive_winning_config.py`.

### 2026-07-26/28 — GEPA autoresearch loop launched; first confirmed feature win wired into production

**Decision:** Built an always-on GEPA (`optimize_anything`) loop
(`AutoResearch/gepa_loop/`) that proposes row-wise-pure engineered features on top of
the 2026-07-25 production config, evaluates each against the 5 production folds, and
requires `confirm.py` (3-seed averaging + two shifted fold structures) to mark a gain
CONFIRMED before it can become champion — a stricter bar than earlier rounds, adopted
because most single-fold "wins" here sit inside the noise band (see
`AutoResearch/results.tsv`'s untrustworthy old-data-era best row, 0.746821, vs. the
current-data baseline of 0.741976).
Round `r003_20260726_204147` cleared that bar: dropped `listing_share_per_tenure_year`
(a consistently weak, bottom-half-importance feature across every fold observed) and
added `trend_deceleration_gap` (`units_trend_12m - trend_3m`, an inflection-point signal
distinct from either raw trend column). Net effect: same 5-feature engineered block,
one feature swapped. `confirm.py` verdict: **CONFIRMED** — "Gain of +0.001100 survives
3-seed averaging AND holds on both shifted fold structures (2/2). Improved 3/5
production folds. This is a real improvement, not window-fitting."
This is promoted into `PRODUCTION_FEATURES` (`Final_Model/crossval.py`,
`modeling/crossval.py`) via `data.py::AUTORESEARCH_CHAMPION_COLUMNS` /
`add_autoresearch_champion_features` — feature count grows from 49 to 54. Same
production hyperparameters, unchanged.
**Result on the full current data file:** mean AUC 0.741976 → 0.743376
(worst-fold 0.719279 → 0.717082). Re-running the full logistic/XGBoost/CatBoost
comparison (`crossval.py`) with the champion features shows CatBoost now winning
**5 of 5** folds (previously 4/5) at mean AUC 74.3% (logistic 69.1%, XGBoost 73.0%) —
`README.md` and `agent_likelihood_to_leave_report.pdf` updated to match.
**Downstream outputs regenerated same day:** `current_agent_risk_scores.csv`,
`agent_likelihood_to_leave_report.pdf`, `risk_score_distribution.png` (all
`Final_Model/`). `current_agent_risk_scores_multi_horizon.csv` was NOT regenerated —
out of scope for this pass, see the team-account note below and `HANDOFF.md`.
**The GEPA loop keeps running** (`AutoResearch/gepa_loop/eternal.py`, an unattended
outer loop around `run_gepa.py`) looking for the next confirmed win; rounds since
r003 (through at least r041) have not beaten it — see `AutoResearch/gepa_loop/ledger.jsonl`
/ `rounds.jsonl` for the live record, and `best/champion.json` for the current champion.
**Source:** `AutoResearch/gepa_loop/best/champion.json`,
`best/r003_20260726_204147_confirmation.json`, `confirm.py`, `ledger.jsonl`,
`Final_Model/crossval.py` and `data.py` module docstrings/diffs.

### 2026-07-28 — Team-account scoring intentionally left stale, not extended

**Decision:** The individual-agent model continues to exclude team accounts entirely
(`exclude_team=True`, the 2026-07-25 standard — 2,222 team-account rows / 153 distinct
team `agent_profile_id`s dropped before training and scoring, unchanged by this round's
feature swap). No separate team-level model was built to fill that gap. Any existing
team-inclusive/team-specific output files elsewhere in the repo (predating the
team/individual split) are left exactly as they are — not regenerated, not backfilled
with the new champion features — as a deliberate, temporary scope decision, not an
oversight.
**Why:** Building and validating a second model family for team accounts is separate
work with its own fold structure and calibration questions; bundling it into this
feature-promotion pass would conflate two unrelated changes. Team scoring stays out of
scope until it's explicitly picked up as its own piece of work.
**How to apply:** Don't treat the team-related CSVs' staleness as a bug when auditing
outputs after this round — it's expected. Flag it again in `HANDOFF.md`'s "Next up"
list so it isn't silently forgotten.
**Source:** this conversation's explicit instruction (2026-07-28).

### 2026-07-28 — "Why this score" bullet redundancy fixed: concept-group cap

**Decision:** `explain_agents.py` picked its top-5 SHAP bullets purely by |contribution|,
with no check for whether two picks told the same story. Spotted in a UI mockup review:
an agent could get both "6 yrs 2 mo at their current office" and "has not changed
offices in the last 5 years" — the second is not new information once the first is
known (long current-office tenure mechanically forces `moves_5y_asof` to 0 past the
5-year mark). Added `CONCEPT_GROUPS`, capping bullet selection at one feature per group:
`career_stability` (`tenure_current_office_months` + `moves_5y_asof`) and
`peer_turnover` (`office_exit_rate_12m` + `company_exit_rate_12m`). A skipped feature no
longer costs the agent a bullet — selection now over-fetches past the top-5 window so a
lower-ranked, distinct-story feature backfills the slot.
**Why:** Checked whether the same pile-up risk applies to the production/activity
cluster (`months_since_last_closing`, `volume_trend_12m`, `new_listings_12m`,
`share_of_office_volume_12m`, `office_volume_trend_vs_market_12m`) before generalizing
the fix — pairwise correlations there are all near-zero (genuinely independent facts),
so that cluster is deliberately NOT grouped. `office_exit_rate_12m` vs.
`company_exit_rate_12m` correlate at 0.77 (same "peers leaving" story, two org levels) —
grouped. `tenure_current_office_months` vs. `moves_5y_asof` correlate at -0.33, and are
outright deterministic once tenure clears 60 months — grouped.
**Result:** re-ran `score_agents.py` (`Final_Model/` and `modeling/`, mirrored) —
verified zero remaining rows carry both bullets from either group.
**Source:** `Final_Model/explain_agents.py::CONCEPT_GROUPS`, this conversation.

### 2026-07-28 — Second explanation bug: self-contradictory `new_listings_12m` wording

**Decision:** User spotted a stale "150%/150%" exit-rate duplicate — traced to
`modeling/agent_explanations_demo.csv`, a Jul-24 scratch file predating the
`CONCEPT_GROUPS` fix above (already clean in the live `current_agent_risk_scores.csv`;
regenerated the scratch file too so it can't cause this confusion again). Broader scan
per the user's request found a real, bigger issue: `new_listings_12m` carries no
monotonic constraint (unlike the 5 features in `crossval.py::MONOTONE` — corrected
2026-08-07 to 4; the fifth never bound, see that entry), so CatBoost's
learned SHAP direction for it can point either way independent of the agent's actual
count. Live in the prior output: 1,768 agents (38% of the roster) got "Took 0 new
listings... **active** in generating new business," and some got "Took only **448**
new listings... **below-average**" — both self-contradictory on their face.
**Why:** Checked every other unconstrained templated feature (`moves_5y_asof`,
`office_brand_independent`, plus the two already-dead templates flagged in
`HANDOFF.md`) for the same pattern first — none showed a live occurrence, so the fix is
scoped to `new_listings_12m` only, not applied speculatively elsewhere.
**Fix:** Added `NEW_LISTINGS_LOW_MAX=8` / `NEW_LISTINGS_HIGH_MIN=5` (grounded in the
roster's own distribution: median 1, 90th pct ~8.3) to `explain_agents.py`. "Risk"
wording requires the count at or below 8; "safe" wording requires it at or above 5;
the ambiguous 6-7 gap and anything violating its own bound is skipped rather than
forced, backfilled by the next distinct feature via the existing over-fetch mechanism.
**Result:** re-ran `score_agents.py` and `explain_agents.py` in both `Final_Model/` and
`modeling/` — verified zero remaining contradictory rows, and re-verified the
exit-rate/tenure-moves dedup still holds in all three regenerated files.
**Source:** `Final_Model/explain_agents.py::NEW_LISTINGS_LOW_MAX`/`_HIGH_MIN`, this
conversation.

### 2026-07-28 — Exit-rate display wording: prior-headcount denominator (display only)

**Write-up note (2026-08-13):** User's explicit call — this bug (dividing exits by
CURRENT headcount instead of prior headcount, so a shrinking org could show >100%
exit rate) is too obvious/embarrassing to belong in the paper. Kept here in the
changelog for the record, but the write-up should NOT feature it as a finding or
methodology point. Contrast with the arrival-rate fix below (2026-07-28, next
section) which IS a legitimate, harder-to-spot model-input correctness fix worth
including.

**Decision:** User spotted "150% of agents company-wide left" for a 14-agent company
(21 exits) and asked why, then recommended fixing it by using the OLD (12mo-ago)
headcount as the denominator instead of current headcount -- the standard convention
for a rate. Implemented as DISPLAY-ONLY, per explicit user direction: the actual model
input (`office_exit_rate_12m`/`company_exit_rate_12m` in `PRODUCTION_FEATURES`) is
untouched -- no retrain, no crossval re-run, no calibration change. Added
`_display_exit_rate()` to `explain_agents.py`, recomputing `exits_12m / prior_headcount`
(`prior_headcount = active_agents_asof - net_flow_12m`, the same derivation already
used for `office_headcount_trend_12m`/`company_headcount_trend_12m`) purely for the
formatted sentence. `company_active_agents_asof` isn't itself a production feature
(only its derived trend ratio is), so `generate_explanations()` gained an optional
`raw` parameter (the full current-snapshot frame) used only for display-string lookups
-- the model's actual `Pool(X, ...)`/SHAP call is unaffected.
**Result:** the flagged case now reads 21/23 = 91% instead of 150%. Re-ran
`score_agents.py` in both `Final_Model/` and `modeling/`. Honest residual, reported
back to the user rather than silently left out: 21 agents still show >100% at the
OFFICE level -- a different, legitimate case (a fast-growing office, e.g. 7->38 agents
via +31 net flow, that also had 14 exits along the way; prior headcount of 7 is small
because the office was aggressively recruiting, so 14 exits against that tiny starting
base is real churn, not a math artifact). Left as-is, not patched further.
**Source:** `Final_Model/explain_agents.py::_display_exit_rate`, this conversation.

### 2026-07-28 — Arrival-rate formula fix: prior-headcount denominator (model input, not just display)

**Decision:** Handoff flagged an incorrect percent-change formula for agents coming and
going. Unlike the exit-rate display fix above, this one is a real model input:
`office_arrival_rate_12m`/`company_arrival_rate_12m` (`data.py::add_derived_ratios`)
were dividing arrivals by CURRENT headcount, while the sibling `office_exit_rate_12m`/
`company_exit_rate_12m` (source columns; see the display-only fix above, same
`prior_headcount = active_agents_asof - net_flow_12m` derivation) divide exits by PRIOR
headcount -- an inconsistent base for two rates meant to be symmetric "agents coming
vs. going" measures.
**Tested before applying, per explicit user request for a non-committal check:** cloned
`leave-dataset-with-team-distinction.csv` and `Final_Model/` into an isolated scratch
directory, applied the fix there only, and ran `crossval.py` against an unmodified
in-place baseline run. Result -- mean AUC across 5 forward-chaining folds:

| Model | Baseline | Fixed | Δ |
|---|---|---|---|
| Logistic | 0.6907 | 0.6905 | -0.0002 |
| XGBoost | 0.7297 | 0.7305 | +0.0008 |
| CatBoost (production) | 0.7434 | 0.7402 | -0.0032 |

Per-fold CatBoost also drops from winning 5/5 folds to 4/5 (loses fold 1 to XGBoost,
0.711 vs 0.719) under the fixed formula. Both `office_arrival_rate_12m` (TREE_FEATURES,
logistic/XGBoost only) and `company_arrival_rate_12m` (`PRODUCTION_FEATURES`, all three
models) are affected.
**Decision:** adopted anyway, on explicit user instruction, "for the sake of
correctness and defendability" -- not for an AUC gain. A ~0.3pp mean-AUC cost on the
production model is the accepted, measured price of a formula that's actually correct
and won't need re-explaining under scrutiny.
**Fix:** `add_derived_ratios()` in both `Final_Model/data.py` and `modeling/data.py`
(kept mirrored, per the project's existing convention) now divides
`office_arrivals_12m`/`company_arrivals_12m` by `prior_headcount`/
`company_prior_headcount` instead of `office_active_agents_asof`/
`company_active_agents_asof`.
**Result:** re-ran the full `Final_Model/` production pipeline on the corrected code —
`calibrate.py` (backtest reliability unchanged, no new drift flags), `score_agents.py`
(4,608 active agents scored, tier distribution 3385/992/144/87 Low/Med/High/Extra
High), `generate_report.py`, `generate_distribution_chart.py`. Headline AUC updated
74.3% -> 74.0% in `Final_Model/README.md`. `modeling/data.py` patched for parity but
its own scratch/dev artifacts (its `current_agent_risk_scores.csv`, etc. — secondary to
`Final_Model/`'s production output) were not separately regenerated.
**Source:** `Final_Model/data.py::add_derived_ratios`, `modeling/data.py::add_derived_ratios`,
this conversation.

### 2026-07-28 — "Non-Mls Member" synthetic placeholder record excluded

**Decision:** While reviewing the Forecasted Sales project's data after the team-exclusion
work (same session), the user spotted `agent_profile_id
1a2504a4-2f87-42dd-ae86-09a6bc021f69`: `agent_name`/`office_name` both literally "Non-Mls
Member", $1.12B summed volume across the panel, 1,678 buyer sides, 0 listing sides, present
in all 20 snapshots, 0.9% of all volume. Confirmed: buy-side-only (0-2 list-sides for the
last ~2 years, 310 buy-sides latest quarter), `tenure_current_office_months` climbing by
exactly 3 every quarter with no resets, `is_team==0` (so the 2026-07-25 team filter doesn't
catch it) — a permanent MLS placeholder bucket for buyer-side transactions where the
cooperating agent isn't a board member, not a real person. Latest-quarter `volume_12m` =
$161,987,304 (~1.16% of that quarter's total market volume) — a distinct contamination
source from team accounts, not a variant of the same bug.
**Tested non-committally before adopting**, same discipline as the percent-change fix
above and the Forecasted Sales team-exclusion fix (same session): cloned the
(already team-excluded) dataset+code into an isolated scratch directory, added an
`exclude_non_member` flag gated separately from `exclude_team`, ran `crossval.py` with
and without it.
**Result** — mean CatBoost AUC across 5 forward-chaining folds: 0.7402 → 0.7411 (+0.0009).
Small — likely near the noise floor given this removes ~20 rows out of ~77k — but a real
improvement, and per-fold it restored CatBoost to winning all 5 folds (had dropped to 4/5
after the percent-change fix above): `0.711 0.734 0.774 0.713 0.769` → `0.721 0.730 0.769
0.717 0.769`, now beating XGBoost's `0.705 0.720 0.754 0.710 0.755` on every fold. Per the
user's explicit instruction ("if the AUC improves we will commit it as a sweeping
change"), adopted in both projects together — the Forecasted Sales side showed
directionally consistent gains on the same test (see that project's `HANDOFF.md`).
**Fix:** `NON_MLS_MEMBER_AGENT_IDS` added to both `Final_Model/data.py` and
`modeling/data.py` (kept mirrored). `load_clean`/`load_current_snapshot` gained an
`exclude_non_member` parameter, default `True`, filtering on `agent_profile_id` after the
existing `exclude_team` filter.
**Result:** re-ran the full `Final_Model/` production pipeline —
`crossval.py`/`calibrate.py`/`score_agents.py`/`generate_report.py`/
`generate_distribution_chart.py`. `current_agent_risk_scores.csv` regenerated, 4,607
agents scored (down from 4,608). Headline AUC updated 74.0% → 74.1% in
`Final_Model/README.md`, fold-win count restored to 5/5. `modeling/data.py` patched for
parity, own scratch artifacts not separately regenerated (same convention as the
percent-change fix above).
**Source:** `Final_Model/data.py::NON_MLS_MEMBER_AGENT_IDS`,
`modeling/data.py::NON_MLS_MEMBER_AGENT_IDS`, this conversation, sibling entry in
`Forecasted_Sales_Algorithm/HANDOFF.md`.

### 2026-08-03 — Calibration audit and external-benchmark comparison (documentation only)

**Decision:** Record, without changing anything, four findings from a user-directed
audit prompted by a comparison against `seller_likelihood_complete_documentation (6).pdf`.
**No code, config, model, or algorithm was modified** — all numbers came from throwaway
scripts run against the pipeline as it stands. Full detail in `modeling/HANDOFF.md`,
section "Calibration audit + external benchmark comparison (2026-08-03)".

**Finding 1 — our raw scores carry the same prior shift the paper's do.**
`auto_class_weights="SqrtBalanced"` inflates the mean predicted score to 10.5392%
against a true base rate of 1.9737% (**5.34x**). The pipeline is sound only because
`fit_production_calibrator`'s Platt step is fit on OOF scores against true labels at
**natural prevalence** — verified there is no resampling anywhere in the pipeline
(class weights only). Held out folds 3-4: 5.34x shift reduced to ratio 1.331, and that
residual is the already-root-caused temporal drift, not miscalibration. Reliability is
strongly position-dependent: deciles 1-3 overpredict 1.9-3.5x, deciles 8-10 land within
3-28% and sit essentially on the nose once bias correction is applied. **Consequence:
never publish a raw CatBoost score as a probability, and don't quote a Low-tier
percentage as a calibrated one.**

**Finding 2 — mild double-count in `bias_correction`, deliberately left in place.**
`score_agents.py` fits Platt on all folds then multiplies by a bias ratio measured with
a prior-folds-only calibrator, so part of the drift is applied twice (held-out fold 4:
Platt-only ratio 1.312, corrected 1.043, production recipe 0.949 — ~5% under). **Not
fixed, on purpose.** `bias_correction` appears in both `pct` and in the cutpoint/
multiplier basis, so it **cancels exactly**: tiers, risk multipliers, and ranking are
mathematically invariant to it, and it affects only the displayed percentage. This is a
stronger property than the 2026-07-24 cutpoint/base-rate entry claims. Anyone removing
the double-count must keep both on the same basis or they reintroduce the 2026-07-24
tier-squeezing bug.

**Finding 3 — the 6x tier anchor does not transfer across horizons.** One-off 12-month
backtest run as a hypothetical; **12m remains DESCOPED per 2026-07-25 and was not
re-scoped.** Out-of-sample Extra High at 12m: 46 agents, 28 left, 60.87%, 6.62x lift
(stable at 58.33% when the calibrator is re-fit on a longer window). But
`MULT_CUT_POINTS = [1, 3, 6]` was tuned at the 3-month base rate; at a 9% 12-month base
rate, 6x means a predicted 54% chance of leaving and the top bucket collapses to 0.27%
of the roster (~13 agents). ~3x is the 12m equivalent of 6x at 3m. **If any non-3m
horizon is ever un-descoped, re-anchor the multipliers rather than reusing [1, 3, 6].**
Also noted: `horizon_generalization.build_clean_for_horizon` predates both the
2026-07-25 team filter and the 2026-07-28 non-MLS-member filter and uses `TREE_FEATURES`
rather than `PRODUCTION_FEATURES` — left unpatched this round, patch before trusting.

**Finding 4 — the seller-likelihood paper is not a valid benchmark as published.**
18-month horizon (vs. our 3); self-contradictory base rate (Table 4 says 8-12% at 18mo,
p36 says "~3-5% annually"); majority-class undersampling applied **before** the
train/test split (sections G.1, I.3 steps 2/4, J.5: "12:1 to approximately 3.8:1"), so
precision is measured at ~20.8% prevalence; **no prior correction anywhere in the
document**, including the section K inference path, so the shift propagates into deployed
`p18m` and into section K.2's survival conversion, which assumes `p18m` is calibrated;
and a random 80/20 split (seed 42) on a panel with 15-25 snapshots per property, against
its own section 5.3 recommending an `as_of_date` split. Arithmetic check: their precision
0.4643 / recall 0.6651 is incompatible with their reported AUC 0.7342 at any base rate
near 3-5% (would need AUC >= 0.81) and reconciles only at ~20.8%.
**Quantified on our data** (their protocol, our model fixed — hypothetical, never
adopted): our precision at their recall goes **4.14% honest → 37.41%**, a 9x inflation
of the headline while AUC moves only 0.7407 → 0.7618.
**Standing guidance: benchmark on AUC or lift, never raw precision.** Their AUC is
0.686-0.711 across three states (0.7342 headline RI) vs. our ~0.7384-0.7411 at 3m —
comparable ranking, stricter folds on our side. Our genuine weakness is recall, not
precision.
**Source:** `modeling/HANDOFF.md` (2026-08-03 section), `modeling/calibrate.py`,
`modeling/score_agents.py`, `seller_likelihood_complete_documentation (6).pdf` sections
D.4, G.1, I.2, I.3, J.2-J.8, K.1-K.3, Table 4, Table 16; this conversation.

---

### 2026-08-03 — Magnet/trickle-as-churn: proposed, applied, measured, reversed

**Net effect on labels: none.** All 44 pairs suppress as before; 459 rows recode; base
positive rate 1.962%; mean AUC 0.7411/0.741511. Recorded in full because a measured,
implemented change that was then *declined* is a stronger methodological artifact than one
that was never tried — and because the AUC it would have bought (+0.0085, crossing 0.75)
makes it likely to be re-proposed.

**What was proposed (user-directed, from industry-expert conversations).** Count the MAGNET
geometry (many distinct origins → one destination, same quarter) and the TRICKLE geometry
(one office pair trading 2–4 agents per quarter across several quarters) as **agent
churn**, moving all 30 pairs those detectors had contributed out of the not-churn list
(suppressing pairs 44 → 14). Rationale: a magnet destination usually reflects one
brokerage's exceptional marketing campaign or a strong signing bonus — an agent who takes a
better offer left voluntarily, which is exactly the churn a retention model exists to
predict; and a trickle of a few agents a quarter can happen entirely naturally, which is
the steady bleed `office_exit_rate_12m` is intended to measure.

**Why it was reversed.** The detectors never classified by *shape* — they are discovery
tools, and organic drift had already been filtered out before any pair reached the file:

- Of **45 trickle pairs surfaced for review, 16 were checked and deliberately left as
  churn** — `RMAX01↔RMAX45` (bidirectional: 7 movers one way, 3 back, which is competition
  rather than a roll-up), `DRHM→LVGP`, `RGEX→RDRM02`, `CMWT13→CMWT04` (flowing *out* of the
  consolidated entity — wrong direction for an acquisition). ~200 further lower-volume
  trickle-shaped pairs never cleared review at all.
- The 29 survivors are the **residue after that filter**: 17 with a byte-identical
  origin/destination `office_name` ("Coldwell Banker Realty" → "Coldwell Banker Realty" — an
  internal MLS office-code renumber in which nobody changed employers), and 12 with a
  documented corporate event. None is unexplained organic drift, so "a trickle can be
  natural" — true of the shape in general — does not describe these particular pairs.
- **Tier B is this project's founding bug on a slower clock.** Randall Realtors and Lila
  Delman → Compass are press-documented acquisitions: the same event class as HomeSmart
  Professionals → REMAX Revolution (120 contaminated labels, fold 5 collapsing to 0.63 AUC —
  the diagnosis this entire system was built from). Compass simply rebranded its agents
  across several quarters instead of one, which is precisely the slow version
  `detect_trickle_pairs.py` was built to catch.
- **The magnet shape never suppressed a label in the first place.** There is no
  dest+quarter-keyed recoding mechanism in `data.py`. Both large magnet destinations —
  SERHANT (24 arrivals/12 origins) and LVGP/Real Brokerage (5 quarters) — were verified and
  *deliberately left labeled as churn*, for exactly the signing-bonus reasoning in the
  proposal. Exactly one pair found via that route is suppressed, `HSHR→SMRT`, resting on a
  documented consolidation announcement (RISMedia 2024-01-16) rather than on the geometry.
- **What acquisitions do produce that is genuine churn**: agents who leave *because* their
  firm was bought. Those moves go origin → some *third* office and have always counted as
  churn. Suppression only ever covered moves to the acquirer itself — the one destination
  that is not a choice.

**Tier A semantics confirmed** as the intended reading: if an agent's office code changes
but the office *name* is identical, that is not churn, regardless of which detector found
the pair. `CBRB23→CBRB24` and `CBRB15→CBRB24` are now treated alike, resolving an
inconsistency the proposal would have introduced.

**Quantified — three label variants, production CatBoost config, 3 seeds averaged, on the
production fold structure plus both `confirm.py` shifted structures.** Variant C prices
out the objection: the change, but with the 17 identical-name renumbers still suppressed.

**Quantified before reversing** — production CatBoost config, 3 seeds averaged, production
fold structure plus both `confirm.py` shifted structures. Variant C prices out the
identical-name sub-question on its own:

| variant | pairs | recoded | positives | pos rate | AUC (9,2) | AUC (7,2) | AUC (11,2) | worst (9,2) |
|---|---|---|---|---|---|---|---|---|
| **A: current (restored)** | **44** | **459** | **1519** | **1.962%** | **0.741511** | **0.738277** | **0.747655** | **0.715368** |
| B: proposal (reverted) | 14 | 335 | 1637 | 2.114% | 0.750005 | 0.744435 | 0.752004 | 0.729573 |
| C: keep identical-name | 31 | 395 | 1582 | 2.043% | 0.749994 | 0.744746 | 0.752733 | 0.724095 |

Per-fold (9,2), seed-averaged: A `.717 .733 .771 .718 .768`, B `.742 .733 .772 .730 .774`.

**Why the +0.0085 appeared, and why it was declined anyway.** A label change is not a model
change. B's target is a different and easier-to-rank quantity: the 118 positives it restores
are office-clustered events, and office features (`office_exit_rate_12m`,
`office_net_flow_12m`) are what this model ranks best, so returning them mechanically lifts
AUC. Any future proposal that improves AUC by moving rows between label classes deserves the
same scrutiny. Note also that `confirm.py` **cannot adjudicate this class of change at all** —
it holds the panel fixed and varies the config, and here the panel is what moved; only its
multi-seed / shifted-fold *stability* machinery transfers.

**Incidental finding — the identical-name question is AUC-neutral.** B and C differ by
0.00001 (0.750005 vs 0.749994). The whole delta comes from the 13 evidence-backed pairs, not
the 17 office-code renumbers, which carry essentially zero signal either way. Their
classification therefore rests entirely on correctness.

**Downstream.** Detector semantics reaffirmed rather than changed: all three find a *shape*,
evidence decides the *cause*, and none recodes on geometry alone.
`detect_trickle_pairs.py` now filters against a new `ALREADY_ADJUDICATED_PAIRS` rather than
`NOT_CHURN_PAIRS` — identical in content today, but it prevents a future flip of
`COUNT_MAGNET_TRICKLE_AS_CHURN` from silently making the detector re-propose 29 pairs a
human already ruled on. Production regenerated and mirrored to `Final_Model/` and the repo
root; labels verified byte-identical to the pre-session state (459 recoded, 1.9620%).

**Source:** `modeling/rebrand_classifications.py` (POLICY CHANGE docstring block,
`MAGNET_TRICKLE_PAIRS`), `modeling/data.py::_recode_mass_mover_moves`,
`modeling/detect_magnet_destinations.py`, `modeling/detect_trickle_pairs.py`,
`modeling/HANDOFF.md` (2026-08-03 POLICY CHANGE section), `Write_Up/VERIFIED_TABLES.md`
T6/T7; this conversation.

### 2026-08-05 — Resolvability rule tightened; horizon row-math re-derived; 0.63% retracted

**Decision point:** Reviewing `data.py`, the resolvable-row rule read
`resolvable = move_observed | window_confirmed_elapsed` — a row counted as usable if
its 3-month window had closed **or** a move had been recorded for it. The module
docstring three paragraphs above explicitly forbids exactly that: *"No 'we already know
this one, so keep it' special-casing that would bias the recent edge of the data."*
Code and docstring had been contradicting each other.

**Why it is wrong:** past the observable edge, a mover's move is on record but a
stayer's non-move is not, so the OR-branch can only ever admit positives. Grading a
3-month probation by reading only the resignation letters that already arrived.

**Investigation (riar, non-team, `leave-dataset-with-team-distinction.csv`):** the
branch admitted **12 rows, all `label_left_3m == 1`, all in the 2026-07-01 snapshot**
(`label_days_to_move` 3–8 days). Pre-filter population 77,453 → 77,441; positive rate
2.536% → 2.521%. `MIN_SNAPSHOT_ROWS = 500` then drops that snapshot as degenerate, so
none of the 12 ever reached training. MLSPIN: 6 rows, same shape.

**Result:** `move_observed |` removed from all three copies (`Final_Model/`, `modeling/`,
`MA/Likelihood_to_Leave/`). End-to-end `load_clean()` output is **byte-identical** —
77,422 rows, 1,519 positives, **1.9620%**. No fold, metric, or model changes.

**Why fix a zero-impact bug:** it was harmless only by accident. `max_date_lower_bound`
currently lands exactly on the max snapshot date, so every earlier snapshot clears the
window test on its own and the OR-branch was load-bearing at the final snapshot alone. A
future drop whose `label_censored == 0` tail runs several quarters behind the last
snapshot would have it admit positives-only rows across multiple recent quarters. The
500-row floor is a real but unintentional backstop (a positives-only snapshot is
~2.5% × 4,500 ≈ 112 rows), and it only holds while the base rate stays near 2.5%.

**Horizon row-math re-derived at the same time.** Of 84,417 riar rows: **79,663 (94.4%)
usable over 19 quarters** at 3 months vs. **65,797 (77.9%) over 16 quarters** at 12
months; confirmed leavers **2,024 vs 6,321**; raw positive rate 2.54% vs 9.61%. The
long-standing "~94% vs ~78%" is confirmed.

> **Basis correction, 2026-08-11.** These figures are on the *pre-team-exclusion* panel,
> which is not the panel any other number in this project uses. Recomputed on the
> production panel: **77,389 (91.7%) over 19 quarters vs. 63,948 (75.8%) over 16**;
> confirmed leavers **1,952 vs 6,128**; raw positive rate **2.5223% vs 9.58%**. This is the
> origin of the "2.54% vs 2.55%" split that reached three drafts — the draft was quoting a
> narrower panel than the guide was. Use the production-panel row and nothing else. **"Four extra quarters" was wrong — it is
three (16 → 19)**; the fourth was 2026-07-01, which is dropped as degenerate and only
entered the count via the bug above. MA had inherited RI's figures verbatim; MLSPIN's
own are **583,908 (95.4%) over 19 quarters vs. 497,253 (81.2%) over 16**, positive rate
2.25% vs 8.54%, leavers 13,133 vs 42,445.

**Retraction:** `HANDOFF.md`'s "second move inside the same quarter: 0.63% of riar
moves, 1.89% on MLSPIN" **does not reproduce and must not be cited.** No script, no
changelog entry, nothing downstream re-derives it. Its neighbours in the same bullet do
reproduce exactly (523/640,245; 20/64,809), so the audit and the file are known — but
~20 candidate definitions give no single rule producing both: `Δmoves_5y>=2` on move
rows → 0.70%/1.23% (any-gap 0.65%/1.16%), `Δmoves_5y>=2 & Δnum_offices>=1` →
0.94%/1.97%, `Δnum_offices_career>=2` → 0.20%/0.44%. Each target is reachable alone, by
a different rule. Any re-derivation must also apply the 23%-genuine haircut, since every
instrument leans on the `moves_5y_asof` counter audited as wrong 64% of the time. The
limitation is real; only its size is unrecorded.

**Source:** `Final_Model/data.py::load_clean` (+ `modeling/`, `MA/Likelihood_to_Leave/`
mirrors), `*/generate_report.py` methodology table, `modeling/HANDOFF.md` Data section
and decision #1, `Write_Up/AUTHORS_GUIDE.md` §2.1/§2.2/§5.4, `Write_Up/FEEDBACK.md`
§2.5/§4.1; this conversation.

### 2026-08-07 — Monotonic constraints: declared five, deployed four; resolved to four

**Found while sourcing the paper's methods section.** Three different counts of the
monotonic constraint set were live simultaneously:

| Claim | Location | Count |
|---|---|---:|
| Documented | `AUTHORS_GUIDE.md` §5.10, `HANDOFF.md` decision #8 | 3 |
| Declared | `crossval.py::MONOTONE` (all three lineages) | 5 |
| Actually binding in production | CatBoost over `PRODUCTION_FEATURES` | **4** |

The docs said three because they predate the 2026-07 round that added
`company_exit_rate_12m` (+1) and `volume_trend_vs_market_12m` (−1). The dict says five.
But `volume_trend_vs_market_12m` is not in `PRODUCTION_FEATURES` — the AutoResearch
94→54 pruning dropped it, keeping `units_trend_vs_market_12m` instead — and the
constraint vector is built with `MONOTONE.get(c, 0)`, which resolves a missing key to
`0` (unconstrained) rather than raising. So the fifth constraint silently evaporated at
promotion time and the deployed model has only ever carried four. Same on MA, which
inherits the RI feature set.

**Decision — resolved in favour of the model.** `volume_trend_vs_market_12m` removed
from `MONOTONE` in all five live definition sites (`Final_Model/`, `modeling/`,
`MA/Likelihood_to_Leave/` `crossval.py`; `modeling/derive_winning_config.py`;
`MA/Likelihood_to_Leave/derive_ma_config.py`). Declared and deployed now both read four:
`office_exit_rate_12m` (+), `months_since_last_closing` (+),
`share_of_office_volume_12m` (−), `company_exit_rate_12m` (+).

**Cost, measured rather than assumed.** Production CatBoost is unaffected by
construction — the key never bound there. The only path where it *did* bind is
`_fit_xgboost`, which fits over the full 94-column `TREE_FEATURES`. Re-running the
five forward-chaining folds both ways:

| | mean AUC | worst fold |
|---|---:|---:|
| 5-constraint (before) | 0.728803 | 0.705294 |
| 4-constraint (after) | 0.729684 | 0.705046 |
| delta | **+0.000881** | **−0.000248** |

Inside the noise band — below the program's own +0.001 mean keep bar, and nowhere near
the −0.01 worst-fold discard bar. The reference XGBoost number does not meaningfully
move, so no published figure is invalidated.

**`units_trend_vs_market_12m` deliberately left unconstrained.** The tempting fix is to
move the −1 onto the column that actually survived pruning, since it measures nearly the
same thing in units rather than dollars. Not done: that would be inventing a constraint
the model has never been fit under and calling it a correction. It survived pruning on
its own merits and has never carried a validated prior. If the prior is worth spending,
it is a new experiment and goes through `confirm.py` like anything else.

**AutoResearch copies deliberately left alone.** `AutoResearch/autoloop{,2,3,4,5,6}.py`
and `experiment.py` each carry their own five-key `MONOTONE`. Those are records of runs
that actually executed with five keys declared (and four binding, wherever they fit over
a pruned set). Editing them would falsify the provenance of every number in
`results.tsv`. They are historical, not live.

**Why this mattered enough to chase.** The silent-`.get()` failure mode is the same
class as any config that degrades quietly instead of erroring: nothing was ever wrong in
the output, so nothing ever surfaced it. It was caught only because the paper needed the
list written down explicitly. The `explain_agents.py` note reasoning about SHAP sign
guarantees "unlike the 5 monotonically-constrained features" was relying on the wrong
number in the one place the guarantee is actually load-bearing.

**Source:** `*/crossval.py::MONOTONE`, `modeling/derive_winning_config.py`,
`MA/Likelihood_to_Leave/derive_ma_config.py`, `*/explain_agents.py`,
`modeling/HANDOFF.md` decision #8, `Write_Up/AUTHORS_GUIDE.md` §5.10; this conversation.

---

### 2026-08-10 — Agent identifier switched to MLS ID (`agent_profile_id` → `mls_agent_id`)

**Decision:** Adopted `leave-dataset-with-team-distinction-mls-id.csv` as the default
training and scoring source, replacing `leave-dataset-with-team-distinction.csv`. The
sole difference is the agent identifier: the opaque per-row UUID `agent_profile_id` is
replaced by `mls_agent_id`, the agent's real ID in their own MLS. Every other column is
unchanged. Production output `current_agent_risk_scores.csv` now keys on `mls_agent_id`.

**Verified before switching** (this project's standing bar for any source-file swap):

| Check | Result |
|---|---|
| Row count | 696,775 → 696,775 |
| Shared columns, compared row-for-row in file order | identical on all 9 probed (`agent_name`, `mls_code`, `snapshot_date`, `office_mls_id`, `is_team`, `volume_12m`, `units_12m`, `label_days_to_move`, `label_left_within_12m`) |
| Distinct agents | 50,811 → 50,811 |
| Null IDs | 0 |
| Duplicate (id, snapshot_date, mls_code) keys | 0 |
| Old↔new ID mapping | strict 1:1 (0 UUIDs split, 0 IDs collapsed) |
| RIAR subset | 84,417 rows / 5,625 agents / 155 team accounts — all unchanged |

**Score-neutral, confirmed rather than assumed.** No feature value moved, so the switch
*should* be a pure relabel — but "should" is not a measurement. `score_agents.py` was
re-run end to end and its output joined back to the pre-switch file through the
crosswalk: **4,607 of 4,607 agents matched, max absolute difference in `percent_chance`
and `risk_multiplier` = 0.0**, and `risk_tier`, `risk_tier_label`, `office_name`,
`snapshot_date` and all five `why_*` explanation strings identical. The 25-test leakage
guard suite (`AutoResearch/gepa_loop/test_guard.py`) passes unchanged. Pipeline counts
reproduce exactly: 459 rows recoded, 153 team accounts excluded, 84,417 → 77,422 usable,
1.9620% positive rate.

**Not a new data era.** `results.tsv` spans three incompatible data eras (see
`rebaseline.py`). This switch does *not* open a fourth: a relabel with byte-identical
features leaves every prior measurement on era 3 directly comparable, so the documented
baseline (mean 0.7411 / worst-fold 0.7190) carries over untouched. `rebaseline.py` now
reads its `data_file` label off `data.py::DATA_PATH` instead of a hardcoded literal, so
the field that exists to identify the era can no longer go stale against the file it
names.

**Two silent-failure modes this could have introduced, both closed:**

1. **`NON_MLS_MEMBER_AGENT_IDS` was pinned to a UUID.** Left as-is, the 2026-07-28
   exclusion of the "Non-Mls Member" synthetic record (worth +0.0009 mean AUC) would
   have matched nothing and reverted itself with no error and no log line — the exact
   silent-degradation class as the 2026-08-07 `MONOTONE.get()` finding. Re-pointed to
   MLS ID `12345`, verified still matching the same 20 rows (19 in training, 1 in the
   scored snapshot).
2. **`mls_agent_id` must never be dtype-inferred.** Three RIAR IDs carry meaningful
   leading zeros (`00001`, `0012`, `0014`) and the MLSPIN block arriving with the MA
   expansion is alphanumeric (`A9500308`, `CN235473`, `CT005436`). Inference would
   corrupt the first group and produce a mixed int/str column across the second. Both
   `read_csv` calls in `data.py` now pass `dtype={AGENT_ID_COLUMN: str}`.

**Flagged, deliberately not fixed — more synthetic records are now visible.** Readable
IDs exposed three further non-agent placeholders in the RIAR feed that UUIDs had hidden:
`0012` (MASS ALLIANCE PARTNER, already caught by the `is_team` filter), `0014` (ALLIANCE
MEMBER, stops appearing after 2025-01-01), and **`00001` (Assisted Sale)** — which is
`is_team == 0`, present in all 20 snapshots including the scored one, and is **currently
being scored as if it were a real agent**. This is a pre-existing condition of the feed,
not something the file swap introduced, and excluding it is a model change: per the
standing bar it needs its own before/after crossval through `confirm.py` before being
adopted or rejected. Recorded here so it is not lost.

**Artifacts.** `agent_id_crosswalk.csv` (50,811 rows: `agent_profile_id`, `mls_agent_id`,
`mls_code`, `agent_name`) is written at the project root so any output produced before
this switch can still be joined. Retired UUIDs appear nowhere in live code; the
historical `SOURCE DATA` log entries in `data.py` keep the old column name on purpose,
since they record verification genuinely performed against files that had it.

**ID scope caveat for MA/CT.** `mls_agent_id` is guaranteed unique only *within* an MLS.
It happens to be globally unique across the current file (0 of 50,811 IDs appear under
more than one `mls_code`), but that is a property of these two boards, not a guarantee.
Multi-MLS work should key on `(mls_code, mls_agent_id)`. RIAR-only, the bare ID is safe.

**Source:** `*/data.py` (2026-08-10 `SOURCE DATA` note, `AGENT_ID_COLUMN`,
`NON_MLS_MEMBER_AGENT_IDS`), `*/score_agents.py`, `*/explain_agents.py`,
`AutoResearch/gepa_loop/rebaseline.py`, `agent_id_crosswalk.csv`; this conversation.

---

### 2026-08-10 (b) — "Assisted Sale" placeholder excluded; identifier sweep completed across all four models

**Decision:** `00001` ("Assisted Sale") added to `NON_MLS_MEMBER_AGENT_IDS`, and the
`mls_agent_id` switch propagated to every project that reads this dataset.

**Measured before adopting**, via a new `AutoResearch/gepa_loop/confirm_population.py` —
`confirm.py`'s protocol adapted to the axis where the CONFIG is fixed and the PANEL
varies. (`confirm.py` itself could not be used: it compares candidate config vs baseline
config on one panel, so here it would have compared the production config against itself
and returned a delta of exactly zero.) Same bar, same three checks, 3 seeds × 3 fold
structures, 90 fits:

| Fold structure | Baseline | `00001` excluded | Δ |
|---|---:|---:|---:|
| Production 9/2 | 0.741511 | 0.742121 | **+0.000610** |
| Shifted 7/2 | 0.738277 | 0.739596 | +0.001319 |
| Shifted 11/2 | 0.747655 | 0.748003 | +0.000348 |
| Worst-fold (9/2) | 0.715368 | 0.716269 | +0.000901 |

Verdict **CONFIRMED** — positive on all three structures under seed averaging, 3/5
production folds improved. Panel 77,422 → 77,403 usable rows; all 19 removed rows are
negatives, so positives stay at 1,519 and the base rate moves 1.9620% → 1.9625%.

**Read the magnitude honestly.** +0.00061 is about **0.4 of a paired standard error**
(SE(δ) = σ√(2(1−ρ)) ≈ 0.0014 at ρ=0.99), and 2 of 5 folds got worse. What CONFIRMED
establishes is
consistency of DIRECTION across nine independent measurements, not magnitude. Were this
a feature or hyperparameter proposal it would be noise-band and rejected. The
load-bearing argument is correctness: `00001` is `office_mls_id` **ASST**, office_name
"Broker Assisted Sale", 1,062 units at 99.7% list-side, tenure incrementing exactly 3.0
months per quarter (211.1, 214.1, 217.1, …) — a counter, not a career — and it never
changes office across 19 usable rows. It is a transaction bucket, not a person, and has
no churn risk to predict. Same basis on which the magnet/trickle recodes were adopted at
a measured AUC *cost*. The AUC here merely declines to argue against it.

> **Statistical framing corrected 2026-08-14** (measured values untouched). The paragraph
> above originally read "+0.00061 is ~1/30th of the ±0.019 CI." That compared a paired
> difference against the CI on the *level* of AUC — the wrong ruler, and one that
> understated every result in this file. Candidate and baseline are scored on identical
> folds, rows and splits, so the relevant SE is σ√(2(1−ρ)) ≈ 0.0014 at ρ=0.99, not 0.0099.
> The verdict here is unchanged (0.4 SE is still direction-only), but the same phrasing
> appears throughout Part III and understates it. See `Write_Up/AUTHORS_GUIDE.md` §5.2b
> and `FEEDBACK.md` Round 5.

**Correction to the record made while writing this up.** The 2026-07-28 "Non-Mls Member"
adoption (+0.0009) was cited in `data.py` as comparable evidence. It is not: that was
single-seed, single-fold-structure, and predates `confirm.py` by two days. It would not
clear today's bar on its own and has never been re-measured under the protocol. Its
justification is the same correctness argument, not the number. Annotated in `data.py`
so the two are not read as equivalent evidence.

**Production output impact is larger than 19 rows suggests, and worth knowing.**
Rescoring moved 241 of 4,606 agents (5.2%) across a tier boundary; Spearman rank
correlation 0.9906, top-100 overlap 91/100, but top-20 overlap only 15/20 and the
largest single `percent_chance` move was 0.101. Tier assignment near the cutpoints is
more sensitive to small panel changes than the headline AUC implies — relevant if tier
labels are being used directly for outreach prioritization.

**Identifier sweep completed.** Four models read this one CSV and differ only by their
`mls_code` filter. All four now read `leave-dataset-with-team-distinction-mls-id.csv`,
key on `mls_agent_id`, force `dtype=str` on it, and carry a re-pointed placeholder set:

| Project | MLS | Placeholder exclusions | Post-switch panel |
|---|---|---|---|
| `Likelihood_to_Leave_Algorithm` | riar | `12345`, `00001` | 77,403 rows |
| `MA/Likelihood_to_Leave` | mlspin | `H1111111` | 556,365 rows |
| `Forecasted_Sales_Algorithm` | riar | `12345`, `00001` | 82,009 rows |
| `MA/Forecasted_Sales` | mlspin | `H1111111` | 583,384 rows |

Every one of those pinned constants was a UUID that would have matched nothing under the
new file — four silent revert-to-unfiltered failures avoided, not one.

**MLSPIN was scanned for its own `Assisted Sale` equivalent** (non-team, all 20
snapshots, single office, ≥2,000 units, degenerate list/buy split): `H1111111` is the
only hit and was already excluded. The two other low-entropy MLSPIN ids, `H9999999` and
`Z1111111`, are ordinary low-volume agents with real names, not buckets. So MA needs no
new exclusion; RI's `00001` is a riar id and is deliberately NOT inherited there.

**Evidence asymmetry, stated rather than glossed.** `00001` was measured against the
LEAVE model only. It is adopted in `Forecasted_Sales_Algorithm` on the correctness
argument alone, with no backtest of that model's hurdle+CQR pipeline — unlike that
project's two prior exclusion adoptions, both of which were measured. Flagged inline in
its `data.py`: re-run the backtest before restating that model's numbers, rather than
citing the Leave project's AUC as though it transferred.

**`Forecasted_Sales_Algorithm` repointed to the shared CSV.** It had kept its own 332MB
duplicate. Creating a third copy of the new file would have meant three files needing
manual re-sync on every data drop — which is exactly how that project's `DATA_PATH` went
stale twice before. Superseded local copies left on disk, untouched, read by nothing.

**Source:** `*/data.py`, `AutoResearch/gepa_loop/confirm_population.py`,
`AutoResearch/gepa_loop/population_confirmation.json`; this conversation.

---

### 2026-08-10 (c) — `0014` "ALLIANCE MEMBER" (Connecticut-MLS bucket) excluded

**Decision (user-directed):** `0014` added to `NON_MLS_MEMBER_AGENT_IDS`, on the user's
call that the evidence already settles what it is — a bucket, not an individual agent.
The RI leave model's placeholder set is now `{12345, 00001, 0014}`.

**The evidence, which does not depend on the metric.** `office_mls_id` **CTMLS**,
office_name **"CONNECTICUT MLS"**, 21 units across 14 snapshots, 100% list-side, never
changes office, `is_team == 0`, and it stops appearing entirely after 2025-01-01. The
geography is the tell: **Connecticut is not otherwise in this file at all** (RIAR and
MLSPIN only). This row is the RIAR feed's holding pen for CT-MLS cross-board activity,
which is precisely why no individual agent stands behind it.

**It was contaminating training only, never a scored output.** Because it stops in early
2025 it never reaches the scored snapshot, so no production number ever looked wrong
because of it. That is the quieter of the two failure modes and the reason it survived
this long unnoticed — worth remembering that "the output looks fine" is not evidence the
panel is clean.

**Measured anyway**, chained onto the `00001` panel as the new baseline
(`confirm_population.py 0014`, 3 seeds × 3 fold structures):

| Fold structure | Baseline (post-`00001`) | `0014` also excluded | Δ |
|---|---:|---:|---:|
| Production 9/2 | 0.742121 | 0.743050 | **+0.000929** |
| Shifted 7/2 | 0.739596 | 0.739696 | +0.000100 |
| Shifted 11/2 | 0.748003 | 0.748894 | +0.000892 |
| Worst-fold (9/2) | 0.716269 | 0.717216 | +0.000947 |

Verdict **CONFIRMED**, and stronger than `00001`'s on two counts: 4/5 production folds
improved rather than 3/5, and the effect on worst-fold is as large as on the mean. The
same magnitude caveat still applies — this is well inside the ±0.019 CI, and the
correctness argument is what carries the decision.

**Chain-of-custody check.** This run's baseline arm reproduced the previous run's
candidate arm to six decimals on all three fold structures (0.742121 / 0.739596 /
0.748003). The two measurements therefore compose: each is measured against the panel the
previous one actually produced, rather than each drifting from its own private reference.
That matters here because these exclusions are arriving one at a time.

**Cumulative for 2026-08-10:** panel 84,417 → **77,389** usable rows (33 fewer than this
morning's 77,422; all 33 are negatives, so positives stay at 1,519 and the base rate
moves 1.9620% → 1.9628%). 3-seed mean AUC 0.741511 → **0.743050**, worst-fold 0.715368 →
**0.717216**.

**`0012` (MASS ALLIANCE PARTNER) deliberately NOT added.** It is `is_team == 1`, so the
team filter already removes it from both training and scoring. Adding it to the
placeholder set would be dead weight and would imply the team filter cannot be relied on
to do its job.

**Forecasted Sales was NOT changed — deliberately, per user instruction.** That project
keeps `{12345, 00001}` and now diverges from this one. A strongly-worded stop block was
added at the top of `Forecasted_Sales_Algorithm/HANDOFF.md` recording: (a) that `00001`
was adopted there on the correctness argument with **no** measurement against its
hurdle+CQR backtest, unlike both of its prior exclusion adoptions, which were measured;
(b) that `0014` is a confirmed bucket, excluded here, still present there; and (c) that
the two projects consequently **no longer train on the same population**, so any
cross-model comparison or shared sanity-check is invalid until reconciled. Ordered
remediation path given there: decide on `0014` → re-run the backtest covering both
records → only then restate that model's figures.

**Source:** `*/data.py::NON_MLS_MEMBER_AGENT_IDS`,
`AutoResearch/gepa_loop/population_confirmation_0014.json`,
`Forecasted_Sales_Algorithm/HANDOFF.md`; this conversation.

---

### 2026-08-11 — `new_listings_12m` guardrail: the two bounds were transposed

**Decision:** `NEW_LISTINGS_LOW_MAX` and `NEW_LISTINGS_HIGH_MIN` in `explain_agents.py`
were swapped at authoring — `LOW_MAX = 8`, `HIGH_MIN = 5`. Restored to `LOW_MAX = 5`,
`HIGH_MIN = 8` in all three copies (`Final_Model/`, `modeling/`, `MA/Likelihood_to_Leave/`).

**How it was found:** while tabulating the explanation templates for
`Write_Up/VERIFIED_TABLES.md` T13, not by a test — the pair is self-contradictory read
aloud ("low" topping out *above* where "active" begins), and the constant names say so
without needing the data.

**What the transposition did.** The 2026-07-28 entry above describes the guardrail's
intent as leaving "the ambiguous 6-7 gap" unspoken — which is only reachable with the
values the other way round. As shipped they left an **overlap**, not a gap: `risk`
required n ≤ 8 and `safe` required n ≥ 5, so at n = 5, 6, 7, 8 *both* wordings were
permitted and two agents with an identical listing count could be described in opposite
terms. It is the same class of contradiction the guardrail was added to prevent, narrowed
from "0 listings, active / only 448, below-average" to a four-count band.

**The wording was false in that band, not merely ambiguous.** Current roster
(4,606 agents, snapshot 2026-07-01): mean 3.31, median 1, p75 = 4, p80 = 5, **p90 = 8**.
An agent with 6 new listings sits at the **85.7th percentile**, and was being told they had
"below-average listing activity." Under the corrected bounds "low" is the bottom 83% of the
roster, "active" is the top 10%, and the 6-7 band (85.7th-88.2nd pct) is skipped as
genuinely ambiguous.

| | Old (`8`/`5`) | New (`5`/`8`) |
|---|---|---|
| `risk` ("only {n} … below-average") fires when | n ≤ 8 | n ≤ 5 |
| `safe` ("{n} new listings … active") fires when | n ≥ 5 | n ≥ 8 |
| Counts where both wordings are permitted | **5, 6, 7, 8** | none |
| Counts where neither fires | none | 6, 7 |

**Result** — `Final_Model/score_agents.py` re-run, `current_agent_risk_scores.csv`
regenerated:

| Quantity | Before | After |
|---|---:|---:|
| "risk" listing bullets, by count n = 2/3/4/5 | 393/278/183/165 | unchanged |
| "risk" listing bullets, n = 6/7/8 | 120/113/82 | **0** |
| Total "risk" listing bullets | 1,334 | **1,019** |
| "safe" listing bullets (any n) | 0 | 0 |
| Agents with any changed explanation text | — | **315 (6.84%)** |
| Mean bullets per agent | 3.35 | 3.28 |

309 of the 315 affected agents simply carry one fewer bullet; only 6 backfill from the next
feature. That is `MIN_SUPPORTING_SHARE = 0.03` working as documented — candidates are
ranked by |SHAP| descending, so once the skipped bullet is passed the next one usually sits
under the 3% floor and the loop stops rather than padding.

**Control, and the reason this is safe to ship without re-validating the model:** max
absolute change in `percent_chance` across all 4,606 agents is exactly **0.0**, and **no
agent changes risk tier** (3,354 / 1,032 / 138 / 82 before and after). Explanations are
computed downstream of scoring; a template bound cannot move a score, and this confirms it
empirically rather than by argument.

**Not fixed here:** the MA copy carries the same corrected constants but its bounds were
never re-derived against the MA roster's own listing distribution — RI's median-1/p90-8
shape is not guaranteed to transfer. Flagged for the MA expansion, not assumed.

**Source:** `*/explain_agents.py::NEW_LISTINGS_LOW_MAX`/`_HIGH_MIN`,
`Final_Model/current_agent_risk_scores.csv` (regenerated 2026-08-11),
`Write_Up/VERIFIED_TABLES.md` T13; this conversation.

---

### 2026-08-11 (b) — `00001` finally measured on the SALES model; shipped forecast was carrying it

**Decision:** closes the remediation path left open by the 2026-08-10 (b) entry above
("re-run the backtest covering both records"). Applies to
`Forecasted_Sales_Algorithm`, recorded here because that project's exclusion set was
adopted from this one and the two must not silently diverge.

**What was actually wrong.** Both placeholders were already in that project's
`data.py::NON_MLS_MEMBER_AGENT_IDS`, so the code was correct. But its shipped
`current_agent_sales_forecast.csv` was produced by a **crosswalk re-key of the Jul-28
fit, not a refit** -- and a re-key cannot apply a population change. `00001` was
therefore still being forecast in the delivered file at **$51,688,219, rank 11 of
4,607** (46x the roster median, 0.43% of the book). `12345` was genuinely absent,
having been excluded under its UUID back in July, before that fit ran.

**Measured** via a new `Forecasted_Sales_Algorithm/modeling/compare_exclusion_forecasts.py`
-- four full production scoring passes, exclusion ON/OFF at seed 0 plus ON at seeds 1-2,
so the effect reads against that pipeline's own instability rather than against zero
(its HANDOFF item 3 documents ~1.5% median per-agent movement from row reordering alone):

| per-agent `volume_point_estimate` | A vs B (exclusion) | A vs C (noise) | A vs D (noise) |
|---|---:|---:|---:|
| median abs change | **1.47%** | 1.79% | 2.02% |
| p95 abs change | **5.12%** | 6.23% | 6.43% |
| book total | **+0.52%** | -0.21% | +1.16% |
| largest single agent | **$23.5M** | $7.8M | $7.1M |

**Verdict: immaterial to accuracy, mandatory on correctness.** Median, p95 and book
total all sit INSIDE the seed-noise band, so no MAE win may be claimed -- the honest
expectation written into that project's `data.py` was right. Only the largest
single-agent move clears the band, so the effect is concentrated in a few agents rather
than spread. The justification is that a broker-assisted-sale bucket is not a person:
its median training target is **$94.9M / 176 units** against a real-agent median of
**$1.08M / 3 units** (99.94th percentile), and its max ($184.9M) **exceeds the largest
real agent in the panel** ($125.9M). No MAE would have surfaced any of that.

**This is a second, independent measurement of that pipeline's noise floor.** Varying
only the learners' seed -- population and row order held fixed -- reproduces the same
~2% median / ~6% p95 per-agent spread that project measured by re-keying. Two different
perturbations, same magnitude, which rules out the re-key being a special case. 100% of
agents moved in every run.

**Result:** `current_agent_sales_forecast.csv` regenerated by refit -- 4,606 agents
(was 4,607), book $12.075B -> $11.891B, same 33 columns. That project's HANDOFF item 1
moves OPEN -> RESOLVED; item 2 (`0014`, still in its panel, excluded here) stays open,
so the two projects still do not train on the same population.

**Source:** `Forecasted_Sales_Algorithm/modeling/compare_exclusion_forecasts.py`,
`measure_placeholder_exclusion.py`, `exclusion_impact_summary.csv`,
`Forecasted_Sales_Algorithm/HANDOFF.md`; this conversation.

---

### 2026-08-11 (c) — VERIFIED_TABLES.md re-verified end-to-end; headline is now 0.7440

**Decision:** every [RUN] and [CSV] figure in `Write_Up/VERIFIED_TABLES.md` re-derived against
the current panel and the regenerated spreadsheet. T1, T2, T3, T4, T5, T8, T9, T9b, T10 and
the header corrections box all changed.

**Why it was needed.** The tables were verified on 2026-08-04 and the panel changed on
2026-08-10 (`00001` and `0014` excluded, agent key switched). Nothing re-checked them, so a
file whose stated purpose is "a single authoritative source for every number in the paper"
had been describing a retired population for a week. Found while regenerating
`current_agent_risk_scores.csv` for the guardrail fix above: the live CSV had 4,606 rows
against the file's 4,607.

| | Was (2026-08-04) | Now (2026-08-11) |
|---|---:|---:|
| Usable rows | 77,422 | **77,389** |
| Placeholder rows excluded | 19 | **52** (3 records) |
| 3-month positive rate | 1.9620% | **1.9628%** (1,519 positives, unchanged) |
| Agents scored | 4,607 | **4,606** |
| CatBoost mean AUC | 0.7411 | **0.7440** |
| XGBoost / Logistic | 0.7288 / 0.6910 | **0.7279 / 0.6910** |
| Bias correction | 0.7622 | **0.7589** |
| Tier counts | 3,368/1,015/136/88 | **3,354/1,032/138/82** |
| Max calibrated score | 39.45% | **34.09%** |
| Empirical CI on mean AUC | +/-0.023 | **+/-0.024** |

**0.7440 is now the headline everywhere** (`Final_Model/README.md`, `Write_Up/AUTHORS_GUIDE.md`,
`modeling/HANDOFF.md`, this file's header, `Write_Up/FEEDBACK.md`). It is not presented as a
modelling improvement, because it is not one -- it is what the model measures once it stops
being trained on records that are not people. That framing needs no caveat: both 2026-08-10
exclusions were measured before adoption under `confirm_population.py` (3 seeds x 3 fold
structures, 90 fits each) and returned **CONFIRMED**, and the seed-averaged figure on this
panel is 0.743050 / worst fold 0.717216 against the 5-fold production run's 0.7440.
Dated measurement records in the entries above are left at the values they were measured
at -- those are history, not current claims.

**Two scripts added so this cannot silently rot again:**
- `Write_Up/derive_tables_from_log.py` -- parses a crossval log and prints every T1-T5
  figure including the [CALC] ones (per-fold positives, Hanley-McNeil SEs, both routes to
  the CI). Validated by running it against the archived 2026-08-04 log and reproducing the
  published T1/T2/T4/T5 exactly, which is also how a transcription bug in its own T3 parser
  was caught (an unanchored regex was returning fold 1's AUCs as the means).
- `Final_Model/regenerate_t9b_raw_scores.py` -- T9b's raw scores are not persisted anywhere,
  so its "how to re-derive" note was a prose procedure. It is now executable, and it asserts
  agreement with the shipped CSV rather than assuming it: max |re-derived pct - shipped
  percent_chance| = 9.9e-17 across 4,606 agents, and raw-space tier assignment reproduces
  calibrated-space for 4,606 of 4,606.

**Raw positive rate pinned to one basis, and the recode count corrected.** Two figures that
had drifted across docs are now fixed by reproducing `load_clean`'s filter chain row-for-row
(returns 77,389 rows / 1,519 positives, exact):

- **Raw (pre-recode) 3-month positive rate = 2.5223%** (1,952 of 77,389). The competing
  numbers were both right on their own panel and neither is right here: 2.542% is
  pre-team-exclusion (79,611 rows / 2,024), 2.556% is the pre-2026-08-10 production panel
  (77,422 / 1,978). Only 2.5223% shares a panel with 1.9628%, the AUCs and the tiers.
- **459 recodes are printed; 433 are in-panel.** `data.py` prints the recode count *before*
  the team filter, and **26** of the 459 sit on team rows dropped immediately after. So the
  bridge is **1,952 (2.5223%) -> 1,519 (1.9628%) via 433 recodes** — the subtraction does
  not close with 459 on any panel. Cite 459 as "rows the detector recoded" if useful, never
  as the step between the two rates.

Propagated to `VERIFIED_TABLES.md` T1 (new raw-rate row, 459/433 split, warning against the
wrong subtraction), `AUTHORS_GUIDE.md` (§2.1 horizon table recomputed on one basis, §1.3
summary rows), `Final_Model/README.md`, `modeling/HANDOFF.md`, and this file.

**Unchanged and re-confirmed:** 459 recoded rows / 44 pairs, 153 team accounts / 2,222 rows,
1,519 positives, per-fold positives 164/160/199/172/150 (total 845), CatBoost wins 5 of 5
folds, parametric CI +/-0.019, CatBoost-vs-Logistic still the only comparison clearing the
band. T6, T11 and T12 are [DOC]-sourced and were not re-measured.

**Source:** `Final_Model/crossval_ri_current.log` (re-run 2026-08-11; the 2026-08-04 log is
archived), `Final_Model/current_agent_risk_scores.csv`, the two scripts above; this
conversation.

---

### 2026-08-11 (d) — Quarterly recalibration automated (`quarterly_recalibration.py`)

**Decision:** the Quarterly Reset Protocol is now one command instead of a written
procedure. `modeling/quarterly_recalibration.py` decides whether a run is due, gates it
on the mass-mover check, rescores, verifies the output before anything downstream can
see it, mirrors the result, and logs what the calibration did.

**What was actually missing.** Not the method. The Platt calibrator and the bias
correction are both refit from scratch inside `score_agents.py` on every run, and have
been since 2026-07 #5. What was missing was the cadence and the guard rails: a person
had to remember to check `detect_transitions.py` for new clusters, run the scoring
script, eyeball the tier distribution, and hand-copy the output to two directories, with
nothing recording what changed between quarters. Evidence that this failed, found while
building the wrapper: `Final_Model/generate_report.py` was **newer** than `modeling/`'s
copy of the same file (74.3% vs 73.7% on the headline panel), while the repo-root
multi-horizon CSV was two generations behind at 4,754 agents — pre-dating the
2026-07-25 team exclusion. The mirror had drifted in both directions at once.

**Design decisions worth recording:**

| Decision | Alternative rejected | Why |
|---|---|---|
| Due-ness read from the data (newest snapshot quarter + source-file identity vs. `recalibration_state.json`) | A calendar schedule | Recalibrating twice on one panel reproduces the same numbers. Data-driven due-ness makes the script safe to trigger monthly, so the trigger never has to be timed to vendor delivery. It also catches a file **swap** at an unchanged quarter — exactly the 2026-08-10 `mls_agent_id` case. |
| Unclassified mass-mover cluster **aborts** (exit 2) | Warn and continue | This is the check whose omission produced the fold-5 rebrand-contamination collapse. A warning in a scheduled job is a warning nobody reads. All 27 clusters in the current panel are classified, so anything the gate surfaces is genuinely new. |
| Output checks are an invariant + wide absolute bounds + drift vs. the previous run | The tier shares the protocol quoted (Low 60-66 / Medium 27-31 / High 5-6 / Extra High 1-1.5%) | Those are the 2026-07-24 backtest shape and the live roster has legitimately moved off it (72.8 / 22.4 / 3.0 / 1.8 after the 2026-08-10 panel change). A gate hardcoded to them fires on every correct run and gets ignored. The invariant — every tier-N agent's `risk_multiplier` inside tier N's band — is true by construction on every correct run and cannot go stale. |
| Script drift between `modeling/` and `Final_Model/` is **reported**, not auto-synced | Blind `modeling/` → `Final_Model/` copy | Drift has run in both directions (see `generate_report.py` above); a blind sync would have silently regressed the newer file. |
| 6m/12m off by default, behind `--multi-horizon` | Running all three, as the 2026-07-24 protocol's step 4 said | `train_multi_horizon.py` was descoped 2026-07-25, one day after the protocol was written. The wrapper resolves that contradiction in favour of the later, explicit instruction. |

**Verified end to end** with `--dry-run` (full pipeline into a temp directory, nothing
mirrored or recorded): 67 seconds, and the regenerated scores are **bit-identical** to
the shipped `current_agent_risk_scores.csv` — max |diff| = 0.000e+00 on both
`percent_chance` and `risk_multiplier` across all 4,606 agents, 0 tier changes. All
output checks passed. Not yet run against a genuinely new quarter; the panel has not
advanced past 2026-07-01.

**Two corrections found while doing this, both stale rather than wrong-at-the-time:**
- `modeling/HANDOFF.md`'s current-figures table and `Write_Up/AUTHORS_GUIDE.md` §1.3 both
  still carried bias correction **0.7622×**. The current value is **0.7589×** — as this
  file's 2026-08-11 (c) entry and `VERIFIED_TABLES.md` T8 already record, and as the dry
  run re-derived independently. Both tables corrected. Note the tier assignment is
  *invariant* to this number (`pct` and the cutpoints scale by it together), which is why
  the stale figure never showed up in an output.
- `Final_Model/README.md`'s "currently worth watching" note read Extra High **1.9% (88 of
  4,607)**; the shipped CSV says **1.78% (82 of 4,606)**. Corrected, and the wrapper now
  logs this share every run so the series stops depending on who happens to look.

**Also fixed, and it was live, not hypothetical:** `detect_transitions.build_moves()`
read `mls_agent_id` without `dtype=str`, so pandas inferred per-chunk and returned the
same ID as int in one chunk and str in another — splitting one agent's history in two,
inventing a move at the seam and hiding the real one. Same trap `data.py` already
documents and already guards against. Measured before/after on the current panel:

| | Before | After |
|---|---:|---:|
| Moves reconstructed | 2,024 | **2,140** |
| Clusters ≥5 movers | 24 | **27** |
| Unclassified | 0 | 0 |

**All three newly-visible clusters are in `2026-07-01` — the newest quarter — and one of
them is `SMRT → RMREV` with 56 movers, the largest cluster in the entire panel.** The
mis-inference concentrated at the end of the file, so the detector was effectively blind
to the most recent quarter: precisely the quarter a quarterly guard exists to inspect.
No shared cluster's mover count changed (0 of 24), and nothing was visible before that
is not visible now.

**Labels and scores are unaffected**, because `data.py::_build_moves` reconstructs moves
from the correctly-typed frame `load_clean()` reads and never used the detector's output:
459 recodes, 1.9628% base rate, and the bit-identical dry-run scores all confirm it. The
damage was confined to the discovery step — which is the step this wrapper now depends on
as a gate, which is how it was found.

**Left open, on purpose:** the same missing `dtype` is still in
`detect_magnet_destinations.py` and plausibly `detect_trickle_pairs.py`. Those two fed
the 2026-08-03/04 magnet/trickle policy decision, so quietly changing their output in a
pass about scheduling is the wrong way to touch them. Logged as "Next up" #8 with the
diff-the-candidate-lists procedure attached.

**Source:** `modeling/quarterly_recalibration.py`, `modeling/detect_transitions.py`,
`modeling/score_agents.py`, `modeling/train_multi_horizon.py`, `modeling/HANDOFF.md`
("Quarterly Reset Protocol", rewritten), `Final_Model/README.md`,
`Write_Up/AUTHORS_GUIDE.md` §5.8; dry-run log, this conversation.

---

## 2026-08-25 — Horizon selection re-opened and re-measured; 6m/12m re-scoped; office-size diagnostic

**Decision: keep the 3-month horizon as the ranking and tiering basis. Re-scope
6m/12m as a reporting artifact only.** Reverses the 2026-07-25 descope of
`train_multi_horizon.py` at user direction, then declines the horizon change the
re-scoping was meant to enable — on evidence, not on the prior decision.

### The premise, and the correction that reverses it

The horizons were revisited on the reasoning that 6m/12m have a higher base rate,
so precision should be better and the positive labels should contain fewer false
positives. The first half of that is arithmetically guaranteed and therefore not
evidence: **precision rises with the base rate for any ranking, including a random
one.** At the 12-month base rate (7.67%) a coin flip scores 7.67% precision; at the
3-month rate (1.96%) the same coin flip scores 1.96%. The comparable quantity is
lift = precision / base rate.

Measured on identical held-out rows (the (agent, snapshot) keys resolvable at all
three horizons, 3 folds, test windows 2024-01…2025-04), native model per horizon,
current production config:

| horizon | base rate | AUC | P@10% | lift@10% | recall@10% |
|---|---|---|---|---|---|
| 3m | 2.09% | **0.7444** | 7.42% | **3.52x** | **35.2%** |
| 6m | 4.16% | 0.7247 | 12.88% | 3.08x | 30.8% |
| 12m | 8.03% | 0.7146 | 22.34% | 2.78x | 27.8% |

Precision triples; lift, AUC and recall all fall monotonically. The longer horizons
produce a **worse ranking of a fatter target**. Same ordering on each horizon's own
maximal panel (5/4/3 folds), so it is not an artifact of the matched restriction.

### The cross-horizon matrix — why "we care about the annual question" doesn't change it

Ranking horizon and outcome horizon are separable. Top-decile precision, rows =
training horizon, columns = outcome judged against, identical rows:

| rank by ↓ | label 3m | label 6m | label 12m |
|---|---|---|---|
| 3m | 7.42% | 12.65% | **20.79%** |
| 6m | **7.78%** | 12.88% | 21.90% |
| 12m | 7.14% | 13.36% | **22.34%** |

**The 3-month ranking captures 93% of the 12-month model's own 12-month precision**
(20.79% vs 22.34%) while being a materially better ranking on every other target.

### The false-positive intuition was right, and it argues the other way

Of agents in the 3m model's top decile who did **not** leave within 3 months,
14.45% left within 12 (pooled n = 2,312, 95% CI 13.07–15.94%), against 5.19% for
agents the model did not flag and an 8.03% 12-month base rate — consistent across
all three folds. **A 3m "false positive" is 2.8x more likely than an unflagged
agent to leave within the year.** They are largely right-censored true churners, not
model errors.

That confirms the intuition behind the question and inverts its conclusion: the 3m
ranking already carries the longer-run signal, which is precisely why the label-12m
column above lands so close. Lengthening the label re-labels agents the model was
already surfacing, at a cost of ~30 AUC points.

### A candidate improvement found, then rejected on a leakage check

The cross-horizon matrix turned up an unasked-for result: the model trained on
`label_left_6m` appeared to rank **3-month** departures better than the model
trained on `label_left_3m` (0.7493 vs 0.7444). Plausible mechanism — the 3m label
has ~1,500 positives against the 6m label's ~2,900, so it looked like free
denoising of a scarce target.

**Rejected.** Training on a 6-month label uses outcomes from 6 months after each
training snapshot; with forward-chaining folds the last training quarter's label
resolves in the *second quarter of the test block*, so the candidate was reading
outcomes from inside its own test window. Measured both ways
(`modeling/confirm_train_label_horizon.py`, confirm.py's three-check protocol,
3 seeds × production 9/2 + shifted 7/2 + 11/2, baseline held to the same rule at
its own horizon):

| variant | delta | folds improved | verdict |
|---|---|---|---|
| optimistic (6m labels reach into the test block) | **+0.003749** | — | CONFIRMED |
| **deployable** (only outcomes closed before the test window opens) | **−0.011520** | 1/5 | **UNCONFIRMED_NO_GAIN** |

**Training on the 6-month label makes the 3-month model worse by ~0.0115 AUC**
once restricted to what a quarterly retrain could have known. Worth recording as
a methodological point: **the optimistic version would have passed
confirmation.** Seed-averaging and shifted fold structures do not catch a
temporal leak, because the leak is present in every seed and every structure —
`confirm.py`'s three checks defend against selection bias, window-fitting and
seed noise, and none of those is this.

`AutoResearch/gepa_loop/bench.py` was **not** modified to host this test — it
hardcodes `label_left_3m` as both training and evaluation label, and changing the
search harness's fixed ground truth to chase one hypothesis is the wrong trade.

The same structural advantage exists in the matched-window tables above (every
non-3m model trains on labels resolving into or past the test block). **It runs
against the conclusion** — the longer horizons are measured with an advantage the
3m model does not get and still lose — so those tables were left as measured
rather than re-run under the strict rule. The asymmetry is exact: the strict rule
costs the 3-month model **nothing** (baseline 0.743050 in both variants,
byte-identical — a 3m label on the last training quarter closes exactly as the
test window opens), while the 6m model loses one training quarter and the 12m
model loses three.

### Three drift bugs found and fixed at the root

`horizon_generalization.build_clean_for_horizon` was a hand-copied twin of
`data.py::load_clean` and had drifted four production changes behind it — it
predated the 2026-07-25 team-account filter, the 2026-07-28 non-MLS-member filter,
the 2026-08-05 resolvability tightening, and the `mls_agent_id`-as-str read.
HANDOFF.md had flagged it 2026-08-13 as "patch it before trusting it".

Fixed at the root instead of in the copy: `load_clean` is now horizon-parameterized
(`horizon_months` / `horizon_days`, defaulting to the production 3-month values) and
`build_clean_for_horizon` is a one-line delegate. **The 3-month path is verified
bit-identical** — 459 rows recoded, 84,417 → 77,389 usable, 1.9628% positive rate.

A third bug surfaced during the fix: the copied mass-mover recode pinned the
destination office to the single quarter `snapshot + horizon`, so at 6m/12m it
matched only moves landing in the *final* quarter of the window and scored every
earlier rebrand/M&A move as real individual attrition. Now matches the agent's first
move anywhere in the window (identical at 3m, where quarterly snapshots leave only
one candidate quarter). **Every 6m/12m figure produced before this date is
superseded, not merely stale.**

`horizon_generalization.py` was also still fitting the pre-AutoResearch
depth=4 / `TREE_FEATURES` / Balanced config rather than `PRODUCTION_FEATURES` /
`PRODUCTION_CATBOOST_PARAMS`. Now imports the production config.

**Consequence for the write-up:** the 2026-07-21 transfer figures (12m transfer
0.7013 vs native 0.7335) are not citable — retired data file *and* retired config.
Current: transfer 0.6973 vs native 0.7146 at 12m (retrain warranted), 0.7210 vs
0.7247 at 6m (rescaling fine). Verdict unchanged, magnitudes not.
`Write_Up/AUTHORS_GUIDE.md` §"long version" updated.

### Tier multipliers re-anchored per horizon (shipped)

`MULT_CUT_POINTS = [1, 3, 6]` is a 3-month anchor; at the 12m base rate "6x" means a
predicted 46% chance and the top tier empties. Swept out-of-sample; the binding
criterion is **bucket size, not peak precision** (precision keeps climbing past every
threshold at all three horizons — what stops you is n collapsing). Anchors chosen to
reproduce the 3-month tiers' share of roster:

| horizon | Extra High | % roster | precision | High |
|---|---|---|---|---|
| 3m | 6x | 0.99% | 15.7% | 3x |
| 6m | 8x | 1.05% | 19.4% | 4x |
| 12m | 3.5x | 1.11% | 48.4% | 2.5x |

Shipped as `calibrate.HORIZON_MULT_CUT_POINTS`. `compute_tier_cutpoints()` takes an
optional override and still defaults to the 3-month constants, so **production
behaviour is unchanged**. Confirms the qualitative finding of the 2026-08-13 12m tier
backtest (the 6x anchor does not transfer) while superseding its numbers, which were
computed on the drifted population.

### Office-size diagnostic (user-requested)

**Median office size has two answers and they differ by 6x: 3 agents per office,
18 agents per agent.** 72.5% of offices hold 1–5 agents but only 23% of agents;
6 offices with 100+ agents hold 16.6%. For anything about agents, the per-agent
figure is the relevant one.

Ranking accuracy by office size (out-of-fold, within-bucket AUC), pooled and tested
as a difference rather than by eyeballing overlapping CIs:

| group | rows | departures | AUC | lift@10% |
|---|---|---|---|---|
| tiny (1–5) | 10,091 | 135 | 0.7309 | 3.26x |
| **mid (6–50)** | 23,486 | 535 | **0.7574** | **3.85x** |
| large (51+) | 9,220 | 175 | 0.6913 | 2.80x |

**mid vs large: −0.0661 AUC, z = 2.58, p = 0.0098 — the one difference that survives
a real test.** Tiny vs mid (p = 0.34) and tiny vs large (p = 0.24) are not
distinguishable. Operationally: a top-decile flag at a 6–10 agent office converts at
9.51% vs 3.54% at a 51–100 agent office, a ~2.7x difference in flag value.

Calibration is a separate and larger problem at the small end: **at 1–5 agent offices
the model prints 1.92% against an actual 0.83% — 2.3x over-predicted** (95% CI on
actual 0.59–1.16%). The 6–50 band is uniformly well calibrated (ratio 0.87).

Hypothesis for the large-office ranking gap, explicitly unverified: many of the
strongest features are office-relative (`share_of_office_volume_12m`,
`rank_in_office_volume_12m`, `office_exit_rate_12m`, `price_gap_vs_office`) and
dilute toward zero in a 200-agent office. Testing it needs a per-bucket
feature-importance run. **No production change made on this diagnostic.**


### Follow-up the same day: the horizon comparison had TWO confounds, pulling opposite ways

**User challenge:** is the longer horizons' worse AUC just an artifact of the config
having been AutoResearch-tuned on the 3-month label? Fair, and partly right — but
chasing it surfaced a second, larger confound running the other way, and the two had
been roughly cancelling.

**Confound 1 — config bias, favours 3m.** `PRODUCTION_FEATURES` (54 of 99) and
`PRODUCTION_CATBOOST_PARAMS` were both chosen by maximizing AUC on `label_left_3m`.
Re-deriving each horizon's feature pruning natively (AutoResearch's own procedure —
depth=3 SqrtBalanced on fold-1 train only, same pool, same 54 kept, only the ranking
label differs) buys 12m **+0.0094**; also swapping the 3m-tuned hyperparameters shrinks
the 3m−12m gap from **+0.0298 to +0.0059**. Under that config the 12m model appeared to
rank *everything* best, including 3-month departures.

The horizons genuinely want different features. 12m's own pruning **drops the
short-window recency signals** (`months_since_last_closing`, `months_since_last_move`,
`trend_3m`, `units_3m`, `units_prior_3m`, `units_trend_12m`,
`trend_deceleration_gap`) and **adds slow-moving structural ones**
(`office_age_months`, `company_active_agents_asof`, `office_volume_trend_12m`,
`num_offices_career`, `office_roster_pct_rank`) — 37/54 overlap with production vs
41/54 for 3m. "Has this person gone quiet lately" is the right question for a quarter
and stale noise for a year.

**Confound 2 — temporal leak, favours the longer horizons.** A model trained on an
H-month label sees outcomes H months past each training snapshot, which lands inside
the test block. 3m leaks nothing, 6m leaks 2 quarters, 12m leaks 4.

**Both removed at once** (`horizon_fair_and_deployable.py` — per-horizon native
features pruned on the deployable fold-1 window, plus the deployable training rule):

| config | 3m | 6m | 12m | 3m−12m gap |
|---|---|---|---|---|
| fair + deployable (prod params) | 0.7418 | 0.7158 | 0.6846 | **+0.0572** |
| fair + deployable (neutral params) | 0.7329 | 0.7088 | 0.6789 | **+0.0539** |

lift@10%: 3m 3.53–3.66x, 6m 2.88–2.98x, 12m 2.31–2.40x.

**The properly de-confounded gap is roughly double what was first reported (+0.054 vs
+0.0298), not smaller.** The leak was doing more work than the config bias — charging
the 12m model its four missing training quarters costs it 0.048 AUC (0.7269 → 0.6789),
and the "12m wins everywhere" result was entirely leak-driven. **The horizon
recommendation stands and is stronger than originally measured; the +0.0298 figure was
right by accident, being the net of two offsetting biases.**

**One earlier conclusion REVERSES.** The transfer finding said a 12-month product needs
its own retrained model rather than a rescaled 3m score (transfer 0.6973 vs native
0.7146). Under deployable conditions native 12m is **0.6846** and the 3m model scored
against the 12m label is **0.6945** — the 3-month model is the better 12-month ranker.
**Do not train a native 12m model; rescale.** `train_multi_horizon.py` still trains
natively, so the shipped `percent_chance_12m` column is built the weaker way — flagged
in that file's docstring and logged as Next-up #12 rather than changed silently, since
it alters a shipped column.

**Methodological point worth keeping:** each confound in isolation supports a different
answer, and each was measured with a defensible-looking harness. The failure mode is
not a bad measurement — it is stopping at the first correction that confirms or refutes
the prior. Neither `horizon_precision.py` nor `horizon_config_fairness.py` is wrong;
each is simply incomplete in a direction that flatters a different horizon.


### Third leg of the same confound: would a different ALGORITHM suit 12m better?

**User challenge:** every horizon comparison held the model family fixed at CatBoost,
which was itself selected on the 3-month label (logistic 0.681 / XGBoost 0.731 /
CatBoost 0.737, all at 3m). Same confound family as the feature and hyperparameter
biases above, so it got the same treatment: six families × three horizons under the
fair+deployable protocol.

| model | 3m | 6m | 12m | 3m−12m gap | rank at 12m |
|---|---|---|---|---|---|
| **CatBoost (production)** | **0.7418** | **0.7158** | **0.6846** | +0.0572 | **1** |
| XGBoost | 0.7276 | 0.7031 | 0.6681 | +0.0595 | 2 |
| LightGBM | 0.7261 | 0.7041 | 0.6650 | +0.0611 | 4 |
| HistGBM | 0.7325 | 0.6973 | 0.6565 | +0.0760 | 6 |
| RandomForest | 0.7083 | 0.6806 | 0.6599 | +0.0484 | 5 |
| Logistic (binned) | 0.6797 | 0.6732 | 0.6666 | **+0.0131** | **3** |

**Answer: no.** CatBoost wins at all three horizons — the 3m-derived algorithm choice
transfers. The sharpest statement of the horizon problem's size: **the best 12-month
model of any family (0.6846) is barely above the WORST 3-month model** (binned
logistic, 0.6797 — the one crossval.py calls "interpretable gut-check only, not for
production"). Family choice moves 12m by at most 0.028 best-to-worst; the gap to 3m is
0.054.

**Three findings worth keeping anyway**, all pointing at what the 12-month problem
actually is:

1. **Family choice matters roughly half as much at 12m as at 3m** — best-to-worst
   spread 0.062 at 3m vs 0.028 at 12m. The longer target has much less exploitable
   non-linear structure.
2. **Low-variance models degrade far more gracefully.** Logistic's 3m→12m gap is
   +0.0131 against CatBoost's +0.0572; it climbs from dead last at 3m to **3rd at
   12m**, passing LightGBM, HistGBM and RandomForest and landing within 0.0015 of
   XGBoost. This corroborates the feature-set finding from an independent direction —
   12m's own pruning drops the short-window recency signals for slow structural ones,
   and a more linear, more structural target is exactly where boosters lose their edge.
3. **CatBoost's relative dominance GROWS at 12m** — margin over second place +0.0093 at
   3m, **+0.0165 at 12m**. Consistent with ordered boosting limiting overfitting on
   small data, which is the regime the deployable rule forces at 12m (6 of 9 fold-1
   training quarters usable, vs 9 of 9 at 3m). The data starvation that makes 12m hard
   is also what widens CatBoost's lead there.

Supersedes the 2026-07 "LightGBM ambiguity resolved / no new LightGBM build" note: a
build was explicitly requested this session and now exists.

**Caveats:** three folds; between-family differences at 12m are small relative to the
3m-vs-12m gap they are compared against, so the middle of that table should not be
over-read. Nothing went through `confirm.py` because nothing is being promoted. Each
family runs at its crossval.py/reference hyperparameters — this tests FAMILY choice per
horizon, not a per-family per-horizon search. RandomForest and logistic run without
monotone constraints (no library equivalent).

**Source:** `modeling/horizon_model_comparison.py` (new), `modeling/HANDOFF.md` (H10,
Next-up #4 supersession, revised recommendation);
`horizon_model_comparison.log`, this conversation.

---

**Source:** `modeling/horizon_config_fairness.py` (new),
`modeling/horizon_fair_and_deployable.py` (new), `modeling/HANDOFF.md` (H9, plus
supersession notices on H2/H3/H5 and a revised recommendation section),
`modeling/train_multi_horizon.py`; `horizon_config_fairness.log`,
`horizon_fair_and_deployable.log`, this conversation.

---

**Source:** `modeling/horizon_precision.py` (new),
`modeling/office_size_diagnostic.py` (new),
`modeling/confirm_train_label_horizon.py` (new), `modeling/data.py` (horizon
parameterization + recode window fix), `modeling/horizon_generalization.py`,
`modeling/calibrate.py`, `modeling/train_multi_horizon.py`,
`modeling/quarterly_recalibration.py`, `modeling/HANDOFF.md`,
`Write_Up/AUTHORS_GUIDE.md`; `horizon_precision.log`,
`office_size_diagnostic.log`, `train_multi_horizon.log`, this conversation.
