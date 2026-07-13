"""Manually verified (origin_office, dest_office) transition classifications.

Built by: detect_transitions.py finds candidate mass-simultaneous office
transitions directly from agent movement data (no web search needed to
FIND candidates), then each candidate is manually verified via web search
(see conversation record) and cached here by office-code pair so the
classification applies to every agent who made that transition, not just
the one investigated first (e.g. SMRT).

Rule (revised 2026-07 #3, current -- supersedes rounds 1-3 below): TWO
different gates depending on how confident we are about a pair:

  CONFIRMED_PAIRS (was REBRAND_PAIRS): a pair we have INDEPENDENT evidence
  for -- web-researched, confirmed same ownership/rebrand, M&A/acquisition,
  or same-brand-family roll-up. Because the evidence is independent of any
  specific quarter's mover count, EVERY move on that exact office pair is
  recoded to not-churn, in ANY quarter, at ANY volume -- including a lone
  agent trickling over a quarter or two after the main event. We know it's
  the same non-event regardless of how many people moved that quarter.

  CLUSTER_GATED_PAIRS (was GENUINE_PAIRS + AMBIGUOUS_UNCONFIRMED_PAIRS): a
  pair with NO independent evidence of any corporate relationship (or none
  found either way). For these, the ONLY reason to disregard a move at all
  is the mass-mover PATTERN itself (>=5 agents, same origin, same dest,
  same quarter -- a team-lead/office-level decision, not individual free
  will). So the per-quarter, per-pair mover-count gate (MIN_CLUSTER_MOVERS,
  data.py::_recode_mass_mover_moves) still applies here: only the actual
  flagged instance(s) get recoded, not every future move on that pair.

Why the split: round 3 (2026-07 #2) required the >=5-same-quarter gate
for EVERY pair, including ones we'd already confirmed via web research.
That under-caught real non-events: several confirmed pairs (KSPG->MGLM,
TIRR->MGLM, the CBRB internal office-code pairs, KELW03->SMRT) had
additional 3-4-mover "trickle" instances in adjacent quarters that were
still counted as real churn purely because that specific quarter didn't
independently clear 5 movers -- even though we already know, from
independent evidence, that ANY move on that pair isn't real churn. The
gate makes sense as a DISCOVERY mechanism (for pairs with no other
evidence) but was too strict as a RECODING mechanism for pairs we've
already confirmed.

MIN_CLUSTER_MOVERS (=5, see data.py) has no sharp statistical elbow
justifying it over 4 or 6 -- full-panel distribution of same-origin/
same-dest/same-quarter cluster sizes is a smooth decay (1457 singleton
moves, 110 pairs of 2, 38 of 3, 18 of 4, then only 27 instances of 5+).
5 is defensible as "top 1.6% of clusters, well past where organic
coincidence dominates" but is not a provably-optimal cutoff. This mainly
matters for CLUSTER_GATED_PAIRS/discovery, since CONFIRMED_PAIRS no
longer depend on it at all.

Historical rounds (superseded, kept for context):
  Round 1 (original, narrowest): only same-ownership/franchise-flag-only
  rebrands recoded; confirmed M&A/acquisitions counted as real churn.
  Round 2 (2026-07 #1): widened to also cover confirmed M&A/acquisitions
  and same-brand-family roll-ups; unrelated-competitor moves still churn.
  Round 3 (2026-07 #2): every >=5-mover cluster excluded from churn,
  confirmed or not, but ALWAYS gated on that exact quarter independently
  clearing 5 movers -- the bug this round's split fixes.

This list is NOT exhaustive -- only candidates that have surfaced via
detect_transitions.py's >=5-mover threshold have ever been reviewed (plus
one exception below: the HomeSmart-family pattern was checked on its own
initiative even though it never cleared 5 movers, and confirmed genuine).
Re-run detect_transitions.py whenever new quarters of data arrive; new
candidates need manual web verification before being added to either
bucket.
"""

CONFIRMED_PAIRS = {
    ("SMRT", "RMREV"): "HomeSmart Professionals -> REMAX Revolution: same owner-broker (Dean deTonnancourt), franchise flag change only",
    ("RMAX09", "FORB"): "RE/MAX Flagship -> Flagship Real Estate Advisors: same leadership (Benjamin Emerick), disaffiliated from RE/MAX franchise",
    ("RNDL05", "CMPM03"): "Randall Realtors Compass -> Compass: final branding step, confirmed same leadership team/staff/marketing retained",
    ("LVGP", "LVGP02"): "Real Broker, LLC -> Real Broker, LLC: identical name, internal office-code change",
    ("CBRB23", "CBRB24"): "Coldwell Banker Realty -> Coldwell Banker Realty: identical name, internal office-code change",
    ("CBRB15", "CBRB06"): "Coldwell Banker Realty -> Coldwell Banker Realty: identical name, internal office-code change",
    ("CBHN", "CBRB18"): "Coldwell Banker Realty -> Coldwell Banker Realty: identical name, internal office-code change",
    ("KELW06", "KELW02"): "Keller Williams Coastal -> Keller Williams Coastal: identical name, internal office-code change",
    ("KSPG", "MGLM"): "Keystone Property Group acquired by Lamacchia Realty: local broker (Jodi Hedrick) stayed on, corporate ownership changed -- acquisition",
    ("TIRR", "MGLM"): "Tirrell Realty acquired by Lamacchia Realty: local leader (Phil Tirrell) stayed on, corporate ownership changed -- acquisition",
    ("CMWT02", "CMWT11"): "Century 21 The Seyboth Team absorbed into Century 21 Limitless: multi-team roll-up/consolidation, same brand family",
    ("CMWT04", "CMWT11"): "Century 21 Premier Agency absorbed into Century 21 Limitless: multi-team roll-up/consolidation, same brand family",
    ("CMWT10", "CMWT11"): "Century 21 Limitless PRG -> Century 21 Limitless: part of the same roll-up/consolidation, same brand family",
    ("PMRG", "CMWT10"): "Premier Realty Group -> Century 21 Limitless PRG: part of the same roll-up/consolidation, same brand family",
}

# Confirmed moves to a genuinely unrelated competitor (no corporate
# relationship of any kind found) -- excluded ONLY at the specific
# >=5-mover cluster instance(s) that got them flagged in the first place
# (see MIN_CLUSTER_MOVERS gate in data.py). A future solo move on these
# same office pairs, outside a genuine cluster, still counts as churn.
GENUINE_PAIRS = {
    ("RESI", "CMPM"): "Residential Properties -> Compass: no merger evidence found; firms shown as competitors trading agents both directions",
    ("CMWT10", "EXPW03"): "Century 21 Limitless PRG -> eXp Realty LLC: different brand family, no corporate relationship found",
    ("CMWT03", "LVGP"): "Century 21 Shoreline -> Real Broker, LLC: different brand family, no corporate relationship found",
    ("KELW03", "SMRT"): "Keller Williams Leading Edge -> HomeSmart Professionals: different brand family, no corporate relationship found",
}

# Unconfirmed/no-evidence-either-way pairs -- same cluster-instance-only
# gating as GENUINE_PAIRS above; listed here only because they were never
# actually researched to a conclusion, not because the gating differs.
AMBIGUOUS_UNCONFIRMED_PAIRS = {
    ("KELW01", "KELW02"): "Keller Williams Realty -> Keller Williams Coastal: no corroborating evidence found either way",
    ("RMAX43", "INOV"): "RE/MAX Innovations -> Innovations Realty: no evidence found; RE/MAX Innovations still shown active as of Dec 2025",
    ("RMAX15", "RMHT"): "RE/MAX 1st Choice -> RE/MAX ONE: no local-specific evidence (broader global Real/RE-MAX $880M deal exists but hadn't closed as of this data)",
    ("CE2111", "CE2139"): "C-21 Butterman & Kryston -> Century 21 AllPoints Realty: no corroborating evidence found",
    ("KELW05", "KELW03"): "Keller Williams Realty Leading -> Keller Williams Leading Edge: not directly searched, name-similarity prior only",
}

# Checked on its own initiative (2026-07 #3): the "HomeSmart"-branded
# family looked suspicious given the confirmed SMRT->RMREV rebrand, but
# neither pair ever cleared MIN_CLUSTER_MOVERS (4 movers each, one quarter
# each) so neither was ever a detect_transitions.py candidate or is in any
# bucket above -- these rows correctly remain counted as real churn.
#   SMRT (HomeSmart Professionals, owner Dean deTonnancourt) -> HSFC
#   (HomeSmart First Class Realty, owner Ryan Cook, MA): DIFFERENT
#   independent franchisee under the same international brand -- genuine.
#   HSHR (HomeSmart Heritage Realty, owner Jason Araujo, MA) -> SMRT:
#   also a DIFFERENT independent franchisee -- genuine.

# Pairs excluded from churn with NO per-quarter/mover-count gate --
# independent evidence already tells us any move on these is a non-event.
CONFIRMED_NOT_CHURN_PAIRS = CONFIRMED_PAIRS

# Pairs excluded from churn ONLY for the specific quarter(s) that
# independently clear MIN_CLUSTER_MOVERS -- data.py checks cluster size
# per (origin, dest, quarter) for these before recoding.
CLUSTER_GATED_PAIRS = {**GENUINE_PAIRS, **AMBIGUOUS_UNCONFIRMED_PAIRS}

# Combined, for anything that just needs "is this pair known at all"
# (e.g. detect_transitions.py cross-checks, documentation).
NOT_CHURN_PAIRS = {**CONFIRMED_NOT_CHURN_PAIRS, **CLUSTER_GATED_PAIRS}
