"""Manually verified (origin_office, dest_office) transition classifications.

Built by: detect_transitions.py finds candidate mass-simultaneous office
transitions directly from agent movement data (no web search needed to
FIND candidates), then each candidate is manually verified via web search
(see conversation record) and cached here by office-code pair so the
classification applies to every agent who made that transition, not just
the one investigated first (e.g. SMRT).

2026-08-04 (CURRENT POLICY, user-directed): magnet and trickle are now
governed SEPARATELY -- magnets count as CHURN, trickles do not, and
documented rebrands/acquisitions (CONFIRMED_PAIRS) do not.

HSHR->SMRT, the single pair the magnet route ever suppressed, was moved
into CONFIRMED_PAIRS the same day, because its suppression always rested
on a documented consolidation announcement (RISMedia 2024-01-16) rather
than on the magnet geometry. Evidence governs it; the magnet policy does
not reach it. _MAGNET_PAIRS is consequently empty BY DESIGN.

Net label effect of the whole 2026-08-04 round: none. 44 pairs suppress,
459 rows recode, and every metric came out identical to 2026-08-03. (Those
metrics read base rate 1.9620% / mean AUC 0.7411 at the time; on the current
panel they are 1.9628% / 0.7440, moved by the 2026-08-10 placeholder
exclusions, not by anything in this file -- the labels here are untouched.) What changed is that the policy is now explicit and
per-geometry rather than bundled, so the reasoning is inspectable and a
future magnet find cannot be suppressed on shape alone.

2026-08-03: a proposed policy change was applied, measured, and then
REVERSED after review. Net effect on labels at the time: none -- all 44
pairs suppressed as they had before. The episode is recorded here because
the reasoning on both sides is worth keeping, and because the sub-bucket
structure it introduced (MAGNET_TRICKLE_PAIRS and its three sub-dicts) is
exactly what made the 2026-08-04 split a one-line change. NOTE that the
2026-08-03 proposal was all-or-nothing across BOTH routes; the current
policy adopts it for magnets only, which is a different call and carries a
much smaller label change (1 pair, not 30).

The proposal: count the MAGNET and TRICKLE detection shapes as agent
churn, on the grounds that a magnet destination often reflects one
brokerage's marketing campaign or signing bonus (an agent taking a better
offer left voluntarily), and that a trickle of a few agents per quarter
can be entirely natural -- the steady bleed office_exit_rate_12m is meant
to measure.

Why it was reversed: the detectors never classified anything BY SHAPE.
They are discovery tools, and natural drift was already filtered out
before any pair reached this file. Of 45 trickle pairs surfaced for
review, 16 were checked and deliberately LEFT AS CHURN (RMAX01<->RMAX45's
bidirectional flow, DRHM->LVGP, RGEX->RDRM02, CMWT13->CMWT04 flowing out
of the consolidated entity), and ~200 lower-volume trickle-shaped pairs
never cleared review at all. The 29 that survived are the residue after
that filter: 17 with a byte-identical origin/destination office_name
(Coldwell Banker Realty -> Coldwell Banker Realty -- an internal MLS
office-code renumber where nobody changed employers), and 12 with a
documented corporate event behind them. None is unexplained organic
drift, so the "trickle can be natural" rationale, though true of the
shape in general, does not describe these specific pairs.

Tier B is the founding bug of this project, phased: Randall Realtors and
Lila Delman -> Compass are press-documented acquisitions, the same event
class as HomeSmart Professionals -> REMAX Revolution (120 contaminated
labels, fold 5 collapsing to 0.63 AUC -- the diagnosis this whole system
was built from). The only difference is that Compass rebranded its agents
over several quarters instead of one. Same cause, slower clock -- which
is precisely what detect_trickle_pairs.py exists to catch.

The one thing acquisitions DO produce that is real churn: agents who
leave BECAUSE their firm was bought. Those moves go origin -> some THIRD
office, and they have always counted as churn. Suppression only ever
covered moves to the acquirer itself, which is the one destination that
isn't a choice.

Measured cost of the reversed proposal, for the record (3 seeds,
production CatBoost, fold structures (9,2)/(7,2)/(11,2)): counting all 30
pairs as churn moved mean AUC 0.741511 -> 0.750005. That was not a model
improvement -- it restored 118 office-clustered positives to a target the
model's office-level features rank well, i.e. it made the target easier.
Reverting gives the 0.0085 back, which is the correct trade. Also
measured: the 17 identical-name pairs are AUC-NEUTRAL either way
(0.749994 vs 0.750005, a 0.00001 difference), so their classification can
always be decided on correctness alone.

Rule (revised 2026-07 #3 -- supersedes rounds 1-3 below): TWO
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
    # Moved here from _MAGNET_PAIRS on 2026-08-04 (user-directed). It was
    # only ever suppressed on documented-consolidation evidence, never on
    # the magnet geometry, so it belongs in the evidence-classified bucket
    # where the magnet policy cannot reach it. See the POLICY block below.
    ("HSHR", "SMRT"): "HomeSmart Heritage Realty (Jason Araujo, Fall River MA) -> HomeSmart Professionals (Dean deTonnancourt): RISMedia 2024-01-16 confirms Heritage formally joined Professionals for shared support/tech/training -- a real consolidation despite HSHR retaining separate ownership on paper. Corrects the 2026-07 #3 'different independent franchisee' verdict, which only checked ownership and missed this announcement.",
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

# =====================================================================
# MAGNET/TRICKLE-DISCOVERED PAIRS -- these are NOT CHURN (suppressed).
#
# Split into sub-dicts by discovery route during the 2026-08-03 proposal
# and reversal (see the module docstring). They are folded back into
# CONFIRMED_NOT_CHURN_PAIRS below via COUNT_MAGNET_TRICKLE_AS_CHURN=False,
# so every move on these pairs is suppressed in ANY quarter at ANY volume,
# exactly as it was before 2026-08-03. The decomposition is kept because
# it is a genuinely useful way to reason about these pairs, not because
# any of them are currently treated differently from one another.
# =====================================================================

# Found by detect_magnet_destinations.py (round 1). DELIBERATELY EMPTY, and
# that emptiness is the point: the magnet SHAPE has never suppressed a label
# in this project, and as of 2026-08-04 it formally never will.
#
# The detector surfaced three magnet destinations. Two -- SERHANT (24
# arrivals/16 origins) and LVGP/Real Broker -- were verified and
# deliberately LEFT LABELED AS CHURN, since both are individual recruitment
# (Real Brokerage's revenue-share model, SERHANT's brand pull) rather than
# an office-level event. The third, HSHR->SMRT, was suppressed on a
# documented consolidation announcement rather than on its geometry, and on
# 2026-08-04 it was moved into CONFIRMED_PAIRS so that the evidence governs
# it directly. Nothing is left here.
#
# Keep this dict rather than deleting it: if a future magnet candidate is
# ever suppressed, it must be because independent evidence says so, in which
# case it belongs in CONFIRMED_PAIRS too -- and an empty dict here is the
# clearest possible statement of that rule.
_MAGNET_PAIRS = {}

# --- 2026-07-25 round 2: detect_trickle_pairs.py sweep. Two tiers below. ---

# Tier A: EXACT origin/dest office_name match -- NOT CHURN, and the
# firmest call in this file. An agent whose office_mls_id changed while
# the office_name stayed byte-identical ("Coldwell Banker Realty" ->
# "Coldwell Banker Realty") did not change brokerages; the MLS renumbered
# an internal office code. Confirmed as the intended reading 2026-08-03:
# same name = same business = not churn, regardless of which detector
# found the pair or how the codes are numbered. Measured as AUC-neutral
# (0.00001), so this rests entirely on correctness, which is where it
# belongs.
_TRICKLE_IDENTICAL_NAME_PAIRS = {
    # No web search performed or
    # needed -- same zero-cost reasoning already used above for LVGP->LVGP02,
    # CBRB23->CBRB24, CBRB15->CBRB06, CBHN->CBRB18, KELW06->KELW02: two office
    # codes reporting the identical business name are, by construction, the
    # same underlying business under a new internal office code.
    ("CBRB15", "CBRB24"): "Coldwell Banker Realty -> Coldwell Banker Realty: identical name, internal office-code change",
    ("CBRB06", "CBRB18"): "Coldwell Banker Realty -> Coldwell Banker Realty: identical name, internal office-code change",
    ("CBRB18", "CBRB24"): "Coldwell Banker Realty -> Coldwell Banker Realty: identical name, internal office-code change",
    ("CBRB17", "CBRB23"): "Coldwell Banker Realty -> Coldwell Banker Realty: identical name, internal office-code change",
    ("C21A04", "CE2129"): "Century 21 Topsail Realty -> Century 21 Topsail Realty: identical name, internal office-code change",
    ("CE2129", "C21A04"): "Century 21 Topsail Realty -> Century 21 Topsail Realty: identical name, internal office-code change (reverse direction of the same pair)",
    ("CE2135", "CE2129"): "Century 21 Topsail Realty -> Century 21 Topsail Realty: identical name, internal office-code change",
    ("CE2132", "CE2104"): "CENTURY 21 Stachurski Agency -> CENTURY 21 Stachurski Agency: identical name, internal office-code change",
    ("CMPM03", "CMPM04"): "Compass -> Compass: identical name, internal office-code change",
    ("CMPM05", "CMPM04"): "Compass -> Compass: identical name, internal office-code change",
    ("CMPM", "CMPM08"): "Compass -> Compass: identical name, internal office-code change",
    ("CMPM", "CMPM02"): "Compass -> Compass: identical name, internal office-code change",
    ("KELW04", "KELW02"): "Keller Williams Coastal -> Keller Williams Coastal: identical name, internal office-code change",
    ("KELW07", "KELW02"): "Keller Williams Coastal -> Keller Williams Coastal: identical name, internal office-code change",
    ("KLWL", "KWRE"): "Keller Williams Realty -> Keller Williams Realty: identical name, internal office-code change",
    ("MOCH05", "MOCH08"): "Mott & Chace Sotheby's Intl. -> Mott & Chace Sotheby's Intl.: identical name, internal office-code change",
    ("MOCH", "MOCH02"): "Mott & Chace Sotheby's Intl. -> Mott & Chace Sotheby's Intl.: identical name, internal office-code change",
}

# Tiers B and C of the same trickle sweep -- NOT CHURN. Pairs backed by
# press coverage of a documented acquisition (B: Randall Realtors and Lila
# Delman into Compass) or by a near-identical name / direct chain
# extension of an already-confirmed pair (C). Tier B is the founding
# fold-5 rebrand bug on a slower clock: a documented acquisition whose
# rebrand was phased across quarters rather than landing in one.
_TRICKLE_EVIDENCE_PAIRS = {
    # Tier B: web-confirmed via press coverage (Randall Realtors / Lila Delman
    # -> Compass, both multi-year phased acquisitions). RNDL05->CMPM03 above
    # was already independently confirmed; these are sibling office codes of
    # the SAME two documented deals -- Compass acquired Lila Delman Real
    # Estate in Jan 2021 (NEREJ, liladelman.com, Newport Buzz), and Randall,
    # Realtors joined Compass the same year, with agents formally
    # transitioning to the Compass brand Nov 8 2023 (Compass newsroom, NEREJ,
    # PBN, Charlestown Patch) after operating as "Randall Realtors Compass"/
    # "Lila Delman Compass" in the interim -- exactly the multi-quarter
    # trickle shape this script was built to catch.
    ("RNDL", "CMPM04"): "Randall, REALTORS Compass -> Compass: same documented Compass acquisition as RNDL05->CMPM03 (Compass newsroom/NEREJ/PBN, agents transitioned to Compass brand 2023-11-08)",
    ("RNDL02", "CMPM04"): "Randall, REALTORS Compass -> Compass: same documented Compass acquisition as RNDL05->CMPM03",
    ("RNDL02", "CMPM05"): "Randall, REALTORS Compass -> Compass: same documented Compass acquisition as RNDL05->CMPM03",
    ("RNDL07", "CMPM06"): "Randall, REALTORS Compass -> Compass / Lila Delman Compass: same documented Compass acquisition, transitional sub-office code",
    ("LILA06", "CMPM"): "Compass / Lila Delman Compass -> Compass: Compass acquired Lila Delman Real Estate Jan 2021 (NEREJ, liladelman.com, Newport Buzz), phased rebrand of the Compass office code",
    ("LILA06", "LILA"): "Compass / Lila Delman Compass -> Lila Delman Compass: same documented Lila Delman/Compass acquisition, internal code consolidation",
    ("LILA07", "CMPM09"): "Compass / Lila Delman Compass -> Compass: same documented Lila Delman/Compass acquisition",

    # Tier C: near-identical name (not byte-exact, but unambiguously the same
    # business) or a direct chain extension of an already-confirmed pair
    # above -- no new web search needed given what's already established.
    ("RMAX40", "RMAX09"): "RE/MAX Flagship -> RE/MAX FLAGSHIP, INC.: same business already identified above (RMAX09/FORB entry) as Benjamin Emerick's RE/MAX Flagship -- name variant, not a different entity",
    ("PLTM", "CMWT08"): "Platinum Real Estate Group -> Century 21 Platinum RE Group: near-identical 'Platinum' branding, consistent with an independent brokerage franchising into Century 21 under its existing name",
    ("CMWT05", "CMWT13"): "Century 21 Visionary Group -> Century 21 Limitless: CMWT13's name is identical to CMWT11's already-confirmed identity (Century 21 Limitless) above -- extension of the same already-verified roll-up under a newer office code",
    ("CMWT02", "CMWT10"): "Century 21 The Seyboth Team -> Century 21 Limitless PRG: CMWT02 is already confirmed above (CMWT02->CMWT11) as part of the Limitless roll-up; CMWT10 is already confirmed above (CMWT10->CMWT11) as an intermediate code in the same chain -- this is an earlier hop in the identical, already-verified consolidation",
    ("CMWT08", "CMWT10"): "Century 21 Platinum RE Group -> Century 21 Limitless PRG: CMWT08 is the PLTM entity above, already tied into the Limitless roll-up chain via CMWT10->CMWT11",
}

# Checked this round (2026-07-25, detect_trickle_pairs.py), NOT confirmed --
# real web searches performed, no corroborating evidence found. Left as
# genuine churn, same as any unconfirmed candidate. Documented here so a
# future session doesn't re-spend a search on the same question from
# scratch.
#   C21A01 (Century 21 Access America) -> CE2137 (Century 21 Guardian
#   Realty): no merger evidence found; Access America's only documented
#   merger on record is into Century 21 Topsail Realty (2014, unrelated).
#   RMAX01 (RE/MAX Profnl. Newport) <-> RMAX45 (RE/MAX Results), RMAX11
#   (RE/MAX Advantage Group) -> RMAX45, RMAX02 (RE/MAX Professionals) ->
#   RMAX45: no merger evidence found. The RMAX01<->RMAX45 flow is
#   BIDIRECTIONAL (7 movers one way, 3 the other, across separate quarters)
#   -- a real one-way roll-up should drain in one direction only, so this
#   pattern reads as ordinary two-way competition between offices, same
#   shape as the already-confirmed-genuine RESI<->CMPM pair.
#   RMAX41 (RE/MAX On The Move) -> RMAX37 (RE/MAX River's Edge), RMAX41 ->
#   RMAX11: no evidence found either way.
#   CMWT04 (Century 21 Premier Agency) -> CMWT12 (Century 21 Luxe Property
#   Grp): different name, no relationship to the already-confirmed Limitless
#   roll-up chain found; not the same as CMWT04's OTHER already-confirmed
#   destination (CMWT11).
#   CMWT13 (Century 21 Limitless) -> CMWT04 (Century 21 Premier Agency):
#   flows OUT of the confirmed Limitless entity back to Premier Agency --
#   opposite direction from the roll-up pattern, reads as an unrelated
#   individual move, not part of the consolidation.
#   RGEX (Realty ONE Group Executives) -> RDRM02 (Realty One Group Dream
#   Makers): shared national franchise brand only, no local relationship
#   evidence found (same shape as the already-confirmed-genuine
#   KELW03->SMRT "different brand family" pattern).
#   PARS/PARS02 (Century 21 Ed Pariseau/Pariseau R.E.) -> NSGR05 (Century 21
#   North East), KELW01 (Keller Williams Realty) -> KELW03 (Keller Williams
#   Leading Edge), KELW02 (Keller Williams Coastal) -> KELW03: no evidence
#   found either way (same status as the existing AMBIGUOUS_UNCONFIRMED_PAIRS
#   KELW05->KELW03 entry above).
#   DRHM (My Dream Home Realty) -> LVGP (Real Broker, LLC), RMAX45 (RE/MAX
#   Results) -> JPAR (JPAR Prime Real Estate): no brand relationship, no
#   evidence found -- most consistent with ordinary individual recruitment,
#   same shape as LVGP's other confirmed-genuine growth (see
#   detect_magnet_destinations.py findings).

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
# each) so neither was ever a detect_transitions.py candidate.
#   SMRT (HomeSmart Professionals, owner Dean deTonnancourt) -> HSFC
#   (HomeSmart First Class Realty, owner Ryan Cook, MA): DIFFERENT
#   independent franchisee under the same international brand -- genuine,
#   still uncorrected, no evidence found either way.
#   HSHR (HomeSmart Heritage Realty, owner Jason Araujo, MA) -> SMRT:
#   REVISED (2026-07-25, via detect_magnet_destinations.py): the original
#   "different independent franchisee" call only checked ownership and
#   missed a real consolidation announcement (RISMedia, 2024-01-16) --
#   moved to CONFIRMED_PAIRS above. Caught because HSHR was one of several
#   small origins feeding SMRT's 2024-01-01 magnet-arrival cluster (8
#   arrivals, 6 origins), a shape the pair-only detector structurally
#   can't see since no single origin ever cleared MIN_CLUSTER_MOVERS here.

# The 30 pairs contributed by the magnet and trickle detectors: 1 magnet +
# 29 trickle (17 Tier A identical-name, 12 Tier B/C evidence-backed).
MAGNET_TRICKLE_PAIRS = {
    **_MAGNET_PAIRS,
    **_TRICKLE_IDENTICAL_NAME_PAIRS,
    **_TRICKLE_EVIDENCE_PAIRS,
}

_TRICKLE_PAIRS = {**_TRICKLE_IDENTICAL_NAME_PAIRS, **_TRICKLE_EVIDENCE_PAIRS}

# =====================================================================
# POLICY (2026-08-04, user-directed). The single combined
# COUNT_MAGNET_TRICKLE_AS_CHURN flag is replaced by one flag per
# discovery route, because the two routes are now governed differently:
#
#   MAGNET  -> counted as CHURN     (not suppressed)
#   TRICKLE -> NOT churn            (suppressed)
#   REBRAND -> NOT churn            (suppressed; CONFIRMED_PAIRS, unchanged
#                                    and never governed by these flags)
#
# Rationale for the split: a magnet destination (many origins converging on
# one office in one quarter) is the signature of individual recruitment --
# a signing bonus or a marketing push -- and an agent who takes a better
# offer left voluntarily. A trickle (one pair bleeding 2-4 agents a quarter
# over many quarters) is the founding fold-5 rebrand bug on a slower clock:
# every pair that survived review is either a byte-identical office-code
# renumber or a documented acquisition.
#
# HOW THE MAGNET RULE AND THE REBRAND RULE COEXIST (resolved 2026-08-04):
# geometry never suppresses; evidence does. The magnet detector is a
# DISCOVERY tool -- it points at destinations worth investigating, and the
# investigation, not the shape, decides the label. Its three finds went:
#   SERHANT      -> churn (individual recruitment, brand pull)
#   LVGP/Real    -> churn (revenue-share model, national recruiting)
#   HSHR->SMRT   -> NOT churn, on a documented consolidation announcement
#                   (RISMedia 2024-01-16) -- and therefore now lives in
#                   CONFIRMED_PAIRS, not here.
# So COUNT_MAGNET_AS_CHURN=True has no members to act on, by construction.
# It encodes the rule "a magnet shape alone is never grounds for
# suppression" and guarantees no future magnet find can be suppressed
# without independent evidence being written down first.
# =====================================================================
COUNT_MAGNET_AS_CHURN = True
COUNT_TRICKLE_AS_CHURN = False

_SUPPRESSED_BY_ROUTE = {
    **({} if COUNT_MAGNET_AS_CHURN else _MAGNET_PAIRS),
    **({} if COUNT_TRICKLE_AS_CHURN else _TRICKLE_PAIRS),
}

# Pairs excluded from churn with NO per-quarter/mover-count gate --
# independent evidence already tells us any move on these is a non-event.
CONFIRMED_NOT_CHURN_PAIRS = {**CONFIRMED_PAIRS, **_SUPPRESSED_BY_ROUTE}

# Pairs excluded from churn ONLY for the specific quarter(s) that
# independently clear MIN_CLUSTER_MOVERS -- data.py checks cluster size
# per (origin, dest, quarter) for these before recoding.
CLUSTER_GATED_PAIRS = {**GENUINE_PAIRS, **AMBIGUOUS_UNCONFIRMED_PAIRS}

# Combined, for anything that just needs "is this pair known at all"
# (e.g. detect_transitions.py cross-checks, documentation).
NOT_CHURN_PAIRS = {**CONFIRMED_NOT_CHURN_PAIRS, **CLUSTER_GATED_PAIRS}

# "Has a human already ruled on this pair?" -- what the detectors filter
# against, so they don't re-propose a web search someone already did.
# Deliberately a SUPERSET of NOT_CHURN_PAIRS: it keeps every adjudicated
# pair here even when a route flag has released it back into the target,
# so setting COUNT_MAGNET_AS_CHURN=True does not make
# detect_magnet_destinations.py re-surface HSHR->SMRT as a fresh candidate.
ALREADY_ADJUDICATED_PAIRS = {**NOT_CHURN_PAIRS, **MAGNET_TRICKLE_PAIRS}
