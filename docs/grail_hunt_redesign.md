# Grail Hunt Redesign

## Goal

Build a system that finds a small set of longshot parlays that are plausibly underpriced and worth taking a swing on.

The goal is not:
- maximizing hit rate on low-payout tickets
- maximizing average leg edge in isolation
- letting an LLM freestyle longshot ideas from a raw slate dump

The goal is:
- generate candidate theses
- express those theses as parlays
- remove junk and broken-feed artifacts
- let an LLM critique and rank the best candidates

This system should optimize for asymmetric upside with believable paths, not conservative betting behavior.

## Design Principles

- Deterministic first, intuition second
- Structured signals before freeform narrative
- LLM as hypothesis/ranking layer, not sole signal source
- Verification upgrades confidence; missing verification lowers confidence
- Data hygiene is required
- "Interesting" is not enough; a ticket needs a coherent thesis

## System Overview

The redesign has five layers:

1. Structured Thesis Lane
2. Intuition Thesis Lane
3. Verification Lane
4. Candidate Builder
5. Judge and Board

Each layer has a distinct job.

## 1. Structured Thesis Lane

This lane produces theses from deterministic detectors and model-backed triggers.

Examples:
- NBA thin rotation / usage concentration
- NBA injury lag vs market pricing
- MLB bullpen exhaustion
- MLB run-environment spike
- MLB starter downgrade or late probable-pitcher change
- handedness/platoon mismatch
- sim-vs-market divergence
- feed integrity anomaly

This lane should be reproducible and testable.

### Structured Thesis Requirements

Each thesis must include:
- `thesis_id`
- `source = structured`
- `type`
- `sport`
- `games`
- `summary`
- `confidence`
- `supporting_facts`
- `missing_information`
- `verification_targets`
- `kill_conditions`
- `candidate_leg_types`

### Example

```json
{
  "thesis_id": "mlb_bullpen_exhaustion_pit_cin_2026_03_30",
  "source": "structured",
  "type": "bullpen_exhaustion",
  "sport": "mlb",
  "games": ["PIT@CIN"],
  "summary": "Bullpen fatigue plus run environment increases late scoring risk.",
  "confidence": 0.64,
  "supporting_facts": [
    "Both bullpens show elevated recent workload",
    "Park supports offense",
    "Weather supports carry"
  ],
  "missing_information": [
    "Confirmed lineups"
  ],
  "verification_targets": [
    "official lineups",
    "probable pitchers"
  ],
  "kill_conditions": [
    "pitching change",
    "multiple regular bats absent"
  ],
  "candidate_leg_types": ["total", "team_total", "hits", "hr"]
}
```

## 2. Intuition Thesis Lane

This lane allows LLM-driven intuition, but in a bounded format.

The purpose is not to replace structured detectors. It is to:
- notice unusual patterns
- propose scenario narratives
- surface "this feels off" hypotheses
- prioritize areas for verification

### Why This Lane Exists

Pure detectors miss pattern combinations that feel meaningful but are hard to encode quickly.

Examples:
- "This favorite feels overpriced because the market is still treating it as a healthy normal roster."
- "This prop cluster looks stale rather than efficiently repriced."
- "This underdog game script feels more live than the current board implies."

### Constraints

The LLM must not generate vague intuition only. Every intuition thesis must be structured and include:
- explicit claim
- supporting facts taken only from supplied payload
- missing facts
- verification targets
- confidence
- kill conditions

### Intuition Thesis Requirements

Each thesis must include:
- `thesis_id`
- `source = intuition`
- `summary`
- `games`
- `supporting_facts`
- `missing_information`
- `verification_targets`
- `confidence`
- `kill_conditions`

### Example

```json
{
  "thesis_id": "intuition_det_okc_underdog_live_2026_03_30",
  "source": "intuition",
  "type": "underdog_environment",
  "sport": "nba",
  "games": ["DET@OKC"],
  "summary": "Detroit feels more live than the board suggests because the favorite still looks priced like a normal healthy favorite.",
  "supporting_facts": [
    "Sim narrows the game more than market",
    "Roster context is noisy and incomplete",
    "Favorite price remains very strong"
  ],
  "missing_information": [
    "Official NBA injury report",
    "Confirmed active list"
  ],
  "verification_targets": [
    "official injury report",
    "beat-reporter rotation notes"
  ],
  "confidence": 0.41,
  "kill_conditions": [
    "late injury report confirms full-strength favorite",
    "underdog missing key usage piece"
  ]
}
```

## 3. Verification Lane

This lane exists to upgrade or downgrade theses when key facts are missing or stale.

Verification should use:
- official league/team sources first
- reliable current reporting second
- structured source capture

Verification is not a separate recommender. It only:
- confirms thesis facts
- identifies contradictions
- updates confidence
- adds source links

### Verification Outcomes

Each thesis should end in one of:
- `verified`
- `partially_verified`
- `unverified`
- `contradicted`

### Confidence Handling

- Verified: confidence can increase
- Partially verified: confidence capped
- Unverified: confidence suppressed
- Contradicted: thesis removed or marked dead

## 4. Candidate Builder

The candidate builder should create parlays as expressions of theses.

It should not try to act like a final judge.

### Builder Responsibilities

- generate candidate tickets per thesis
- allow mixed leg types
- allow same-game combinations when thesis supports them
- avoid only hard contradictions
- score candidates as thesis expressions

### Candidate Inputs

- thesis object
- recognized legs
- model probabilities
- market probabilities
- executable entry cost
- compatibility map
- data-quality flags

### Candidate Outputs

Each candidate should include:
- `candidate_id`
- `thesis_id`
- `legs`
- `payout_estimate`
- `model_joint_prob`
- `market_joint_prob`
- `expression_score`
- `correlation_flags`
- `data_quality_flags`
- `kill_conditions`

### Hard Exclusions

Hard exclusions are allowed for:
- impossible or contradictory leg combinations
- broken market mappings
- obviously bad feed values
- duplicate legs

These are data hygiene rules, not betting-style paternalism.

### Soft Penalties

Soft penalties can include:
- stale lineup state
- fallback pricing
- unresolved starter conflicts
- weak role certainty

The builder should still generate the candidate; the judge decides how much to trust it.

## 5. Judge and Board

The judge layer should use the LLM to:
- critique theses
- compare candidate expressions
- identify fragility
- rank the final board

The LLM should not originate the raw candidate pool by itself.

### LLM Responsibilities

- summarize the thesis
- decide whether the thesis is coherent
- explain what could kill it
- pick the strongest candidate expression
- identify feed/pathology concerns
- say when no candidate should survive

### LLM Output Schema

For each thesis:
- `verdict`: `back`, `lean`, `watch`, `pass`
- `confidence`
- `best_candidate_id`
- `why`
- `kill_conditions`
- `source_notes`

## Final Dashboard Board

The final board should show thesis-driven outputs, not generic parlays.

Suggested cards:
- `Best Thesis`
- `Best 50x Thesis`
- `Best 100x Thesis`
- `Wildcard 250x+`
- `Fragile But Interesting`
- `Discarded Thesis`

Each card should show:
- thesis title
- source type: structured / intuition / both
- summary
- best candidate expression
- payout estimate
- model joint probability
- verification status
- kill conditions
- LLM verdict

## Objective Function

The system should not optimize a single naive scalar.

The practical objective is:

- find candidates above the payout target
- maximize believable path to hitting
- favor model disagreement when present
- prefer verified theses over unverified ones
- penalize broken or stale data

This is closer to:
- thesis quality
- candidate expression quality
- verification status
- model plausibility

than to raw average edge.

## Phase Plan

### Phase 1

Add thesis infrastructure.

- define thesis schema
- create structured thesis lane
- emit thesis objects in oracle output
- show theses in dashboard

### Phase 2

Add intuition lane.

- LLM prompt for structured intuition theses
- merge intuition theses with structured theses
- preserve provenance

### Phase 3

Add verification lane.

- source-backed verification for missing/stale facts
- confidence updates
- contradiction handling

### Phase 4

Replace current parlay recommender with thesis-based candidate builder.

- generate candidates per thesis
- attach data-quality flags
- attach payout-target buckets

### Phase 5

Move LLM judgment to thesis-bundle evaluation.

- rank theses
- choose best candidate expression
- explain kill conditions

## Prompting Guidance

The LLM should receive:
- structured theses
- candidate tickets
- data-quality flags
- verification status
- missing information

The LLM should not receive an instruction like:
- "look at this slate and find the best longshot"

It should instead receive:
- "evaluate these thesis objects and candidate expressions; identify which are coherent, fragile, or dead"

## Non-Goals

This redesign is not trying to:
- create a watchable sports simulation
- maximize low-payout hit rate
- let the LLM freestyle parlay ideas from scratch
- replace structured modeling with vibes

## Summary

The grail-hunt system should treat intuition as valuable but bounded.

The right balance is:
- structured signals provide discipline
- intuition provides hypothesis generation
- verification upgrades confidence
- the builder creates candidate expressions
- the LLM judges and explains

That gives the system room for insight without letting it drift into noise.
