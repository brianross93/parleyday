# DraftKings DFS Extension Spec

## Goal

Extend the current thesis-based sports engine into a DraftKings DFS lineup system.

The goal is not:
- building a generic fantasy optimizer with no slate context
- chasing only median projection
- treating DFS like straight betting with a salary cap

The goal is:
- turn the same slate data, matchup context, and theses into DFS lineup decisions
- generate lineups that can beat human opponents, not just a sportsbook line
- balance projection, ceiling, ownership, leverage, and correlation
- produce contest-aware recommendations instead of one-size-fits-all lineups

This should become a second output product built on the same data stack:
- thesis-driven betting board
- thesis-driven DFS lineup engine

## Why DFS Fits

DFS is a strong extension for this repo because:
- the game is human-vs-human, not only model-vs-book
- late news, ownership lag, and lineup-building mistakes create softer edges
- the existing thesis layer naturally translates into stacks, bring-backs, fades, and leverage
- the current simulator and context pipeline already produce much of the raw information we need

DFS is not easier than betting, but it is a different optimization problem with different edge sources.

## Design Principles

- Reuse the existing data pipelines and thesis architecture
- Do not start over from scratch
- Contest type matters
- Projection is necessary but not sufficient
- Ceiling and leverage matter more in large-field GPPs
- Correlation should be thesis-driven, not generic
- Ownership should be modeled explicitly, not guessed by prose alone
- The LLM should critique lineup theses, not hallucinate player pools

## Product Scope

The first version should support:
- DraftKings NBA Classic
- DraftKings MLB Classic

Later extensions:
- NBA Showdown
- MLB Showdown
- multi-entry portfolio construction
- late swap

## System Overview

The DFS redesign has seven layers:

1. Slate and Rules Ingest
2. Projection Layer
3. Ownership and Field Model
4. Thesis-to-Lineup Translation
5. Lineup Generator
6. Judge and Portfolio Layer
7. Tracking and Learning

Each layer has a distinct job.

## 1. Slate and Rules Ingest

This layer ingests the DraftKings contest slate.

Required inputs:
- player names
- DraftKings player ids
- salaries
- team
- opponent
- roster positions
- game start times
- contest type metadata if available

Required rule support:

NBA Classic:
- salary cap: 50000
- positions: PG, SG, SF, PF, C, G, F, UTIL

MLB Classic:
- salary cap: 50000
- positions: P, P, C, 1B, 2B, 3B, SS, OF, OF, OF

Output:
- normalized DFS player pool
- roster constraints
- slate metadata

### DFS Player Object

Each DFS player should normalize into something like:

```json
{
  "player_id": "dk_12345",
  "name": "Bam Adebayo",
  "sport": "nba",
  "team": "MIA",
  "opponent": "PHI",
  "positions": ["C"],
  "salary": 8300,
  "game": "PHI@MIA",
  "start_time": "2026-03-30T23:00:00Z"
}
```

## 2. Projection Layer

This layer converts current matchup data into DraftKings fantasy-point projections.

The system already has:
- simulated prop probabilities
- player role context
- team context
- injuries and rotation data
- MLB lineup and matchup data

Those need to become DFS outputs:
- median projection
- ceiling projection
- floor projection
- volatility score
- role stability

### NBA Projection Inputs

Use:
- minutes expectation
- points
- rebounds
- assists
- steals/blocks if available later
- usage and rotation compression
- injury-driven role changes

### MLB Projection Inputs

Use:
- lineup slot
- pitcher handedness and quality
- park/weather
- bullpen context
- stolen-base chance later if available
- hit, walk, XBH, HR, run, RBI environment

### Projection Output

Each DFS player should have:
- `median_fpts`
- `ceiling_fpts`
- `floor_fpts`
- `volatility`
- `projection_confidence`

Example:

```json
{
  "player_id": "dk_12345",
  "median_fpts": 40.8,
  "ceiling_fpts": 58.7,
  "floor_fpts": 27.1,
  "volatility": 0.31,
  "projection_confidence": 0.72
}
```

## 3. Ownership and Field Model

This is the main new component that the betting engine does not have.

DFS is not just about raw projection. It is about:
- what the field will play
- what the field will stack
- where humans will overconcentrate

Ownership modeling should estimate:
- player ownership
- stack ownership
- chalk tiers
- likely field constructions

### Ownership Inputs

- salary
- recent game logs
- obvious value created by injury/news
- Vegas totals and game environments
- site-visible popularity signals if available later
- public-facing narratives
- slate size and position scarcity

### Ownership Outputs

Each player:
- `projected_ownership`
- `chalk_flag`
- `leverage_score`

Each lineup:
- `cumulative_ownership`
- `duplication_risk`
- `leverage_score`

This layer is essential for GPP play.

## 4. Thesis-to-Lineup Translation

This is the bridge between the existing engine and DFS.

A thesis should not directly create a bet anymore. It should create lineup construction ideas.

Examples:

NBA:
- `thin_rotation` -> prioritize concentrated usage players, maybe run-back from opposing game environment
- `model_market_divergence` -> overweight underowned player role or fade overpriced chalk
- `rebound_control` -> big-man rebound core, opposing guard scoring fade

MLB:
- `run_environment` -> stack top-of-order bats and correlation pieces
- `bullpen_exhaustion` -> full-stack or mini-stack candidates
- `contact_over_power` -> hit/run producers rather than pure HR hunting

### DFS Thesis Schema

Each thesis should be extended or translated into DFS-native fields:
- `stack_teams`
- `focus_players`
- `fade_players`
- `secondary_players`
- `roster_construction_bias`
- `contest_fit`

Example:

```json
{
  "thesis_id": "mlb_run_environment_min_kc_2026_03_30",
  "sport": "mlb",
  "games": ["MIN@KC"],
  "summary": "Warm weather and wind support a hit-chain environment.",
  "stack_teams": ["MIN", "KC"],
  "focus_players": ["Bobby Witt Jr.", "Byron Buxton", "Austin Martin"],
  "fade_players": [],
  "secondary_players": ["Josh Bell"],
  "roster_construction_bias": "mini_stack_or_full_stack",
  "contest_fit": ["single_entry_gpp", "large_field_gpp"]
}
```

## 5. Lineup Generator

This layer builds legal DraftKings lineups from the player pool and thesis objects.

It should support:
- pure projection builds
- pure leverage builds
- thesis-driven builds
- blended builds

### Generator Constraints

Must obey:
- salary cap
- position rules
- team limits where applicable
- player uniqueness
- optional stack rules

### Generator Modes

- `cash`
  - favor median
  - lower volatility
  - less ownership concern

- `single_entry_gpp`
  - balanced projection and leverage
  - coherent thesis stack
  - moderate ownership discipline

- `large_field_gpp`
  - higher ceiling
  - lower duplication
  - stronger correlation and leverage

### Lineup Output

Each lineup should include:
- player list
- salary used
- median projection
- ceiling projection
- projected ownership
- duplication risk
- lineup thesis
- why it exists

Example:

```json
{
  "lineup_id": "nba_se_001",
  "contest_type": "single_entry_gpp",
  "salary_used": 49800,
  "median_fpts": 312.4,
  "ceiling_fpts": 381.9,
  "projected_ownership": 108.0,
  "duplication_risk": 0.17,
  "thesis_ids": ["nba_thin_rotation_min_dal_2026_03_30"],
  "players": ["..."],
  "summary": "Dallas frontcourt concentration plus selective run-back."
}
```

## 6. Judge and Portfolio Layer

The LLM should not generate raw DFS lineups from scratch.

Its role is to:
- rank lineup bundles
- identify the best expression of each thesis
- point out fragility
- explain why a lineup is viable or over-owned
- separate sharp constructions from generic optimizer chalk

### LLM Role

The LLM may:
- compare lineups
- critique thesis alignment
- identify over-owned or misaligned constructions
- use web search to verify late news when needed

The LLM should not:
- invent player pools
- invent ownership
- ignore roster rules
- choose players not already generated in candidate lineups

## 7. Tracking and Learning

DFS should become a learning system over time.

Track:
- actual ownership vs projected ownership
- lineup percentile outcomes
- thesis success rates
- stack success rates
- duplication surprises
- late news misses

This is how the system improves beyond static optimization.

## Contest Framework

The product should support at least three recommendation surfaces:

- `Best Cash Lineup`
- `Best Single-Entry GPP Lineup`
- `Best Large-Field GPP Lineup`

Optional later:
- `Best Contrarian Stack`
- `Best Leverage Core`
- `Late Swap Targets`

## Hard Exclusions

Allowed hard exclusions:
- injured / ruled-out players
- ineligible position configurations
- invalid salary-cap lineups
- obvious stale or broken player mappings
- missing DraftKings player identifiers

These are data-integrity rules, not style preferences.

## Soft Penalties

Soft penalties may include:
- uncertain minutes
- stale news
- projected chalk concentration
- poor thesis alignment
- low ceiling for GPPs
- high duplication risk

## UI Design

The DFS surface should not look like the current debug-heavy board.

The default UI should show:
- slate overview
- contest mode selector
- top lineups
- thesis-aligned core plays
- fades
- leverage pivots

Debug should be hidden behind disclosure panels:
- raw projections
- ownership model
- stack rules
- thesis details

## Implementation Plan

### Phase 1

- Add DFS spec and schema
- Define DraftKings player pool format
- Add salary and position ingest
- Build lineup rule models

### Phase 2

- Convert current player outputs into DFS fantasy-point projections
- Add basic lineup generator
- Output legal projection-first lineups

### Phase 3

- Add thesis-to-lineup translation
- Add stack logic and correlation rules
- Add contest-type lineup variants

### Phase 4

- Add ownership model
- Add leverage and duplication scoring
- Add LLM critique of lineup bundles

### Phase 5

- Add late-news verification and late swap
- Add lineup tracking and learning loop

## Initial Build Priorities

The highest-value first steps are:

1. DraftKings slate ingest
2. DFS projection object
3. Lineup legality engine
4. NBA and MLB classic lineup generation
5. Thesis-to-lineup translation

That order preserves momentum and reuses what already works in the repo.

## Non-Goals For V1

- full optimizer exposure control
- multi-hundred lineup portfolio builder
- automatic contest-entry upload
- showdown support
- full ownership simulation of the field

Those can come later.

## Summary

This extension should treat DFS as:
- a lineup-construction problem
- a thesis-expression problem
- a field-behavior problem

It should not be a generic fantasy optimizer.

The edge comes from:
- better context
- better theses
- better projection and role interpretation
- better ownership and leverage modeling
- better discipline against human field mistakes
