# Basketball Sim Project Spec

## Vision

Build a basketball simulation system that is tactically rich, statistically disciplined, and useful for real prediction work.

The long-term aspiration is an FM-style basketball simulation engine with strong tactical depth. The near-term product goal is more pragmatic:

- simulate possessions and games coherently
- extract high-value game-state and lineup interaction features
- feed those features into the existing DFS and market modeling stack
- validate relentlessly against real NBA outcomes

This project should not become a fake realism engine. "Looks like basketball" is not enough. It has to provide signal.

## Product Goals

1. Model basketball at the possession level.
2. Represent team tactics and lineup interactions explicitly.
3. Produce explainable event logs and box-score outputs.
4. Improve downstream prediction quality for DFS and betting.
5. Support simple 2D viewing later without requiring graphics complexity now.

## Non-Goals

1. Full physics-body simulation.
2. 3D rendering or animation-heavy gameplay.
3. Building an entire basketball game frontend before the engine works.
4. Hand-authoring giant playbooks with arbitrary hardcoded outcome numbers.
5. Replacing the existing prediction stack immediately.

## Why This Matters

The current projection stack can model player outputs, matchup features, and DFS lineup optimization, but it still abstracts away a lot of how games actually unfold:

- possession sequencing
- lineup interactions
- tactical changes
- bench drop-off
- late-game behavior

A possession simulator can surface features the current stack does not express directly, such as:

- projected pace mismatch
- creation-pressure mismatch
- bench minutes quality gap
- closing-lineup edge
- rebound environment by lineup archetype
- expected shot-type mix under specific coverage interactions

Those features can improve:

- DFS projections
- game totals
- win probability
- prop context

## Rating Scale

The canonical simulation attribute scale is `1-20`, not `0-100`.

Reason:

- it is easier to reason about
- it matches the Football Manager style of tactical interpretation
- it avoids fake precision early

Internally, we can still derive traits from continuous percentiles and compress them into `1-20` buckets.

Detailed attribute definitions and mechanic wiring live in:

- [basketball_attribute_mechanics.md](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\docs\basketball_attribute_mechanics.md)

## Project Strategy

### Phase 1 Strategy

Use the sim as a feature generator first.

That means:

- simulate possessions and games
- extract summary features
- feed those features into the existing Kalshi / DFS / prediction stack
- measure whether those features improve out-of-sample performance

Only after that should the sim be trusted as a direct standalone forecasting engine.

### Why This Is The Right Path

It lowers risk.

Instead of requiring:
- perfect full-game calibration immediately

we only require:
- useful, stable, measurable sim-derived features

That lets the project add value early.

## System Components

### Schema

File:
- [basketball_sim_schema.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_sim_schema.py)

Purpose:
- define the contracts for players, tactics, lineups, possession state, and outputs

### Possession Engine

Proposed file:
- `basketball_possession_engine.py`

Purpose:
- resolve one possession into an event chain and outcome

### Rotation Engine

Proposed file:
- `basketball_rotation_engine.py`

Purpose:
- manage substitutions, fatigue, foul pressure, and closing groups

### Tactics Engine

Proposed file:
- `basketball_tactics_engine.py`

Purpose:
- choose play families and defensive coverages probabilistically from team tendencies

### Game Engine

Proposed file:
- `basketball_game_engine.py`

Purpose:
- simulate a full game possession by possession

### Feature Extraction

Proposed file:
- `basketball_sim_features.py`

Purpose:
- turn simulated games into prediction features for the existing pipeline

### Calibration

Proposed file:
- `basketball_calibration.py`

Purpose:
- compare simulated outputs to real data and tune the system

## Calibration Objectives

This must be explicit.

### Primary Objectives

1. Win probability calibration
- objective:
  - log-loss / Brier score on game winner probability

2. Game total distribution calibration
- objective:
  - CRPS or distributional log-loss if we model full score distributions
  - RMSE / MAE on totals only as secondary diagnostics

3. Margin and total variance calibration
- objective:
  - simulated spread and total variance should match historical outcome dispersion

### Secondary Objectives

1. Team score RMSE
2. Possessions per game error
3. Player minutes error
4. Player box-score RMSE
5. Shot-type distribution error

### Important Rule

A sim that matches average points but misses variance is not good enough for:

- totals
- parlays
- tail-event forecasting

So distributional calibration matters, not just means.

## Data Philosophy

### Hardcode

These are acceptable to hardcode:

- play families
- coverage families
- phase/state machine
- basic rotation grammar
- court zones

### Derive

These should be derived or calibrated:

- player trait weights
- team tactical tendencies
- shot quality adjustments
- turnover/foul/rebound probabilities
- lineup interaction effects

### Caution

Traits like:

- switchability
- screen value
- short-roll playmaking

are worth modeling, but they do not have perfect public data pipelines. Early versions should use simpler and more defensible proxies.

## Initial Scope

The initial engine slice should only support:

- play families:
  - `HIGH_PICK_AND_ROLL`
  - `ISO`
- coverages:
  - `DROP`
  - `SWITCH`
- outcomes:
  - shot
  - turnover
  - foul
  - rebound

No `HORNS`, `SPAIN_PNR`, or richer tactical trees should be added until this minimal slice is validated.

## Milestones

### Milestone 1

Deliver:
- possession engine skeleton
- event chain output
- basic sanity tests

### Milestone 2

Deliver:
- full-game loop
- team score / possession outputs
- team-level calibration checks

### Milestone 3

Deliver:
- rotation engine
- player minutes and box-score outputs
- player-level calibration checks

### Milestone 4

Deliver:
- sim feature extraction into the current model stack
- feature backtests

### Milestone 5

Deliver:
- deeper tactics
- better lineup interactions
- stronger calibration

### Milestone 6

Deliver:
- simple 2D viewer for debugging and presentation

## Risks

1. Overfitting realism
- building a complex sim that looks plausible but adds no predictive value

2. Magic-number creep
- adding too many hand-tuned coefficients without calibration discipline

3. Data quality gaps
- trying to infer advanced tactical traits from weak public proxies

4. Scope explosion
- adding viewer or FM-style layers before the core engine is trustworthy

## Immediate Next Step

Build the possession engine first, and only for:

- `HIGH_PICK_AND_ROLL`
- `ISO`
- `DROP`
- `SWITCH`

Then validate:

- shot mix
- turnover rate
- foul rate
- offensive rebound rate
- points per possession

If that slice is not sane, do not broaden the playbook yet.
