# Basketball Sim Overview

## Goal

Build a basketball simulation engine that is:

- tactically rich enough to model meaningful team and lineup differences
- simple enough to calibrate against real NBA outcomes
- useful as a predictive substrate for DFS, projection work, and game analysis
- explainable, so we can audit why the engine produced a given outcome

The first practical role of the sim is not "replace the whole prediction stack." The first practical role is:

- generate richer basketball features for the existing prediction pipeline
- improve DFS/game-state signal
- eventually earn the right to be used as a direct forecasting component

This is not a graphics-first project. The core deliverable is a possession-level simulation model. Any 2D viewer is a debug and visualization layer on top of the engine.

## Product Principles

1. Event-driven first
- Start with possession state, play calls, coverage responses, and event outcomes.
- Do not begin with continuous movement or visual realism.

2. Basketball grammar can be hardcoded
- Play families, coverages, and rotation concepts are legitimate hardcoded structures.
- Outcome weights and team tendencies should be learned or calibrated when possible.

3. Calibration is mandatory
- A more complex sim is only better if it improves out-of-sample realism or predictive value.
- Every major layer must be validated against real games.
- Matching mean stat lines is not enough. Distributional accuracy matters.

4. Rendering is secondary
- Early Football Manager-style circles are enough.
- The engine must stand on its own without visual polish.

## High-Level Architecture

### Schema Layer

File:
- [basketball_sim_schema.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_sim_schema.py)

Responsibilities:
- shared contracts for players, traits, tactics, possession state, event logs, and outputs

### Possession Engine

Proposed file:
- `basketball_possession_engine.py`

Responsibilities:
- choose offensive action path within a play family
- choose defensive response/coverage
- resolve the possession into an event chain and final outcome

### Rotation Engine

Proposed file:
- `basketball_rotation_engine.py`

Responsibilities:
- manage substitutions, target minutes, stagger patterns, fatigue, foul trouble, and closing lineups

### Tactics Engine

Proposed file:
- `basketball_tactics_engine.py`

Responsibilities:
- select play families and defensive coverages based on:
  - team tendencies
  - lineup context
  - clock/score state
  - opponent traits

### Game Engine

Proposed file:
- `basketball_game_engine.py`

Responsibilities:
- simulate full games possession by possession
- maintain clock, score, fouls, lineup state, fatigue, and event logs

### Calibration Layer

Proposed file:
- `basketball_calibration.py`

Responsibilities:
- compare simulation output against real NBA data
- tune rates, scales, and trait effects
- reject features that do not improve predictive performance

Primary objective:
- use the sim as a feature generator for the deployed prediction stack first

Secondary objective:
- develop toward a standalone calibrated game simulator

### Viewer

Proposed file:
- `basketball_viewer.py`

Responsibilities:
- optional later
- simple 2D representation of the engine state
- event playback and debugging

## Development Phases

### Phase 1: Possession Prototype

Deliverables:
- single-possession resolver
- initial play families:
  - transition push
  - high pick and roll
  - handoff
  - iso
  - reset
- initial coverages:
  - man
  - switch
  - drop

Validation targets:
- possession outcome frequencies
- turnover/foul/shot mix sanity
- basic shot profile realism

### Phase 2: Full-Game Skeleton

Deliverables:
- full game loop
- clock and possession management
- simple score state handling
- team box score aggregation

Validation targets:
- possessions per game
- team points
- basic variance distribution

### Phase 3: Rotation and Fatigue

Deliverables:
- substitution engine
- target minute allocation
- fatigue and foul pressure
- end-of-quarter / closing group logic

Validation targets:
- player minutes
- bench share
- late-game lineup behavior

### Phase 4: Tactical Depth

Deliverables:
- more play families
- more defensive coverages
- tactic-driven selection logic
- lineup interaction effects

Validation targets:
- shot distribution by team archetype
- assist rates
- rim pressure / three-point rate differences by style

### Phase 5: Calibration and Backtesting

Deliverables:
- parameter fitting
- historical backtests
- error reports by team, lineup type, and player archetype
- feature extraction path into the current market/DFS models

Validation targets:
- prediction error on:
  - team score
  - player minutes
  - points
  - rebounds
  - assists
- win probability calibration
- total score distribution calibration
- variance and tail-behavior calibration

### Phase 6: Viewer

Deliverables:
- simple 2D possession playback
- event timeline
- lineup and tactic inspection

## Success Criteria

The project is successful when:

- simulated games look coherent in event logs
- team-level outputs are statistically close to real NBA distributions
- player-level minute and stat outputs are directionally reliable
- the engine produces useful signal for DFS and game-level forecasting
- we can explain why a simulated game unfolded the way it did
- sim-derived features measurably improve the existing model stack

## Immediate Next Step

Implement `basketball_possession_engine.py` for a minimal first slice:

- play families:
  - `HIGH_PICK_AND_ROLL`
  - `ISO`
- coverages:
  - `DROP`
  - `SWITCH`
- outcomes:
  - shot attempt
  - turnover
  - foul
  - rebound

That is the smallest meaningful engine slice.
