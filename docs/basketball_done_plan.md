# Basketball Sim Done Plan

## Purpose

This document defines the complete path from the current basketball sim state to a finished, useful system.

"Done" does not mean:

- every imaginable play family exists
- every tracking-data proxy is perfect
- the sim is indistinguishable from optical tracking

"Done" means:

1. the possession engine expresses real basketball progression
2. the game engine produces coherent full-game outputs
3. the calibration harness passes the agreed target bands for the supported scope
4. sim-derived features improve the downstream prediction stack
5. the architecture is stable enough to extend without rewriting foundations

This is the implementation roadmap and completion spec.

---

## Current State

Already in place:

- canonical 1-20 attribute model
- mechanic-level resolvers:
  - `resolve_drive_attempt(...)`
  - `resolve_pullup(...)`
  - `resolve_catch_and_shoot(...)`
  - `resolve_rebound(...)`
- team tactics layer
- time-driven game loop
- rotation engine with basic foul trouble handling
- calibration harness with explicit target bands
- first possession progression slice:
  - schema-level `ProgressionState`
  - creation/exploitation split in `HIGH_PNR` and `ISO`
  - initial second-side loop
  - initial swing-pass path

Not done yet:

- offensive rebound 14-second re-entry
- transition / early offense loop
- robust off-ball availability model
- proper second-side calibration
- stable foul environment
- final box-score / feature quality
- downstream feature validation

---

## Definition Of Done

The basketball sim is considered done when all of these are true.

### Engine Completeness

The supported engine scope includes:

- possession entry types:
  - `NORMAL`
  - `OREB`
  - `TIMEOUT`
  - `TRANSITION`
- creation loops:
  - `HIGH_PNR`
  - `ISO`
- continuation loops:
  - `kickout`
  - `swing`
  - `attack_closeout`
  - `reset`
- terminal mechanics:
  - pull-up
  - catch-and-shoot
  - drive attempt
  - free throws
  - rebound
  - turnover

### Calibration Completeness

The supported-scope calibration harness passes its target bands.

### Product Completeness

The sim produces feature sets that improve the downstream prediction stack on held-out data.

### Extension Safety

Adding a new play family or tendency should not require rewriting the routing model.

---

## Guiding Rule

Do not tune around missing loops.

If a calibration failure is caused by missing gameplay topology, build the missing topology first.

Examples:

- too much first-action terminal offense -> build second-side / transition loops
- too much 3PA concentration from a narrow action set -> add the missing pass-chain / off-ball paths
- bad late-clock behavior -> build clock-aware bailout logic, not just shot-quality hacks

Only tune once the relevant loop exists.

---

## Build Order

The work should proceed in this order.

1. progression completion
2. possession entry realism
3. off-ball realism
4. full-game realism
5. calibration pass
6. feature extraction and downstream validation
7. extension layer

---

## Stage 1: Progression Completion

### Goal

Finish the core possession progression loops so the engine no longer behaves like first-action basketball.

### Required Work

1. Make `ProgressionState` the canonical routing object everywhere.
- no parallel ad hoc routing state
- every progression decision reads and writes the same state

2. Finish `HIGH_PNR` and `ISO` as true creation loops.
- creation loop outputs:
  - `PAINT_TOUCH`
  - `FORCED_HELP`
  - `PULL_UP_SPACE`
  - `NONE`
  - terminal foul / turnover

3. Complete `_second_side_loop(...)`.
- decisions:
  - `catch_and_shoot`
  - `attack_closeout`
  - `swing`
  - `reset`
- state-aware:
  - `shot_clock_remaining`
  - `clock_urgency`
  - `swing_count`
  - `last_passer_id`
  - `pass_chain`

4. Implement canonical eligible-receiver selection.
- filter by off-ball state
- use:
  - `shooter_distribution_weights`
  - `corner_spacing_bias`
  - openness proxy

5. Preserve pass chains through the whole possession.
- support:
  - assist
  - hockey-assist logging
  - second-side attribution

### Exit Criteria

- all basketball tests pass
- event logs regularly contain:
  - `PASS` -> `PASS` -> `SHOT`
  - not just one-pass terminal possessions
- no possession-routing code depends on hidden local state outside `ProgressionState`

### Files

- [basketball_sim_schema.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_sim_schema.py)
- [basketball_possession_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_possession_engine.py)
- [tests/test_basketball_possession_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\tests\test_basketball_possession_engine.py)

---

## Stage 2: Possession Entry Realism

### Goal

Make possessions start differently depending on how they begin.

### Required Work

1. Add offensive-rebound 14-second re-entry.
- after offensive rebound:
  - re-enter with `EntryType.OREB`
  - shot clock = 14
- add putback check
- if no putback, flow into compressed halfcourt progression

2. Add transition check.
- inputs:
  - `transition_frequency`
  - `pace_target`
  - possession origin
- outputs:
  - `TRANSITION` loop
  - halfcourt with clock carry-forward

3. Add minimal transition loop.
- actions:
  - rim run
  - pitch-ahead
  - trail three
  - early drag
- if no clean advantage:
  - flow into halfcourt with reduced clock

4. Make play selection clock-aware.
- suppress slow-developing actions when clock is already compressed
- increase bailout frequency late

### Exit Criteria

- possessions no longer all begin in identical halfcourt conditions
- OREB possessions use 14-second logic
- transition possessions exist in the event log
- possession-count / PPP / shot-mix metrics move closer to target without hand-waving

### Files

- [basketball_game_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_game_engine.py)
- [basketball_possession_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_possession_engine.py)
- [basketball_calibration.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_calibration.py)

---

## Stage 3: Off-Ball Realism

### Goal

Stop treating non-primary players as static catch-and-shoot endpoints.

### Required Work

1. Add real off-ball states beyond `spotted_up` / `corner_drift`.
- `relocating`
- `cutting`
- `screening_away`
- `crashing`

2. Add weak-side availability updates.
- determine who is available when the ball moves
- determine who is not

3. Add cut-find gating.
- use:
  - `pass_vision`
  - `pass_accuracy`
  - `decision_making`

4. Add weak-side participation to second-side routing.
- cutters
- corner drift
- relocation

### Exit Criteria

- shot target selection depends on state and location, not only ratings
- corner-vs-wing-vs-cut outcomes become explainable from off-ball state
- event logs show:
  - cut pass attempts
  - weak-side availability-driven passes

### Files

- [basketball_possession_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_possession_engine.py)
- [basketball_sim_schema.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_sim_schema.py)
- [tests/test_basketball_possession_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\tests\test_basketball_possession_engine.py)

---

## Stage 4: Full-Game Realism

### Goal

Make full games behave like NBA games, not just a pile of coherent possessions.

### Required Work

1. Improve foul ecosystem.
- separate:
  - foul volume
  - foul concentration
  - foul type mix
- stabilize:
  - shooting foul rate
  - FTA/FGA
  - archetype foul bands

2. Improve assist environment.
- use pass chains
- credit primary creators and second-side creators correctly
- avoid fake assist inflation

3. Improve turnover environment.
- late-clock turnovers
- pass-chain turnovers
- better separation between live-ball and dead-ball turnovers

4. Improve rebound environment.
- OREB re-entry
- putback logic
- role-aware crash decisions

5. Tighten box-score attribution.
- assists
- hockey assists in logs
- steals
- blocks
- foul types

### Exit Criteria

- player box scores are directionally believable
- team box scores are directionally believable
- event logs tell a coherent basketball story

### Files

- [basketball_game_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_game_engine.py)
- [basketball_possession_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_possession_engine.py)
- [basketball_rotation_engine.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_rotation_engine.py)

---

## Stage 5: Supported-Scope Calibration

### Goal

Get the supported engine scope through the harness honestly.

### Rule

Do not widen or relax bands to hide solvable model failures.

The only acceptable relaxed band is where topology is knowingly incomplete and documented.

### Required Work

1. Recalibrate possession mix.
- targets:
  - PPP
  - turnover rate
  - shooting foul rate
  - OREB rate
  - 3PA share
  - FTA/FGA
  - rim share
  - midrange share
  - corner / above-break split

2. Recalibrate usage and concentration.
- usage share
- assist concentration
- foul concentration
- 3PA concentration within supported-scope band

3. Recalibrate archetype outputs.
- low-end, mid-tier, elite, and historic foul drawers
- role-player vs star shot ownership

4. Recalibrate game variance.
- total variance
- possession variance
- margin realism

### Exit Criteria

- `python audit_basketball_calibration.py --assert-targets` passes for the documented supported scope

### Files

- [basketball_calibration.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_calibration.py)
- [audit_basketball_calibration.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\audit_basketball_calibration.py)
- [tests/test_basketball_calibration.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\tests\test_basketball_calibration.py)

---

## Stage 6: Feature Extraction And Validation

### Goal

Prove the sim adds prediction value.

### Required Work

1. Define the first sim-derived feature set.
- usage concentration mismatch
- bench quality gap
- rebound environment
- pace mismatch
- foul-pressure environment
- closing-lineup edge

2. Build extraction code.
- one file that takes simulated games and returns stable features

3. Run downstream A/B tests.
- baseline stack without sim features
- stack with sim features
- compare held-out performance

4. Cut features that do not add signal.

### Exit Criteria

- at least one sim-derived feature group improves held-out model performance
- improvements are reproducible, not one-off noise

### Files

- new:
  - [basketball_sim_features.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_sim_features.py)
- likely consumers:
  - [dfs_nba.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\dfs_nba.py)
  - [dfs_results.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\dfs_results.py)

---

## Stage 7: Extension Layer

### Goal

Only after the supported scope works, expand tactical coverage.

### Candidate Additions

Play families:

- `HANDOFF`
- `POST_TOUCH`
- `HORNS`
- `DOUBLE_DRAG`
- `STAGGER`
- `SPAIN_PICK_AND_ROLL`

Coverages:

- `HEDGE`
- `BLITZ`
- `ICE`
- `ZONE`
- `SCRAM`

### Rule

Each addition must:

- fit the progression model
- add meaningful calibration value
- not break the supported-scope harness

### Exit Criteria

- new additions are incremental, not foundational rewrites

---

## Hard Gates

These are the explicit gates between stages.

### Gate A: Progression Gate

Before serious tuning resumes:

- `ProgressionState` is canonical
- second-side loop exists
- OREB re-entry and transition entry exist

### Gate B: Calibration Gate

Before feature work:

- harness passes supported-scope target bands

### Gate C: Product Gate

Before declaring the sim done:

- sim-derived features improve downstream performance

---

## What To Build Next

Immediate next implementation steps:

1. offensive rebound 14-second re-entry
2. shot-clock-aware second-side behavior
3. transition check and minimal transition loop
4. stronger eligible-receiver / off-ball state integration

That is the shortest path from the current engine to the next meaningful completion gate.

---

## Final Done State

The project is done when:

1. possessions progress through real basketball loops
2. game outputs are coherent and calibrated for the supported scope
3. sim-derived features improve the real prediction stack
4. the engine can be extended with new play families without structural rewrites

That is the finish line.

