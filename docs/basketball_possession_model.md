# Basketball Possession Model

## Objective

Define how one offensive possession is represented and resolved.

The possession model should answer:

- what play did the offense try to run?
- how did the defense respond?
- who touched the ball?
- what shot or turnover event occurred?
- what rebound or dead-ball state followed?

## Possession State Machine

Each possession moves through these phases:

1. `ADVANCE`
- ball enters frontcourt
- transition opportunity may be available

2. `INITIATION`
- offense sets alignment
- offense selects play family
- defense selects primary coverage

3. `PRIMARY_ACTION`
- main play action occurs:
  - pick and roll
  - handoff
  - post touch
  - isolation
  - off-ball screening action

4. `SECONDARY_ACTION`
- first action did not directly end the possession
- offense may flow into:
  - swing-swing shot
  - drive and kick
  - re-screen
  - reset iso
  - late-clock bailout

5. `SHOT`
- shot type and shot quality are resolved

6. `REBOUND`
- if miss:
  - defensive rebound
  - offensive rebound
  - loose ball foul
  - dead ball

7. `DEAD_BALL`
- foul shots
- violation
- turnover inbound
- timeout or substitution opportunities

## Core Inputs

Each possession consumes:

- offense lineup
- defense lineup
- player traits
- player condition
- team tactics
- game clock state
- score state
- floor spacing state
- defensive assignments

These are represented in:
- [basketball_sim_schema.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_sim_schema.py)

## Play Family Design

### Initial Play Families

The first engine version should support:

- `HIGH_PICK_AND_ROLL`
- `ISO`
- `RESET`

Then expand to:

- `TRANSITION_PUSH`
- `HANDOFF`
- `HORNS`
- `DOUBLE_DRAG`
- `POST_TOUCH`
- `STAGGER`
- `SPAIN_PICK_AND_ROLL`

### What a Play Family Does

A play family defines:

- default actor roles
- likely floor zones
- likely pass sequence
- likely defender involvement
- likely shot profile
- likely turnover/foul pattern

Example: `HIGH_PICK_AND_ROLL`

- primary actor:
  - lead guard or primary creator
- screener:
  - roll big or pop big
- likely defensive coverages:
  - drop
  - switch
  - blitz
- likely outcomes:
  - pull-up
  - pocket pass
  - rim finish
  - weak-side kickout
  - live-ball turnover

## Defensive Response Model

### Initial Defensive Coverages

Start with:

- `MAN`
- `DROP`
- `SWITCH`

Then add:

- `HEDGE`
- `BLITZ`
- `ICE`
- `ZONE`
- `SCRAM`

### Coverage Responsibilities

Each coverage changes:

- on-ball pressure
- help timing
- rim access
- passing windows
- recovery probability
- mismatch risk

Example: `DROP`

- suppresses direct rim access less than switch
- encourages pull-up and pocket-pass outcomes
- preserves rim protection
- reduces switch mismatch creation

Example: `SWITCH`

- reduces clean pull-up windows
- increases mismatch creation
- changes rebound matchups
- may flatten the offense into isolation

## Event Resolution Order

Each possession resolves in this order:

1. select play family
2. select primary ball handler and action participants
3. select defensive coverage response
4. compute action advantage
5. choose branch:
   - shot
   - pass continuation
   - turnover
   - foul
   - reset
6. if shot:
   - determine shot type
   - determine shot quality
   - determine make/miss
7. if miss:
   - determine rebound result
8. emit event log

## Action Advantage

The key engine quantity is action advantage.

This should be computed from:

- ball handler creation trait
- screener value
- spacing score
- defender containment
- help defender quality
- rim protection
- tactic fit
- fatigue
- clock pressure

The result should be a bounded value, not an unbounded hand-tuned score.

Example usage:

- positive advantage:
  - increases high-quality shot branches
  - increases assist completion
  - decreases turnover risk
- negative advantage:
  - forces reset
  - increases late-clock bailout
  - increases poor shot and turnover rates

## Shot Quality Model

Shot quality should be derived from:

- shot type
- location
- defender proximity proxy
- play success
- help rotation timing
- passer quality
- shooter skill
- fatigue

The engine does not need real geometric collision math initially.
It only needs enough state to distinguish:

- open corner three
- contested pull-up three
- clean rim attempt
- tough paint floater
- late-clock midrange bailout

## Rebound Model

On misses, rebound outcome should depend on:

- shot location
- lineup rebounding scores
- crash-glass tactic
- defensive box-out quality
- floor balance

Do not treat rebounds as pure random team-level coin flips.

## First Engine Contract

The first possession engine function should look conceptually like:

```python
def simulate_possession(context: PossessionContext, rng) -> PossessionOutcome:
    ...
```

It should:

- consume the full possession context
- return:
  - points scored
  - turnover/foul flags
  - player attribution
  - detailed event log

## Validation Targets

For the first possession engine, test:

- shot type distribution
- turnover rate
- foul rate
- offensive rebound rate
- points per possession by lineup/tactic archetype

If those are not directionally sane, do not add more tactical complexity yet.
