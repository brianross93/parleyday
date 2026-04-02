# Basketball Data Model

## Objective

Define what data the simulation should consume and what inferred traits it should produce.

The goal is to derive as much as possible from measurable basketball data while hardcoding only the structural grammar of the sport.

## Data Layers

### Layer 1: Observable Inputs

These are directly measurable:

- minutes
- points
- rebounds
- assists
- turnovers
- fouls
- field goal attempts
- three-point attempts
- free throw attempts
- offensive rebounds
- defensive rebounds
- starting status
- on/off lineup combinations

### Layer 2: Derived Player Traits

These are inferred from observable data and stabilized over rolling windows.

Canonical scale:

- all player attributes should be stored and discussed on a `1-20` scale
- `10` is role-adjusted league average
- `15` is strong
- `18` is elite
- `20` is true outlier territory

Examples:

- usage
- touch share
- rim pressure
- pull-up shooting
- catch-and-shoot strength
- finishing
- passing creation
- screen value
- short-roll playmaking
- post scoring
- offensive rebounding
- defensive rebounding
- perimeter defense
- interior defense
- rim protection
- switchability
- steal pressure
- foul tendency
- turnover tendency
- transition speed
- stamina

### Layer 3: Team Tactical Tendencies

These are inferred from team behavior over time.

Examples:

- pace target
- transition frequency
- crash-glass rate
- switch rate
- drop rate
- zone rate
- no-middle tendency
- late-clock isolation rate
- early-offense rate
- pick-and-roll frequency
- handoff frequency
- post-touch frequency
- off-ball screen frequency

### Layer 4: Lineup Interaction Effects

These are derived from 5-man unit combinations.

Examples:

- spacing score
- creation redundancy
- rim pressure score
- rebounding score
- switchability score
- rim protection score

## Trait Philosophy

Traits should not be pure vibes.

They should come from:

- directly measured rates
- rolling averages
- lineup context
- regularized or capped transforms

If a trait cannot be explained by data inputs and validated later, it should not exist.

Detailed mechanic wiring lives in:

- [basketball_attribute_mechanics.md](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\docs\basketball_attribute_mechanics.md)

## Example Trait Mapping

### Usage

Derived from:

- shot share
- assist chance creation
- turnover involvement
- touch concentration

### Rim Pressure

Derived from:

- rim attempt share
- paint touch proxy
- foul draw tendency
- transition rim frequency

### Passing Creation

Derived from:

- assist rate
- secondary assist proxies if available later
- turnover-adjusted creation value

### Rim Protection

Derived from:

- block rate
- opponent rim suppression proxies
- defensive rebounding support
- foul tradeoff

### Switchability

Derived from:

- size/position flexibility proxies
- foul stability
- perimeter and interior defensive competence blend

## Initial Realistic Data Plan

For version 1, use simpler inferred traits that can be built from current data:

- minutes-based role strength
- points / rebounds / assists production
- usage-like offensive share
- rebounding strength
- basic shooting role classification
- basic defender classification

This is enough to support a first possession engine.

## Team Tactics Representation

Team tactics should be weights, not deterministic scripts.

Example:

- `pick_and_roll_rate = 0.34`
- `handoff_rate = 0.11`
- `post_touch_rate = 0.08`
- `switch_rate = 0.27`
- `drop_rate = 0.43`

The engine then chooses actions probabilistically from those weights.

## Rotation Data

Rotation planning should use:

- starter status
- recent minute distribution
- closing lineup frequency
- foul sensitivity
- bench replacement hierarchy

Even if those start as heuristics, they should be explicit inputs to the rotation engine.

## Calibration Targets

The sim must be measured against:

### Team-Level

- points scored
- points allowed
- possessions
- offensive rating
- shot mix
- turnovers
- offensive rebound rate

### Player-Level

- minutes
- points
- rebounds
- assists
- usage concentration
- variance by role/archetype

### Lineup-Level

- bench drop-off
- stagger impact
- closing lineup performance
- style-driven shot distribution

## Validation Process

Every new derived trait or tactic effect should go through:

1. feature definition
2. implementation
3. backtest against historical games
4. keep / tune / cut decision

No feature should survive because it "sounds like basketball."

## Current Schema Mapping

These concepts currently map to:

- `PlayerTraitProfile`
- `PlayerCondition`
- `PlayerSimProfile`
- `TeamTactics`
- `LineupUnit`
- `RotationPlan`

in [basketball_sim_schema.py](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\basketball_sim_schema.py)

## Next Design Step

Translate this data model into:

- a small builder that can create `PlayerSimProfile`
- a minimal `TeamTactics` profile generator
- a possession engine that consumes those objects

That should be done before any viewer work.
