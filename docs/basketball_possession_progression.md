# Basketball Possession Progression Spec

## Purpose

This document defines how a possession unfolds before it reaches a terminal mechanic.

The attribute and mechanics model answers:

- what each player can do
- how a shot, drive, foul, or rebound resolves once chosen

This document answers:

- how the offense gets from initiation to terminal outcome
- when the ball moves vs. when the possession ends
- when the defense rotates, stays home, or switches
- when advantage is exploited immediately vs. moved to the second side
- where team tendencies influence progression

This is the gameplay-loop layer of the sim. Without it, the engine becomes a collection of isolated mechanic calls instead of a basketball possession model.

---

## Design Goal

Real NBA possessions are usually not:

1. primary creator acts
2. possession immediately ends

They are more often:

1. offense initiates action
2. first advantage is created or denied
3. defense chooses how to respond
4. offense exploits, moves, or resets the advantage
5. possession either ends or flows into a second-side action

The progression model exists to represent that sequence.

---

## Core Principle

Mechanics are terminal or near-terminal.

Progression decides:

- which mechanic gets called
- in what order
- after what preceding events
- with which actors

The mechanic resolver should stay pure:

- inputs: players, context, traits, coverage, shot type
- outputs: quality, branch result, make/miss/foul/turnover/rebound

The progression layer should handle:

- possession flow
- continuation logic
- pass-chain generation
- second-side movement
- team-specific play tendencies

---

## Possession Loop Model

Every possession moves through up to five loops:

1. entry loop
2. primary advantage loop
3. defensive reaction loop
4. exploitation / continuation loop
5. terminal loop

Not every possession uses every loop, but every possession should fit this shape.

---

## Loop 1: Entry

The offense enters the possession in one of two broad states:

- `transition_or_early`
- `set_halfcourt`

### Transition Or Early

This is not yet a full transition engine. It is a possession entry state that allows:

- rim run
- pitch-ahead
- trail three
- early drag
- flow into halfcourt if nothing materializes

**Controlled by team tendencies:**

- `pace_target`
- `transition_frequency`
- `early_offense_rate`

**Typical outputs:**

- immediate rim attempt
- early drag PnR
- wing catch-and-shoot three
- no advantage -> flow into set halfcourt

### Set Halfcourt

The offense enters its chosen play family and spacing shell.

**Controlled by team tendencies:**

- `play_family_weights`
- `star_usage_bias`
- `corner_spacing_bias`

**Typical outputs:**

- play family selected
- primary actor selected
- screener / secondary actor selected
- weak-side spacing map established

---

## Loop 2: Primary Advantage

This is the first real attempt to bend the defense.

Examples:

- `HIGH_PNR`
- `ISO`
- later: `HANDOFF`, `DOUBLE_DRAG`, `STAGGER`, `POST_TOUCH`, `SPAIN_PICK_AND_ROLL`

The purpose of this loop is not necessarily to end the possession. Its purpose is to answer:

- was an advantage created?
- where was it created?
- who forced the defense to react?

### Primary Advantage States

The primary action should resolve into one of these abstract states:

- `no_advantage`
- `perimeter_advantage`
- `paint_touch`
- `switch_mismatch`
- `forced_help`
- `direct_terminal`

### State Definitions

#### `no_advantage`

The action failed to bend the defense.

Typical reasons:

- defender stayed attached
- screen nav neutralized the screen
- handler did not gain leverage
- clock pressure remains

Typical next step:

- reset
- second-side swing
- late-clock bailout

#### `perimeter_advantage`

The handler or receiver has a live closeout or partial step.

Typical next step:

- pull-up
- attack closeout
- swing to the next perimeter player

#### `paint_touch`

The ball or player reaches the paint.

This is a key state because paint touches force real defensive decisions.

Typical next step:

- rim attempt
- dump-off
- kickout
- corner drift pass
- foul drawn

#### `switch_mismatch`

The coverage flattened the screen action but created a mismatch.

Typical next step:

- guard attacks big
- big seals small
- offense delays to isolate the mismatch

#### `forced_help`

The primary action caused a second defender to commit.

Typical next step:

- pocket pass
- kickout
- swing
- hockey-assist chain

#### `direct_terminal`

The possession already has a terminal path.

Examples:

- clean pull-up
- direct rim finish
- strip turnover
- shooting foul

Typical next step:

- call terminal mechanic immediately

---

## Loop 3: Defensive Reaction

After the primary action, the defense must choose how it reacts.

This is where the possession becomes five-man basketball instead of two-man basketball.

### Defensive Reaction States

The defense should choose one of:

- `stay_home`
- `tag_roller`
- `full_help`
- `late_rotate`
- `switch_hold`
- `scramble_recovery`

### Definitions

#### `stay_home`

The defense refuses to help off shooters.

Consequences:

- stronger on kickout denial
- weaker rim resistance if the handler gets downhill cleanly

#### `tag_roller`

The weak-side or nail defender checks the roll man but does not fully sell out.

Consequences:

- pocket pass is harder
- kickout window may open

#### `full_help`

The defense commits fully to stopping the ball at the rim or in the paint.

Consequences:

- strong rim contest
- high kickout / extra-pass potential

#### `late_rotate`

The defense reacts, but not on time.

Consequences:

- open catch-and-shoot
- attackable closeout

#### `switch_hold`

The switch happened and the defense trusts the matchup.

Consequences:

- no immediate rotation
- offense must exploit the mismatch itself

#### `scramble_recovery`

The defense is now rotating out of shape.

Consequences:

- swing-swing threes
- relocation threes
- long closeouts

### Defensive Inputs

The defensive reaction loop should consume:

- `coverage_weights`
- `help_aggressiveness`
- `switch_rate`
- `pre_switch_rate`
- `help_rotation`
- `closeout`
- `rim_protect`
- `interior_def`

The exact shot resolution still belongs to the mechanic layer. This loop only determines how the possession proceeds.

---

## Loop 4: Exploitation Or Continuation

This is the missing layer in the current engine.

Once advantage exists, the offense should choose whether to:

- end the possession now
- move the ball
- attack the next defender
- reset into a second-side action

### Continuation States

The offense should have at least these progression branches:

- `immediate_shot`
- `attack_closeout`
- `pocket_pass`
- `kickout`
- `swing`
- `reset_to_creator`
- `second_side_action`

### Definitions

#### `immediate_shot`

The current actor should finish the possession.

Examples:

- clean pull-up
- open catch-and-shoot
- clean rim finish

#### `attack_closeout`

The receiver catches an advantage and drives the rotating defender.

This is not a new play family. It is a continuation state.

Typical outputs:

- drive attempt
- paint touch
- foul
- next kickout

#### `pocket_pass`

The handler exploits rim pressure with a pass to the roller or dunker.

Typical outputs:

- roll finish
- dump-off
- help-side strip

#### `kickout`

The ball leaves the paint to the perimeter after help commits.

This should not usually be terminal by itself.

A kickout should feed one of:

- immediate shot
- attack closeout
- swing
- second-side reset

#### `swing`

The receiver moves the ball to the next perimeter player without trying to score first.

This is the core hockey-assist branch.

Typical sequence:

1. drive
2. kickout
3. swing
4. shot

This branch is essential for distributing three-point volume more realistically across the lineup.

#### `reset_to_creator`

The offense gives the ball back to a creator and reforms shape.

Typical next step:

- late-clock pull-up
- re-screen
- second-side PnR

#### `second_side_action`

The possession flows into a new action after the first advantage was moved.

Examples:

- kickout -> swing -> side PnR
- kickout -> handoff
- reversal -> weak-side flare
- reset -> ISO against a shifted defense

### Continuation Inputs

This loop should consume:

- `closeout_attack_rate`
- `second_side_rate`
- `pass_vision`
- `pass_accuracy`
- `decision_making`
- `catch_shoot`
- `separation`
- `burst`
- `closeout`
- `help_rotation`

This is also where the new team-level tendency fields matter most.

---

## Loop 5: Terminal

Only after progression has chosen the final exploitation path should the terminal mechanic run.

Terminal categories:

- `pullup`
- `catch_and_shoot`
- `drive_attempt`
- `post_touch`
- `free_throw_sequence`
- `turnover`
- `rebound_resolution`

The terminal loop should be the only place where the possession officially ends.

That keeps the engine from becoming first-action terminal by default.

---

## Team Tendency Integration

The progression layer should read from `TeamTactics`, not from a separate style object.

### Existing Tactics Fields Already Relevant

- `pace_target`
- `transition_frequency`
- `crash_glass_rate`
- `switch_rate`
- `pick_and_roll_rate`
- `play_family_weights`
- `coverage_weights`

### New Progression Fields

- `star_usage_bias`
- `closeout_attack_rate`
- `second_side_rate`
- `corner_spacing_bias`
- `shooter_distribution_weights`

### What Each Field Should Influence

#### `star_usage_bias`

Controls:

- how steeply primary creation is concentrated
- how often the offense resets back to its star
- how often elite foul drawers remain involved

#### `closeout_attack_rate`

Controls:

- how often a catchout receiver attacks instead of shooting
- how often the possession re-enters the paint after a kickout

#### `second_side_rate`

Controls:

- how often the ball moves again after the first kickout
- how often possessions become pass-pass-shot instead of pass-shot

#### `corner_spacing_bias`

Controls:

- whether weak-side spacing favors corner occupation vs slot/wing occupation
- corner vs above-break shot distribution

#### `shooter_distribution_weights`

Controls:

- which players are preferred as kickout or swing targets
- how team-specific shot distribution differs even with similar player ratings

---

## Canonical Progression Paths

The engine should support these canonical paths before adding many more play families.

### Path A: Direct Strong-Side Finish

1. initiate action
2. create direct advantage
3. no strong help arrives
4. terminal mechanic:
   - pull-up
   - rim finish
   - foul

### Path B: Paint Touch Kickout

1. initiate action
2. paint touch created
3. defense helps
4. kickout to perimeter
5. terminal mechanic:
   - catch-and-shoot
   - attack closeout

### Path C: Kickout Swing Shot

1. initiate action
2. force help
3. kickout
4. swing to next player
5. catch-and-shoot

This is the most important missing path in the current engine.

### Path D: Kickout Attack Closeout

1. initiate action
2. kickout
3. receiver attacks closeout
4. drive attempt or next kickout

### Path E: First Action Fails, Second Side Begins

1. primary action creates no direct finish
2. offense resets or swings
3. second-side creator attacks
4. possession ends from the second side

### Path F: Early Offense

1. possession enters in transition or early offense
2. immediate rim or trail-three chance
3. if unavailable, flow into halfcourt without resetting the possession model

---

## Static Players Vs Active Participants

The current engine still treats non-primary players too often as passive targets.

The progression layer must instead treat them as active participants with roles:

- `strong_side_actor`
- `screening_actor`
- `roller_or_popper`
- `weak_side_lifter`
- `corner_spacer`
- `slot_spacer`
- `trail_big`
- `second_side_creator`

These are not new player attributes. They are temporary possession roles.

The same player can occupy different roles on different possessions.

---

## Weak-Side Availability Model

The engine does not need full optical-tracking realism to model weak-side movement.

It only needs to represent whether a weak-side player is:

- stationary and covered
- spaced and available
- relocating into a better window
- freed by weak-side screening

### Minimal Weak-Side States

- `occupied`
- `available`
- `relocating`
- `screen_freed`

These states should influence:

- who is the next pass target
- whether the shot is corner vs above-break
- whether the pass becomes a swing or an immediate shot

---

## Event Logging Requirements

The progression model should emit event chains that make the possession legible.

Examples:

### Drive Kickout Shot

1. `SCREEN`
2. `DRIVE`
3. `PASS` note=`pnr kickout`
4. `SHOT` note=`catch_and_shoot`

### Hockey Assist Three

1. `SCREEN`
2. `DRIVE`
3. `PASS` note=`paint kickout`
4. `PASS` note=`swing`
5. `SHOT` note=`catch_and_shoot`

### Second-Side Reset

1. `SCREEN`
2. `PASS` note=`kickout reset`
3. `PASS` note=`second_side_entry`
4. `DRIVE`
5. `SHOT`

The engine does not need official NBA scorer logic for hockey assists yet, but it should preserve the event chain.

---

## Initial Implementation Order

Build the progression layer in this order:

1. add `swing` as a real continuation branch after kickouts
2. make `second_side_rate` produce pass-pass-shot chains
3. allow kickout receivers to become passers, not only terminal shooters or drivers
4. add transition / early-offense entry branch
5. add weak-side availability states
6. later add explicit off-ball actions:
   - `HANDOFF`
   - `STAGGER`
   - `SPAIN_PICK_AND_ROLL`
   - `DOUBLE_DRAG`

Do not add many new play families until the basic continuation loops are working.

---

## Calibration Implications

Several current calibration misses are probably progression problems, not mechanic-weight problems:

- `three_pa_concentration` too high:
  - too many possessions end at first kickout
  - too few swing / second-side threes

- `assist_concentration` too low:
  - not enough possessions flow through pass-pass-shot chains
  - primary creators are not generating enough multi-pass sequences

- overly repetitive shot ownership:
  - same terminal actors absorb too many outcomes

The progression layer should reduce these problems naturally.

---

## Non-Goals

This document does not define:

- exact shot-quality math
- exact attribute weights
- exact foul branch formulas
- full playbook diagrams

Those belong in:

- [basketball_attribute_mechanics.md](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\docs\basketball_attribute_mechanics.md)
- [basketball_possession_model.md](C:\Users\brssn\OneDrive\Desktop\parleysubstrate\docs\basketball_possession_model.md)

This document only defines how possessions progress from one decision point to the next.

