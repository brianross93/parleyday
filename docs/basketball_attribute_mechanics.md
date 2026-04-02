# Basketball Attribute & Mechanics Model

## Purpose

This document defines:

1. every player attribute the sim engine uses
2. what each attribute means
3. how each attribute is sourced or inferred
4. which mechanics each attribute feeds
5. initial bounded influence ranges
6. what must be hand-specified vs. what can be learned

This is the load-bearing design decision of the project. If the attribute model is wrong, possessions, tactics, box scores, game flow, and downstream prediction signal are all fake.

---

## Design Principles

**Small and real over large and vague.** Every attribute must connect to at least one mechanic. No attribute exists "because 2K has it."

**Data source honesty.** Every attribute has a data source. Some sources are strong (FT%, block rate, rim FG%). Some are weak proxies (lineup-level defensive 3P% for closeout). Some are positional defaults (screen setting). All attributes are active in the engine regardless of data quality, but weak-source attributes have their mechanic weights capped until the source is validated or upgraded. The doc marks each attribute's source quality explicitly.

**Bounded 1-20 scale.** All attributes use a 1-20 integer scale. This forces meaningful distinctions. The scale represents position-group-relative ability within the current NBA player population.

**Composable, not monolithic.** Mechanics consume small combinations of attributes (2-5), not a single composite "overall" rating. The interaction happens in the mechanic, not in the attribute.

**No junk drawers.** If an attribute is doing three different jobs, it needs to be three attributes.

---

## Scale Definition: 1-20

| Rating | Meaning | Approximate Percentile |
|--------|---------|----------------------|
| 1-3 | Liability. Actively exploitable at this skill. | Bottom 10% |
| 4-6 | Below average. Opponent can target this. | 10th-30th |
| 7-9 | Serviceable. Not a weakness, not a weapon. | 30th-50th |
| 10-11 | Average to slightly above. Solid starter. | 50th-65th |
| 12-14 | Above average. Clear positive contributor. | 65th-85th |
| 15-16 | Very good. Top-tier at this skill. | 85th-93rd |
| 17-18 | Elite. One of the best in the league. | 93rd-98th |
| 19-20 | Historically elite. Reserved for best-in-league seasons. | 98th+ |

**Position-group normalization still applies.** A center with containment 10 is average for a center. A guard with containment 10 is average for a guard.

---

## Attribute Registry

### Offensive Attributes (13)

| # | Attribute | Short Name | Source Quality | Definition |
|---|-----------|-----------|---------------|------------|
| O1 | Ball Security | `ball_security` | Strong | Protecting the dribble under pressure. |
| O2 | Separation Creation | `separation` | Moderate | Ability to create space from a defender using handles, footwork, and change of direction. |
| O3 | Burst | `burst` | Proxy | First-step acceleration and downhill explosion. |
| O4 | Finishing | `finishing` | Strong | Converting at the rim and in the paint. |
| O5 | Pull-Up Shooting | `pullup_shooting` | Strong | Creating and converting jump shots off the dribble. |
| O6 | Catch-and-Shoot | `catch_shoot` | Strong | Converting spot-up and catch-and-shoot attempts. |
| O7 | Passing Vision | `pass_vision` | Weak Proxy | Reading defensive rotations and identifying passing windows before they fully open. |
| O8 | Passing Accuracy | `pass_accuracy` | Positional Default | Delivering the ball on time and on target through traffic. |
| O9 | Decision-Making | `decision_making` | Moderate | Compressed cognitive attribute for shot/pass/reset quality. |
| O10 | Screen Setting | `screen_setting` | Positional Default | Quality of ball screens and off-ball screens. |
| O11 | Offensive Rebounding | `oreb` | Strong | Pursuing and securing offensive rebounds. |
| O12 | Free Throw | `free_throw` | Direct | Dual representation: `free_throw_rating` for profiles and `ft_pct_raw` for mechanics. |
| O13 | Foul Drawing | `foul_drawing` | Strong | Ability to initiate and earn contact on drives and shots. |

### Defensive Attributes (9)

| # | Attribute | Short Name | Source Quality | Definition |
|---|-----------|-----------|---------------|------------|
| D1 | Containment | `containment` | Moderate | Staying in front of a ball handler. |
| D2 | Closeout Quality | `closeout` | Weak Proxy | Recovering to a shooter on kick-outs and rotations. Mechanic weights capped at 0.20 until validated. |
| D3 | Screen Navigation | `screen_nav` | Weak Proxy | Fighting through or recovering from ball screens. |
| D4 | Interior Defense | `interior_def` | Strong | Defending in the paint. |
| D5 | Rim Protection | `rim_protect` | Strong | Contesting and altering shots at the rim. |
| D6 | Steal Pressure | `steal_pressure` | Strong | Active hands and anticipation in passing lanes and on the ball. |
| D7 | Defensive Rebounding | `dreb` | Strong | Securing defensive rebounds. |
| D8 | Foul Discipline | `foul_discipline` | Strong | Defending without fouling. |
| D9 | Help Rotation | `help_rotation` | Weak Proxy | Speed and quality of help-side defensive rotations. |

### Context Attributes (3)

| # | Attribute | Short Name | Source Quality | Definition |
|---|-----------|-----------|---------------|------------|
| C1 | Stamina | `stamina` | Strong | Ability to maintain performance over a game. |
| C2 | Role Consistency | `role_consistency` | Moderate | How stable a player's production is game-to-game. |
| C3 | Clutch Modifier | `clutch` | Weak Proxy | Performance deviation in high-leverage situations. |

### Physical Attributes (2)

| # | Attribute | Short Name | Source Quality | Definition |
|---|-----------|-----------|---------------|------------|
| P1 | Size | `size` | Direct | Composite of height, weight, and strength. |
| P2 | Reach | `reach` | Direct | Wingspan and standing reach. |

**Total: 27 attributes**

`size` and `reach` are Direct source quality because height, weight, and wingspan are publicly available for every NBA player from combine and roster data.

---

## Attribute Split Rationale

### Why `handle` became three attributes

| Job | Now Handled By | Why It Matters |
|-----|---------------|---------------|
| Don't lose the ball under pressure | `ball_security` | Different turnover profiles need different ratings. |
| Get past your man with craft | `separation` | Craft and burst are not the same thing. |
| Make the right choice with the ball | `decision_making` | Decision quality changes possession outcomes independently of craft. |

### Why `perimeter_def` became three attributes

| Task | Now Handled By | Why It Matters |
|------|---------------|---------------|
| Stay in front of the ball handler | `containment` | Core on-ball defense skill. |
| Recover to a shooter after helping | `closeout` | Different movement pattern than containment. |
| Fight through a screen | `screen_nav` | Determines whether PnR creates a clean advantage. |

These three splits are the minimum resolution needed for `DROP` vs `SWITCH` to produce different outcomes for different defenders.

---

## Attribute Data Sources

Every attribute is active in the engine. Source quality determines how much to trust the mechanic weight.

| Attribute | Source Quality | Data Source |
|-----------|--------------|-------------|
| `ball_security` | Strong | Inverse turnover rate under high usage. |
| `separation` | Moderate | Rim attempt frequency among perimeter players plus pull-up shooting volume. |
| `burst` | Proxy | Rim attempt share plus foul draw rate. |
| `finishing` | Strong | Rim FG% or best available paint finishing proxy. |
| `pullup_shooting` | Strong | Pull-up FG% if available, otherwise weighted midrange plus 3P proxy. |
| `catch_shoot` | Strong | Catch-and-shoot FG% if available, otherwise 3P proxy. |
| `pass_vision` | Weak Proxy | Assist rate plus potential assists when available. |
| `pass_accuracy` | Positional Default | Flat modifier by position group until better data exists. |
| `decision_making` | Moderate | eFG relative to expected plus AST/TO composite. |
| `screen_setting` | Positional Default | C=12, PF=10, SF=7, SG=5, PG=5. |
| `oreb` | Strong | Offensive rebound rate per minute. |
| `free_throw` | Direct | FT%. |
| `foul_drawing` | Strong | FTA per FGA plus experience modifier. |
| `containment` | Moderate | Steal rate plus inverse foul rate plus position-adjusted defensive signals. |
| `closeout` | Weak Proxy | Opponent catch-and-shoot 3P% when player is in lineup. |
| `screen_nav` | Weak Proxy | Opponent PnR efficiency when on-ball defender, otherwise `containment - 2`. |
| `interior_def` | Strong | Block rate plus defensive rebounding rate plus position context. |
| `rim_protect` | Strong | Block rate with positional context. |
| `steal_pressure` | Strong | Steal rate. |
| `dreb` | Strong | Defensive rebound rate per minute. |
| `foul_discipline` | Strong | Inverse personal foul rate per minute. |
| `help_rotation` | Weak Proxy | `interior_def` proxy for bigs, `closeout` for perimeter players. |
| `stamina` | Strong | MPG / 48 adjusted for games played and production maintenance. |
| `role_consistency` | Moderate | Computed from game log variance. |
| `clutch` | Weak Proxy | Clutch split data when available, otherwise 0. |
| `size` | Direct | Composite of height, weight, wingspan. |
| `reach` | Direct | Wingspan plus standing reach. |

### Source Quality Legend

| Quality | Meaning | Mechanic Weight Policy |
|---------|---------|----------------------|
| Direct | Directly measurable, no inference | Full weight |
| Strong | Clean box-score or play-by-play derivation | Full weight |
| Moderate | Defensible composite proxy | Full weight, monitor in calibration |
| Proxy | Indirect inference | Full weight but flagged for upgrade |
| Weak Proxy | Noisy or lineup-dependent | Mechanic weights capped at 0.20 until validated |
| Positional Default | No player-level data | Full weight but flagged for per-player override |

---

## Attribute Inference Pipeline

### General Pipeline

```text
raw stat -> per-minute or per-possession rate -> rolling window (15-25 games)
         -> position-group percentile -> map to 1-20 scale
```

Rolling window:
- 20 games default
- 10 games for injury returns or role changes
- 30+ games for stable veterans

Position groups:
- Guards: PG, SG
- Wings: SF, SG/SF tweeners
- Bigs: PF, C

### Percentile-to-Rating Mapping

| Percentile Range | Rating |
|-----------------|--------|
| 0-5% | 1-2 |
| 5-15% | 3-4 |
| 15-30% | 5-6 |
| 30-45% | 7-8 |
| 45-55% | 9-10 |
| 55-65% | 11-12 |
| 65-80% | 13-14 |
| 80-90% | 15-16 |
| 90-97% | 17-18 |
| 97-100% | 19-20 |

### Proxy and Default Formulas

`burst`:

```text
burst_raw = 0.6 * rim_attempt_share + 0.4 * foul_draw_rate
```

`screen_setting` defaults:

```text
C=12, PF=10, SF=7, SG=5, PG=5
```

`screen_nav` proxy:

```text
If play-type PnR defensive data is available:
    screen_nav = percentile_to_rating(opponent_pnr_efficiency, position_group, inverted)
Else:
    screen_nav = containment - 2 (floored at 1)
```

---

## Mechanics Map

Each mechanic specifies:
- which offensive attributes participate and at what weight
- which defensive attributes participate and at what weight
- what outcome branches exist
- how attribute advantage maps to branch probabilities

All attribute values are on the 1-20 scale.

### Mechanic 1: Drive Attempt

Offense:
- `separation` (0.25)
- `burst` (0.30)
- `finishing` (0.15)
- `decision_making` (0.10)
- `ball_security` (0.10)
- `foul_drawing` (0.10)

Defense:
- `containment` (0.35)
- `foul_discipline` (0.15)
- `interior_def` (0.20)
- `rim_protect` (0.20)
- `steal_pressure` (0.10)

Branches:
- rim attempt (clean)
- rim attempt (contested)
- foul drawn
- cut off to kick-out
- strip turnover
- charge
- bailout pull-up

### Mechanic 2: Pull-Up Jump Shot

Offense:
- `pullup_shooting` (0.50)
- `separation` (0.30)
- `burst` (0.20)

Defense:
- `containment` (0.45)
- `closeout` (0.20, capped until validated)
- `screen_nav` (0.35)

Base pull-up make rate:
- about `0.40`

### Mechanic 3: Catch-and-Shoot

Offense:
- `catch_shoot` (0.60)
- pass quality modifier (0.15)
- play success modifier (0.25)

Defense:
- `containment` (0.40)
- `closeout` (0.20, capped)
- `reach` (0.20)
- `size` (0.20)

Base catch-and-shoot make rate:
- about `0.375`

### Mechanic 4: Post Touch

Offense:
- `finishing` (0.40)
- `ball_security` (0.15)
- `burst` (0.15)
- `decision_making` (0.15)
- `foul_drawing` (0.15)

Defense:
- `interior_def` (0.40)
- `foul_discipline` (0.25)
- `steal_pressure` (0.20)
- `size` (0.15)

### Mechanic 5: Pick-and-Roll Resolution

Screen phase:

```text
screen_advantage = screener.screen_setting - on_ball_def.screen_nav
```

`DROP` action score:

```text
off_score = 0.30 * handler.pullup_shooting
          + 0.20 * handler.separation
          + 0.15 * handler.burst
          + 0.15 * screener.screen_setting
          + 0.10 * handler.ball_security
          + 0.10 * spacing_score

def_score = 0.30 * on_ball_def.screen_nav
          + 0.20 * on_ball_def.containment
          + 0.25 * drop_man.interior_def
          + 0.15 * drop_man.rim_protect
          + 0.10 * on_ball_def.closeout
```

`SWITCH` action score:

```text
off_score = 0.30 * handler.separation
          + 0.25 * handler.burst
          + 0.15 * handler.finishing
          + 0.15 * handler.pullup_shooting
          + 0.05 * handler.ball_security
          + 0.10 * spacing_score

def_score = 0.35 * switching_big.containment
          + 0.20 * switching_big.foul_discipline
          + 0.20 * help_def.rim_protect
          + 0.15 * help_def.interior_def
          + 0.10 * switching_big.closeout
```

Primary branches:
- pull-up jumper
- drive
- pocket pass to roller
- kick-out
- foul
- turnover
- reset / swing

### Mechanic 6: ISO Resolution

Offense:
- `separation` (0.25)
- `burst` (0.20)
- `pullup_shooting` (0.20)
- `finishing` (0.10)
- `decision_making` (0.10)
- `ball_security` (0.10)
- `foul_drawing` (0.05)

Defense:
- `containment` (0.40)
- `foul_discipline` (0.15)
- `steal_pressure` (0.15)
- help `rim_protect` (0.20)
- help `closeout` (0.10)

Primary branches:
- drive
- pull-up midrange
- pull-up three
- foul
- kick-out
- turnover
- reset

### Mechanic 7: Rebound Resolution

Offense:
- `oreb`
- team `crash_glass_rate`

Defense:
- `dreb`
- `rim_protect`

Base offensive rebound rate:
- about `0.27` on misses

### Mechanic 8: Free Throw Resolution

Attribute consumed:
- `ft_pct_raw`

Each free throw is an independent Bernoulli trial.

---

## Spacing Score

Spacing is lineup-level, not a player attribute.

```text
spacing_score = mean(catch_shoot for 3 non-handler, non-screener players on perimeter)
```

Typical range on the 1-20 scale:
- about `5-15`

Higher spacing:
- widens driving lanes
- improves kick-out value
- suppresses help-side rim resistance

---

## Fatigue System

```text
fatigue = minutes_played / (stamina_threshold * stamina_multiplier)
```

Penalty curve:

```text
if fatigue < 0.7:
    penalty = 0.0
elif fatigue < 0.9:
    penalty = (fatigue - 0.7) * 0.5
else:
    penalty = 0.10 + (fatigue - 0.9) * 1.5
```

Physical attributes take the full penalty.
Mental attributes take half the penalty.

---

## Sigmoid Normalization Function

```python
def sigmoid_normalize(raw_advantage: float, k: float = 0.15) -> float:
    return 1.0 / (1.0 + math.exp(-k * raw_advantage))
```

Interpretation on the 1-20 scale:
- `0` difference -> `0.500`
- `+-2` -> `0.574 / 0.426`
- `+-4` -> `0.646 / 0.354`
- `+-6` -> `0.711 / 0.289`

---

## Branch Probability Adjustment

```python
def adjust_branches(base_rates: dict, advantage: float,
                    branch_effects: dict, sensitivity: float = 1.5) -> dict:
    shift = advantage - 0.5
    adjusted = {}
    for branch, base in base_rates.items():
        effect = branch_effects.get(branch, 0)
        adjusted[branch] = max(base * (1.0 + effect * shift * sensitivity), 0.001)
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}
```

Calibration knobs:
- `k` for sigmoid steepness
- `sensitivity` for branch shifting

---

## What Must Be Hand-Specified vs. Learned

### Hand-Specified

- play family definitions
- coverage family definitions
- attribute-to-mechanic wiring
- outcome branch sets
- possession state machine

### Calibrated

- base branch rates
- mechanic weights
- sigmoid steepness
- branch sensitivity
- fatigue curve
- spacing multiplier
- free throw scaling

### Calibratable

- weak-proxy attribute values
- team coverage probabilities
- team play-family probabilities
- court-zone shot quality adjustments
- rebound geometry by shot location

---

## Calibration Framework

### Priority 1: Possession-Level Distribution Fit

Loss:
- chi-squared / squared-error fit on outcome frequencies

Targets:
- PPP: `1.05-1.15`
- turnover rate: `12%-16%`
- shooting foul rate: `15%-22%`
- OREB on misses: `24%-30%`
- 3PA share: `35%-42%`
- FT rate: `0.22-0.30`
- rim attempt share: `28%-36%`
- midrange share: `8%-16%`

Shot-type targets:
- rim: about `32%`
- short midrange / paint: about `10%`
- long midrange: about `12%`
- above-break 3: about `27%`
- corner 3: about `10%`

If these are wrong, fix base rates, `k`, and `sensitivity` before adding more tactical complexity.

### Priority 2: Attribute Sensitivity Validation

Required directional checks:
- higher `separation` -> more rim pressure, fewer turnovers
- higher `containment` -> more cutoffs / resets / turnovers
- higher `catch_shoot` lineup average -> stronger kick-out outcomes
- higher `foul_drawing` -> materially more FTA
- higher `rim_protect` in `DROP` -> lower rim conversion
- higher `screen_nav` -> lower handler pull-up quality in PnR
- better switched big `containment` -> less mismatch damage

### Priority 3: Usage Concentration

Loss:
- MAE on usage concentration metrics

Targets:
- top scorer possession share: `25%-36%`
- top 2 usage share: `44%-58%`
- top 3 usage share: `58%-74%`
- assist concentration: `30%-50%`
- FTA concentration: `25%-50%`
- 3PA concentration: `20%-40%`

`offensive_load`:

```text
offensive_load = (0.35 * separation
                + 0.25 * pullup_shooting
                + 0.15 * ball_security
                + 0.15 * finishing
                + 0.10 * foul_drawing)
```

### FTA-by-Archetype Double-Count Check

Monitor interaction between:
- `foul_drawing`
- `burst`
- `separation`

Target FTA/game bands:
- `1-4` foul drawing: `0.5-2.0`
- `5-8`: `1.5-4.0`
- `9-12`: `3.5-6.5`
- `13-16`: `6.0-10.0`
- `17-20`: `9.0-16.0`

If archetypes blow past the band, reduce double-counting in the foul branch.

### Priority 4: Game-Level Calibration

Primary losses:
- Brier score on win probability
- CRPS on game total distributions
- KS-style distribution checks on margin

Secondary targets:
- team score RMSE < `12`
- possessions MAE < `3`
- starter minute MAE < `3`
- starter point RMSE < `6`
- starter assist RMSE < `2`
- starter rebound RMSE < `2`

Variance matters:
- real NBA game total sigma is roughly `20-22`
- if simulated sigma is far below that, totals/spreads/parlays become unusable

### Priority 5: Feature Extraction Validation

The sim is allowed to prove itself first as a feature generator.

Candidate features:
1. bench minutes quality gap
2. closing lineup net rating estimate
3. usage concentration mismatch
4. rebounding environment by lineup archetype
5. projected pace mismatch
6. foul trouble cascade probability

### Calibration Sequencing Rule

Do not advance to a later calibration priority until the earlier one is passing.

---

## Usage Concentration: Foul Drawing Deep Dive

Real NBA FTA distribution is wide:
- low foul-draw role player: `0-2`
- moderate slashing wing: `3-6`
- elite foul drawer: `8-16`

Approximate `foul_drawing` to FTA/game map:

| `foul_drawing` | Expected FTA/game |
|---|---|
| 1-4 | 0.5-1.5 |
| 5-8 | 1.5-3.5 |
| 9-12 | 3.5-6.0 |
| 13-15 | 6.0-9.0 |
| 16-18 | 9.0-13.0 |
| 19-20 | 13.0-16.0+ |

Experience modifier:
- veterans (8+ years): `+1`
- rookies: `-1`

The engine should eventually model foul-trouble cascades when elite foul drawers repeatedly attack the same defender.

---

## Mismatch Mechanics: Size and Reach

`size` and `reach` are modifiers, not replacements for skill.

### `size` enters:
- post offense
- post defense
- switch mismatch logic
- contested rebounding
- future screen-setting enhancement

### `reach` enters:
- shot contest quality
- rim protection
- steal pressure / passing lane disruption
- finishing over defenders

Illustrative reach modifier:

```text
effective_contest = base_contest * (1.0 + (defender.reach - 10) * 0.03)
```

---

## Example Player Profiles

### Shai Gilgeous-Alexander

Directional example:
- `ball_security 15`
- `separation 17`
- `burst 14`
- `finishing 18`
- `pullup_shooting 16`
- `catch_shoot 10`
- `decision_making 16`
- `foul_drawing 19`
- `oreb 4`
- `free_throw 17 display / 0.874 raw`
- `containment 14`
- `closeout 13`
- `screen_nav 12`
- `interior_def 5`
- `rim_protect 3`
- `steal_pressure 15`
- `dreb 10`
- `foul_discipline 13`
- `stamina 16`
- `size 12`
- `reach 15`

Derived load example:

```text
offensive_load = 0.35(17) + 0.25(16) + 0.15(15) + 0.15(18) + 0.10(19) = 16.70
```

### Rudy Gobert

Directional example:
- `ball_security 4`
- `separation 2`
- `burst 6`
- `finishing 14`
- `pullup_shooting 1`
- `catch_shoot 2`
- `decision_making 11`
- `foul_drawing 8`
- `oreb 16`
- `free_throw 12 display / 0.659 raw`
- `containment 8`
- `closeout 4`
- `screen_nav 3`
- `interior_def 18`
- `rim_protect 19`
- `steal_pressure 7`
- `dreb 17`
- `foul_discipline 12`
- `stamina 14`
- `size 18`
- `reach 19`

Interpretation:
- `DROP` should emphasize Gobert’s rim deterrence
- `SWITCH` should expose his containment / closeout limitations against fast guards

---

## Data Quality Improvement Targets

These are upgrade targets, not milestones.

### Attribute Source Upgrades

| Attribute | Current Source | Target Source | Expected Impact |
|-----------|--------------|--------------|----------------|
| `closeout` | Lineup-level defensive 3P% | Tracking closeout speed data | High |
| `screen_nav` | `containment - 2` fallback | Defender-level PnR data | High |
| `burst` | Rim attempt share + foul draw rate | Acceleration tracking | Moderate |
| `separation` | Rim attempts + pull-up volume | On-ball creation tracking | Moderate |
| `pass_vision` | Assist-rate proxy | Potential assists + secondary assists | Moderate |
| `pass_accuracy` | Positional default | Pass-turnover classification | Moderate |
| `screen_setting` | Positional default | PnR frequency + roll man production | Low-moderate |
| `help_rotation` | `interior_def` / `closeout` proxy | Better lineup or tracking rotation signals | Low-moderate |
| `role_consistency` | Game log variance | Better implementation of same | Low |
| `clutch` | Default 0 | Clutch split data | Low |

### Play Family and Coverage Additions

Play families to add as needed:
- `TRANSITION_PUSH`
- `HANDOFF`
- `POST_TOUCH`
- `HORNS`
- `DOUBLE_DRAG`
- `STAGGER`
- `SPAIN_PICK_AND_ROLL`

Coverages to add as needed:
- `HEDGE`
- `BLITZ`
- `ICE`
- `ZONE`
- `SCRAM`

---

## Resolved Design Decisions

### `free_throw` scale mapping

Decision:
- `free_throw_rating` exists for profile display
- `ft_pct_raw` is consumed directly by the FT mechanic

### `foul_drawing` as a standalone attribute

Decision:
- yes, it is standalone
- the FTA distribution is too wide to emerge reliably from indirect attributes alone

### `size` as a standalone attribute

Decision:
- yes, it is direct-source and should remain explicit

### Hot-hand / confidence effects

Decision:
- deferred
- streaks should emerge from repeated probabilistic outcomes before adding explicit momentum logic

---

## Open Questions

1. Is `27` attributes too many?
2. Should `offensive_load` stay derived or become explicit?
3. How should foul trouble cascade be modeled once the engine is calibrated enough to justify it?
