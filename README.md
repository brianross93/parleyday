# parleyday

Local Flask dashboard for MLB and NBA slate analysis, Monte Carlo pricing, and parlay construction using live market data plus cached sports context.

This repo is built as an internal decision-support tool. The goal is not to predict winners in the abstract. The goal is to:

- refresh a live slate
- estimate probabilities for supported game and player markets
- compare model probabilities against market-implied prices
- surface candidate edges
- assemble parlays more intelligently than a raw high-probability sort

## What The App Does

At a high level, the app has five layers:

1. Data ingestion and caching
2. Market recognition and normalization
3. Simulation and direct pricing models
4. Edge and trust scoring
5. Parlay construction

The dashboard is a local Flask frontend over a Python backend. The heavy work is server-side, not client-side:

- live slate pulls
- lineup and injury refreshes
- Monte Carlo simulation
- parlay generation

## Architecture

Core files:

- [dashboard_app.py](/Users/brianross/Desktop/parleyday/dashboard_app.py): Flask app and dashboard form handling
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py): main slate loader, scoring engine, simulation integration, and parlay builder
- [refresh_slate.py](/Users/brianross/Desktop/parleyday/refresh_slate.py): same-day refresh for volatile inputs and recognized legs
- [daily_ingest.py](/Users/brianross/Desktop/parleyday/daily_ingest.py): baseline cache refresh
- [monte_carlo/mlb.py](/Users/brianross/Desktop/parleyday/monte_carlo/mlb.py): MLB plate-appearance Monte Carlo engine
- [monte_carlo/nba.py](/Users/brianross/Desktop/parleyday/monte_carlo/nba.py): NBA possession Monte Carlo engine
- [data_pipeline/cache.py](/Users/brianross/Desktop/parleyday/data_pipeline/cache.py): SQLite snapshot store
- [data_pipeline/mlb_profiles.py](/Users/brianross/Desktop/parleyday/data_pipeline/mlb_profiles.py): live MLB player and matchup profiles
- [data_pipeline/nba_profiles.py](/Users/brianross/Desktop/parleyday/data_pipeline/nba_profiles.py): NBA player stat profiles
- [data_pipeline/mlb_park_factors.py](/Users/brianross/Desktop/parleyday/data_pipeline/mlb_park_factors.py): park-factor helpers

## Local Workflow

Install and run:

```bash
pip install -r requirements.txt
python dashboard_app.py
```

Open:

```text
http://127.0.0.1:5000
```

Recommended usage pattern:

1. Run a baseline ingest in the morning.
2. Run a same-day refresh closer to lock.
3. Use the dashboard in `sim` mode.
4. Re-refresh when lineups, injuries, or weather change.

## Cache-First Data Model

The project is intentionally cache-first.

SQLite path by default:

```text
data/parleyday.sqlite
```

We do not want to pull every remote source on every dashboard request if we can avoid it. The app separates:

- stable daily inputs
- volatile same-day inputs

### Daily baseline pull

```bash
python3 daily_ingest.py --date 2026-03-29
```

This is for slower-moving inputs like:

- schedules
- baseline team form
- historical profile inputs

### Same-day refresh

```bash
python3 refresh_slate.py --date 2026-03-29 --sport both --kalshi-pages 25
```

This is for volatile inputs like:

- recognized market legs
- MLB lineups
- MLB probable pitchers
- MLB weather
- MLB bullpen freshness
- MLB roster and transaction context
- NBA injury-report snapshots
- NBA matchup context

The dashboard can do this refresh automatically before a run.

## Data Sources

### MLB

- MLB Stats API for schedule, standings, rosters, and boxscore-derived context
- live lineup and probable-pitcher context from MLB feeds when available
- Statcast-style park-factor inputs through [data_pipeline/mlb_park_factors.py](/Users/brianross/Desktop/parleyday/data_pipeline/mlb_park_factors.py)
- weather pulled during same-day refresh

### NBA

- Kalshi schedule and market feeds for slate construction
- ESPN/NBA endpoints for matchup and profile context
- official NBA injury report PDFs for availability status

### Market data

- Kalshi REST API for market discovery and pricing

## Market Recognition

The app does not use raw exchange payloads directly. It first recognizes and normalizes them into internal `Leg` objects in [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py).

Supported leg types today:

- MLB moneylines
- MLB totals
- MLB HR props
- MLB hits props
- MLB strikeout props
- NBA moneylines
- NBA totals
- NBA `PTS`, `REB`, and `AST` props

Each leg gets:

- a normalized label
- a game key
- a category
- an implied market probability
- matchup notes

Important detail:

- the app now preserves threshold labels more carefully for props so low-bar markets do not get collapsed into the wrong line

## Simulation Methods

### MLB game model

The MLB engine in [monte_carlo/mlb.py](/Users/brianross/Desktop/parleyday/monte_carlo/mlb.py) is a plate-appearance Monte Carlo simulator.

At a high level:

- each plate appearance resolves probabilistically
- event rates combine batter profile, pitcher profile, and league baseline
- runner advancement uses probabilistic base-state logic
- fatigue adjusts pitcher outcomes
- the engine rolls full games thousands of times

The live MLB pipeline uses:

- cached lineup hitter profiles
- cached probable pitcher profiles
- weather context
- bullpen freshness
- park factors
- matchup-level run calibration

That live calibration path is wired through:

- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `expected_mlb_runs()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `simulate_live_mlb_leg_probabilities()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `build_calibrated_live_mlb_contexts()`

#### MLB context adjustments currently modeled

- park effects
- light weather adjustment
- lineup confirmation state
- bullpen fatigue
- probable-pitcher availability risk
- calibrated total environment
- side compression / home-edge adjustments

#### MLB fallback behavior

If the lineup is not available, the app does not try to force a player-prop simulation. It falls back instead.

This is deliberate:

- game-level estimates can tolerate rougher inputs
- player-level props should not be simulated confidently without actual hitters in the lineup

### NBA game and player model

The NBA path now has a real possession-by-possession Monte Carlo engine in [monte_carlo/nba.py](/Users/brianross/Desktop/parleyday/monte_carlo/nba.py).

At a high level:

- the app first estimates team scoring environment with `expected_nba_points()`
- player profiles are projected into a live rotation with minute and usage redistribution
- the simulator rolls possessions, not just final scores
- each possession can resolve as:
  - turnover
  - free-throw trip
  - 2-point attempt
  - 3-point attempt
  - offensive rebound continuation
  - defensive rebound ending the possession
- scoring, rebounds, and assists are attributed to players directly from the possession flow

The current NBA live path uses:

- team-form snapshots
- home-court adjustment
- official injury availability context
- recent player stat profiles from [data_pipeline/nba_profiles.py](/Users/brianross/Desktop/parleyday/data_pipeline/nba_profiles.py)
- projected minute redistribution
- possession-level event rolls for supported props

Relevant functions:

- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `expected_nba_points()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `project_nba_player_means()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `build_live_nba_team_context()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `simulate_live_nba_leg_probabilities()`
- [monte_carlo/nba.py](/Users/brianross/Desktop/parleyday/monte_carlo/nba.py) `NBAGameSimulator`

Today that simulator supports:

- NBA moneylines
- NBA totals
- NBA `PTS`, `REB`, and `AST` props

It is still an MVP in the sense that rotations are profile-based rather than driven by a full historical substitution model, but it is no longer just a direct stat sampler.

## Score Modes

The dashboard supports multiple score modes.

### `implied`

Pure market-implied pricing.

Use this as:

- a baseline
- a fast mode
- a fallback sanity check

### `sim`

Live matchup simulation.

This is the main custom model path and uses:

- game-level simulation for moneylines and totals
- live player-stat simulation for supported props
- market fallback where sim support is unavailable

### `residual`

Residual-style adjustment layer used mainly for the cash tier.

### `heuristic`

Direct market probabilities plus lightweight hand-tuned bias adjustments.

### `ising`

Experimental Gibbs / Ising-style sampling path.

## Pricing Coverage And Trust

Not every leg on a slate is priced with the same confidence.

The app tracks pricing source per leg:

- `Monte Carlo`
- `Market implied`
- `Market fallback`
- `Residual model`
- `Heuristic model`

The dashboard also shows a trust score.

Trust is currently influenced by:

- whether a leg was fully sim-priced or fell back to market
- whether the edge is suspiciously extreme
- whether the market type is inherently noisier

This is not a formal posterior confidence interval. It is a practical ranking aid for internal use.

## Parlay Construction Method

The parlay builder has been tightened beyond a simple “highest probabilities win” rule.

Current design goals:

- require positive edge for inclusion
- apply minimum trust thresholds
- penalize fallback legs, especially in cash-style tickets
- penalize suspiciously extreme edges
- avoid obviously incompatible pairings
- reward better correlation-aware joint fit
- keep payout bands aligned with the tier label

Tier definitions live in [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py).

Current tiers:

- `Cash`
- `Decent Bet`
- `Longshot`

### How parlay selection works now

For each tier, the builder:

1. Filters legs by minimum edge and trust.
2. Applies extra restrictions for the tier.
3. Trims to a candidate set.
4. Searches combinations of the target size.
5. Scores combinations on:
   - mean edge
   - trust
   - correlation-aware joint model probability
   - market-implied joint probability
   - payout-band fit
   - state-search compatibility score

This is implemented mainly in:

- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `build_tiered_parlays()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `partial_state_score()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `model_parlay_probability()`
- [quantum_parlay_oracle.py](/Users/brianross/Desktop/parleyday/quantum_parlay_oracle.py) `leg_trust_score()`

### Important practical note

An empty tier is acceptable.

If no combination satisfies:

- positive edge
- trust requirements
- payout-band fit

the correct behavior is to show no parlay for that tier rather than invent a bad one.

## Fallback Philosophy

The app is intentionally designed to degrade gracefully.

Examples:

- missing MLB lineup: MLB player-prop sim falls back instead of forcing a broken simulation
- missing cached profile: prop falls back to market
- empty recognized slate: the dashboard returns an empty result instead of crashing
- missing live source: cache is preferred where possible

The dashboard now also exposes specific reasons for some fallbacks, such as:

- `Lineup not confirmed`

## Backtesting

This repo includes historical game-level backtests for 2025.

Scripts:

- [backtest_monte_carlo_2025.py](/Users/brianross/Desktop/parleyday/backtest_monte_carlo_2025.py)
- [backtest_nba_2025.py](/Users/brianross/Desktop/parleyday/backtest_nba_2025.py)
- [backtest_2025_all.py](/Users/brianross/Desktop/parleyday/backtest_2025_all.py)
- [train_holdout_parlay_backtest.py](/Users/brianross/Desktop/parleyday/train_holdout_parlay_backtest.py)

What these do well:

- calibrate game-level model behavior
- compare variants out of sample
- evaluate synthetic parlay selection logic

What they do not yet do:

- replay real historical market snapshots end to end
- prove true historical profitability

We do not currently have a full archive of historical odds, so parlay-profitability backtests are still proxies rather than true P&L replays.

## Limits And Current Scope

This is an internal tool, not a finished commercial betting platform.

Current limitations:

- MLB player props depend heavily on lineup availability
- NBA player model is still an MVP rotation model, even though pricing now runs through a possession engine
- some edges can still be false positives due to market parsing or sparse inputs
- historical profitability is not fully known without archived odds
- trust score is heuristic, not fully statistical

Best use right now:

- slate triage
- candidate edge discovery
- comparing model vs market
- generating candidate parlays for human review

Less appropriate use right now:

- blind automation
- aggressive bankroll sizing
- assuming every displayed edge is true EV

## Useful Commands

Run the dashboard:

```bash
python dashboard_app.py
```

Run the main oracle from CLI:

```bash
python3 quantum_parlay_oracle.py --date 2026-03-29 --sport both --score-source sim
```

Run daily baseline ingest:

```bash
python3 daily_ingest.py --date 2026-03-29
```

Run same-day refresh:

```bash
python3 refresh_slate.py --date 2026-03-29 --sport both --kalshi-pages 25
```

Run the standalone MLB MVP sim:

```bash
python3 simulate_mlb_mvp.py
```

Run a live cached MLB matchup:

```bash
python3 simulate_live_mlb.py --date 2026-03-29 --matchup TB@STL
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

## Deploy

This repo includes:

- [wsgi.py](/Users/brianross/Desktop/parleyday/wsgi.py)
- [Procfile](/Users/brianross/Desktop/parleyday/Procfile)
- [render.yaml](/Users/brianross/Desktop/parleyday/render.yaml)

The deployed app runs the same server-side Python logic as local mode. The browser is only a frontend shell; refresh, simulation, and parlay construction all happen on the backend.
