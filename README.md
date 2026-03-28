# parleyday

Local Flask dashboard for live MLB/NBA parlay state search using Kalshi market data, MLB Stats API inputs, and a backtest framework for residual-model experiments.

The repo now also includes an MVP MLB Monte Carlo simulator foundation under `monte_carlo/` that can:

- simulate plate appearances with a log5-style matchup model
- roll full games across thousands of trials
- aggregate player stat distributions
- compare sim-derived prop probabilities against market prices

## Run locally

```powershell
pip install -r requirements.txt
python dashboard_app.py
```

Then open `http://127.0.0.1:5000`.

## Monte Carlo MVP

Run the standalone simulator demo:

```powershell
python3 simulate_mlb_mvp.py
```

This prints a JSON payload with average scores plus scored prop edges for a demo MLB matchup.

Current scope:

- MLB only
- starter-vs-lineup plate appearance engine
- basic fatigue and runner advancement logic
- player props for hitters plus starter strikeouts

This is the Phase 1 foundation for the broader spec; it is not yet wired into the Flask dashboard or live Statcast ingestion.

## Live Matchup Score Mode

The existing parlay search app now supports a `sim` score mode in both the Flask UI and CLI:

```powershell
python3 quantum_parlay_oracle.py --date 2026-03-28 --sport both --score-source sim
```

What `sim` does today:

- loads real MLB and NBA matchups for the selected slate
- simulates game-level score distributions for each matchup
- prices moneylines and totals from those simulated outcomes
- falls back to market-implied probabilities for player props until the full live player-data layer is added

In the Flask dashboard, `Analyze` can now refresh same-day slate inputs first before scoring/parlay construction.

## Cache-First Workflow

The project now supports a cache-first ingest flow backed by SQLite at `data/parleyday.sqlite` by default.

Daily baseline pull:

```powershell
python3 daily_ingest.py --date 2026-03-28
```

Same-day volatile refresh:

```powershell
python3 refresh_slate.py --date 2026-03-28 --sport both --kalshi-pages 10
```

Recommended cadence:

- run `daily_ingest.py` once in the morning to cache schedules and baseline team-form inputs
- run `refresh_slate.py` closer to lock to refresh same-day market data and volatile inputs
- run simulations from the cached data repeatedly during the day

Notes:

- the simulator now prefers cached team-form snapshots and only falls back to live pulls if cache is missing or stale
- `refresh_slate.py` now caches per-game context snapshots for MLB and NBA even if the live market leg refresh is slow or fails
- MLB game contexts currently include confirmed lineups, probable pitchers, venue metadata, weather, roster-status snapshots, recent transactions, and bullpen freshness
- NBA game contexts currently include matchup status and venue metadata, with a reserved availability slot for a future injury source

MLB availability signals now come from:

- active and 40-man roster status snapshots
- recent MLB transaction feed entries
- recent reliever usage from completed game boxscores

NBA availability signals now come from:

- official NBA injury report PDFs when available

## Live MLB Player Sim

After `refresh_slate.py` caches same-day MLB matchup profiles, you can run a real matchup through the local player-level MLB simulator:

```powershell
python3 simulate_live_mlb.py --date 2026-03-28 --matchup TB@STL
```

This reads cached lineup-player and probable-pitcher profiles from SQLite and runs the Monte Carlo engine against that matchup locally.

## Deploy

This repo includes:

- `wsgi.py`
- `Procfile`
- `render.yaml`

so it can be deployed on a simple Python host like Render or exposed from a local machine with a tunnel.
