import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from phase3_backtest import (
    HistoricalGame,
    daterange,
    default_team_state,
    fetch_historical_schedule,
    team_runs_allowed,
    team_runs_scored,
    update_team_stats,
)
from quantum_parlay_oracle import expected_mlb_runs


RESULTS_DIR = Path("results") / "monte_carlo_backtest"
LEAGUE_AVG_RUNS = 4.5


def build_team_form_snapshot(team_stats: dict) -> dict[str, dict]:
    snapshot = {}
    for code, state in team_stats.items():
        snapshot[code] = {
            "runs_scored_pg": team_runs_scored(state),
            "runs_allowed_pg": team_runs_allowed(state),
            "recent_win_pct": float(state["recent_win_rate"]),
            "win_pct": (float(state["wins"]) / max(float(state["games"]), 1.0)) if state["games"] else 0.5,
            "run_diff_pg": team_runs_scored(state) - team_runs_allowed(state),
            "home_win_pct": 0.5,
            "away_win_pct": 0.5,
        }
    return snapshot


def simulate_historical_game(
    game: HistoricalGame,
    team_form: dict[str, dict],
    n_simulations: int,
    seed: int,
) -> dict:
    away_mean, home_mean = expected_mlb_runs(
        game.away_code,
        game.home_code,
        team_form,
        {
            "venue": {"name": game.venue_name},
            "game_time": game.official_date,
        },
    )
    rng = np.random.default_rng(seed)
    away_scores = rng.poisson(lam=away_mean, size=n_simulations)
    home_scores = rng.poisson(lam=home_mean, size=n_simulations)
    tie_mask = away_scores == home_scores
    while np.any(tie_mask):
        away_scores[tie_mask] += rng.poisson(lam=0.6, size=int(np.sum(tie_mask)))
        home_scores[tie_mask] += rng.poisson(lam=0.6, size=int(np.sum(tie_mask)))
        tie_mask = away_scores == home_scores
    away_win_prob = float(np.mean(away_scores > home_scores))
    home_win_prob = 1.0 - away_win_prob
    projected_total = float(np.mean(away_scores + home_scores))
    return {
        "away_mean": away_mean,
        "home_mean": home_mean,
        "away_win_prob": away_win_prob,
        "home_win_prob": home_win_prob,
        "projected_total": projected_total,
    }


def brier_score(probability: float, outcome: bool) -> float:
    target = 1.0 if outcome else 0.0
    return (probability - target) ** 2


def log_loss(probability: float, outcome: bool) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    target = 1.0 if outcome else 0.0
    return -((target * math.log(clipped)) + ((1.0 - target) * math.log(1.0 - clipped)))


def calibration_bucket(probability: float) -> str:
    lower = math.floor(probability * 10.0) / 10.0
    upper = min(lower + 0.1, 1.0)
    return f"{lower:.1f}-{upper:.1f}"


def summarize_buckets(rows: list[dict]) -> list[dict]:
    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        label = calibration_bucket(float(row["away_win_prob"]))
        bucket = buckets.setdefault(label, {"count": 0, "prob_sum": 0.0, "hit_sum": 0.0})
        bucket["count"] += 1
        bucket["prob_sum"] += float(row["away_win_prob"])
        bucket["hit_sum"] += 1.0 if row["away_win_hit"] else 0.0
    summary = []
    for label in sorted(buckets):
        bucket = buckets[label]
        count = int(bucket["count"])
        summary.append(
            {
                "bucket": label,
                "count": count,
                "avg_prob": bucket["prob_sum"] / count,
                "actual_rate": bucket["hit_sum"] / count,
            }
        )
    return summary


def write_outputs(run_name: str, game_rows: list[dict], summary: dict) -> None:
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (run_dir / "games.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(game_rows[0].keys()) if game_rows else ["date"])
        writer.writeheader()
        for row in game_rows:
            writer.writerow(row)


def run_backtest(
    start_date: str,
    end_date: str,
    n_simulations: int,
    seed: int,
) -> dict:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    team_stats = defaultdict(default_team_state)
    game_rows: list[dict] = []

    for current_date in daterange(start, end):
        games = fetch_historical_schedule(current_date.isoformat())
        if not games:
            continue
        team_form = build_team_form_snapshot(team_stats)
        for game in games:
            result = simulate_historical_game(
                game,
                team_form,
                n_simulations=n_simulations,
                seed=seed + abs(hash((game.official_date, game.matchup))) % (2**32),
            )
            away_win_hit = game.away_score > game.home_score
            projected_winner = game.away_code if result["away_win_prob"] >= 0.5 else game.home_code
            game_rows.append(
                {
                    "date": game.official_date,
                    "matchup": game.matchup,
                    "away_code": game.away_code,
                    "home_code": game.home_code,
                    "actual_away_score": game.away_score,
                    "actual_home_score": game.home_score,
                    "actual_total": game.total_runs,
                    "away_mean": round(result["away_mean"], 4),
                    "home_mean": round(result["home_mean"], 4),
                    "away_win_prob": round(result["away_win_prob"], 4),
                    "home_win_prob": round(result["home_win_prob"], 4),
                    "projected_total": round(result["projected_total"], 4),
                    "away_win_hit": int(away_win_hit),
                    "winner_correct": int(projected_winner == game.winner_code),
                    "brier_away_ml": round(brier_score(result["away_win_prob"], away_win_hit), 6),
                    "log_loss_away_ml": round(log_loss(result["away_win_prob"], away_win_hit), 6),
                    "total_abs_error": round(abs(result["projected_total"] - game.total_runs), 4),
                }
            )
        update_team_stats(team_stats, games)

    if not game_rows:
        raise RuntimeError("No completed historical games found for the selected date range")

    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "n_simulations": n_simulations,
        "games_processed": len(game_rows),
        "winner_accuracy": sum(int(row["winner_correct"]) for row in game_rows) / len(game_rows),
        "away_ml_brier": sum(float(row["brier_away_ml"]) for row in game_rows) / len(game_rows),
        "away_ml_log_loss": sum(float(row["log_loss_away_ml"]) for row in game_rows) / len(game_rows),
        "total_mae": sum(float(row["total_abs_error"]) for row in game_rows) / len(game_rows),
        "avg_projected_total": sum(float(row["projected_total"]) for row in game_rows) / len(game_rows),
        "avg_actual_total": sum(float(row["actual_total"]) for row in game_rows) / len(game_rows),
        "calibration_buckets": summarize_buckets(game_rows),
    }
    return {"summary": summary, "game_rows": game_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical 2025 MLB Monte Carlo backtest")
    parser.add_argument("--start-date", default="2025-04-01")
    parser.add_argument("--end-date", default="2025-04-30")
    parser.add_argument("--n-simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        n_simulations=args.n_simulations,
        seed=args.seed,
    )
    run_name = args.run_name or f"mc_{args.start_date}_to_{args.end_date}".replace("-", "")
    write_outputs(run_name, result["game_rows"], result["summary"])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
