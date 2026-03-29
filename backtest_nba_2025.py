import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

from quantum_parlay_oracle import expected_nba_points


RESULTS_DIR = Path("results") / "nba_monte_carlo_backtest"
ESPN_NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
NBA_BASELINE_POINTS = 112.0
NBA_HOME_COURT_POINTS = 2.5
INITIAL_NET_RATING = 0.0
RECENT_ALPHA = 0.18


@dataclass(frozen=True)
class HistoricalNBAGame:
    official_date: str
    away_name: str
    home_name: str
    away_code: str
    home_code: str
    away_score: int
    home_score: int

    @property
    def matchup(self) -> str:
        return f"{self.away_code}@{self.home_code}"

    @property
    def total_points(self) -> int:
        return self.away_score + self.home_score

    @property
    def winner_code(self) -> str:
        return self.away_code if self.away_score > self.home_score else self.home_code


def daterange(start_date: datetime.date, end_date: datetime.date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def default_team_state() -> dict:
    return {
        "games": 0,
        "wins": 0,
        "points_scored": 0,
        "points_allowed": 0,
        "recent_win_rate": 0.5,
        "recent_point_diff": 0.0,
        "recent_total_points": NBA_BASELINE_POINTS * 2.0,
        "net_rating_proxy": INITIAL_NET_RATING,
    }


def canonical_nba_code(team: dict) -> str:
    display_name = str(team.get("displayName", "")).strip()
    abbreviation = str(team.get("abbreviation", "")).strip()
    mapping = {
        "SA": "SAS",
        "GS": "GSW",
        "NO": "NOP",
        "UTAH": "UTA",
    }
    if abbreviation in mapping:
        return mapping[abbreviation]
    if abbreviation:
        return abbreviation
    words = display_name.split()
    if not words:
        return ""
    if display_name == "Philadelphia 76ers":
        return "PHI"
    if len(words) == 1:
        return words[0][:3].upper()
    return "".join(part[0] for part in words[:3]).upper()


def fetch_historical_nba_schedule(target_date: str) -> list[HistoricalNBAGame]:
    response = requests.get(
        ESPN_NBA_SCOREBOARD_URL,
        params={"dates": target_date.replace("-", "")},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    games = []
    for event in payload.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue
        competitors = competition.get("competitors", [])
        away = next((team for team in competitors if team.get("homeAway") == "away"), None)
        home = next((team for team in competitors if team.get("homeAway") == "home"), None)
        if not away or not home:
            continue
        away_score = away.get("score")
        home_score = home.get("score")
        if away_score is None or home_score is None:
            continue
        away_team = away.get("team", {})
        home_team = home.get("team", {})
        away_name = away_team.get("displayName", "")
        home_name = home_team.get("displayName", "")
        games.append(
            HistoricalNBAGame(
                official_date=target_date,
                away_name=away_name,
                home_name=home_name,
                away_code=canonical_nba_code(away_team),
                home_code=canonical_nba_code(home_team),
                away_score=int(away_score),
                home_score=int(home_score),
            )
        )
    return games


def build_team_form_snapshot(team_stats: dict) -> dict[str, dict]:
    snapshot = {}
    for code, state in team_stats.items():
        games = max(int(state["games"]), 1)
        snapshot[code] = {
            "win_pct": float(state["wins"]) / games if state["games"] else 0.5,
            "games_played": games,
            "net_rating_proxy": float(state["net_rating_proxy"]),
        }
    return snapshot


def simulate_historical_nba_game(
    game: HistoricalNBAGame,
    team_form: dict[str, dict],
    n_simulations: int,
    seed: int,
) -> dict:
    away_mean, home_mean = expected_nba_points(game.away_code, game.home_code, team_form, None)
    rng = np.random.default_rng(seed)
    away_std = max(10.0, away_mean * 0.11)
    home_std = max(10.0, home_mean * 0.11)
    away_scores = np.rint(rng.normal(loc=away_mean, scale=away_std, size=n_simulations)).astype(int)
    home_scores = np.rint(rng.normal(loc=home_mean, scale=home_std, size=n_simulations)).astype(int)
    away_scores = np.clip(away_scores, 75, 170)
    home_scores = np.clip(home_scores, 75, 170)
    tie_mask = away_scores == home_scores
    while np.any(tie_mask):
        away_scores[tie_mask] += rng.integers(3, 11, size=int(np.sum(tie_mask)))
        home_scores[tie_mask] += rng.integers(2, 10, size=int(np.sum(tie_mask)))
        tie_mask = away_scores == home_scores
    away_win_prob = float(np.mean(away_scores > home_scores))
    return {
        "away_mean": away_mean,
        "home_mean": home_mean,
        "away_win_prob": away_win_prob,
        "home_win_prob": 1.0 - away_win_prob,
        "projected_total": float(np.mean(away_scores + home_scores)),
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


def update_team_stats(team_stats: dict, games: list[HistoricalNBAGame]) -> None:
    for game in games:
        away = team_stats[game.away_code]
        home = team_stats[game.home_code]
        away["games"] += 1
        home["games"] += 1
        away["points_scored"] += game.away_score
        away["points_allowed"] += game.home_score
        home["points_scored"] += game.home_score
        home["points_allowed"] += game.away_score
        if game.away_score > game.home_score:
            away["wins"] += 1
        else:
            home["wins"] += 1
        away_win = 1.0 if game.away_score > game.home_score else 0.0
        home_win = 1.0 - away_win
        total_points = float(game.total_points)
        away_diff = float(game.away_score - game.home_score)
        home_diff = -away_diff
        away["recent_win_rate"] = ((1.0 - RECENT_ALPHA) * float(away["recent_win_rate"])) + (RECENT_ALPHA * away_win)
        home["recent_win_rate"] = ((1.0 - RECENT_ALPHA) * float(home["recent_win_rate"])) + (RECENT_ALPHA * home_win)
        away["recent_point_diff"] = ((1.0 - RECENT_ALPHA) * float(away["recent_point_diff"])) + (RECENT_ALPHA * away_diff)
        home["recent_point_diff"] = ((1.0 - RECENT_ALPHA) * float(home["recent_point_diff"])) + (RECENT_ALPHA * home_diff)
        away["recent_total_points"] = ((1.0 - RECENT_ALPHA) * float(away["recent_total_points"])) + (RECENT_ALPHA * total_points)
        home["recent_total_points"] = ((1.0 - RECENT_ALPHA) * float(home["recent_total_points"])) + (RECENT_ALPHA * total_points)

        away_games = max(int(away["games"]), 1)
        home_games = max(int(home["games"]), 1)
        away_net = (float(away["points_scored"]) - float(away["points_allowed"])) / away_games
        home_net = (float(home["points_scored"]) - float(home["points_allowed"])) / home_games
        away["net_rating_proxy"] = (0.65 * away_net) + (0.35 * float(away["recent_point_diff"]))
        home["net_rating_proxy"] = (0.65 * home_net) + (0.35 * float(home["recent_point_diff"]))


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


def run_backtest(start_date: str, end_date: str, n_simulations: int, seed: int) -> dict:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    team_stats = defaultdict(default_team_state)
    game_rows: list[dict] = []

    for current_date in daterange(start, end):
        games = fetch_historical_nba_schedule(current_date.isoformat())
        if not games:
            continue
        team_form = build_team_form_snapshot(team_stats)
        for game in games:
            result = simulate_historical_nba_game(
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
                    "actual_total": game.total_points,
                    "away_mean": round(result["away_mean"], 4),
                    "home_mean": round(result["home_mean"], 4),
                    "away_win_prob": round(result["away_win_prob"], 4),
                    "home_win_prob": round(result["home_win_prob"], 4),
                    "projected_total": round(result["projected_total"], 4),
                    "away_win_hit": int(away_win_hit),
                    "winner_correct": int(projected_winner == game.winner_code),
                    "brier_away_ml": round(brier_score(result["away_win_prob"], away_win_hit), 6),
                    "log_loss_away_ml": round(log_loss(result["away_win_prob"], away_win_hit), 6),
                    "total_abs_error": round(abs(result["projected_total"] - game.total_points), 4),
                }
            )
        update_team_stats(team_stats, games)

    if not game_rows:
        raise RuntimeError("No completed historical NBA games found for the selected date range")

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
    parser = argparse.ArgumentParser(description="Historical 2025 NBA Monte Carlo backtest")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--n-simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
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
    run_name = args.run_name or f"nba_mc_{args.start_date}_to_{args.end_date}".replace("-", "")
    write_outputs(run_name, result["game_rows"], result["summary"])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
