import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests

from quantum_parlay_oracle import (
    Leg,
    build_coupling_matrix,
    build_greedy_parlay,
    compute_biases,
    gibbs_sample,
)


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
RESULTS_DIR = Path("results") / "phase3_backtest"
DEFAULT_BETAS = [0.8, 1.0, 1.2, 1.5, 2.0]
LEAGUE_AVG_RUNS = 4.5
STAT_PRIOR_GAMES = 8.0
MIN_TOTAL_LINE = 6.5
MAX_TOTAL_LINE = 11.5
INITIAL_ELO = 1500.0
HOME_FIELD_ELO = 35.0
ELO_K = 20.0
RUN_DIFF_SCALE = 3.0
HOME_FIELD_RUNS = 0.2
DEFAULT_COUPLING_PRIOR = 12.0
DEFAULT_MAX_COUPLING_MAGNITUDE = 1.5
RECENT_ALPHA = 0.15

TEAM_CODES = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


@dataclass(frozen=True)
class HistoricalGame:
    official_date: str
    away_name: str
    home_name: str
    away_code: str
    home_code: str
    away_score: int
    home_score: int
    venue_name: str = ""

    @property
    def matchup(self) -> str:
        return f"{self.away_code}@{self.home_code}"

    @property
    def total_runs(self) -> int:
        return self.away_score + self.home_score

    @property
    def winner_code(self) -> str:
        return self.away_code if self.away_score > self.home_score else self.home_code


class PseudoEntropySource:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.total_consumed = 0
        self.source = f"NumPy PRNG (seed={seed})"

    def next_float(self) -> float:
        self.total_consumed += 1
        return float(self.rng.random())

    def next_floats(self, n: int) -> np.ndarray:
        self.total_consumed += n
        return self.rng.random(n)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def fetch_historical_schedule(target_date: str) -> list[HistoricalGame]:
    last_error = None
    payload = None
    for attempt in range(3):
        try:
            response = requests.get(
                MLB_SCHEDULE_URL,
                params={"sportId": 1, "date": target_date},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 2:
                raise
    if payload is None:
        raise RuntimeError(f"Unable to load MLB schedule for {target_date}: {last_error}")
    games = []

    for date_block in payload.get("dates", []):
        for item in date_block.get("games", []):
            away = item["teams"]["away"]
            home = item["teams"]["home"]
            away_score = away.get("score")
            home_score = home.get("score")
            if away_score is None or home_score is None:
                continue
            away_name = away["team"]["name"]
            home_name = home["team"]["name"]
            away_code = TEAM_CODES.get(away_name)
            home_code = TEAM_CODES.get(home_name)
            if away_code is None or home_code is None:
                continue
            games.append(
                HistoricalGame(
                    official_date=item["officialDate"],
                    away_name=away_name,
                    home_name=home_name,
                    away_code=away_code,
                    home_code=home_code,
                    away_score=int(away_score),
                    home_score=int(home_score),
                    venue_name=item.get("venue", {}).get("name", ""),
                )
            )
    return games


def daterange(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def default_team_state():
    return {
        "games": 0,
        "wins": 0,
        "runs_scored": 0,
        "runs_allowed": 0,
        "elo": INITIAL_ELO,
        "recent_win_rate": 0.5,
        "recent_run_diff": 0.0,
        "recent_total_runs": LEAGUE_AVG_RUNS * 2.0,
    }


def team_win_pct(state: dict) -> float:
    return (state["wins"] + (STAT_PRIOR_GAMES * 0.5)) / (state["games"] + STAT_PRIOR_GAMES)


def team_runs_scored(state: dict) -> float:
    return (state["runs_scored"] + (STAT_PRIOR_GAMES * LEAGUE_AVG_RUNS)) / (state["games"] + STAT_PRIOR_GAMES)


def team_runs_allowed(state: dict) -> float:
    return (state["runs_allowed"] + (STAT_PRIOR_GAMES * LEAGUE_AVG_RUNS)) / (state["games"] + STAT_PRIOR_GAMES)


def team_run_diff_per_game(state: dict) -> float:
    return team_runs_scored(state) - team_runs_allowed(state)


def elo_expected(away_elo: float, home_elo: float) -> tuple[float, float]:
    away_expected = 1.0 / (1.0 + (10.0 ** (((home_elo + HOME_FIELD_ELO) - away_elo) / 400.0)))
    return away_expected, 1.0 - away_expected


def elo_update_margin_multiplier(run_diff: int) -> float:
    return math.log(max(run_diff, 1) + 1.0) * (2.2 / ((run_diff * 0.001) + 2.2))


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    pmf = math.exp(-lam)
    cdf = pmf
    for value in range(1, k + 1):
        pmf *= lam / value
        cdf += pmf
    return min(max(cdf, 0.0), 1.0)


def poisson_total_probs(total_line: float, total_lambda: float) -> tuple[float, float]:
    over_prob = 1.0 - poisson_cdf(math.floor(total_line), total_lambda)
    under_prob = poisson_cdf(math.ceil(total_line) - 1, total_lambda)
    return max(over_prob, 0.0), max(under_prob, 0.0)


def synthesize_legs(games: list[HistoricalGame], team_stats: dict) -> tuple[list[Leg], dict]:
    legs = []
    outcomes = {}
    leg_id = 0

    for game in games:
        away_state = team_stats[game.away_code]
        home_state = team_stats[game.home_code]

        away_prob, home_prob = elo_expected(float(away_state["elo"]), float(home_state["elo"]))

        away_lambda = max(2.0, (team_runs_scored(away_state) + team_runs_allowed(home_state)) / 2.0)
        home_lambda = max(2.0, ((team_runs_scored(home_state) + team_runs_allowed(away_state)) / 2.0) + HOME_FIELD_RUNS)
        projected_total = away_lambda + home_lambda
        total_line = round(projected_total * 2.0) / 2.0
        total_line = min(MAX_TOTAL_LINE, max(MIN_TOTAL_LINE, total_line))
        over_prob, under_prob = poisson_total_probs(total_line, projected_total)

        away_leg = Leg(leg_id, f"{game.away_code} ML", "ml", game.matchup, away_prob, "Phase 3 Elo prior", "mlb")
        outcomes[leg_id] = game.winner_code == game.away_code
        legs.append(away_leg)
        leg_id += 1

        home_leg = Leg(leg_id, f"{game.home_code} ML", "ml", game.matchup, home_prob, "Phase 3 Elo prior", "mlb")
        outcomes[leg_id] = game.winner_code == game.home_code
        legs.append(home_leg)
        leg_id += 1

        over_leg = Leg(leg_id, f"{game.matchup} O{total_line:g}", "total", game.matchup, over_prob, "Phase 3 Poisson total", "mlb")
        outcomes[leg_id] = game.total_runs > total_line
        legs.append(over_leg)
        leg_id += 1

        under_leg = Leg(leg_id, f"{game.matchup} U{total_line:g}", "total", game.matchup, under_prob, "Phase 3 Poisson total", "mlb")
        outcomes[leg_id] = game.total_runs < total_line
        legs.append(under_leg)
        leg_id += 1

    return legs, outcomes


def leg_total_line(leg: Leg) -> float | None:
    if leg.category != "total":
        return None
    match = math.nan
    import re
    parsed = re.search(r"[OU](\d+(?:\.\d+)?)", leg.label)
    if not parsed:
        return None
    return float(parsed.group(1))


def leg_selected_team(leg: Leg) -> str | None:
    if leg.category != "ml":
        return None
    return leg.label.split()[0]


def game_lookup(games: list[HistoricalGame]) -> dict[str, HistoricalGame]:
    return {game.matchup: game for game in games}


def residual_adjustment(leg: Leg, game: HistoricalGame, team_stats: dict) -> float:
    if leg.category == "ml":
        selected = leg_selected_team(leg)
        if selected is None:
            return 0.0
        opponent = game.home_code if selected == game.away_code else game.away_code
        selected_state = team_stats[selected]
        opponent_state = team_stats[opponent]
        win_edge = team_win_pct(selected_state) - team_win_pct(opponent_state)
        run_edge = team_run_diff_per_game(selected_state) - team_run_diff_per_game(opponent_state)
        recent_edge = float(selected_state["recent_win_rate"]) - float(opponent_state["recent_win_rate"])
        momentum_edge = float(selected_state["recent_run_diff"]) - float(opponent_state["recent_run_diff"])
        adjustment = (
            0.045 * math.tanh(win_edge / 0.12)
            + 0.035 * math.tanh(run_edge / 1.25)
            + 0.02 * math.tanh(recent_edge / 0.10)
            + 0.015 * math.tanh(momentum_edge / 1.5)
        )
        return adjustment

    if leg.category == "total":
        line = leg_total_line(leg)
        if line is None:
            return 0.0
        away_state = team_stats[game.away_code]
        home_state = team_stats[game.home_code]
        away_lambda = max(
            2.0,
            (
                team_runs_scored(away_state)
                + float(home_state["recent_total_runs"]) / 2.0
                + team_runs_allowed(home_state)
            )
            / 3.0,
        )
        home_lambda = max(
            2.0,
            (
                team_runs_scored(home_state)
                + float(away_state["recent_total_runs"]) / 2.0
                + team_runs_allowed(away_state)
            )
            / 3.0
            + HOME_FIELD_RUNS,
        )
        projected_total = away_lambda + home_lambda
        total_edge = projected_total - line
        recent_total_edge = (
            (float(away_state["recent_total_runs"]) + float(home_state["recent_total_runs"])) / 2.0
        ) - (LEAGUE_AVG_RUNS * 2.0)
        adjustment = (
            0.06 * math.tanh(total_edge / 1.2)
            + 0.025 * math.tanh(recent_total_edge / 2.0)
        )
        if " U" in f" {leg.label}":
            adjustment *= -1.0
        return adjustment

    return 0.0


def leg_tag(leg: Leg) -> str:
    if leg.category == "ml":
        side = leg.label.split()[0]
        away_code, home_code = leg.game.split("@")
        is_home = side == home_code
        role = "fav" if leg.implied_prob >= 0.5 else "dog"
        location = "home" if is_home else "away"
        return f"ml:{location}:{role}"
    if leg.category == "total":
        direction = "over" if " O" in f" {leg.label}" else "under"
        return f"total:{direction}"
    return f"prop:{leg.category}"


def pair_feature_key(first: Leg, second: Leg) -> tuple[str, str]:
    tags = sorted([leg_tag(first), leg_tag(second)])
    return tags[0], tags[1]


def default_coupling_stats() -> dict[str, float]:
    return {
        "recommended": 0,
        "both_hit": 0,
        "mixed": 0,
    }


def calibrated_coupling_adjustment(
    stats: dict[str, float],
    learning_rate: float,
    prior_strength: float,
    max_magnitude: float,
) -> float:
    evidence = float(stats["recommended"])
    if evidence <= 0:
        return 0.0
    net_signal = float(-stats["both_hit"])
    shrink = evidence / (evidence + prior_strength)
    adjustment = learning_rate * net_signal * shrink
    return float(max(-max_magnitude, min(max_magnitude, adjustment)))


def build_coupling_adjustments(
    coupling_stats: dict,
    learning_rate: float,
    prior_strength: float,
    max_magnitude: float,
) -> dict:
    adjustments = {}
    for key, stats in coupling_stats.items():
        value = calibrated_coupling_adjustment(stats, learning_rate, prior_strength, max_magnitude)
        if abs(value) > 1e-9:
            adjustments[key] = value
    return adjustments


def apply_coupling_adjustments(legs: list[Leg], base: np.ndarray, adjustments: dict) -> np.ndarray:
    matrix = base.copy()
    for i, first in enumerate(legs):
        for j in range(i + 1, len(legs)):
            second = legs[j]
            delta = adjustments.get(pair_feature_key(first, second), 0.0)
            if delta:
                matrix[i, j] += delta
                matrix[j, i] += delta
    return matrix


def run_custom_ensemble(
    legs: list[Leg],
    coupling_adjustments: dict,
    entropy,
    samples_per_beta: int,
    warmup: int,
    thin: int,
) -> np.ndarray:
    biases = compute_biases(legs)
    base = build_coupling_matrix(legs)
    coupling = apply_coupling_adjustments(legs, base, coupling_adjustments)
    all_samples = []
    for beta in DEFAULT_BETAS:
        all_samples.append(
            gibbs_sample(
                biases=biases,
                coupling=coupling,
                beta=beta,
                qrng=entropy,
                n_warmup=warmup,
                n_samples=samples_per_beta,
                thin=thin,
            )
        )
    return np.concatenate(all_samples, axis=0)


def eligible_indices_for_strategy(legs: list[Leg], requested_size: int, strategy: str) -> list[int]:
    if strategy == "totals_focus" and requested_size == 3:
        return [idx for idx, leg in enumerate(legs) if leg.category == "total"]
    return list(range(len(legs)))


def remapped_sublegs(legs: list[Leg], eligible: list[int]) -> list[Leg]:
    return [replace(legs[idx], id=sub_idx) for sub_idx, idx in enumerate(eligible)]


def build_strategy_parlay(
    legs: list[Leg],
    activation: np.ndarray,
    co_activation: np.ndarray,
    requested_size: int,
    strategy: str,
) -> list[int]:
    eligible = eligible_indices_for_strategy(legs, requested_size, strategy)
    if not eligible:
        return []
    sub_legs = remapped_sublegs(legs, eligible)
    sub_activation = activation[eligible]
    sub_co = co_activation[np.ix_(eligible, eligible)]
    sub_parlay = build_greedy_parlay(sub_legs, sub_activation, sub_co, requested_size)
    return [eligible[idx] for idx in sub_parlay]


def direct_pair_bonus(first: Leg, second: Leg, mode: str) -> float:
    if first.game == second.game and first.category == second.category:
        return -0.95

    bonus = 0.0
    first_tag = leg_tag(first)
    second_tag = leg_tag(second)
    tags = {first_tag, second_tag}

    if mode == "heuristic":
        if first.category == "total" and second.category == "total":
            if tags == {"total:under"}:
                bonus += 0.08
            elif tags == {"total:over"}:
                bonus += 0.03
            else:
                bonus -= 0.04
        if first.category == "ml" and second.category == "ml":
            if "ml:away:dog" in tags and len(tags) == 1:
                bonus += 0.04
            if "ml:home:fav" in tags and len(tags) == 1:
                bonus += 0.02
        if tags == {"ml:home:fav", "total:under"}:
            bonus += 0.05
        if tags == {"ml:away:dog", "total:over"}:
            bonus += 0.04

    return bonus


def direct_activation_and_coactivation(
    legs: list[Leg],
    score_source: str,
    games: list[HistoricalGame] | None = None,
    team_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    implied = np.array([float(leg.implied_prob) for leg in legs], dtype=np.float64)
    activation = implied.copy()

    if score_source == "heuristic":
        bias_bonus = -compute_biases(legs) * 0.35
        activation = activation + bias_bonus
        for idx, leg in enumerate(legs):
            if leg.category == "total" and " U" in f" {leg.label}":
                activation[idx] += 0.03
            elif leg.category == "total" and " O" in f" {leg.label}":
                activation[idx] += 0.01
            elif leg.category == "ml" and leg.implied_prob < 0.5:
                activation[idx] += 0.02
    elif score_source == "residual":
        lookup = game_lookup(games or [])
        stats = team_stats or {}
        for idx, leg in enumerate(legs):
            game = lookup.get(leg.game)
            if game is None:
                continue
            activation[idx] += residual_adjustment(leg, game, stats)

    activation = np.clip(activation, 0.02, 0.98)
    co_activation = np.outer(activation, activation)

    for i, first in enumerate(legs):
        for j in range(i + 1, len(legs)):
            second = legs[j]
            bonus = direct_pair_bonus(first, second, score_source)
            if bonus:
                adjusted = float(np.clip(co_activation[i, j] + bonus, 0.0, 1.0))
                co_activation[i, j] = adjusted
                co_activation[j, i] = adjusted

    np.fill_diagonal(co_activation, activation)
    return activation, co_activation


def activation_and_parlays(
    legs: list[Leg],
    samples: np.ndarray,
    strategy: str,
) -> tuple[np.ndarray, list[list[int]]]:
    binary = (samples + 1.0) / 2.0
    activation = np.mean(binary, axis=0)
    co_activation = (binary.T @ binary) / samples.shape[0]
    parlays = [
        build_strategy_parlay(legs, activation, co_activation, requested_size, strategy)
        for requested_size in (3, 4, 5)
    ]
    return activation, parlays


def direct_activation_and_parlays(
    legs: list[Leg],
    strategy: str,
    score_source: str,
    games: list[HistoricalGame] | None = None,
    team_stats: dict | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    activation, co_activation = direct_activation_and_coactivation(
        legs,
        score_source,
        games=games,
        team_stats=team_stats,
    )
    parlays = [
        build_strategy_parlay(legs, activation, co_activation, requested_size, strategy)
        for requested_size in (3, 4, 5)
    ]
    return activation, parlays


def parlay_hit(parlay: list[int], outcomes: dict[int, bool]) -> bool:
    return bool(parlay) and all(outcomes[idx] for idx in parlay)


def parlay_market_probability(parlay: list[int], legs: list[Leg]) -> float:
    if not parlay:
        return 0.0
    probability = 1.0
    for idx in parlay:
        probability *= float(legs[idx].implied_prob)
    return probability


def parlay_payout_estimate(parlay: list[int], legs: list[Leg]) -> float | None:
    implied = parlay_market_probability(parlay, legs)
    if implied <= 0.0:
        return None
    return 1.0 / implied


def parlay_profit_units(parlay: list[int], legs: list[Leg], outcomes: dict[int, bool]) -> float:
    payout = parlay_payout_estimate(parlay, legs)
    if payout is None:
        return 0.0
    return (payout - 1.0) if parlay_hit(parlay, outcomes) else -1.0


def random_baseline(legs: list[Leg], rng: np.random.Generator, size: int) -> list[int]:
    if len(legs) <= size:
        return list(range(len(legs)))
    choices = rng.choice(len(legs), size=size, replace=False)
    return [int(idx) for idx in choices]


def top_implied_baseline(legs: list[Leg], size: int) -> list[int]:
    ranked = sorted(range(len(legs)), key=lambda idx: legs[idx].implied_prob, reverse=True)
    return ranked[: min(size, len(ranked))]


def top_edge_baseline(legs: list[Leg], activation: np.ndarray, size: int) -> list[int]:
    ranked = sorted(
        range(len(legs)),
        key=lambda idx: float(activation[idx]) - float(legs[idx].implied_prob),
        reverse=True,
    )
    return ranked[: min(size, len(ranked))]


def filtered_random_baseline(
    legs: list[Leg],
    rng: np.random.Generator,
    size: int,
    strategy: str,
) -> list[int]:
    eligible = eligible_indices_for_strategy(legs, size, strategy)
    if len(eligible) <= size:
        return eligible
    picks = rng.choice(eligible, size=size, replace=False)
    return [int(idx) for idx in picks]


def filtered_implied_baseline(
    legs: list[Leg],
    size: int,
    strategy: str,
) -> list[int]:
    eligible = eligible_indices_for_strategy(legs, size, strategy)
    ranked = sorted(eligible, key=lambda idx: legs[idx].implied_prob, reverse=True)
    return ranked[: min(size, len(ranked))]


def filtered_edge_baseline(
    legs: list[Leg],
    activation: np.ndarray,
    size: int,
    strategy: str,
) -> list[int]:
    eligible = eligible_indices_for_strategy(legs, size, strategy)
    ranked = sorted(
        eligible,
        key=lambda idx: float(activation[idx]) - float(legs[idx].implied_prob),
        reverse=True,
    )
    return ranked[: min(size, len(ranked))]


def update_team_stats(team_stats: dict, games: list[HistoricalGame]) -> None:
    for game in games:
        away = team_stats[game.away_code]
        home = team_stats[game.home_code]

        away["games"] += 1
        home["games"] += 1
        away["runs_scored"] += game.away_score
        away["runs_allowed"] += game.home_score
        home["runs_scored"] += game.home_score
        home["runs_allowed"] += game.away_score

        if game.away_score > game.home_score:
            away["wins"] += 1
        else:
            home["wins"] += 1

        away_win = 1.0 if game.away_score > game.home_score else 0.0
        home_win = 1.0 - away_win
        total_runs = float(game.away_score + game.home_score)
        away_run_diff = float(game.away_score - game.home_score)
        home_run_diff = -away_run_diff
        away["recent_win_rate"] = ((1.0 - RECENT_ALPHA) * float(away["recent_win_rate"])) + (RECENT_ALPHA * away_win)
        home["recent_win_rate"] = ((1.0 - RECENT_ALPHA) * float(home["recent_win_rate"])) + (RECENT_ALPHA * home_win)
        away["recent_run_diff"] = ((1.0 - RECENT_ALPHA) * float(away["recent_run_diff"])) + (RECENT_ALPHA * away_run_diff)
        home["recent_run_diff"] = ((1.0 - RECENT_ALPHA) * float(home["recent_run_diff"])) + (RECENT_ALPHA * home_run_diff)
        away["recent_total_runs"] = ((1.0 - RECENT_ALPHA) * float(away["recent_total_runs"])) + (RECENT_ALPHA * total_runs)
        home["recent_total_runs"] = ((1.0 - RECENT_ALPHA) * float(home["recent_total_runs"])) + (RECENT_ALPHA * total_runs)

        away_expected, home_expected = elo_expected(float(away["elo"]), float(home["elo"]))
        away_actual = 1.0 if game.away_score > game.home_score else 0.0
        home_actual = 1.0 - away_actual
        margin = abs(game.away_score - game.home_score)
        multiplier = elo_update_margin_multiplier(margin)
        elo_delta = ELO_K * multiplier
        away["elo"] += elo_delta * (away_actual - away_expected)
        home["elo"] += elo_delta * (home_actual - home_expected)


def write_outputs(run_name: str, daily_rows: list[dict], summary: dict, oracle_leg_rows: list[dict]) -> None:
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    daily_csv = run_dir / "daily_results.csv"
    with daily_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily_rows[0].keys()) if daily_rows else ["date"])
        writer.writeheader()
        for row in daily_rows:
            writer.writerow(row)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    leg_csv = run_dir / "oracle_legs.csv"
    with leg_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(oracle_leg_rows[0].keys()) if oracle_leg_rows else ["date"],
        )
        writer.writeheader()
        for row in oracle_leg_rows:
            writer.writerow(row)

    coupling_rows = summary.get("coupling_report", [])
    coupling_csv = run_dir / "coupling_report.csv"
    with coupling_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(coupling_rows[0].keys()) if coupling_rows else ["pair_key"],
        )
        writer.writeheader()
        for row in coupling_rows:
            writer.writerow(row)


def run_backtest(
    start_date: str,
    end_date: str,
    samples_per_beta: int,
    warmup: int,
    thin: int,
    learning_rate: float,
    seed: int,
    strategy: str,
    coupling_prior: float,
    max_coupling_magnitude: float,
    score_source: str,
) -> dict:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    team_stats = defaultdict(default_team_state)
    coupling_stats = defaultdict(default_coupling_stats)
    coupling_adjustments = {}
    baseline_rng = np.random.default_rng(seed)
    daily_rows = []

    oracle_hits = {3: 0, 4: 0, 5: 0}
    random_hits = {3: 0, 4: 0, 5: 0}
    implied_hits = {3: 0, 4: 0, 5: 0}
    edge_hits = {3: 0, 4: 0, 5: 0}
    oracle_profit = {3: 0.0, 4: 0.0, 5: 0.0}
    random_profit = {3: 0.0, 4: 0.0, 5: 0.0}
    implied_profit = {3: 0.0, 4: 0.0, 5: 0.0}
    edge_profit = {3: 0.0, 4: 0.0, 5: 0.0}
    oracle_legs = []

    for current_date in daterange(start, end):
        games = fetch_historical_schedule(current_date.isoformat())
        if not games:
            continue

        legs, outcomes = synthesize_legs(games, team_stats)
        if score_source == "ising":
            entropy = PseudoEntropySource(seed + int(current_date.strftime("%j")))
            samples = run_custom_ensemble(
                legs=legs,
                coupling_adjustments=coupling_adjustments,
                entropy=entropy,
                samples_per_beta=samples_per_beta,
                warmup=warmup,
                thin=thin,
            )
            activation, parlays = activation_and_parlays(legs, samples, strategy)
        else:
            activation, parlays = direct_activation_and_parlays(
                legs,
                strategy,
                score_source,
                games=games,
                team_stats=team_stats,
            )

        recommended_pairs = set()
        row = {
            "date": current_date.isoformat(),
            "games": len(games),
            "legs": len(legs),
        }

        for size, parlay in zip((3, 4, 5), parlays):
            hit = parlay_hit(parlay, outcomes)
            random_parlay = filtered_random_baseline(legs, baseline_rng, size, strategy)
            implied_parlay = filtered_implied_baseline(legs, size, strategy)
            edge_parlay = filtered_edge_baseline(legs, activation, size, strategy)
            random_hit = parlay_hit(random_parlay, outcomes)
            implied_hit = parlay_hit(implied_parlay, outcomes)
            edge_hit = parlay_hit(edge_parlay, outcomes)

            oracle_hits[size] += int(hit)
            random_hits[size] += int(random_hit)
            implied_hits[size] += int(implied_hit)
            edge_hits[size] += int(edge_hit)
            oracle_profit[size] += parlay_profit_units(parlay, legs, outcomes)
            random_profit[size] += parlay_profit_units(random_parlay, legs, outcomes)
            implied_profit[size] += parlay_profit_units(implied_parlay, legs, outcomes)
            edge_profit[size] += parlay_profit_units(edge_parlay, legs, outcomes)
            row[f"oracle_{size}_hit"] = int(hit)
            row[f"random_{size}_hit"] = int(random_hit)
            row[f"implied_{size}_hit"] = int(implied_hit)
            row[f"edge_{size}_hit"] = int(edge_hit)
            row[f"oracle_{size}_profit"] = oracle_profit[size]
            row[f"random_{size}_profit"] = random_profit[size]
            row[f"implied_{size}_profit"] = implied_profit[size]
            row[f"edge_{size}_profit"] = edge_profit[size]
            row[f"oracle_{size}_daily_profit"] = parlay_profit_units(parlay, legs, outcomes)
            row[f"random_{size}_daily_profit"] = parlay_profit_units(random_parlay, legs, outcomes)
            row[f"implied_{size}_daily_profit"] = parlay_profit_units(implied_parlay, legs, outcomes)
            row[f"edge_{size}_daily_profit"] = parlay_profit_units(edge_parlay, legs, outcomes)
            row[f"oracle_{size}_legs"] = " | ".join(legs[idx].label for idx in parlay)
            row[f"random_{size}_legs"] = " | ".join(legs[idx].label for idx in random_parlay)
            row[f"implied_{size}_legs"] = " | ".join(legs[idx].label for idx in implied_parlay)
            row[f"edge_{size}_legs"] = " | ".join(legs[idx].label for idx in edge_parlay)

            for idx in parlay:
                oracle_legs.append(
                    {
                        "date": current_date.isoformat(),
                        "parlay_size": size,
                        "label": legs[idx].label,
                        "category": legs[idx].category,
                        "game": legs[idx].game,
                        "activation": float(activation[idx]),
                        "implied_prob": float(legs[idx].implied_prob),
                        "edge": float(activation[idx]) - float(legs[idx].implied_prob),
                        "hit": int(outcomes[idx]),
                    }
                )

            for i, first in enumerate(parlay):
                for second in parlay[i + 1 :]:
                    recommended_pairs.add((first, second))

        for first_idx, second_idx in recommended_pairs:
            feature_key = pair_feature_key(legs[first_idx], legs[second_idx])
            stats = coupling_stats[feature_key]
            stats["recommended"] += 1
            first_hit = outcomes[first_idx]
            second_hit = outcomes[second_idx]
            if first_hit and second_hit:
                stats["both_hit"] += 1
            else:
                stats["mixed"] += 1

        coupling_adjustments = build_coupling_adjustments(
            coupling_stats=coupling_stats,
            learning_rate=learning_rate,
            prior_strength=coupling_prior,
            max_magnitude=max_coupling_magnitude,
        )

        daily_rows.append(row)
        update_team_stats(team_stats, games)

    leg_hit_rate = (
        sum(item["hit"] for item in oracle_legs) / len(oracle_legs) if oracle_legs else 0.0
    )
    avg_activation = (
        sum(item["activation"] for item in oracle_legs) / len(oracle_legs) if oracle_legs else 0.0
    )
    avg_implied = (
        sum(item["implied_prob"] for item in oracle_legs) / len(oracle_legs) if oracle_legs else 0.0
    )

    coupling_report = []
    for (left, right), stats in sorted(coupling_stats.items()):
        recommended = int(stats["recommended"])
        if recommended <= 0:
            continue
        both_hit = int(stats["both_hit"])
        mixed = int(stats["mixed"])
        adjustment = calibrated_coupling_adjustment(
            stats=stats,
            learning_rate=learning_rate,
            prior_strength=coupling_prior,
            max_magnitude=max_coupling_magnitude,
        )
        win_rate = both_hit / recommended
        coupling_report.append(
            {
                "pair_key": f"{left}|{right}",
                "recommended": recommended,
                "both_hit": both_hit,
                "mixed": mixed,
                "win_rate": win_rate,
                "shrink_factor": recommended / (recommended + coupling_prior),
                "adjustment": adjustment,
            }
        )

    coupling_report.sort(key=lambda row: abs(float(row["adjustment"])), reverse=True)

    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "strategy": strategy,
        "score_source": score_source,
        "days_processed": len(daily_rows),
        "oracle_hits": oracle_hits,
        "random_hits": random_hits,
        "implied_hits": implied_hits,
        "edge_hits": edge_hits,
        "oracle_profit_units": oracle_profit,
        "random_profit_units": random_profit,
        "implied_profit_units": implied_profit,
        "edge_profit_units": edge_profit,
        "oracle_roi": {size: (oracle_profit[size] / max(len(daily_rows), 1)) for size in oracle_profit},
        "random_roi": {size: (random_profit[size] / max(len(daily_rows), 1)) for size in random_profit},
        "implied_roi": {size: (implied_profit[size] / max(len(daily_rows), 1)) for size in implied_profit},
        "edge_roi": {size: (edge_profit[size] / max(len(daily_rows), 1)) for size in edge_profit},
        "oracle_leg_hit_rate": leg_hit_rate,
        "oracle_avg_activation": avg_activation,
        "oracle_avg_implied_prob": avg_implied,
        "coupling_prior": coupling_prior,
        "max_coupling_magnitude": max_coupling_magnitude,
        "learned_coupling_adjustments": {
            f"{left}|{right}": value
            for (left, right), value in sorted(coupling_adjustments.items())
            if abs(value) > 1e-9
        },
        "coupling_report": coupling_report,
    }
    return {"daily_rows": daily_rows, "summary": summary, "oracle_leg_rows": oracle_legs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 historical MLB backtest")
    parser.add_argument("--start-date", default="2025-04-01")
    parser.add_argument("--end-date", default="2025-04-07")
    parser.add_argument("--samples-per-beta", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--thin", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["standard", "totals_focus"], default="standard")
    parser.add_argument("--coupling-prior", type=float, default=DEFAULT_COUPLING_PRIOR)
    parser.add_argument("--max-coupling-magnitude", type=float, default=DEFAULT_MAX_COUPLING_MAGNITUDE)
    parser.add_argument("--score-source", choices=["ising", "implied", "heuristic", "residual"], default="ising")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        samples_per_beta=args.samples_per_beta,
        warmup=args.warmup,
        thin=args.thin,
        learning_rate=args.learning_rate,
        seed=args.seed,
        strategy=args.strategy,
        coupling_prior=args.coupling_prior,
        max_coupling_magnitude=args.max_coupling_magnitude,
        score_source=args.score_source,
    )
    run_name = args.run_name or f"{args.start_date}_to_{args.end_date}"
    write_outputs(run_name, result["daily_rows"], result["summary"], result["oracle_leg_rows"])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
