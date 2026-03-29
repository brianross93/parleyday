from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NBAPlayerProfile:
    name: str
    minutes: float
    points: float
    rebounds: float
    assists: float
    status: str = "active"
    games_sample: float = 1.0
    position: str | None = None


@dataclass(frozen=True)
class NBATeamContext:
    code: str
    players: list[NBAPlayerProfile]
    expected_points: float


@dataclass(frozen=True)
class NBAGameConfig:
    n_simulations: int = 900
    random_seed: int = 0
    min_possessions: int = 90
    max_possessions: int = 108


@dataclass(frozen=True)
class NBAPropDistribution:
    samples: np.ndarray
    mean: float
    percentiles: dict[int, float]
    over_probabilities: dict[float, float]


@dataclass(frozen=True)
class NBAGameSimulationResult:
    away_scores: np.ndarray
    home_scores: np.ndarray
    player_props: dict[tuple[str, str], NBAPropDistribution]


class NBAGameSimulator:
    def __init__(self, config: NBAGameConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def simulate_game(
        self,
        *,
        away: NBATeamContext,
        home: NBATeamContext,
        tracked_props: set[tuple[str, str]] | None = None,
    ) -> NBAGameSimulationResult:
        tracked_props = tracked_props or set()
        away_state = _PreparedTeam.from_context(away)
        home_state = _PreparedTeam.from_context(home)
        pace = self._shared_possessions(away.expected_points, home.expected_points)

        away_scores = np.zeros(self.config.n_simulations, dtype=np.int16)
        home_scores = np.zeros(self.config.n_simulations, dtype=np.int16)
        tracked_samples = {
            key: np.zeros(self.config.n_simulations, dtype=np.int16)
            for key in tracked_props
        }

        for sim_idx in range(self.config.n_simulations):
            possessions = int(
                np.clip(
                    np.rint(self.rng.normal(loc=pace, scale=2.8)),
                    self.config.min_possessions,
                    self.config.max_possessions,
                )
            )
            away_box = self._simulate_team_possessions(away_state, home_state, possessions)
            home_box = self._simulate_team_possessions(home_state, away_state, possessions)
            away_scores[sim_idx] = np.int16(away_box["__team__"]["points"])
            home_scores[sim_idx] = np.int16(home_box["__team__"]["points"])
            for key, samples in tracked_samples.items():
                player_name, stat = key
                samples[sim_idx] = np.int16(
                    away_box.get(player_name, {}).get(stat, 0) + home_box.get(player_name, {}).get(stat, 0)
                )

        player_props = {
            key: NBAPropDistribution(
                samples=samples,
                mean=float(np.mean(samples)),
                percentiles={p: float(np.percentile(samples, p)) for p in (25, 50, 75, 90)},
                over_probabilities={},
            )
            for key, samples in tracked_samples.items()
        }
        return NBAGameSimulationResult(
            away_scores=away_scores,
            home_scores=home_scores,
            player_props=player_props,
        )

    def _shared_possessions(self, away_points: float, home_points: float) -> float:
        implied_total = max(away_points + home_points, 180.0)
        baseline = implied_total / 2.22
        return float(np.clip(baseline, self.config.min_possessions, self.config.max_possessions))

    def _simulate_team_possessions(
        self,
        offense: "_PreparedTeam",
        defense: "_PreparedTeam",
        possessions: int,
    ) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {
            player.name: {"points": 0, "rebounds": 0, "assists": 0}
            for player in offense.players + defense.players
        }
        stats["__team__"] = {"points": 0, "rebounds": 0, "assists": 0}

        for _ in range(possessions):
            live = True
            while live:
                if self.rng.random() < offense.turnover_rate:
                    live = False
                    continue

                if self.rng.random() < offense.free_throw_trip_rate:
                    shooter = offense.pick_scorer(self.rng)
                    attempts = 3 if self.rng.random() < 0.08 else 2
                    made = int(self.rng.binomial(attempts, offense.free_throw_pct))
                    stats[shooter.name]["points"] += made
                    stats["__team__"]["points"] += made
                    live = False
                    continue

                shooter = offense.pick_scorer(self.rng)
                is_three = self.rng.random() < offense.three_point_rate
                make_pct = offense.three_point_pct if is_three else offense.two_point_pct
                points = 3 if is_three else 2
                if self.rng.random() < make_pct:
                    stats[shooter.name]["points"] += points
                    stats["__team__"]["points"] += points
                    if self.rng.random() < offense.assist_rate:
                        passer = offense.pick_assister(self.rng, exclude=shooter.name)
                        if passer is not None:
                            stats[passer.name]["assists"] += 1
                            stats["__team__"]["assists"] += 1
                    live = False
                elif self.rng.random() < offense.offensive_rebound_rate:
                    rebounder = offense.pick_rebounder(self.rng)
                    stats[rebounder.name]["rebounds"] += 1
                    stats["__team__"]["rebounds"] += 1
                else:
                    rebounder = defense.pick_rebounder(self.rng)
                    stats[rebounder.name]["rebounds"] += 1
                    live = False
        return stats


@dataclass(frozen=True)
class _PreparedTeam:
    code: str
    players: list[NBAPlayerProfile]
    scorer_weights: np.ndarray
    rebound_weights: np.ndarray
    assist_weights: np.ndarray
    turnover_rate: float
    free_throw_trip_rate: float
    free_throw_pct: float
    three_point_rate: float
    three_point_pct: float
    two_point_pct: float
    offensive_rebound_rate: float
    assist_rate: float

    @classmethod
    def from_context(cls, context: NBATeamContext) -> "_PreparedTeam":
        players = sorted(context.players, key=lambda player: player.minutes, reverse=True)[:10]
        if not players:
            players = [NBAPlayerProfile(name=f"{context.code} Placeholder", minutes=24.0, points=10.0, rebounds=4.0, assists=3.0)]

        scorer_weights = _normalize_weights([max(player.points, 0.35) * max(player.minutes, 10.0) ** 0.45 for player in players])
        rebound_weights = _normalize_weights([max(player.rebounds, 0.2) * max(player.minutes, 10.0) ** 0.35 for player in players])
        assist_weights = _normalize_weights([max(player.assists, 0.15) * max(player.minutes, 10.0) ** 0.35 for player in players])

        team_points = sum(player.points for player in players)
        team_rebounds = sum(player.rebounds for player in players)
        team_assists = sum(player.assists for player in players)
        star_share = float(np.max(scorer_weights))
        assist_quality = team_assists / max(team_points / 2.6, 1.0)
        rebound_strength = team_rebounds / max(len(players) * 4.8, 1.0)
        target_ppp = context.expected_points / max(float(np.clip((context.expected_points * 2.0) / 2.22, 90.0, 108.0)), 1.0)

        turnover_rate = float(np.clip(0.122 + (star_share - 0.27) * 0.06 - (assist_quality - 0.58) * 0.04, 0.095, 0.16))
        free_throw_trip_rate = float(np.clip(0.095 + (star_share - 0.25) * 0.05, 0.075, 0.14))
        three_point_rate = float(np.clip(0.37 + (assist_quality - 0.58) * 0.12, 0.30, 0.49))
        offensive_rebound_rate = float(np.clip(0.235 + (rebound_strength - 1.0) * 0.045, 0.18, 0.31))
        assist_rate = float(np.clip(0.54 + (assist_quality - 0.58) * 0.28, 0.44, 0.76))

        baseline_ppp = (1.0 - turnover_rate) * (
            free_throw_trip_rate * 1.52
            + (1.0 - free_throw_trip_rate) * ((three_point_rate * 1.08) + ((1.0 - three_point_rate) * 1.02))
            + offensive_rebound_rate * 0.22
        )
        efficiency_scale = float(np.clip(target_ppp / max(baseline_ppp, 0.75), 0.86, 1.18))

        three_point_pct = float(np.clip(0.355 * efficiency_scale**0.72, 0.29, 0.45))
        two_point_pct = float(np.clip(0.525 * efficiency_scale**0.70, 0.45, 0.63))
        free_throw_pct = float(np.clip(0.775 * efficiency_scale**0.18, 0.68, 0.87))

        return cls(
            code=context.code,
            players=players,
            scorer_weights=scorer_weights,
            rebound_weights=rebound_weights,
            assist_weights=assist_weights,
            turnover_rate=turnover_rate,
            free_throw_trip_rate=free_throw_trip_rate,
            free_throw_pct=free_throw_pct,
            three_point_rate=three_point_rate,
            three_point_pct=three_point_pct,
            two_point_pct=two_point_pct,
            offensive_rebound_rate=offensive_rebound_rate,
            assist_rate=assist_rate,
        )

    def pick_scorer(self, rng: np.random.Generator) -> NBAPlayerProfile:
        idx = int(rng.choice(len(self.players), p=self.scorer_weights))
        return self.players[idx]

    def pick_rebounder(self, rng: np.random.Generator) -> NBAPlayerProfile:
        idx = int(rng.choice(len(self.players), p=self.rebound_weights))
        return self.players[idx]

    def pick_assister(self, rng: np.random.Generator, exclude: str) -> NBAPlayerProfile | None:
        mask = np.array([player.name != exclude for player in self.players], dtype=bool)
        if not np.any(mask):
            return None
        weights = self.assist_weights[mask]
        weights = weights / weights.sum()
        candidates = [player for player in self.players if player.name != exclude]
        idx = int(rng.choice(len(candidates), p=weights))
        return candidates[idx]


def _normalize_weights(values: list[float]) -> np.ndarray:
    weights = np.array([max(float(value), 0.001) for value in values], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total
