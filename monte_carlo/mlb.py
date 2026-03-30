from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable

import numpy as np


LEAGUE_RATES = {
    "k": 0.225,
    "bb": 0.085,
    "hbp": 0.010,
    "single": 0.155,
    "double": 0.045,
    "triple": 0.005,
    "hr": 0.032,
}
OUTCOME_ORDER = (
    "k",
    "bb",
    "hbp",
    "single",
    "double",
    "triple",
    "hr",
    "out",
)
HIT_EVENTS = {"single", "double", "triple", "hr"}


@dataclass(frozen=True)
class BatterProfile:
    player_id: str
    name: str
    hand: str
    pa_share: float
    strikeout_rate: float
    walk_rate: float
    hbp_rate: float
    single_rate: float
    double_rate: float
    triple_rate: float
    home_run_rate: float
    speed_factor: float = 1.0
    vs_left_factor: float = 1.0
    vs_right_factor: float = 1.0

    def event_rates(self) -> dict[str, float]:
        return {
            "k": self.strikeout_rate,
            "bb": self.walk_rate,
            "hbp": self.hbp_rate,
            "single": self.single_rate,
            "double": self.double_rate,
            "triple": self.triple_rate,
            "hr": self.home_run_rate,
        }


@dataclass(frozen=True)
class PitcherProfile:
    player_id: str
    name: str
    hand: str
    strikeout_rate: float
    walk_rate: float
    hbp_rate: float
    single_rate: float
    double_rate: float
    triple_rate: float
    home_run_rate: float
    fatigue_start: int = 75
    fatigue_full: int = 100

    def event_rates(self) -> dict[str, float]:
        return {
            "k": self.strikeout_rate,
            "bb": self.walk_rate,
            "hbp": self.hbp_rate,
            "single": self.single_rate,
            "double": self.double_rate,
            "triple": self.triple_rate,
            "hr": self.home_run_rate,
        }


@dataclass(frozen=True)
class TeamContext:
    team_code: str
    lineup: tuple[BatterProfile, ...]
    starter: PitcherProfile
    bullpen: tuple[PitcherProfile, ...] = ()
    park_hit_factor: float = 1.0
    park_hr_factor: float = 1.0
    offense_factor: float = 1.0


@dataclass(frozen=True)
class MLBGameConfig:
    n_simulations: int = 10_000
    innings: int = 9
    extras_runner_on_second: bool = True
    random_seed: int | None = None


@dataclass(frozen=True)
class MarketProp:
    player_name: str
    stat: str
    line: float
    side: str
    market_price: float
    event_id: str = ""
    market_source: str = "manual"


@dataclass(frozen=True)
class PropDistribution:
    player_name: str
    stat: str
    values: tuple[int, ...]
    mean: float
    median: float
    percentiles: dict[int, float]
    over_probabilities: dict[float, float]
    under_probabilities: dict[float, float]


@dataclass(frozen=True)
class EdgeEvaluation:
    player_name: str
    stat: str
    line: float
    side: str
    sim_probability: float
    market_price: float
    edge: float
    edge_pct: float
    ev_per_dollar: float
    confidence: float
    category: str
    event_id: str = ""
    market_source: str = "manual"


@dataclass(frozen=True)
class GameSimulationResult:
    home_team: str
    away_team: str
    home_scores: tuple[int, ...]
    away_scores: tuple[int, ...]
    winners: tuple[str, ...]
    player_props: dict[tuple[str, str], PropDistribution]


@dataclass
class _HalfInningState:
    outs: int = 0
    first: str | None = None
    second: str | None = None
    third: str | None = None


class MLBGameSimulator:
    def __init__(self, config: MLBGameConfig | None = None):
        self.config = config or MLBGameConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def simulate_game(
        self,
        away: TeamContext,
        home: TeamContext,
        market_props: Iterable[MarketProp] | None = None,
    ) -> GameSimulationResult:
        n = self.config.n_simulations
        away_scores: list[int] = []
        home_scores: list[int] = []
        winners: list[str] = []
        tracked_stats = {
            (player.name, stat): []
            for team in (away, home)
            for player in team.lineup
            for stat in ("hits", "home_runs", "walks", "strikeouts", "runs", "rbi", "total_bases", "plate_appearances")
        }
        for team in (away, home):
            for pitcher in (team.starter,) + tuple(team.bullpen):
                for stat in ("strikeouts", "walks", "hits_allowed", "earned_runs"):
                    tracked_stats[(pitcher.name, stat)] = []

        for _ in range(n):
            game = self._simulate_single_game(away, home)
            away_scores.append(game["away_score"])
            home_scores.append(game["home_score"])
            winners.append(game["winner"])
            for key, value in game["player_stats"].items():
                tracked_stats[key].append(value)

        player_props = {}
        requested_lines = self._market_lines_by_stat(market_props or [])
        for key, values in tracked_stats.items():
            player_name, stat = key
            values_tuple = tuple(int(v) for v in values)
            lines = requested_lines.get(key, ())
            player_props[key] = self._build_distribution(player_name, stat, values_tuple, lines)

        return GameSimulationResult(
            home_team=home.team_code,
            away_team=away.team_code,
            home_scores=tuple(home_scores),
            away_scores=tuple(away_scores),
            winners=tuple(winners),
            player_props=player_props,
        )

    def evaluate_edges(
        self,
        result: GameSimulationResult,
        market_props: Iterable[MarketProp],
    ) -> list[EdgeEvaluation]:
        edges = []
        for market in market_props:
            distribution = result.player_props.get((market.player_name, market.stat))
            if distribution is None:
                continue
            if market.side.lower() == "over":
                sim_prob = distribution.over_probabilities.get(market.line)
            else:
                sim_prob = distribution.under_probabilities.get(market.line)
            if sim_prob is None:
                continue
            edge = sim_prob - market.market_price
            confidence = self._confidence_from_distribution(distribution.values)
            edges.append(
                EdgeEvaluation(
                    player_name=market.player_name,
                    stat=market.stat,
                    line=market.line,
                    side=market.side,
                    sim_probability=sim_prob,
                    market_price=market.market_price,
                    edge=edge,
                    edge_pct=(edge / market.market_price) if market.market_price else 0.0,
                    ev_per_dollar=(sim_prob * (1.0 - market.market_price)) - ((1.0 - sim_prob) * market.market_price),
                    confidence=confidence,
                    category=self._categorize_edge(sim_prob, market.market_price, market.side),
                    event_id=market.event_id,
                    market_source=market.market_source,
                )
            )
        return sorted(edges, key=lambda item: abs(item.edge) * item.confidence, reverse=True)

    def _simulate_single_game(self, away: TeamContext, home: TeamContext) -> dict:
        scoreboard = {"away": 0, "home": 0}
        batter_index = {"away": 0, "home": 0}
        pitch_count = {
            pitcher.name: 0
            for team in (away, home)
            for pitcher in ((team.starter,) + tuple(team.bullpen))
        }
        staff_state = {
            "away": {"current": away.starter, "bullpen_index": 0},
            "home": {"current": home.starter, "bullpen_index": 0},
        }
        lines = {
            (player.name, stat): 0
            for team in (away, home)
            for player in team.lineup
            for stat in ("hits", "home_runs", "walks", "strikeouts", "runs", "rbi", "total_bases", "plate_appearances")
        }
        for team in (away, home):
            for pitcher in (team.starter,) + tuple(team.bullpen):
                for stat in ("strikeouts", "walks", "hits_allowed", "earned_runs"):
                    lines[(pitcher.name, stat)] = 0

        inning = 1
        while True:
            scoreboard["away"] += self._simulate_half_inning(
                offense=away,
                defense=home,
                batting_key="away",
                defense_key="home",
                inning=inning,
                batter_index=batter_index,
                pitch_count=pitch_count,
                staff_state=staff_state,
                player_lines=lines,
            )
            if inning >= self.config.innings and scoreboard["home"] > scoreboard["away"]:
                break

            scoreboard["home"] += self._simulate_half_inning(
                offense=home,
                defense=away,
                batting_key="home",
                defense_key="away",
                inning=inning,
                batter_index=batter_index,
                pitch_count=pitch_count,
                staff_state=staff_state,
                player_lines=lines,
                walk_off_enabled=inning >= self.config.innings,
                current_home_score=scoreboard["home"],
                current_away_score=scoreboard["away"],
            )
            if inning >= self.config.innings and scoreboard["home"] != scoreboard["away"]:
                break
            inning += 1

        winner = away.team_code if scoreboard["away"] > scoreboard["home"] else home.team_code
        return {
            "away_score": scoreboard["away"],
            "home_score": scoreboard["home"],
            "winner": winner,
            "player_stats": lines,
        }

    def _simulate_half_inning(
        self,
        offense: TeamContext,
        defense: TeamContext,
        batting_key: str,
        defense_key: str,
        inning: int,
        batter_index: dict[str, int],
        pitch_count: dict[str, int],
        staff_state: dict[str, dict[str, object]],
        player_lines: dict[tuple[str, str], int],
        walk_off_enabled: bool = False,
        current_home_score: int = 0,
        current_away_score: int = 0,
    ) -> int:
        state = _HalfInningState()
        if inning > self.config.innings and self.config.extras_runner_on_second:
            state.second = "__ghost_runner__"

        runs = 0
        while state.outs < 3:
            batter = offense.lineup[batter_index[batting_key] % len(offense.lineup)]
            batter_index[batting_key] += 1
            player_lines[(batter.name, "plate_appearances")] = player_lines.get((batter.name, "plate_appearances"), 0) + 1
            pitcher = self._current_pitcher(defense=defense, defense_key=defense_key, inning=inning, pitch_count=pitch_count, staff_state=staff_state)
            outcome = self._resolve_plate_appearance(
                batter=batter,
                pitcher=pitcher,
                park_hit_factor=offense.park_hit_factor,
                park_hr_factor=offense.park_hr_factor,
                offense_factor=offense.offense_factor,
                pitch_count=pitch_count[pitcher.name],
            )
            pitch_count[pitcher.name] += self._pitch_count_delta(outcome)
            runs += self._apply_outcome(batter.name, pitcher.name, outcome, state, player_lines)
            if walk_off_enabled and batting_key == "home" and current_home_score + runs > current_away_score:
                break
        return runs

    def _current_pitcher(
        self,
        *,
        defense: TeamContext,
        defense_key: str,
        inning: int,
        pitch_count: dict[str, int],
        staff_state: dict[str, dict[str, object]],
    ) -> PitcherProfile:
        state = staff_state[defense_key]
        current = state["current"]
        assert isinstance(current, PitcherProfile)
        bullpen_index = int(state["bullpen_index"])
        should_switch = False
        if current.name == defense.starter.name:
            if pitch_count[current.name] >= current.fatigue_full:
                should_switch = True
            elif inning >= 7 and pitch_count[current.name] >= current.fatigue_start:
                should_switch = True
        elif pitch_count[current.name] >= current.fatigue_full:
            should_switch = True
        if should_switch and bullpen_index < len(defense.bullpen):
            current = defense.bullpen[bullpen_index]
            state["current"] = current
            state["bullpen_index"] = bullpen_index + 1
        return current

    def _resolve_plate_appearance(
        self,
        batter: BatterProfile,
        pitcher: PitcherProfile,
        park_hit_factor: float,
        park_hr_factor: float,
        offense_factor: float,
        pitch_count: int,
    ) -> str:
        probabilities = {}
        batter_rates = batter.event_rates()
        pitcher_rates = pitcher.event_rates()
        fatigue = self._fatigue_multiplier(pitch_count, pitcher)
        split_factor = batter.vs_left_factor if pitcher.hand == "L" else batter.vs_right_factor if pitcher.hand == "R" else 1.0

        for outcome in LEAGUE_RATES:
            base = batter_rates[outcome] * pitcher_rates[outcome] / LEAGUE_RATES[outcome]
            if outcome == "k":
                base *= float(np.clip(1.0 + ((1.0 - split_factor) * 0.55), 0.82, 1.18))
            elif outcome == "bb":
                base *= float(np.clip(1.0 + ((split_factor - 1.0) * 0.30), 0.88, 1.12))
            elif outcome == "hr":
                base *= float(np.clip(1.0 + ((split_factor - 1.0) * 0.80), 0.78, 1.24))
            elif outcome in {"single", "double", "triple"}:
                base *= float(np.clip(1.0 + ((split_factor - 1.0) * 0.60), 0.82, 1.18))
            if outcome == "hr":
                base *= park_hr_factor * fatigue * offense_factor
            elif outcome in {"single", "double", "triple"}:
                base *= park_hit_factor * fatigue * offense_factor
            elif outcome in {"bb", "hbp"}:
                base *= fatigue * offense_factor
            probabilities[outcome] = max(base, 0.0)

        total_non_out = sum(probabilities.values())
        if total_non_out >= 0.98:
            scale = 0.98 / total_non_out
            probabilities = {key: value * scale for key, value in probabilities.items()}
            total_non_out = sum(probabilities.values())
        probabilities["out"] = 1.0 - total_non_out

        bucket = self.rng.random()
        running = 0.0
        for outcome in OUTCOME_ORDER:
            running += probabilities[outcome]
            if bucket <= running:
                return outcome
        return "out"

    def _apply_outcome(
        self,
        batter_name: str,
        pitcher_name: str,
        outcome: str,
        state: _HalfInningState,
        player_lines: dict[tuple[str, str], int],
    ) -> int:
        runs = 0
        if outcome == "k":
            state.outs += 1
            player_lines[(batter_name, "strikeouts")] += 1
            player_lines[(pitcher_name, "strikeouts")] += 1
            return 0
        if outcome == "out":
            if state.first is not None and state.outs <= 1 and self.rng.random() < 0.11:
                state.first = None
                state.outs += 2
            else:
                state.outs += 1
            return 0
        if outcome in {"bb", "hbp"}:
            player_lines[(batter_name, "walks")] += 1
            player_lines[(pitcher_name, "walks")] += 1
            if state.first is not None:
                if state.second is not None:
                    if state.third is not None:
                        runs += 1
                        player_lines[(batter_name, "rbi")] += 1
                        self._score_runner(state.third, player_lines, pitcher_name)
                    state.third = state.second
                state.second = state.first
            state.first = batter_name
            return runs

        player_lines[(batter_name, "hits")] += 1
        player_lines[(pitcher_name, "hits_allowed")] += 1
        if outcome == "hr":
            player_lines[(batter_name, "home_runs")] += 1
            player_lines[(batter_name, "total_bases")] += 4
            baserunners = sum(1 for runner in (state.first, state.second, state.third) if runner is not None)
            runs += 1 + baserunners
            player_lines[(batter_name, "rbi")] += 1 + baserunners
            for runner in (state.first, state.second, state.third):
                if runner is not None:
                    self._score_runner(runner, player_lines, pitcher_name)
            player_lines[(batter_name, "runs")] += 1
            state.first = state.second = state.third = None
            return runs

        if outcome == "single":
            player_lines[(batter_name, "total_bases")] += 1
            if state.third is not None:
                runs += 1
                self._score_runner(state.third, player_lines, pitcher_name)
                state.third = None
            if state.second is not None and self.rng.random() < min(0.60 * self._speed_guess(batter_name, player_lines), 0.95):
                runs += 1
                self._score_runner(state.second, player_lines, pitcher_name)
                state.second = None
            elif state.second is not None:
                state.third = state.second
                state.second = None
            if state.first is not None and self.rng.random() < min(0.25 * self._speed_guess(batter_name, player_lines), 0.55):
                state.third = state.first
                state.first = None
            elif state.first is not None:
                state.second = state.first
            state.first = batter_name
        elif outcome == "double":
            player_lines[(batter_name, "total_bases")] += 2
            if state.third is not None:
                runs += 1
                self._score_runner(state.third, player_lines, pitcher_name)
                state.third = None
            if state.second is not None:
                runs += 1
                self._score_runner(state.second, player_lines, pitcher_name)
                state.second = None
            if state.first is not None and self.rng.random() < 0.55:
                runs += 1
                self._score_runner(state.first, player_lines, pitcher_name)
                state.first = None
            elif state.first is not None:
                state.third = state.first
                state.first = None
            state.second = batter_name
        elif outcome == "triple":
            player_lines[(batter_name, "total_bases")] += 3
            for runner in (state.first, state.second, state.third):
                if runner is not None:
                    runs += 1
                    self._score_runner(runner, player_lines, pitcher_name)
            state.first = None
            state.second = None
            state.third = batter_name

        if runs:
            player_lines[(batter_name, "rbi")] += runs
        return runs

    def _score_runner(
        self,
        runner_name: str | None,
        player_lines: dict[tuple[str, str], int],
        pitcher_name: str,
    ) -> None:
        if runner_name is None:
            return
        if (runner_name, "runs") in player_lines:
            player_lines[(runner_name, "runs")] += 1
        player_lines[(pitcher_name, "earned_runs")] += 1

    def _pitch_count_delta(self, outcome: str) -> int:
        if outcome in {"bb", "hbp"}:
            return int(self.rng.integers(4, 7))
        if outcome in HIT_EVENTS:
            return int(self.rng.integers(2, 6))
        return int(self.rng.integers(3, 7))

    def _fatigue_multiplier(self, pitch_count: int, pitcher: PitcherProfile) -> float:
        if pitch_count < pitcher.fatigue_start:
            return 1.0
        if pitch_count < pitcher.fatigue_full:
            return 1.0 + 0.005 * (pitch_count - pitcher.fatigue_start)
        return 1.0 + 0.01 * (pitch_count - pitcher.fatigue_start)

    def _market_lines_by_stat(self, market_props: Iterable[MarketProp]) -> dict[tuple[str, str], tuple[float, ...]]:
        grouped: dict[tuple[str, str], set[float]] = {}
        for prop in market_props:
            grouped.setdefault((prop.player_name, prop.stat), set()).add(prop.line)
        return {key: tuple(sorted(values)) for key, values in grouped.items()}

    def _build_distribution(
        self,
        player_name: str,
        stat: str,
        values: tuple[int, ...],
        lines: Iterable[float],
    ) -> PropDistribution:
        array = np.asarray(values, dtype=np.int32)
        over_probabilities = {line: float(np.mean(array > line)) for line in lines}
        under_probabilities = {line: float(np.mean(array <= line)) for line in lines}
        return PropDistribution(
            player_name=player_name,
            stat=stat,
            values=values,
            mean=float(mean(values)) if values else 0.0,
            median=float(median(values)) if values else 0.0,
            percentiles={pct: float(np.percentile(array, pct)) for pct in (5, 25, 50, 75, 95)} if values else {},
            over_probabilities=over_probabilities,
            under_probabilities=under_probabilities,
        )

    def _confidence_from_distribution(self, values: tuple[int, ...]) -> float:
        if not values:
            return 0.0
        array = np.asarray(values, dtype=np.float64)
        normalized_std = float(np.std(array) / max(np.mean(array) + 1.0, 1.0))
        confidence = 1.0 - min(normalized_std, 1.0)
        return max(0.05, confidence)

    def _categorize_edge(self, sim_prob: float, market_price: float, side: str) -> str:
        side_lower = side.lower()
        if side_lower == "under" and sim_prob > 0.95 and market_price < 0.95:
            return "no_value"
        if side_lower == "over" and sim_prob > 0.90 and market_price < 0.85:
            return "lock"
        return "standard"

    def _speed_guess(self, batter_name: str, player_lines: dict[tuple[str, str], int]) -> float:
        del batter_name, player_lines
        return 1.0


def build_demo_mlb_matchup() -> tuple[TeamContext, TeamContext, list[MarketProp]]:
    away = TeamContext(
        team_code="NYM",
        lineup=(
            BatterProfile("1", "Brandon Nimmo", "L", 0.12, 0.170, 0.105, 0.006, 0.165, 0.050, 0.004, 0.040),
            BatterProfile("2", "Francisco Lindor", "S", 0.12, 0.160, 0.090, 0.004, 0.160, 0.050, 0.004, 0.042),
            BatterProfile("3", "Juan Soto", "L", 0.12, 0.180, 0.170, 0.003, 0.150, 0.040, 0.002, 0.065),
            BatterProfile("4", "Pete Alonso", "R", 0.12, 0.220, 0.095, 0.003, 0.130, 0.045, 0.001, 0.070),
            BatterProfile("5", "Starling Marte", "R", 0.11, 0.210, 0.070, 0.004, 0.165, 0.040, 0.004, 0.028),
            BatterProfile("6", "Jeff McNeil", "L", 0.11, 0.110, 0.075, 0.003, 0.175, 0.035, 0.003, 0.020),
            BatterProfile("7", "Mark Vientos", "R", 0.10, 0.260, 0.080, 0.002, 0.130, 0.040, 0.001, 0.060),
            BatterProfile("8", "Francisco Alvarez", "R", 0.10, 0.280, 0.070, 0.002, 0.120, 0.035, 0.001, 0.055),
            BatterProfile("9", "Tyrone Taylor", "R", 0.10, 0.230, 0.060, 0.003, 0.145, 0.030, 0.002, 0.030),
        ),
        starter=PitcherProfile("P1", "Paul Skenes", "R", 0.330, 0.070, 0.004, 0.110, 0.030, 0.002, 0.025),
        park_hit_factor=0.98,
        park_hr_factor=0.93,
    )
    home = TeamContext(
        team_code="PIT",
        lineup=(
            BatterProfile("10", "Oneil Cruz", "L", 0.12, 0.310, 0.105, 0.003, 0.135, 0.040, 0.006, 0.060),
            BatterProfile("11", "Bryan Reynolds", "S", 0.12, 0.180, 0.085, 0.003, 0.165, 0.042, 0.003, 0.045),
            BatterProfile("12", "Ke'Bryan Hayes", "R", 0.11, 0.155, 0.070, 0.004, 0.160, 0.035, 0.004, 0.020),
            BatterProfile("13", "Andrew McCutchen", "R", 0.11, 0.210, 0.110, 0.003, 0.140, 0.036, 0.001, 0.040),
            BatterProfile("14", "Joey Bart", "R", 0.11, 0.250, 0.070, 0.002, 0.145, 0.035, 0.002, 0.040),
            BatterProfile("15", "Nick Gonzales", "R", 0.11, 0.195, 0.065, 0.002, 0.155, 0.032, 0.003, 0.025),
            BatterProfile("16", "Jack Suwinski", "L", 0.11, 0.315, 0.100, 0.002, 0.125, 0.034, 0.002, 0.050),
            BatterProfile("17", "Henry Davis", "R", 0.10, 0.280, 0.080, 0.002, 0.125, 0.030, 0.002, 0.040),
            BatterProfile("18", "Jared Triolo", "R", 0.11, 0.220, 0.080, 0.003, 0.145, 0.035, 0.003, 0.020),
        ),
        starter=PitcherProfile("P2", "Kodai Senga", "R", 0.290, 0.090, 0.005, 0.120, 0.032, 0.002, 0.028),
        park_hit_factor=0.98,
        park_hr_factor=0.93,
    )
    props = [
        MarketProp("Juan Soto", "hits", 0.5, "over", 0.72, "demo-soto-hit"),
        MarketProp("Juan Soto", "home_runs", 0.5, "over", 0.22, "demo-soto-hr"),
        MarketProp("Paul Skenes", "strikeouts", 7.5, "over", 0.54, "demo-skenes-k"),
    ]
    return away, home, props
