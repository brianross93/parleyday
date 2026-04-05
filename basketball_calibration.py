from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from dataclasses import replace

from basketball_game_engine import _assignments, _clock_state, _floor_states, _lineup_from_ids, simulate_game
from basketball_possession_engine import _select_coverage, _select_play_call, simulate_possession
from basketball_sim_schema import (
    DefensiveCoverage,
    DefensiveRole,
    GameSimulationInput,
    OffensiveRole,
    PlayFamily,
    PlayerCondition,
    PlayerSimProfile,
    PlayerTraitProfile,
    PossessionContext,
    PossessionPhase,
    ScoreState,
    RotationPlan,
    TeamTactics,
)

TARGET_BANDS = {
    "possession_mix": {
        "points_per_possession": (1.05, 1.15),
        "turnover_rate": (0.12, 0.16),
        "shooting_foul_rate": (0.15, 0.22),
        "oreb_rate_on_misses": (0.24, 0.30),
        "three_pa_share": (0.35, 0.42),
        "fta_per_fga": (0.22, 0.30),
        "rim_attempt_share": (0.28, 0.36),
        "midrange_share": (0.08, 0.16),
        "above_break_three_share": (0.23, 0.31),
        "corner_three_share": (0.07, 0.13),
    },
    "usage_concentration": {
        "top_scorer_possession_share": (0.25, 0.36),
        "top_2_usage_share": (0.44, 0.58),
        "top_3_usage_share": (0.58, 0.74),
        "assist_concentration": (0.30, 0.50),
        "fta_concentration": (0.25, 0.50),
        # Temporary band for the current engine scope. With only HIGH_PNR + ISO
        # active and no transition/off-ball three-point generation paths, the
        # top two shooters absorb a larger share of team 3PA than a full NBA
        # offense would. Tighten this once more play families are live.
        "three_pa_concentration": (0.50, 0.70),
    },
    "fta_by_archetype": {
        "spot_up_role": (0.5, 2.0),
        "average_starter": (1.5, 4.0),
        "aggressive_driver": (3.5, 6.5),
        "elite_drawer": (6.0, 10.0),
        "historic_drawer": (9.0, 16.0),
    },
    "game_variance": {
        "game_total_std": (16.0, 28.0),
        "possessions_mean": (190.0, 210.0),
        "possessions_std": (4.0, 12.0),
    },
}


def build_calibration_game_input(
    *,
    home_star_traits: dict[str, float] | None = None,
    away_star_traits: dict[str, float] | None = None,
) -> GameSimulationInput:
    home_players = _team_profiles("HOM", home_star_traits)
    away_players = _team_profiles("AWY", away_star_traits)
    players = home_players + away_players
    return GameSimulationInput(
        game_id="AWY@HOM",
        home_team_code="HOM",
        away_team_code="AWY",
        players=players,
        home_tactics=_team_tactics("HOM"),
        away_tactics=_team_tactics("AWY"),
        home_rotation=_rotation_plan(home_players),
        away_rotation=_rotation_plan(away_players),
        opening_tip_winner="HOM",
    )


def measure_possession_mix(samples: int = 5000, rng_seed: int = 7) -> dict[str, float]:
    sim_input = build_calibration_game_input()
    rng = random.Random(rng_seed)
    totals = Counter()
    shot_types = Counter()

    for possession_number in range(1, samples + 1):
        outcome = simulate_possession(_base_context(sim_input, offense_home=(possession_number % 2 == 1), possession_number=possession_number), rng)
        totals["possessions"] += 1
        totals["points"] += outcome.points_scored
        totals["turnovers"] += int(outcome.turnover)
        totals["fouls"] += int(outcome.foul_committed)
        totals["misses"] += int((not outcome.made_shot) and any(event.event_type.name == "SHOT" for event in outcome.events))
        totals["off_rebounds"] += int(outcome.offensive_rebound)
        totals["fta"] += outcome.free_throws_attempted
        for event in outcome.events:
            if event.event_type.name != "SHOT" or event.shot_type is None:
                continue
            totals["fga"] += 1
            shot_types[event.shot_type.value] += 1
            if event.shot_type.name in {"ABOVE_BREAK_THREE", "CORNER_THREE"}:
                totals["threes"] += 1
            if event.shot_type.name == "RIM":
                totals["rim"] += 1
            if event.shot_type.name == "MIDRANGE":
                totals["midrange"] += 1

    misses = max(totals["misses"], 1)
    fga = max(totals["fga"], 1)
    return {
        "points_per_possession": totals["points"] / max(totals["possessions"], 1),
        "turnover_rate": totals["turnovers"] / max(totals["possessions"], 1),
        "shooting_foul_rate": totals["fouls"] / max(totals["possessions"], 1),
        "oreb_rate_on_misses": totals["off_rebounds"] / misses,
        "three_pa_share": totals["threes"] / fga,
        "fta_per_fga": totals["fta"] / fga,
        "rim_attempt_share": totals["rim"] / fga,
        "midrange_share": totals["midrange"] / fga,
        "corner_three_share": shot_types["corner_three"] / fga,
        "above_break_three_share": shot_types["above_break_three"] / fga,
        "paint_share": shot_types["paint"] / fga,
    }


def measure_usage_concentration(samples: int = 5000, rng_seed: int = 11) -> dict[str, float]:
    sim_input = build_calibration_game_input(
        home_star_traits={
            "separation": 17.0,
            "burst": 15.0,
            "pullup_shooting": 16.0,
            "finishing": 17.0,
            "ball_security": 15.0,
            "foul_drawing": 18.0,
            "decision_making": 16.0,
            "pass_vision": 16.0,
            "pass_accuracy": 15.0,
        }
    )
    rng = random.Random(rng_seed)
    primary_actor_counts = Counter()
    assists = Counter()
    fta = Counter()
    threes = Counter()

    for possession_number in range(1, samples + 1):
        context = _base_context(sim_input, offense_home=True, possession_number=possession_number)
        play_call = _select_play_call(context, rng)
        coverage = _select_coverage(context, rng)
        outcome = simulate_possession(replace(context, play_call=play_call, coverage=coverage), rng)
        primary_actor_counts[play_call.primary_actor_id] += 1
        if outcome.assisting_player_id and outcome.made_shot:
            assists[outcome.assisting_player_id] += 1
        if outcome.foul_committed and outcome.shooting_player_id:
            fta[outcome.shooting_player_id] += outcome.free_throws_attempted
        for event in outcome.events:
            if event.event_type.name == "SHOT" and event.shot_type and event.shot_type.name in {"ABOVE_BREAK_THREE", "CORNER_THREE"} and outcome.shooting_player_id:
                threes[outcome.shooting_player_id] += 1

    primary_shares = _top_share_metrics(primary_actor_counts)
    return {
        "top_scorer_possession_share": primary_shares["top_1_share"],
        "top_2_usage_share": primary_shares["top_2_share"],
        "top_3_usage_share": primary_shares["top_3_share"],
        "assist_concentration": _top_n_share(assists, 1),
        "fta_concentration": _top_n_share(fta, 1),
        "three_pa_concentration": _top_n_share(threes, 2),
    }


def measure_fta_by_archetype(games_per_archetype: int = 120, rng_seed: int = 19) -> dict[str, float]:
    archetypes = {
        "spot_up_role": {
            "foul_drawing": 3.0,
            "separation": 8.0,
            "burst": 8.0,
            "pullup_shooting": 9.0,
            "finishing": 9.0,
            "ball_security": 10.0,
            "decision_making": 11.0,
            "catch_shoot": 15.0,
        },
        "average_starter": {
            "foul_drawing": 7.0,
            "separation": 11.0,
            "burst": 11.0,
            "pullup_shooting": 12.0,
            "finishing": 12.0,
            "ball_security": 12.0,
            "decision_making": 12.0,
        },
        "aggressive_driver": {
            "foul_drawing": 11.0,
            "separation": 14.0,
            "burst": 14.0,
            "pullup_shooting": 13.0,
            "finishing": 14.0,
            "ball_security": 13.0,
            "decision_making": 13.0,
        },
        "elite_drawer": {
            "foul_drawing": 15.0,
            "separation": 16.0,
            "burst": 15.0,
            "pullup_shooting": 15.0,
            "finishing": 16.0,
            "ball_security": 14.0,
            "decision_making": 15.0,
        },
        "historic_drawer": {
            "foul_drawing": 19.0,
            "separation": 17.0,
            "burst": 16.0,
            "pullup_shooting": 16.0,
            "finishing": 17.0,
            "ball_security": 15.0,
            "decision_making": 16.0,
        },
    }
    results: dict[str, float] = {}
    possessions_per_game = 100

    for index, (name, archetype_traits) in enumerate(archetypes.items()):
        rng = random.Random(rng_seed + index)
        fta_games: list[float] = []
        sim_input = build_calibration_game_input(home_star_traits=archetype_traits)
        star_id = "HOM_pg"
        for game_index in range(games_per_archetype):
            star_fta = 0.0
            for possession_number in range(1, possessions_per_game + 1):
                context = _base_context(sim_input, offense_home=True, possession_number=(game_index * possessions_per_game) + possession_number)
                play_call = _select_play_call(context, rng)
                coverage = _select_coverage(context, rng)
                outcome = simulate_possession(replace(context, play_call=play_call, coverage=coverage), rng)
                if outcome.foul_committed and outcome.shooting_player_id == star_id:
                    star_fta += outcome.free_throws_attempted
            fta_games.append(star_fta)
        results[name] = sum(fta_games) / max(len(fta_games), 1)
    return results


def measure_game_variance(game_count: int = 200, rng_seed: int = 23) -> dict[str, float]:
    totals = []
    possessions = []
    home_rebounds = []
    home_assists = []
    for offset in range(game_count):
        sim_input = _style_variant_game_input(offset)
        result = simulate_game(sim_input, rng_seed=rng_seed + offset)
        totals.append(result.home_score + result.away_score)
        possessions.append(result.possession_count)
        home_players = [player for player in result.player_box_scores if player.player_id.startswith("HOM_")]
        home_rebounds.append(sum(player.rebounds for player in home_players))
        home_assists.append(sum(player.assists for player in home_players))
    return {
        "game_total_mean": statistics.fmean(totals),
        "game_total_std": statistics.pstdev(totals),
        "possessions_mean": statistics.fmean(possessions),
        "possessions_std": statistics.pstdev(possessions),
        "home_rebounds_mean": statistics.fmean(home_rebounds),
        "home_assists_mean": statistics.fmean(home_assists),
    }


def run_calibration_suite(
    *,
    possession_samples: int = 5000,
    usage_samples: int = 5000,
    archetype_games: int = 120,
    game_count: int = 200,
    rng_seed: int = 7,
) -> dict[str, object]:
    return {
        "possession_mix": measure_possession_mix(samples=possession_samples, rng_seed=rng_seed),
        "usage_concentration": measure_usage_concentration(samples=usage_samples, rng_seed=rng_seed + 100),
        "fta_by_archetype": measure_fta_by_archetype(games_per_archetype=archetype_games, rng_seed=rng_seed + 200),
        "game_variance": measure_game_variance(game_count=game_count, rng_seed=rng_seed + 300),
    }


def suite_as_json(**kwargs) -> str:
    return json.dumps(run_calibration_suite(**kwargs), indent=2, sort_keys=True)


def evaluate_calibration_targets(report: dict[str, object]) -> list[str]:
    violations: list[str] = []
    for section, metrics in TARGET_BANDS.items():
        actual_metrics = report.get(section, {})
        if not isinstance(actual_metrics, dict):
            violations.append(f"{section}: missing section")
            continue
        for metric_name, (lower, upper) in metrics.items():
            value = actual_metrics.get(metric_name)
            if value is None:
                violations.append(f"{section}.{metric_name}: missing metric")
                continue
            if not (lower <= float(value) <= upper):
                violations.append(
                    f"{section}.{metric_name}: {value:.4f} outside target band [{lower:.4f}, {upper:.4f}]"
                )
    return violations


def assert_calibration_targets(report: dict[str, object]) -> None:
    violations = evaluate_calibration_targets(report)
    if violations:
        raise AssertionError("Calibration target failures:\n" + "\n".join(violations))


def _base_context(sim_input: GameSimulationInput, *, offense_home: bool, possession_number: int) -> PossessionContext:
    home_ids = sim_input.home_rotation.starters
    away_ids = sim_input.away_rotation.starters
    player_lookup = {player.player_id: player for player in sim_input.players}
    offense_team = sim_input.home_team_code if offense_home else sim_input.away_team_code
    defense_team = sim_input.away_team_code if offense_home else sim_input.home_team_code
    offense_ids = home_ids if offense_home else away_ids
    defense_ids = away_ids if offense_home else home_ids
    offense_lineup = _lineup_from_ids(offense_team, offense_ids, player_lookup)
    defense_lineup = _lineup_from_ids(defense_team, defense_ids, player_lookup)
    return PossessionContext(
        offense_team_code=offense_team,
        defense_team_code=defense_team,
        clock=_clock_state(42.0 * 60.0, possession_number, 15.0),
        score=ScoreState(offense_score=55, defense_score=54),
        offense_lineup=offense_lineup,
        defense_lineup=defense_lineup,
        offensive_tactics=sim_input.home_tactics if offense_home else sim_input.away_tactics,
        defensive_tactics=sim_input.away_tactics if offense_home else sim_input.home_tactics,
        floor_players=_floor_states(offense_ids, defense_ids),
        defensive_assignments=_assignments(offense_ids, defense_ids),
        player_pool=sim_input.players,
        current_phase=PossessionPhase.PRIMARY_ACTION,
        play_call=None,
        coverage=None,
    )


def _team_profiles(team_code: str, star_traits: dict[str, float] | None) -> tuple[PlayerSimProfile, ...]:
    traits = {
        "pg": {
            "offensive_role": OffensiveRole.PRIMARY_CREATOR,
            "defensive_role": DefensiveRole.POINT_OF_ATTACK,
            "separation": 15.0,
            "burst": 13.0,
            "pullup_shooting": 14.0,
            "catch_shoot": 11.0,
            "finishing": 14.0,
            "pass_vision": 15.0,
            "pass_accuracy": 14.0,
            "decision_making": 14.0,
            "ball_security": 14.0,
            "foul_drawing": 12.0,
            "containment": 13.0,
            "closeout": 12.0,
            "screen_nav": 11.0,
            "steal_pressure": 13.0,
            "dreb": 8.0,
            "size": 9.0,
            "reach": 10.0,
        },
        "sg": {
            "offensive_role": OffensiveRole.MOVEMENT_SHOOTER,
            "defensive_role": DefensiveRole.WING_STOPPER,
            "separation": 11.0,
            "burst": 10.0,
            "pullup_shooting": 11.0,
            "catch_shoot": 16.0,
            "finishing": 11.0,
            "pass_vision": 9.0,
            "pass_accuracy": 10.0,
            "decision_making": 11.0,
            "ball_security": 10.0,
            "foul_drawing": 6.0,
            "containment": 13.0,
            "closeout": 12.0,
            "screen_nav": 11.0,
            "steal_pressure": 11.0,
            "dreb": 9.0,
            "size": 10.0,
            "reach": 11.0,
        },
        "sf": {
            "offensive_role": OffensiveRole.SLASHER,
            "defensive_role": DefensiveRole.WING_STOPPER,
            "separation": 11.0,
            "burst": 12.0,
            "pullup_shooting": 10.0,
            "catch_shoot": 12.0,
            "finishing": 12.0,
            "pass_vision": 9.0,
            "pass_accuracy": 10.0,
            "decision_making": 11.0,
            "ball_security": 10.0,
            "foul_drawing": 8.0,
            "containment": 12.0,
            "closeout": 12.0,
            "screen_nav": 11.0,
            "steal_pressure": 10.0,
            "dreb": 10.0,
            "size": 11.0,
            "reach": 12.0,
        },
        "pf": {
            "offensive_role": OffensiveRole.GLUE,
            "defensive_role": DefensiveRole.HELPER,
            "separation": 8.0,
            "burst": 9.0,
            "pullup_shooting": 8.0,
            "catch_shoot": 10.0,
            "finishing": 12.0,
            "pass_vision": 8.0,
            "pass_accuracy": 8.0,
            "decision_making": 11.0,
            "ball_security": 9.0,
            "foul_drawing": 7.0,
            "screen_setting": 12.0,
            "oreb": 12.0,
            "containment": 10.0,
            "closeout": 10.0,
            "screen_nav": 8.0,
            "interior_def": 12.0,
            "rim_protect": 11.0,
            "dreb": 13.0,
            "size": 14.0,
            "reach": 14.0,
        },
        "c": {
            "offensive_role": OffensiveRole.ROLL_BIG,
            "defensive_role": DefensiveRole.RIM_PROTECTOR,
            "separation": 5.0,
            "burst": 8.0,
            "pullup_shooting": 4.0,
            "catch_shoot": 5.0,
            "finishing": 14.0,
            "pass_vision": 7.0,
            "pass_accuracy": 7.0,
            "decision_making": 11.0,
            "ball_security": 8.0,
            "foul_drawing": 8.0,
            "screen_setting": 14.0,
            "oreb": 14.0,
            "containment": 8.0,
            "closeout": 7.0,
            "screen_nav": 6.0,
            "interior_def": 15.0,
            "rim_protect": 15.0,
            "dreb": 15.0,
            "size": 17.0,
            "reach": 17.0,
        },
        "sixth": {
            "offensive_role": OffensiveRole.SECONDARY_CREATOR,
            "defensive_role": DefensiveRole.POINT_OF_ATTACK,
            "separation": 13.0,
            "burst": 12.0,
            "pullup_shooting": 13.0,
            "catch_shoot": 11.0,
            "finishing": 12.0,
            "pass_vision": 12.0,
            "pass_accuracy": 12.0,
            "decision_making": 12.0,
            "ball_security": 11.0,
            "foul_drawing": 10.0,
            "containment": 11.0,
            "closeout": 10.0,
            "screen_nav": 10.0,
            "steal_pressure": 11.0,
            "dreb": 8.0,
            "size": 9.0,
            "reach": 9.0,
            "stamina": 13.0,
        },
        "wing": {
            "offensive_role": OffensiveRole.SPACER,
            "defensive_role": DefensiveRole.WING_STOPPER,
            "separation": 9.0,
            "burst": 9.0,
            "pullup_shooting": 8.0,
            "catch_shoot": 13.0,
            "finishing": 9.0,
            "pass_vision": 8.0,
            "pass_accuracy": 8.0,
            "decision_making": 11.0,
            "ball_security": 10.0,
            "foul_drawing": 5.0,
            "containment": 12.0,
            "closeout": 12.0,
            "screen_nav": 11.0,
            "steal_pressure": 9.0,
            "dreb": 9.0,
            "size": 10.5,
            "reach": 11.5,
            "stamina": 13.0,
        },
        "combo_big": {
            "offensive_role": OffensiveRole.POP_BIG,
            "defensive_role": DefensiveRole.HELPER,
            "separation": 7.0,
            "burst": 8.0,
            "pullup_shooting": 7.0,
            "catch_shoot": 10.0,
            "finishing": 11.0,
            "pass_vision": 7.0,
            "pass_accuracy": 8.0,
            "decision_making": 10.0,
            "ball_security": 9.0,
            "foul_drawing": 6.0,
            "screen_setting": 12.0,
            "oreb": 11.0,
            "containment": 9.0,
            "closeout": 9.0,
            "screen_nav": 8.0,
            "interior_def": 11.0,
            "rim_protect": 10.0,
            "dreb": 12.0,
            "size": 13.5,
            "reach": 13.5,
            "stamina": 12.0,
        },
        "big": {
            "offensive_role": OffensiveRole.ROLL_BIG,
            "defensive_role": DefensiveRole.REBOUNDER,
            "separation": 4.0,
            "burst": 6.5,
            "pullup_shooting": 3.0,
            "catch_shoot": 4.0,
            "finishing": 12.0,
            "pass_vision": 5.0,
            "pass_accuracy": 6.0,
            "decision_making": 9.0,
            "ball_security": 8.0,
            "foul_drawing": 7.0,
            "screen_setting": 13.0,
            "oreb": 13.0,
            "containment": 7.0,
            "closeout": 7.0,
            "screen_nav": 6.0,
            "interior_def": 12.0,
            "rim_protect": 12.0,
            "dreb": 13.0,
            "size": 15.0,
            "reach": 15.5,
            "stamina": 11.5,
        },
        "guard": {
            "offensive_role": OffensiveRole.SPACER,
            "defensive_role": DefensiveRole.POINT_OF_ATTACK,
            "separation": 9.5,
            "burst": 9.5,
            "pullup_shooting": 9.0,
            "catch_shoot": 12.0,
            "finishing": 9.0,
            "pass_vision": 8.5,
            "pass_accuracy": 9.0,
            "decision_making": 10.0,
            "ball_security": 10.5,
            "foul_drawing": 5.0,
            "containment": 11.0,
            "closeout": 10.5,
            "screen_nav": 10.0,
            "steal_pressure": 10.0,
            "dreb": 7.0,
            "size": 8.5,
            "reach": 9.0,
            "stamina": 13.5,
        },
    }
    home_star_traits = star_traits or {}
    profiles = []
    for slot in ("pg", "sg", "sf", "pf", "c", "sixth", "wing", "combo_big", "big", "guard"):
        slot_traits = dict(traits[slot])
        if slot == "pg":
            slot_traits.update(home_star_traits)
        profiles.append(_player_profile(f"{team_code}_{slot}", team_code, slot, slot_traits))
    return tuple(profiles)


def _player_profile(player_id: str, team_code: str, slot: str, values: dict[str, float]) -> PlayerSimProfile:
    positions = {
        "pg": ("PG",),
        "sg": ("SG",),
        "sf": ("SF",),
        "pf": ("PF",),
        "c": ("C",),
        "sixth": ("PG", "SG"),
        "wing": ("SG", "SF"),
        "combo_big": ("PF", "C"),
        "big": ("C",),
        "guard": ("PG", "SG"),
    }[slot]
    return PlayerSimProfile(
        player_id=player_id,
        name=player_id,
        team_code=team_code,
        positions=positions,
        offensive_role=values.get("offensive_role", OffensiveRole.GLUE),
        defensive_role=values.get("defensive_role", DefensiveRole.HELPER),
        traits=PlayerTraitProfile(
            ball_security=values.get("ball_security", 10.0),
            separation=values.get("separation", 10.0),
            burst=values.get("burst", 10.0),
            pullup_shooting=values.get("pullup_shooting", 10.0),
            catch_shoot=values.get("catch_shoot", 10.0),
            finishing=values.get("finishing", 10.0),
            pass_vision=values.get("pass_vision", 10.0),
            pass_accuracy=values.get("pass_accuracy", 10.0),
            decision_making=values.get("decision_making", 10.0),
            screen_setting=values.get("screen_setting", 7.0),
            rebounding=((values.get("oreb", 9.0) + values.get("dreb", 10.0)) / 2.0),
            free_throw_rating=values.get("free_throw_rating", 14.0),
            ft_pct_raw=values.get("ft_pct_raw", 0.79),
            foul_drawing=values.get("foul_drawing", 10.0),
            containment=values.get("containment", 10.0),
            closeout=values.get("closeout", 10.0),
            screen_nav=values.get("screen_nav", 9.0),
            interior_def=values.get("interior_def", 10.0),
            rim_protect=values.get("rim_protect", 9.0),
            steal_pressure=values.get("steal_pressure", 10.0),
            foul_discipline=values.get("foul_discipline", 11.0),
            help_rotation=values.get("help_rotation", 10.0),
            stamina=values.get("stamina", 14.0),
            role_consistency=values.get("role_consistency", 11.0),
            clutch=values.get("clutch", 0.0),
            size=values.get("size", 10.0),
            reach=values.get("reach", 10.0),
        ),
        condition=PlayerCondition(),
    )


def _team_tactics(team_code: str) -> TeamTactics:
    return TeamTactics(
        pace_target=99.0,
        transition_frequency=0.14,
        crash_glass_rate=0.22,
        help_aggressiveness=0.48,
        switch_rate=0.32,
        zone_rate=0.0,
        no_middle_rate=0.1,
        pre_switch_rate=0.1,
        rotation_tightness=0.78,
        late_clock_isolation_rate=0.15,
        early_offense_rate=0.18,
        pick_and_roll_rate=0.42,
        handoff_rate=0.08,
        post_touch_rate=0.08,
        off_ball_screen_rate=0.12,
        play_family_weights={
            PlayFamily.HIGH_PICK_AND_ROLL: 0.68,
            PlayFamily.ISO: 0.32,
        },
        coverage_weights={
            DefensiveCoverage.DROP: 0.62,
            DefensiveCoverage.SWITCH: 0.38,
        },
        star_usage_bias=1.18,
        closeout_attack_rate=0.42,
        second_side_rate=0.31,
        corner_spacing_bias=0.53,
        shooter_distribution_weights={
            f"{team_code}_pg": 0.86,
            f"{team_code}_sg": 1.26,
            f"{team_code}_sf": 1.04,
            f"{team_code}_pf": 1.12,
            f"{team_code}_c": 0.72,
            f"{team_code}_sixth": 1.12,
            f"{team_code}_wing": 1.02,
            f"{team_code}_combo_big": 0.84,
            f"{team_code}_big": 0.56,
            f"{team_code}_guard": 0.94,
        },
    )


def _style_variant_game_input(offset: int) -> GameSimulationInput:
    style_index = offset % 4
    home_tactics = _team_tactics("HOM")
    away_tactics = _team_tactics("AWY")
    home_star = {
        "separation": 17.0,
        "burst": 15.0,
        "pullup_shooting": 16.0,
        "finishing": 17.0,
        "foul_drawing": 17.0,
    }
    away_star = {
        "separation": 15.0,
        "burst": 14.0,
        "pullup_shooting": 15.0,
        "finishing": 16.0,
        "foul_drawing": 14.0,
    }

    if style_index == 0:
        home_tactics = replace(home_tactics, pace_target=102.0, crash_glass_rate=0.28, switch_rate=0.28)
        away_tactics = replace(away_tactics, pace_target=100.0, switch_rate=0.42)
        home_star["pullup_shooting"] = 17.0
    elif style_index == 1:
        home_tactics = replace(home_tactics, pace_target=97.0, crash_glass_rate=0.18, switch_rate=0.46)
        away_tactics = replace(away_tactics, pace_target=95.0, crash_glass_rate=0.20, switch_rate=0.48)
        home_star["foul_drawing"] = 15.0
        away_star["pullup_shooting"] = 16.0
    elif style_index == 2:
        home_tactics = replace(home_tactics, pace_target=100.0, crash_glass_rate=0.24, switch_rate=0.22)
        away_tactics = replace(away_tactics, pace_target=103.0, crash_glass_rate=0.27, switch_rate=0.30)
        away_star["burst"] = 16.0
        away_star["finishing"] = 17.0
    else:
        home_tactics = replace(home_tactics, pace_target=96.0, crash_glass_rate=0.26, switch_rate=0.35)
        away_tactics = replace(away_tactics, pace_target=101.0, crash_glass_rate=0.19, switch_rate=0.40)
        home_star["separation"] = 16.0
        away_star["foul_drawing"] = 16.0

    sim_input = build_calibration_game_input(home_star_traits=home_star, away_star_traits=away_star)
    return replace(sim_input, home_tactics=home_tactics, away_tactics=away_tactics)


def _rotation_plan(players: tuple[PlayerSimProfile, ...]) -> RotationPlan:
    ordered_ids = tuple(player.player_id for player in players)
    starters = ordered_ids[:5]
    closing_group = (ordered_ids[0], ordered_ids[1], ordered_ids[2], ordered_ids[3], ordered_ids[5])
    return RotationPlan(
        starters=starters,
        closing_group=closing_group,
        stagger_pairs=((ordered_ids[0], ordered_ids[1]),),
        target_minutes={
            ordered_ids[0]: 36.0,
            ordered_ids[1]: 34.0,
            ordered_ids[2]: 32.0,
            ordered_ids[3]: 30.0,
            ordered_ids[4]: 28.0,
            ordered_ids[5]: 28.0,
            ordered_ids[6]: 22.0,
            ordered_ids[7]: 18.0,
            ordered_ids[8]: 14.0,
            ordered_ids[9]: 10.0,
        },
        max_stint_minutes={
            ordered_ids[0]: 10.0,
            ordered_ids[1]: 10.0,
            ordered_ids[2]: 9.0,
            ordered_ids[3]: 9.0,
            ordered_ids[4]: 8.0,
            ordered_ids[5]: 8.0,
            ordered_ids[6]: 7.0,
            ordered_ids[7]: 7.0,
            ordered_ids[8]: 6.0,
            ordered_ids[9]: 5.0,
        },
        backup_priority={
            ordered_ids[0]: (ordered_ids[5], ordered_ids[9]),
            ordered_ids[1]: (ordered_ids[5], ordered_ids[6]),
            ordered_ids[2]: (ordered_ids[6], ordered_ids[7]),
            ordered_ids[3]: (ordered_ids[7], ordered_ids[8]),
            ordered_ids[4]: (ordered_ids[8], ordered_ids[7]),
            ordered_ids[5]: (),
            ordered_ids[6]: (),
            ordered_ids[7]: (),
            ordered_ids[8]: (),
            ordered_ids[9]: (),
        },
    )


def _top_share_metrics(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    ordered = sorted(counter.values(), reverse=True)
    if total <= 0:
        return {"top_1_share": 0.0, "top_2_share": 0.0, "top_3_share": 0.0}
    return {
        "top_1_share": sum(ordered[:1]) / total,
        "top_2_share": sum(ordered[:2]) / total,
        "top_3_share": sum(ordered[:3]) / total,
    }


def _top_n_share(counter: Counter[str], n: int) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return sum(value for _, value in counter.most_common(n)) / total
