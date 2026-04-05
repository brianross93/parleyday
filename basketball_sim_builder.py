from __future__ import annotations

from collections import defaultdict

from basketball_db import import_draftkings_slate_to_db, load_nba_sim_profiles_for_slate
from data_pipeline.cache import DEFAULT_DB_PATH
from basketball_sim_schema import (
    DefensiveCoverage,
    DefensiveRole,
    GameSimulationInput,
    OffensiveRole,
    PlayFamily,
    PlayerCondition,
    PlayerSimProfile,
    PlayerTraitProfile,
    RotationPlan,
    TeamTactics,
)
from dfs_ingest import DraftKingsSlate
from dfs_nba import DraftKingsNBAProjection, build_nba_dk_projections
from nba_matchup_features import load_nba_matchup_features
from player_name_utils import dfs_name_key


def build_nba_sim_inputs_from_dk_csv(date_str: str, salary_csv_path: str, *, db_path: str | None = None) -> list[GameSimulationInput]:
    slate = import_draftkings_slate_to_db(date_str, salary_csv_path, sport="nba", db_path=db_path or DEFAULT_DB_PATH)
    return build_nba_sim_inputs_from_db_slate(date_str, slate, db_path=db_path or DEFAULT_DB_PATH)


def build_nba_sim_inputs_from_db_slate(date_str: str, slate: DraftKingsSlate, *, db_path: str = DEFAULT_DB_PATH) -> list[GameSimulationInput]:
    if slate.sport != "nba":
        return []
    profiles_by_key = load_nba_sim_profiles_for_slate(date_str, slate, db_path=db_path)
    by_game: dict[str, list[PlayerSimProfile]] = defaultdict(list)
    for player in slate.players:
        if not player.game or "@" not in player.game:
            continue
        profile = profiles_by_key.get((player.team, dfs_name_key(player.name)))
        if profile is None:
            continue
        by_game[player.game].append(profile)
    results: list[GameSimulationInput] = []
    for game, game_players in sorted(by_game.items()):
        away_code, home_code = game.split("@", 1)
        team_features = load_nba_matchup_features(date_str, (away_code, home_code))
        away_players = tuple(player for player in game_players if player.team_code == away_code)
        home_players = tuple(player for player in game_players if player.team_code == home_code)
        if len(away_players) < 5 or len(home_players) < 5:
            continue
        players = tuple(game_players)
        results.append(
            GameSimulationInput(
                game_id=game,
                home_team_code=home_code,
                away_team_code=away_code,
                players=players,
                home_tactics=_build_team_tactics(home_code, home_players, team_features.get(home_code, {}), team_features.get(away_code, {})),
                away_tactics=_build_team_tactics(away_code, away_players, team_features.get(away_code, {}), team_features.get(home_code, {})),
                home_rotation=_build_rotation_plan(home_players),
                away_rotation=_build_rotation_plan(away_players),
            )
        )
    return results


def build_nba_sim_inputs(date_str: str, slate: DraftKingsSlate) -> list[GameSimulationInput]:
    if slate.sport != "nba":
        return []
    projections = build_nba_dk_projections(date_str, slate)
    by_game: dict[str, list[DraftKingsNBAProjection]] = defaultdict(list)
    for projection in projections:
        if projection.game and "@" in projection.game:
            by_game[projection.game].append(projection)
    results: list[GameSimulationInput] = []
    for game, game_projections in sorted(by_game.items()):
        away_code, home_code = game.split("@", 1)
        team_features = load_nba_matchup_features(date_str, (away_code, home_code))
        players = tuple(_projection_to_sim_profile(item) for item in game_projections)
        away_players = tuple(player for player in players if player.team_code == away_code)
        home_players = tuple(player for player in players if player.team_code == home_code)
        if len(away_players) < 5 or len(home_players) < 5:
            continue
        results.append(
            GameSimulationInput(
                game_id=game,
                home_team_code=home_code,
                away_team_code=away_code,
                players=players,
                home_tactics=_build_team_tactics(home_code, home_players, team_features.get(home_code, {}), team_features.get(away_code, {})),
                away_tactics=_build_team_tactics(away_code, away_players, team_features.get(away_code, {}), team_features.get(home_code, {})),
                home_rotation=_build_rotation_plan(home_players),
                away_rotation=_build_rotation_plan(away_players),
            )
        )
    return results


def _projection_to_sim_profile(projection: DraftKingsNBAProjection) -> PlayerSimProfile:
    offensive_role = _infer_offensive_role(projection)
    defensive_role = _infer_defensive_role(projection)
    positions = set(projection.positions) | set(projection.roster_positions)
    assist_per_minute = projection.assists / max(projection.minutes, 1.0)
    rebound_per_minute = projection.rebounds / max(projection.minutes, 1.0)
    scoring_per_minute = projection.points / max(projection.minutes, 1.0)
    ft_pct_raw = _bounded(0.55 + (projection.projection_confidence * 0.3), 0.45, 0.92)
    return PlayerSimProfile(
        player_id=projection.player_id or dfs_name_key(projection.name),
        name=projection.name,
        team_code=projection.team,
        positions=projection.positions,
        offensive_role=offensive_role,
        defensive_role=defensive_role,
        traits=PlayerTraitProfile(
            ball_security=_to_rating((projection.projection_confidence * 0.45) + (0.35 * (1.0 - projection.volatility)) + (0.20 * min(1.0, assist_per_minute * 2.0))),
            separation=_to_rating((scoring_per_minute * 0.55) + (0.25 * min(1.0, assist_per_minute * 1.8)) + (0.20 * projection.projection_confidence)),
            burst=_to_rating((scoring_per_minute * 0.6) + (0.4 * min(1.0, projection.points / max(projection.salary / 350.0, 1.0)))),
            finishing=_to_rating((scoring_per_minute * 0.7) + (rebound_per_minute * 0.3)),
            pullup_shooting=_to_rating((scoring_per_minute * 0.75) + (projection.volatility * 0.25)),
            catch_shoot=_to_rating((projection.median_fpts / max(projection.salary / 1000.0, 1.0)) / 7.0),
            pass_vision=_to_rating((assist_per_minute * 0.8) + (projection.projection_confidence * 0.2)),
            pass_accuracy=_to_rating((assist_per_minute * 0.65) + ((1.0 - projection.volatility) * 0.35)),
            decision_making=_to_rating((projection.projection_confidence * 0.55) + ((1.0 - projection.volatility) * 0.25) + (min(1.0, assist_per_minute * 1.2) * 0.2)),
            screen_setting=_screen_value(projection),
            rebounding=_to_rating(rebound_per_minute * 1.7),
            free_throw_rating=_to_rating(ft_pct_raw),
            ft_pct_raw=ft_pct_raw,
            foul_drawing=_to_rating((scoring_per_minute * 0.65) + (projection.volatility * 0.15) + (projection.projection_confidence * 0.2)),
            containment=_perimeter_defense_value(projection),
            closeout=_closeout_value(projection),
            screen_nav=_screen_navigation_value(projection),
            interior_def=_interior_defense_value(projection),
            rim_protect=_rim_protection_value(projection),
            steal_pressure=_to_rating((projection.volatility * 0.2) + (projection.projection_confidence * 0.35) + (0.45 if {"PG", "SG", "SF", "G"} & positions else 0.2)),
            foul_discipline=_to_rating((1.0 - projection.volatility) * 0.65 + (projection.projection_confidence * 0.35)),
            help_rotation=_to_rating((_interior_defense_value(projection) / 20.0 * 0.55) + (_closeout_value(projection) / 20.0 * 0.45)),
            stamina=_to_rating((projection.recent_minutes_avg / 36.0) * projection.role_stability),
            role_consistency=_to_rating(projection.role_stability),
            clutch=10.0,
            size=_size_value(projection),
            reach=_reach_value(projection),
        ),
        condition=PlayerCondition(
            energy=_bounded(projection.projection_confidence, 0.2, 1.0),
            fatigue=_bounded(1.0 - (projection.recent_minutes_avg / 36.0), 0.0, 0.7),
            foul_count=0,
            confidence=_bounded((projection.recent_fpts_weighted / max(projection.recent_fpts_avg, 1.0)) * 0.5, 0.2, 0.9),
            minutes_played=0.0,
            available=projection.availability_status != "out",
        ),
    )


def _build_team_tactics(
    team_code: str,
    team_players: tuple[PlayerSimProfile, ...],
    team_features: dict[str, float],
    opponent_features: dict[str, float],
) -> TeamTactics:
    pace_target = float(team_features.get("recent_pace") or 99.0)
    avg_usage = _avg(player.traits.offensive_load for player in team_players)
    avg_creation = _avg(((player.traits.pass_vision + player.traits.pass_accuracy + player.traits.decision_making) / 3.0) for player in team_players)
    avg_screen = _avg(player.traits.screen_setting for player in team_players)
    avg_switchability = _avg(((player.traits.containment + player.traits.closeout + player.traits.screen_nav) / 3.0) for player in team_players)
    opp_orb = float(opponent_features.get("opp_orb_rate_allowed") or 0.28)
    transition_frequency = _bounded((pace_target - 94.0) / 14.0, 0.08, 0.28)
    pick_and_roll_rate = _bounded((avg_creation * 0.42) + (avg_screen * 0.22), 0.18, 0.48)
    handoff_rate = _bounded(0.08 + ((avg_creation - 0.4) * 0.12), 0.04, 0.18)
    post_touch_rate = _bounded(0.06 + (_avg(player.traits.finishing for player in team_players) / 20.0 * 0.08), 0.03, 0.16)
    off_ball_screen_rate = _bounded(0.1 + (_avg(player.traits.catch_shoot for player in team_players) / 20.0 * 0.08), 0.06, 0.2)
    iso_rate = _bounded(0.06 + (avg_usage * 0.18), 0.05, 0.22)
    reset_rate = _bounded(0.14 - ((avg_creation - 0.45) * 0.08), 0.08, 0.2)
    switch_rate = _bounded((avg_switchability * 0.42), 0.12, 0.48)
    drop_rate = _bounded(0.48 + ((0.55 - avg_switchability) * 0.24), 0.2, 0.62)
    crash_glass_rate = _bounded(0.16 + ((opp_orb - 0.26) * 0.6), 0.1, 0.3)
    sorted_loads = sorted((player.traits.offensive_load for player in team_players), reverse=True)
    load_gap = (sorted_loads[0] - sorted_loads[1]) if len(sorted_loads) > 1 else 0.0
    star_usage_bias = _bounded(1.0 + (load_gap / 12.0), 1.0, 1.55)
    closeout_attack_rate = _bounded(
        0.34
        + (_avg((player.traits.separation + player.traits.burst) / 2.0 for player in team_players) / 20.0) * 0.22
        - (_avg(player.traits.catch_shoot for player in team_players) / 20.0) * 0.08,
        0.26,
        0.64,
    )
    second_side_rate = _bounded(
        0.14
        + (_avg((player.traits.pass_vision + player.traits.pass_accuracy) / 2.0 for player in team_players) / 20.0) * 0.18
        + (_avg(player.traits.decision_making for player in team_players) / 20.0) * 0.08,
        0.12,
        0.42,
    )
    spacer_share = sum(1 for player in team_players if player.offensive_role in {OffensiveRole.SPACER, OffensiveRole.MOVEMENT_SHOOTER}) / max(len(team_players), 1)
    corner_spacing_bias = _bounded(
        0.42
        + spacer_share * 0.18
        + (_avg(player.traits.catch_shoot for player in team_players) / 20.0) * 0.08
        - (_avg(player.traits.size for player in team_players) / 20.0) * 0.06,
        0.30,
        0.72,
    )
    shooter_distribution_weights = {
        player.player_id: _bounded(
            0.88
            + (player.traits.catch_shoot / 20.0) * 0.22
            + (0.12 if player.offensive_role in {OffensiveRole.SPACER, OffensiveRole.MOVEMENT_SHOOTER} else 0.0)
            - (0.08 if player.offensive_role == OffensiveRole.ROLL_BIG else 0.0),
            0.72,
            1.28,
        )
        for player in team_players
    }
    return TeamTactics(
        pace_target=pace_target,
        transition_frequency=transition_frequency,
        crash_glass_rate=crash_glass_rate,
        help_aggressiveness=_bounded(0.42 + ((_avg(player.traits.containment for player in team_players) / 20.0) * 0.12), 0.28, 0.72),
        switch_rate=switch_rate,
        zone_rate=0.03,
        no_middle_rate=0.1,
        pre_switch_rate=_bounded(switch_rate * 0.38, 0.04, 0.2),
        rotation_tightness=0.72,
        late_clock_isolation_rate=_bounded(iso_rate * 0.9, 0.08, 0.22),
        early_offense_rate=_bounded(transition_frequency * 0.9, 0.08, 0.24),
        pick_and_roll_rate=pick_and_roll_rate,
        handoff_rate=handoff_rate,
        post_touch_rate=post_touch_rate,
        off_ball_screen_rate=off_ball_screen_rate,
        play_family_weights={
            PlayFamily.HIGH_PICK_AND_ROLL: pick_and_roll_rate,
            PlayFamily.ISO: iso_rate,
            PlayFamily.HANDOFF: handoff_rate,
            PlayFamily.POST_TOUCH: post_touch_rate,
            PlayFamily.RESET: reset_rate,
        },
        coverage_weights={
            DefensiveCoverage.DROP: drop_rate,
            DefensiveCoverage.SWITCH: switch_rate,
        },
        star_usage_bias=star_usage_bias,
        closeout_attack_rate=closeout_attack_rate,
        second_side_rate=second_side_rate,
        corner_spacing_bias=corner_spacing_bias,
        shooter_distribution_weights=shooter_distribution_weights,
    )


def _build_rotation_plan(team_players: tuple[PlayerSimProfile, ...]) -> RotationPlan:
    ordered = sorted(
        team_players,
        key=lambda item: (
            item.condition.available,
            item.condition.energy,
            item.traits.offensive_load + item.traits.decision_making + item.traits.stamina,
        ),
        reverse=True,
    )
    starters = tuple(player.player_id for player in ordered[:5])
    closers = tuple(player.player_id for player in ordered[:5])
    target_minutes = {
        player.player_id: _bounded_minutes(player)
        for player in ordered[:10]
    }
    return RotationPlan(
        starters=starters,
        closing_group=closers,
        target_minutes=target_minutes,
        max_stint_minutes={player.player_id: min(12.0, target_minutes[player.player_id] / 2.0 + 2.0) for player in ordered[:10]},
        backup_priority={starter: tuple(player.player_id for player in ordered[5:10]) for starter in starters},
    )


def _infer_offensive_role(projection: DraftKingsNBAProjection) -> OffensiveRole:
    positions = set(projection.positions) | set(projection.roster_positions)
    if {"PG", "G"} & positions and projection.assists >= 6.0:
        return OffensiveRole.PRIMARY_CREATOR
    if {"PG", "SG", "G"} & positions and projection.assists >= 4.0:
        return OffensiveRole.SECONDARY_CREATOR
    if "C" in positions and projection.assists >= 4.0:
        return OffensiveRole.POST_HUB
    if "C" in positions:
        return OffensiveRole.ROLL_BIG
    if projection.salary <= 4500:
        return OffensiveRole.SPACER
    if {"SF", "PF", "F"} & positions:
        return OffensiveRole.SLASHER
    return OffensiveRole.GLUE


def _infer_defensive_role(projection: DraftKingsNBAProjection) -> DefensiveRole:
    positions = set(projection.positions) | set(projection.roster_positions)
    if "C" in positions:
        return DefensiveRole.RIM_PROTECTOR
    if {"PF", "F"} & positions:
        return DefensiveRole.HELPER
    if {"SF", "SG"} & positions:
        return DefensiveRole.WING_STOPPER
    return DefensiveRole.POINT_OF_ATTACK


def _screen_value(projection: DraftKingsNBAProjection) -> float:
    positions = set(projection.positions) | set(projection.roster_positions)
    base = 12.0 if "C" in positions else 10.0 if {"PF", "F"} & positions else 7.0 if "SF" in positions else 5.0
    return _bounded(base + ((projection.rebounds / max(projection.minutes, 1.0)) * 4.0), 1.0, 20.0)


def _perimeter_defense_value(projection: DraftKingsNBAProjection) -> float:
    positions = set(projection.positions) | set(projection.roster_positions)
    base = 0.62 if {"PG", "SG", "SF", "G"} & positions else 0.45
    return _to_rating(base + (projection.projection_confidence - 0.5) * 0.25)


def _closeout_value(projection: DraftKingsNBAProjection) -> float:
    return _to_rating((_perimeter_defense_value(projection) / 20.0 * 0.7) + (projection.projection_confidence * 0.3))


def _screen_navigation_value(projection: DraftKingsNBAProjection) -> float:
    return _bounded(_perimeter_defense_value(projection) - 2.0, 1.0, 20.0)


def _interior_defense_value(projection: DraftKingsNBAProjection) -> float:
    positions = set(projection.positions) | set(projection.roster_positions)
    base = 0.68 if "C" in positions else 0.42
    return _to_rating(base + ((projection.rebounds / max(projection.minutes, 1.0)) * 0.35))


def _rim_protection_value(projection: DraftKingsNBAProjection) -> float:
    positions = set(projection.positions) | set(projection.roster_positions)
    base = 0.62 if "C" in positions else 0.28
    return _to_rating(base + ((projection.rebounds / max(projection.minutes, 1.0)) * 0.2))


def _size_value(projection: DraftKingsNBAProjection) -> float:
    positions = set(projection.positions) | set(projection.roster_positions)
    if "C" in positions:
        return 17.0
    if {"PF", "F"} & positions:
        return 13.0
    if {"SF"} & positions:
        return 11.0
    return 8.0


def _reach_value(projection: DraftKingsNBAProjection) -> float:
    positions = set(projection.positions) | set(projection.roster_positions)
    if "C" in positions:
        return 17.0
    if {"PF", "F", "SF"} & positions:
        return 13.0
    return 10.0


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _to_rating(value: float) -> float:
    return _bounded(1.0 + (float(value) * 19.0), 1.0, 20.0)


def _avg(values) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / max(len(values), 1)


def _bounded_minutes(player: PlayerSimProfile) -> float:
    estimated = 18.0 + (player.condition.energy * 8.0) + ((player.traits.stamina / 20.0) * 8.0)
    return _bounded(estimated, 10.0, 38.0)
