from __future__ import annotations

import random
from enum import Enum
from typing import Any, TypeVar

from basketball_calibration import build_calibration_game_input
from basketball_choreography import build_match_choreography
from basketball_game_engine import _assignments, _floor_states, _lineup_from_ids, simulate_game
from basketball_possession_engine import simulate_possession
from basketball_sim_builder import _build_rotation_plan, _build_team_tactics, build_nba_sim_inputs_from_dk_csv
from basketball_sim_schema import (
    DefensiveCoverage,
    EventType,
    GameSimulationInput,
    OffensiveRole,
    PlayerCondition,
    PlayerSimProfile,
    PlayerTraitProfile,
    EntrySource,
    EntryType,
    PlayCall,
    PlayFamily,
    PossessionContext,
    PossessionPhase,
    ScoreState,
)
from nba_matchup_features import load_nba_matchup_features
from dfs_ingest import parse_draftkings_salary_csv
from quantum_parlay_oracle import (
    load_game_context_snapshot,
    load_nba_matchup_profile_snapshot,
    nba_availability_status_lookup,
    team_name_from_code,
)
from refresh_slate import fetch_nba_game_contexts
from data_pipeline.nba_profiles import fetch_nba_team_player_profiles


SUPPORTED_PLAY_FAMILIES = [PlayFamily.HIGH_PICK_AND_ROLL, PlayFamily.DOUBLE_DRAG, PlayFamily.HANDOFF, PlayFamily.ISO]
SUPPORTED_COVERAGES = [DefensiveCoverage.DROP, DefensiveCoverage.SWITCH, DefensiveCoverage.ICE, DefensiveCoverage.HEDGE]
SUPPORTED_ENTRY_TYPES = [EntryType.NORMAL, EntryType.TRANSITION, EntryType.OREB]
SUPPORTED_ENTRY_SOURCES = [
    EntrySource.DEAD_BALL,
    EntrySource.LIVE_TURNOVER_BREAK,
    EntrySource.DEFENSIVE_REBOUND_PUSH,
    EntrySource.MADE_BASKET_FLOW,
    EntrySource.OREB_GATHER,
]
SUPPORTED_VIEW_MODES = ["game", "single"]
SUPPORTED_DATA_MODES = ["calibration", "live", "draftkings_csv"]
EnumT = TypeVar("EnumT", bound=Enum)


def default_viewer_form() -> dict[str, Any]:
    return {
        "view_mode": "game",
        "data_mode": "calibration",
        "date": "",
        "matchup": "",
        "csv_path": "/Users/brianross/Downloads/DKSalaries.csv",
        "seed": 7,
        "play_family": PlayFamily.HIGH_PICK_AND_ROLL.value,
        "coverage": DefensiveCoverage.DROP.value,
        "entry_type": EntryType.NORMAL.value,
        "entry_source": EntrySource.DEAD_BALL.value,
        "offense_team": "HOM",
    }


def build_possession_view_payload(
    *,
    view_mode: str,
    data_mode: str,
    date: str | None,
    matchup: str | None,
    csv_path: str | None,
    seed: int,
    play_family: str,
    coverage: str,
    entry_type: str,
    entry_source: str,
    offense_team: str = "HOM",
) -> dict[str, Any]:
    sim_input, source_meta = _load_sim_input(
        view_mode=view_mode,
        data_mode=data_mode,
        date=date,
        matchup=matchup,
        csv_path=csv_path,
    )
    play_family_enum = _coerce_enum(PlayFamily, play_family)
    coverage_enum = _coerce_enum(DefensiveCoverage, coverage)
    entry_type_enum = _coerce_enum(EntryType, entry_type)
    entry_source_enum = _coerce_enum(EntrySource, entry_source)
    player_lookup = {player.player_id: player for player in sim_input.players}
    home_lineup_ids = sim_input.home_rotation.starters
    away_lineup_ids = sim_input.away_rotation.starters
    offense_home = offense_team == sim_input.home_team_code
    offense_ids = home_lineup_ids if offense_home else away_lineup_ids
    defense_ids = away_lineup_ids if offense_home else home_lineup_ids
    offense_lineup = _lineup_from_ids(sim_input.home_team_code if offense_home else sim_input.away_team_code, offense_ids, player_lookup)
    defense_lineup = _lineup_from_ids(sim_input.away_team_code if offense_home else sim_input.home_team_code, defense_ids, player_lookup)
    offensive_tactics = sim_input.home_tactics if offense_home else sim_input.away_tactics
    defensive_tactics = sim_input.away_tactics if offense_home else sim_input.home_tactics

    creator = max(
        (player_lookup[player_id] for player_id in offense_ids if player_id in player_lookup),
        key=lambda player: player.traits.offensive_load,
    )
    screener = None
    secondary_screener = None
    if play_family_enum in {PlayFamily.HIGH_PICK_AND_ROLL, PlayFamily.DOUBLE_DRAG, PlayFamily.HANDOFF}:
        ordered_screeners = sorted(
            (player_lookup[player_id] for player_id in offense_ids if player_id != creator.player_id and player_id in player_lookup),
            key=lambda player: player.traits.screen_setting,
            reverse=True,
        )
        screener = ordered_screeners[0] if ordered_screeners else None
        secondary_screener = ordered_screeners[1] if play_family_enum == PlayFamily.DOUBLE_DRAG and len(ordered_screeners) > 1 else None

    context = PossessionContext(
        offense_team_code=offense_lineup.team_code,
        defense_team_code=defense_lineup.team_code,
        clock=sim_input and _base_clock(),
        score=ScoreState(offense_score=54, defense_score=51),
        offense_lineup=offense_lineup,
        defense_lineup=defense_lineup,
        offensive_tactics=offensive_tactics,
        defensive_tactics=defensive_tactics,
        floor_players=_floor_states(offense_ids, defense_ids),
        defensive_assignments=_assignments(offense_ids, defense_ids),
        player_pool=sim_input.players,
        current_phase=PossessionPhase.PRIMARY_ACTION,
        play_call=PlayCall(
            family=play_family_enum,
            primary_actor_id=creator.player_id,
            secondary_actor_id=secondary_screener.player_id if secondary_screener is not None else None,
            screener_id=screener.player_id if screener is not None else None,
        ),
        coverage=coverage_enum,
        entry_type=entry_type_enum,
        entry_source=entry_source_enum,
    )
    if view_mode == "game":
        game_result = simulate_game(sim_input, rng_seed=seed)
        beats = _build_game_match_beats(game_result, player_lookup)
        home_display_name = _team_display_name(sim_input.home_team_code)
        away_display_name = _team_display_name(sim_input.away_team_code)
        initial_scoreboard = {
            "period": 1,
            "clock_display": "12:00",
            "shot_clock": 24.0,
            "offense_score": 0,
            "defense_score": 0,
        }
        summary = {
            "offense_team": sim_input.home_team_code,
            "defense_team": sim_input.away_team_code,
            "entry_type": "game_sequence",
            "entry_source": source_meta["data_mode"],
            "points_scored": game_result.home_score + game_result.away_score,
            "made_shot": False,
            "turnover": False,
            "foul_committed": False,
            "offensive_rebound": False,
            "shooter": "",
            "assister": "",
            "rebounder": "",
            "turnover_player": "",
        }
        initial_positions = _build_initial_positions_for_matchup(sim_input, home_lineup_ids, away_lineup_ids)
        raw_events = _serialize_events(game_result.event_log, player_lookup)
        match = {
            "home_team": sim_input.home_team_code,
            "away_team": sim_input.away_team_code,
            "home_team_name": home_display_name,
            "away_team_name": away_display_name,
            "initial_scoreboard": initial_scoreboard,
            "beats": beats,
        }
    else:
        outcome = simulate_possession(context, rng=random.Random(seed))
        beats = _build_match_beats(outcome.events, player_lookup, context)
        home_display_name = _team_display_name(context.offense_team_code)
        away_display_name = _team_display_name(context.defense_team_code)
        initial_scoreboard = {
            "period": context.clock.period,
            "clock_display": _format_clock(context.clock.seconds_remaining_in_period),
            "shot_clock": round(context.clock.shot_clock, 1),
            "offense_score": context.score.offense_score,
            "defense_score": context.score.defense_score,
        }
        summary = {
            "offense_team": context.offense_team_code,
            "defense_team": context.defense_team_code,
            "entry_type": context.entry_type.value,
            "entry_source": context.entry_source.value,
            "points_scored": outcome.points_scored,
            "made_shot": outcome.made_shot,
            "turnover": outcome.turnover,
            "foul_committed": outcome.foul_committed,
            "offensive_rebound": outcome.offensive_rebound,
            "shooter": _player_name(player_lookup, outcome.shooting_player_id),
            "assister": _player_name(player_lookup, outcome.assisting_player_id),
            "rebounder": _player_name(player_lookup, outcome.rebounder_id),
            "turnover_player": _player_name(player_lookup, outcome.turnover_player_id),
        }
        initial_positions = [
            {
                "player_id": state.player_id,
                "side": state.side.value,
                "x": state.location.x,
                "y": state.location.y,
                "zone": state.location.zone.value,
                "has_ball": state.has_ball,
            }
            for state in context.floor_players
        ]
        raw_events = _serialize_events(outcome.events, player_lookup)
        match = {
            "home_team": context.offense_team_code,
            "away_team": context.defense_team_code,
            "home_team_name": home_display_name,
            "away_team_name": away_display_name,
            "initial_scoreboard": initial_scoreboard,
            "beats": beats,
        }
    choreography = build_match_choreography(
        players=[
            {
                "player_id": player.player_id,
                "name": player.name,
                "team_code": player.team_code,
                "side": "offense" if player.player_id in offense_ids else "defense",
                "offensive_role": player.offensive_role.value,
                "defensive_role": player.defensive_role.value,
            }
            for player in sim_input.players
            if player.player_id in offense_ids or player.player_id in defense_ids
        ],
        initial_positions=initial_positions,
        beats=beats,
        events=raw_events,
        home_team=match["home_team"],
        away_team=match["away_team"],
        assignments=[
            {
                "defender_id": assignment.defender_id,
                "offensive_player_id": assignment.offensive_player_id,
                "matchup_strength": assignment.matchup_strength,
                "on_ball": assignment.on_ball,
                "help_priority": assignment.help_priority,
            }
            for assignment in _assignments(offense_ids, defense_ids)
        ],
    )
    return {
        "seed": seed,
        "form": {
            "view_mode": view_mode,
            "data_mode": data_mode,
            "date": date or "",
            "matchup": matchup or "",
            "csv_path": csv_path or "",
            "play_family": play_family,
            "coverage": coverage,
            "entry_type": entry_type,
            "entry_source": entry_source,
            "offense_team": offense_team,
        },
        "summary": summary,
        "match": match,
        "choreography": choreography,
        "source_meta": source_meta,
        "players": [
            {
                "player_id": player.player_id,
                "name": player.name,
                "team_code": player.team_code,
                "side": "offense" if player.player_id in offense_ids else "defense",
                "offensive_role": player.offensive_role.value,
                "defensive_role": player.defensive_role.value,
            }
            for player in sim_input.players
            if player.player_id in offense_ids or player.player_id in defense_ids
        ],
        "initial_positions": initial_positions,
        "events": raw_events,
        "options": {
            "view_modes": SUPPORTED_VIEW_MODES,
            "data_modes": SUPPORTED_DATA_MODES,
            "play_families": [item.value for item in SUPPORTED_PLAY_FAMILIES],
            "coverages": [item.value for item in SUPPORTED_COVERAGES],
            "entry_types": [item.value for item in SUPPORTED_ENTRY_TYPES],
            "entry_sources": [item.value for item in SUPPORTED_ENTRY_SOURCES],
            "offense_teams": [sim_input.home_team_code, sim_input.away_team_code],
            "matchups": _available_matchups(data_mode=data_mode, date_str=date or "", csv_path=csv_path or ""),
        },
    }


def _base_clock():
    from basketball_sim_schema import GameClockState

    return GameClockState(period=1, seconds_remaining_in_period=420.0, shot_clock=14.0, possession_number=12)


def _player_name(player_lookup: dict[str, Any], player_id: str | None) -> str:
    if not player_id:
        return ""
    player = player_lookup.get(player_id)
    return player.name if player is not None else str(player_id)


def _team_display_name(team_code: str) -> str:
    return team_name_from_code(team_code) or str(team_code)


def _coerce_enum(enum_cls: type[EnumT], raw_value: str) -> EnumT:
    try:
        return enum_cls(raw_value)
    except ValueError:
        normalized = raw_value.strip().upper()
        try:
            return enum_cls[normalized]
        except KeyError as exc:
            supported = ", ".join(item.value for item in enum_cls)
            raise ValueError(f"Unsupported {enum_cls.__name__}: {raw_value}. Expected one of: {supported}") from exc


def _load_sim_input(*, view_mode: str, data_mode: str, date: str | None, matchup: str | None, csv_path: str | None) -> tuple[GameSimulationInput, dict[str, Any]]:
    if data_mode == "draftkings_csv":
        return _build_dk_csv_game_input(date_str=date or "", matchup=matchup or "", csv_path=csv_path or "")
    if data_mode == "live":
        return _build_live_game_input(date_str=date or "", matchup=matchup or "")
    sim_input = build_calibration_game_input()
    return sim_input, {
        "data_mode": "calibration",
        "date": "",
        "matchup": sim_input.game_id,
        "title": "Calibration Fixture",
    }


def _available_matchups(*, data_mode: str, date_str: str, csv_path: str) -> list[str]:
    if data_mode == "draftkings_csv" and csv_path:
        try:
            slate = parse_draftkings_salary_csv(csv_path, sport="nba")
            return sorted({player.game for player in slate.players if player.game and "@" in player.game})
        except Exception:
            return []
    if data_mode == "live" and date_str:
        try:
            return [item["matchup"] for item in fetch_nba_game_contexts(date_str)]
        except Exception:
            return []
    return []


def _build_live_game_input(*, date_str: str, matchup: str) -> tuple[GameSimulationInput, dict[str, Any]]:
    if not date_str:
        raise ValueError("Live mode needs a slate date.")
    contexts = fetch_nba_game_contexts(date_str)
    if not contexts:
        raise ValueError(f"No live NBA contexts found for {date_str}.")
    selected = next((item for item in contexts if item["matchup"] == matchup), contexts[0] if contexts else None)
    if selected is None:
        raise ValueError(f"No live NBA matchup available for {date_str}.")
    matchup = selected["matchup"]
    away_code, home_code = matchup.split("@", 1)
    context_snapshot = load_game_context_snapshot(date_str, "nba", matchup) or selected
    profile_snapshot = load_nba_matchup_profile_snapshot(date_str, matchup) or {}
    away_profiles = list(profile_snapshot.get("away_profiles") or [])
    home_profiles = list(profile_snapshot.get("home_profiles") or [])
    if not away_profiles:
        away_profiles = fetch_nba_team_player_profiles(str(selected["away_team_id"]), date_str)
    if not home_profiles:
        home_profiles = fetch_nba_team_player_profiles(str(selected["home_team_id"]), date_str)
    if len(away_profiles) < 5 or len(home_profiles) < 5:
        raise ValueError(f"Not enough live player profiles to build {matchup}.")
    team_features = load_nba_matchup_features(date_str, (away_code, home_code))
    availability = (context_snapshot.get("availability") or {}) if context_snapshot else {}
    away_players = tuple(_profile_to_sim_player(profile, away_code, (availability.get("away") or [])) for profile in away_profiles)
    home_players = tuple(_profile_to_sim_player(profile, home_code, (availability.get("home") or [])) for profile in home_profiles)
    players = away_players + home_players
    sim_input = GameSimulationInput(
        game_id=matchup,
        home_team_code=home_code,
        away_team_code=away_code,
        players=players,
        home_tactics=_build_team_tactics(home_code, home_players, team_features.get(home_code, {}), team_features.get(away_code, {})),
        away_tactics=_build_team_tactics(away_code, away_players, team_features.get(away_code, {}), team_features.get(home_code, {})),
        home_rotation=_build_rotation_plan(home_players),
        away_rotation=_build_rotation_plan(away_players),
        opening_tip_winner=home_code,
    )
    return sim_input, {
        "data_mode": "live",
        "date": date_str,
        "matchup": matchup,
        "title": f"{away_code} @ {home_code}",
    }


def _build_dk_csv_game_input(*, date_str: str, matchup: str, csv_path: str) -> tuple[GameSimulationInput, dict[str, Any]]:
    if not csv_path:
        raise ValueError("DraftKings CSV mode needs a CSV path.")
    if not date_str:
        raise ValueError("DraftKings CSV mode needs a slate date.")
    sim_inputs = build_nba_sim_inputs_from_dk_csv(date_str, csv_path)
    if not sim_inputs:
        raise ValueError(f"No NBA games could be built from {csv_path}.")
    selected = next((item for item in sim_inputs if item.game_id == matchup), sim_inputs[0])
    return selected, {
        "data_mode": "draftkings_csv",
        "date": date_str,
        "matchup": selected.game_id,
        "title": f"DraftKings CSV · {selected.game_id}",
    }


def _profile_to_sim_player(profile: dict[str, Any], team_code: str, availability_entries: list[dict]) -> PlayerSimProfile:
    status_lookup = nba_availability_status_lookup(availability_entries)
    name = str(profile.get("name", "")).strip() or f"{team_code}_player"
    status = status_lookup.get(name.lower(), str(profile.get("status", "active")).strip().lower())
    position = str(profile.get("position") or "G")
    positions = (position,)
    minutes = max(float(profile.get("minutes", 18.0)), 8.0)
    points = max(float(profile.get("points", 6.0)), 0.5)
    rebounds = max(float(profile.get("rebounds", 2.0)), 0.2)
    assists = max(float(profile.get("assists", 1.5)), 0.2)
    games_sample = max(float(profile.get("games_sample", 1.0)), 1.0)
    offensive_role = _infer_live_offensive_role(position, points, assists)
    available = status not in {"out"}
    energy = 0.9 if available else 0.0
    return PlayerSimProfile(
        player_id=str(profile.get("player_id") or name),
        name=name,
        team_code=team_code,
        positions=positions,
        offensive_role=offensive_role,
        defensive_role=_infer_live_defensive_role(position),
        traits=PlayerTraitProfile(
            ball_security=_rating((assists / max(minutes, 1.0)) * 1.6 + 0.55),
            separation=_rating((points / max(minutes, 1.0)) * 1.8 + 0.45),
            burst=_rating((points / max(minutes, 1.0)) * 1.7 + 0.40),
            pullup_shooting=_rating((points / max(minutes, 1.0)) * 1.7 + 0.35),
            catch_shoot=_rating((points / max(minutes, 1.0)) * 1.4 + 0.50),
            finishing=_rating((points / max(minutes, 1.0)) * 1.5 + 0.50),
            pass_vision=_rating((assists / max(minutes, 1.0)) * 2.0 + 0.45),
            pass_accuracy=_rating((assists / max(minutes, 1.0)) * 1.8 + 0.42),
            decision_making=_rating((games_sample / 8.0) * 0.3 + 0.5),
            screen_setting=_screen_rating(position, rebounds),
            oreb=_rating((rebounds / max(minutes, 1.0)) * 1.2 + (0.25 if position == "C" else 0.05)),
            free_throw_rating=_rating(0.62),
            ft_pct_raw=0.78,
            foul_drawing=_rating((points / max(minutes, 1.0)) * 1.5 + 0.35),
            containment=_defense_rating(position),
            closeout=_defense_rating(position) - 1.0,
            screen_nav=max(1.0, _defense_rating(position) - 2.0),
            interior_def=_interior_rating(position, rebounds),
            rim_protect=_rim_rating(position, rebounds),
            steal_pressure=_rating(0.48 if position in {"PG", "SG", "SF"} else 0.34),
            dreb=_rating((rebounds / max(minutes, 1.0)) * 1.4 + 0.20),
            foul_discipline=_rating(0.60),
            help_rotation=_rating(0.58),
            stamina=_rating(min(minutes / 34.0, 1.0)),
            role_consistency=_rating(min(games_sample / 8.0, 1.0)),
            clutch=10.0,
            size=_size_rating(position),
            reach=_reach_rating(position),
        ),
        condition=PlayerCondition(
            energy=energy,
            fatigue=max(0.0, 1.0 - min(minutes / 36.0, 1.0)),
            foul_count=0,
            confidence=0.55,
            minutes_played=0.0,
            available=available,
        ),
    )


def _infer_live_offensive_role(position: str, points: float, assists: float) -> OffensiveRole:
    if position in {"PG", "G"} and assists >= 5.5:
        return OffensiveRole.PRIMARY_CREATOR
    if position in {"PG", "SG", "G"} and assists >= 3.5:
        return OffensiveRole.SECONDARY_CREATOR
    if position == "C" and assists >= 3.5:
        return OffensiveRole.POST_HUB
    if position == "C":
        return OffensiveRole.ROLL_BIG
    if points >= 18.0:
        return OffensiveRole.SLASHER
    return OffensiveRole.SPACER


def _infer_live_defensive_role(position: str):
    from basketball_sim_schema import DefensiveRole
    if position == "C":
        return DefensiveRole.RIM_PROTECTOR
    if position in {"PF", "F"}:
        return DefensiveRole.HELPER
    if position in {"SF", "SG"}:
        return DefensiveRole.WING_STOPPER
    return DefensiveRole.POINT_OF_ATTACK


def _rating(value: float) -> float:
    return max(1.0, min(20.0, 1.0 + (float(value) * 19.0)))


def _size_rating(position: str) -> float:
    return {"C": 17.0, "PF": 14.0, "F": 13.0, "SF": 11.0}.get(position, 8.5)


def _reach_rating(position: str) -> float:
    return {"C": 17.0, "PF": 14.0, "F": 13.0, "SF": 12.0}.get(position, 10.0)


def _screen_rating(position: str, rebounds: float) -> float:
    base = {"C": 14.0, "PF": 11.5, "F": 10.5, "SF": 8.5}.get(position, 6.5)
    return max(1.0, min(20.0, base + (rebounds * 0.25)))


def _defense_rating(position: str) -> float:
    return {"PG": 11.5, "G": 11.5, "SG": 11.0, "SF": 11.2, "PF": 10.2, "F": 10.4, "C": 9.0}.get(position, 10.0)


def _interior_rating(position: str, rebounds: float) -> float:
    base = {"C": 14.5, "PF": 11.5, "F": 10.5}.get(position, 8.0)
    return max(1.0, min(20.0, base + (rebounds * 0.18)))


def _rim_rating(position: str, rebounds: float) -> float:
    base = {"C": 14.0, "PF": 10.0, "F": 8.5}.get(position, 5.5)
    return max(1.0, min(20.0, base + (rebounds * 0.14)))


def _serialize_events(events: tuple[Any, ...], player_lookup: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": idx,
            "event_type": event.event_type.value,
            "actor_id": event.actor_id,
            "actor_name": _player_name(player_lookup, event.actor_id),
            "receiver_id": event.receiver_id,
            "receiver_name": _player_name(player_lookup, event.receiver_id),
            "defender_id": event.defender_id,
            "defender_name": _player_name(player_lookup, event.defender_id),
            "location": {
                "x": event.location.x if event.location else None,
                "y": event.location.y if event.location else None,
                "zone": event.location.zone.value if event.location else None,
            },
            "shot_type": event.shot_type.value if event.shot_type else None,
            "turnover_type": event.turnover_type.value if event.turnover_type else None,
            "success_probability": event.success_probability,
            "realized_success": event.realized_success,
            "points_scored": event.points_scored,
            "foul_drawn": event.foul_drawn,
            "notes": event.notes,
        }
        for idx, event in enumerate(events)
    ]


def _build_initial_positions_for_matchup(sim_input: GameSimulationInput, home_ids: tuple[str, ...], away_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": state.player_id,
            "team_code": sim_input.home_team_code if state.player_id in home_ids else sim_input.away_team_code,
            "side": state.side.value,
            "x": state.location.x,
            "y": state.location.y,
            "zone": state.location.zone.value,
            "has_ball": state.has_ball,
        }
        for state in _floor_states(home_ids, away_ids)
    ]


def _build_match_beats(
    events: list[Any],
    player_lookup: dict[str, Any],
    context: PossessionContext,
    *,
    event_index_offset: int = 0,
    offense_team_code: str | None = None,
) -> list[dict[str, Any]]:
    beats: list[dict[str, Any]] = []
    otc = offense_team_code or context.offense_team_code
    seconds_remaining = float(context.clock.seconds_remaining_in_period)
    shot_clock = float(context.clock.shot_clock)
    offense_score = context.score.offense_score
    defense_score = context.score.defense_score
    last_ball_handler = next(
        (state.player_id for state in context.floor_players if state.side.value == "offense" and state.has_ball),
        context.play_call.primary_actor_id if context.play_call else None,
    )

    for idx, event in enumerate(events):
        elapsed = _event_elapsed_seconds(event)
        seconds_remaining = max(0.0, seconds_remaining - elapsed)
        shot_clock = max(0.0, shot_clock - elapsed)
        if event.points_scored:
            offense_score += int(event.points_scored)
        if event.event_type == EventType.REBOUND and event.actor_id and event.actor_id in context.defense_lineup.player_ids:
            last_ball_handler = None
        elif event.receiver_id:
            last_ball_handler = event.receiver_id
        elif event.actor_id and event.event_type in {EventType.DRIVE, EventType.SHOT, EventType.ADVANCE, EventType.HANDOFF, EventType.POST_ENTRY}:
            last_ball_handler = event.actor_id

        label, commentary = _describe_event(event, player_lookup)
        beats.append(
            {
                "index": idx,
                "event_apply_index": event_index_offset + idx,
                "offense_team_code": otc,
                "label": label,
                "commentary": commentary,
                "event_type": event.event_type.value,
                "focus_player_id": event.actor_id or event.receiver_id,
                "ball_player_id": last_ball_handler,
                "receiver_id": event.receiver_id,
                "defender_id": event.defender_id,
                "location": {
                    "x": event.location.x if event.location else None,
                    "y": event.location.y if event.location else None,
                    "zone": event.location.zone.value if event.location else None,
                },
                "clock_display": _format_clock(seconds_remaining),
                "shot_clock": round(shot_clock, 1),
                "offense_score": offense_score,
                "defense_score": defense_score,
                "duration_ms": _event_duration_ms(event),
            }
        )
    return beats


def _build_game_match_beats(game_result: Any, player_lookup: dict[str, Any]) -> list[dict[str, Any]]:
    beats: list[dict[str, Any]] = []
    home_score = 0
    away_score = 0
    beat_index = 0
    event_offset = 0
    previous_possession = None
    opening_possession = game_result.possessions[0] if game_result.possessions else None
    if opening_possession is not None:
        tip_positions = _build_tipoff_positions(game_result, player_lookup)
        tip_winner = _reset_ball_handler(tip_positions)
        beats.append(
            {
                "index": beat_index,
                "event_apply_index": -1,
                "offense_team_code": opening_possession.offense_team_code,
                "label": "Tip-Off",
                "commentary": f"{game_result.home_team_code} and {game_result.away_team_code} meet at center court",
                "event_type": "jump_ball_setup",
                "focus_player_id": None,
                "ball_player_id": None,
                "location": {"x": 0.0, "y": 47.0, "zone": "backcourt"},
                "clock_display": "12:00",
                "shot_clock": 24.0,
                "offense_score": 0,
                "defense_score": 0,
                "duration_ms": 1200,
                "reset_positions": tip_positions,
            }
        )
        beat_index += 1
        beats.append(
            {
                "index": beat_index,
                "event_apply_index": -1,
                "offense_team_code": opening_possession.offense_team_code,
                "label": "Tap",
                "commentary": f"{opening_possession.offense_team_code} controls the tip and brings it into flow",
                "event_type": "jump_ball",
                "focus_player_id": tip_winner,
                "ball_player_id": tip_winner,
                "location": {"x": 0.0, "y": 47.0, "zone": "backcourt"},
                "clock_display": "12:00",
                "shot_clock": 24.0,
                "offense_score": 0,
                "defense_score": 0,
                "duration_ms": 1000,
                "reset_positions": tip_positions,
            }
        )
        beat_index += 1
    for possession in game_result.possessions:
        if previous_possession is not None:
            transition_label, transition_commentary = _describe_possession_change(previous_possession, possession)
            transition_reset = _build_possession_reset_positions(
                game_result,
                player_lookup,
                possession.offense_team_code,
                possession.defense_team_code,
                stage="backcourt",
            )
            beats.append(
                {
                    "index": beat_index,
                    "event_apply_index": event_offset - 1,
                    "offense_team_code": possession.offense_team_code,
                    "label": transition_label,
                    "commentary": transition_commentary,
                    "event_type": "possession_change",
                    "focus_player_id": None,
                    "ball_player_id": _reset_ball_handler(transition_reset),
                    "location": {"x": 0.0, "y": 40.0, "zone": "backcourt"},
                    "clock_display": _format_clock(possession.start_clock),
                    "shot_clock": round(possession.start_shot_clock, 1),
                    "offense_score": home_score,
                    "defense_score": away_score,
                    "duration_ms": 900,
                    "reset_positions": transition_reset,
                }
            )
            beat_index += 1
        possession_label = (
            f"{possession.offense_team_code} transition"
            if possession.entry_type == EntryType.TRANSITION
            else f"{possession.offense_team_code} second chance"
            if possession.entry_type == EntryType.OREB
            else f"{possession.offense_team_code} halfcourt"
        )
        possession_reset = _build_possession_reset_positions(
            game_result,
            player_lookup,
            possession.offense_team_code,
            possession.defense_team_code,
            stage="advance",
        )
        beats.append(
            {
                "index": beat_index,
                "event_apply_index": event_offset - 1,
                "offense_team_code": possession.offense_team_code,
                "label": "Possession",
                "commentary": possession_label,
                "event_type": "possession_start",
                "focus_player_id": None,
                "ball_player_id": _reset_ball_handler(possession_reset),
                "location": {"x": 0.0, "y": 32.0, "zone": "backcourt"},
                "clock_display": _format_clock(possession.start_clock),
                "shot_clock": round(possession.start_shot_clock, 1),
                "offense_score": home_score,
                "defense_score": away_score,
                "duration_ms": 650,
                "reset_positions": possession_reset,
            }
        )
        beat_index += 1
        possession_context = type(
            "Ctx",
            (),
            {
                "offense_team_code": possession.offense_team_code,
                "clock": type("Clock", (), {"seconds_remaining_in_period": possession.start_clock, "shot_clock": possession.start_shot_clock})(),
                "score": type("Score", (), {"offense_score": possession.start_offense_score, "defense_score": possession.start_defense_score})(),
                "floor_players": (),
                "play_call": None,
                "defense_lineup": type("Lineup", (), {"player_ids": ()})(),
            },
        )()
        event_beats = _build_match_beats(
            list(possession.events),
            player_lookup,
            possession_context,
            event_index_offset=event_offset,
            offense_team_code=possession.offense_team_code,
        )
        for beat in event_beats:
            if possession.offense_team_code == game_result.home_team_code:
                beat["offense_score"] = beat["offense_score"] - possession.start_offense_score + home_score
                beat["defense_score"] = beat["defense_score"] - possession.start_defense_score + away_score
            else:
                beat["offense_score"] = home_score
                beat["defense_score"] = away_score
                if beat.get("event_type") == "shot" and any(event.realized_success for event in possession.events if event.event_type == EventType.SHOT):
                    beat["defense_score"] = away_score + max(0, possession.points_scored)
            beat["index"] = beat_index
            beats.append(beat)
            beat_index += 1
        event_offset += len(possession.events)
        if possession.offense_team_code == game_result.home_team_code:
            home_score = possession.end_offense_score
            away_score = possession.end_defense_score
        else:
            away_score = possession.end_offense_score
            home_score = possession.end_defense_score
        previous_possession = possession
    return beats


def _describe_possession_change(previous_possession: Any, next_possession: Any) -> tuple[str, str]:
    if any(event.event_type == EventType.TURNOVER for event in previous_possession.events):
        return "Turnover", f"{next_possession.offense_team_code} races the other way off the takeaway"
    if previous_possession.points_scored > 0:
        return "Inbound", f"{next_possession.offense_team_code} inbounds and flows into the next action"
    if any(event.event_type == EventType.REBOUND and "defensive_rebound" in (event.notes or "") for event in previous_possession.events):
        return "Outlet", f"{next_possession.offense_team_code} turns the rebound into the next push"
    return "Change", f"{next_possession.offense_team_code} takes over for the next possession"


def _build_possession_reset_positions(
    game_result: Any,
    player_lookup: dict[str, Any],
    offense_team_code: str,
    defense_team_code: str,
    *,
    stage: str,
) -> list[dict[str, Any]]:
    offense_ids = tuple(
        box.player_id
        for box in sorted(
            (box for box in game_result.player_box_scores if getattr(player_lookup.get(box.player_id), "team_code", None) == offense_team_code),
            key=lambda box: box.minutes,
            reverse=True,
        )[:5]
    )
    defense_ids = tuple(
        box.player_id
        for box in sorted(
            (box for box in game_result.player_box_scores if getattr(player_lookup.get(box.player_id), "team_code", None) == defense_team_code),
            key=lambda box: box.minutes,
            reverse=True,
        )[:5]
    )
    if not offense_ids or not defense_ids:
        return []
    base_states = list(_floor_states(offense_ids, defense_ids))
    offense_layout = (
        [
            (0.0, 40.0, "backcourt", True),
            (-15.0, 36.0, "left_wing", False),
            (15.0, 36.0, "right_wing", False),
            (-10.0, 30.0, "left_wing", False),
            (10.0, 28.0, "paint", False),
        ]
        if stage == "backcourt"
        else [
            (0.0, 32.0, "top", True),
            (-17.0, 26.0, "left_wing", False),
            (17.0, 26.0, "right_wing", False),
            (-21.0, 8.0, "left_corner", False),
            (0.0, 16.0, "paint", False),
        ]
    )
    defense_layout = (
        [
            (0.0, 27.0, "top"),
            (-16.0, 24.0, "left_wing"),
            (16.0, 24.0, "right_wing"),
            (-20.0, 10.0, "left_corner"),
            (0.0, 14.0, "paint"),
        ]
        if stage == "backcourt"
        else [
            (0.0, 24.0, "top"),
            (-16.0, 21.0, "left_wing"),
            (16.0, 21.0, "right_wing"),
            (-20.0, 8.0, "left_corner"),
            (0.0, 12.0, "paint"),
        ]
    )

    reset_positions: list[dict[str, Any]] = []
    offense_index = 0
    defense_index = 0
    for state in base_states:
        if state.player_id in offense_ids:
            x, y, zone, has_ball = offense_layout[min(offense_index, len(offense_layout) - 1)]
            offense_index += 1
            reset_positions.append(
                {
                    "player_id": state.player_id,
                    "team_code": offense_team_code,
                    "side": state.side.value,
                    "x": x,
                    "y": y,
                    "zone": zone,
                    "has_ball": has_ball,
                }
            )
        else:
            x, y, zone = defense_layout[min(defense_index, len(defense_layout) - 1)]
            defense_index += 1
            reset_positions.append(
                {
                    "player_id": state.player_id,
                    "team_code": defense_team_code,
                    "side": state.side.value,
                    "x": x,
                    "y": y,
                    "zone": zone,
                    "has_ball": False,
                }
            )
    return reset_positions


def _reset_ball_handler(reset_positions: list[dict[str, Any]]) -> str | None:
    for item in reset_positions:
        if item.get("has_ball"):
            return str(item["player_id"])
    return None


def _build_tipoff_positions(
    game_result: Any,
    player_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    home_ids = tuple(
        box.player_id
        for box in sorted(
            (box for box in game_result.player_box_scores if getattr(player_lookup.get(box.player_id), "team_code", None) == game_result.home_team_code),
            key=lambda box: box.minutes,
            reverse=True,
        )[:5]
    )
    away_ids = tuple(
        box.player_id
        for box in sorted(
            (box for box in game_result.player_box_scores if getattr(player_lookup.get(box.player_id), "team_code", None) == game_result.away_team_code),
            key=lambda box: box.minutes,
            reverse=True,
        )[:5]
    )
    if not home_ids or not away_ids:
        return []

    home_layout = [
        (0.0, 45.0, "top", False),
        (-12.0, 50.0, "left_wing", False),
        (12.0, 50.0, "right_wing", False),
        (-18.0, 41.0, "left_wing", False),
        (18.0, 41.0, "right_wing", False),
    ]
    away_layout = [
        (0.0, 49.0, "top", True),
        (-12.0, 44.0, "left_wing", False),
        (12.0, 44.0, "right_wing", False),
        (-18.0, 53.0, "left_wing", False),
        (18.0, 53.0, "right_wing", False),
    ]

    if game_result.possessions and game_result.possessions[0].offense_team_code == game_result.home_team_code:
        home_layout[0] = (0.0, 45.0, "top", True)
        away_layout[0] = (0.0, 49.0, "top", False)

    positions: list[dict[str, Any]] = []
    for player_id, (x, y, zone, has_ball) in zip(home_ids, home_layout, strict=False):
        positions.append(
            {
                "player_id": player_id,
                "team_code": game_result.home_team_code,
                "side": "offense",
                "x": x,
                "y": y,
                "zone": zone,
                "has_ball": has_ball,
            }
        )
    for player_id, (x, y, zone, has_ball) in zip(away_ids, away_layout, strict=False):
        positions.append(
            {
                "player_id": player_id,
                "team_code": game_result.away_team_code,
                "side": "defense",
                "x": x,
                "y": y,
                "zone": zone,
                "has_ball": has_ball,
            }
        )
    return positions


def _describe_event(event: Any, player_lookup: dict[str, Any]) -> tuple[str, str]:
    actor = _player_name(player_lookup, event.actor_id)
    receiver = _player_name(player_lookup, event.receiver_id)
    if event.event_type == EventType.ADVANCE:
        return "Push", f"{actor or 'Ball handler'} brings it up"
    if event.event_type == EventType.SCREEN:
        return "Screen", f"{actor or 'Big'} gets into the action for {receiver or 'the handler'}"
    if event.event_type == EventType.PASS:
        return "Pass", f"{actor or 'Ball handler'} moves it to {receiver or 'the next side'}"
    if event.event_type == EventType.DRIVE:
        return "Drive", f"{actor or 'Creator'} attacks the gap"
    if event.event_type == EventType.CUT:
        return "Cut", f"{actor or 'Off-ball cutter'} dives behind the defense"
    if event.event_type == EventType.SHOT:
        shot_name = (event.shot_type.value.replace('_', ' ') if event.shot_type else 'shot').title()
        if event.realized_success:
            return "Bucket", f"{actor or 'Shooter'} knocks down the {shot_name.lower()}"
        return "Shot", f"{actor or 'Shooter'} puts up a {shot_name.lower()}"
    if event.event_type == EventType.REBOUND:
        return "Rebound", f"{actor or 'Rebounder'} secures it"
    if event.event_type == EventType.FOUL:
        return "Foul", f"{actor or 'Attacker'} draws contact"
    if event.event_type == EventType.TURNOVER:
        return "Turnover", f"{actor or 'Ball handler'} gives it away"
    return event.event_type.value.replace("_", " ").title(), event.notes or "Action unfolds"


def _event_elapsed_seconds(event: Any) -> float:
    mapping = {
        EventType.ADVANCE: 1.6,
        EventType.SCREEN: 1.0,
        EventType.PASS: 0.8,
        EventType.DRIVE: 1.3,
        EventType.CUT: 1.0,
        EventType.SHOT: 0.7,
        EventType.REBOUND: 0.6,
        EventType.FOUL: 0.4,
        EventType.TURNOVER: 0.4,
    }
    return mapping.get(event.event_type, 0.8)


def _event_duration_ms(event: Any) -> int:
    mapping = {
        EventType.ADVANCE: 950,
        EventType.SCREEN: 900,
        EventType.PASS: 700,
        EventType.DRIVE: 1000,
        EventType.CUT: 850,
        EventType.SHOT: 950,
        EventType.REBOUND: 850,
        EventType.FOUL: 700,
        EventType.TURNOVER: 700,
    }
    return mapping.get(event.event_type, 800)


def _format_clock(seconds_remaining: float) -> str:
    whole = max(0, int(round(seconds_remaining)))
    minutes = whole // 60
    seconds = whole % 60
    return f"{minutes}:{seconds:02d}"
