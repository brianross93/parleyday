from __future__ import annotations

import random
from collections import defaultdict

from basketball_possession_engine import simulate_possession
from basketball_rotation_engine import select_lineup, update_stint_minutes
from basketball_sim_schema import (
    BasketballSide,
    CourtPoint,
    CourtZone,
    DefensiveAssignment,
    EntrySource,
    EntryType,
    FloorPlayerState,
    GameClockState,
    GameSimulationInput,
    GameSimulationResult,
    LineupUnit,
    PlayerBoxScoreProjection,
    PossessionContext,
    PossessionPhase,
    ScoreState,
    SimulatedPossession,
    TeamBoxScoreProjection,
)


def simulate_game(sim_input: GameSimulationInput, rng_seed: int | None = None) -> GameSimulationResult:
    rng = random.Random(rng_seed if rng_seed is not None else hash(sim_input.game_id) & 0xFFFFFFFF)
    player_lookup = {player.player_id: player for player in sim_input.players}
    home_lineup_ids = sim_input.home_rotation.starters
    away_lineup_ids = sim_input.away_rotation.starters

    home_score = 0
    away_score = 0
    event_log = []
    player_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"minutes": 0.0, "points": 0.0, "rebounds": 0.0, "assists": 0.0, "turnovers": 0.0, "fouls": 0.0, "steals": 0.0, "blocks": 0.0})
    team_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"points": 0.0, "fga": 0.0, "threes": 0.0, "fta": 0.0, "turnovers": 0.0, "oreb": 0.0, "dreb": 0.0, "assists": 0.0})
    home_stint_minutes: dict[str, float] = defaultdict(float)
    away_stint_minutes: dict[str, float] = defaultdict(float)
    foul_counts: dict[str, int] = defaultdict(int)
    possession_count = 0
    possessions: list[SimulatedPossession] = []
    current_floor_states: tuple[FloorPlayerState, ...] | None = None

    offense_home = sim_input.opening_tip_winner == sim_input.home_team_code
    if sim_input.opening_tip_winner is None:
        offense_home = True

    game_seconds_remaining = 48.0 * 60.0
    carry_over_rebound = False
    next_entry_type = EntryType.NORMAL
    next_entry_source = EntrySource.DEAD_BALL

    while game_seconds_remaining > 0:
        possession_count += 1
        previous_home_lineup_ids = home_lineup_ids
        previous_away_lineup_ids = away_lineup_ids
        home_minutes_played = {player_id: player_totals[player_id]["minutes"] for player_id in sim_input.home_rotation.target_minutes.keys()}
        away_minutes_played = {player_id: player_totals[player_id]["minutes"] for player_id in sim_input.away_rotation.target_minutes.keys()}
        home_lineup_ids = select_lineup(
            sim_input.home_rotation,
            home_minutes_played,
            home_lineup_ids,
            home_stint_minutes,
            game_seconds_remaining,
            foul_counts=foul_counts,
        )
        away_lineup_ids = select_lineup(
            sim_input.away_rotation,
            away_minutes_played,
            away_lineup_ids,
            away_stint_minutes,
            game_seconds_remaining,
            foul_counts=foul_counts,
        )
        home_lineup = _lineup_from_ids(sim_input.home_team_code, home_lineup_ids, player_lookup)
        away_lineup = _lineup_from_ids(sim_input.away_team_code, away_lineup_ids, player_lookup)
        offense_team = sim_input.home_team_code if offense_home else sim_input.away_team_code
        defense_team = sim_input.away_team_code if offense_home else sim_input.home_team_code
        offense_ids = home_lineup_ids if offense_home else away_lineup_ids
        defense_ids = away_lineup_ids if offense_home else home_lineup_ids
        offense_lineup = home_lineup if offense_home else away_lineup
        defense_lineup = away_lineup if offense_home else home_lineup
        offensive_tactics = sim_input.home_tactics if offense_home else sim_input.away_tactics
        defensive_tactics = sim_input.away_tactics if offense_home else sim_input.home_tactics

        seconds_per_possession = min(
            game_seconds_remaining,
            _draw_possession_seconds(sim_input, rng, offensive_rebound=carry_over_rebound),
        )
        clock = _clock_state(game_seconds_remaining, possession_count, seconds_per_possession)
        score = ScoreState(offense_score=home_score if offense_home else away_score, defense_score=away_score if offense_home else home_score)
        start_offense_score = score.offense_score
        start_defense_score = score.defense_score
        context = PossessionContext(
            offense_team_code=offense_team,
            defense_team_code=defense_team,
            clock=clock,
            score=score,
            offense_lineup=offense_lineup,
            defense_lineup=defense_lineup,
            offensive_tactics=offensive_tactics,
            defensive_tactics=defensive_tactics,
            floor_players=_next_floor_states(offense_ids, defense_ids, current_floor_states, player_lookup),
            defensive_assignments=_assignments(offense_ids, defense_ids),
            player_pool=sim_input.players,
            current_phase=PossessionPhase.PRIMARY_ACTION,
            play_call=None,
            coverage=None,
            entry_type=next_entry_type,
            entry_source=next_entry_source,
        )
        outcome = simulate_possession(context, rng)
        event_log.extend(outcome.events)
        if offense_home:
            home_score += outcome.points_scored
        else:
            away_score += outcome.points_scored
        end_offense_score = home_score if offense_home else away_score
        end_defense_score = away_score if offense_home else home_score
        possessions.append(
            SimulatedPossession(
                possession_number=possession_count,
                offense_team_code=offense_team,
                defense_team_code=defense_team,
                period=clock.period,
                start_clock=clock.seconds_remaining_in_period,
                end_clock=max(0.0, clock.seconds_remaining_in_period - seconds_per_possession),
                start_shot_clock=clock.shot_clock,
                end_shot_clock=max(0.0, clock.shot_clock - seconds_per_possession),
                entry_type=context.entry_type,
                entry_source=context.entry_source,
                start_offense_score=start_offense_score,
                start_defense_score=start_defense_score,
                end_offense_score=end_offense_score,
                end_defense_score=end_defense_score,
                points_scored=outcome.points_scored,
                events=outcome.events,
            )
        )
        _apply_outcome_to_boxscore(outcome, offense_team, defense_team, player_totals, team_totals, foul_counts)
        added_minutes = seconds_per_possession / 60.0
        home_stint_minutes = update_stint_minutes(previous_home_lineup_ids, home_lineup_ids, home_stint_minutes, added_minutes)
        away_stint_minutes = update_stint_minutes(previous_away_lineup_ids, away_lineup_ids, away_stint_minutes, added_minutes)
        for player_id in offense_ids + defense_ids:
            player_totals[player_id]["minutes"] += seconds_per_possession / 60.0
        carry_over_rebound = outcome.offensive_rebound
        if outcome.offensive_rebound:
            next_entry_type = EntryType.OREB
            next_entry_source = EntrySource.OREB_GATHER
            next_offense_ids = offense_ids
            next_defense_ids = defense_ids
        else:
            offense_home = not offense_home
            next_offense_tactics = sim_input.home_tactics if offense_home else sim_input.away_tactics
            next_entry_type, next_entry_source = _next_entry_state(outcome, next_offense_tactics, rng)
            next_offense_ids = home_lineup_ids if offense_home else away_lineup_ids
            next_defense_ids = away_lineup_ids if offense_home else home_lineup_ids
        current_floor_states = _carry_forward_floor_states(
            context.floor_players,
            outcome.events,
            next_offense_ids,
            next_defense_ids,
            player_lookup,
        )
        game_seconds_remaining = max(0.0, game_seconds_remaining - seconds_per_possession)

    player_box_scores = tuple(
        PlayerBoxScoreProjection(
            player_id=player.player_id,
            name=player.name,
            minutes=round(player_totals[player.player_id]["minutes"], 2),
            points=round(player_totals[player.player_id]["points"], 2),
            rebounds=round(player_totals[player.player_id]["rebounds"], 2),
            assists=round(player_totals[player.player_id]["assists"], 2),
            steals=round(player_totals[player.player_id]["steals"], 2),
            blocks=round(player_totals[player.player_id]["blocks"], 2),
            turnovers=round(player_totals[player.player_id]["turnovers"], 2),
            fouls=round(player_totals[player.player_id]["fouls"], 2),
        )
        for player in sim_input.players
    )
    team_box_scores = tuple(
        TeamBoxScoreProjection(
            team_code=team_code,
            points=round(values["points"], 2),
            field_goal_attempts=round(values["fga"], 2),
            threes_attempted=round(values["threes"], 2),
            free_throws_attempted=round(values["fta"], 2),
            turnovers=round(values["turnovers"], 2),
            offensive_rebounds=round(values["oreb"], 2),
            defensive_rebounds=round(values["dreb"], 2),
            assists=round(values["assists"], 2),
        )
        for team_code, values in sorted(team_totals.items())
    )
    return GameSimulationResult(
        game_id=sim_input.game_id,
        home_team_code=sim_input.home_team_code,
        away_team_code=sim_input.away_team_code,
        home_score=home_score,
        away_score=away_score,
        possession_count=possession_count,
        possessions=tuple(possessions),
        event_log=tuple(event_log),
        player_box_scores=player_box_scores,
        team_box_scores=team_box_scores,
    )


def simulate_games(sim_inputs: list[GameSimulationInput], rng_seed: int | None = None) -> list[GameSimulationResult]:
    results = []
    base_seed = rng_seed if rng_seed is not None else 17
    for offset, sim_input in enumerate(sim_inputs):
        results.append(simulate_game(sim_input, rng_seed=base_seed + offset))
    return results


def _estimated_total_possessions(sim_input: GameSimulationInput) -> int:
    avg_pace = (sim_input.home_tactics.pace_target + sim_input.away_tactics.pace_target) / 2.0
    return max(150, min(220, round(avg_pace * 2.0)))


def _draw_possession_seconds(sim_input: GameSimulationInput, rng: random.Random, offensive_rebound: bool) -> float:
    avg_pace = (sim_input.home_tactics.pace_target + sim_input.away_tactics.pace_target) / 2.0
    mean_seconds = (48.0 * 60.0) / max(avg_pace * 2.0, 1.0)
    if offensive_rebound:
        mean_seconds *= 0.55
    return max(3.5, min(24.0, rng.gauss(mean_seconds, 4.0)))


def _next_entry_state(outcome, offensive_tactics, rng: random.Random) -> tuple[EntryType, EntrySource]:
    if outcome.foul_committed:
        return EntryType.NORMAL, EntrySource.DEAD_BALL
    if outcome.turnover:
        transition_prob = min(0.82, 0.28 + (offensive_tactics.transition_frequency * 1.15) + (offensive_tactics.early_offense_rate * 0.45))
        if rng.random() < transition_prob:
            return EntryType.TRANSITION, EntrySource.LIVE_TURNOVER_BREAK
        return EntryType.NORMAL, EntrySource.DEAD_BALL
    if outcome.made_shot:
        early_prob = min(0.16, offensive_tactics.early_offense_rate * 0.22)
        if rng.random() < early_prob:
            return EntryType.TRANSITION, EntrySource.MADE_BASKET_FLOW
        return EntryType.NORMAL, EntrySource.DEAD_BALL
    if outcome.rebounder_id:
        transition_prob = min(0.58, 0.12 + (offensive_tactics.transition_frequency * 0.9) + (offensive_tactics.early_offense_rate * 0.35))
        if rng.random() < transition_prob:
            return EntryType.TRANSITION, EntrySource.DEFENSIVE_REBOUND_PUSH
        return EntryType.NORMAL, EntrySource.DEAD_BALL
    return EntryType.NORMAL, EntrySource.DEAD_BALL


def _lineup_from_ids(team_code: str, lineup_ids: tuple[str, ...], player_lookup: dict[str, object]) -> LineupUnit:
    players = [player_lookup[player_id] for player_id in lineup_ids if player_id in player_lookup]
    return LineupUnit(
        team_code=team_code,
        player_ids=lineup_ids,
        spacing_score=_avg(player.traits.catch_shoot for player in players),
        creation_score=_avg(max(player.traits.offensive_load, player.traits.pass_vision) for player in players),
        rim_pressure_score=_avg((player.traits.separation + player.traits.burst + player.traits.foul_drawing) / 3.0 for player in players),
        rebounding_score=_avg((player.traits.oreb + player.traits.dreb) / 2.0 for player in players),
        switchability_score=_avg((player.traits.containment + player.traits.closeout + player.traits.screen_nav) / 3.0 for player in players),
        rim_protection_score=_avg(player.traits.rim_protect for player in players),
    )


def _clock_state(game_seconds_remaining: float, possession_number: int, seconds_per_possession: float) -> GameClockState:
    elapsed = (48.0 * 60.0) - game_seconds_remaining
    period = min(4, int(elapsed // (12.0 * 60.0)) + 1)
    period_elapsed = elapsed % (12.0 * 60.0)
    seconds_remaining_in_period = max(0.0, (12.0 * 60.0) - period_elapsed)
    shot_clock = min(24.0, seconds_per_possession)
    return GameClockState(
        period=period,
        seconds_remaining_in_period=seconds_remaining_in_period,
        shot_clock=shot_clock,
        possession_number=possession_number,
    )


def _floor_states(offense_ids: tuple[str, ...], defense_ids: tuple[str, ...]) -> tuple[FloorPlayerState, ...]:
    offense_layout = [
        CourtPoint(0.0, 22.0, CourtZone.TOP),
        CourtPoint(-18.0, 21.0, CourtZone.LEFT_WING),
        CourtPoint(18.0, 21.0, CourtZone.RIGHT_WING),
        CourtPoint(-22.0, 3.0, CourtZone.LEFT_CORNER),
        CourtPoint(3.0, 20.0, CourtZone.TOP),
    ]
    defense_layout = [
        CourtPoint(0.0, 21.0, CourtZone.TOP),
        CourtPoint(-17.0, 20.0, CourtZone.LEFT_WING),
        CourtPoint(17.0, 20.0, CourtZone.RIGHT_WING),
        CourtPoint(-21.0, 4.0, CourtZone.LEFT_CORNER),
        CourtPoint(2.0, 18.0, CourtZone.PAINT),
    ]
    states = []
    for idx, player_id in enumerate(offense_ids[:5]):
        states.append(FloorPlayerState(player_id=player_id, side=BasketballSide.OFFENSE, location=offense_layout[idx], has_ball=(idx == 0)))
    for idx, player_id in enumerate(defense_ids[:5]):
        states.append(FloorPlayerState(player_id=player_id, side=BasketballSide.DEFENSE, location=defense_layout[idx], has_ball=False))
    return tuple(states)


def _next_floor_states(
    offense_ids: tuple[str, ...],
    defense_ids: tuple[str, ...],
    previous_states: tuple[FloorPlayerState, ...] | None,
    player_lookup: dict[str, object],
) -> tuple[FloorPlayerState, ...]:
    fallback_states = _role_anchored_floor_states(offense_ids, defense_ids, player_lookup)
    if previous_states is None:
        return fallback_states
    previous_by_id = {state.player_id: state for state in previous_states}
    carried: list[FloorPlayerState] = []
    for fallback in fallback_states:
        previous = previous_by_id.get(fallback.player_id)
        location = _blend_locations(previous.location, fallback.location, keep_ratio=0.58) if previous is not None else fallback.location
        carried.append(
            FloorPlayerState(
                player_id=fallback.player_id,
                side=fallback.side,
                location=location,
                has_ball=fallback.has_ball,
            )
        )
    return tuple(carried)


def _carry_forward_floor_states(
    floor_states: tuple[FloorPlayerState, ...],
    events: tuple,
    next_offense_ids: tuple[str, ...],
    next_defense_ids: tuple[str, ...],
    player_lookup: dict[str, object],
) -> tuple[FloorPlayerState, ...]:
    anchored_states = _role_anchored_floor_states(next_offense_ids, next_defense_ids, player_lookup)
    anchored_by_id = {state.player_id: state for state in anchored_states}
    positions = {state.player_id: state.location for state in floor_states}
    has_ball: dict[str, bool] = {state.player_id: state.has_ball for state in floor_states}
    for player_id in has_ball:
        has_ball[player_id] = False
    last_ball_player: str | None = None
    for event in events:
        if event.location and event.actor_id in positions:
            positions[event.actor_id] = event.location
        if event.location and event.receiver_id in positions:
            positions[event.receiver_id] = event.location
        if event.receiver_id:
            last_ball_player = event.receiver_id
        elif event.actor_id and event.event_type.value in {"advance", "drive", "shot", "handoff", "post_entry"}:
            last_ball_player = event.actor_id
    if last_ball_player in has_ball:
        has_ball[last_ball_player] = True
    next_states: list[FloorPlayerState] = []
    for player_id in next_offense_ids:
        anchor = anchored_by_id.get(player_id)
        location = positions.get(player_id, anchor.location if anchor else CourtPoint(0.0, 22.0, CourtZone.TOP))
        if anchor is not None:
            location = _blend_locations(location, anchor.location, keep_ratio=0.72)
        next_states.append(FloorPlayerState(player_id=player_id, side=BasketballSide.OFFENSE, location=location, has_ball=has_ball.get(player_id, False)))
    for player_id in next_defense_ids:
        anchor = anchored_by_id.get(player_id)
        location = positions.get(player_id, anchor.location if anchor else CourtPoint(0.0, 21.0, CourtZone.TOP))
        if anchor is not None:
            location = _blend_locations(location, anchor.location, keep_ratio=0.72)
        next_states.append(FloorPlayerState(player_id=player_id, side=BasketballSide.DEFENSE, location=location, has_ball=has_ball.get(player_id, False)))
    return tuple(next_states)


def _role_anchored_floor_states(
    offense_ids: tuple[str, ...],
    defense_ids: tuple[str, ...],
    player_lookup: dict[str, object],
) -> tuple[FloorPlayerState, ...]:
    states: list[FloorPlayerState] = []
    for index, player_id in enumerate(offense_ids[:5]):
        player = player_lookup.get(player_id)
        point = _offense_anchor(player, index)
        states.append(FloorPlayerState(player_id=player_id, side=BasketballSide.OFFENSE, location=point, has_ball=(index == 0)))
    for index, player_id in enumerate(defense_ids[:5]):
        player = player_lookup.get(player_id)
        point = _defense_anchor(player, index)
        states.append(FloorPlayerState(player_id=player_id, side=BasketballSide.DEFENSE, location=point, has_ball=False))
    return tuple(states)


def _offense_anchor(player, index: int) -> CourtPoint:
    role = getattr(player, "offensive_role", None)
    positions = tuple(getattr(player, "positions", ()) or ())
    if role and role.value == "primary_creator":
        return CourtPoint(0.0, 22.0, CourtZone.TOP)
    if role and role.value == "secondary_creator":
        return CourtPoint(-17.0, 21.0, CourtZone.LEFT_WING)
    if role and role.value in {"movement_shooter", "spacer"}:
        return CourtPoint(22.0 if index % 2 == 0 else -22.0, 3.0, CourtZone.RIGHT_CORNER if index % 2 == 0 else CourtZone.LEFT_CORNER)
    if role and role.value == "slasher":
        return CourtPoint(18.0, 21.0, CourtZone.RIGHT_WING)
    if role and role.value in {"roll_big", "post_hub"}:
        return CourtPoint(3.0 if index % 2 == 0 else -3.0, 9.0, CourtZone.RIGHT_DUNKER if index % 2 == 0 else CourtZone.LEFT_DUNKER)
    if "PG" in positions or "SG" in positions or "G" in positions:
        return CourtPoint(-18.0 if index % 2 else 18.0, 21.0, CourtZone.LEFT_WING if index % 2 else CourtZone.RIGHT_WING)
    if "C" in positions:
        return CourtPoint(0.0, 10.0, CourtZone.PAINT)
    if "PF" in positions or "F" in positions:
        return CourtPoint(-20.0 if index % 2 else 20.0, 5.0, CourtZone.LEFT_CORNER if index % 2 else CourtZone.RIGHT_CORNER)
    fallback_layout = [
        CourtPoint(0.0, 22.0, CourtZone.TOP),
        CourtPoint(-18.0, 21.0, CourtZone.LEFT_WING),
        CourtPoint(18.0, 21.0, CourtZone.RIGHT_WING),
        CourtPoint(-22.0, 3.0, CourtZone.LEFT_CORNER),
        CourtPoint(3.0, 20.0, CourtZone.TOP),
    ]
    return fallback_layout[min(index, len(fallback_layout) - 1)]


def _defense_anchor(player, index: int) -> CourtPoint:
    role = getattr(player, "defensive_role", None)
    positions = tuple(getattr(player, "positions", ()) or ())
    if role and role.value == "point_of_attack":
        return CourtPoint(0.0, 21.0, CourtZone.TOP)
    if role and role.value == "rim_protector":
        return CourtPoint(0.0, 12.0, CourtZone.PAINT)
    if role and role.value in {"wing_stopper", "helper"}:
        return CourtPoint(-17.0 if index % 2 else 17.0, 20.0, CourtZone.LEFT_WING if index % 2 else CourtZone.RIGHT_WING)
    if "C" in positions:
        return CourtPoint(0.0, 12.0, CourtZone.PAINT)
    if "G" in positions or "PG" in positions or "SG" in positions:
        return CourtPoint(-16.0 if index % 2 else 16.0, 20.0, CourtZone.LEFT_WING if index % 2 else CourtZone.RIGHT_WING)
    return CourtPoint(-19.0 if index % 2 else 19.0, 6.0, CourtZone.LEFT_CORNER if index % 2 else CourtZone.RIGHT_CORNER)


def _blend_locations(previous: CourtPoint, anchor: CourtPoint, keep_ratio: float) -> CourtPoint:
    keep_ratio = max(0.0, min(1.0, keep_ratio))
    anchor_ratio = 1.0 - keep_ratio
    return CourtPoint(
        x=(previous.x * keep_ratio) + (anchor.x * anchor_ratio),
        y=(previous.y * keep_ratio) + (anchor.y * anchor_ratio),
        zone=anchor.zone if anchor_ratio >= 0.25 else previous.zone,
    )


def _assignments(offense_ids: tuple[str, ...], defense_ids: tuple[str, ...]) -> tuple[DefensiveAssignment, ...]:
    pairs = zip(offense_ids[:5], defense_ids[:5], strict=False)
    return tuple(
        DefensiveAssignment(
            defender_id=defender_id,
            offensive_player_id=offense_id,
            matchup_strength=0.55,
            on_ball=(idx == 0),
            help_priority=0.6 if idx == 4 else 0.3,
        )
        for idx, (offense_id, defender_id) in enumerate(pairs)
    )


def _apply_outcome_to_boxscore(
    outcome,
    offense_team: str,
    defense_team: str,
    player_totals: dict[str, dict[str, float]],
    team_totals: dict[str, dict[str, float]],
    foul_counts: dict[str, int],
) -> None:
    if outcome.turnover and outcome.turnover_player_id:
        player_totals[outcome.turnover_player_id]["turnovers"] += 1.0
        team_totals[offense_team]["turnovers"] += 1.0
        if outcome.steal_player_id:
            player_totals[outcome.steal_player_id]["steals"] += 1.0
    if outcome.foul_committed and outcome.shooting_player_id:
        team_totals[offense_team]["fta"] += float(outcome.free_throws_attempted)
        foul_event = next((event for event in outcome.events if event.event_type.name == "FOUL"), None)
        if foul_event and foul_event.defender_id:
            player_totals[foul_event.defender_id]["fouls"] += 1.0
            foul_counts[foul_event.defender_id] += 1
    shot_events = [event for event in outcome.events if event.event_type.name == "SHOT"]
    for event in shot_events:
        team_totals[offense_team]["fga"] += 1.0
        if event.shot_type and event.shot_type.name in {"ABOVE_BREAK_THREE", "CORNER_THREE"}:
            team_totals[offense_team]["threes"] += 1.0
    if outcome.block_player_id:
        player_totals[outcome.block_player_id]["blocks"] += 1.0
    if outcome.points_scored and outcome.shooting_player_id:
        player_totals[outcome.shooting_player_id]["points"] += outcome.points_scored
        team_totals[offense_team]["points"] += outcome.points_scored
    if outcome.made_shot and outcome.shooting_player_id:
        if outcome.assisting_player_id:
            player_totals[outcome.assisting_player_id]["assists"] += 1.0
            team_totals[offense_team]["assists"] += 1.0
    if outcome.rebounder_id:
        player_totals[outcome.rebounder_id]["rebounds"] += 1.0
        if outcome.offensive_rebound:
            team_totals[offense_team]["oreb"] += 1.0
        else:
            team_totals[defense_team]["dreb"] += 1.0


def _avg(values) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / max(len(values), 1)
