from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Iterable

from basketball_sim_schema import (
    AdvantageState,
    CourtZone,
    DefensiveAssignment,
    DefensiveCoverage,
    EntrySource,
    EntryType,
    EventContext,
    EventType,
    FoulOutcomeType,
    OffensiveRole,
    PlayCall,
    PlayFamily,
    PlayerSimProfile,
    PossessionContext,
    PossessionOutcome,
    PossessionPhase,
    ProgressionState,
    ShotType,
    TurnoverType,
)


SUPPORTED_PLAY_FAMILIES = {PlayFamily.HIGH_PICK_AND_ROLL, PlayFamily.DOUBLE_DRAG, PlayFamily.ISO, PlayFamily.HANDOFF}
SUPPORTED_COVERAGES = {DefensiveCoverage.DROP, DefensiveCoverage.SWITCH, DefensiveCoverage.ICE, DefensiveCoverage.HEDGE}
WEAK_PROXY_WEIGHT_CAP = 0.20


def simulate_possession(context: PossessionContext, rng: random.Random | None = None) -> PossessionOutcome:
    rng = rng or random.Random()
    prepared_context, lead_events, direct_outcome = _prepare_entry_context(context, rng)
    if direct_outcome is not None:
        return direct_outcome
    play_call = prepared_context.play_call or _select_play_call(prepared_context, rng)
    if play_call.family not in SUPPORTED_PLAY_FAMILIES:
        raise ValueError(f"Unsupported play family for first engine slice: {play_call.family}")
    coverage = prepared_context.coverage or _select_coverage(prepared_context, rng)
    if coverage not in SUPPORTED_COVERAGES:
        raise ValueError(f"Unsupported coverage for first engine slice: {coverage}")
    if play_call.family == PlayFamily.HIGH_PICK_AND_ROLL:
        return _prepend_events(resolve_pnr(prepared_context, play_call, coverage, rng), lead_events)
    if play_call.family == PlayFamily.DOUBLE_DRAG:
        return _prepend_events(resolve_double_drag(prepared_context, play_call, coverage, rng), lead_events)
    if play_call.family == PlayFamily.ISO:
        return _prepend_events(resolve_iso(prepared_context, play_call, coverage, rng), lead_events)
    if play_call.family == PlayFamily.HANDOFF:
        return _prepend_events(resolve_handoff(prepared_context, play_call, coverage, rng), lead_events)
    raise ValueError(f"Unsupported play family: {play_call.family}")


def _prepare_entry_context(
    context: PossessionContext,
    rng: random.Random,
) -> tuple[PossessionContext, tuple[EventContext, ...], PossessionOutcome | None]:
    if context.entry_type == EntryType.OREB:
        return _prepare_oreb_entry(context, rng)
    if context.entry_type == EntryType.TRANSITION:
        return _prepare_transition_entry(context, rng)
    return context, (), None


def _prepare_oreb_entry(
    context: PossessionContext,
    rng: random.Random,
) -> tuple[PossessionContext, tuple[EventContext, ...], PossessionOutcome | None]:
    rebounder = max(
        _all_players(context, offense=True),
        key=lambda player: player.traits.oreb + (0.22 * player.traits.finishing) + (0.12 * player.traits.size),
        default=_get_player(context, context.offense_lineup.player_ids[0]),
    )
    capped_context = _with_entry_clock(context, 14.0)
    entry_event = EventContext(
        event_type=EventType.ADVANCE,
        actor_id=rebounder.player_id if rebounder else None,
        location=_court_point("paint"),
        success_probability=1.0,
        realized_success=True,
        notes=f"oreb_reentry_14s source={context.entry_source.value}",
    )
    if rebounder is None:
        return capped_context, (entry_event,), None

    tip_in_probability = _clamp(
        0.12
        + (context.offensive_tactics.crash_glass_rate * 0.22)
        + ((rebounder.traits.oreb + rebounder.traits.finishing + rebounder.traits.reach - 34.0) / 50.0) * 0.18,
        0.08,
        0.34,
    )
    tip_out_probability = _clamp(
        0.24
        + (context.offensive_tactics.second_side_rate * 0.24)
        + (context.offense_lineup.spacing_score / 20.0) * 0.08
        + ((rebounder.traits.oreb + rebounder.traits.pass_accuracy - 20.0) / 40.0) * 0.08,
        0.20,
        0.52,
    )
    gather_reset_probability = _clamp(
        0.22
        + (context.offensive_tactics.second_side_rate * 0.12)
        + (0.08 if capped_context.clock.shot_clock >= 10.0 else 0.0)
        - (tip_in_probability * 0.15),
        0.16,
        0.44,
    )
    decision = _weighted_choice(
        rng,
        [
            ("tip_in", tip_in_probability),
            ("tip_out", tip_out_probability),
            ("gather_reset", gather_reset_probability),
        ],
        default="gather_reset",
    )
    if decision == "tip_in":
        defender = _pick_help_defender(capped_context, exclude_ids=set())
        shot_result = {
            "result_type": "shot",
            "events": (),
            "shooter": rebounder,
            "shot_type": ShotType.RIM,
            "shot_quality": _shot_quality_from_advantage(
                rebounder.traits.finishing + (0.14 * rebounder.traits.oreb) + (0.10 * rebounder.traits.reach),
                ((defender.traits.rim_protect if defender else 10.0) * 0.92) + (0.06 * (defender.traits.size if defender else 10.0)),
                0.64,
                base=0.61,
            ),
        }
        tip_event = EventContext(
            event_type=EventType.REBOUND,
            actor_id=rebounder.player_id,
            defender_id=defender.player_id if defender else None,
            location=_court_point("rim"),
            success_probability=tip_in_probability,
            realized_success=True,
            notes="oreb_tip_in",
        )
        outcome = _materialize_shot_result(
            context=capped_context,
            result=shot_result,
            lead_events=(entry_event, tip_event),
            assister_id=None,
            default_off_rebounder=rebounder,
            defender=defender,
            rng=rng,
        )
        return capped_context, (), outcome
    if decision == "tip_out":
        receiver = _best_shooter(capped_context, exclude_ids={rebounder.player_id}, rng=rng)
        defender = _pick_closeout_defender(capped_context, receiver.player_id, _pick_help_defender(capped_context, exclude_ids=set()))
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=rebounder.player_id,
            receiver_id=receiver.player_id,
            defender_id=defender.player_id if defender else None,
            location=_court_point("wing"),
            success_probability=0.9,
            realized_success=True,
            notes="oreb_tip_out",
        )
        action_result = _resolve_kickout_action(
            capped_context,
            receiver,
            defender,
            _pick_help_defender(capped_context, exclude_ids={defender.player_id if defender else ""}),
            0.58,
            rng,
            last_passer_id=rebounder.player_id,
            shot_clock_remaining=min(capped_context.clock.shot_clock, 14.0),
        )
        outcome = _materialize_action_result(
            context=capped_context,
            result=action_result,
            lead_events=(entry_event, pass_event),
            default_assister_id=rebounder.player_id,
            default_off_rebounder=rebounder,
            defender=defender,
            rng=rng,
        )
        return capped_context, (), outcome
    gather_event = EventContext(
        event_type=EventType.ADVANCE,
        actor_id=rebounder.player_id,
        defender_id=None,
        location=_court_point("paint"),
        success_probability=gather_reset_probability,
        realized_success=True,
        notes="oreb_gather_reset",
    )
    next_play = PlayCall(
        family=PlayFamily.HIGH_PICK_AND_ROLL if capped_context.offensive_tactics.pick_and_roll_rate >= 0.28 else PlayFamily.ISO,
        primary_actor_id=(rebounder.player_id if rebounder.traits.pass_accuracy >= 10.0 else _primary_creator_id(capped_context, rng)),
        screener_id=(rebounder.player_id if rebounder.traits.screen_setting >= 11.0 else None),
        target_zone=CourtZone.TOP,
    )
    return replace(capped_context, play_call=next_play, entry_type=EntryType.NORMAL, entry_source=EntrySource.OREB_GATHER), (entry_event, gather_event), None


def _prepare_transition_entry(
    context: PossessionContext,
    rng: random.Random,
) -> tuple[PossessionContext, tuple[EventContext, ...], PossessionOutcome | None]:
    capped_context = _with_entry_clock(context, 18.0)
    primary_actor_id = _primary_creator_id(capped_context, rng)
    ball_handler = _get_player(capped_context, primary_actor_id)
    primary_assignment = _find_assignment(capped_context.defensive_assignments, primary_actor_id)
    primary_defender = _get_player(capped_context, primary_assignment.defender_id if primary_assignment else None)
    help_defender = _pick_help_defender(capped_context, exclude_ids={primary_defender.player_id if primary_defender else ""})
    advance_event = EventContext(
        event_type=EventType.ADVANCE,
        actor_id=ball_handler.player_id if ball_handler else None,
        defender_id=primary_defender.player_id if primary_defender else None,
        location=_court_point("top"),
        success_probability=1.0,
        realized_success=True,
        notes=f"transition_entry source={context.entry_source.value}",
    )
    if ball_handler is None:
        return capped_context, (advance_event,), None

    if context.entry_source == EntrySource.LIVE_TURNOVER_BREAK:
        rim_push_probability = _clamp(
            0.28
            + (capped_context.offensive_tactics.transition_frequency * 0.38)
            + ((ball_handler.traits.burst + ball_handler.traits.separation - 24.0) / 40.0) * 0.15,
            0.20,
            0.56,
        )
        trail_three_probability = _clamp(0.10 + (capped_context.offensive_tactics.early_offense_rate * 0.18), 0.06, 0.22)
        early_drag_probability = _clamp(0.14 + (capped_context.offensive_tactics.pick_and_roll_rate * 0.28), 0.10, 0.30)
    elif context.entry_source == EntrySource.DEFENSIVE_REBOUND_PUSH:
        rim_push_probability = _clamp(
            0.15
            + (capped_context.offensive_tactics.transition_frequency * 0.26)
            + ((ball_handler.traits.burst + ball_handler.traits.separation - 24.0) / 40.0) * 0.10,
            0.10,
            0.34,
        )
        trail_three_probability = _clamp(
            0.18
            + (capped_context.offensive_tactics.early_offense_rate * 0.30)
            + (capped_context.offense_lineup.spacing_score / 20.0) * 0.10,
            0.12,
            0.38,
        )
        early_drag_probability = _clamp(0.24 + (capped_context.offensive_tactics.pick_and_roll_rate * 0.40), 0.18, 0.48)
    else:
        rim_push_probability = _clamp(
            0.16
            + (capped_context.offensive_tactics.transition_frequency * 0.34)
            + ((ball_handler.traits.burst + ball_handler.traits.separation - 24.0) / 40.0) * 0.12,
            0.12,
            0.42,
        )
        trail_three_probability = _clamp(
            0.14
            + (capped_context.offensive_tactics.early_offense_rate * 0.28)
            + (capped_context.offense_lineup.spacing_score / 20.0) * 0.10,
            0.10,
            0.34,
        )
        early_drag_probability = _clamp(
            0.18 + (capped_context.offensive_tactics.pick_and_roll_rate * 0.42),
            0.15,
            0.46,
        )
    decision = _weighted_choice(
        rng,
        [
            ("rim_push", rim_push_probability),
            ("trail_three", trail_three_probability),
            ("early_drag", early_drag_probability),
            ("flow", max(0.10, 1.0 - (rim_push_probability + trail_three_probability + early_drag_probability))),
        ],
        default="flow",
    )
    if decision == "rim_push":
        action_result = resolve_drive_attempt(capped_context, ball_handler, primary_defender, help_defender, rng)
        outcome = _materialize_action_result(
            context=capped_context,
            result=action_result,
            lead_events=(advance_event,),
            default_assister_id=None,
            default_off_rebounder=ball_handler,
            defender=primary_defender,
            rng=rng,
        )
        return capped_context, (), outcome
    if decision == "trail_three":
        shooter = _best_shooter(capped_context, exclude_ids={ball_handler.player_id}, rng=rng)
        shooter_defender = _pick_closeout_defender(capped_context, shooter.player_id, primary_defender)
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=ball_handler.player_id,
            receiver_id=shooter.player_id,
            defender_id=primary_defender.player_id if primary_defender else None,
            location=_court_point("wing"),
            success_probability=0.9,
            realized_success=True,
            notes="transition_pitch_ahead",
        )
        shot_result = resolve_catch_and_shoot(capped_context, shooter, shooter_defender, 0.68, rng)
        outcome = _materialize_shot_result(
            context=capped_context,
            result=shot_result,
            lead_events=(advance_event, pass_event),
            assister_id=ball_handler.player_id,
            default_off_rebounder=shooter,
            defender=shooter_defender,
            rng=rng,
        )
        return capped_context, (), outcome
    if decision == "early_drag":
        screener = max(
            (player for player in _all_players(capped_context, offense=True) if player.player_id != ball_handler.player_id),
            key=lambda player: player.traits.screen_setting,
            default=None,
        )
        play_call = PlayCall(
            family=PlayFamily.HIGH_PICK_AND_ROLL,
            primary_actor_id=ball_handler.player_id,
            screener_id=screener.player_id if screener else None,
            target_zone=capped_context.play_call.target_zone if capped_context.play_call else None,
        )
        return replace(capped_context, play_call=play_call, entry_type=EntryType.NORMAL, entry_source=EntrySource.DEAD_BALL), (advance_event,), None
    return replace(capped_context, entry_type=EntryType.NORMAL, entry_source=EntrySource.DEAD_BALL), (advance_event,), None


def resolve_pnr(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> PossessionOutcome:
    progression_state = _resolve_pnr_creation_state(context, play_call, coverage, rng)
    return _resolve_progression_state(context, progression_state, rng)


def resolve_iso(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> PossessionOutcome:
    progression_state = _resolve_iso_creation_state(context, play_call, coverage, rng)
    return _resolve_progression_state(context, progression_state, rng)


def resolve_handoff(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> PossessionOutcome:
    progression_state = _resolve_handoff_creation_state(context, play_call, coverage, rng)
    return _resolve_progression_state(context, progression_state, rng)


def resolve_double_drag(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> PossessionOutcome:
    progression_state = _resolve_double_drag_creation_state(context, play_call, coverage, rng)
    return _resolve_progression_state(context, progression_state, rng)


def _resolve_pnr_creation_state(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> ProgressionState:
    handler = _get_player(context, play_call.primary_actor_id)
    screener = _get_player(context, play_call.screener_id)
    on_ball_assignment = _find_assignment(context.defensive_assignments, play_call.primary_actor_id)
    on_ball_defender = _get_player(context, on_ball_assignment.defender_id if on_ball_assignment else None)
    help_defender = _pick_help_defender(context, exclude_ids={on_ball_defender.player_id if on_ball_defender else ""})

    screen_advantage = (screener.traits.screen_setting if screener else 8.0) - (on_ball_defender.traits.screen_nav if on_ball_defender else 10.0)
    advantage = _resolve_pnr_advantage(context, handler, screener, on_ball_defender, help_defender, coverage)

    screen_event = EventContext(
        event_type=EventType.SCREEN,
        actor_id=screener.player_id if screener else None,
        receiver_id=handler.player_id,
        defender_id=on_ball_defender.player_id if on_ball_defender else None,
        location=_court_point("top"),
        success_probability=_clamp(0.5 + ((screen_advantage / 20.0) * 0.2), 0.25, 0.85),
        realized_success=True,
        notes=f"coverage={coverage.value};screen_advantage={screen_advantage:.2f}",
    )

    branch_rates = adjust_branches(
        base_rates={
            "pullup": 0.18 if coverage == DefensiveCoverage.DROP else 0.08 if coverage == DefensiveCoverage.SWITCH else 0.07 if coverage == DefensiveCoverage.HEDGE else 0.06,
            "drive": 0.27 if coverage == DefensiveCoverage.DROP else 0.34 if coverage == DefensiveCoverage.SWITCH else 0.16 if coverage == DefensiveCoverage.HEDGE else 0.20,
            "roller": 0.21 if coverage == DefensiveCoverage.DROP else 0.13 if coverage == DefensiveCoverage.SWITCH else 0.18 if coverage == DefensiveCoverage.HEDGE else 0.08,
            "kickout": 0.14 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0.22 if coverage == DefensiveCoverage.ICE else 0.20,
            "foul": 0.16 if coverage == DefensiveCoverage.DROP else 0.19 if coverage == DefensiveCoverage.SWITCH else 0.14,
            "turnover": 0.06 if coverage == DefensiveCoverage.DROP else 0.08 if coverage == DefensiveCoverage.SWITCH else 0.12 if coverage == DefensiveCoverage.HEDGE else 0.12,
            "reset": 0.02 if coverage == DefensiveCoverage.DROP else 0.05 if coverage == DefensiveCoverage.SWITCH else 0.13 if coverage == DefensiveCoverage.HEDGE else 0.16,
        },
        advantage=advantage,
        branch_effects={
            "pullup": 1 if coverage == DefensiveCoverage.DROP else -1 if coverage == DefensiveCoverage.ICE else 0,
            "drive": 1 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else -1,
            "roller": 1 if coverage in {DefensiveCoverage.DROP, DefensiveCoverage.HEDGE} else -1 if coverage == DefensiveCoverage.ICE else 0,
            "kickout": 1 if coverage in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0,
            "foul": 1 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0,
            "turnover": -1,
            "reset": -1 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 1,
        },
    )
    foul_scale = _foul_branch_scale(handler, on_ball_defender, advantage, switch_bonus=(0.12 if coverage == DefensiveCoverage.SWITCH else 0.0))
    branch_rates = _rescale_branch_probability(branch_rates, "foul", foul_scale)
    branch = _weighted_choice(rng, list(branch_rates.items()), default="pullup")

    if branch == "turnover":
        turnover_event = EventContext(
            event_type=EventType.TURNOVER,
            actor_id=handler.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("top"),
            turnover_type=TurnoverType.BAD_PASS if coverage == DefensiveCoverage.SWITCH else TurnoverType.STRIP,
            success_probability=branch_rates["turnover"],
            realized_success=True,
            notes="pnr turnover",
        )
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=screener.player_id if screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(screen_event,),
            terminal_result={
                "kind": "outcome",
                "outcome": _turnover_outcome(
                    handler.player_id,
                    (screen_event, turnover_event),
                    steal_player_id=on_ball_defender.player_id if on_ball_defender and turnover_event.turnover_type in {TurnoverType.STRIP, TurnoverType.BAD_PASS} else None,
                ),
            },
        )

    if branch == "foul":
        foul_event = EventContext(
            event_type=EventType.FOUL,
            actor_id=handler.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("paint"),
            foul_drawn=True,
            success_probability=branch_rates["foul"],
            realized_success=True,
            notes="pnr foul drawn",
        )
        foul_type = _choose_perimeter_foul_type(handler, rng)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=screener.player_id if screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(screen_event,),
            terminal_result={
                "kind": "outcome",
                "outcome": _foul_outcome(
                    handler.player_id,
                    handler.traits.ft_pct_raw,
                    (screen_event, foul_event),
                    rng,
                    assisting_player_id=screener.player_id if screener and foul_type == FoulOutcomeType.AND_ONE else None,
                    foul_type=foul_type,
                    shot_type=ShotType.ABOVE_BREAK_THREE if foul_type == FoulOutcomeType.THREE_SHOT else ShotType.RIM,
                ),
            },
        )

    if branch == "drive":
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.PAINT_TOUCH,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=screener.player_id if screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(screen_event,),
            paint_touched=True,
            help_committed=coverage == DefensiveCoverage.SWITCH,
        )

    if branch == "pullup":
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.PULL_UP_SPACE,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=screener.player_id if screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(screen_event,),
        )

    if branch == "roller" and screener is not None:
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=handler.player_id,
            receiver_id=screener.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("paint"),
            success_probability=_clamp(0.45 + ((handler.traits.pass_vision + handler.traits.pass_accuracy) / 40.0) * 0.35, 0.3, 0.9),
            realized_success=True,
            notes="pocket pass to roller",
        )
        roller_result = {
            "result_type": "shot",
            "shooter": screener,
            "shot_type": ShotType.RIM if screener.traits.finishing >= 12.0 else ShotType.PAINT,
            "shot_quality": _shot_quality_from_advantage(
                screener.traits.finishing + (0.12 * screener.traits.size) + (0.05 * screener.traits.reach),
                ((help_defender.traits.rim_protect if help_defender else 10.0) * 0.88) + (0.07 * (help_defender.traits.reach if help_defender else 10.0)),
                advantage,
                base=0.64,
            ),
        }
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=screener.player_id,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=screener.player_id,
            coverage=coverage,
            pass_chain=(handler.player_id,),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=(screen_event, pass_event),
            last_passer_id=handler.player_id,
            terminal_result={
                "kind": "shot",
                "result": roller_result,
                "assister_id": handler.player_id,
                "default_off_rebounder": screener,
                "defender": help_defender or on_ball_defender,
            },
        )

    if branch == "kickout":
        receiver = _best_shooter(context, exclude_ids={handler.player_id, screener.player_id if screener else ""}, rng=rng)
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=handler.player_id,
            receiver_id=receiver.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("wing"),
            success_probability=_clamp(0.5 + ((handler.traits.pass_vision + handler.traits.pass_accuracy) / 40.0) * 0.3, 0.35, 0.92),
            realized_success=True,
            notes="pnr kickout",
        )
        shot_defender = _pick_closeout_defender(context, receiver.player_id, on_ball_defender)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.FORCED_HELP,
            ball_handler_id=handler.player_id,
            current_receiver_id=receiver.player_id,
            primary_defender_id=shot_defender.player_id if shot_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=screener.player_id if screener else None,
            coverage=coverage,
            pass_chain=(handler.player_id,),
            off_ball_states=_initial_off_ball_states(context, {handler.player_id, receiver.player_id}),
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=(screen_event, pass_event),
            last_passer_id=handler.player_id,
            paint_touched=True,
            help_committed=True,
        )

    secondary = _secondary_creator(context, exclude_ids={handler.player_id, screener.player_id if screener else ""})
    reset_pass = EventContext(
        event_type=EventType.PASS,
        actor_id=handler.player_id,
        receiver_id=secondary.player_id if secondary else handler.player_id,
        defender_id=on_ball_defender.player_id if on_ball_defender else None,
        location=_court_point("top"),
        success_probability=0.88,
        realized_success=True,
        notes="pnr reset",
    )
    reset_actor = secondary or handler
    return ProgressionState(
        shot_clock_remaining=max(0.0, context.clock.shot_clock - 7.0),
        entry_type=EntryType.NORMAL,
        advantage_state=AdvantageState.NONE,
        ball_handler_id=handler.player_id,
        current_receiver_id=reset_actor.player_id,
        primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
        help_defender_id=help_defender.player_id if help_defender else None,
        screener_id=screener.player_id if screener else None,
        coverage=coverage,
        pass_chain=(handler.player_id,),
        off_ball_states={},
        clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 7.0)),
        possession_events=(screen_event, reset_pass),
        last_passer_id=handler.player_id,
    )


def _resolve_double_drag_creation_state(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> ProgressionState:
    handler = _get_player(context, play_call.primary_actor_id)
    first_screener = _get_player(context, play_call.screener_id)
    second_screener = _get_player(context, play_call.secondary_actor_id)
    on_ball_assignment = _find_assignment(context.defensive_assignments, play_call.primary_actor_id)
    on_ball_defender = _get_player(context, on_ball_assignment.defender_id if on_ball_assignment else None)
    help_defender = _pick_help_defender(context, exclude_ids={on_ball_defender.player_id if on_ball_defender else ""})
    first_advantage = (first_screener.traits.screen_setting if first_screener else 9.0) - (on_ball_defender.traits.screen_nav if on_ball_defender else 10.0)
    second_advantage = (second_screener.traits.screen_setting if second_screener else 9.0) - (help_defender.traits.screen_nav if help_defender else 10.0)
    advantage = _resolve_pnr_advantage(context, handler, second_screener or first_screener, on_ball_defender, help_defender, coverage)

    first_screen = EventContext(
        event_type=EventType.SCREEN,
        actor_id=first_screener.player_id if first_screener else None,
        receiver_id=handler.player_id if handler else None,
        defender_id=on_ball_defender.player_id if on_ball_defender else None,
        location=_court_point("top"),
        success_probability=_clamp(0.52 + ((first_advantage / 20.0) * 0.18), 0.26, 0.88),
        realized_success=True,
        notes=f"coverage={coverage.value};double_drag_1={first_advantage:.2f}",
    )
    second_screen = EventContext(
        event_type=EventType.SCREEN,
        actor_id=second_screener.player_id if second_screener else None,
        receiver_id=handler.player_id if handler else None,
        defender_id=help_defender.player_id if help_defender else None,
        location=_court_point("top"),
        success_probability=_clamp(0.54 + ((second_advantage / 20.0) * 0.18), 0.28, 0.90),
        realized_success=True,
        notes=f"coverage={coverage.value};double_drag_2={second_advantage:.2f}",
    )
    branch_rates = adjust_branches(
        base_rates={
            "turn_corner": 0.18 if coverage != DefensiveCoverage.HEDGE else 0.10,
            "middle_pullup": 0.14 if coverage != DefensiveCoverage.HEDGE else 0.08,
            "second_roller": 0.20 if second_screener else 0.0,
            "pop_big": 0.10 if first_screener else 0.0,
            "kickout": 0.18 if coverage != DefensiveCoverage.HEDGE else 0.24,
            "foul": 0.12,
            "turnover": 0.06 if coverage != DefensiveCoverage.HEDGE else 0.11,
            "reset": 0.02 if coverage != DefensiveCoverage.HEDGE else 0.12,
        },
        advantage=advantage,
        branch_effects={
            "turn_corner": 1 if coverage != DefensiveCoverage.HEDGE else -1,
            "middle_pullup": 0,
            "second_roller": 1,
            "pop_big": 0,
            "kickout": 1 if coverage == DefensiveCoverage.HEDGE else 0,
            "foul": 1,
            "turnover": -1,
            "reset": 1 if coverage == DefensiveCoverage.HEDGE else -1,
        },
    )
    branch = _weighted_choice(rng, list(branch_rates.items()), default="turn_corner")
    lead_events = (first_screen, second_screen)
    active_screener = second_screener or first_screener

    if branch == "turnover":
        turnover_event = EventContext(
            event_type=EventType.TURNOVER,
            actor_id=handler.player_id if handler else None,
            defender_id=help_defender.player_id if help_defender else on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("top"),
            turnover_type=TurnoverType.BAD_PASS if coverage == DefensiveCoverage.HEDGE else TurnoverType.STRIP,
            success_probability=branch_rates["turnover"],
            realized_success=True,
            notes="double_drag turnover",
        )
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=active_screener.player_id if active_screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=lead_events,
            terminal_result={"kind": "outcome", "outcome": _turnover_outcome(handler.player_id, lead_events + (turnover_event,))},
        )

    if branch == "foul":
        foul_event = EventContext(
            event_type=EventType.FOUL,
            actor_id=handler.player_id if handler else None,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("paint"),
            foul_drawn=True,
            success_probability=branch_rates["foul"],
            realized_success=True,
            notes="double_drag foul drawn",
        )
        foul_type = _choose_perimeter_foul_type(handler, rng)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=active_screener.player_id if active_screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=lead_events,
            terminal_result={
                "kind": "outcome",
                "outcome": _foul_outcome(
                    handler.player_id,
                    handler.traits.ft_pct_raw,
                    lead_events + (foul_event,),
                    rng,
                    foul_type=foul_type,
                    shot_type=ShotType.RIM if foul_type != FoulOutcomeType.THREE_SHOT else ShotType.ABOVE_BREAK_THREE,
                ),
            },
        )

    if branch == "turn_corner":
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.PAINT_TOUCH,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=active_screener.player_id if active_screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=lead_events,
            paint_touched=True,
            help_committed=coverage in {DefensiveCoverage.SWITCH, DefensiveCoverage.HEDGE},
        )

    if branch == "middle_pullup":
        pullup_event = EventContext(
            event_type=EventType.DRIVE,
            actor_id=handler.player_id if handler else None,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("top"),
            success_probability=_clamp(0.44 + ((advantage - 0.5) * 0.20), 0.22, 0.82),
            realized_success=True,
            notes="double_drag middle_pullup",
        )
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.5),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.PULL_UP_SPACE,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=active_screener.player_id if active_screener else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.5)),
            possession_events=lead_events + (pullup_event,),
        )

    if branch == "second_roller" and active_screener is not None:
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=handler.player_id,
            receiver_id=active_screener.player_id,
            defender_id=help_defender.player_id if help_defender else None,
            location=_court_point("paint"),
            success_probability=0.84,
            realized_success=True,
            notes="double_drag hit_second_roller",
        )
        roller_result = {
            "result_type": "shot",
            "shooter": active_screener,
            "shot_type": ShotType.RIM if active_screener.traits.finishing >= 12.0 else ShotType.PAINT,
            "shot_quality": _shot_quality_from_advantage(
                active_screener.traits.finishing + (0.10 * active_screener.traits.size),
                ((help_defender.traits.rim_protect if help_defender else 10.0) * 0.88),
                advantage,
                base=0.61,
            ),
        }
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 7.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=active_screener.player_id,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=active_screener.player_id,
            coverage=coverage,
            pass_chain=(handler.player_id,),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 7.0)),
            possession_events=lead_events + (pass_event,),
            last_passer_id=handler.player_id,
            terminal_result={
                "kind": "shot",
                "result": roller_result,
                "assister_id": handler.player_id,
                "default_off_rebounder": active_screener,
                "defender": help_defender or on_ball_defender,
            },
        )

    if branch == "pop_big" and first_screener is not None:
        receiver = _best_shooter(context, exclude_ids={handler.player_id, active_screener.player_id if active_screener else ""}, rng=rng)
        if receiver.player_id == first_screener.player_id:
            receiver = first_screener
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=handler.player_id,
            receiver_id=receiver.player_id,
            defender_id=help_defender.player_id if help_defender else None,
            location=_court_point("left_wing" if (play_call.target_zone or CourtZone.LEFT_WING) == CourtZone.LEFT_WING else "right_wing"),
            success_probability=0.86,
            realized_success=True,
            notes="double_drag pop",
        )
        shot_defender = _pick_closeout_defender(context, receiver.player_id, help_defender or on_ball_defender)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.5),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.FORCED_HELP,
            ball_handler_id=handler.player_id,
            current_receiver_id=receiver.player_id,
            primary_defender_id=shot_defender.player_id if shot_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=active_screener.player_id if active_screener else None,
            coverage=coverage,
            pass_chain=(handler.player_id,),
            off_ball_states=_initial_off_ball_states(context, {handler.player_id, receiver.player_id}),
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.5)),
            possession_events=lead_events + (pass_event,),
            last_passer_id=handler.player_id,
            help_committed=True,
        )

    if branch == "kickout":
        receiver = _best_shooter(context, exclude_ids={handler.player_id, active_screener.player_id if active_screener else "", first_screener.player_id if first_screener else ""}, rng=rng)
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=handler.player_id,
            receiver_id=receiver.player_id,
            defender_id=help_defender.player_id if help_defender else None,
            location=_court_point("wing"),
            success_probability=0.86,
            realized_success=True,
            notes="double_drag kickout",
        )
        shot_defender = _pick_closeout_defender(context, receiver.player_id, help_defender or on_ball_defender)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.5),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.FORCED_HELP,
            ball_handler_id=handler.player_id,
            current_receiver_id=receiver.player_id,
            primary_defender_id=shot_defender.player_id if shot_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=active_screener.player_id if active_screener else None,
            coverage=coverage,
            pass_chain=(handler.player_id,),
            off_ball_states=_initial_off_ball_states(context, {handler.player_id, receiver.player_id}),
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.5)),
            possession_events=lead_events + (pass_event,),
            last_passer_id=handler.player_id,
            paint_touched=coverage != DefensiveCoverage.HEDGE,
            help_committed=True,
        )

    reset_pass = EventContext(
        event_type=EventType.PASS,
        actor_id=handler.player_id,
        receiver_id=handler.player_id,
        defender_id=on_ball_defender.player_id if on_ball_defender else None,
        location=_court_point("top"),
        success_probability=0.90,
        realized_success=True,
        notes="double_drag reset",
    )
    return ProgressionState(
        shot_clock_remaining=max(0.0, context.clock.shot_clock - 7.0),
        entry_type=EntryType.NORMAL,
        advantage_state=AdvantageState.NONE,
        ball_handler_id=handler.player_id,
        current_receiver_id=handler.player_id,
        primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
        help_defender_id=help_defender.player_id if help_defender else None,
        screener_id=active_screener.player_id if active_screener else None,
        coverage=coverage,
        pass_chain=(handler.player_id,),
        off_ball_states={},
        clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 7.0)),
        possession_events=lead_events + (reset_pass,),
        last_passer_id=handler.player_id,
    )


def _resolve_iso_creation_state(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> ProgressionState:
    handler = _get_player(context, play_call.primary_actor_id)
    on_ball_assignment = _find_assignment(context.defensive_assignments, play_call.primary_actor_id)
    on_ball_defender = _get_player(context, on_ball_assignment.defender_id if on_ball_assignment else None)
    help_defender = _pick_help_defender(context, exclude_ids={on_ball_defender.player_id if on_ball_defender else ""})
    advantage = _resolve_iso_advantage(handler, on_ball_defender, help_defender)

    drive_event = EventContext(
        event_type=EventType.DRIVE,
        actor_id=handler.player_id,
        defender_id=on_ball_defender.player_id if on_ball_defender else None,
        location=_court_point("top"),
        success_probability=_clamp(0.45 + ((advantage - 0.5) * 0.4), 0.15, 0.9),
        realized_success=True,
        notes=f"coverage={coverage.value};iso_advantage={advantage:.3f}",
    )

    branch_rates = adjust_branches(
        base_rates={
            "drive": 0.40,
            "midrange": 0.05,
            "three": 0.09 if handler.traits.pullup_shooting >= 10.0 else 0.03,
            "foul": 0.22,
            "kickout": 0.12,
            "turnover": 0.10,
            "reset": 0.04,
        },
        advantage=advantage,
        branch_effects={
            "drive": 1,
            "midrange": 0,
            "three": 0,
            "foul": 1,
            "kickout": -1,
            "turnover": -1,
            "reset": -1,
        },
    )
    branch_rates = _rescale_branch_probability(branch_rates, "foul", _foul_branch_scale(handler, on_ball_defender, advantage))
    branch = _weighted_choice(rng, list(branch_rates.items()), default="drive")

    if branch == "turnover":
        turnover_event = EventContext(
            event_type=EventType.TURNOVER,
            actor_id=handler.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("top"),
            turnover_type=TurnoverType.STRIP,
            success_probability=branch_rates["turnover"],
            realized_success=True,
            notes="iso strip turnover",
        )
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(drive_event,),
            terminal_result={
                "kind": "outcome",
                "outcome": _turnover_outcome(
                    handler.player_id,
                    (drive_event, turnover_event),
                    steal_player_id=on_ball_defender.player_id if on_ball_defender and turnover_event.turnover_type == TurnoverType.STRIP else None,
                ),
            },
        )

    if branch == "foul":
        foul_event = EventContext(
            event_type=EventType.FOUL,
            actor_id=handler.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("paint"),
            foul_drawn=True,
            success_probability=branch_rates["foul"],
            realized_success=True,
            notes="iso foul drawn",
        )
        foul_type = _choose_perimeter_foul_type(handler, rng)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(drive_event,),
            terminal_result={
                "kind": "outcome",
                "outcome": _foul_outcome(
                    handler.player_id,
                    handler.traits.ft_pct_raw,
                    (drive_event, foul_event),
                    rng,
                    foul_type=foul_type,
                    shot_type=ShotType.ABOVE_BREAK_THREE if foul_type == FoulOutcomeType.THREE_SHOT else ShotType.RIM,
                ),
            },
        )

    if branch == "drive":
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.PAINT_TOUCH,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(drive_event,),
            paint_touched=True,
        )

    if branch in {"midrange", "three"}:
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.PULL_UP_SPACE,
            ball_handler_id=handler.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(drive_event,),
        )

    if branch == "kickout":
        receiver = _best_shooter(context, exclude_ids={handler.player_id}, rng=rng)
        shot_defender = _pick_closeout_defender(context, receiver.player_id, on_ball_defender)
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=handler.player_id,
            receiver_id=receiver.player_id,
            defender_id=help_defender.player_id if help_defender else None,
            location=_court_point("wing"),
            success_probability=0.87,
            realized_success=True,
            notes="iso kickout",
        )
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.FORCED_HELP,
            ball_handler_id=handler.player_id,
            current_receiver_id=receiver.player_id,
            primary_defender_id=shot_defender.player_id if shot_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=None,
            coverage=coverage,
            pass_chain=(handler.player_id,),
            off_ball_states=_initial_off_ball_states(context, {handler.player_id, receiver.player_id}),
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=(drive_event, pass_event),
            last_passer_id=handler.player_id,
            paint_touched=True,
            help_committed=True,
        )

    secondary = _secondary_creator(context, exclude_ids={handler.player_id})
    reset_pass = EventContext(
        event_type=EventType.PASS,
        actor_id=handler.player_id,
        receiver_id=secondary.player_id if secondary else handler.player_id,
        defender_id=on_ball_defender.player_id if on_ball_defender else None,
        location=_court_point("top"),
        success_probability=0.9,
        realized_success=True,
        notes="iso bailout reset",
    )
    bailout_actor = secondary or handler
    return ProgressionState(
        shot_clock_remaining=max(0.0, context.clock.shot_clock - 7.0),
        entry_type=EntryType.NORMAL,
        advantage_state=AdvantageState.NONE,
        ball_handler_id=handler.player_id,
        current_receiver_id=bailout_actor.player_id,
        primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
        help_defender_id=help_defender.player_id if help_defender else None,
        screener_id=None,
        coverage=coverage,
        pass_chain=(handler.player_id,),
        off_ball_states={},
        clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 7.0)),
        possession_events=(drive_event, reset_pass),
        last_passer_id=handler.player_id,
    )


def _resolve_handoff_creation_state(
    context: PossessionContext,
    play_call: PlayCall,
    coverage: DefensiveCoverage,
    rng: random.Random,
) -> ProgressionState:
    receiver = _get_player(context, play_call.primary_actor_id)
    handoff_big = _get_player(context, play_call.screener_id) or _pick_re_screen_big(context, exclude_ids={receiver.player_id if receiver else ""})
    if receiver is None:
        receiver = _get_player(context, context.offense_lineup.player_ids[0])
    on_ball_assignment = _find_assignment(context.defensive_assignments, receiver.player_id if receiver else "")
    on_ball_defender = _get_player(context, on_ball_assignment.defender_id if on_ball_assignment else None)
    help_defender = _pick_help_defender(context, exclude_ids={on_ball_defender.player_id if on_ball_defender else ""})
    handoff_event = EventContext(
        event_type=EventType.HANDOFF,
        actor_id=handoff_big.player_id if handoff_big else None,
        receiver_id=receiver.player_id if receiver else None,
        defender_id=on_ball_defender.player_id if on_ball_defender else None,
        location=_court_point("wing"),
        success_probability=_clamp(0.74 + (((handoff_big.traits.screen_setting if handoff_big else 10.0) - 10.0) / 40.0), 0.58, 0.90),
        realized_success=True,
        notes=f"coverage={coverage.value};handoff_flow",
    )
    advantage = _resolve_pnr_advantage(context, receiver, handoff_big, on_ball_defender, help_defender, coverage)
    branch_rates = adjust_branches(
        base_rates={
            "turn_corner": 0.28 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0.10 if coverage == DefensiveCoverage.ICE else 0.12,
            "snake_pullup": 0.18 if coverage == DefensiveCoverage.DROP else 0.08 if coverage != DefensiveCoverage.HEDGE else 0.10,
            "kickout": 0.16 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0.24 if coverage == DefensiveCoverage.ICE else 0.22,
            "flip_reset": 0.08 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0.24 if coverage == DefensiveCoverage.ICE else 0.18,
            "foul": 0.16,
            "turnover": 0.07 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0.12 if coverage == DefensiveCoverage.ICE else 0.10,
            "big_short_roll": 0.16 if handoff_big and coverage != DefensiveCoverage.ICE else 0.20 if handoff_big and coverage == DefensiveCoverage.HEDGE else 0.0,
        },
        advantage=advantage,
        branch_effects={
            "turn_corner": 1 if coverage not in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else -1,
            "snake_pullup": 0,
            "kickout": 1 if coverage in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else 0,
            "flip_reset": 1 if coverage in {DefensiveCoverage.ICE, DefensiveCoverage.HEDGE} else -1,
            "foul": 1,
            "turnover": -1,
            "big_short_roll": 1 if coverage == DefensiveCoverage.HEDGE else 0,
        },
    )
    branch = _weighted_choice(rng, list(branch_rates.items()), default="turn_corner")

    if branch == "turnover":
        turnover_event = EventContext(
            event_type=EventType.TURNOVER,
            actor_id=receiver.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("wing"),
            turnover_type=TurnoverType.BAD_PASS if coverage == DefensiveCoverage.ICE else TurnoverType.STRIP,
            success_probability=branch_rates["turnover"],
            realized_success=True,
            notes="handoff turnover",
        )
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=receiver.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=handoff_big.player_id if handoff_big else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(handoff_event,),
            terminal_result={"kind": "outcome", "outcome": _turnover_outcome(receiver.player_id, (handoff_event, turnover_event))},
        )

    if branch == "foul":
        foul_event = EventContext(
            event_type=EventType.FOUL,
            actor_id=receiver.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("paint"),
            foul_drawn=True,
            success_probability=branch_rates["foul"],
            realized_success=True,
            notes="handoff foul drawn",
        )
        foul_type = _choose_perimeter_foul_type(receiver, rng)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=receiver.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=handoff_big.player_id if handoff_big else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(handoff_event,),
            terminal_result={
                "kind": "outcome",
                "outcome": _foul_outcome(
                    receiver.player_id,
                    receiver.traits.ft_pct_raw,
                    (handoff_event, foul_event),
                    rng,
                    assisting_player_id=handoff_big.player_id if handoff_big and foul_type == FoulOutcomeType.AND_ONE else None,
                    foul_type=foul_type,
                    shot_type=ShotType.RIM if foul_type != FoulOutcomeType.THREE_SHOT else ShotType.ABOVE_BREAK_THREE,
                ),
            },
        )

    if branch == "snake_pullup":
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.PULL_UP_SPACE,
            ball_handler_id=receiver.player_id,
            current_receiver_id=None,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=handoff_big.player_id if handoff_big else None,
            coverage=coverage,
            pass_chain=(),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
            possession_events=(handoff_event,),
        )

    if branch == "big_short_roll" and handoff_big is not None:
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=receiver.player_id,
            receiver_id=handoff_big.player_id,
            defender_id=help_defender.player_id if help_defender else None,
            location=_court_point("paint"),
            success_probability=0.84,
            realized_success=True,
            notes="handoff short roll",
        )
        roller_result = {
            "result_type": "shot",
            "shooter": handoff_big,
            "shot_type": ShotType.PAINT if handoff_big.traits.finishing < 13.0 else ShotType.RIM,
            "shot_quality": _shot_quality_from_advantage(
                handoff_big.traits.finishing + (0.10 * handoff_big.traits.size),
                ((help_defender.traits.rim_protect if help_defender else 10.0) * 0.88),
                advantage,
                base=0.60,
            ),
        }
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.TERMINAL,
            ball_handler_id=receiver.player_id,
            current_receiver_id=handoff_big.player_id,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=handoff_big.player_id,
            coverage=coverage,
            pass_chain=(receiver.player_id,),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=(handoff_event, pass_event),
            last_passer_id=receiver.player_id,
            terminal_result={
                "kind": "shot",
                "result": roller_result,
                "assister_id": receiver.player_id,
                "default_off_rebounder": handoff_big,
                "defender": help_defender or on_ball_defender,
            },
        )

    if branch == "kickout":
        wing = _best_shooter(context, exclude_ids={receiver.player_id, handoff_big.player_id if handoff_big else ""}, rng=rng)
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=receiver.player_id,
            receiver_id=wing.player_id,
            defender_id=help_defender.player_id if help_defender else None,
            location=_court_point("wing"),
            success_probability=0.86,
            realized_success=True,
            notes="handoff kickout",
        )
        closeout_defender = _pick_closeout_defender(context, wing.player_id, on_ball_defender)
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.5),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.FORCED_HELP,
            ball_handler_id=receiver.player_id,
            current_receiver_id=wing.player_id,
            primary_defender_id=closeout_defender.player_id if closeout_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=handoff_big.player_id if handoff_big else None,
            coverage=coverage,
            pass_chain=(receiver.player_id,),
            off_ball_states=_initial_off_ball_states(context, {receiver.player_id, wing.player_id}),
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.5)),
            possession_events=(handoff_event, pass_event),
            last_passer_id=receiver.player_id,
        paint_touched=coverage != DefensiveCoverage.ICE,
        help_committed=True,
    )

    if branch == "flip_reset":
        secondary = _secondary_creator(context, exclude_ids={receiver.player_id}) or receiver
        reset_pass = EventContext(
            event_type=EventType.PASS,
            actor_id=receiver.player_id,
            receiver_id=secondary.player_id,
            defender_id=on_ball_defender.player_id if on_ball_defender else None,
            location=_court_point("top"),
            success_probability=0.9,
            realized_success=True,
            notes="handoff flip reset",
        )
        return ProgressionState(
            shot_clock_remaining=max(0.0, context.clock.shot_clock - 6.0),
            entry_type=EntryType.NORMAL,
            advantage_state=AdvantageState.NONE,
            ball_handler_id=receiver.player_id,
            current_receiver_id=secondary.player_id,
            primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
            help_defender_id=help_defender.player_id if help_defender else None,
            screener_id=handoff_big.player_id if handoff_big else None,
            coverage=coverage,
            pass_chain=(receiver.player_id,),
            off_ball_states={},
            clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 6.0)),
            possession_events=(handoff_event, reset_pass),
            last_passer_id=receiver.player_id,
        )

    return ProgressionState(
        shot_clock_remaining=max(0.0, context.clock.shot_clock - 5.0),
        entry_type=EntryType.NORMAL,
        advantage_state=AdvantageState.PAINT_TOUCH,
        ball_handler_id=receiver.player_id,
        current_receiver_id=None,
        primary_defender_id=on_ball_defender.player_id if on_ball_defender else None,
        help_defender_id=help_defender.player_id if help_defender else None,
        screener_id=handoff_big.player_id if handoff_big else None,
        coverage=coverage,
        pass_chain=(),
        off_ball_states={},
        clock_urgency=_clock_urgency(max(0.0, context.clock.shot_clock - 5.0)),
        possession_events=(handoff_event,),
        paint_touched=True,
        help_committed=coverage != DefensiveCoverage.ICE,
    )


def resolve_drive_attempt(
    context: PossessionContext,
    attacker: PlayerSimProfile,
    primary_defender: PlayerSimProfile | None,
    help_defender: PlayerSimProfile | None,
    rng: random.Random,
) -> dict[str, object]:
    off_score = weighted_sum(
        [
            (attacker.traits.separation, 0.25),
            (attacker.traits.burst, 0.30),
            (attacker.traits.finishing, 0.15),
            (attacker.traits.decision_making, 0.10),
            (attacker.traits.ball_security, 0.10),
            (attacker.traits.foul_drawing, 0.10),
        ]
    )
    def_score = weighted_sum(
        [
            (primary_defender.traits.containment if primary_defender else 10.0, 0.35),
            (primary_defender.traits.foul_discipline if primary_defender else 10.0, 0.15),
            (help_defender.traits.interior_def if help_defender else 10.0, 0.20),
            (help_defender.traits.rim_protect if help_defender else 10.0, 0.20),
            (primary_defender.traits.steal_pressure if primary_defender else 10.0, 0.10),
        ]
    )
    drive_advantage = sigmoid_normalize(off_score - def_score)
    branch_rates = adjust_branches(
        base_rates={
            "rim_clean": 0.32,
            "rim_contested": 0.26,
            "foul": 0.22,
            "kickout": 0.08,
            "strip": 0.08,
            "charge": 0.04,
            "bail_pullup": 0.02,
        },
        advantage=drive_advantage,
        branch_effects={
            "rim_clean": 1,
            "rim_contested": 0,
            "foul": 1,
            "kickout": -1,
            "strip": -1,
            "charge": -1,
            "bail_pullup": -1,
        },
    )
    branch_rates = _rescale_branch_probability(
        branch_rates,
        "foul",
        _foul_branch_scale(attacker, primary_defender, drive_advantage),
    )
    branch = _weighted_choice(rng, list(branch_rates.items()), default="rim_contested")
    drive_event = EventContext(
        event_type=EventType.DRIVE,
        actor_id=attacker.player_id,
        defender_id=primary_defender.player_id if primary_defender else None,
        location=_court_point("paint"),
        success_probability=_clamp(branch_rates.get(branch, 0.3), 0.05, 0.95),
        realized_success=branch not in {"strip", "charge"},
        notes=f"drive_branch={branch};advantage={drive_advantage:.3f}",
    )

    if branch == "strip":
        return {"result_type": "turnover", "events": (drive_event,), "turnover_player_id": attacker.player_id, "turnover_type": TurnoverType.STRIP}
    if branch == "charge":
        return {"result_type": "turnover", "events": (drive_event,), "turnover_player_id": attacker.player_id, "turnover_type": TurnoverType.CHARGE}
    if branch == "foul":
        foul_event = EventContext(
            event_type=EventType.FOUL,
            actor_id=attacker.player_id,
            defender_id=primary_defender.player_id if primary_defender else None,
            location=_court_point("paint"),
            foul_drawn=True,
            success_probability=branch_rates["foul"],
            realized_success=True,
            notes="drive foul",
        )
        foul_type = _choose_drive_foul_type(attacker, drive_advantage, rng)
        shot_type = ShotType.RIM if foul_type == FoulOutcomeType.AND_ONE else None
        return {"result_type": "foul", "events": (drive_event, foul_event), "shooter": attacker, "foul_type": foul_type, "shot_type": shot_type}
    if branch == "kickout":
        receiver = _best_shooter(context, exclude_ids={attacker.player_id}, rng=rng)
        pass_event = EventContext(
            event_type=EventType.PASS,
            actor_id=attacker.player_id,
            receiver_id=receiver.player_id,
            defender_id=help_defender.player_id if help_defender else None,
            location=_court_point("wing"),
            success_probability=0.86,
            realized_success=True,
            notes="drive kickout",
        )
        shot_defender = _pick_closeout_defender(context, receiver.player_id, primary_defender)
        return {
            "result_type": "nested_action",
            "events": (drive_event, pass_event),
            "result": _resolve_kickout_action(context, receiver, shot_defender, help_defender, drive_advantage, rng),
            "assister_id": attacker.player_id,
            "default_off_rebounder": attacker,
            "defender": shot_defender,
        }
    if branch == "bail_pullup":
        shot_result = resolve_pullup(attacker, primary_defender, DefensiveCoverage.DROP, off_screen=False, shot_type=ShotType.MIDRANGE)
        return {"result_type": "nested_shot", "events": (drive_event,), "result": shot_result, "assister_id": None, "default_off_rebounder": attacker, "defender": primary_defender}

    shot_type = ShotType.RIM if branch == "rim_clean" else ShotType.PAINT
    contest_boost = 1.0 if branch == "rim_clean" else -0.8
    shot_quality = _shot_quality_from_advantage(
        attacker.traits.finishing + (0.10 * attacker.traits.reach),
        ((help_defender.traits.rim_protect if help_defender else 10.0) * 0.90) + (0.06 * (help_defender.traits.reach if help_defender else 10.0)),
        drive_advantage,
        base=0.64 + contest_boost * 0.04,
    )
    return {"result_type": "shot", "events": (drive_event,), "shooter": attacker, "shot_type": shot_type, "shot_quality": shot_quality}


def resolve_pullup(
    shooter: PlayerSimProfile,
    defender: PlayerSimProfile | None,
    coverage: DefensiveCoverage,
    *,
    off_screen: bool,
    shot_type: ShotType | None = None,
) -> dict[str, object]:
    off_score = weighted_sum([(shooter.traits.pullup_shooting, 0.50), (shooter.traits.separation, 0.30), (shooter.traits.burst, 0.20)])
    closeout_value = min(defender.traits.closeout if defender else 10.0, 20.0)
    if off_screen:
        def_pairs = [
            (defender.traits.containment if defender else 10.0, 0.45),
            (closeout_value, WEAK_PROXY_WEIGHT_CAP),
            (defender.traits.screen_nav if defender else 10.0, 0.35),
        ]
    else:
        def_pairs = [
            (defender.traits.containment if defender else 10.0, 0.55),
            (closeout_value, WEAK_PROXY_WEIGHT_CAP),
            (defender.traits.reach if defender else 10.0, 0.25),
        ]
    def_score = weighted_sum(def_pairs)
    quality = sigmoid_normalize(off_score - def_score)
    resolved_shot_type = shot_type or (ShotType.ABOVE_BREAK_THREE if shooter.traits.pullup_shooting >= 12.0 else ShotType.MIDRANGE)
    base = 0.39 if resolved_shot_type == ShotType.ABOVE_BREAK_THREE else 0.43
    event = EventContext(
        event_type=EventType.SHOT,
        actor_id=shooter.player_id,
        defender_id=defender.player_id if defender else None,
        location=_court_point("three" if resolved_shot_type == ShotType.ABOVE_BREAK_THREE else "midrange"),
        shot_type=resolved_shot_type,
        success_probability=_shot_quality_from_advantage(off_score, def_score, quality, base=base),
        realized_success=None,
        notes=f"pullup off_screen={off_screen};coverage={coverage.value}",
    )
    return {"result_type": "shot", "events": (event,), "shooter": shooter, "shot_type": resolved_shot_type, "shot_quality": event.success_probability}


def resolve_catch_and_shoot(
    context: PossessionContext,
    shooter: PlayerSimProfile,
    defender: PlayerSimProfile | None,
    upstream_advantage: float,
    rng: random.Random,
) -> dict[str, object]:
    off_score = weighted_sum([(shooter.traits.catch_shoot, 0.60), (10.0 + (upstream_advantage - 0.5) * 10.0, 0.25), (10.0, 0.15)])
    def_score = weighted_sum(
        [
            (defender.traits.containment if defender else 10.0, 0.40),
            (min(defender.traits.closeout if defender else 10.0, 20.0), WEAK_PROXY_WEIGHT_CAP),
            (defender.traits.reach if defender else 10.0, 0.20),
            (defender.traits.size if defender else 10.0, 0.20),
        ]
    )
    quality = sigmoid_normalize(off_score - def_score)
    shot_type = ShotType.CORNER_THREE if rng.random() < _corner_three_probability(context, shooter, upstream_advantage) else ShotType.ABOVE_BREAK_THREE
    event = EventContext(
        event_type=EventType.SHOT,
        actor_id=shooter.player_id,
        defender_id=defender.player_id if defender else None,
        location=_court_point("corner" if shot_type == ShotType.CORNER_THREE else "three"),
        shot_type=shot_type,
        success_probability=_shot_quality_from_advantage(off_score, def_score, quality, base=0.375),
        realized_success=None,
        notes="catch_and_shoot",
    )
    return {"result_type": "shot", "events": (event,), "shooter": shooter, "shot_type": shot_type, "shot_quality": event.success_probability}


def resolve_rebound(
    context: PossessionContext,
    shooter: PlayerSimProfile,
    shot_type: ShotType,
    rng: random.Random,
    default_off_rebounder: PlayerSimProfile | None = None,
) -> tuple[EventContext, bool, str | None]:
    offensive_players = list(_all_players(context, offense=True))
    defensive_players = list(_all_players(context, offense=False))
    crash_candidates = sorted(
        offensive_players,
        key=lambda player: (
            player.traits.oreb + (0.35 * player.traits.size) + (0.15 * player.traits.reach),
            player.player_id != shooter.player_id,
        ),
        reverse=True,
    )
    crashers = crash_candidates[:3]
    boxers = sorted(defensive_players, key=lambda player: (player.traits.dreb, player.traits.size), reverse=True)[:4]
    rim_protector = max(defensive_players, key=lambda player: player.traits.rim_protect, default=None)

    crash_glass_modifier = 0.95 + (context.offensive_tactics.crash_glass_rate * 1.45)
    off_score = sum(player.traits.oreb + (0.15 * player.traits.size) for player in crashers) * crash_glass_modifier
    def_score = sum(player.traits.dreb + (0.15 * player.traits.size) for player in boxers)
    if rim_protector is not None:
        def_score += rim_protector.traits.rim_protect * 0.3
    location_modifier = {
        ShotType.RIM: 1.15,
        ShotType.PAINT: 1.08,
        ShotType.MIDRANGE: 1.0,
        ShotType.CORNER_THREE: 0.92,
        ShotType.ABOVE_BREAK_THREE: 0.88,
    }.get(shot_type, 1.0)
    rebound_edge = off_score / max((off_score + def_score) / 2.0, 1.0)
    oreb_probability = _clamp(0.275 * location_modifier * rebound_edge, 0.14, 0.40)
    offensive_rebound = rng.random() < oreb_probability

    if offensive_rebound:
        rebounder = _weighted_choice(
            rng,
            [
                (player, max(0.1, player.traits.oreb + (0.2 * player.traits.size)))
                for player in crashers
            ],
            default=default_off_rebounder,
        )
    else:
        rebounder = _weighted_choice(
            rng,
            [
                (player, max(0.1, player.traits.dreb + (0.2 * player.traits.size)))
                for player in boxers
            ],
            default=rim_protector,
        )
    rebound_event = EventContext(
        event_type=EventType.REBOUND,
        actor_id=rebounder.player_id if rebounder else None,
        location=_court_point("paint" if shot_type in {ShotType.RIM, ShotType.PAINT} else "top"),
        success_probability=oreb_probability if offensive_rebound else (1.0 - oreb_probability),
        realized_success=True,
        notes="offensive_rebound" if offensive_rebound else "defensive_rebound",
    )
    return rebound_event, offensive_rebound, rebounder.player_id if rebounder else None


def weighted_sum(pairs: Iterable[tuple[float, float]]) -> float:
    return sum(value * weight for value, weight in pairs)


def sigmoid_normalize(raw_advantage: float, k: float = 0.15) -> float:
    return 1.0 / (1.0 + math.exp(-k * raw_advantage))


def adjust_branches(
    base_rates: dict[str, float],
    advantage: float,
    branch_effects: dict[str, int],
    sensitivity: float = 1.5,
) -> dict[str, float]:
    shift = advantage - 0.5
    adjusted = {}
    for branch, base in base_rates.items():
        effect = branch_effects.get(branch, 0)
        adjusted[branch] = max(base * (1.0 + effect * shift * sensitivity), 0.001)
    total = sum(adjusted.values())
    return {key: value / total for key, value in adjusted.items()}


def _resolve_pnr_advantage(
    context: PossessionContext,
    handler: PlayerSimProfile,
    screener: PlayerSimProfile | None,
    on_ball_defender: PlayerSimProfile | None,
    help_defender: PlayerSimProfile | None,
    coverage: DefensiveCoverage,
) -> float:
    spacing_score = context.offense_lineup.spacing_score
    if coverage == DefensiveCoverage.DROP:
        off_score = weighted_sum(
            [
                (handler.traits.pullup_shooting, 0.30),
                (handler.traits.separation, 0.20),
                (handler.traits.burst, 0.15),
                (screener.traits.screen_setting if screener else 8.0, 0.15),
                (handler.traits.ball_security, 0.10),
                (spacing_score, 0.10),
            ]
        )
        def_score = weighted_sum(
            [
                (on_ball_defender.traits.screen_nav if on_ball_defender else 10.0, 0.30),
                (on_ball_defender.traits.containment if on_ball_defender else 10.0, 0.20),
                (help_defender.traits.interior_def if help_defender else 10.0, 0.25),
                (help_defender.traits.rim_protect if help_defender else 10.0, 0.15),
                (min(on_ball_defender.traits.closeout if on_ball_defender else 10.0, 20.0), 0.10),
            ]
        )
    else:
        off_score = weighted_sum(
            [
                (handler.traits.separation, 0.30),
                (handler.traits.burst, 0.25),
                (handler.traits.finishing, 0.15),
                (handler.traits.pullup_shooting, 0.15),
                (handler.traits.ball_security, 0.05),
                (spacing_score, 0.10),
            ]
        )
        def_score = weighted_sum(
            [
                (on_ball_defender.traits.containment if on_ball_defender else 10.0, 0.35),
                (on_ball_defender.traits.foul_discipline if on_ball_defender else 10.0, 0.20),
                (help_defender.traits.rim_protect if help_defender else 10.0, 0.20),
                (help_defender.traits.interior_def if help_defender else 10.0, 0.15),
                (min(on_ball_defender.traits.closeout if on_ball_defender else 10.0, 20.0), 0.10),
            ]
        )
    return sigmoid_normalize(off_score - def_score)


def _resolve_iso_advantage(
    handler: PlayerSimProfile,
    on_ball_defender: PlayerSimProfile | None,
    help_defender: PlayerSimProfile | None,
) -> float:
    off_score = weighted_sum(
        [
            (handler.traits.separation, 0.25),
            (handler.traits.burst, 0.20),
            (handler.traits.pullup_shooting, 0.20),
            (handler.traits.finishing, 0.10),
            (handler.traits.decision_making, 0.10),
            (handler.traits.ball_security, 0.10),
            (handler.traits.foul_drawing, 0.05),
        ]
    )
    def_score = weighted_sum(
        [
            (on_ball_defender.traits.containment if on_ball_defender else 10.0, 0.40),
            (on_ball_defender.traits.foul_discipline if on_ball_defender else 10.0, 0.15),
            (on_ball_defender.traits.steal_pressure if on_ball_defender else 10.0, 0.15),
            (help_defender.traits.rim_protect if help_defender else 10.0, 0.20),
            (min(help_defender.traits.closeout if help_defender else 10.0, 20.0), 0.10),
        ]
    )
    return sigmoid_normalize(off_score - def_score)


def _clock_urgency(shot_clock_remaining: float) -> float:
    return _clamp((24.0 - shot_clock_remaining) / 10.0, 0.0, 1.0)


def _initial_off_ball_states(context: PossessionContext, exclude_ids: set[str]) -> dict[str, str]:
    states: dict[str, str] = {}
    for player in _all_players(context, offense=True):
        if player.player_id in exclude_ids:
            continue
        if player.offensive_role == OffensiveRole.SPACER:
            states[player.player_id] = "corner_drift"
        elif player.offensive_role in {OffensiveRole.SLASHER, OffensiveRole.MOVEMENT_SHOOTER} or (
            player.traits.separation + player.traits.burst >= player.traits.catch_shoot + 3.0
        ):
            states[player.player_id] = "cutting"
        elif player.offensive_role in {OffensiveRole.ROLL_BIG, OffensiveRole.POST_HUB}:
            states[player.player_id] = "crashing"
        else:
            states[player.player_id] = "spotted_up"
    return states


def _eligible_receivers(
    context: PossessionContext,
    state: ProgressionState,
    exclude_ids: set[str],
) -> list[PlayerSimProfile]:
    candidates: list[PlayerSimProfile] = []
    for player in _all_players(context, offense=True):
        if player.player_id in exclude_ids:
            continue
        off_ball_state = state.off_ball_states.get(player.player_id, "spotted_up")
        if off_ball_state not in {"spotted_up", "corner_drift", "relocated", "cutting", "crashing"}:
            continue
        candidates.append(player)
    return candidates


def _second_side_loop(
    context: PossessionContext,
    state: ProgressionState,
    rng: random.Random,
) -> dict[str, object]:
    receiver = _get_player(context, state.current_receiver_id or state.ball_handler_id)
    shot_defender = _get_player(context, state.primary_defender_id)
    help_defender = _get_player(context, state.help_defender_id)
    if receiver is None:
        fallback = _get_player(context, state.ball_handler_id)
        return resolve_pullup(fallback, shot_defender, state.coverage, off_screen=False, shot_type=ShotType.MIDRANGE)

    if state.shot_clock_remaining <= 4.0 or state.swing_count >= 3:
        bailout_type = ShotType.ABOVE_BREAK_THREE if receiver.traits.pullup_shooting >= 12.0 else ShotType.MIDRANGE
        return {
            "result_type": "nested_shot",
            "events": (),
            "result": resolve_pullup(receiver, shot_defender, state.coverage, off_screen=False, shot_type=bailout_type),
            "assister_id": None,
            "default_off_rebounder": receiver,
            "defender": shot_defender,
        }

    contest_level = ((shot_defender.traits.closeout if shot_defender else 10.0) * 0.6) + ((shot_defender.traits.containment if shot_defender else 10.0) * 0.4)
    openness = max(0.0, receiver.traits.catch_shoot - contest_level)
    shot_aggression = (receiver.traits.catch_shoot * 0.55) + (receiver.traits.decision_making * 0.25) + (receiver.traits.offensive_load * 0.20)
    shoot_probability = _clamp(
        0.34
        + (openness / 20.0) * 0.35
        + (shot_aggression / 20.0) * 0.20
        + max(0.0, (8.0 - state.shot_clock_remaining) / 8.0) * 0.25
        - (context.offensive_tactics.second_side_rate * 0.15),
        0.12,
        0.86,
    )
    attack_probability = _clamp(
        context.offensive_tactics.closeout_attack_rate * 0.40
        + ((receiver.traits.separation + receiver.traits.burst - receiver.traits.catch_shoot) / 40.0) * 0.25
        + max(0.0, 0.5 - openness / 20.0) * 0.10,
        0.08,
        0.46,
    )
    cut_probability = _clamp(
        0.05
        + (context.offensive_tactics.second_side_rate * 0.12)
        + (0.08 if state.paint_touched else 0.0)
        + (0.06 if state.help_committed else 0.0),
        0.03,
        0.24,
    )
    swing_probability = _clamp(
        (1.0 - shoot_probability) * 0.55
        + (context.offensive_tactics.second_side_rate * 0.25)
        - max(0.0, (6.0 - state.shot_clock_remaining) / 6.0) * 0.30,
        0.0,
        0.55,
    )
    reset_probability = _clamp(
        0.08
        + (context.offensive_tactics.second_side_rate * 0.14)
        + (0.10 if state.shot_clock_remaining >= 8.0 else 0.0)
        - (shoot_probability * 0.10)
        - (attack_probability * 0.18),
        0.05,
        0.30,
    )
    re_screen_probability = _clamp(
        0.06
        + (context.offensive_tactics.pick_and_roll_rate * 0.18)
        + (context.offensive_tactics.second_side_rate * 0.12)
        + (0.08 if state.shot_clock_remaining >= 9.0 else 0.0),
        0.04,
        0.28,
    )
    total = shoot_probability + attack_probability + cut_probability + swing_probability + reset_probability + re_screen_probability
    if total > 1.0:
        scale = 1.0 / total
        shoot_probability *= scale
        attack_probability *= scale
        cut_probability *= scale
        swing_probability *= scale
        reset_probability *= scale
        re_screen_probability *= scale

    decision = _weighted_choice(
        rng,
        [
            ("shoot", shoot_probability),
            ("attack", attack_probability),
            ("cut", cut_probability),
            ("swing", swing_probability),
            ("re_screen", re_screen_probability),
            ("reset", reset_probability),
        ],
        default="shoot",
    )
    if decision == "shoot":
        return resolve_catch_and_shoot(context, receiver, shot_defender, 0.5 + (openness / 40.0), rng)
    if decision == "attack":
        closeout_event = EventContext(
            event_type=EventType.DRIVE,
            actor_id=receiver.player_id,
            defender_id=shot_defender.player_id if shot_defender else None,
            location=_court_point("wing"),
            success_probability=_clamp(0.48 + (openness / 20.0) * 0.14 + (context.offensive_tactics.closeout_attack_rate * 0.12), 0.28, 0.88),
            realized_success=True,
            notes="attack_closeout",
        )
        return {
            "result_type": "nested_action",
            "events": (closeout_event,),
            "result": resolve_drive_attempt(context, receiver, shot_defender, help_defender, rng),
            "assister_id": state.last_passer_id,
            "default_off_rebounder": receiver,
            "defender": shot_defender,
        }
    if decision == "cut":
        cutter = _pick_cutter(context, state, exclude_ids={receiver.player_id, state.last_passer_id or ""}, rng=rng)
        if cutter is not None:
            cut_defender = _pick_closeout_defender(context, cutter.player_id, help_defender or shot_defender)
            cut_event = EventContext(
                event_type=EventType.CUT,
                actor_id=cutter.player_id,
                receiver_id=receiver.player_id,
                defender_id=cut_defender.player_id if cut_defender else None,
                location=_court_point("paint"),
                success_probability=_clamp(
                    0.52
                    + ((receiver.traits.pass_vision + receiver.traits.pass_accuracy + receiver.traits.decision_making - 30.0) / 60.0) * 0.18
                    + ((cutter.traits.separation + cutter.traits.finishing - 22.0) / 40.0) * 0.12,
                    0.32,
                    0.88,
                ),
                realized_success=True,
                notes="cut_find",
            )
            shot_quality = _shot_quality_from_advantage(
                cutter.traits.finishing + (0.08 * cutter.traits.separation) + (0.05 * cutter.traits.size),
                ((cut_defender.traits.interior_def if cut_defender else 10.0) * 0.84) + (0.08 * (cut_defender.traits.rim_protect if cut_defender else 10.0)),
                0.63,
                base=0.59,
            )
            return {
                "result_type": "nested_shot",
                "events": (cut_event,),
                "result": {
                    "result_type": "shot",
                    "events": (),
                    "shooter": cutter,
                    "shot_type": ShotType.RIM if cutter.traits.finishing >= 12.0 else ShotType.PAINT,
                    "shot_quality": shot_quality,
                },
                "assister_id": receiver.player_id,
                "default_off_rebounder": cutter,
                "defender": cut_defender,
            }
    if decision == "re_screen":
        side_zone = _side_pnr_zone_for_player(context, receiver.player_id)
        screener = _pick_re_screen_big(context, exclude_ids={receiver.player_id, state.last_passer_id or ""})
        re_screen_event = EventContext(
            event_type=EventType.SCREEN,
            actor_id=screener.player_id if screener else None,
            receiver_id=receiver.player_id,
            defender_id=shot_defender.player_id if shot_defender else None,
            location=_court_point("left_wing" if side_zone == CourtZone.LEFT_WING else "right_wing"),
            success_probability=_clamp(0.70 + ((screener.traits.screen_setting if screener else 10.0) - 10.0) / 40.0, 0.58, 0.92),
            realized_success=True,
            notes="second_side re_screen",
        )
        side_call = PlayCall(
            family=PlayFamily.HIGH_PICK_AND_ROLL,
            primary_actor_id=receiver.player_id,
            screener_id=screener.player_id if screener else None,
            target_zone=side_zone,
        )
        side_context = replace(
            context,
            clock=replace(context.clock, shot_clock=max(5.0, state.shot_clock_remaining - 2.0)),
            current_phase=PossessionPhase.SECONDARY_ACTION,
            play_call=side_call,
            coverage=state.coverage,
            entry_type=EntryType.NORMAL,
            entry_source=EntrySource.DEAD_BALL,
        )
        side_outcome = simulate_possession(side_context, rng)
        return {
            "result_type": "outcome",
            "outcome": _prepend_events(side_outcome, (re_screen_event,)),
        }
    if decision == "reset":
        secondary = _secondary_creator(context, exclude_ids={receiver.player_id})
        secondary = secondary or _get_player(context, state.ball_handler_id)
        secondary_assignment = _find_assignment(context.defensive_assignments, secondary.player_id)
        secondary_defender = _get_player(context, secondary_assignment.defender_id if secondary_assignment else None)
        reset_pass = EventContext(
            event_type=EventType.PASS,
            actor_id=receiver.player_id,
            receiver_id=secondary.player_id,
            defender_id=shot_defender.player_id if shot_defender else None,
            location=_court_point("top"),
            success_probability=0.88,
            realized_success=True,
            notes="second_side reset",
        )
        if state.shot_clock_remaining > 6.5:
            next_play = PlayCall(
                family=PlayFamily.ISO if secondary.traits.separation >= 12.0 else PlayFamily.HIGH_PICK_AND_ROLL,
                primary_actor_id=secondary.player_id,
                screener_id=state.screener_id if state.screener_id and state.screener_id != secondary.player_id else None,
                target_zone=CourtZone.TOP,
            )
            reset_context = replace(
                context,
                clock=replace(context.clock, shot_clock=max(4.0, state.shot_clock_remaining - 2.0)),
                play_call=next_play,
                coverage=state.coverage,
                current_phase=PossessionPhase.SECONDARY_ACTION,
                entry_type=EntryType.NORMAL,
            )
            reset_outcome = simulate_possession(reset_context, rng)
            return {
                "result_type": "outcome",
                "outcome": _prepend_events(reset_outcome, (reset_pass,)),
            }
        return {
            "result_type": "nested_shot",
            "events": (reset_pass,),
            "result": resolve_pullup(
                secondary,
                secondary_defender,
                state.coverage,
                off_screen=False,
                shot_type=ShotType.ABOVE_BREAK_THREE if secondary.traits.pullup_shooting >= 14.0 else ShotType.MIDRANGE,
            ),
            "assister_id": None,
            "default_off_rebounder": receiver,
            "defender": secondary_defender,
        }

    eligible = _eligible_receivers(context, state, {receiver.player_id, state.last_passer_id or ""})
    if not eligible:
        return resolve_catch_and_shoot(context, receiver, shot_defender, 0.5 + (openness / 40.0), rng)
    weighted = []
    for player in eligible:
        off_ball_state = state.off_ball_states.get(player.player_id, "spotted_up")
        corner_bonus = 1.0 if off_ball_state == "corner_drift" else 0.0
        openness_estimate = max(0.0, player.traits.catch_shoot - ((shot_defender.traits.closeout if shot_defender else 10.0) * 0.6))
        weight = (
            (player.traits.catch_shoot * 0.40)
            + (_shooter_distribution_weight(context, player) * 5.0 * 0.25)
            + (corner_bonus * 20.0 * 0.20)
            + (openness_estimate * 0.15)
        )
        weighted.append((player, max(0.1, weight)))
    next_receiver = _weighted_choice(rng, weighted, default=eligible[0])
    next_defender = _pick_closeout_defender(context, next_receiver.player_id, shot_defender)
    swing_event = EventContext(
        event_type=EventType.PASS,
        actor_id=receiver.player_id,
        receiver_id=next_receiver.player_id,
        defender_id=shot_defender.player_id if shot_defender else None,
        location=_court_point("wing"),
        success_probability=_clamp(0.82 + ((receiver.traits.pass_accuracy + receiver.traits.pass_vision - 20.0) / 40.0) * 0.15, 0.72, 0.95),
        realized_success=True,
        notes="swing pass",
    )
    next_state = ProgressionState(
        shot_clock_remaining=max(0.0, state.shot_clock_remaining - 2.5),
        entry_type=state.entry_type,
        advantage_state=AdvantageState.FORCED_HELP,
        ball_handler_id=state.ball_handler_id,
        current_receiver_id=next_receiver.player_id,
        primary_defender_id=next_defender.player_id if next_defender else None,
        help_defender_id=state.help_defender_id,
        screener_id=state.screener_id,
        coverage=state.coverage,
        pass_chain=state.pass_chain + (receiver.player_id,),
        off_ball_states={**state.off_ball_states, receiver.player_id: "relocated"},
        clock_urgency=_clock_urgency(max(0.0, state.shot_clock_remaining - 2.5)),
        possession_events=(swing_event,),
        last_passer_id=receiver.player_id,
        paint_touched=state.paint_touched,
        help_committed=state.help_committed,
        swing_count=state.swing_count + 1,
    )
    return {
        "result_type": "nested_action",
        "events": (swing_event,),
        "result": _second_side_loop(context, next_state, rng),
        "assister_id": state.last_passer_id,
        "default_off_rebounder": receiver,
        "defender": next_defender,
    }


def _resolve_progression_state(
    context: PossessionContext,
    state: ProgressionState,
    rng: random.Random,
) -> PossessionOutcome:
    ball_handler = _get_player(context, state.ball_handler_id)
    current_actor = _get_player(context, state.current_receiver_id or state.ball_handler_id)
    screener = _get_player(context, state.screener_id)
    primary_defender = _get_player(context, state.primary_defender_id)
    help_defender = _get_player(context, state.help_defender_id)
    if state.terminal_result is not None:
        if state.terminal_result["kind"] == "outcome":
            return state.terminal_result["outcome"]
        return _materialize_shot_result(
            context=context,
            result=state.terminal_result["result"],
            lead_events=state.possession_events,
            assister_id=state.terminal_result.get("assister_id"),
            default_off_rebounder=state.terminal_result.get("default_off_rebounder", screener or current_actor),
            defender=state.terminal_result.get("defender", primary_defender),
            rng=rng,
        )

    if state.advantage_state == AdvantageState.PAINT_TOUCH:
        drive_result = resolve_drive_attempt(context, current_actor, primary_defender, help_defender, rng)
        return _materialize_action_result(
            context=context,
            result=drive_result,
            lead_events=state.possession_events,
            default_assister_id=None,
            default_off_rebounder=screener or current_actor,
            defender=primary_defender,
            rng=rng,
        )

    if state.advantage_state == AdvantageState.PULL_UP_SPACE:
        shot_type = ShotType.ABOVE_BREAK_THREE if current_actor.traits.pullup_shooting >= 13.0 else ShotType.MIDRANGE
        if state.shot_clock_remaining <= 5.0 and current_actor.traits.pullup_shooting < 12.0:
            shot_type = ShotType.MIDRANGE
        pullup_result = resolve_pullup(current_actor, primary_defender, state.coverage, off_screen=(screener is not None), shot_type=shot_type)
        return _materialize_shot_result(
            context=context,
            result=pullup_result,
            lead_events=state.possession_events,
            assister_id=None,
            default_off_rebounder=screener or current_actor,
            defender=primary_defender,
            rng=rng,
        )

    if state.advantage_state == AdvantageState.FORCED_HELP:
        kickout_result = _second_side_loop(context, state, rng)
        return _materialize_action_result(
            context=context,
            result=kickout_result,
            lead_events=state.possession_events,
            default_assister_id=state.last_passer_id,
            default_off_rebounder=screener or ball_handler,
            defender=primary_defender,
            rng=rng,
        )

    reset_actor = current_actor
    reset_result = resolve_pullup(
        reset_actor,
        primary_defender,
        state.coverage,
        off_screen=False,
        shot_type=ShotType.ABOVE_BREAK_THREE if reset_actor.traits.pullup_shooting >= 14.0 and reset_actor.traits.separation >= 12.0 else ShotType.MIDRANGE,
    )
    return _materialize_shot_result(
        context=context,
        result=reset_result,
        lead_events=state.possession_events,
        assister_id=None,
        default_off_rebounder=screener or ball_handler,
        defender=primary_defender,
        rng=rng,
    )


def _materialize_action_result(
    context: PossessionContext,
    result: dict[str, object],
    lead_events: tuple[EventContext, ...],
    default_assister_id: str | None,
    default_off_rebounder: PlayerSimProfile,
    defender: PlayerSimProfile | None,
    rng: random.Random,
) -> PossessionOutcome:
    result_type = result["result_type"]
    if result_type == "outcome":
        return _prepend_events(result["outcome"], lead_events)
    if result_type == "turnover":
        turnover_event = EventContext(
            event_type=EventType.TURNOVER,
            actor_id=result["turnover_player_id"],
            defender_id=defender.player_id if defender else None,
            location=_court_point("paint"),
            turnover_type=result["turnover_type"],
            success_probability=1.0,
            realized_success=True,
            notes="drive turnover",
        )
        return _turnover_outcome(
            result["turnover_player_id"],
            lead_events + tuple(result["events"]) + (turnover_event,),
            steal_player_id=defender.player_id if defender and result["turnover_type"] == TurnoverType.STRIP else None,
        )
    if result_type == "foul":
        return _foul_outcome(
            result["shooter"].player_id,
            result["shooter"].traits.ft_pct_raw,
            lead_events + tuple(result["events"]),
            rng,
            assisting_player_id=default_assister_id,
            foul_type=result.get("foul_type", FoulOutcomeType.TWO_SHOT),
            shot_type=result.get("shot_type"),
        )
    if result_type == "nested_action":
        return _materialize_action_result(
            context=context,
            result=result["result"],
            lead_events=lead_events + tuple(result["events"]),
            default_assister_id=result.get("assister_id", default_assister_id),
            default_off_rebounder=result.get("default_off_rebounder", default_off_rebounder),
            defender=result.get("defender", defender),
            rng=rng,
        )
    if result_type == "nested_shot":
        return _materialize_shot_result(
            context=context,
            result=result["result"],
            lead_events=lead_events + tuple(result["events"]),
            assister_id=result.get("assister_id", default_assister_id),
            default_off_rebounder=result.get("default_off_rebounder", default_off_rebounder),
            defender=result.get("defender", defender),
            rng=rng,
        )
    return _materialize_shot_result(
        context=context,
        result=result,
        lead_events=lead_events,
        assister_id=default_assister_id,
        default_off_rebounder=default_off_rebounder,
        defender=defender,
        rng=rng,
    )


def _materialize_shot_result(
    context: PossessionContext,
    result: dict[str, object],
    lead_events: tuple[EventContext, ...],
    assister_id: str | None,
    default_off_rebounder: PlayerSimProfile,
    defender: PlayerSimProfile | None,
    rng: random.Random,
) -> PossessionOutcome:
    shooter = result["shooter"]
    shot_type = result["shot_type"]
    shot_quality = _clamp(float(result["shot_quality"]), 0.08, 0.92)
    existing_events = tuple(result.get("events", ()))
    pre_shot_events = tuple(event for event in existing_events if event.event_type != EventType.SHOT)
    made = rng.random() < shot_quality
    block_player_id = None
    if not made and defender is not None:
        if shot_type in {ShotType.RIM, ShotType.PAINT}:
            block_prob = _clamp(0.01 + (((defender.traits.rim_protect + defender.traits.reach) / 40.0) * 0.08), 0.01, 0.12)
        else:
            block_prob = _clamp(0.002 + ((defender.traits.reach / 20.0) * 0.02), 0.002, 0.03)
        if rng.random() < block_prob:
            block_player_id = defender.player_id
    points = _shot_points(shot_type) if made else 0
    credited_assister_id = assister_id if made and _is_assist_eligible(lead_events + pre_shot_events, shot_type) else None
    shot_event = EventContext(
        event_type=EventType.SHOT,
        actor_id=shooter.player_id,
        defender_id=defender.player_id if defender else None,
        location=_court_point(_shot_location_key(shot_type)),
        shot_type=shot_type,
        success_probability=shot_quality,
        realized_success=made,
        points_scored=points,
        notes="mechanic_resolved" if block_player_id is None else "mechanic_resolved;blocked",
    )
    if made:
        return PossessionOutcome(
            points_scored=points,
            made_shot=True,
            turnover=False,
            foul_committed=False,
            offensive_rebound=False,
            shooting_player_id=shooter.player_id,
            assisting_player_id=credited_assister_id,
            rebounder_id=None,
            turnover_player_id=None,
            events=lead_events + pre_shot_events + (shot_event,),
        free_throws_attempted=0,
        foul_type=None,
        steal_player_id=None,
        block_player_id=None,
        )
    rebound_event, offensive_rebound, rebounder_id = resolve_rebound(context, shooter, shot_type, rng, default_off_rebounder)
    return PossessionOutcome(
        points_scored=0,
        made_shot=False,
        turnover=False,
        foul_committed=False,
        offensive_rebound=offensive_rebound,
        shooting_player_id=shooter.player_id,
        assisting_player_id=credited_assister_id,
        rebounder_id=rebounder_id,
        turnover_player_id=None,
        events=lead_events + pre_shot_events + (shot_event, rebound_event),
        free_throws_attempted=0,
        foul_type=None,
        steal_player_id=None,
        block_player_id=block_player_id,
    )


def _turnover_outcome(turnover_player_id: str, events: tuple[EventContext, ...], steal_player_id: str | None = None) -> PossessionOutcome:
    return PossessionOutcome(
        points_scored=0,
        made_shot=False,
        turnover=True,
        foul_committed=False,
        offensive_rebound=False,
        shooting_player_id=None,
        assisting_player_id=None,
        rebounder_id=None,
        turnover_player_id=turnover_player_id,
        events=events,
        free_throws_attempted=0,
        foul_type=None,
        steal_player_id=steal_player_id,
        block_player_id=None,
    )


def _foul_outcome(
    shooter_id: str,
    ft_pct_raw: float,
    events: tuple[EventContext, ...],
    rng: random.Random,
    assisting_player_id: str | None = None,
    foul_type: FoulOutcomeType = FoulOutcomeType.TWO_SHOT,
    shot_type: ShotType | None = None,
) -> PossessionOutcome:
    free_throws_attempted = {
        FoulOutcomeType.AND_ONE: 1,
        FoulOutcomeType.TWO_SHOT: 2,
        FoulOutcomeType.THREE_SHOT: 3,
    }[foul_type]
    make_prob = _clamp(ft_pct_raw, 0.0, 1.0)
    free_throw_points = sum(int(rng.random() < make_prob) for _ in range(max(free_throws_attempted, 0)))
    made_shot = foul_type == FoulOutcomeType.AND_ONE
    resolved_shot_type = shot_type or ShotType.RIM
    field_goal_points = _shot_points(resolved_shot_type) if made_shot else 0
    points_scored = field_goal_points + free_throw_points
    full_events = events
    if made_shot and not any(event.event_type == EventType.SHOT for event in events):
        foul_event = next((event for event in reversed(events) if event.event_type == EventType.FOUL), None)
        full_events = events + (
            EventContext(
                event_type=EventType.SHOT,
                actor_id=shooter_id,
                defender_id=foul_event.defender_id if foul_event else None,
                location=_court_point(_shot_location_key(resolved_shot_type)),
                shot_type=resolved_shot_type,
                success_probability=1.0,
                realized_success=True,
                points_scored=field_goal_points,
                notes="foul_and_one",
            ),
        )
    return PossessionOutcome(
        points_scored=points_scored,
        made_shot=made_shot,
        turnover=False,
        foul_committed=True,
        offensive_rebound=False,
        shooting_player_id=shooter_id,
        assisting_player_id=assisting_player_id,
        rebounder_id=None,
        turnover_player_id=None,
        events=full_events,
        free_throws_attempted=free_throws_attempted,
        foul_type=foul_type,
        steal_player_id=None,
        block_player_id=None,
    )


def _shot_quality_from_advantage(off_score: float, def_score: float, advantage: float, base: float) -> float:
    raw_delta = (off_score - def_score) / 20.0
    advantage_delta = (advantage - 0.5) * 0.24
    return _clamp(base + (raw_delta * 0.08) + advantage_delta, 0.12, 0.82)


def _select_play_call(context: PossessionContext, rng: random.Random) -> PlayCall:
    supported_weights = [(family, weight) for family, weight in context.offensive_tactics.play_family_weights.items() if family in SUPPORTED_PLAY_FAMILIES]
    family = _weighted_choice(rng, supported_weights, default=PlayFamily.HIGH_PICK_AND_ROLL)
    primary = _primary_creator_id(context, rng)
    screener = None
    secondary = None
    if family in {PlayFamily.HIGH_PICK_AND_ROLL, PlayFamily.HANDOFF, PlayFamily.DOUBLE_DRAG}:
        ordered_screeners = sorted(
            (player for player in _all_players(context, offense=True) if player.player_id != primary),
            key=lambda player: player.traits.screen_setting,
            reverse=True,
        )
        screener = ordered_screeners[0] if ordered_screeners else None
        secondary = ordered_screeners[1] if family == PlayFamily.DOUBLE_DRAG and len(ordered_screeners) > 1 else None
    return PlayCall(
        family=family,
        primary_actor_id=primary,
        secondary_actor_id=secondary.player_id if secondary else None,
        screener_id=screener.player_id if screener else None,
        target_zone=_side_pnr_zone_for_player(context, primary) if family in {PlayFamily.HANDOFF, PlayFamily.DOUBLE_DRAG} else None,
    )


def _select_coverage(context: PossessionContext, rng: random.Random) -> DefensiveCoverage:
    supported_weights = [(coverage, weight) for coverage, weight in context.defensive_tactics.coverage_weights.items() if coverage in SUPPORTED_COVERAGES]
    return _weighted_choice(rng, supported_weights, default=DefensiveCoverage.DROP)


def _find_assignment(assignments: Iterable[DefensiveAssignment], offensive_player_id: str) -> DefensiveAssignment | None:
    for assignment in assignments:
        if assignment.offensive_player_id == offensive_player_id:
            return assignment
    return None


def _primary_creator_id(context: PossessionContext, rng: random.Random) -> str:
    bias = _clamp(context.offensive_tactics.star_usage_bias, 0.75, 1.8)
    weighted_players = []
    for player_id in context.offense_lineup.player_ids:
        player = _get_player(context, player_id)
        if player is None:
            continue
        base_load = max(player.traits.offensive_load, 0.1) ** bias
        weight = (base_load * 0.82) + (player.traits.pass_vision * 0.12) + (player.traits.pass_accuracy * 0.06)
        weighted_players.append((player_id, max(0.1, weight)))
    return _weighted_choice(rng, weighted_players, default=context.offense_lineup.player_ids[0])


def _best_shooter(context: PossessionContext, exclude_ids: set[str], rng: random.Random) -> PlayerSimProfile:
    candidates = [player for player in _all_players(context, offense=True) if player.player_id not in exclude_ids]
    weighted_candidates = []
    for player in candidates:
        role_penalty = 2.5 if player.offensive_role == OffensiveRole.SPACER else 0.0
        weight = (
            (player.traits.catch_shoot * 0.36)
            + (player.traits.decision_making * 0.10)
            + (max(1.0, 15.0 - player.traits.offensive_load) * 0.38)
            + (player.traits.pass_vision * 0.10)
            - role_penalty
        )
        weight *= _shooter_distribution_weight(context, player)
        weighted_candidates.append((player, max(0.1, weight)))
    return _weighted_choice(rng, weighted_candidates, default=_get_player(context, context.offense_lineup.player_ids[0]))


def _resolve_kickout_action(
    context: PossessionContext,
    receiver: PlayerSimProfile,
    shot_defender: PlayerSimProfile | None,
    help_defender: PlayerSimProfile | None,
    upstream_advantage: float,
    rng: random.Random,
    *,
    last_passer_id: str | None = None,
    shot_clock_remaining: float | None = None,
    swing_depth: int = 0,
) -> dict[str, object]:
    state = ProgressionState(
        shot_clock_remaining=context.clock.shot_clock if shot_clock_remaining is None else shot_clock_remaining,
        entry_type=EntryType.NORMAL,
        advantage_state=AdvantageState.FORCED_HELP,
        ball_handler_id=last_passer_id or receiver.player_id,
        current_receiver_id=receiver.player_id,
        primary_defender_id=shot_defender.player_id if shot_defender else None,
        help_defender_id=help_defender.player_id if help_defender else None,
        screener_id=None,
        coverage=DefensiveCoverage.DROP,
        pass_chain=((last_passer_id,) if last_passer_id else ()),
        off_ball_states=_initial_off_ball_states(context, {receiver.player_id, last_passer_id or ""}),
        clock_urgency=_clock_urgency(context.clock.shot_clock if shot_clock_remaining is None else shot_clock_remaining),
        possession_events=(),
        last_passer_id=last_passer_id,
        paint_touched=upstream_advantage >= 0.55,
        help_committed=True,
        swing_count=swing_depth,
    )
    return _second_side_loop(context, state, rng)


def _secondary_creator(context: PossessionContext, exclude_ids: set[str]) -> PlayerSimProfile | None:
    candidates = [player for player in _all_players(context, offense=True) if player.player_id not in exclude_ids]
    return max(
        candidates,
        key=lambda player: (player.traits.offensive_load * 0.65) + (player.traits.pass_vision * 0.2) + (player.traits.pass_accuracy * 0.15),
        default=None,
    )


def _pick_help_defender(context: PossessionContext, exclude_ids: set[str]) -> PlayerSimProfile | None:
    candidates = [player for player in _all_players(context, offense=False) if player.player_id not in exclude_ids]
    return max(candidates, key=lambda player: (player.traits.help_rotation + player.traits.rim_protect + player.traits.interior_def), default=None)


def _pick_closeout_defender(context: PossessionContext, receiver_id: str, fallback: PlayerSimProfile | None) -> PlayerSimProfile | None:
    assignment = _find_assignment(context.defensive_assignments, receiver_id)
    if assignment is not None:
        return _get_player(context, assignment.defender_id)
    return fallback


def _pick_cutter(
    context: PossessionContext,
    state: ProgressionState,
    exclude_ids: set[str],
    rng: random.Random,
) -> PlayerSimProfile | None:
    candidates = []
    for player in _eligible_receivers(context, state, exclude_ids):
        off_ball_state = state.off_ball_states.get(player.player_id, "spotted_up")
        if off_ball_state not in {"cutting", "relocated", "crashing"}:
            continue
        weight = (
            (player.traits.separation * 0.35)
            + (player.traits.finishing * 0.30)
            + (player.traits.decision_making * 0.15)
            + (player.traits.size * 0.10)
            + (0.12 * player.traits.reach)
            + (4.0 if off_ball_state == "cutting" else 0.0)
        )
        candidates.append((player, max(0.1, weight)))
    return _weighted_choice(rng, candidates, default=None)


def _pick_re_screen_big(context: PossessionContext, exclude_ids: set[str]) -> PlayerSimProfile | None:
    candidates = [player for player in _all_players(context, offense=True) if player.player_id not in exclude_ids]
    return max(
        candidates,
        key=lambda player: (
            (player.traits.screen_setting * 0.52)
            + (player.traits.size * 0.18)
            + (player.traits.finishing * 0.12)
            + (4.0 if player.offensive_role in {OffensiveRole.ROLL_BIG, OffensiveRole.POST_HUB, OffensiveRole.POP_BIG} else 0.0)
        ),
        default=None,
    )


def _side_pnr_zone_for_player(context: PossessionContext, player_id: str) -> CourtZone:
    for state in context.floor_players:
        if state.player_id != player_id:
            continue
        if state.location.zone in {CourtZone.LEFT_WING, CourtZone.LEFT_CORNER, CourtZone.ELBOW_LEFT}:
            return CourtZone.LEFT_WING
        if state.location.zone in {CourtZone.RIGHT_WING, CourtZone.RIGHT_CORNER, CourtZone.ELBOW_RIGHT}:
            return CourtZone.RIGHT_WING
        return CourtZone.LEFT_WING if state.location.x < 0 else CourtZone.RIGHT_WING
    return CourtZone.RIGHT_WING


def _all_players(context: PossessionContext, *, offense: bool) -> list[PlayerSimProfile]:
    pool = context.player_pool
    player_ids = set(context.offense_lineup.player_ids if offense else context.defense_lineup.player_ids)
    return [player for player in pool if player.player_id in player_ids]


def _get_player(context: PossessionContext, player_id: str | None) -> PlayerSimProfile | None:
    if player_id is None:
        return None
    if not context.player_pool:
        raise ValueError("PossessionContext.player_pool is required for player lookup")
    for player in context.player_pool:
        if player.player_id == player_id:
            return player
    return None


def _with_entry_clock(context: PossessionContext, max_shot_clock: float) -> PossessionContext:
    return replace(
        context,
        clock=replace(context.clock, shot_clock=min(context.clock.shot_clock, max_shot_clock)),
        entry_type=EntryType.NORMAL,
        current_phase=PossessionPhase.INITIATION,
    )


def _prepend_events(outcome: PossessionOutcome, lead_events: tuple[EventContext, ...]) -> PossessionOutcome:
    if not lead_events:
        return outcome
    return replace(outcome, events=lead_events + outcome.events)


def _weighted_choice(rng: random.Random, items: list[tuple[object, float]], default):
    total = sum(max(float(weight), 0.0) for _, weight in items)
    if total <= 0:
        return default
    draw = rng.random() * total
    running = 0.0
    for value, weight in items:
        running += max(float(weight), 0.0)
        if draw <= running:
            return value
    return items[-1][0] if items else default


def _shot_points(shot_type: ShotType) -> int:
    return 3 if shot_type in {ShotType.ABOVE_BREAK_THREE, ShotType.CORNER_THREE} else 2


def _court_point(kind: str):
    from basketball_sim_schema import CourtPoint, CourtZone

    mapping = {
        "top": CourtZone.TOP,
        "wing": CourtZone.RIGHT_WING,
        "left_wing": CourtZone.LEFT_WING,
        "right_wing": CourtZone.RIGHT_WING,
        "paint": CourtZone.PAINT,
        "midrange": CourtZone.ELBOW_RIGHT,
        "three": CourtZone.TOP,
        "corner": CourtZone.LEFT_CORNER,
        "left_corner": CourtZone.LEFT_CORNER,
        "right_corner": CourtZone.RIGHT_CORNER,
        "rim": CourtZone.RIM,
    }
    coords = {
        CourtZone.TOP: (0.0, 22.0),
        CourtZone.LEFT_WING: (-18.0, 21.0),
        CourtZone.RIGHT_WING: (18.0, 21.0),
        CourtZone.PAINT: (0.0, 8.0),
        CourtZone.ELBOW_RIGHT: (8.0, 16.0),
        CourtZone.LEFT_CORNER: (-22.0, 3.0),
        CourtZone.RIGHT_CORNER: (22.0, 3.0),
        CourtZone.RIM: (0.0, 1.5),
    }
    zone = mapping[kind]
    x, y = coords[zone]
    return CourtPoint(x=x, y=y, zone=zone)


def _shot_location_key(shot_type: ShotType) -> str:
    if shot_type == ShotType.RIM:
        return "rim"
    if shot_type == ShotType.PAINT:
        return "paint"
    if shot_type == ShotType.MIDRANGE:
        return "midrange"
    if shot_type == ShotType.CORNER_THREE:
        return "corner"
    return "three"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rescale_branch_probability(branch_rates: dict[str, float], branch: str, scale: float) -> dict[str, float]:
    adjusted = dict(branch_rates)
    adjusted[branch] = max(adjusted.get(branch, 0.0) * scale, 0.001)
    total = sum(adjusted.values())
    if total <= 0:
        return branch_rates
    return {key: value / total for key, value in adjusted.items()}


def _foul_branch_scale(
    attacker: PlayerSimProfile,
    defender: PlayerSimProfile | None,
    advantage: float,
    *,
    switch_bonus: float = 0.0,
) -> float:
    defender_discipline = defender.traits.foul_discipline if defender else 10.0
    foul_draw_rating = _clamp(attacker.traits.foul_drawing, 1.0, 20.0)
    draw_factor = ((foul_draw_rating - 1.0) / 19.0) ** 2.3
    discipline_edge = (10.0 - defender_discipline) / 10.0
    advantage_edge = advantage - 0.5
    scale = 0.16 + (draw_factor * 0.76) + (discipline_edge * 0.14) + (advantage_edge * 0.08) + (switch_bonus * 0.22)
    return _clamp(scale, 0.10, 1.08)


def _choose_drive_foul_type(attacker: PlayerSimProfile, advantage: float, rng: random.Random) -> FoulOutcomeType:
    and_one_rate = _clamp(0.15 + ((attacker.traits.finishing - 10.0) / 20.0) * 0.10 + ((advantage - 0.5) * 0.07), 0.09, 0.28)
    if rng.random() < and_one_rate:
        return FoulOutcomeType.AND_ONE
    return FoulOutcomeType.TWO_SHOT


def _choose_perimeter_foul_type(handler: PlayerSimProfile, rng: random.Random) -> FoulOutcomeType:
    foul_draw_level = _clamp((handler.traits.foul_drawing - 1.0) / 19.0, 0.0, 1.0)
    three_shot_rate = _clamp(0.015 + ((handler.traits.pullup_shooting - 10.0) / 10.0) * 0.025 + (foul_draw_level * 0.02), 0.01, 0.05)
    and_one_rate = _clamp(0.06 + ((handler.traits.finishing - 10.0) / 10.0) * 0.05 + (foul_draw_level * 0.03), 0.04, 0.11)
    draw = rng.random()
    if draw < three_shot_rate:
        return FoulOutcomeType.THREE_SHOT
    if draw < (three_shot_rate + and_one_rate):
        return FoulOutcomeType.AND_ONE
    return FoulOutcomeType.TWO_SHOT


def _shooter_distribution_weight(context: PossessionContext, player: PlayerSimProfile) -> float:
    weights = context.offensive_tactics.shooter_distribution_weights
    if player.player_id in weights:
        return max(0.1, weights[player.player_id])
    role_key = player.offensive_role.value
    if role_key in weights:
        return max(0.1, weights[role_key])
    return 1.0


def _corner_three_probability(context: PossessionContext, shooter: PlayerSimProfile, upstream_advantage: float) -> float:
    probability = 0.225
    if shooter.offensive_role in {OffensiveRole.MOVEMENT_SHOOTER, OffensiveRole.SPACER}:
        probability += 0.10
    if shooter.traits.catch_shoot >= 14.0:
        probability += 0.05
    if shooter.traits.size >= 13.0:
        probability -= 0.04
    if shooter.traits.separation >= 12.0:
        probability -= 0.04
    probability += max(0.0, upstream_advantage - 0.56) * 0.10
    probability += (context.offensive_tactics.corner_spacing_bias - 0.5) * 0.20
    return _clamp(probability, 0.10, 0.37)


def _is_assist_eligible(events: tuple[EventContext, ...], shot_type: ShotType) -> bool:
    pass_events = [event for event in events if event.event_type == EventType.PASS]
    if not pass_events:
        return False
    pass_notes = " ".join(event.notes for event in pass_events).lower()
    if "reset" in pass_notes or "bailout" in pass_notes:
        return False
    if "pocket pass" in pass_notes:
        return True
    if "kickout" in pass_notes and shot_type in {ShotType.CORNER_THREE, ShotType.ABOVE_BREAK_THREE}:
        return True
    if shot_type in {ShotType.CORNER_THREE, ShotType.ABOVE_BREAK_THREE}:
        return True
    return any(event.event_type in {EventType.SCREEN, EventType.DRIVE} for event in events)
