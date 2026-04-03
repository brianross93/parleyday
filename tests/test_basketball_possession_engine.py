import random
from collections import Counter
from dataclasses import replace

from basketball_possession_engine import _best_shooter, _get_player, _pick_closeout_defender, _primary_creator_id, _resolve_kickout_action, simulate_possession
from basketball_sim_schema import (
    BasketballSide,
    CourtPoint,
    CourtZone,
    DefensiveAssignment,
    DefensiveCoverage,
    DefensiveRole,
    EntrySource,
    EntryType,
    EventType,
    FoulOutcomeType,
    FloorPlayerState,
    GameClockState,
    LineupUnit,
    OffensiveRole,
    PlayCall,
    PlayFamily,
    PlayerCondition,
    PlayerSimProfile,
    PlayerTraitProfile,
    PossessionContext,
    PossessionPhase,
    ScoreState,
    TeamTactics,
)


def _player(
    player_id: str,
    team_code: str,
    *,
    separation: float = 12.0,
    burst: float = 12.0,
    pullup_shooting: float = 12.0,
    screen_setting: float = 8.0,
    pass_vision: float = 12.0,
    pass_accuracy: float = 12.0,
    decision_making: float = 12.0,
    ball_security: float = 12.0,
    finishing: float = 12.0,
    catch_shoot: float = 11.0,
    containment: float = 11.0,
) -> PlayerSimProfile:
    return PlayerSimProfile(
        player_id=player_id,
        name=player_id,
        team_code=team_code,
        positions=("G",),
        offensive_role=OffensiveRole.PRIMARY_CREATOR,
        defensive_role=DefensiveRole.POINT_OF_ATTACK,
        traits=PlayerTraitProfile(
            ball_security=ball_security,
            separation=separation,
            burst=burst,
            pullup_shooting=pullup_shooting,
            catch_shoot=catch_shoot,
            finishing=finishing,
            pass_vision=pass_vision,
            pass_accuracy=pass_accuracy,
            decision_making=decision_making,
            screen_setting=screen_setting,
            oreb=9.0,
            free_throw_rating=12.0,
            ft_pct_raw=0.78,
            foul_drawing=11.0,
            containment=containment,
            closeout=max(1.0, containment - 1.0),
            screen_nav=max(1.0, containment - 2.0),
            interior_def=9.0,
            rim_protect=8.0,
            steal_pressure=11.0,
            dreb=10.0,
            foul_discipline=11.0,
            help_rotation=10.0,
            stamina=12.0,
            size=10.0,
            reach=10.0,
        ),
        condition=PlayerCondition(),
    )


def _context(play_family: PlayFamily, coverage: DefensiveCoverage) -> PossessionContext:
    tactics = TeamTactics(
        pace_target=99.0,
        transition_frequency=0.14,
        crash_glass_rate=0.22,
        help_aggressiveness=0.48,
        switch_rate=0.3,
        zone_rate=0.0,
        no_middle_rate=0.1,
        pre_switch_rate=0.1,
        rotation_tightness=0.75,
        late_clock_isolation_rate=0.16,
        early_offense_rate=0.18,
        pick_and_roll_rate=0.38,
        handoff_rate=0.08,
        post_touch_rate=0.06,
        off_ball_screen_rate=0.12,
        play_family_weights={PlayFamily.HIGH_PICK_AND_ROLL: 0.45, PlayFamily.DOUBLE_DRAG: 0.20, PlayFamily.HANDOFF: 0.10, PlayFamily.ISO: 0.25},
        coverage_weights={DefensiveCoverage.DROP: 0.45, DefensiveCoverage.SWITCH: 0.25, DefensiveCoverage.ICE: 0.15, DefensiveCoverage.HEDGE: 0.15},
    )
    context = PossessionContext(
        offense_team_code="OFF",
        defense_team_code="DEF",
        clock=GameClockState(period=1, seconds_remaining_in_period=500.0, shot_clock=15.0, possession_number=10),
        score=ScoreState(offense_score=18, defense_score=17),
        offense_lineup=LineupUnit(
            team_code="OFF",
            player_ids=("creator", "wing", "shooter", "forward", "big"),
            spacing_score=12.0,
            creation_score=12.0,
            rim_pressure_score=12.0,
            rebounding_score=10.0,
            switchability_score=10.0,
            rim_protection_score=9.0,
        ),
        defense_lineup=LineupUnit(
            team_code="DEF",
            player_ids=("d1", "d2", "d3", "d4", "d5"),
            spacing_score=0.0,
            creation_score=0.0,
            rim_pressure_score=0.0,
            rebounding_score=11.0,
            switchability_score=12.0,
            rim_protection_score=11.0,
        ),
        offensive_tactics=tactics,
        defensive_tactics=tactics,
        floor_players=(
            FloorPlayerState("creator", BasketballSide.OFFENSE, CourtPoint(0.0, 22.0, CourtZone.TOP), has_ball=True),
            FloorPlayerState("big", BasketballSide.OFFENSE, CourtPoint(3.0, 20.0, CourtZone.TOP)),
            FloorPlayerState("d1", BasketballSide.DEFENSE, CourtPoint(0.0, 21.0, CourtZone.TOP)),
        ),
        defensive_assignments=(
            DefensiveAssignment(defender_id="d1", offensive_player_id="creator", matchup_strength=0.58, on_ball=True),
        ),
        player_pool=(),
        current_phase=PossessionPhase.PRIMARY_ACTION,
        play_call=PlayCall(
            family=play_family,
            primary_actor_id="creator",
            secondary_actor_id="forward" if play_family == PlayFamily.DOUBLE_DRAG else None,
            screener_id="big" if play_family in {PlayFamily.HIGH_PICK_AND_ROLL, PlayFamily.DOUBLE_DRAG, PlayFamily.HANDOFF} else None,
            target_zone=CourtZone.TOP,
        ),
        coverage=coverage,
    )
    player_pool = (
        _player("creator", "OFF", separation=15.0, burst=14.0, pullup_shooting=15.0, pass_vision=16.0, pass_accuracy=15.0, decision_making=15.0, ball_security=15.0),
        _player("wing", "OFF"),
        _player("shooter", "OFF", catch_shoot=16.0, pullup_shooting=11.0),
        _player("forward", "OFF", separation=11.0, burst=11.0, pullup_shooting=10.0),
        _player("big", "OFF", screen_setting=15.0, finishing=13.0, pass_vision=8.0, pass_accuracy=8.0, separation=8.0, burst=9.0),
        _player("d1", "DEF", containment=15.0, separation=4.0, burst=4.0, pullup_shooting=4.0),
        _player("d2", "DEF", containment=13.0),
        _player("d3", "DEF", containment=12.0),
        _player("d4", "DEF", containment=11.0),
        _player("d5", "DEF", containment=10.0),
    )
    return PossessionContext(
        offense_team_code=context.offense_team_code,
        defense_team_code=context.defense_team_code,
        clock=context.clock,
        score=context.score,
        offense_lineup=context.offense_lineup,
        defense_lineup=context.defense_lineup,
        offensive_tactics=context.offensive_tactics,
        defensive_tactics=context.defensive_tactics,
        floor_players=context.floor_players,
        defensive_assignments=context.defensive_assignments,
        player_pool=player_pool,
        current_phase=context.current_phase,
        play_call=context.play_call,
        coverage=context.coverage,
    )


def test_simulate_high_pick_and_roll_returns_valid_minimal_outcome() -> None:
    outcome = simulate_possession(_context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP), random.Random(7))
    assert outcome.events


def test_simulate_double_drag_with_hedge_returns_two_screen_flow() -> None:
    outcome = simulate_possession(_context(PlayFamily.DOUBLE_DRAG, DefensiveCoverage.HEDGE), random.Random(7))
    assert outcome.events
    screen_count = sum(1 for event in outcome.events if event.event_type == EventType.SCREEN)
    assert screen_count >= 2


def test_star_usage_bias_steepens_primary_creator_selection() -> None:
    context = _context(PlayFamily.ISO, DefensiveCoverage.SWITCH)
    low_bias = PossessionContext(
        offense_team_code=context.offense_team_code,
        defense_team_code=context.defense_team_code,
        clock=context.clock,
        score=context.score,
        offense_lineup=context.offense_lineup,
        defense_lineup=context.defense_lineup,
        offensive_tactics=replace(context.offensive_tactics, star_usage_bias=1.0),
        defensive_tactics=context.defensive_tactics,
        floor_players=context.floor_players,
        defensive_assignments=context.defensive_assignments,
        player_pool=context.player_pool,
        current_phase=context.current_phase,
        play_call=context.play_call,
        coverage=context.coverage,
    )
    high_bias = PossessionContext(
        offense_team_code=context.offense_team_code,
        defense_team_code=context.defense_team_code,
        clock=context.clock,
        score=context.score,
        offense_lineup=context.offense_lineup,
        defense_lineup=context.defense_lineup,
        offensive_tactics=replace(context.offensive_tactics, star_usage_bias=1.55),
        defensive_tactics=context.defensive_tactics,
        floor_players=context.floor_players,
        defensive_assignments=context.defensive_assignments,
        player_pool=context.player_pool,
        current_phase=context.current_phase,
        play_call=context.play_call,
        coverage=context.coverage,
    )
    low_counts = Counter(_primary_creator_id(low_bias, random.Random(seed)) for seed in range(250))
    high_counts = Counter(_primary_creator_id(high_bias, random.Random(seed)) for seed in range(250))
    assert high_counts["creator"] > low_counts["creator"]


def test_shooter_distribution_weights_influence_kickout_targeting() -> None:
    context = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    weighted_context = PossessionContext(
        offense_team_code=context.offense_team_code,
        defense_team_code=context.defense_team_code,
        clock=context.clock,
        score=context.score,
        offense_lineup=context.offense_lineup,
        defense_lineup=context.defense_lineup,
        offensive_tactics=replace(
            context.offensive_tactics,
            shooter_distribution_weights={"forward": 1.9, "shooter": 0.6, "wing": 0.8},
        ),
        defensive_tactics=context.defensive_tactics,
        floor_players=context.floor_players,
        defensive_assignments=context.defensive_assignments,
        player_pool=context.player_pool,
        current_phase=context.current_phase,
        play_call=context.play_call,
        coverage=context.coverage,
    )
    picks = Counter(
        _best_shooter(weighted_context, exclude_ids={"creator", "big"}, rng=random.Random(seed)).player_id
        for seed in range(250)
    )
    assert picks["forward"] > picks["shooter"]


def test_kickout_action_can_swing_to_second_side() -> None:
    context = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    context = replace(
        context,
        offensive_tactics=replace(
            context.offensive_tactics,
            closeout_attack_rate=0.2,
            second_side_rate=0.95,
            shooter_distribution_weights={"wing": 1.4, "forward": 1.3, "shooter": 0.7},
        ),
    )
    receiver = _get_player(context, "wing")
    shot_defender = _pick_closeout_defender(context, receiver.player_id, _get_player(context, "d1"))
    found_swing = False
    for seed in range(40):
        result = _resolve_kickout_action(
            context,
            receiver,
            shot_defender,
            _get_player(context, "d5"),
            0.72,
            random.Random(seed),
            last_passer_id="creator",
            shot_clock_remaining=12.0,
        )
        if result["result_type"] == "nested_action" and result["events"][0].notes == "swing pass":
            found_swing = True
            break
    assert found_swing


def test_simulate_iso_returns_valid_minimal_outcome() -> None:
    outcome = simulate_possession(_context(PlayFamily.ISO, DefensiveCoverage.SWITCH), random.Random(11))
    assert outcome.events
    assert outcome.events[0].event_type.name == "DRIVE"
    assert outcome.turnover or outcome.foul_committed or any(event.event_type.name == "SHOT" for event in outcome.events)


def test_simulate_possession_rejects_unsupported_play_family() -> None:
    context = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    object.__setattr__(
        context,
        "play_call",
        PlayCall(family=PlayFamily.HORNS, primary_actor_id="creator", target_zone=CourtZone.TOP),
    )
    try:
        simulate_possession(context, random.Random(1))
    except ValueError as exc:
        assert "Unsupported play family" in str(exc)
    else:
        raise AssertionError("Expected unsupported play family to raise")


def test_simulate_possession_can_select_play_and_coverage_from_tactics() -> None:
    context = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    object.__setattr__(context, "play_call", None)
    object.__setattr__(context, "coverage", None)
    outcome = simulate_possession(context, random.Random(3))
    assert outcome.events
    assert outcome.events[0].event_type.name in {"SCREEN", "DRIVE"}


def test_high_pick_and_roll_can_flow_to_pass_branch() -> None:
    found_pass = False
    found_non_primary_shot = False
    for seed in range(1, 30):
        outcome = simulate_possession(_context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP), random.Random(seed))
        if any(event.event_type.name == "PASS" for event in outcome.events):
            found_pass = True
            if outcome.shooting_player_id and outcome.shooting_player_id != "creator":
                found_non_primary_shot = True
                break
    assert found_pass
    assert found_non_primary_shot


def test_iso_can_flow_to_kickout_or_bailout_branch() -> None:
    found_pass = False
    for seed in range(1, 40):
        outcome = simulate_possession(_context(PlayFamily.ISO, DefensiveCoverage.SWITCH), random.Random(seed))
        if any(event.event_type.name == "PASS" for event in outcome.events):
            found_pass = True
            break
    assert found_pass


def test_shot_based_possessions_emit_single_shot_event() -> None:
    for seed in range(1, 20):
        outcome = simulate_possession(_context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP), random.Random(seed))
        shot_events = [event for event in outcome.events if event.event_type.name == "SHOT"]
        if shot_events:
            assert len(shot_events) == 1
            break
    else:
        raise AssertionError("Expected at least one shot possession in sample")


def test_foul_outcome_points_stay_within_two_free_throws() -> None:
    seen_attempts = set()
    for seed in range(1, 50):
        outcome = simulate_possession(_context(PlayFamily.ISO, DefensiveCoverage.SWITCH), random.Random(seed))
        if outcome.foul_committed:
            assert outcome.free_throws_attempted in {1, 2, 3}
            assert outcome.foul_type in {FoulOutcomeType.AND_ONE, FoulOutcomeType.TWO_SHOT, FoulOutcomeType.THREE_SHOT}
            assert 0 <= outcome.points_scored <= 4
            seen_attempts.add(outcome.free_throws_attempted)
    assert seen_attempts


def test_transition_entry_adds_advance_event() -> None:
    context = replace(
        _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP),
        entry_type=EntryType.TRANSITION,
        entry_source=EntrySource.LIVE_TURNOVER_BREAK,
    )
    outcome = simulate_possession(context, random.Random(5))
    assert outcome.events
    assert outcome.events[0].event_type == EventType.ADVANCE
    assert "transition_entry" in outcome.events[0].notes
    assert "live_turnover_break" in outcome.events[0].notes


def test_oreb_entry_adds_reentry_event() -> None:
    base = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    context = replace(
        base,
        entry_type=EntryType.OREB,
        entry_source=EntrySource.OREB_GATHER,
        clock=replace(base.clock, shot_clock=20.0),
    )
    outcome = simulate_possession(context, random.Random(8))
    assert outcome.events
    assert outcome.events[0].event_type == EventType.ADVANCE
    assert "oreb_reentry_14s" in outcome.events[0].notes


def test_second_side_can_attack_closeout_or_reset_loop() -> None:
    base = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    attack_context = replace(
        base,
        offensive_tactics=replace(
            base.offensive_tactics,
            closeout_attack_rate=0.85,
            second_side_rate=0.95,
        ),
    )
    reset_context = replace(
        base,
        offensive_tactics=replace(
            base.offensive_tactics,
            closeout_attack_rate=0.05,
            second_side_rate=0.98,
        ),
    )
    observed_attack = False
    observed_reset = False
    for seed in range(1, 40):
        attack_outcome = simulate_possession(attack_context, random.Random(seed))
        attack_notes = " ".join(event.notes for event in attack_outcome.events).lower()
        if "attack_closeout" in attack_notes:
            observed_attack = True
        reset_outcome = simulate_possession(reset_context, random.Random(seed))
        reset_notes = " ".join(event.notes for event in reset_outcome.events).lower()
        if "second_side reset" in reset_notes:
            observed_reset = True
        if observed_attack and observed_reset:
            break
    assert observed_attack
    assert observed_reset


def test_second_side_can_find_cutter() -> None:
    base = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    context = replace(
        base,
        offensive_tactics=replace(
            base.offensive_tactics,
            closeout_attack_rate=0.05,
            second_side_rate=0.98,
            shooter_distribution_weights={"shooter": 0.6, "wing": 0.8, "forward": 1.2},
        ),
    )
    seen_cut = False
    for seed in range(1, 80):
        outcome = simulate_possession(context, random.Random(seed))
        notes = " ".join(event.notes for event in outcome.events).lower()
        if "cut_find" in notes:
            seen_cut = True
            break
    assert seen_cut


def test_transition_source_changes_profile() -> None:
    base = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    turnover_context = replace(base, entry_type=EntryType.TRANSITION, entry_source=EntrySource.LIVE_TURNOVER_BREAK)
    rebound_context = replace(base, entry_type=EntryType.TRANSITION, entry_source=EntrySource.DEFENSIVE_REBOUND_PUSH)
    turnover_drive_count = 0
    rebound_trail_count = 0
    for seed in range(120):
        turnover_outcome = simulate_possession(turnover_context, random.Random(seed))
        rebound_outcome = simulate_possession(rebound_context, random.Random(seed))
        turnover_notes = " ".join(event.notes or "" for event in turnover_outcome.events)
        rebound_notes = " ".join(event.notes or "" for event in rebound_outcome.events)
        if any(event.event_type == EventType.DRIVE for event in turnover_outcome.events) and "coverage=" not in turnover_notes:
            turnover_drive_count += 1
        if "transition_pitch_ahead" in rebound_notes:
            rebound_trail_count += 1
    assert turnover_drive_count > 0
    assert rebound_trail_count > 0


def test_oreb_branch_can_tip_in_tip_out_and_gather_reset() -> None:
    base = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    context = replace(
        base,
        entry_type=EntryType.OREB,
        entry_source=EntrySource.OREB_GATHER,
        clock=replace(base.clock, shot_clock=20.0),
    )
    seen = set()
    for seed in range(180):
        outcome = simulate_possession(context, random.Random(seed))
        notes = " ".join(event.notes or "" for event in outcome.events)
        for tag in ("oreb_tip_in", "oreb_tip_out", "oreb_gather_reset"):
            if tag in notes:
                seen.add(tag)
        if len(seen) == 3:
            break
    assert seen == {"oreb_tip_in", "oreb_tip_out", "oreb_gather_reset"}


def test_second_side_can_re_screen_into_side_pnr() -> None:
    base = _context(PlayFamily.HIGH_PICK_AND_ROLL, DefensiveCoverage.DROP)
    context = replace(
        base,
        offensive_tactics=replace(
            base.offensive_tactics,
            closeout_attack_rate=0.05,
            second_side_rate=0.98,
            pick_and_roll_rate=0.98,
        ),
    )
    seen_re_screen = False
    for seed in range(1, 220):
        outcome = simulate_possession(context, random.Random(seed))
        notes = " ".join(event.notes or "" for event in outcome.events).lower()
        if "second_side re_screen" in notes:
            seen_re_screen = True
            break
    assert seen_re_screen
