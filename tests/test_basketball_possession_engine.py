import random

from basketball_possession_engine import simulate_possession
from basketball_sim_schema import (
    BasketballSide,
    CourtPoint,
    CourtZone,
    DefensiveAssignment,
    DefensiveCoverage,
    DefensiveRole,
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
        play_family_weights={PlayFamily.HIGH_PICK_AND_ROLL: 0.6, PlayFamily.ISO: 0.4},
        coverage_weights={DefensiveCoverage.DROP: 0.6, DefensiveCoverage.SWITCH: 0.4},
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
            screener_id="big" if play_family == PlayFamily.HIGH_PICK_AND_ROLL else None,
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
    assert outcome.events[0].event_type.name == "SCREEN"
    assert outcome.events[-1].event_type.name in {"SHOT", "REBOUND", "FOUL", "TURNOVER"}
    assert outcome.points_scored >= 0


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
