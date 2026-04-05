import random

from basketball_court import build_shot_context, sample_shot_point
from basketball_sim_schema import CourtPoint, CourtZone, ShotType


def test_corner_three_points_land_in_corner_geometry() -> None:
    point = sample_shot_point(ShotType.CORNER_THREE, rng=random.Random(7), preferred_side="left")
    assert point.zone.value == "left_corner"
    assert point.x < 0
    assert point.y <= 11.0


def test_shot_quality_respects_contest_distance() -> None:
    open_shot = build_shot_context(
        ShotType.ABOVE_BREAK_THREE,
        shot_style="catch",
        shooter_skill=15.0,
        defender_distance_feet=7.0,
        advantage=0.62,
        rng=random.Random(3),
    )
    tight_shot = build_shot_context(
        ShotType.ABOVE_BREAK_THREE,
        shot_style="catch",
        shooter_skill=15.0,
        defender_distance_feet=1.5,
        advantage=0.62,
        rng=random.Random(3),
    )
    assert open_shot.make_probability > tight_shot.make_probability


def test_rim_attempts_grade_higher_than_midrange_for_same_shooter() -> None:
    rim = build_shot_context(
        ShotType.RIM,
        shot_style="pullup",
        shooter_skill=14.0,
        defender_distance_feet=3.5,
        advantage=0.58,
        rng=random.Random(11),
    )
    middy = build_shot_context(
        ShotType.MIDRANGE,
        shot_style="pullup",
        shooter_skill=14.0,
        defender_distance_feet=3.5,
        advantage=0.58,
        rng=random.Random(11),
    )
    assert rim.make_probability > middy.make_probability


def test_real_defender_position_tightens_shot_context() -> None:
    open_context = build_shot_context(
        ShotType.ABOVE_BREAK_THREE,
        shot_style="catch",
        shooter_skill=14.0,
        defender_distance_feet=6.8,
        advantage=0.60,
        rng=random.Random(17),
        preferred_side="left",
        origin=CourtPoint(-18.0, 24.0, CourtZone.LEFT_WING),
        defender_point=CourtPoint(-25.0, 30.0, CourtZone.LEFT_WING),
        template="flare_three",
    )
    tight_context = build_shot_context(
        ShotType.ABOVE_BREAK_THREE,
        shot_style="catch",
        shooter_skill=14.0,
        defender_distance_feet=6.8,
        advantage=0.60,
        rng=random.Random(17),
        preferred_side="left",
        origin=CourtPoint(-18.0, 24.0, CourtZone.LEFT_WING),
        defender_point=CourtPoint(-15.2, 24.7, CourtZone.LEFT_WING),
        template="flare_three",
    )
    assert tight_context.defender_distance_feet < open_context.defender_distance_feet
    assert tight_context.make_probability < open_context.make_probability


def test_action_templates_land_in_distinct_shot_areas() -> None:
    handoff = sample_shot_point(ShotType.MIDRANGE, rng=random.Random(7), preferred_side="left", template="handoff_pullup")
    side_pnr = sample_shot_point(ShotType.MIDRANGE, rng=random.Random(7), preferred_side="left", template="side_pnr_pullup")
    flare = sample_shot_point(ShotType.ABOVE_BREAK_THREE, rng=random.Random(7), preferred_side="left", template="flare_three")
    corner = sample_shot_point(ShotType.CORNER_THREE, rng=random.Random(7), preferred_side="left", template="corner_catch")
    assert handoff.x > side_pnr.x
    assert side_pnr.y >= handoff.y
    assert flare.y > side_pnr.y
    assert abs(corner.x) > abs(flare.x)
    assert corner.y < 10.0
