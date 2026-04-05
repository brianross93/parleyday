from __future__ import annotations

import math
import random
from dataclasses import dataclass

from basketball_sim_schema import CourtPoint, CourtZone, ShotType

COURT_LENGTH_FEET = 94.0
COURT_WIDTH_FEET = 50.0
HALF_COURT_LENGTH_FEET = 47.0
BASKET_CENTER_X = 0.0
BASKET_CENTER_Y = 5.25
THREE_POINT_RADIUS_FEET = 23.75
CORNER_THREE_X_FEET = 22.0


@dataclass(frozen=True)
class ShotContext:
    point: CourtPoint
    distance_feet: float
    zone_label: str
    defender_distance_feet: float
    make_probability: float


ZONE_BASE_FG = {
    ShotType.RIM: 0.66,
    ShotType.PAINT: 0.49,
    ShotType.MIDRANGE: 0.41,
    ShotType.CORNER_THREE: 0.39,
    ShotType.ABOVE_BREAK_THREE: 0.35,
}

ZONE_CENTER_DISTANCE = {
    ShotType.RIM: 1.5,
    ShotType.PAINT: 9.0,
    ShotType.MIDRANGE: 17.0,
    ShotType.CORNER_THREE: 22.0,
    ShotType.ABOVE_BREAK_THREE: 24.0,
}

PULLUP_PENALTY = {
    ShotType.RIM: 0.00,
    ShotType.PAINT: 0.02,
    ShotType.MIDRANGE: 0.04,
    ShotType.CORNER_THREE: 0.04,
    ShotType.ABOVE_BREAK_THREE: 0.03,
}


def build_shot_context(
    shot_type: ShotType,
    *,
    shot_style: str,
    shooter_skill: float,
    defender_distance_feet: float,
    advantage: float,
    rng: random.Random,
    preferred_side: str = "center",
    origin: CourtPoint | None = None,
    defender_point: CourtPoint | None = None,
    template: str | None = None,
) -> ShotContext:
    point = sample_shot_point(shot_type, rng=rng, preferred_side=preferred_side, template=template)
    template_anchor = _template_anchor(template, preferred_side)
    if template_anchor is not None:
        anchor_blend = 0.58 if origin is not None else 0.76
        point = court_point(
            (point.x * (1.0 - anchor_blend)) + (template_anchor.x * anchor_blend),
            (point.y * (1.0 - anchor_blend)) + (template_anchor.y * anchor_blend),
        )
    if origin is not None:
        blend = 0.62 if shot_style == "pullup" else 0.78
        point = court_point(
            (point.x * (1.0 - blend)) + (origin.x * blend),
            (point.y * (1.0 - blend)) + (origin.y * blend),
        )
    if template_anchor is not None:
        distance = distance_to_basket(point)
        if template in {"flare_three", "slot_three", "corner_catch"} and distance < 22.2:
            point = court_point(
                (point.x * 0.42) + (template_anchor.x * 0.58),
                (point.y * 0.42) + (template_anchor.y * 0.58),
            )
        elif template in {"handoff_pullup", "side_pnr_pullup"} and distance > 22.0:
            point = court_point(
                (point.x * 0.50) + (template_anchor.x * 0.50),
                (point.y * 0.50) + (template_anchor.y * 0.50),
            )
    if defender_point is not None:
        actual_defender_distance = math.hypot(point.x - defender_point.x, point.y - defender_point.y)
        # Use the real floor position as the primary contest signal while still
        # letting defender skill slightly tighten or loosen the contest.
        defender_distance_feet = clamp((actual_defender_distance * 0.82) + (defender_distance_feet * 0.18), 0.8, 9.5)
    distance_feet = distance_to_basket(point)
    base_fg = ZONE_BASE_FG[shot_type]
    center_distance = ZONE_CENTER_DISTANCE[shot_type]
    distance_mod = -(abs(distance_feet - center_distance) * 0.012)
    skill_mod = ((shooter_skill - 10.0) / 10.0) * 0.14
    contest_mod = (defender_distance_feet - 4.0) * 0.022
    advantage_mod = (advantage - 0.5) * 0.14
    style_penalty = -PULLUP_PENALTY[shot_type] if shot_style == "pullup" else 0.0
    make_probability = clamp(base_fg + distance_mod + skill_mod + contest_mod + advantage_mod + style_penalty, 0.16, 0.82)
    return ShotContext(
        point=point,
        distance_feet=distance_feet,
        zone_label=basic_zone_label(point),
        defender_distance_feet=defender_distance_feet,
        make_probability=make_probability,
    )


def sample_shot_point(shot_type: ShotType, *, rng: random.Random, preferred_side: str = "center", template: str | None = None) -> CourtPoint:
    side = preferred_side
    if side not in {"left", "right", "center"}:
        side = "center"
    if side == "center":
        draw = rng.random()
        if draw < 0.33:
            side = "left"
        elif draw < 0.66:
            side = "right"
    sign = -1.0 if side == "left" else 1.0
    if template == "handoff_pullup":
        x = sign * rng.uniform(4.5, 10.5)
        y = rng.uniform(16.5, 20.5)
        return court_point(x, y)
    if template == "side_pnr_pullup":
        x = sign * rng.uniform(10.0, 17.0)
        y = rng.uniform(17.5, 23.5)
        return court_point(x, y)
    if template == "flare_three":
        x = sign * rng.uniform(16.0, 22.0)
        y = rng.uniform(21.5, 28.5)
        return court_point(x, y)
    if template == "corner_catch":
        x = sign * rng.uniform(21.6, 22.0)
        y = rng.uniform(4.5, 8.5)
        return court_point(x, y)
    if template == "slot_three":
        x = sign * rng.uniform(4.5, 11.5)
        y = rng.uniform(23.5, 28.5)
        return court_point(x, y)
    if shot_type == ShotType.RIM:
        x = rng.uniform(-2.5, 2.5)
        y = rng.uniform(3.8, 7.0)
        return court_point(x, y)
    if shot_type == ShotType.PAINT:
        x = rng.uniform(-7.5, 7.5)
        y = rng.uniform(8.0, 16.0)
        return court_point(x, y)
    if shot_type == ShotType.MIDRANGE:
        x = sign * rng.uniform(5.0, 18.0) if side != "center" else rng.uniform(-10.0, 10.0)
        y = rng.uniform(15.0, 24.0)
        return court_point(x, y)
    if shot_type == ShotType.CORNER_THREE:
        x = sign * rng.uniform(21.5, 22.0)
        y = rng.uniform(3.5, 10.0)
        return court_point(x, y)
    x = sign * rng.uniform(0.0, 20.0) if side != "center" else rng.uniform(-18.0, 18.0)
    y = rng.uniform(22.0, 29.0)
    return court_point(x, y)


def court_point(x: float, y: float) -> CourtPoint:
    return CourtPoint(x=float(x), y=float(y), zone=zone_for_point(float(x), float(y)))


def distance_to_basket(point: CourtPoint) -> float:
    return math.hypot(point.x - BASKET_CENTER_X, point.y - BASKET_CENTER_Y)


def basic_zone_label(point: CourtPoint) -> str:
    if point.zone == CourtZone.RIM:
        return "restricted_area"
    if point.zone == CourtZone.PAINT:
        return "paint"
    if point.zone in {CourtZone.LEFT_CORNER, CourtZone.RIGHT_CORNER}:
        return "corner_three"
    if point.zone in {CourtZone.TOP, CourtZone.LEFT_WING, CourtZone.RIGHT_WING} and distance_to_basket(point) >= 22.5:
        return "above_break_three"
    if point.zone == CourtZone.BACKCOURT:
        return "backcourt"
    return "midrange"


def zone_for_point(x: float, y: float) -> CourtZone:
    if y >= HALF_COURT_LENGTH_FEET:
        return CourtZone.BACKCOURT
    distance = math.hypot(x - BASKET_CENTER_X, y - BASKET_CENTER_Y)
    if distance <= 4.0:
        return CourtZone.RIM
    if abs(x) <= 8.0 and y <= 19.0:
        return CourtZone.PAINT
    if y <= 11.0 and abs(x) >= 19.5:
        return CourtZone.LEFT_CORNER if x < 0 else CourtZone.RIGHT_CORNER
    if abs(x) <= 3.0:
        return CourtZone.TOP
    if x < 0:
        if y < 18.0:
            return CourtZone.ELBOW_LEFT
        return CourtZone.LEFT_WING
    if y < 18.0:
        return CourtZone.ELBOW_RIGHT
    return CourtZone.RIGHT_WING


def contest_distance_from_traits(
    *,
    coverage_gap: float,
    defender_closeout: float,
    defender_reach: float,
    off_screen: bool = False,
    catch_and_shoot: bool = False,
) -> float:
    base = 4.4 + (coverage_gap * 1.8)
    base += (1.2 if off_screen else 0.0)
    base += (1.0 if catch_and_shoot else 0.0)
    base -= ((defender_closeout - 10.0) / 10.0) * 1.5
    base -= ((defender_reach - 10.0) / 10.0) * 0.6
    return clamp(base, 0.8, 8.0)


def _template_anchor(template: str | None, preferred_side: str) -> CourtPoint | None:
    side = preferred_side if preferred_side in {"left", "right"} else "right"
    sign = -1.0 if side == "left" else 1.0
    if template == "handoff_pullup":
        return court_point(sign * 8.0, 18.5)
    if template == "side_pnr_pullup":
        return court_point(sign * 13.0, 20.5)
    if template == "flare_three":
        return court_point(sign * 18.5, 24.5)
    if template == "corner_catch":
        return court_point(sign * 21.9, 6.5)
    if template == "slot_three":
        return court_point(sign * 8.5, 25.5)
    return None


def shot_type_from_point(point: CourtPoint) -> ShotType:
    if point.zone == CourtZone.RIM:
        return ShotType.RIM
    if point.zone == CourtZone.PAINT:
        return ShotType.PAINT
    if point.zone in {CourtZone.LEFT_CORNER, CourtZone.RIGHT_CORNER}:
        return ShotType.CORNER_THREE
    if basic_zone_label(point) == "above_break_three":
        return ShotType.ABOVE_BREAK_THREE
    return ShotType.MIDRANGE


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
