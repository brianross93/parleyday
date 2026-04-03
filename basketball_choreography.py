from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MotionPoint:
    x: float
    y: float


@dataclass(frozen=True)
class ActorTrack:
    player_id: str
    team_code: str
    start: MotionPoint
    end: MotionPoint
    easing: str


@dataclass(frozen=True)
class BallTrack:
    owner_player_id: str | None
    start: MotionPoint
    end: MotionPoint
    easing: str
    mode: str
    arc_height: float
    dribble_count: int
    control: MotionPoint | None


@dataclass(frozen=True)
class CameraTrack:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    start_scale: float
    end_scale: float
    easing: str


@dataclass(frozen=True)
class ChoreographySegment:
    beat_index: int
    event_type: str
    label: str
    commentary: str
    start_ms: int
    end_ms: int
    duration_ms: int
    offense_team_code: str | None
    actor_tracks: tuple[ActorTrack, ...]
    ball_track: BallTrack
    camera_track: CameraTrack


@dataclass(frozen=True)
class MatchChoreography:
    total_duration_ms: int
    segments: tuple[ChoreographySegment, ...]


def build_match_choreography(
    *,
    players: list[dict[str, Any]],
    initial_positions: list[dict[str, Any]],
    beats: list[dict[str, Any]],
    events: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    assignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    player_lookup = {player["player_id"]: player for player in players}
    team_players = {
        home_team: [player["player_id"] for player in players if player["team_code"] == home_team],
        away_team: [player["player_id"] for player in players if player["team_code"] == away_team],
    }
    assignment_map = {
        item["defender_id"]: item
        for item in (assignments or [])
        if item.get("defender_id")
    }
    current_positions = {
        item["player_id"]: MotionPoint(float(item["x"]), float(item["y"]))
        for item in initial_positions
    }
    current_ball = _initial_ball_point(initial_positions)
    current_camera = _camera_target_from_point(current_ball, home_team, "possession_start")
    event_cursor = -1
    elapsed_ms = 0
    segments: list[ChoreographySegment] = []

    for beat_index, beat in enumerate(beats):
        working_positions = dict(current_positions)
        reset_positions = beat.get("reset_positions") or []
        if reset_positions:
            working_positions = _blend_reset_positions(
                current_positions=current_positions,
                reset_positions=reset_positions,
                beat=beat,
                player_lookup=player_lookup,
            )

        offense_team_code = beat.get("offense_team_code") or home_team
        defense_team_code = away_team if offense_team_code == home_team else home_team
        ball_owner = beat.get("ball_player_id")
        focus_ids = {beat.get("focus_player_id"), ball_owner}

        end = beat.get("event_apply_index", beat_index)
        if end is None:
            end = beat_index
        for idx in range(event_cursor + 1, end + 1):
            if idx < 0 or idx >= len(events):
                continue
            event = events[idx]
            working_positions = _apply_event_formation(
                current_positions=working_positions,
                event=event,
                beat=beat,
                player_lookup=player_lookup,
                team_players=team_players,
                offense_team_code=offense_team_code,
                defense_team_code=defense_team_code,
                assignment_map=assignment_map,
            )
            event_cursor = idx

        resolved_positions = _resolve_collisions(
            raw_positions=working_positions,
            player_ids=[player["player_id"] for player in players],
            focus_ids=focus_ids,
        )
        ball_end = _resolve_ball_end(ball_owner, beat, resolved_positions, current_ball)
        duration_ms = max(260, int(beat.get("duration_ms") or 800))

        actor_tracks = []
        for player in players:
            pid = player["player_id"]
            start_point = current_positions.get(pid, resolved_positions.get(pid, MotionPoint(0.0, 22.0)))
            end_point = resolved_positions.get(pid, start_point)
            actor_tracks.append(
                ActorTrack(
                    player_id=pid,
                    team_code=player["team_code"],
                    start=start_point,
                    end=end_point,
                    easing=_actor_easing(beat, player["player_id"]),
                )
            )

        camera_end = _camera_target_from_point(ball_end, offense_team_code, str(beat.get("event_type") or ""))
        segment = ChoreographySegment(
            beat_index=beat_index,
            event_type=str(beat.get("event_type") or "unknown"),
            label=str(beat.get("label") or ""),
            commentary=str(beat.get("commentary") or ""),
            start_ms=elapsed_ms,
            end_ms=elapsed_ms + duration_ms,
            duration_ms=duration_ms,
            offense_team_code=offense_team_code,
            actor_tracks=tuple(actor_tracks),
            ball_track=BallTrack(
                owner_player_id=ball_owner,
                start=current_ball,
                end=ball_end,
                easing=_ball_easing(beat),
                mode=_ball_mode(beat),
                arc_height=_ball_arc_height(beat, current_ball, ball_end),
                dribble_count=_ball_dribble_count(beat, current_ball, ball_end, duration_ms),
                control=_ball_control_point(beat, current_ball, ball_end),
            ),
            camera_track=CameraTrack(
                start_x=current_camera[0],
                start_y=current_camera[1],
                end_x=camera_end[0],
                end_y=camera_end[1],
                start_scale=current_camera[2],
                end_scale=camera_end[2],
                easing=_camera_easing(beat),
            ),
        )
        segments.append(segment)
        current_positions = resolved_positions
        current_ball = ball_end
        current_camera = camera_end
        elapsed_ms += duration_ms

    return asdict(MatchChoreography(total_duration_ms=elapsed_ms, segments=tuple(segments)))


def _apply_event_formation(
    *,
    current_positions: dict[str, MotionPoint],
    event: dict[str, Any],
    beat: dict[str, Any],
    player_lookup: dict[str, dict[str, Any]],
    team_players: dict[str, list[str]],
    offense_team_code: str,
    defense_team_code: str,
    assignment_map: dict[str, dict[str, Any]],
) -> dict[str, MotionPoint]:
    positions = dict(current_positions)
    event_type = str(event.get("event_type") or beat.get("event_type") or "")
    location = _event_location_point(event) or _beat_location_point(beat) or MotionPoint(0.0, 22.0)
    actor_id = event.get("actor_id") or beat.get("focus_player_id")
    receiver_id = event.get("receiver_id")
    defender_id = event.get("defender_id")
    notes = str(event.get("notes") or "")
    ball_side = _ball_side(location.x)
    offense_ids = team_players.get(offense_team_code, [])
    defense_ids = team_players.get(defense_team_code, [])

    offensive_shape = _build_offensive_shape(
        offense_ids=offense_ids,
        player_lookup=player_lookup,
        actor_id=actor_id,
        receiver_id=receiver_id,
        location=location,
        event_type=event_type,
        notes=notes,
        ball_side=ball_side,
    )
    defensive_shape = _build_defensive_shape(
        defense_ids=defense_ids,
        player_lookup=player_lookup,
        location=location,
        event_type=event_type,
        notes=notes,
        ball_side=ball_side,
        assignment_map=assignment_map,
        actor_id=actor_id,
        defender_id=defender_id,
        receiver_id=receiver_id,
    )

    positions.update(offensive_shape)
    positions.update(defensive_shape)
    return positions


def _build_offensive_shape(
    *,
    offense_ids: list[str],
    player_lookup: dict[str, dict[str, Any]],
    actor_id: str | None,
    receiver_id: str | None,
    location: MotionPoint,
    event_type: str,
    notes: str,
    ball_side: str,
) -> dict[str, MotionPoint]:
    shape = {
        pid: _role_anchor(player_lookup.get(pid, {}), ball_side)
        for pid in offense_ids
    }
    if actor_id in shape:
        shape[actor_id] = _clamp_point(location)

    if event_type == "screen" and actor_id in shape and receiver_id in shape:
        screen_side = -1 if ball_side == "left" else 1
        handler_point = MotionPoint(location.x + (screen_side * 2.4), location.y + 1.8)
        shape[receiver_id] = _clamp_point(handler_point)
        if "double_drag_2" in notes:
            second_screener = next((pid for pid in offense_ids if pid not in {actor_id, receiver_id} and _role_name(player_lookup.get(pid, {})) in {"roll_big", "pop_big", "glue"}), None)
            if second_screener:
                shape[second_screener] = MotionPoint(location.x + (screen_side * 5.2), location.y + 2.5)
        for pid in offense_ids:
            if pid not in {actor_id, receiver_id}:
                shape[pid] = _off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="screen")

    elif event_type == "handoff" and actor_id in shape and receiver_id in shape:
        handoff_side = -1 if location.x < 0 else 1
        shape[actor_id] = _clamp_point(MotionPoint(location.x - (handoff_side * 0.9), location.y))
        shape[receiver_id] = _clamp_point(MotionPoint(location.x + (handoff_side * 1.3), location.y + 0.8))
        for pid in offense_ids:
            if pid not in {actor_id, receiver_id}:
                shape[pid] = _off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="handoff")

    elif event_type == "drive" and actor_id in shape:
        for pid in offense_ids:
            if pid == actor_id:
                continue
            role = _role_name(player_lookup.get(pid, {}))
            shape[pid] = _drive_spacing(role, ball_side, location)

    elif event_type == "advance" and actor_id in shape:
        for pid in offense_ids:
            role = _role_name(player_lookup.get(pid, {}))
            if pid == actor_id:
                shape[pid] = _clamp_point(location)
            else:
                shape[pid] = _transition_spacing(role, ball_side, location, notes)

    elif event_type == "pass":
        target_id = receiver_id or actor_id
        if target_id in shape:
            shape[target_id] = _clamp_point(location)
        if actor_id in shape and receiver_id:
            shape[actor_id] = _passer_relocate(player_lookup.get(actor_id, {}), ball_side, location)
        if "kickout" in notes:
            for pid in offense_ids:
                if pid not in {actor_id, receiver_id}:
                    shape[pid] = _off_ball_relocation(player_lookup.get(pid, {}), _ball_side(location.x), phase="kickout")
        elif "pocket pass" in notes:
            for pid in offense_ids:
                if pid not in {actor_id, receiver_id}:
                    shape[pid] = _off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="roller_pass")
        for pid in offense_ids:
            if pid not in {actor_id, receiver_id}:
                shape[pid] = _off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="pass")

    elif event_type == "shot" and actor_id in shape:
        for pid in offense_ids:
            role = _role_name(player_lookup.get(pid, {}))
            if pid == actor_id:
                shape[pid] = _clamp_point(location)
            elif "big" in role or role in {"roll_big", "post_hub"}:
                shape[pid] = MotionPoint(-4.0 if pid.endswith("pf") else 4.0, 6.5)
            else:
                shape[pid] = _off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="shot")

    elif event_type == "rebound":
        for pid in offense_ids:
            role = _role_name(player_lookup.get(pid, {}))
            if pid == actor_id:
                shape[pid] = _clamp_point(location)
            elif "big" in role or role in {"roll_big", "post_hub"}:
                shape[pid] = MotionPoint(-5.5 if pid.endswith("pf") else 5.5, 8.0)
            else:
                shape[pid] = _off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="rebound")

    return shape


def _build_defensive_shape(
    *,
    defense_ids: list[str],
    player_lookup: dict[str, dict[str, Any]],
    location: MotionPoint,
    event_type: str,
    notes: str,
    ball_side: str,
    assignment_map: dict[str, dict[str, Any]],
    actor_id: str | None,
    defender_id: str | None,
    receiver_id: str | None,
) -> dict[str, MotionPoint]:
    shape = {
        pid: _defensive_anchor(player_lookup.get(pid, {}), ball_side)
        for pid in defense_ids
    }
    primary_defender = defender_id if defender_id in shape else defense_ids[0] if defense_ids else None
    matchup_strengths = {
        pid: float(assignment_map.get(pid, {}).get("matchup_strength", 0.5))
        for pid in defense_ids
    }
    help_priorities = {
        pid: float(assignment_map.get(pid, {}).get("help_priority", 0.5))
        for pid in defense_ids
    }
    if primary_defender in shape:
        if event_type in {"drive", "shot"}:
            shape[primary_defender] = _clamp_point(MotionPoint(location.x - 1.2, location.y + 2.0))
        else:
            shape[primary_defender] = _clamp_point(MotionPoint(location.x - 1.8, location.y + 1.1))

    if event_type in {"screen", "handoff"}:
        help_targets = sorted(
            [pid for pid in defense_ids if pid != primary_defender],
            key=lambda pid: help_priorities.get(pid, 0.5),
            reverse=True,
        )[:2]
        for idx, pid in enumerate(help_targets):
            shade = 4.0 + (matchup_strengths.get(pid, 0.5) * 2.5)
            shape[pid] = MotionPoint(((-1 if idx == 0 else 1) * shade), 15.0)
    elif event_type == "drive":
        helpers = sorted(
            [pid for pid in defense_ids if pid != primary_defender],
            key=lambda pid: help_priorities.get(pid, 0.5),
            reverse=True,
        )
        for idx, pid in enumerate(helpers):
            role = _def_role_name(player_lookup.get(pid, {}))
            if role in {"rim_protector", "helper", "rebounder"}:
                shade = 3.5 + (help_priorities.get(pid, 0.5) * 2.0)
                shape[pid] = MotionPoint((-shade if idx % 2 == 0 else shade), 10.5)
            else:
                tether = 10.0 + (matchup_strengths.get(pid, 0.5) * 4.0)
                shape[pid] = MotionPoint(-tether if ball_side == "right" else tether, 18.0 + (idx * 2.0))
    elif event_type == "advance":
        for idx, pid in enumerate(defense_ids):
            role = _def_role_name(player_lookup.get(pid, {}))
            if pid == primary_defender:
                shape[pid] = _clamp_point(MotionPoint(location.x - 0.8, location.y + 2.8))
            elif role in {"rim_protector", "helper", "rebounder"}:
                shape[pid] = MotionPoint(-4.0 if idx % 2 == 0 else 4.0, max(9.0, location.y - 8.0))
            else:
                lane_x = 10.0 if idx % 2 == 0 else -10.0
                shape[pid] = MotionPoint(lane_x, max(14.0, location.y - 2.0))
    elif event_type == "pass":
        rotating = sorted(
            [pid for pid in defense_ids if pid != primary_defender],
            key=lambda pid: matchup_strengths.get(pid, 0.5),
            reverse=True,
        )
        for idx, pid in enumerate(rotating):
            role = _def_role_name(player_lookup.get(pid, {}))
            if "kickout" in notes and receiver_id:
                assigned = assignment_map.get(pid, {})
                if assigned.get("offensive_player_id") == receiver_id:
                    shape[pid] = MotionPoint(location.x - 1.0, location.y + 1.4)
                    continue
            if role in {"wing_stopper", "point_of_attack"}:
                stay_home_bias = 1.0 + matchup_strengths.get(pid, 0.5)
                shape[pid] = MotionPoint(location.x - (1.2 * stay_home_bias), location.y + 1.4 + (idx * 0.8))
            else:
                shape[pid] = _defensive_anchor(player_lookup.get(pid, {}), ball_side)
        if "kickout" in notes:
            for pid in rotating:
                if pid not in shape:
                    continue
                if help_priorities.get(pid, 0.5) > 0.55:
                    recover_x = -11.0 if ball_side == "right" else 11.0
                    shape[pid] = MotionPoint(recover_x, 14.0 + (rotating.index(pid) * 1.2))
    elif event_type in {"shot", "rebound"}:
        for idx, pid in enumerate(defense_ids):
            if pid == primary_defender:
                continue
            role = _def_role_name(player_lookup.get(pid, {}))
            if role in {"rim_protector", "rebounder", "switch_big"}:
                shape[pid] = MotionPoint(-3.5 if idx % 2 == 0 else 3.5, 8.5)
            else:
                stay_attached = 11.0 + (matchup_strengths.get(pid, 0.5) * 5.0)
                shape[pid] = MotionPoint(-stay_attached if ball_side == "right" else stay_attached, 17.0 + (idx * 1.4))

    if "ice" in notes:
        for pid in defense_ids:
            if pid == primary_defender:
                continue
            role = _def_role_name(player_lookup.get(pid, {}))
            if role == "rim_protector":
                shape[pid] = MotionPoint(6.0 if ball_side == "right" else -6.0, 12.5)
    if "hedge" in notes:
        hedger = next(
            (pid for pid in defense_ids if _def_role_name(player_lookup.get(pid, {})) in {"switch_big", "rim_protector", "helper"}),
            None,
        )
        if hedger and hedger != primary_defender:
            shape[hedger] = MotionPoint(location.x - 0.6, location.y + 1.8)

    return shape


def _role_anchor(player: dict[str, Any], ball_side: str) -> MotionPoint:
    role = _role_name(player)
    weak_x = -19.0 if ball_side == "right" else 19.0
    strong_x = 19.0 if ball_side == "right" else -19.0
    if role == "primary_creator":
        return MotionPoint(0.0, 24.0)
    if role == "secondary_creator":
        return MotionPoint(12.0 if ball_side == "right" else -12.0, 20.0)
    if role in {"movement_shooter", "spacer"}:
        return MotionPoint(weak_x, 8.5 if role == "spacer" else 14.5)
    if role in {"roll_big", "pop_big"}:
        return MotionPoint(4.5 if ball_side == "right" else -4.5, 22.5)
    if role == "post_hub":
        return MotionPoint(strong_x * 0.55, 12.0)
    if role == "slasher":
        return MotionPoint(strong_x * 0.7, 11.0)
    return MotionPoint(weak_x * 0.7, 18.0)


def _defensive_anchor(player: dict[str, Any], ball_side: str) -> MotionPoint:
    role = _def_role_name(player)
    if role == "point_of_attack":
        return MotionPoint(0.0, 23.5)
    if role == "wing_stopper":
        return MotionPoint(-12.0 if ball_side == "left" else 12.0, 17.0)
    if role == "helper":
        return MotionPoint(0.0, 14.5)
    if role == "rim_protector":
        return MotionPoint(0.0, 8.0)
    if role == "rebounder":
        return MotionPoint(-6.0 if ball_side == "left" else 6.0, 10.0)
    return MotionPoint(7.0 if ball_side == "right" else -7.0, 14.0)


def _off_ball_relocation(player: dict[str, Any], ball_side: str, *, phase: str) -> MotionPoint:
    role = _role_name(player)
    weak_x = -20.0 if ball_side == "right" else 20.0
    strong_x = 15.0 if ball_side == "right" else -15.0
    if phase == "screen":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, 9.0 if role == "spacer" else 14.0)
        if role in {"roll_big", "pop_big"}:
            return MotionPoint(strong_x * 0.45, 19.0)
    if phase == "handoff":
        if role in {"movement_shooter", "secondary_creator"}:
            return MotionPoint(strong_x, 18.5)
        if role in {"slasher", "glue"}:
            return MotionPoint(weak_x * 0.6, 11.0)
    if phase == "pass":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, 7.5 if role == "spacer" else 16.0)
        if role == "secondary_creator":
            return MotionPoint(strong_x, 22.0)
    if phase == "kickout":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, 6.0 if role == "spacer" else 14.0)
        if role == "secondary_creator":
            return MotionPoint(strong_x * 0.7, 21.0)
    if phase == "roller_pass":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, 8.0 if role == "spacer" else 15.0)
        if role in {"roll_big", "pop_big"}:
            return MotionPoint(strong_x * 0.35, 9.0)
    if phase in {"shot", "rebound"}:
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, 8.0 if role == "spacer" else 15.5)
        if role in {"roll_big", "post_hub"}:
            return MotionPoint(strong_x * 0.35, 7.5)
    return _role_anchor(player, ball_side)


def _drive_spacing(role: str, ball_side: str, location: MotionPoint) -> MotionPoint:
    weak_x = -21.0 if ball_side == "right" else 21.0
    strong_x = 16.0 if ball_side == "right" else -16.0
    if role in {"movement_shooter", "spacer"}:
        return MotionPoint(weak_x, 6.5 if role == "spacer" else 14.0)
    if role in {"roll_big", "pop_big"}:
        return MotionPoint(strong_x * 0.25, 8.5)
    if role == "secondary_creator":
        return MotionPoint(strong_x, max(14.0, location.y + 5.0))
    if role == "slasher":
        return MotionPoint(-3.0 if ball_side == "right" else 3.0, 9.0)
    return MotionPoint(weak_x * 0.55, 18.0)


def _transition_spacing(role: str, ball_side: str, location: MotionPoint, notes: str) -> MotionPoint:
    lane_x = -15.0 if ball_side == "left" else 15.0
    weak_lane_x = -lane_x
    if "turnover_break" in notes:
        push_depth = max(24.0, location.y + 5.0)
    else:
        push_depth = max(22.0, location.y + 3.0)
    if role in {"movement_shooter", "spacer"}:
        return MotionPoint(weak_lane_x, push_depth - (6.0 if role == "spacer" else 2.5))
    if role == "secondary_creator":
        return MotionPoint(lane_x, push_depth)
    if role in {"roll_big", "pop_big", "post_hub"}:
        return MotionPoint(lane_x * 0.35, max(12.0, push_depth - 10.0))
    if role == "slasher":
        return MotionPoint(lane_x * 0.7, max(13.0, push_depth - 6.0))
    return MotionPoint(weak_lane_x * 0.65, max(15.0, push_depth - 4.5))


def _passer_relocate(player: dict[str, Any], ball_side: str, location: MotionPoint) -> MotionPoint:
    role = _role_name(player)
    if role == "primary_creator":
        drift = -11.0 if ball_side == "right" else 11.0
        return MotionPoint(location.x + drift, min(27.5, location.y + 3.0))
    if role in {"roll_big", "post_hub", "pop_big"}:
        block_x = -7.0 if ball_side == "right" else 7.0
        return MotionPoint(block_x, max(6.0, location.y - 6.0))
    if role in {"movement_shooter", "spacer"}:
        fill_x = 17.0 if ball_side == "right" else -17.0
        return MotionPoint(fill_x, 19.0)
    drift = -7.5 if ball_side == "right" else 7.5
    return MotionPoint(location.x + drift, min(26.0, location.y + 3.0))


def _resolve_ball_end(
    ball_owner: str | None,
    beat: dict[str, Any],
    resolved_positions: dict[str, MotionPoint],
    current_ball: MotionPoint,
) -> MotionPoint:
    if ball_owner and ball_owner in resolved_positions:
        return resolved_positions[ball_owner]
    fallback = _beat_location_point(beat)
    return fallback or current_ball


def _ball_control_point(beat: dict[str, Any], start: MotionPoint, end: MotionPoint) -> MotionPoint | None:
    event_type = str(beat.get("event_type") or "")
    mid_x = (start.x + end.x) / 2.0
    mid_y = (start.y + end.y) / 2.0
    label = str(beat.get("label") or "").lower()
    commentary = str(beat.get("commentary") or "").lower()
    if event_type == "pass":
        lateral = 3.5 if end.x >= start.x else -3.5
        lift = 4.5 if abs(end.x - start.x) > 8.0 else 2.5
        return MotionPoint(mid_x + lateral, mid_y + lift)
    if event_type == "possession_change" and label == "outlet":
        lateral = 1.8 if end.x >= start.x else -1.8
        return MotionPoint(mid_x + lateral, mid_y + 3.0)
    if event_type == "possession_start" and "transition" in commentary:
        lateral = 6.0 if end.x >= start.x else -6.0
        return MotionPoint(mid_x + lateral, mid_y + 6.5)
    return None


def _ball_mode(beat: dict[str, Any]) -> str:
    event_type = str(beat.get("event_type") or "")
    commentary = str(beat.get("commentary") or "").lower()
    label = str(beat.get("label") or "").lower()
    if event_type in {"pass", "jump_ball"}:
        return "air"
    if event_type == "possession_change" and ("outlet" in label or "inbounds" in commentary):
        return "air"
    if event_type in {"drive", "advance", "possession_change", "possession_start"}:
        return "dribble"
    return "held"


def _ball_arc_height(beat: dict[str, Any], start: MotionPoint, end: MotionPoint) -> float:
    event_type = str(beat.get("event_type") or "")
    label = str(beat.get("label") or "").lower()
    commentary = str(beat.get("commentary") or "").lower()
    distance = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
    if event_type == "jump_ball":
        return 16.0
    if event_type == "possession_change" and label == "outlet":
        return max(4.0, min(10.0, distance * 0.28))
    if event_type == "possession_change" and "inbounds" in commentary:
        return max(2.0, min(6.0, distance * 0.18))
    if event_type == "pass":
        return max(5.0, min(14.0, distance * 0.45))
    return 0.0


def _ball_dribble_count(beat: dict[str, Any], start: MotionPoint, end: MotionPoint, duration_ms: int) -> int:
    event_type = str(beat.get("event_type") or "")
    commentary = str(beat.get("commentary") or "").lower()
    if event_type not in {"drive", "advance", "possession_change", "possession_start"}:
        return 0
    if event_type in {"possession_change", "possession_start"} and ("outlet" in commentary or "transition" in commentary or "inbounds" in commentary):
        return 0
    distance = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
    pace_factor = max(1.0, duration_ms / 340.0)
    return max(1, int(round((distance / 5.5) * pace_factor)))


def _blend_reset_positions(
    *,
    current_positions: dict[str, MotionPoint],
    reset_positions: list[dict[str, Any]],
    beat: dict[str, Any],
    player_lookup: dict[str, dict[str, Any]],
) -> dict[str, MotionPoint]:
    event_type = str(beat.get("event_type") or "")
    commentary = str(beat.get("commentary") or "").lower()
    label = str(beat.get("label") or "").lower()
    if event_type == "jump_ball_setup":
        return {
            item["player_id"]: MotionPoint(float(item["x"]), float(item["y"]))
            for item in reset_positions
        }
    blended: dict[str, MotionPoint] = {}
    for item in reset_positions:
        pid = item["player_id"]
        target = MotionPoint(float(item["x"]), float(item["y"]))
        prior = current_positions.get(pid, target)
        player = player_lookup.get(pid, {})
        defensive_role = _def_role_name(player)
        blend = 0.72
        if event_type == "possession_change":
            blend = 0.58 if label == "outlet" else 0.5 if "turnover" in label else 0.65
        elif event_type == "possession_start":
            blend = 0.7 if "transition" in commentary else 0.82
        if defensive_role in {"rim_protector", "helper", "rebounder"}:
            blend -= 0.08
        blended[pid] = MotionPoint(
            (prior.x * (1 - blend)) + (target.x * blend),
            (prior.y * (1 - blend)) + (target.y * blend),
        )
    return blended


def _event_location_point(event: dict[str, Any]) -> MotionPoint | None:
    location = event.get("location") or {}
    if location.get("x") is None or location.get("y") is None:
        return None
    return MotionPoint(float(location["x"]), float(location["y"]))


def _initial_ball_point(initial_positions: list[dict[str, Any]]) -> MotionPoint:
    for item in initial_positions:
        if item.get("has_ball"):
            return MotionPoint(float(item["x"]), float(item["y"]))
    return MotionPoint(0.0, 22.0)


def _beat_location_point(beat: dict[str, Any]) -> MotionPoint | None:
    location = beat.get("location") or {}
    if location.get("x") is None or location.get("y") is None:
        return None
    return MotionPoint(float(location["x"]), float(location["y"]))


def _resolve_collisions(
    raw_positions: dict[str, MotionPoint],
    player_ids: list[str],
    focus_ids: set[str | None],
) -> dict[str, MotionPoint]:
    positions = {pid: MotionPoint(point.x, point.y) for pid, point in raw_positions.items()}
    anchors = {pid: MotionPoint(point.x, point.y) for pid, point in raw_positions.items()}
    min_distance = 4.4
    locked = {pid for pid in focus_ids if pid}

    for _ in range(14):
        for i, a_id in enumerate(player_ids):
            a = positions.get(a_id)
            if a is None:
                continue
            for b_id in player_ids[i + 1 :]:
                b = positions.get(b_id)
                if b is None:
                    continue
                dx = b.x - a.x
                dy = b.y - a.y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist >= min_distance:
                    continue
                safe_dist = dist if dist > 0.001 else 0.001
                overlap = (min_distance - safe_dist) / 2.0
                ux = dx / safe_dist
                uy = dy / safe_dist
                a_locked = a_id in locked
                b_locked = b_id in locked
                if not a_locked and not b_locked:
                    a = MotionPoint(a.x - ux * overlap, a.y - uy * overlap)
                    b = MotionPoint(b.x + ux * overlap, b.y + uy * overlap)
                elif not a_locked:
                    a = MotionPoint(a.x - ux * overlap * 2.0, a.y - uy * overlap * 2.0)
                elif not b_locked:
                    b = MotionPoint(b.x + ux * overlap * 2.0, b.y + uy * overlap * 2.0)
                positions[a_id] = _clamp_point(a)
                positions[b_id] = _clamp_point(b)
        for pid in player_ids:
            current = positions.get(pid)
            anchor = anchors.get(pid)
            if current is None or anchor is None or pid in locked:
                continue
            positions[pid] = _clamp_point(
                MotionPoint(
                    (current.x * 0.92) + (anchor.x * 0.08),
                    (current.y * 0.92) + (anchor.y * 0.08),
                )
            )
    return positions


def _clamp_point(point: MotionPoint) -> MotionPoint:
    return MotionPoint(
        x=max(-23.5, min(23.5, point.x)),
        y=max(1.5, min(45.5, point.y)),
    )


def _camera_target_from_point(point: MotionPoint, offense_team_code: str, event_type: str) -> tuple[float, float, float]:
    x_shift = max(-10.0, min(10.0, (point.x / 25.0) * -8.0))
    py = _full_court_pct_y(point.y, offense_team_code)
    offense_is_away = offense_team_code.startswith("AWY") or offense_team_code == "AWY"
    desired_screen_y = 62.0 if offense_is_away else 38.0
    y_shift = max(-22.0, min(22.0, ((py - desired_screen_y) / 50.0) * 24.0))
    if event_type in {"drive", "shot"}:
        scale = 1.42
    elif event_type in {"pass", "possession_change", "possession_start"}:
        scale = 1.24
    else:
        scale = 1.32
    return (x_shift, y_shift, scale)


def _full_court_pct_y(y: float, offense_team_code: str) -> float:
    offense_is_away = offense_team_code.startswith("AWY") or offense_team_code == "AWY"
    if offense_is_away:
        return (y / 47.0) * 50.0
    return 100.0 - (y / 47.0) * 50.0


def _actor_easing(beat: dict[str, Any], player_id: str) -> str:
    event_type = str(beat.get("event_type") or "")
    if player_id == beat.get("focus_player_id"):
        if event_type in {"drive", "cut", "jump_ball"}:
            return "ease_in_out"
        if event_type in {"screen", "pass", "handoff"}:
            return "ease_out"
    if player_id == beat.get("ball_player_id"):
        return "linear"
    if event_type in {"possession_change", "possession_start", "jump_ball_setup"}:
        return "ease_in_out"
    return "ease_out"


def _ball_easing(beat: dict[str, Any]) -> str:
    event_type = str(beat.get("event_type") or "")
    if event_type in {"pass", "jump_ball"}:
        return "linear"
    if event_type in {"drive", "possession_change", "possession_start"}:
        return "ease_in_out"
    return "ease_out"


def _camera_easing(beat: dict[str, Any]) -> str:
    event_type = str(beat.get("event_type") or "")
    if event_type in {"possession_change", "possession_start", "jump_ball_setup", "jump_ball"}:
        return "ease_in_out"
    return "ease_out"


def _ball_side(x: float) -> str:
    if x < -4.0:
        return "left"
    if x > 4.0:
        return "right"
    return "middle"


def _role_name(player: dict[str, Any]) -> str:
    return str(player.get("offensive_role") or "glue")


def _def_role_name(player: dict[str, Any]) -> str:
    return str(player.get("defensive_role") or "helper")
