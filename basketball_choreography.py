from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from basketball_court import BASKET_CENTER_Y, court_point
from basketball_sim_schema import CourtZone


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
    control: MotionPoint | None
    easing: str
    delay: float
    tempo: float
    locomotion: str


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
    current_offense_team_code = str(beats[0].get("offense_team_code") or home_team) if beats else home_team
    current_camera = _camera_target_from_point(current_ball, current_offense_team_code, "possession_start")
    event_cursor = -1
    elapsed_ms = 0
    segments: list[ChoreographySegment] = []

    for beat_index, beat in enumerate(beats):
        offense_team_code = beat.get("offense_team_code") or home_team
        if offense_team_code != current_offense_team_code:
            current_positions = {
                pid: _flip_possession_frame(point)
                for pid, point in current_positions.items()
            }
            current_ball = _flip_possession_frame(current_ball)
            current_offense_team_code = str(offense_team_code)
        working_positions = dict(current_positions)
        reset_positions = beat.get("reset_positions") or []
        if reset_positions:
            working_positions = _blend_reset_positions(
                current_positions=current_positions,
                reset_positions=reset_positions,
                beat=beat,
                player_lookup=player_lookup,
            )

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
        event_type = str(beat.get("event_type") or "")
        base_duration_ms = max(260, int(beat.get("duration_ms") or 800))
        actor_motion_budget = _actor_motion_budget(event_type)
        max_actor_distance = 0.0

        constrained_positions: dict[str, MotionPoint] = {}
        for player in players:
            pid = player["player_id"]
            start_point = current_positions.get(pid, resolved_positions.get(pid, MotionPoint(0.0, 22.0)))
            desired_end = resolved_positions.get(pid, start_point)
            end_point = _limit_track_target(
                start=start_point,
                desired_end=desired_end,
                max_distance=actor_motion_budget.focus_max if pid in focus_ids else actor_motion_budget.role_max,
            )
            constrained_positions[pid] = end_point
            max_actor_distance = max(max_actor_distance, _point_distance(start_point, end_point))

        resolved_positions = constrained_positions
        ball_end = _resolve_ball_end(ball_owner, beat, resolved_positions, current_ball)
        ball_distance = _point_distance(current_ball, ball_end)
        duration_ms = _segment_duration_ms(
            event_type=event_type,
            base_duration_ms=base_duration_ms,
            max_actor_distance=max_actor_distance,
            ball_distance=ball_distance,
        )

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
                    control=_actor_control_point(beat, player, player["player_id"], start_point, end_point),
                    easing=_actor_easing(beat, player["player_id"]),
                    delay=_actor_motion_delay(beat, player, player["player_id"]),
                    tempo=_actor_motion_tempo(beat, player, player["player_id"]),
                    locomotion=_actor_locomotion(beat, player, player["player_id"], start_point, end_point),
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
    ball_owner_id = beat.get("ball_player_id")
    receiver_id = event.get("receiver_id")
    defender_id = event.get("defender_id")
    notes = " ".join(
        part
        for part in (
            str(event.get("notes") or "").lower(),
            str(beat.get("label") or "").lower(),
            str(beat.get("commentary") or "").lower(),
        )
        if part
    )
    ball_side = _ball_side(location.x)
    offense_ids = team_players.get(offense_team_code, [])
    defense_ids = team_players.get(defense_team_code, [])
    beat_index = int(beat.get("index", 0) or 0)

    offensive_shape = _build_offensive_shape(
        offense_ids=offense_ids,
        player_lookup=player_lookup,
        current_positions=current_positions,
        actor_id=actor_id,
        ball_owner_id=ball_owner_id,
        receiver_id=receiver_id,
        location=location,
        event_type=event_type,
        notes=notes,
        ball_side=ball_side,
        beat_index=beat_index,
    )
    defensive_shape = _build_defensive_shape(
        defense_ids=defense_ids,
        player_lookup=player_lookup,
        current_positions=current_positions,
        location=location,
        event_type=event_type,
        notes=notes,
        ball_side=ball_side,
        assignment_map=assignment_map,
        actor_id=actor_id,
        defender_id=defender_id,
        receiver_id=receiver_id,
        beat_index=beat_index,
    )

    positions.update(offensive_shape)
    positions.update(defensive_shape)
    return positions


def _build_offensive_shape(
    *,
    offense_ids: list[str],
    player_lookup: dict[str, dict[str, Any]],
    current_positions: dict[str, MotionPoint],
    actor_id: str | None,
    ball_owner_id: str | None,
    receiver_id: str | None,
    location: MotionPoint,
    event_type: str,
    notes: str,
    ball_side: str,
    beat_index: int,
) -> dict[str, MotionPoint]:
    shape = {
        pid: _existing_or_anchor(current_positions, pid, _role_anchor(player_lookup.get(pid, {}), ball_side), preserve=0.68)
        for pid in offense_ids
    }
    if actor_id in shape:
        shape[actor_id] = _clamp_point(location)

    if event_type in {"possession_change", "possession_start"}:
        phase = (
            "inbound"
            if "inbound" in notes
            else "push"
            if any(token in notes for token in ("outlet", "turnover", "transition", "push"))
            else "walkup"
        )
        live_ball_owner = ball_owner_id if ball_owner_id in shape else actor_id if actor_id in shape else None
        for idx, pid in enumerate(offense_ids):
            role = _role_name(player_lookup.get(pid, {}))
            target = _possession_offense_spacing(role, ball_side, location, phase=phase, lane_index=idx)
            if pid != live_ball_owner:
                target = _jitter_target(target, pid, beat_index, amplitude=1.1 if phase == "inbound" else 1.4)
            weight = 0.54 if pid == live_ball_owner else 0.32 if phase == "push" else 0.28
            shape[pid] = _toward(current_positions.get(pid), target, weight)
        if live_ball_owner in shape:
            ball_target = _possession_ball_handler_point(location, phase=phase)
            shape[live_ball_owner] = _toward(current_positions.get(live_ball_owner), ball_target, 0.62 if phase == "push" else 0.56)

    elif event_type == "screen" and actor_id in shape and receiver_id in shape:
        screen_side = -1 if ball_side == "left" else 1
        handler_point = MotionPoint(location.x + (screen_side * 2.4), location.y + 1.8)
        shape[receiver_id] = _toward(current_positions.get(receiver_id), handler_point, 0.72)
        if "double_drag_2" in notes:
            second_screener = next((pid for pid in offense_ids if pid not in {actor_id, receiver_id} and _role_name(player_lookup.get(pid, {})) in {"roll_big", "pop_big", "glue"}), None)
            if second_screener:
                shape[second_screener] = _toward(current_positions.get(second_screener), MotionPoint(location.x + (screen_side * 5.2), location.y + 2.5), 0.68)
        for pid in offense_ids:
            if pid not in {actor_id, receiver_id}:
                role = _role_name(player_lookup.get(pid, {}))
                drift_target = _weak_side_activity(role, ball_side, location, phase="screen", lane_index=offense_ids.index(pid))
                drift_target = _jitter_target(drift_target, pid, beat_index, amplitude=1.0)
                shape[pid] = _toward(current_positions.get(pid), drift_target, 0.34)

    elif event_type == "handoff" and actor_id in shape and receiver_id in shape:
        handoff_side = -1 if location.x < 0 else 1
        shape[actor_id] = _toward(current_positions.get(actor_id), MotionPoint(location.x - (handoff_side * 0.9), location.y), 0.82)
        shape[receiver_id] = _toward(current_positions.get(receiver_id), MotionPoint(location.x + (handoff_side * 1.3), location.y + 0.8), 0.76)
        for pid in offense_ids:
            if pid not in {actor_id, receiver_id}:
                role = _role_name(player_lookup.get(pid, {}))
                drift_target = _weak_side_activity(role, ball_side, location, phase="handoff", lane_index=offense_ids.index(pid))
                drift_target = _jitter_target(drift_target, pid, beat_index, amplitude=1.0)
                shape[pid] = _toward(current_positions.get(pid), drift_target, 0.32)

    elif event_type == "drive" and actor_id in shape:
        for pid in offense_ids:
            if pid == actor_id:
                continue
            role = _role_name(player_lookup.get(pid, {}))
            drive_target = _drive_spacing(role, ball_side, location)
            alive_target = _weak_side_activity(role, ball_side, location, phase="drive", lane_index=offense_ids.index(pid))
            drift_target = _jitter_target(_blend_points(drive_target, alive_target, weight=0.35), pid, beat_index, amplitude=1.25)
            shape[pid] = _toward(current_positions.get(pid), drift_target, 0.34)

    elif event_type == "advance" and actor_id in shape:
        for pid in offense_ids:
            role = _role_name(player_lookup.get(pid, {}))
            if pid == actor_id:
                shape[pid] = _clamp_point(location)
            else:
                target = _jitter_target(_transition_spacing(role, ball_side, location, notes), pid, beat_index, amplitude=1.0)
                shape[pid] = _toward(current_positions.get(pid), target, 0.42)

    elif event_type == "pass":
        target_id = receiver_id or actor_id
        if target_id in shape:
            shape[target_id] = _toward(current_positions.get(target_id), location, 0.84)
        if actor_id in shape and receiver_id:
            shape[actor_id] = _toward(current_positions.get(actor_id), _passer_relocate(player_lookup.get(actor_id, {}), ball_side, location), 0.52)
        if "kickout" in notes:
            for pid in offense_ids:
                if pid not in {actor_id, receiver_id}:
                    target = _jitter_target(_off_ball_relocation(player_lookup.get(pid, {}), _ball_side(location.x), phase="kickout"), pid, beat_index, amplitude=1.15)
                    shape[pid] = _toward(current_positions.get(pid), target, 0.34)
        elif "pocket pass" in notes:
            for pid in offense_ids:
                if pid not in {actor_id, receiver_id}:
                    target = _jitter_target(_off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="roller_pass"), pid, beat_index, amplitude=0.95)
                    shape[pid] = _toward(current_positions.get(pid), target, 0.38)
        for pid in offense_ids:
            if pid not in {actor_id, receiver_id}:
                target = _jitter_target(_off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="pass"), pid, beat_index, amplitude=0.9)
                shape[pid] = _toward(current_positions.get(pid), target, 0.26)

    elif event_type == "shot" and actor_id in shape:
        for pid in offense_ids:
            role = _role_name(player_lookup.get(pid, {}))
            if pid == actor_id:
                shape[pid] = _clamp_point(location)
            elif "big" in role or role in {"roll_big", "post_hub"}:
                shape[pid] = _toward(current_positions.get(pid), MotionPoint(-4.0 if pid.endswith("pf") else 4.0, 6.5), 0.7)
            else:
                target = _jitter_target(_off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="shot"), pid, beat_index, amplitude=0.8)
                shape[pid] = _toward(current_positions.get(pid), target, 0.22)

    elif event_type == "rebound":
        for pid in offense_ids:
            role = _role_name(player_lookup.get(pid, {}))
            if pid == actor_id:
                shape[pid] = _clamp_point(location)
            elif "big" in role or role in {"roll_big", "post_hub"}:
                shape[pid] = _toward(current_positions.get(pid), MotionPoint(-5.5 if pid.endswith("pf") else 5.5, 8.0), 0.82)
            else:
                target = _jitter_target(_off_ball_relocation(player_lookup.get(pid, {}), ball_side, phase="rebound"), pid, beat_index, amplitude=0.8)
                shape[pid] = _toward(current_positions.get(pid), target, 0.28)

    return shape


def _build_defensive_shape(
    *,
    defense_ids: list[str],
    player_lookup: dict[str, dict[str, Any]],
    current_positions: dict[str, MotionPoint],
    location: MotionPoint,
    event_type: str,
    notes: str,
    ball_side: str,
    assignment_map: dict[str, dict[str, Any]],
    actor_id: str | None,
    defender_id: str | None,
    receiver_id: str | None,
    beat_index: int,
) -> dict[str, MotionPoint]:
    shape = {
        pid: _existing_or_anchor(current_positions, pid, _defensive_anchor(player_lookup.get(pid, {}), ball_side), preserve=0.72)
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
            shape[primary_defender] = _toward(current_positions.get(primary_defender), MotionPoint(location.x - 1.2, location.y + 2.0), 0.68)
        else:
            shape[primary_defender] = _toward(current_positions.get(primary_defender), MotionPoint(location.x - 1.8, location.y + 1.1), 0.58)

    if event_type in {"possession_change", "possession_start"}:
        phase = (
            "inbound"
            if "inbound" in notes
            else "push"
            if any(token in notes for token in ("outlet", "turnover", "transition", "push"))
            else "walkup"
        )
        sorted_ids = sorted(
            defense_ids,
            key=lambda pid: (
                0 if pid == primary_defender else 1,
                -help_priorities.get(pid, 0.5),
                -matchup_strengths.get(pid, 0.5),
            ),
        )
        for idx, pid in enumerate(sorted_ids):
            role = _def_role_name(player_lookup.get(pid, {}))
            target = _possession_defense_spacing(
                role,
                ball_side,
                location,
                phase=phase,
                lane_index=idx,
                is_primary=(pid == primary_defender),
            )
            if pid != primary_defender:
                target = _jitter_target(target, pid, beat_index, amplitude=0.9 if phase == "push" else 0.7)
            weight = 0.42 if pid == primary_defender else 0.28 if phase == "push" else 0.22
            shape[pid] = _toward(current_positions.get(pid), target, weight)

    elif event_type in {"screen", "handoff"}:
        help_targets = sorted(
            [pid for pid in defense_ids if pid != primary_defender],
            key=lambda pid: help_priorities.get(pid, 0.5),
            reverse=True,
        )[:2]
        for idx, pid in enumerate(help_targets):
            shade = 4.0 + (matchup_strengths.get(pid, 0.5) * 2.5)
            target = _jitter_target(MotionPoint(((-1 if idx == 0 else 1) * shade), 15.0), pid, beat_index, amplitude=0.75)
            shape[pid] = _toward(current_positions.get(pid), target, 0.34)
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
                target = _jitter_target(MotionPoint((-shade if idx % 2 == 0 else shade), 10.5), pid, beat_index, amplitude=0.65)
                shape[pid] = _toward(current_positions.get(pid), target, 0.52)
            else:
                stunt_target = _defensive_stunt(ball_side, location, lane_index=idx, matchup_strength=matchup_strengths.get(pid, 0.5))
                stunt_target = _jitter_target(stunt_target, pid, beat_index, amplitude=0.8)
                shape[pid] = _toward(current_positions.get(pid), stunt_target, 0.26)
    elif event_type == "advance":
        for idx, pid in enumerate(defense_ids):
            role = _def_role_name(player_lookup.get(pid, {}))
            if pid == primary_defender:
                shape[pid] = _toward(current_positions.get(pid), MotionPoint(location.x - 0.8, location.y + 2.8), 0.82)
            elif role in {"rim_protector", "helper", "rebounder"}:
                target = _jitter_target(MotionPoint(-4.0 if idx % 2 == 0 else 4.0, max(9.0, location.y - 8.0)), pid, beat_index, amplitude=0.75)
                shape[pid] = _toward(current_positions.get(pid), target, 0.42)
            else:
                lane_x = 10.0 if idx % 2 == 0 else -10.0
                target = _jitter_target(MotionPoint(lane_x, max(14.0, location.y - 2.0)), pid, beat_index, amplitude=0.85)
                shape[pid] = _toward(current_positions.get(pid), target, 0.34)
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
                    shape[pid] = _toward(current_positions.get(pid), MotionPoint(location.x - 1.0, location.y + 1.4), 0.86)
                    continue
            if role in {"wing_stopper", "point_of_attack"}:
                stay_home_bias = 1.0 + matchup_strengths.get(pid, 0.5)
                target = _jitter_target(MotionPoint(location.x - (1.2 * stay_home_bias), location.y + 1.4 + (idx * 0.8)), pid, beat_index, amplitude=0.7)
                shape[pid] = _toward(current_positions.get(pid), target, 0.58)
            else:
                recover_target = _defensive_stunt(ball_side, location, lane_index=idx, matchup_strength=matchup_strengths.get(pid, 0.5))
                recover_target = _jitter_target(recover_target, pid, beat_index, amplitude=0.8)
                shape[pid] = _toward(current_positions.get(pid), recover_target, 0.20)
        if "kickout" in notes:
            for pid in rotating:
                if pid not in shape:
                    continue
                if help_priorities.get(pid, 0.5) > 0.55:
                    recover_x = -11.0 if ball_side == "right" else 11.0
                    target = _jitter_target(MotionPoint(recover_x, 14.0 + (rotating.index(pid) * 1.2)), pid, beat_index, amplitude=0.75)
                    shape[pid] = _toward(current_positions.get(pid), target, 0.40)
    elif event_type in {"shot", "rebound"}:
        for idx, pid in enumerate(defense_ids):
            if pid == primary_defender:
                continue
            role = _def_role_name(player_lookup.get(pid, {}))
            if role in {"rim_protector", "rebounder", "switch_big"}:
                target = _jitter_target(MotionPoint(-3.5 if idx % 2 == 0 else 3.5, 8.5), pid, beat_index, amplitude=0.65)
                shape[pid] = _toward(current_positions.get(pid), target, 0.60)
            else:
                stay_attached = 11.0 + (matchup_strengths.get(pid, 0.5) * 5.0)
                target = _jitter_target(MotionPoint(-stay_attached if ball_side == "right" else stay_attached, 17.0 + (idx * 1.4)), pid, beat_index, amplitude=0.7)
                shape[pid] = _toward(current_positions.get(pid), target, 0.36)

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
            shape[hedger] = _toward(current_positions.get(hedger), MotionPoint(location.x - 0.6, location.y + 1.8), 0.84)

    return shape


def _role_anchor(player: dict[str, Any], ball_side: str) -> MotionPoint:
    role = _role_name(player)
    weak_corner = _zone_point(CourtZone.LEFT_CORNER if ball_side == "right" else CourtZone.RIGHT_CORNER)
    strong_wing = _zone_point(CourtZone.RIGHT_WING if ball_side == "right" else CourtZone.LEFT_WING)
    top = _zone_point(CourtZone.TOP)
    paint = _zone_point(CourtZone.PAINT)
    if role == "primary_creator":
        return MotionPoint(top.x, top.y)
    if role == "secondary_creator":
        return MotionPoint(strong_wing.x * 0.7, strong_wing.y)
    if role in {"movement_shooter", "spacer"}:
        return MotionPoint(weak_corner.x, 8.0 if role == "spacer" else 15.0)
    if role in {"roll_big", "pop_big"}:
        return MotionPoint(strong_wing.x * 0.25, top.y - 2.5)
    if role == "post_hub":
        return MotionPoint(strong_wing.x * 0.6, paint.y + 1.0)
    if role == "slasher":
        return MotionPoint(strong_wing.x * 0.78, paint.y)
    return MotionPoint(weak_corner.x * 0.72, 18.0)


def _defensive_anchor(player: dict[str, Any], ball_side: str) -> MotionPoint:
    role = _def_role_name(player)
    top = _zone_point(CourtZone.TOP)
    paint = _zone_point(CourtZone.PAINT)
    if role == "point_of_attack":
        return MotionPoint(top.x, top.y - 1.5)
    if role == "wing_stopper":
        return MotionPoint(-12.0 if ball_side == "left" else 12.0, 18.0)
    if role == "helper":
        return MotionPoint(0.0, paint.y + 2.5)
    if role == "rim_protector":
        return MotionPoint(0.0, BASKET_CENTER_Y + 3.5)
    if role == "rebounder":
        return MotionPoint(-6.0 if ball_side == "left" else 6.0, paint.y)
    return MotionPoint(7.0 if ball_side == "right" else -7.0, paint.y + 4.0)


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


def _possession_ball_handler_point(location: MotionPoint, *, phase: str) -> MotionPoint:
    if phase == "inbound":
        return MotionPoint(max(-22.0, min(22.0, location.x)), max(82.0, location.y))
    if phase == "push":
        return MotionPoint(location.x * 0.35, max(62.0, min(78.0, location.y)))
    return MotionPoint(location.x * 0.22, max(50.0, min(60.0, location.y)))


def _possession_offense_spacing(
    role: str,
    ball_side: str,
    location: MotionPoint,
    *,
    phase: str,
    lane_index: int,
) -> MotionPoint:
    strong_x = 16.0 if ball_side == "right" else -16.0
    weak_x = -strong_x
    if phase == "inbound":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x * 1.18, 78.0 - (lane_index * 1.6))
        if role in {"secondary_creator", "slasher"}:
            return MotionPoint(strong_x * 0.72, 74.0 - (lane_index * 1.4))
        if role in {"roll_big", "pop_big", "post_hub"}:
            return MotionPoint(strong_x * 0.28, 66.0)
        return MotionPoint(weak_x * 0.44, 71.0)
    if phase == "push":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, max(58.0, location.y - 8.0 - lane_index))
        if role in {"secondary_creator", "slasher"}:
            return MotionPoint(strong_x * 0.84, max(56.0, location.y - 6.0))
        if role in {"roll_big", "pop_big", "post_hub"}:
            return MotionPoint(strong_x * 0.25, max(46.0, location.y - 14.0))
        return MotionPoint(weak_x * 0.58, max(54.0, location.y - 7.0))
    if role in {"movement_shooter", "spacer"}:
        return MotionPoint(weak_x, 24.5 - (lane_index * 0.7))
    if role in {"secondary_creator", "slasher"}:
        return MotionPoint(strong_x * 0.72, 25.5)
    if role in {"roll_big", "pop_big", "post_hub"}:
        return MotionPoint(strong_x * 0.24, 18.5)
    return MotionPoint(weak_x * 0.46, 22.5)


def _possession_defense_spacing(
    role: str,
    ball_side: str,
    location: MotionPoint,
    *,
    phase: str,
    lane_index: int,
    is_primary: bool,
) -> MotionPoint:
    strong_x = 14.0 if ball_side == "right" else -14.0
    weak_x = -strong_x
    if is_primary:
        if phase == "inbound":
            return MotionPoint(location.x + 5.5, max(72.0, location.y - 9.0))
        if phase == "push":
            return MotionPoint(location.x - 0.6, max(54.0, location.y - 8.0))
        return MotionPoint(location.x - 0.8, max(24.0, location.y - 7.0))
    if role in {"rim_protector", "helper", "rebounder"}:
        if phase == "inbound":
            return MotionPoint(0.0 if lane_index % 2 == 0 else strong_x * 0.3, 52.0 + (lane_index * 1.4))
        if phase == "push":
            return MotionPoint(strong_x * 0.24 if lane_index % 2 == 0 else weak_x * 0.24, max(38.0, location.y - 14.0))
        return MotionPoint(strong_x * 0.18 if lane_index % 2 == 0 else weak_x * 0.18, 15.0 + (lane_index * 0.7))
    if role in {"wing_stopper", "point_of_attack"}:
        if phase == "push":
            return MotionPoint(strong_x if lane_index % 2 == 0 else weak_x, max(42.0, location.y - 10.0))
        return MotionPoint(strong_x * 0.86 if lane_index % 2 == 0 else weak_x * 0.86, 21.0 + (lane_index * 0.8))
    if phase == "inbound":
        return MotionPoint(strong_x * 0.62 if lane_index % 2 == 0 else weak_x * 0.62, 58.0 + (lane_index * 1.2))
    if phase == "push":
        return MotionPoint(strong_x * 0.74 if lane_index % 2 == 0 else weak_x * 0.74, max(44.0, location.y - 9.0))
    return MotionPoint(strong_x * 0.52 if lane_index % 2 == 0 else weak_x * 0.52, 20.5 + (lane_index * 0.8))


def _weak_side_activity(role: str, ball_side: str, location: MotionPoint, *, phase: str, lane_index: int) -> MotionPoint:
    weak_x = -20.0 if ball_side == "right" else 20.0
    strong_x = -weak_x
    if phase == "screen":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, 11.0 + (lane_index * 1.2))
        if role in {"secondary_creator", "slasher"}:
            return MotionPoint(strong_x * 0.55, 18.0 + (lane_index * 0.8))
        if role in {"roll_big", "pop_big", "post_hub"}:
            return MotionPoint(strong_x * 0.18, 17.0)
    if phase == "handoff":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x * 0.92, 12.5 + (lane_index * 0.9))
        if role in {"secondary_creator", "slasher"}:
            return MotionPoint(strong_x * 0.68, 17.5)
        if role in {"roll_big", "pop_big", "post_hub"}:
            return MotionPoint(strong_x * 0.24, 13.5)
    if phase == "drive":
        if role in {"movement_shooter", "spacer"}:
            return MotionPoint(weak_x, 7.5 if role == "spacer" else 14.5)
        if role in {"secondary_creator", "slasher"}:
            return MotionPoint(strong_x * 0.86, max(16.0, location.y + 3.0))
        if role in {"roll_big", "pop_big", "post_hub"}:
            return MotionPoint(strong_x * 0.22, 10.0)
    return _off_ball_relocation({"offensive_role": role}, ball_side, phase="pass")


def _defensive_stunt(ball_side: str, location: MotionPoint, *, lane_index: int, matchup_strength: float) -> MotionPoint:
    side_x = -11.0 if ball_side == "right" else 11.0
    recover_band = 15.0 + (lane_index * 1.4)
    stunt_depth = max(13.0, location.y - (2.5 + (matchup_strength * 2.5)))
    return MotionPoint(side_x * (0.78 if lane_index % 2 == 0 else 1.02), min(recover_band, stunt_depth + 4.5))


def _toward(current: MotionPoint | None, target: MotionPoint, weight: float) -> MotionPoint:
    if current is None:
        return _clamp_point(target)
    return _clamp_point(
        MotionPoint(
            (current.x * (1.0 - weight)) + (target.x * weight),
            (current.y * (1.0 - weight)) + (target.y * weight),
        )
    )


def _blend_points(a: MotionPoint, b: MotionPoint, *, weight: float) -> MotionPoint:
    return _clamp_point(
        MotionPoint(
            (a.x * (1.0 - weight)) + (b.x * weight),
            (a.y * (1.0 - weight)) + (b.y * weight),
        )
    )


def _existing_or_anchor(
    current_positions: dict[str, MotionPoint],
    player_id: str,
    anchor: MotionPoint,
    *,
    preserve: float,
) -> MotionPoint:
    current = current_positions.get(player_id)
    if current is None:
        return anchor
    return _toward(current, anchor, 1.0 - preserve)


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
    event_type = str(beat.get("event_type") or "")
    if event_type == "shot":
        shot_origin = _beat_location_point(beat) or (resolved_positions.get(ball_owner) if ball_owner else None) or current_ball
        return _shot_ball_target(beat, shot_origin)
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
    if event_type == "shot":
        shot_origin = _beat_location_point(beat) or start
        rim_x = 0.0
        rim_y = BASKET_CENTER_Y + 0.4
        dx = rim_x - shot_origin.x
        dy = rim_y - shot_origin.y
        distance = (dx * dx + dy * dy) ** 0.5
        lift = max(8.0, min(22.0, 6.0 + (distance * 0.35)))
        return MotionPoint((shot_origin.x + rim_x) / 2.0, max(shot_origin.y, rim_y) + lift)
    return None


def _ball_mode(beat: dict[str, Any]) -> str:
    event_type = str(beat.get("event_type") or "")
    commentary = str(beat.get("commentary") or "").lower()
    label = str(beat.get("label") or "").lower()
    if event_type in {"pass", "jump_ball"}:
        return "air"
    if event_type == "shot":
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
    if event_type == "shot":
        return max(10.0, min(24.0, 7.0 + (distance * 0.32)))
    return 0.0


def _shot_ball_target(beat: dict[str, Any], shot_origin: MotionPoint) -> MotionPoint:
    made = str(beat.get("label") or "").lower() == "bucket"
    rim_y = BASKET_CENTER_Y + 0.4
    if made:
        return MotionPoint(0.0, rim_y)
    x_side = -1.6 if shot_origin.x < 0 else 1.6
    distance = abs(shot_origin.x) + max(0.0, shot_origin.y - rim_y)
    if distance > 26.0:
        return MotionPoint(x_side * 1.8, rim_y + 3.0)
    return MotionPoint(x_side, rim_y + 1.4)


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
            blend = 0.44 if label == "outlet" else 0.36 if "turnover" in label else 0.40 if "inbound" in label else 0.48
        elif event_type == "possession_start":
            blend = 0.58 if "transition" in commentary else 0.64 if "push" in commentary else 0.70
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
        y=max(BASKET_CENTER_Y - 0.5, min(93.5, point.y)),
    )


def _zone_point(zone: CourtZone) -> MotionPoint:
    cp = court_point({
        CourtZone.TOP: 0.0,
    }.get(zone, 0.0), {
        CourtZone.TOP: 25.0,
    }.get(zone, 25.0))
    if zone == CourtZone.LEFT_CORNER:
        cp = court_point(-22.0, 6.0)
    elif zone == CourtZone.RIGHT_CORNER:
        cp = court_point(22.0, 6.0)
    elif zone == CourtZone.LEFT_WING:
        cp = court_point(-19.0, 22.5)
    elif zone == CourtZone.RIGHT_WING:
        cp = court_point(19.0, 22.5)
    elif zone == CourtZone.PAINT:
        cp = court_point(0.0, 12.0)
    elif zone == CourtZone.RIM:
        cp = court_point(0.0, 6.75)
    return MotionPoint(cp.x, cp.y)


def _camera_target_from_point(point: MotionPoint, offense_team_code: str, event_type: str) -> tuple[float, float, float]:
    cam_x = point.x * 0.30
    cam_y = (point.y - 47.0) * 0.20
    if event_type in {"drive", "shot", "cut"}:
        cam_x = point.x * 0.45
        cam_y = (point.y - 47.0) * 0.24
        scale = 1.12
    elif event_type in {"pass", "screen", "handoff"}:
        scale = 1.05
    elif event_type in {"possession_change", "possession_start", "advance"}:
        scale = 1.01
    else:
        scale = 1.03
    return (cam_x, cam_y, scale)


def _full_court_pct_y(y: float, offense_team_code: str) -> float:
    offense_is_away = offense_team_code.startswith("AWY") or offense_team_code == "AWY"
    if offense_is_away:
        return (y / 94.0) * 100.0
    return 100.0 - (y / 94.0) * 100.0


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


def _trait(player: dict[str, Any], name: str, default: float = 10.0) -> float:
    traits = player.get("traits") or {}
    try:
        value = float(traits.get(name, default))
    except (TypeError, ValueError):
        value = default
    return value


def _actor_motion_tempo(beat: dict[str, Any], player: dict[str, Any], player_id: str) -> float:
    event_type = str(beat.get("event_type") or "")
    is_focus = player_id == beat.get("focus_player_id")
    is_ball = player_id == beat.get("ball_player_id")
    is_receiver = player_id == beat.get("receiver_id")
    is_defender = player_id == beat.get("defender_id")
    speed = _trait(player, "speed")
    burst = _trait(player, "burst")
    stamina = _trait(player, "stamina")
    containment = _trait(player, "containment")
    screen_nav = _trait(player, "screen_nav")
    closeout = _trait(player, "closeout")
    ball_handle = _trait(player, "ball_handle")
    mobility = ((speed * 0.44) + (burst * 0.36) + (stamina * 0.20) - 10.0) / 10.0
    defensive_mobility = ((containment * 0.38) + (screen_nav * 0.34) + (closeout * 0.28) - 10.0) / 10.0
    handler_bonus = ((ball_handle - 10.0) / 10.0) * 0.10 if (is_ball or is_focus) else 0.0
    tempo = 1.0
    if event_type in {"drive", "cut"}:
        tempo += (mobility * 0.26) + handler_bonus
        if is_defender:
            tempo += defensive_mobility * 0.20
    elif event_type in {"advance", "possession_change", "possession_start"}:
        tempo += mobility * 0.22
        if is_defender:
            tempo += defensive_mobility * 0.14
    elif event_type in {"screen", "handoff"}:
        tempo += mobility * 0.14
        if is_receiver or is_defender:
            tempo += defensive_mobility * 0.10
    elif event_type == "pass":
        tempo += mobility * 0.12
        if is_receiver or is_defender:
            tempo += defensive_mobility * 0.10
    else:
        tempo += mobility * 0.08
    if is_focus or is_ball:
        tempo += 0.04
    return max(0.78, min(1.26, tempo))


def _actor_motion_delay(beat: dict[str, Any], player: dict[str, Any], player_id: str) -> float:
    event_type = str(beat.get("event_type") or "")
    is_focus = player_id == beat.get("focus_player_id")
    is_ball = player_id == beat.get("ball_player_id")
    is_receiver = player_id == beat.get("receiver_id")
    is_defender = player_id == beat.get("defender_id")
    side = str(player.get("side") or "")
    speed = _trait(player, "speed")
    burst = _trait(player, "burst")
    stamina = _trait(player, "stamina")
    containment = _trait(player, "containment")
    screen_nav = _trait(player, "screen_nav")
    help_rotation = _trait(player, "help_rotation")
    reaction = ((speed * 0.28) + (burst * 0.34) + (stamina * 0.14) + (containment * 0.12) + (screen_nav * 0.07) + (help_rotation * 0.05) - 10.0) / 10.0
    if is_focus or is_ball or is_receiver:
        delay = 0.01
    elif is_defender:
        delay = 0.09 if event_type in {"drive", "shot", "pass"} else 0.07
        delay -= reaction * 0.06
        if help_rotation >= 12.0 and event_type in {"drive", "pass"}:
            delay += 0.02
    elif side == "offense":
        delay = 0.14 if event_type in {"drive", "screen", "handoff", "pass"} else 0.11
        delay -= reaction * 0.04
    else:
        delay = 0.10 - reaction * 0.04
    return max(0.0, min(0.35, delay))


def _actor_control_point(
    beat: dict[str, Any],
    player: dict[str, Any],
    player_id: str,
    start: MotionPoint,
    end: MotionPoint,
) -> MotionPoint | None:
    event_type = str(beat.get("event_type") or "")
    is_focus = player_id == beat.get("focus_player_id")
    is_ball = player_id == beat.get("ball_player_id")
    is_defender = player_id == beat.get("defender_id")
    is_receiver = player_id == beat.get("receiver_id")
    locomotion = _actor_locomotion(beat, player, player_id, start, end)
    speed = _trait(player, "speed")
    burst = _trait(player, "burst")
    stamina = _trait(player, "stamina")
    containment = _trait(player, "containment")
    screen_nav = _trait(player, "screen_nav")
    closeout = _trait(player, "closeout")
    size = _trait(player, "size")
    reach = _trait(player, "reach")
    dx = end.x - start.x
    dy = end.y - start.y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance < 2.0:
        return None
    mid_x = (start.x + end.x) / 2.0
    mid_y = (start.y + end.y) / 2.0
    ux = dx / distance
    uy = dy / distance
    px = -uy
    py = ux
    mobility = ((speed * 0.42) + (burst * 0.38) + (stamina * 0.20) - 10.0) / 10.0
    defensive_curve = ((containment * 0.40) + (screen_nav * 0.34) + (closeout * 0.26) - 10.0) / 10.0
    size_drag = ((size * 0.55) + (reach * 0.45) - 10.0) / 10.0
    curvature = 0.0
    lead_pull = 0.0
    if event_type in {"drive", "cut"}:
        curvature = (0.18 + (mobility * 0.10)) * distance
        lead_pull = min(4.5, distance * (0.18 + max(0.0, mobility) * 0.08))
        if is_focus or is_ball:
            curvature *= 0.55
            lead_pull *= 1.18
        elif is_defender:
            curvature *= 0.72 - min(0.16, defensive_curve * 0.18)
            lead_pull *= 0.92
    elif event_type in {"screen", "handoff"}:
        curvature = (0.14 + (mobility * 0.06)) * distance
        lead_pull = min(3.2, distance * 0.12)
        if is_receiver:
            lead_pull *= 1.18
        if is_defender:
            curvature *= 0.70 - min(0.14, defensive_curve * 0.16)
    elif event_type in {"possession_change", "possession_start", "advance"}:
        curvature = (0.10 + max(0.0, mobility) * 0.05 + max(0.0, size_drag) * 0.02) * distance
        lead_pull = min(3.8, distance * (0.10 + max(0.0, mobility) * 0.05))
        if size_drag > 0.2:
            curvature *= 1.14
            lead_pull *= 0.86
    elif event_type in {"pass", "shot", "rebound"}:
        curvature = 0.06 * distance
        lead_pull = min(2.0, distance * 0.08)
        if is_defender:
            curvature *= 0.82
    else:
        return None
    if locomotion == "sprint":
        curvature *= 0.72
        lead_pull *= 1.28
    elif locomotion == "shuffle":
        curvature *= 1.12
        lead_pull *= 0.82
    elif locomotion == "backpedal":
        curvature *= 0.92
        lead_pull *= 0.68
    elif locomotion == "set":
        curvature *= 0.45
        lead_pull *= 0.5
    side = -1.0 if dx >= 0 else 1.0
    if is_defender:
        side *= -1.0
    if player.get("side") == "defense" and not is_defender:
        side *= -1.0
    control_x = mid_x + (px * curvature * side) + (ux * lead_pull)
    control_y = mid_y + (py * curvature * side) + (uy * lead_pull)
    return _clamp_point(MotionPoint(control_x, control_y))


def _jitter_target(target: MotionPoint, player_id: str, beat_index: int, *, amplitude: float = 1.2) -> MotionPoint:
    seed = sum((idx + 1) * ord(char) for idx, char in enumerate(f"{player_id}:{beat_index}"))
    dx = (((seed % 1000) / 500.0) - 1.0) * amplitude
    dy = ((((seed // 7) % 1000) / 500.0) - 1.0) * amplitude * 0.6
    return _clamp_point(MotionPoint(target.x + dx, target.y + dy))


def _actor_locomotion(
    beat: dict[str, Any],
    player: dict[str, Any],
    player_id: str,
    start: MotionPoint,
    end: MotionPoint,
) -> str:
    event_type = str(beat.get("event_type") or "")
    is_focus = player_id == beat.get("focus_player_id")
    is_ball = player_id == beat.get("ball_player_id")
    is_defender = player_id == beat.get("defender_id")
    is_receiver = player_id == beat.get("receiver_id")
    speed = _trait(player, "speed")
    burst = _trait(player, "burst")
    containment = _trait(player, "containment")
    screen_nav = _trait(player, "screen_nav")
    closeout = _trait(player, "closeout")
    size = _trait(player, "size")
    dx = end.x - start.x
    dy = end.y - start.y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance < 1.5:
        return "set"
    if event_type in {"drive", "cut"} and (is_focus or is_ball):
        return "sprint" if (burst + speed) >= 28.0 else "jog"
    if event_type == "advance":
        return "sprint" if speed >= 13.5 else "jog"
    if event_type in {"possession_change", "possession_start"}:
        if is_defender:
            return "backpedal" if distance >= 5.0 and containment >= 11.0 else "jog"
        return "jog"
    if event_type in {"screen", "handoff"}:
        if is_defender:
            return "shuffle" if (containment + screen_nav + closeout) >= 34.0 else "jog"
        if is_receiver:
            return "sprint" if burst >= 13.0 else "jog"
        return "jog"
    if event_type == "pass":
        if is_defender:
            return "shuffle" if containment >= 12.0 else "jog"
        return "jog"
    if event_type in {"shot", "rebound"}:
        if is_defender and size < 11.0:
            return "shuffle"
        return "jog"
    return "jog"


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


@dataclass(frozen=True)
class _ActorMotionBudget:
    focus_max: float
    role_max: float


def _actor_motion_budget(event_type: str) -> _ActorMotionBudget:
    if event_type in {"possession_change", "substitution"}:
        return _ActorMotionBudget(focus_max=8.5, role_max=6.8)
    if event_type in {"possession_start", "advance"}:
        return _ActorMotionBudget(
            focus_max=18.0 if event_type == "possession_start" else 12.5,
            role_max=14.0 if event_type == "possession_start" else 9.5,
        )
    if event_type in {"drive", "cut"}:
        return _ActorMotionBudget(focus_max=6.6, role_max=4.2)
    if event_type in {"pass", "screen", "handoff"}:
        return _ActorMotionBudget(focus_max=5.4, role_max=3.8)
    if event_type in {"shot", "rebound"}:
        return _ActorMotionBudget(focus_max=4.8, role_max=3.6)
    return _ActorMotionBudget(focus_max=5.8, role_max=4.4)


def _segment_duration_ms(
    *,
    event_type: str,
    base_duration_ms: int,
    max_actor_distance: float,
    ball_distance: float,
) -> int:
    event_floor = {
        "possession_change": 1180,
        "possession_start": 960,
        "substitution": 1080,
        "advance": 900,
        "drive": 760,
        "pass": 620,
        "screen": 700,
        "handoff": 760,
        "shot": 700,
        "rebound": 760,
    }.get(event_type, 720)
    motion_duration = 340 + int(max_actor_distance * 80) + int(ball_distance * 22)
    return max(event_floor, base_duration_ms, min(1900, motion_duration))


def _limit_track_target(*, start: MotionPoint, desired_end: MotionPoint, max_distance: float) -> MotionPoint:
    dx = desired_end.x - start.x
    dy = desired_end.y - start.y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance <= max_distance or distance <= 0.001:
        return _clamp_point(desired_end)
    scale = max_distance / distance
    return _clamp_point(MotionPoint(start.x + (dx * scale), start.y + (dy * scale)))


def _point_distance(a: MotionPoint, b: MotionPoint) -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    return (dx * dx + dy * dy) ** 0.5


def _flip_possession_frame(point: MotionPoint) -> MotionPoint:
    return MotionPoint(point.x, 94.0 - point.y)


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
