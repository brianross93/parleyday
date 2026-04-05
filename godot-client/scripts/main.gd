extends Node2D

const COURT_WIDTH := 1000.0
const COURT_HEIGHT := 1880.0
const PLAYER_RADIUS := 24.0
const BALL_RADIUS := 10.0
const DATA_PATH := "res://data/latest_game.json"

const HOME_COLOR := Color("1f9d6d")
const AWAY_COLOR := Color("c44536")
const COURT_COLOR := Color("d58b42")
const LINE_COLOR := Color("f8ead8")
const DARK_LINE := Color("7b4f26")
const BALL_COLOR := Color("cb6b1f")

var camera_node: Camera2D

var using_payload := false
var payload: Dictionary = {}
var players: Array[Dictionary] = []
var player_lookup: Dictionary = {}
var choreography_segments: Array = []
var choreography_total_ms := 0.0
var timeline_ms := 0.0
var ball_position := Vector2.ZERO
var last_ball_position := Vector2.ZERO
var current_ball_visual: Dictionary = {"mode": "held"}
var current_actor_visuals: Dictionary = {}
var current_segment: Dictionary = {}
var current_beat: Dictionary = {}
var current_focus: Dictionary = {}
var current_camera_zoom := 1.0
var current_camera_target := Vector2.ZERO
var current_ball_target := Vector2.ZERO
var current_scorebug: Dictionary = {}
var time_accum := 0.0


func _ready() -> void:
	camera_node = $Camera2D
	_try_load_payload()
	if not using_payload:
		_setup_demo_state()
	queue_redraw()


func _process(delta: float) -> void:
	time_accum += delta
	if using_payload and choreography_segments.size() > 0:
		timeline_ms = fmod(timeline_ms + (delta * 1000.0), max(1.0, choreography_total_ms))
		_apply_payload_frame()
		_step_payload_motion(delta)
	else:
		_animate_demo_state()
	var basket_world: Vector2 = _court_to_world(Vector2(0.0, 5.25))
	var target_pos: Vector2 = ball_position.lerp(basket_world, 0.35) + Vector2(0.0, 60.0)
	current_camera_target = target_pos
	camera_node.position = camera_node.position.lerp(current_camera_target, min(1.0, delta * 3.2))
	var base_zoom: float = 1.8
	var target_zoom: float = clamp(base_zoom * current_camera_zoom, 1.6, 2.2)
	camera_node.zoom = camera_node.zoom.lerp(Vector2.ONE * target_zoom, min(1.0, delta * 3.8))
	queue_redraw()


func _draw() -> void:
	_draw_court()
	_draw_event_overlay()
	_draw_players()
	_draw_ball()
	_draw_scorebug()


func _try_load_payload() -> void:
	if not FileAccess.file_exists(DATA_PATH):
		return
	var file := FileAccess.open(DATA_PATH, FileAccess.READ)
	if file == null:
		return
	var parsed = JSON.parse_string(file.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		return
	payload = parsed
	var payload_players = payload.get("players", [])
	var payload_choreo = payload.get("choreography", {})
	if typeof(payload_players) != TYPE_ARRAY or typeof(payload_choreo) != TYPE_DICTIONARY:
		return
	players.clear()
	player_lookup.clear()
	for item in payload_players:
		if typeof(item) != TYPE_DICTIONARY:
			continue
		var pos := _court_to_world(_dict_point(item))
		var team_code := str(item.get("team_code", ""))
		var display := {
			"id": str(item.get("player_id", "")),
			"name": str(item.get("name", item.get("player_id", ""))),
			"team_code": team_code,
			"color": _team_color(team_code),
			"traits": item.get("traits", {}),
			"pos": pos,
			"target_pos": pos,
			"velocity": Vector2.ZERO,
			"prev_velocity": Vector2.ZERO,
			"facing": 0.0,
		}
		players.append(display)
		player_lookup[display["id"]] = display
	choreography_segments = payload_choreo.get("segments", [])
	choreography_total_ms = float(payload_choreo.get("total_duration_ms", 0.0))
	using_payload = choreography_segments.size() > 0 and players.size() > 0
	if using_payload:
		_seed_player_positions_from_first_segment()
		timeline_ms = 0.0
		_apply_payload_frame()


func _seed_player_positions_from_first_segment() -> void:
	if choreography_segments.is_empty():
		return
	var first_segment_variant: Variant = choreography_segments[0]
	if typeof(first_segment_variant) != TYPE_DICTIONARY:
		return
	var first_segment: Dictionary = first_segment_variant
	var tracks_variant: Variant = first_segment.get("actor_tracks", [])
	if typeof(tracks_variant) != TYPE_ARRAY:
		return
	for track_variant in tracks_variant:
		if typeof(track_variant) != TYPE_DICTIONARY:
			continue
		var track: Dictionary = track_variant
		var player_id: String = str(track.get("player_id", ""))
		if not player_lookup.has(player_id):
			continue
		var start_pos: Vector2 = _point_at_track(track, 0.0)
		player_lookup[player_id]["pos"] = start_pos
		player_lookup[player_id]["target_pos"] = start_pos
		player_lookup[player_id]["facing"] = _facing_angle(track, 0.0)
	for idx in range(players.size()):
		var player_id: String = str(players[idx].get("id", ""))
		if player_lookup.has(player_id):
			players[idx] = player_lookup[player_id]


func _apply_payload_frame() -> void:
	var segment: Dictionary = _segment_at_time(timeline_ms)
	if segment.is_empty():
		return
	current_segment = segment
	current_beat = _beat_for_segment(segment)
	current_scorebug = _scorebug_for_segment(segment)
	var duration_ms: float = max(1.0, float(segment.get("duration_ms", 1.0)))
	var progress: float = clamp((timeline_ms - float(segment.get("start_ms", 0.0))) / duration_ms, 0.0, 1.0)
	var ball_track: Dictionary = segment.get("ball_track", {})
	current_ball_target = _point_at_track(ball_track, progress)
	current_ball_visual = {
		"mode": str(ball_track.get("mode", "held")),
		"progress": progress,
		"arc_height": float(ball_track.get("arc_height", 0.0)),
		"dribble_count": int(ball_track.get("dribble_count", 0)),
		"start": _point_at_track(ball_track, 0.0),
		"end": _point_at_track(ball_track, 1.0),
		"owner_player_id": str(ball_track.get("owner_player_id", "")),
	}
	current_focus = _infer_segment_focus(segment, progress, ball_track)
	current_actor_visuals.clear()
	for track_variant in segment.get("actor_tracks", []):
		if typeof(track_variant) != TYPE_DICTIONARY:
			continue
		var track: Dictionary = track_variant
		var player_id := str(track.get("player_id", ""))
		if not player_lookup.has(player_id):
			continue
		var player_target: Vector2 = _point_at_track(track, progress)
		player_lookup[player_id]["target_pos"] = player_target
		current_actor_visuals[player_id] = {
			"locomotion": str(track.get("locomotion", "jog")),
			"phase": _locomotion_phase(str(track.get("locomotion", "jog")), str(segment.get("event_type", "")), progress, float(track.get("delay", 0.0))),
			"angle": _facing_angle(track, progress),
			"is_focus": player_id == str(current_focus.get("focus_player_id", "")),
			"is_receiver": player_id == str(current_focus.get("receiver_id", "")),
			"is_defender": player_id == str(current_focus.get("defender_id", "")),
			"has_ball": player_id == str(current_focus.get("ball_player_id", "")),
		}
	var camera_track: Dictionary = segment.get("camera_track", {})
	if not camera_track.is_empty():
		var eased: float = _ease(str(camera_track.get("easing", "ease_out")), progress)
		current_camera_zoom = lerp(float(camera_track.get("start_scale", 1.0)), float(camera_track.get("end_scale", 1.0)), eased)


func _step_payload_motion(delta: float) -> void:
	last_ball_position = ball_position
	for idx in range(players.size()):
		var player: Dictionary = players[idx]
		var player_id: String = str(player.get("id", ""))
		var visual: Dictionary = current_actor_visuals.get(player_id, {})
		var target_pos: Vector2 = player.get("target_pos", player.get("pos", Vector2.ZERO))
		if not bool(visual.get("is_focus", false)) and not bool(visual.get("is_receiver", false)) and not bool(visual.get("has_ball", false)):
			target_pos += _off_ball_jitter(player_id, time_accum, 4.0 if bool(visual.get("is_defender", false)) else 3.0)
		player["prev_velocity"] = player.get("velocity", Vector2.ZERO)
		_move_player_toward(player, target_pos, visual, delta)
		var target_facing: float = _resolve_facing_target(player, visual)
		var current_facing: float = float(player.get("facing", 0.0))
		var rotation_speed: float = _rotation_speed(player, visual)
		var diff: float = wrapf(target_facing - current_facing, -PI, PI)
		player["facing"] = current_facing + clamp(diff, -rotation_speed * delta, rotation_speed * delta)
		players[idx] = player
		player_lookup[player_id] = player
	_resolve_ball_motion(delta)


func _move_player_toward(player: Dictionary, target: Vector2, visual: Dictionary, delta: float) -> void:
	var pos: Vector2 = player.get("pos", Vector2.ZERO)
	var to_target: Vector2 = target - pos
	var distance: float = to_target.length()
	var velocity: Vector2 = player.get("velocity", Vector2.ZERO)
	if distance < 2.0:
		player["pos"] = target
		player["velocity"] = Vector2.ZERO
		return
	var direction: Vector2 = to_target.normalized()
	var profile: Dictionary = _movement_profile(player, str(visual.get("locomotion", "jog")))
	var max_speed: float = float(profile.get("max_speed", 220.0))
	var acceleration: float = float(profile.get("acceleration", 700.0))
	var deceleration: float = float(profile.get("deceleration", 560.0))
	var current_speed: float = velocity.length()
	var brake_distance: float = (current_speed * current_speed) / max(1.0, 2.0 * deceleration)
	if distance <= brake_distance:
		velocity = velocity.move_toward(Vector2.ZERO, deceleration * delta)
	else:
		velocity = velocity.move_toward(direction * max_speed, acceleration * delta)
	player["pos"] = pos + (velocity * delta)
	player["velocity"] = velocity


func _movement_profile(player: Dictionary, locomotion: String) -> Dictionary:
	var traits: Dictionary = player.get("traits", {})
	var speed_trait: float = float(traits.get("speed", 10.0))
	var burst_trait: float = float(traits.get("burst", 10.0))
	var stamina_trait: float = float(traits.get("stamina", 10.0))
	var size_trait: float = float(traits.get("size", 10.0))
	var max_speed: float = 180.0 + (speed_trait * 12.0) - (size_trait * 4.0)
	var acceleration: float = 500.0 + (burst_trait * 40.0) - (size_trait * 15.0)
	var deceleration: float = 400.0 + (burst_trait * 25.0) + (stamina_trait * 8.0)
	match locomotion:
		"sprint":
			max_speed *= 1.30
			acceleration *= 1.40
		"shuffle":
			max_speed *= 0.55
			acceleration *= 0.70
		"backpedal":
			max_speed *= 0.45
			acceleration *= 0.60
		"set":
			max_speed = 0.0
			acceleration = 0.0
			deceleration *= 1.2
	return {
		"max_speed": max(0.0, max_speed),
		"acceleration": max(0.0, acceleration),
		"deceleration": max(120.0, deceleration),
	}


func _off_ball_jitter(player_id: String, time_value: float, amplitude: float = 3.0) -> Vector2:
	var hash_seed: float = float(abs(player_id.hash()) % 1000) / 100.0
	return Vector2(
		sin(time_value * 1.8 + hash_seed) * amplitude,
		cos(time_value * 1.3 + (hash_seed * 0.7)) * amplitude * 0.6
	)


func _resolve_facing_target(player: Dictionary, visual: Dictionary) -> float:
	var velocity: Vector2 = player.get("velocity", Vector2.ZERO)
	if velocity.length() > 8.0:
		return velocity.angle()
	return float(visual.get("angle", float(player.get("facing", 0.0))))


func _rotation_speed(player: Dictionary, visual: Dictionary) -> float:
	var traits: Dictionary = player.get("traits", {})
	var containment: float = float(traits.get("containment", 10.0))
	var burst_trait: float = float(traits.get("burst", 10.0))
	var size_trait: float = float(traits.get("size", 10.0))
	var base_speed: float = 6.2 + ((containment - 10.0) * 0.14) + ((burst_trait - 10.0) * 0.10) - ((size_trait - 10.0) * 0.07)
	if str(visual.get("locomotion", "jog")) == "shuffle":
		base_speed += 0.7
	return clamp(base_speed, 3.2, 9.0)


func _resolve_ball_motion(delta: float) -> void:
	var mode: String = str(current_ball_visual.get("mode", "held"))
	var owner_id: String = str(current_ball_visual.get("owner_player_id", ""))
	if mode == "dribble" and owner_id != "" and player_lookup.has(owner_id):
		var owner: Dictionary = player_lookup[owner_id]
		var handler_pos: Vector2 = owner.get("pos", ball_position)
		var handler_vel: Vector2 = owner.get("velocity", Vector2.ZERO)
		var lead: Vector2 = handler_vel.normalized() * 12.0 if handler_vel.length() > 10.0 else Vector2(8.0, 0.0)
		var hand_offset: Vector2 = Vector2(10.0, 5.0)
		ball_position = handler_pos + lead + hand_offset
		return
	if mode == "held" and owner_id != "" and player_lookup.has(owner_id):
		var owner: Dictionary = player_lookup[owner_id]
		var handler_pos: Vector2 = owner.get("pos", ball_position)
		var handler_vel: Vector2 = owner.get("velocity", Vector2.ZERO)
		var hold_offset: Vector2 = handler_vel.normalized() * 8.0 if handler_vel.length() > 6.0 else Vector2(8.0, -2.0)
		ball_position = handler_pos + hold_offset
		return
	ball_position = ball_position.lerp(current_ball_target, min(1.0, delta * 10.0))


func _segment_at_time(ms: float) -> Dictionary:
	for variant in choreography_segments:
		if typeof(variant) != TYPE_DICTIONARY:
			continue
		var segment: Dictionary = variant
		if ms <= float(segment.get("end_ms", 0.0)):
			return segment
	return choreography_segments[-1] if choreography_segments.size() > 0 else {}


func _beat_for_segment(segment: Dictionary) -> Dictionary:
	if not payload.has("match"):
		return {}
	var match_info: Dictionary = payload["match"]
	var beats_variant: Variant = match_info.get("beats", [])
	if typeof(beats_variant) != TYPE_ARRAY:
		return {}
	var beats: Array = beats_variant
	var idx: int = int(segment.get("beat_index", -1))
	if idx < 0 or idx >= beats.size():
		return {}
	var beat_variant: Variant = beats[idx]
	return beat_variant if typeof(beat_variant) == TYPE_DICTIONARY else {}


func _scorebug_for_segment(segment: Dictionary) -> Dictionary:
	var beat: Dictionary = _beat_for_segment(segment)
	var match_info: Dictionary = payload.get("match", {})
	var initial_scoreboard: Dictionary = match_info.get("initial_scoreboard", {})
	return {
		"home_team": str(match_info.get("home_team", "")),
		"away_team": str(match_info.get("away_team", "")),
		"home_team_name": str(match_info.get("home_team_name", str(match_info.get("home_team", "")))),
		"away_team_name": str(match_info.get("away_team_name", str(match_info.get("away_team", "")))),
		"period": int(beat.get("period", int(initial_scoreboard.get("period", 1)))),
		"clock_display": str(beat.get("clock_display", str(initial_scoreboard.get("clock_display", "12:00")))),
		"shot_clock": float(beat.get("shot_clock", float(initial_scoreboard.get("shot_clock", 24.0)))),
		"offense_team_code": str(beat.get("offense_team_code", "")),
		"offense_score": int(beat.get("offense_score", int(initial_scoreboard.get("offense_score", 0)))),
		"defense_score": int(beat.get("defense_score", int(initial_scoreboard.get("defense_score", 0)))),
		"label": str(beat.get("label", segment.get("label", ""))),
	}


func _point_at_track(track: Dictionary, progress: float) -> Vector2:
	var start: Vector2 = _dict_point(track.get("start", {}))
	var finish: Vector2 = _dict_point(track.get("end", {}))
	var delay: float = clamp(float(track.get("delay", 0.0)), 0.0, 0.35)
	var tempo: float = clamp(float(track.get("tempo", 1.0)), 0.5, 1.5)
	var delayed: float = 0.0 if progress <= delay else min(1.0, (progress - delay) / max(0.001, 1.0 - delay))
	var eased: float = min(1.0, _ease(str(track.get("easing", "ease_out")), delayed) * tempo)
	if track.has("control") and typeof(track["control"]) == TYPE_DICTIONARY:
		var control: Vector2 = _dict_point(track["control"])
		var u: float = 1.0 - eased
		return _court_to_world(Vector2(
			(u * u * start.x) + (2.0 * u * eased * control.x) + (eased * eased * finish.x),
			(u * u * start.y) + (2.0 * u * eased * control.y) + (eased * eased * finish.y)
		))
	return _court_to_world(start.lerp(finish, eased))


func _point_at_track_direct(track: Dictionary, progress: float, apply_timing: bool = true) -> Vector2:
	var start: Vector2 = _dict_point(track.get("start", {}))
	var finish: Vector2 = _dict_point(track.get("end", {}))
	var eased: float = progress
	if apply_timing:
		var delay: float = clamp(float(track.get("delay", 0.0)), 0.0, 0.35)
		var tempo: float = clamp(float(track.get("tempo", 1.0)), 0.5, 1.5)
		var delayed: float = 0.0 if progress <= delay else min(1.0, (progress - delay) / max(0.001, 1.0 - delay))
		eased = min(1.0, _ease(str(track.get("easing", "ease_out")), delayed) * tempo)
	if track.has("control") and typeof(track["control"]) == TYPE_DICTIONARY:
		var control: Vector2 = _dict_point(track["control"])
		var u: float = 1.0 - eased
		return _court_to_world(Vector2(
			(u * u * start.x) + (2.0 * u * eased * control.x) + (eased * eased * finish.x),
			(u * u * start.y) + (2.0 * u * eased * control.y) + (eased * eased * finish.y)
		))
	return _court_to_world(start.lerp(finish, eased))


func _dict_point(value: Variant) -> Vector2:
	if typeof(value) != TYPE_DICTIONARY:
		return Vector2.ZERO
	var point: Dictionary = value
	return Vector2(float(point.get("x", 0.0)), float(point.get("y", 0.0)))


func _court_to_world(point: Vector2) -> Vector2:
	var x: float = (point.x / 25.0) * (COURT_WIDTH * 0.5)
	var y: float = ((point.y - 47.0) / 47.0) * (COURT_HEIGHT * 0.5)
	return Vector2(x, y)


func _ease(ease_name: String, value: float) -> float:
	var clamped: float = clamp(value, 0.0, 1.0)
	if ease_name == "ease_in_out":
		return 4.0 * clamped * clamped * clamped if clamped < 0.5 else 1.0 - pow(-2.0 * clamped + 2.0, 3.0) / 2.0
	if ease_name == "ease_out":
		return 1.0 - pow(1.0 - clamped, 3.0)
	return clamped


func _facing_angle(track: Dictionary, progress: float) -> float:
	var current_point: Vector2 = _point_at_track_direct(track, progress)
	var next_point: Vector2 = _point_at_track_direct(track, min(1.0, progress + 0.08))
	var delta: Vector2 = next_point - current_point
	if delta.length() < 2.0:
		return 0.0
	return delta.angle()


func _locomotion_phase(locomotion: String, event_type: String, progress: float, delay: float) -> String:
	var delayed: float = 0.0 if progress <= delay else min(1.0, (progress - delay) / max(0.001, 1.0 - delay))
	if locomotion == "set":
		return "ready"
	if locomotion == "sprint":
		if delayed < 0.18:
			return "load"
		if delayed < 0.72:
			return "burst"
		return "recover"
	if locomotion == "shuffle":
		if delayed < 0.28:
			return "ready"
		return "burst" if event_type in ["drive", "pass"] and delayed < 0.68 else "recover"
	if locomotion == "backpedal":
		if delayed < 0.34:
			return "ready"
		return "recover" if delayed < 0.72 else "burst"
	if delayed < 0.2:
		return "ready"
	if event_type == "drive" and delayed < 0.7:
		return "burst"
	return "recover"


func _infer_segment_focus(segment: Dictionary, progress: float, ball_track: Dictionary) -> Dictionary:
	var owner_id: String = str(ball_track.get("owner_player_id", ""))
	var event_type: String = str(segment.get("event_type", ""))
	var owner_track: Dictionary = _track_for_player(segment, owner_id)
	var owner_team: String = str(owner_track.get("team_code", ""))
	var owner_pos: Vector2 = _point_at_track_direct(owner_track, progress) if not owner_track.is_empty() else ball_position
	var receiver_id: String = ""
	if event_type == "pass" or (event_type == "shot" and str(ball_track.get("mode", "")) == "air"):
		receiver_id = _nearest_teammate_to_point(segment, owner_team, owner_id, _dict_point(ball_track.get("end", {})))
	var defender_id: String = _nearest_opponent_to_world_point(segment, owner_team, owner_pos, progress)
	var focus_id: String = owner_id
	if event_type == "pass" and receiver_id != "":
		focus_id = receiver_id if progress > 0.62 else owner_id
	elif event_type == "shot":
		focus_id = owner_id
	elif event_type == "screen":
		var screener_id: String = _nearest_teammate_to_point(segment, owner_team, owner_id, _dict_point(ball_track.get("start", {})))
		if screener_id != "":
			focus_id = screener_id if progress < 0.38 else owner_id
	return {
		"ball_player_id": owner_id,
		"focus_player_id": focus_id,
		"receiver_id": receiver_id,
		"defender_id": defender_id,
	}


func _track_for_player(segment: Dictionary, player_id: String) -> Dictionary:
	if player_id == "":
		return {}
	var tracks_variant: Variant = segment.get("actor_tracks", [])
	if typeof(tracks_variant) != TYPE_ARRAY:
		return {}
	for track_variant in tracks_variant:
		if typeof(track_variant) != TYPE_DICTIONARY:
			continue
		var track: Dictionary = track_variant
		if str(track.get("player_id", "")) == player_id:
			return track
	return {}


func _nearest_teammate_to_point(segment: Dictionary, team_code: String, exclude_id: String, court_point: Vector2) -> String:
	var best_id: String = ""
	var best_distance: float = INF
	var tracks_variant: Variant = segment.get("actor_tracks", [])
	if typeof(tracks_variant) != TYPE_ARRAY:
		return ""
	for track_variant in tracks_variant:
		if typeof(track_variant) != TYPE_DICTIONARY:
			continue
		var track: Dictionary = track_variant
		var player_id: String = str(track.get("player_id", ""))
		if player_id == "" or player_id == exclude_id or str(track.get("team_code", "")) != team_code:
			continue
		var end_point: Vector2 = _dict_point(track.get("end", {}))
		var distance_to_target: float = end_point.distance_to(court_point)
		if distance_to_target < best_distance:
			best_distance = distance_to_target
			best_id = player_id
	return best_id


func _nearest_opponent_to_world_point(segment: Dictionary, offense_team: String, world_point: Vector2, progress: float) -> String:
	var best_id: String = ""
	var best_distance: float = INF
	var tracks_variant: Variant = segment.get("actor_tracks", [])
	if typeof(tracks_variant) != TYPE_ARRAY:
		return ""
	for track_variant in tracks_variant:
		if typeof(track_variant) != TYPE_DICTIONARY:
			continue
		var track: Dictionary = track_variant
		if str(track.get("team_code", "")) == offense_team:
			continue
		var player_id: String = str(track.get("player_id", ""))
		if player_id == "":
			continue
		var point: Vector2 = _point_at_track_direct(track, progress)
		var distance_to_target: float = point.distance_to(world_point)
		if distance_to_target < best_distance:
			best_distance = distance_to_target
			best_id = player_id
	return best_id


func _display_label(player_name: String) -> String:
	if player_name == "":
		return ""
	var parts: PackedStringArray = player_name.split(" ", false)
	if parts.size() >= 2:
		return parts[0].substr(0, 1) + parts[parts.size() - 1].substr(0, 1)
	return player_name.substr(0, min(2, player_name.length())).to_upper()


func _team_color(team_code: String) -> Color:
	if payload.has("match"):
		var match_info: Dictionary = payload["match"]
		if team_code == str(match_info.get("home_team", "")):
			return HOME_COLOR
		if team_code == str(match_info.get("away_team", "")):
			return AWAY_COLOR
	return HOME_COLOR if team_code.begins_with("HOM") else AWAY_COLOR


func _setup_demo_state() -> void:
	var center_x: float = 0.0
	var backcourt_y: float = 730.0
	var frontcourt_y: float = -260.0

	players = [
		{"id": "away_pg", "name": "PG", "team_code": "AWY", "color": AWAY_COLOR, "pos": Vector2(center_x - 120.0, backcourt_y), "role": "handler"},
		{"id": "away_sg", "name": "SG", "team_code": "AWY", "color": AWAY_COLOR, "pos": Vector2(center_x - 260.0, backcourt_y - 120.0), "role": "wing"},
		{"id": "away_sf", "name": "SF", "team_code": "AWY", "color": AWAY_COLOR, "pos": Vector2(center_x + 120.0, backcourt_y - 150.0), "role": "wing"},
		{"id": "away_pf", "name": "PF", "team_code": "AWY", "color": AWAY_COLOR, "pos": Vector2(center_x - 60.0, frontcourt_y + 180.0), "role": "big"},
		{"id": "away_c", "name": "C", "team_code": "AWY", "color": AWAY_COLOR, "pos": Vector2(center_x + 90.0, frontcourt_y + 120.0), "role": "big"},
		{"id": "home_pg", "name": "PG", "team_code": "HOM", "color": HOME_COLOR, "pos": Vector2(center_x - 40.0, backcourt_y - 200.0), "role": "defender"},
		{"id": "home_sg", "name": "SG", "team_code": "HOM", "color": HOME_COLOR, "pos": Vector2(center_x - 210.0, backcourt_y - 230.0), "role": "defender"},
		{"id": "home_sf", "name": "SF", "team_code": "HOM", "color": HOME_COLOR, "pos": Vector2(center_x + 150.0, backcourt_y - 260.0), "role": "defender"},
		{"id": "home_pf", "name": "PF", "team_code": "HOM", "color": HOME_COLOR, "pos": Vector2(center_x - 80.0, frontcourt_y + 40.0), "role": "helper"},
		{"id": "home_c", "name": "C", "team_code": "HOM", "color": HOME_COLOR, "pos": Vector2(center_x + 20.0, frontcourt_y + 10.0), "role": "rim"},
	]
	ball_position = players[0]["pos"] + Vector2(20.0, -8.0)


func _animate_demo_state() -> void:
	var phase: float = fmod(time_accum, 6.0)
	for idx in players.size():
		var player: Dictionary = players[idx]
		var start_pos: Vector2 = player["pos"]
		var role: String = str(player.get("role", ""))
		var wave: float = sin(time_accum * 1.4 + float(idx)) * 6.0
		if role == "handler":
			player["pos"] = Vector2(-140.0 + phase * 55.0, 740.0 - min(phase, 2.6) * 120.0)
		elif role == "wing":
			player["pos"] = start_pos + Vector2(sin(time_accum * 0.8 + idx) * 10.0, wave)
		elif role == "big":
			player["pos"] = start_pos + Vector2(cos(time_accum * 0.7 + idx) * 8.0, sin(time_accum * 0.9 + idx) * 12.0)
		elif role == "defender":
			player["pos"] = start_pos + Vector2(sin(time_accum * 1.1 + idx) * 12.0, -abs(cos(time_accum * 0.9 + idx) * 14.0))
		elif role == "helper":
			player["pos"] = start_pos + Vector2(cos(time_accum * 0.9) * 10.0, sin(time_accum * 1.3) * 10.0)
		elif role == "rim":
			player["pos"] = start_pos + Vector2(sin(time_accum * 0.6) * 6.0, 0.0)
		players[idx] = player
	var handler_pos: Vector2 = players[0]["pos"]
	var dribble: float = float(abs(sin(time_accum * 6.0)) * 18.0)
	ball_position = handler_pos + Vector2(20.0, -dribble)


func _draw_court() -> void:
	var rect := Rect2(Vector2(-COURT_WIDTH * 0.5, -COURT_HEIGHT * 0.5), Vector2(COURT_WIDTH, COURT_HEIGHT))
	draw_rect(rect, COURT_COLOR, true)
	draw_rect(rect, DARK_LINE, false, 6.0)
	_draw_court_line(Vector2(-25.0, 47.0), Vector2(25.0, 47.0), LINE_COLOR, 6.0)
	_draw_court_circle(Vector2(0.0, 47.0), 6.0, LINE_COLOR, 6.0)
	_draw_half_court_geometry(true)
	_draw_half_court_geometry(false)


func _draw_half_court_geometry(is_top: bool) -> void:
	var rim_y: float = 5.25 if is_top else 94.0 - 5.25
	var backboard_y: float = 4.0 if is_top else 94.0 - 4.0
	var ft_y: float = 19.0 if is_top else 94.0 - 19.0
	var lane_near_y: float = 0.0 if is_top else 94.0
	var lane_far_y: float = 19.0 if is_top else 94.0 - 19.0
	var corner_end_y: float = 14.0 if is_top else 94.0 - 14.0
	var restricted_start: float = 0.0 if is_top else PI
	var restricted_end: float = PI if is_top else TAU
	var three_start: float = deg_to_rad(22.0) if is_top else deg_to_rad(202.0)
	var three_end: float = deg_to_rad(158.0) if is_top else deg_to_rad(338.0)

	_draw_court_line(Vector2(-3.0, backboard_y), Vector2(3.0, backboard_y), DARK_LINE, 6.0)
	_draw_court_circle(Vector2(0.0, rim_y), 0.75, LINE_COLOR, 4.0)
	_draw_court_arc(Vector2(0.0, rim_y), 4.0, restricted_start, restricted_end, LINE_COLOR, 4.0)

	var lane_rect := Rect2(
		_court_to_world(Vector2(-8.0, min(lane_near_y, lane_far_y))),
		_court_to_world(Vector2(8.0, max(lane_near_y, lane_far_y))) - _court_to_world(Vector2(-8.0, min(lane_near_y, lane_far_y)))
	)
	draw_rect(lane_rect, Color(0, 0, 0, 0), false, 6.0)
	_draw_court_arc(Vector2(0.0, ft_y), 6.0, 0.0, TAU, LINE_COLOR, 6.0)

	_draw_court_line(Vector2(-22.0, lane_near_y), Vector2(-22.0, corner_end_y), LINE_COLOR, 6.0)
	_draw_court_line(Vector2(22.0, lane_near_y), Vector2(22.0, corner_end_y), LINE_COLOR, 6.0)
	_draw_court_arc(Vector2(0.0, rim_y), 23.75, three_start, three_end, LINE_COLOR, 6.0)


func _draw_court_line(start_court: Vector2, end_court: Vector2, color: Color, width: float) -> void:
	draw_line(_court_to_world(start_court), _court_to_world(end_court), color, width)


func _draw_court_circle(center_court: Vector2, radius_feet: float, color: Color, width: float) -> void:
	draw_arc(_court_to_world(center_court), radius_feet * 20.0, 0.0, TAU, 64, color, width)


func _draw_court_arc(center_court: Vector2, radius_feet: float, start_angle: float, end_angle: float, color: Color, width: float) -> void:
	draw_arc(_court_to_world(center_court), radius_feet * 20.0, start_angle, end_angle, 64, color, width)


func _draw_players() -> void:
	for player in players:
		var pos: Vector2 = player["pos"]
		var color: Color = player["color"]
		var visual: Dictionary = current_actor_visuals.get(player.get("id", ""), {})
		var locomotion: String = str(visual.get("locomotion", "jog"))
		var phase: String = str(visual.get("phase", "ready"))
		var angle: float = float(player.get("facing", float(visual.get("angle", 0.0))))
		var is_focus: bool = bool(visual.get("is_focus", false))
		var is_receiver: bool = bool(visual.get("is_receiver", false))
		var is_defender: bool = bool(visual.get("is_defender", false))
		var has_ball: bool = bool(visual.get("has_ball", false))
		var velocity: Vector2 = player.get("velocity", Vector2.ZERO)
		var prev_velocity: Vector2 = player.get("prev_velocity", Vector2.ZERO)
		var accel_vec: Vector2 = velocity - prev_velocity
		var lean_angle: float = clamp(accel_vec.x * 0.0008, -0.12, 0.12)
		var radius_x: float = PLAYER_RADIUS
		var radius_y: float = PLAYER_RADIUS
		if locomotion == "sprint" or phase == "burst":
			radius_x *= 1.06
			radius_y *= 0.94
		elif locomotion == "backpedal":
			radius_x *= 0.96
			radius_y *= 1.02
		elif locomotion == "shuffle":
			radius_x *= 1.02
			radius_y *= 0.98
		if is_focus or has_ball:
			radius_x *= 1.08
			radius_y *= 1.08
		elif is_receiver or is_defender:
			radius_x *= 1.04
			radius_y *= 1.04

		var draw_radius: float = max(radius_x, radius_y)
		var glow_color: Color = Color(1, 1, 1, 0.0)
		if has_ball:
			glow_color = Color(1.0, 0.92, 0.52, 0.22)
		elif is_focus:
			glow_color = Color(1.0, 1.0, 1.0, 0.14)
		elif is_receiver:
			glow_color = Color(1.0, 0.88, 0.42, 0.16)
		elif is_defender:
			glow_color = Color(0.72, 0.9, 1.0, 0.14)

		draw_set_transform(pos, angle + lean_angle, Vector2.ONE)
		if glow_color.a > 0.0:
			draw_circle(Vector2.ZERO, draw_radius + 8.0, glow_color)
		draw_circle(Vector2.ZERO, draw_radius, color)
		draw_arc(Vector2.ZERO, draw_radius, 0.0, TAU, 24, Color.WHITE, 3.0 if not is_defender else 2.0)
		var nose := Vector2(cos(angle), sin(angle)) * (draw_radius - 6.0)
		draw_circle(nose, 5.0, Color(1, 1, 1, 0.92))
		var shadow_offset := Vector2(0.0, 10.0)
		draw_circle(shadow_offset, draw_radius * 0.78, Color(0, 0, 0, 0.10))
		if has_ball:
			draw_arc(Vector2.ZERO, draw_radius + 5.0, 0.0, TAU, 24, Color(1.0, 0.92, 0.56, 0.85), 2.0)
		draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)
		var label: String = _display_label(str(player["name"]))
		draw_string(ThemeDB.fallback_font, pos + Vector2(-11.0, 6.0), label, HORIZONTAL_ALIGNMENT_LEFT, -1, 19, Color.WHITE)


func _draw_event_overlay() -> void:
	if current_segment.is_empty():
		return
	var event_type: String = str(current_segment.get("event_type", ""))
	if event_type == "pass":
		_draw_pass_overlay()
	elif event_type == "drive":
		_draw_drive_overlay()
	elif event_type == "shot":
		_draw_shot_overlay()
	elif event_type == "possession_change":
		_draw_possession_change_overlay()


func _draw_pass_overlay() -> void:
	var start: Vector2 = Vector2(current_ball_visual.get("start", ball_position))
	var finish: Vector2 = Vector2(current_ball_visual.get("end", ball_position))
	draw_line(start, finish, Color(1.0, 0.95, 0.72, 0.34), 6.0)
	draw_circle(finish, 16.0, Color(1.0, 0.9, 0.4, 0.12))
	var defender_id: String = str(current_focus.get("defender_id", ""))
	if defender_id != "" and player_lookup.has(defender_id):
		draw_line(player_lookup[defender_id]["pos"], finish, Color(0.72, 0.88, 1.0, 0.18), 3.0)


func _draw_drive_overlay() -> void:
	var owner_id: String = str(current_focus.get("ball_player_id", ""))
	var start: Vector2 = Vector2(current_ball_visual.get("start", ball_position))
	if owner_id != "" and player_lookup.has(owner_id):
		start = player_lookup[owner_id]["pos"]
	var finish: Vector2 = ball_position
	draw_line(start, finish, Color(0.98, 0.82, 0.35, 0.22), 14.0)
	draw_line(start, finish, Color(1.0, 0.92, 0.62, 0.46), 4.0)
	var defender_id: String = str(current_focus.get("defender_id", ""))
	if defender_id != "" and player_lookup.has(defender_id):
		draw_line(player_lookup[defender_id]["pos"], finish, Color(0.75, 0.9, 1.0, 0.2), 4.0)


func _draw_shot_overlay() -> void:
	var start: Vector2 = Vector2(current_ball_visual.get("start", ball_position))
	var finish: Vector2 = Vector2(current_ball_visual.get("end", ball_position))
	var peak: float = 80.0 + float(current_ball_visual.get("arc_height", 0.0)) * 4.0
	var control := Vector2((start.x + finish.x) * 0.5, min(start.y, finish.y) - peak)
	var points := PackedVector2Array()
	for i in range(17):
		var t: float = float(i) / 16.0
		var u: float = 1.0 - t
		points.append(Vector2(
			(u * u * start.x) + (2.0 * u * t * control.x) + (t * t * finish.x),
			(u * u * start.y) + (2.0 * u * t * control.y) + (t * t * finish.y)
		))
	draw_polyline(points, Color(1.0, 0.96, 0.78, 0.55), 4.0)
	draw_circle(finish, 18.0, Color(1.0, 0.95, 0.6, 0.14))
	var defender_id: String = str(current_focus.get("defender_id", ""))
	if defender_id != "" and player_lookup.has(defender_id):
		draw_line(player_lookup[defender_id]["pos"], finish, Color(0.72, 0.88, 1.0, 0.18), 3.0)


func _draw_possession_change_overlay() -> void:
	var start: Vector2 = Vector2(current_ball_visual.get("start", ball_position))
	var finish: Vector2 = Vector2(current_ball_visual.get("end", ball_position))
	draw_line(start, finish, Color(0.72, 0.88, 1.0, 0.24), 5.0)
	draw_circle(start, 16.0, Color(0.72, 0.88, 1.0, 0.10))
	draw_circle(finish, 14.0, Color(0.72, 0.88, 1.0, 0.08))


func _draw_ball() -> void:
	var mode: String = str(current_ball_visual.get("mode", "held"))
	var draw_radius: float = BALL_RADIUS
	if mode == "dribble":
		var bounce_progress: float = float(current_ball_visual.get("progress", 0.0))
		var count: int = int(current_ball_visual.get("dribble_count", 1))
		var bounce: float = abs(sin(PI * float(max(1, count)) * bounce_progress))
		draw_radius = BALL_RADIUS * (0.92 + (bounce * 0.08))
	if mode == "air":
		for idx in range(3):
			var t: float = float(idx + 1) / 4.0
			var trail: Vector2 = ball_position.lerp(last_ball_position, t)
			draw_circle(trail, BALL_RADIUS * (1.0 - t * 0.25), Color(0.80, 0.42, 0.12, 0.10 + (0.12 * (1.0 - t))))
	draw_circle(ball_position + Vector2(-6.0, 12.0), BALL_RADIUS * 0.9, Color(0, 0, 0, 0.12))
	draw_circle(ball_position, draw_radius, BALL_COLOR)
	draw_arc(ball_position, draw_radius, 0.0, TAU, 24, Color("7b3f15"), 2.0)


func _draw_scorebug() -> void:
	if current_scorebug.is_empty():
		return
	var viewport_size: Vector2 = get_viewport_rect().size
	var zoom: Vector2 = camera_node.zoom
	var top_left: Vector2 = camera_node.position - Vector2((viewport_size.x * zoom.x) * 0.5, (viewport_size.y * zoom.y) * 0.5)
	var panel_pos: Vector2 = top_left + Vector2(26.0 * zoom.x, 24.0 * zoom.y)
	var panel_size: Vector2 = Vector2(312.0 * zoom.x, 98.0 * zoom.y)
	var away_team: String = str(current_scorebug.get("away_team", "AWY"))
	var home_team: String = str(current_scorebug.get("home_team", "HOM"))
	var offense_team: String = str(current_scorebug.get("offense_team_code", ""))
	var offense_score: int = int(current_scorebug.get("offense_score", 0))
	var defense_score: int = int(current_scorebug.get("defense_score", 0))
	var away_score: int = offense_score if offense_team == away_team else defense_score
	var home_score: int = offense_score if offense_team == home_team else defense_score
	var panel_rect := Rect2(panel_pos, panel_size)
	var accent_rect := Rect2(panel_pos + Vector2(panel_size.x - (88.0 * zoom.x), 0.0), Vector2(88.0 * zoom.x, panel_size.y))
	draw_rect(panel_rect, Color(0.05, 0.08, 0.12, 0.88), true)
	draw_rect(panel_rect, Color(1, 1, 1, 0.08), false, 2.0 * zoom.x)
	draw_rect(accent_rect, Color(0.11, 0.15, 0.21, 0.92), true)
	var font := ThemeDB.fallback_font
	var title_size: int = int(24.0 * zoom.x)
	var meta_size: int = int(18.0 * zoom.x)
	var label_size: int = int(15.0 * zoom.x)
	draw_string(font, panel_pos + Vector2(18.0 * zoom.x, 30.0 * zoom.y), away_team, HORIZONTAL_ALIGNMENT_LEFT, -1, title_size, Color.WHITE)
	draw_string(font, panel_pos + Vector2(18.0 * zoom.x, 60.0 * zoom.y), home_team, HORIZONTAL_ALIGNMENT_LEFT, -1, title_size, Color.WHITE)
	draw_string(font, panel_pos + Vector2(128.0 * zoom.x, 30.0 * zoom.y), str(away_score), HORIZONTAL_ALIGNMENT_LEFT, -1, title_size, Color.WHITE)
	draw_string(font, panel_pos + Vector2(128.0 * zoom.x, 60.0 * zoom.y), str(home_score), HORIZONTAL_ALIGNMENT_LEFT, -1, title_size, Color.WHITE)
	draw_string(font, panel_pos + Vector2(210.0 * zoom.x, 30.0 * zoom.y), "Q" + str(current_scorebug.get("period", 1)), HORIZONTAL_ALIGNMENT_LEFT, -1, meta_size, Color(0.86, 0.9, 0.96, 0.95))
	draw_string(font, panel_pos + Vector2(210.0 * zoom.x, 56.0 * zoom.y), str(current_scorebug.get("clock_display", "12:00")), HORIZONTAL_ALIGNMENT_LEFT, -1, meta_size, Color.WHITE)
	draw_string(font, panel_pos + Vector2(210.0 * zoom.x, 80.0 * zoom.y), "SHOT " + str(int(round(float(current_scorebug.get("shot_clock", 24.0))))), HORIZONTAL_ALIGNMENT_LEFT, -1, label_size, Color(1.0, 0.88, 0.46, 0.96))
	draw_string(font, panel_pos + Vector2(panel_size.x - (78.0 * zoom.x), 32.0 * zoom.y), "LIVE", HORIZONTAL_ALIGNMENT_LEFT, -1, label_size, Color(1.0, 0.78, 0.48, 0.96))
	draw_string(font, panel_pos + Vector2(panel_size.x - (78.0 * zoom.x), 62.0 * zoom.y), str(current_scorebug.get("label", "")), HORIZONTAL_ALIGNMENT_LEFT, -1, label_size, Color(0.88, 0.92, 0.98, 0.92))
