from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from dfs_optimizer import solve_dfs_lineups
from dfs_ingest import DraftKingsPlayer, DraftKingsSlate
from nba_matchup_features import load_nba_matchup_features
from player_name_utils import canonicalize_player_name, dfs_name_key
from refresh_slate import fetch_nba_game_contexts, fetch_nba_injury_context_details, fetch_nba_team_player_profiles
from quantum_parlay_oracle import (
    build_live_nba_team_context,
    expected_nba_points,
    load_game_context_snapshot,
    load_nba_matchup_profile_snapshot,
    load_team_form_snapshot,
    nba_availability_status_lookup,
    nba_roster_offense_penalty,
)


@dataclass(frozen=True)
class DraftKingsNBAProjection:
    player_id: str
    name: str
    team: str
    opponent: str
    salary: int
    positions: tuple[str, ...]
    roster_positions: tuple[str, ...]
    game: str
    median_fpts: float
    ceiling_fpts: float
    floor_fpts: float
    volatility: float
    projection_confidence: float
    minutes: float
    points: float
    rebounds: float
    assists: float
    availability_status: str
    availability_source: str
    recent_games_sample: float = 0.0
    recent_minutes_avg: float = 0.0
    participation_rate: float = 0.0
    role_stability: float = 0.0
    recent_fpts_avg: float = 0.0
    recent_fpts_weighted: float = 0.0
    recent_form_delta: float = 0.0


@dataclass(frozen=True)
class DraftKingsLineup:
    players: tuple[DraftKingsNBAProjection, ...]
    salary_used: int
    median_fpts: float
    ceiling_fpts: float
    floor_fpts: float
    average_confidence: float
    unknown_count: int


def draftkings_nba_fpts(points: float, rebounds: float, assists: float) -> float:
    return float(points + (rebounds * 1.25) + (assists * 1.5))


def build_nba_dk_projections(date_str: str, slate: DraftKingsSlate) -> list[DraftKingsNBAProjection]:
    if slate.sport != "nba":
        return []
    team_form = load_team_form_snapshot(date_str, "nba")
    game_cache: dict[str, dict[str, DraftKingsNBAProjection]] = {}
    live_profile_lookup = _load_live_nba_profiles(date_str, slate)
    live_profile_statuses = {
        key: (_profile_injury_status(profile), "live_profile")
        for key, profile in live_profile_lookup.items()
    }
    injury_context_by_game = _load_fallback_nba_injury_context(date_str, slate)
    projections: list[DraftKingsNBAProjection] = []
    for player in slate.players:
        game = str(player.game or "")
        if not game or "@" not in game:
            continue
        if game not in game_cache:
            game_cache[game] = _build_game_projection_lookup(
                date_str,
                game,
                team_form,
                live_profile_lookup=live_profile_lookup,
            )
        lookup = game_cache.get(game, {})
        projected = lookup.get(dfs_name_key(player.name))
        if projected is None:
            fallback_status = _fallback_status_from_sources(
                player,
                live_profile_statuses=live_profile_statuses,
                injury_context=injury_context_by_game.get(game),
            )
            projected = _fallback_projection(
                player,
                fallback_status=fallback_status,
                live_profile=live_profile_lookup.get((player.team, dfs_name_key(player.name))),
            )
        if projected.availability_status == "out":
            continue
        projections.append(projected)
    return sorted(projections, key=lambda item: (item.median_fpts, -item.salary), reverse=True)


def _build_game_projection_lookup(
    date_str: str,
    game: str,
    team_form: dict[str, dict],
    *,
    live_profile_lookup: dict[tuple[str, str], dict[str, object]],
) -> dict[str, DraftKingsNBAProjection]:
    away_code, home_code = game.split("@", 1)
    matchup_features = load_nba_matchup_features(date_str, (away_code, home_code))
    game_context = load_game_context_snapshot(date_str, "nba", game) or {}
    payload = load_nba_matchup_profile_snapshot(date_str, game) or {}
    availability = game_context.get("availability") or {}
    away_mean, home_mean = expected_nba_points(away_code, home_code, team_form, game_context)
    away_mean = max(
        90.0,
        away_mean - nba_roster_offense_penalty(payload.get("away_profiles", []), availability.get("away", [])),
    )
    home_mean = max(
        90.0,
        home_mean - nba_roster_offense_penalty(payload.get("home_profiles", []), availability.get("home", [])),
    )
    away_context = build_live_nba_team_context(
        away_code,
        payload.get("away_profiles", []),
        away_mean,
        availability.get("away", []),
    )
    home_context = build_live_nba_team_context(
        home_code,
        payload.get("home_profiles", []),
        home_mean,
        availability.get("home", []),
    )
    results: dict[str, DraftKingsNBAProjection] = {}
    for team_context, opponent in ((away_context, home_code), (home_context, away_code)):
        if team_context is None:
            continue
        side_key = "away" if team_context.code == away_code else "home"
        side_submitted = _availability_side_submitted(availability, side_key)
        confidence = 0.68 if side_submitted else 0.54
        side_status_lookup = nba_availability_status_lookup(availability.get(side_key, []))
        raw_profile_lookup = {
            dfs_name_key(str(item.get("name") or "")): item
            for item in (payload.get(f"{side_key}_profiles", []) or [])
            if str(item.get("name") or "").strip()
        }
        for profile in team_context.players:
            raw_profile = raw_profile_lookup.get(dfs_name_key(profile.name), {})
            live_profile = live_profile_lookup.get((team_context.code, dfs_name_key(profile.name)), {})
            status = side_status_lookup.get(
                canonicalize_player_name(profile.name),
                _profile_injury_status(live_profile or raw_profile or profile),
            )
            adjusted = _apply_availability_discount(
                float(profile.minutes),
                float(profile.points),
                float(profile.rebounds),
                float(profile.assists),
                status,
            )
            adjusted = _apply_matchup_feature_adjustments(
                adjusted,
                team_code=team_context.code,
                opponent_code=opponent,
                position=str(raw_profile.get("position") or getattr(profile, "position", "") or ""),
                team_features=matchup_features.get(team_context.code, {}),
                opponent_features=matchup_features.get(opponent, {}),
            )
            role_meta = _profile_role_meta(live_profile or raw_profile)
            form_meta = _profile_form_meta(live_profile or raw_profile)
            median = draftkings_nba_fpts(adjusted["points"], adjusted["rebounds"], adjusted["assists"])
            median *= _recent_form_scale(form_meta["recent_fpts_avg"], form_meta["recent_fpts_weighted"])
            volatility = _estimate_volatility(adjusted["minutes"], adjusted["points"], adjusted["rebounds"], adjusted["assists"])
            projection = DraftKingsNBAProjection(
                player_id="",
                name=profile.name,
                team=team_context.code,
                opponent=opponent,
                salary=0,
                positions=tuple(),
                roster_positions=tuple(),
                game=game,
                median_fpts=median,
                ceiling_fpts=median * (1.0 + volatility * 0.9),
                floor_fpts=max(0.0, median * (1.0 - volatility * 0.65)),
                volatility=volatility,
                projection_confidence=_adjust_confidence_for_status(confidence, status),
                minutes=adjusted["minutes"],
                points=adjusted["points"],
                rebounds=adjusted["rebounds"],
                assists=adjusted["assists"],
                availability_status=status,
                availability_source="game_context" if canonicalize_player_name(profile.name) in side_status_lookup else "profile",
                recent_games_sample=role_meta["games_sample"],
                recent_minutes_avg=role_meta["recent_minutes_avg"],
                participation_rate=role_meta["participation_rate"],
                role_stability=role_meta["role_stability"],
                recent_fpts_avg=form_meta["recent_fpts_avg"],
                recent_fpts_weighted=form_meta["recent_fpts_weighted"],
                recent_form_delta=form_meta["recent_form_delta"],
            )
            results[dfs_name_key(profile.name)] = projection
    return results


def _apply_matchup_feature_adjustments(
    adjusted: dict[str, float],
    *,
    team_code: str,
    opponent_code: str,
    position: str,
    team_features: dict[str, float],
    opponent_features: dict[str, float],
) -> dict[str, float]:
    if not team_features or not opponent_features:
        return adjusted
    league_pace = 100.0
    league_3pa_allowed = 37.0
    league_orb_rate_allowed = 0.28
    league_guard_ast_allowed = 10.0
    league_center_reb_allowed = 13.0

    team_pace = float(team_features.get("recent_pace") or 0.0)
    opp_pace = float(opponent_features.get("recent_pace") or 0.0)
    pace_basis = (team_pace + opp_pace) / 2.0 if team_pace and opp_pace else max(team_pace, opp_pace, 0.0)
    pace_scale = _bounded_ratio(pace_basis, league_pace, 0.94, 1.06)

    pos = str(position or "").upper()
    points_scale = pace_scale
    rebounds_scale = pace_scale ** 0.5
    assists_scale = pace_scale ** 0.5

    if pos in {"G", "PG", "SG"}:
        guard_ast_scale = _bounded_ratio(float(opponent_features.get("opp_guard_ast_allowed_pg") or 0.0), league_guard_ast_allowed, 0.92, 1.08)
        threes_scale = _bounded_ratio(float(opponent_features.get("opp_3pa_allowed_pg") or 0.0), league_3pa_allowed, 0.95, 1.05)
        assists_scale *= guard_ast_scale
        points_scale *= threes_scale
    if pos in {"C"}:
        center_reb_scale = _bounded_ratio(float(opponent_features.get("opp_center_reb_allowed_pg") or 0.0), league_center_reb_allowed, 0.92, 1.08)
        orb_scale = _bounded_ratio(float(opponent_features.get("opp_orb_rate_allowed") or 0.0), league_orb_rate_allowed, 0.95, 1.05)
        rebounds_scale *= center_reb_scale * orb_scale
    if pos in {"F", "PF", "SF"}:
        wing_threes_scale = _bounded_ratio(float(opponent_features.get("opp_3pa_allowed_pg") or 0.0), league_3pa_allowed, 0.96, 1.04)
        orb_scale = _bounded_ratio(float(opponent_features.get("opp_orb_rate_allowed") or 0.0), league_orb_rate_allowed, 0.96, 1.04)
        points_scale *= wing_threes_scale
        rebounds_scale *= orb_scale

    return {
        "minutes": adjusted["minutes"],
        "points": adjusted["points"] * points_scale,
        "rebounds": adjusted["rebounds"] * rebounds_scale,
        "assists": adjusted["assists"] * assists_scale,
    }


def _bounded_ratio(value: float, baseline: float, lower: float, upper: float) -> float:
    if value <= 0.0 or baseline <= 0.0:
        return 1.0
    return max(lower, min(upper, value / baseline))


def _fallback_projection(
    player: DraftKingsPlayer,
    *,
    fallback_status: tuple[str, str] | None = None,
    live_profile: dict[str, object] | None = None,
) -> DraftKingsNBAProjection:
    avg = float(player.avg_points_per_game)
    status, source = fallback_status or ("unknown", "fallback")
    role_meta = _profile_role_meta(live_profile or {})
    form_meta = _profile_form_meta(live_profile or {})
    base_minutes = float(role_meta["recent_minutes_avg"] or 32.0 or 32.0)
    adjusted = _apply_availability_discount(base_minutes, avg * 0.52, avg * 0.24, avg * 0.16, status)
    median = avg if status == "active" else draftkings_nba_fpts(adjusted["points"], adjusted["rebounds"], adjusted["assists"])
    median *= _recent_form_scale(form_meta["recent_fpts_avg"], form_meta["recent_fpts_weighted"])
    volatility = 0.32
    return DraftKingsNBAProjection(
        player_id=player.player_id,
        name=player.name,
        team=player.team,
        opponent=player.opponent,
        salary=player.salary,
        positions=player.positions,
        roster_positions=player.roster_positions,
        game=player.game,
        median_fpts=median,
        ceiling_fpts=median * 1.28,
        floor_fpts=max(0.0, median * 0.7),
        volatility=volatility,
        projection_confidence=_adjust_confidence_for_status(0.4, status),
        minutes=adjusted["minutes"],
        points=adjusted["points"],
        rebounds=adjusted["rebounds"],
        assists=adjusted["assists"],
        availability_status=status,
        availability_source=source,
        recent_games_sample=role_meta["games_sample"],
        recent_minutes_avg=role_meta["recent_minutes_avg"],
        participation_rate=role_meta["participation_rate"],
        role_stability=role_meta["role_stability"],
        recent_fpts_avg=form_meta["recent_fpts_avg"],
        recent_fpts_weighted=form_meta["recent_fpts_weighted"],
        recent_form_delta=form_meta["recent_form_delta"],
    )


def _load_fallback_nba_injury_context(date_str: str, slate: DraftKingsSlate) -> dict[str, dict]:
    slate_games = sorted({str(player.game or "").strip() for player in slate.players if str(player.game or "").strip()})
    if not slate_games:
        return {}
    try:
        contexts = [ctx for ctx in fetch_nba_game_contexts(date_str) if str(ctx.get("matchup") or "").strip() in slate_games]
        if not contexts:
            return {}
        details = fetch_nba_injury_context_details(date_str, contexts)
        return details.get("parsed", {}) or {}
    except Exception:
        return {}


def _load_live_nba_profiles(date_str: str, slate: DraftKingsSlate) -> dict[tuple[str, str], dict[str, object]]:
    slate_games = sorted({str(player.game or "").strip() for player in slate.players if str(player.game or "").strip()})
    if not slate_games:
        return {}
    try:
        contexts = [ctx for ctx in fetch_nba_game_contexts(date_str) if str(ctx.get("matchup") or "").strip() in slate_games]
    except Exception:
        return {}
    profiles_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for context in contexts:
        for side, team_key in (("away", "away_team_id"), ("home", "home_team_id")):
            team_id = context.get(team_key)
            team_code = context["matchup"].split("@", 1)[0] if side == "away" else context["matchup"].split("@", 1)[1]
            if not team_id:
                continue
            try:
                profiles = fetch_nba_team_player_profiles(str(team_id), date_str)
            except Exception:
                continue
            for profile in profiles:
                name = str(profile.get("name") or "").strip()
                if not name:
                    continue
                profiles_by_key[(team_code, dfs_name_key(name))] = dict(profile)
    return profiles_by_key


def _fallback_status_from_sources(
    player: DraftKingsPlayer,
    *,
    live_profile_statuses: dict[tuple[str, str], tuple[str, str]],
    injury_context: dict | None,
) -> tuple[str, str]:
    live_status = live_profile_statuses.get((player.team, dfs_name_key(player.name)))
    if live_status is not None:
        return live_status
    return _fallback_status_from_injury_context(player, injury_context)


def _fallback_status_from_injury_context(player: DraftKingsPlayer, injury_context: dict | None) -> tuple[str, str]:
    availability = (injury_context or {}).get("availability") or {}
    game = str(player.game or "")
    if "@" not in game:
        return ("unknown", "fallback")
    away_team, home_team = game.split("@", 1)
    if player.team == away_team:
        side = "away"
    elif player.team == home_team:
        side = "home"
    else:
        return ("unknown", "fallback")
    side_info = availability.get(side) or {}
    submitted = bool(side_info.get("submitted"))
    entries = side_info.get("entries") or []
    if submitted:
        lookup = nba_availability_status_lookup(entries)
        matched = lookup.get(canonicalize_player_name(player.name))
        if matched:
            return (matched, "injury_report")
        return ("active", "injury_report")
    return ("unknown", "fallback")


def _estimate_volatility(minutes: float, points: float, rebounds: float, assists: float) -> float:
    base = 0.24
    minutes_adj = 0.08 if minutes < 26.0 else 0.03
    usage_adj = 0.05 if points >= 22.0 else 0.0
    peripheral_adj = 0.04 if rebounds + assists >= 12.0 else 0.0
    return min(0.5, base + minutes_adj + usage_adj + peripheral_adj)


def _profile_injury_status(profile) -> str:
    injuries = getattr(profile, "injuries", None)
    if injuries is None and isinstance(profile, dict):
        injuries = profile.get("injuries", [])
    for injury in injuries or []:
        status = str(injury.get("status") or "").strip().lower()
        if status:
            return status
    raw_status = getattr(profile, "status", None)
    if raw_status is None and isinstance(profile, dict):
        raw_status = profile.get("status")
    return str(raw_status or "active").strip().lower()


def _availability_side_submitted(availability: dict, side_key: str) -> bool:
    direct = availability.get(f"{side_key}_submitted")
    if isinstance(direct, bool):
        return direct
    side_value = availability.get(side_key)
    if isinstance(side_value, dict):
        return bool(side_value.get("submitted"))
    return False


def _apply_availability_discount(minutes: float, points: float, rebounds: float, assists: float, status: str) -> dict[str, float]:
    status = str(status or "").strip().lower()
    minutes_scale = {
        "out": 0.0,
        "doubtful": 0.18,
        "day to day": 0.72,
        "day-to-day": 0.72,
        "questionable": 0.74,
        "probable": 0.94,
    }.get(status, 1.0)
    production_scale = {
        "out": 0.0,
        "doubtful": 0.12,
        "day to day": 0.68,
        "day-to-day": 0.68,
        "questionable": 0.7,
        "probable": 0.96,
    }.get(status, 1.0)
    return {
        "minutes": max(0.0, minutes * minutes_scale),
        "points": max(0.0, points * production_scale),
        "rebounds": max(0.0, rebounds * production_scale),
        "assists": max(0.0, assists * production_scale),
    }


def _adjust_confidence_for_status(confidence: float, status: str) -> float:
    status = str(status or "").strip().lower()
    if status == "out":
        return 0.0
    if status in {"doubtful", "day to day", "day-to-day", "questionable"}:
        return max(0.15, confidence * 0.55)
    if status == "probable":
        return confidence * 0.9
    return confidence


def _profile_role_meta(profile: dict | object) -> dict[str, float]:
    games_sample = _safe_profile_float(profile, "games_sample")
    recent_minutes_avg = _safe_profile_float(profile, "minutes")
    participation_rate = max(0.0, min(1.0, games_sample / 8.0)) if games_sample > 0.0 else 0.0
    normalized_minutes = max(0.0, min(1.0, recent_minutes_avg / 30.0)) if recent_minutes_avg > 0.0 else 0.0
    role_stability = participation_rate * normalized_minutes
    return {
        "games_sample": games_sample,
        "recent_minutes_avg": recent_minutes_avg,
        "participation_rate": participation_rate,
        "role_stability": role_stability,
    }


def _profile_form_meta(profile: dict | object) -> dict[str, float]:
    recent_fpts_avg = _safe_profile_float(profile, "recent_fpts_avg")
    recent_fpts_weighted = _safe_profile_float(profile, "recent_fpts_weighted")
    return {
        "recent_fpts_avg": recent_fpts_avg,
        "recent_fpts_weighted": recent_fpts_weighted,
        "recent_form_delta": recent_fpts_weighted - recent_fpts_avg,
    }


def _recent_form_scale(recent_fpts_avg: float, recent_fpts_weighted: float) -> float:
    if recent_fpts_avg <= 0.0 or recent_fpts_weighted <= 0.0:
        return 1.0
    return max(0.94, min(1.08, recent_fpts_weighted / recent_fpts_avg))


def _safe_profile_float(profile: dict | object, field: str) -> float:
    if isinstance(profile, dict):
        value = profile.get(field)
    else:
        value = getattr(profile, field, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def attach_salary_metadata(
    slate: DraftKingsSlate,
    projections: Iterable[DraftKingsNBAProjection],
) -> list[DraftKingsNBAProjection]:
    by_name = {dfs_name_key(player.name): player for player in slate.players}
    enriched: list[DraftKingsNBAProjection] = []
    for projection in projections:
        source = by_name.get(dfs_name_key(projection.name))
        if source is None:
            enriched.append(projection)
            continue
        enriched.append(
            DraftKingsNBAProjection(
                player_id=source.player_id,
                name=projection.name,
                team=projection.team or source.team,
                opponent=projection.opponent or source.opponent,
                salary=source.salary,
                positions=source.positions,
                roster_positions=source.roster_positions,
                game=projection.game or source.game,
                median_fpts=projection.median_fpts,
                ceiling_fpts=projection.ceiling_fpts,
                floor_fpts=projection.floor_fpts,
                volatility=projection.volatility,
                projection_confidence=projection.projection_confidence,
                minutes=projection.minutes,
                points=projection.points,
                rebounds=projection.rebounds,
                assists=projection.assists,
                availability_status=projection.availability_status,
                availability_source=projection.availability_source,
                recent_games_sample=projection.recent_games_sample,
                recent_minutes_avg=projection.recent_minutes_avg,
                participation_rate=projection.participation_rate,
                role_stability=projection.role_stability,
                recent_fpts_avg=projection.recent_fpts_avg,
                recent_fpts_weighted=projection.recent_fpts_weighted,
                recent_form_delta=projection.recent_form_delta,
            )
        )
    return enriched


def optimize_nba_classic_lineups(
    projections: list[DraftKingsNBAProjection],
    salary_cap: int = 50000,
    max_candidates: int | None = None,
    limit: int = 10,
    contest_type: str = "cash",
    focus_players: set[str] | None = None,
    fade_players: set[str] | None = None,
    game_boosts: dict[str, float] | None = None,
    stack_targets: set[str] | None = None,
    bring_back_targets: set[str] | None = None,
    one_off_targets: set[str] | None = None,
    max_players_per_game: int | None = None,
    preferred_salary_shape: str | None = None,
    objective_noise_scale: float = 0.0,
    max_exposure: float | None = None,
    locked_players: set[str] | None = None,
    excluded_players: set[str] | None = None,
) -> list[DraftKingsLineup]:
    normalized_contest = _normalize_contest_type(contest_type)
    focus_keys = {dfs_name_key(name) for name in (focus_players or set()) if dfs_name_key(name)}
    fade_keys = {dfs_name_key(name) for name in (fade_players or set()) if dfs_name_key(name)}
    stack_keys = {dfs_name_key(name) for name in (stack_targets or set()) if dfs_name_key(name)}
    bring_back_keys = {dfs_name_key(name) for name in (bring_back_targets or set()) if dfs_name_key(name)}
    one_off_keys = {dfs_name_key(name) for name in (one_off_targets or set()) if dfs_name_key(name)}
    locked_keys = {dfs_name_key(name) for name in (locked_players or set()) if dfs_name_key(name)}
    excluded_keys = {dfs_name_key(name) for name in (excluded_players or set()) if dfs_name_key(name)}
    environment_boosts = dict(game_boosts or {})
    eligible = [
        item for item in projections
        if item.availability_status != "out"
        and dfs_name_key(item.name) not in excluded_keys
    ]
    if normalized_contest == "cash":
        cash_safe = [item for item in eligible if _is_cash_role_viable(item)]
        if len(cash_safe) >= 8:
            eligible = cash_safe
    ranked_eligible = sorted(
        eligible,
        key=lambda item: (
            _lineup_pool_score(
                item,
                normalized_contest,
                focus_keys,
                fade_keys,
                environment_boosts,
                stack_keys,
                bring_back_keys,
                one_off_keys,
            ),
            item.ceiling_fpts,
            -item.salary,
        ),
        reverse=True,
    )
    if max_candidates is None:
        candidate_pool = ranked_eligible
    else:
        value_slice = max(10, max_candidates // 2)
        salary_slice = max(8, max_candidates // 4)
        ranked_by_median = ranked_eligible[:value_slice]
        ranked_by_value = sorted(
            eligible,
            key=lambda item: (
                (
                    _lineup_pool_score(
                        item,
                        normalized_contest,
                        focus_keys,
                        fade_keys,
                        environment_boosts,
                        stack_keys,
                        bring_back_keys,
                        one_off_keys,
                    )
                )
                / max(item.salary, 1),
                item.ceiling_fpts,
            ),
            reverse=True,
        )[:value_slice]
        ranked_by_salary_relief = sorted(
            eligible,
            key=lambda item: (
                item.salary,
                -_lineup_pool_score(
                    item,
                    normalized_contest,
                    focus_keys,
                    fade_keys,
                    environment_boosts,
                    stack_keys,
                    bring_back_keys,
                    one_off_keys,
                ),
            ),
        )[:salary_slice]
        candidate_pool = list(
            {
                item.player_id or item.name: item
                for item in (ranked_by_median + ranked_by_value + ranked_by_salary_relief)
            }.values()
        )
        if normalized_contest in {"single_entry_gpp", "large_field_gpp"}:
            core_keys = {item.player_id or item.name for item in candidate_pool}
            remaining_pool = [item for item in eligible if (item.player_id or item.name) not in core_keys]
            tail_size = min(max(4, max_candidates // 8), len(remaining_pool))
            if tail_size > 0:
                rng = random.Random(f"nba:{normalized_contest}:{date_signature(candidate_pool)}:{salary_cap}:{limit}")
                candidate_pool.extend(rng.sample(remaining_pool, tail_size))
    lineups: list[DraftKingsLineup] = []
    player_scores = [
        _lineup_pool_score(
            item,
            normalized_contest,
            focus_keys,
            fade_keys,
            environment_boosts,
            stack_keys,
            bring_back_keys,
            one_off_keys,
        )
        for item in candidate_pool
    ]
    required_player_indices = [
        idx for idx, item in enumerate(candidate_pool)
        if dfs_name_key(item.name) in locked_keys
    ]
    solved = solve_dfs_lineups(
        player_count=len(candidate_pool),
        slot_names=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        salary_cap=salary_cap,
        salaries=[item.salary for item in candidate_pool],
        player_scores=player_scores,
        eligibility_fn=lambda player_idx, slot_name: _positions_can_fill_slot(set(candidate_pool[player_idx].positions), slot_name),
        lineups_to_generate=limit,
        max_players_per_game=max_players_per_game,
        game_keys=[item.game for item in candidate_pool],
        game_countable_fn=lambda player_idx: True,
        objective_noise_scale=objective_noise_scale,
        rng_seed=17,
        required_player_indices=required_player_indices,
        max_exposure=max_exposure,
    )
    for solved_lineup in solved:
        lineup_players = tuple(candidate_pool[idx] for idx in solved_lineup.player_indices)
        salary_used = sum(player.salary for player in lineup_players)
        median = sum(player.median_fpts for player in lineup_players)
        ceiling = sum(player.ceiling_fpts for player in lineup_players)
        floor = sum(player.floor_fpts for player in lineup_players)
        average_confidence = sum(player.projection_confidence for player in lineup_players) / len(lineup_players)
        unknown_count = sum(1 for player in lineup_players if player.availability_status == "unknown")
        lineups.append(
            DraftKingsLineup(
                players=tuple(sorted(lineup_players, key=lambda item: item.salary, reverse=True)),
                salary_used=salary_used,
                median_fpts=median,
                ceiling_fpts=ceiling,
                floor_fpts=floor,
                average_confidence=average_confidence,
                unknown_count=unknown_count,
            )
        )
    lineups.sort(
        key=lambda item: _lineup_rank_key(
            item,
            normalized_contest,
            salary_cap,
            environment_boosts,
            preferred_salary_shape,
            stack_keys,
            bring_back_keys,
            one_off_keys,
        ),
        reverse=True,
    )
    return lineups[:limit]


def _is_cash_role_viable(player: DraftKingsNBAProjection) -> bool:
    salary = float(player.salary or 0.0)
    status = str(player.availability_status or "").strip().lower()
    if status not in {"active", "probable", "available"}:
        return False
    if salary >= 6500:
        return True
    role_stability = float(player.role_stability or 0.0)
    recent_minutes = float(player.recent_minutes_avg or 0.0)
    games_sample = float(player.recent_games_sample or 0.0)
    if role_stability <= 0.0 and recent_minutes <= 0.0 and games_sample <= 0.0:
        return salary > 4000
    if salary <= 4000 and (role_stability < 0.58 or recent_minutes < 22.0 or games_sample < 6.0):
        return False
    if salary <= 5500 and (role_stability < 0.45 or recent_minutes < 18.0 or games_sample < 5.0):
        return False
    return True


def _lineup_pool_score(
    player: DraftKingsNBAProjection,
    contest_type: str,
    focus_keys: set[str],
    fade_keys: set[str],
    game_boosts: dict[str, float],
    stack_keys: set[str],
    bring_back_keys: set[str],
    one_off_keys: set[str],
) -> float:
    name_key = dfs_name_key(player.name)
    focus_bonus = _focus_bonus(player, contest_type, name_key in focus_keys)
    fade_penalty = 8.0 if name_key in fade_keys else 0.0
    stack_bonus = 4.5 if name_key in stack_keys else 0.0
    bring_back_bonus = 3.0 if name_key in bring_back_keys else 0.0
    one_off_bonus = 2.0 if name_key in one_off_keys else 0.0
    base_conf = player.projection_confidence * 8.0
    environment_bonus = float(game_boosts.get(player.game, 0.0) or 0.0)
    recent_form_bonus = _recent_form_bonus(player, contest_type)
    if contest_type == "large_field_gpp":
        return (
            player.ceiling_fpts
            + (player.volatility * 8.0)
            + base_conf
            + focus_bonus
            + stack_bonus
            + bring_back_bonus
            + one_off_bonus
            + recent_form_bonus
            + (environment_bonus * 11.0)
            - fade_penalty
        )
    if contest_type == "single_entry_gpp":
        return (
            ((player.median_fpts * 0.7) + (player.ceiling_fpts * 0.45))
            + base_conf
            + focus_bonus
            + (stack_bonus * 0.85)
            + (bring_back_bonus * 0.75)
            + one_off_bonus
            + recent_form_bonus
            + (environment_bonus * 8.0)
            - fade_penalty
        )
    return (
        ((player.median_fpts * 1.0) + (player.floor_fpts * 0.35))
        + base_conf
        + focus_bonus
        + (stack_bonus * 0.25)
        + (bring_back_bonus * 0.25)
        + (one_off_bonus * 0.4)
        + recent_form_bonus
        + (environment_bonus * 4.5)
        - _cash_fragility_penalty(player)
        - fade_penalty
    )


def _cash_fragility_penalty(player: DraftKingsNBAProjection) -> float:
    salary = float(player.salary or 0.0)
    if salary > 6000:
        return 0.0
    recent_minutes = float(player.recent_minutes_avg or 0.0)
    games_sample = float(player.recent_games_sample or 0.0)
    role_stability = float(player.role_stability or 0.0)
    if recent_minutes <= 0.0 and games_sample <= 0.0 and role_stability <= 0.0:
        return 2.5 if salary <= 4000 else 0.0
    penalty = 0.0
    if salary <= 4000:
        penalty += max(0.0, 22.0 - recent_minutes) * 0.45
        penalty += max(0.0, 6.0 - games_sample) * 1.0
        penalty += max(0.0, 0.58 - role_stability) * 10.0
    elif salary <= 5500:
        penalty += max(0.0, 18.0 - recent_minutes) * 0.25
        penalty += max(0.0, 5.0 - games_sample) * 0.7
        penalty += max(0.0, 0.45 - role_stability) * 6.0
    # Cheap low-floor players are more damaging in cash than median alone implies.
    if salary <= 5000:
        penalty += max(0.0, 18.0 - float(player.floor_fpts or 0.0)) * 0.18
    return penalty


def _recent_form_bonus(player: DraftKingsNBAProjection, contest_type: str) -> float:
    if player.recent_fpts_avg <= 0.0 or player.recent_fpts_weighted <= 0.0:
        return 0.0
    form_ratio = max(0.92, min(1.10, player.recent_fpts_weighted / player.recent_fpts_avg))
    form_edge = form_ratio - 1.0
    if abs(form_edge) < 0.01:
        return 0.0
    scale = 42.0 if contest_type == "cash" else 26.0
    return form_edge * scale


def _focus_bonus(player: DraftKingsNBAProjection, contest_type: str, is_focus: bool) -> float:
    if not is_focus:
        return 0.0
    salary_relief = max(0.0, (7000.0 - float(player.salary)) / 500.0)
    if contest_type == "large_field_gpp":
        return 11.0 + (salary_relief * 2.4)
    if contest_type == "single_entry_gpp":
        return 9.5 + (salary_relief * 1.8)
    return 8.0 + (salary_relief * 1.2)


def _lineup_rank_key(
    lineup: DraftKingsLineup,
    contest_type: str,
    salary_cap: int,
    game_boosts: dict[str, float],
    preferred_salary_shape: str | None,
    stack_keys: set[str],
    bring_back_keys: set[str],
    one_off_keys: set[str],
) -> tuple[float, float, float, float]:
    confidence_bonus = lineup.average_confidence * 10.0
    environment_bonus = _lineup_environment_bonus(lineup, contest_type, salary_cap, game_boosts)
    salary_shape_bonus = _lineup_salary_shape_bonus(lineup, salary_cap, preferred_salary_shape)
    guidance_bonus = _lineup_guidance_bonus(lineup, contest_type, stack_keys, bring_back_keys, one_off_keys)
    if contest_type == "large_field_gpp":
        return (
            lineup.ceiling_fpts + confidence_bonus + environment_bonus + salary_shape_bonus + guidance_bonus,
            lineup.median_fpts,
            lineup.floor_fpts,
            -lineup.salary_used,
        )
    if contest_type == "single_entry_gpp":
        blended = (lineup.median_fpts * 0.65) + (lineup.ceiling_fpts * 0.45)
        return (
            blended + confidence_bonus + environment_bonus + salary_shape_bonus + guidance_bonus,
            lineup.ceiling_fpts,
            lineup.floor_fpts,
            -lineup.salary_used,
        )
    blended = lineup.median_fpts + (lineup.floor_fpts * 0.35)
    return (
        blended + confidence_bonus + environment_bonus + salary_shape_bonus + guidance_bonus,
        lineup.median_fpts,
        lineup.ceiling_fpts,
        -lineup.salary_used,
    )


def _lineup_environment_bonus(
    lineup: DraftKingsLineup,
    contest_type: str,
    salary_cap: int,
    game_boosts: dict[str, float],
) -> float:
    if not game_boosts:
        return 0.0
    game_counts: dict[str, int] = {}
    base_bonus = 0.0
    for player in lineup.players:
        if not player.game:
            continue
        game_counts[player.game] = game_counts.get(player.game, 0) + 1
        base_bonus += float(game_boosts.get(player.game, 0.0) or 0.0)
    stack_bonus = sum(max(count - 1, 0) * float(game_boosts.get(game, 0.0) or 0.0) for game, count in game_counts.items())
    salary_remaining = max(0, salary_cap - lineup.salary_used)
    if contest_type == "large_field_gpp":
        uniqueness_bonus = min(salary_remaining, 1500) / 180.0
        return (base_bonus * 1.3) + (stack_bonus * 2.6) + uniqueness_bonus
    if contest_type == "single_entry_gpp":
        uniqueness_bonus = min(salary_remaining, 1200) / 300.0
        return (base_bonus * 0.95) + (stack_bonus * 1.4) + uniqueness_bonus
    concentration_penalty = sum(max(count - 2, 0) * 1.4 for count in game_counts.values())
    return (base_bonus * 0.7) - concentration_penalty


def _normalize_contest_type(contest_type: str) -> str:
    normalized = str(contest_type or "").strip().lower()
    if normalized == "head_to_head":
        return "cash"
    if normalized == "tournament":
        return "large_field_gpp"
    return normalized or "cash"


def date_signature(players: list[DraftKingsNBAProjection]) -> str:
    return "|".join(sorted((player.player_id or player.name) for player in players[:12]))


def _lineup_game_count_exceeds(players: tuple[DraftKingsNBAProjection, ...], max_players_per_game: int) -> bool:
    counts: dict[str, int] = {}
    for player in players:
        if not player.game:
            continue
        counts[player.game] = counts.get(player.game, 0) + 1
        if counts[player.game] > max_players_per_game:
            return True
    return False


def _lineup_salary_shape_bonus(lineup: DraftKingsLineup, salary_cap: int, preferred_salary_shape: str | None) -> float:
    shape = str(preferred_salary_shape or "").strip().lower()
    salary_remaining = max(0, salary_cap - lineup.salary_used)
    salaries = [player.salary for player in lineup.players]
    if not salaries:
        return 0.0
    avg_salary = sum(salaries) / len(salaries)
    variance = sum((salary - avg_salary) ** 2 for salary in salaries) / len(salaries)
    std_dev = variance ** 0.5
    if shape == "balanced":
        return max(0.0, 8.0 - (std_dev / 550.0)) - (salary_remaining / 900.0)
    if shape == "stars_and_scrubs":
        high_salary_count = sum(1 for salary in salaries if salary >= 8500)
        punt_count = sum(1 for salary in salaries if salary <= 4500)
        return (high_salary_count * 2.5) + (punt_count * 1.8) + (std_dev / 1200.0)
    if shape == "leave_salary":
        return min(salary_remaining, 1800) / 220.0
    return 0.0


def _lineup_guidance_bonus(
    lineup: DraftKingsLineup,
    contest_type: str,
    stack_keys: set[str],
    bring_back_keys: set[str],
    one_off_keys: set[str],
) -> float:
    players_by_game: dict[str, list[DraftKingsNBAProjection]] = {}
    stack_hits = 0
    bring_back_hits = 0
    one_off_hits = 0
    for player in lineup.players:
        key = dfs_name_key(player.name)
        if key in stack_keys:
            stack_hits += 1
        if key in bring_back_keys:
            bring_back_hits += 1
        if key in one_off_keys:
            one_off_hits += 1
        if player.game:
            players_by_game.setdefault(player.game, []).append(player)
    game_correlation_bonus = 0.0
    for players in players_by_game.values():
        teams = {player.team for player in players if player.team}
        if len(players) >= 2 and len(teams) >= 2 and any(dfs_name_key(player.name) in bring_back_keys for player in players):
            game_correlation_bonus += 1.5
    if contest_type == "large_field_gpp":
        return (stack_hits * 2.6) + (bring_back_hits * 1.9) + one_off_hits + game_correlation_bonus
    if contest_type == "single_entry_gpp":
        return (stack_hits * 1.8) + (bring_back_hits * 1.2) + (one_off_hits * 0.8) + (game_correlation_bonus * 0.8)
    return (stack_hits * 0.4) + (bring_back_hits * 0.3) + (one_off_hits * 0.5)


def _is_valid_nba_classic_lineup(players: tuple[DraftKingsNBAProjection, ...]) -> bool:
    if len(players) != 8:
        return False
    player_positions = [set(player.positions) for player in players]
    return _can_fill_slots(player_positions, ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"])


def _can_fill_slots(player_positions: list[set[str]], required_slots: list[str]) -> bool:
    slots = list(required_slots)

    def assign(slot_index: int, used: set[int]) -> bool:
        if slot_index >= len(slots):
            return True
        slot = slots[slot_index]
        for idx, positions in enumerate(player_positions):
            if idx in used:
                continue
            if not _positions_can_fill_slot(positions, slot):
                continue
            used.add(idx)
            if assign(slot_index + 1, used):
                return True
            used.remove(idx)
        return False

    return assign(0, set())


def _positions_can_fill_slot(positions: set[str], slot: str) -> bool:
    if slot == "UTIL":
        return bool(positions)
    if slot == "G":
        return bool(positions & {"PG", "SG"})
    if slot == "F":
        return bool(positions & {"SF", "PF"})
    return slot in positions
