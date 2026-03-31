from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import requests

from data_pipeline.cache import DEFAULT_DB_PATH, SnapshotStore
from refresh_slate import fetch_nba_game_contexts


ESPN_NBA_TEAM_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule"
ESPN_NBA_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
FEATURE_SOURCE = "nba_matchup_features"
FEATURE_ENTITY_TYPE = "team_defense_features"


def load_nba_matchup_features(date_str: str, team_codes: Iterable[str], db_path: str = DEFAULT_DB_PATH) -> dict[str, dict[str, float]]:
    store = SnapshotStore(db_path)
    code_to_id = _team_code_to_id_map(date_str)
    features: dict[str, dict[str, float]] = {}
    for team_code in sorted({str(code or "").strip() for code in team_codes if str(code or "").strip()}):
        cached = store.get_snapshot(
            source=FEATURE_SOURCE,
            sport="nba",
            entity_type=FEATURE_ENTITY_TYPE,
            entity_key=team_code,
            as_of_date=date_str,
            max_age_hours=None,
        )
        if cached is not None:
            features[team_code] = dict(cached["payload"] or {})
            continue
        team_id = code_to_id.get(team_code)
        if not team_id:
            continue
        payload = fetch_nba_team_matchup_features(str(team_id), team_code, date_str)
        store.upsert_snapshot(
            source=FEATURE_SOURCE,
            sport="nba",
            entity_type=FEATURE_ENTITY_TYPE,
            entity_key=team_code,
            as_of_date=date_str,
            payload=payload,
            is_volatile=False,
        )
        features[team_code] = payload
    return features


def fetch_nba_team_matchup_features(team_id: str, team_code: str, as_of_date: str, last_n_games: int = 8) -> dict[str, float]:
    schedule_response = requests.get(ESPN_NBA_TEAM_SCHEDULE_URL.format(team_id=team_id), timeout=20)
    schedule_response.raise_for_status()
    schedule_payload = schedule_response.json()
    completed_events: list[str] = []
    for event in schedule_payload.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status", {}).get("type", {})
        if status.get("completed") and event.get("date", "") <= f"{as_of_date}T23:59Z":
            completed_events.append(str(event["id"]))
    completed_events = completed_events[-last_n_games:]
    if not completed_events:
        return _empty_features()

    game_pace: list[float] = []
    opp_3pa: list[float] = []
    opp_orb_rate: list[float] = []
    opp_guard_ast: list[float] = []
    opp_center_reb: list[float] = []

    for event_id in completed_events:
        summary_response = requests.get(ESPN_NBA_SUMMARY_URL, params={"event": event_id}, timeout=20)
        summary_response.raise_for_status()
        summary_payload = summary_response.json()

        team_box = _find_team_box(summary_payload, team_code)
        opp_box = _find_opponent_box(summary_payload, team_code)
        team_players = _find_player_block(summary_payload, team_code)
        opp_players = _find_opponent_player_block(summary_payload, team_code)
        if not team_box or not opp_box or not opp_players:
            continue

        team_possessions = _team_possessions(team_box)
        opp_possessions = _team_possessions(opp_box)
        if team_possessions > 0.0 and opp_possessions > 0.0:
            game_pace.append((team_possessions + opp_possessions) / 2.0)

        opp_3pa.append(_attempts_from_display(opp_box.get("3PT", "0-0")))
        team_dreb = _safe_float(team_box.get("DR", team_box.get("DREB", 0.0)))
        opp_or = _safe_float(opp_box.get("OR", opp_box.get("OREB", 0.0)))
        if team_dreb + opp_or > 0:
            opp_orb_rate.append(opp_or / (team_dreb + opp_or))

        guard_ast = 0.0
        center_reb = 0.0
        for athlete in opp_players:
            position = str((((athlete.get("athlete") or {}).get("position") or {}).get("abbreviation")) or "").upper()
            stats = athlete.get("stats") or []
            if not stats:
                continue
            labels = athlete.get("_labels") or {}
            ast = _safe_float(stats[labels["AST"]]) if "AST" in labels else 0.0
            reb = _safe_float(stats[labels["REB"]]) if "REB" in labels else 0.0
            if "G" in position:
                guard_ast += ast
            if "C" in position:
                center_reb += reb
        opp_guard_ast.append(guard_ast)
        opp_center_reb.append(center_reb)

    if not game_pace:
        return _empty_features()

    return {
        "games_sample": float(len(game_pace)),
        "recent_pace": _avg(game_pace),
        "opp_3pa_allowed_pg": _avg(opp_3pa),
        "opp_orb_rate_allowed": _avg(opp_orb_rate),
        "opp_guard_ast_allowed_pg": _avg(opp_guard_ast),
        "opp_center_reb_allowed_pg": _avg(opp_center_reb),
    }


def _team_code_to_id_map(date_str: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for context in fetch_nba_game_contexts(date_str):
        matchup = str(context.get("matchup") or "")
        if "@" not in matchup:
            continue
        away_code, home_code = matchup.split("@", 1)
        away_id = context.get("away_team_id")
        home_id = context.get("home_team_id")
        if away_id:
            mapping[away_code] = str(away_id)
        if home_id:
            mapping[home_code] = str(home_id)
    return mapping


def _find_team_box(summary_payload: dict[str, Any], team_code: str) -> dict[str, Any] | None:
    for item in summary_payload.get("boxscore", {}).get("teams", []):
        if str((item.get("team") or {}).get("abbreviation") or "") == team_code:
            return {str(stat.get("abbreviation") or stat.get("label") or ""): stat.get("displayValue") for stat in item.get("statistics", [])}
    return None


def _find_opponent_box(summary_payload: dict[str, Any], team_code: str) -> dict[str, Any] | None:
    for item in summary_payload.get("boxscore", {}).get("teams", []):
        if str((item.get("team") or {}).get("abbreviation") or "") != team_code:
            return {str(stat.get("abbreviation") or stat.get("label") or ""): stat.get("displayValue") for stat in item.get("statistics", [])}
    return None


def _find_player_block(summary_payload: dict[str, Any], team_code: str) -> list[dict[str, Any]]:
    for block in summary_payload.get("boxscore", {}).get("players", []):
        if str((block.get("team") or {}).get("abbreviation") or "") != team_code:
            continue
        for stat_group in block.get("statistics", []):
            labels = stat_group.get("labels", [])
            label_idx = {label: idx for idx, label in enumerate(labels)}
            athletes = []
            for athlete_block in stat_group.get("athletes", []):
                copied = dict(athlete_block)
                copied["_labels"] = label_idx
                athletes.append(copied)
            return athletes
    return []


def _find_opponent_player_block(summary_payload: dict[str, Any], team_code: str) -> list[dict[str, Any]]:
    for block in summary_payload.get("boxscore", {}).get("players", []):
        if str((block.get("team") or {}).get("abbreviation") or "") == team_code:
            continue
        for stat_group in block.get("statistics", []):
            labels = stat_group.get("labels", [])
            label_idx = {label: idx for idx, label in enumerate(labels)}
            athletes = []
            for athlete_block in stat_group.get("athletes", []):
                copied = dict(athlete_block)
                copied["_labels"] = label_idx
                athletes.append(copied)
            return athletes
    return []


def _team_possessions(team_box: dict[str, Any]) -> float:
    fga = _attempts_from_display(team_box.get("FG", "0-0"))
    fta = _attempts_from_display(team_box.get("FT", "0-0"))
    turnovers = _safe_float(team_box.get("TO", 0.0))
    offensive_rebounds = _safe_float(team_box.get("OR", team_box.get("OREB", 0.0)))
    return max(0.0, fga - offensive_rebounds + turnovers + (0.44 * fta))


def _attempts_from_display(value: Any) -> float:
    text = str(value or "")
    if "-" not in text:
        return 0.0
    _made, attempts = text.split("-", 1)
    return _safe_float(attempts)


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _empty_features() -> dict[str, float]:
    return {
        "games_sample": 0.0,
        "recent_pace": 0.0,
        "opp_3pa_allowed_pg": 0.0,
        "opp_orb_rate_allowed": 0.0,
        "opp_guard_ast_allowed_pg": 0.0,
        "opp_center_reb_allowed_pg": 0.0,
    }
