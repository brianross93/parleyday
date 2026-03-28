from __future__ import annotations

from typing import Any

import requests


ESPN_NBA_TEAM_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
ESPN_NBA_TEAM_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule"
ESPN_NBA_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def fetch_nba_team_player_profiles(team_id: str, as_of_date: str, last_n_games: int = 8) -> list[dict[str, Any]]:
    roster_response = requests.get(ESPN_NBA_TEAM_ROSTER_URL.format(team_id=team_id), timeout=20)
    roster_response.raise_for_status()
    roster_payload = roster_response.json()
    athletes = roster_payload.get("athletes", [])
    roster_lookup = {athlete["id"]: athlete for athlete in athletes}

    schedule_response = requests.get(ESPN_NBA_TEAM_SCHEDULE_URL.format(team_id=team_id), timeout=20)
    schedule_response.raise_for_status()
    schedule_payload = schedule_response.json()
    completed_events = []
    for event in schedule_payload.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status", {}).get("type", {})
        if status.get("completed") and event.get("date", "") <= f"{as_of_date}T23:59Z":
            completed_events.append(event["id"])
    completed_events = completed_events[-last_n_games:]

    accum: dict[str, dict[str, float]] = {}
    for event_id in completed_events:
        summary_response = requests.get(ESPN_NBA_SUMMARY_URL, params={"event": event_id}, timeout=20)
        summary_response.raise_for_status()
        summary_payload = summary_response.json()
        for team_block in summary_payload.get("boxscore", {}).get("players", []):
            team = team_block.get("team", {})
            if str(team.get("id")) != str(team_id):
                continue
            for stat_group in team_block.get("statistics", []):
                labels = stat_group.get("labels", [])
                label_idx = {label: idx for idx, label in enumerate(labels)}
                for athlete_block in stat_group.get("athletes", []):
                    athlete = athlete_block.get("athlete", {})
                    athlete_id = athlete.get("id")
                    if athlete_id not in roster_lookup:
                        continue
                    stats = athlete_block.get("stats", [])
                    if athlete_block.get("didNotPlay") or not stats:
                        continue
                    entry = accum.setdefault(
                        athlete_id,
                        {"games": 0.0, "minutes": 0.0, "points": 0.0, "rebounds": 0.0, "assists": 0.0},
                    )
                    entry["games"] += 1.0
                    entry["minutes"] += _parse_minutes(stats[label_idx["MIN"]]) if "MIN" in label_idx else 0.0
                    entry["points"] += _parse_float(stats[label_idx["PTS"]]) if "PTS" in label_idx else 0.0
                    entry["rebounds"] += _parse_float(stats[label_idx["REB"]]) if "REB" in label_idx else 0.0
                    entry["assists"] += _parse_float(stats[label_idx["AST"]]) if "AST" in label_idx else 0.0

    profiles = []
    for athlete_id, athlete in roster_lookup.items():
        statline = accum.get(athlete_id, {"games": 0.0, "minutes": 0.0, "points": 0.0, "rebounds": 0.0, "assists": 0.0})
        games = max(statline["games"], 1.0)
        injuries = athlete.get("injuries", []) or []
        profiles.append(
            {
                "player_id": athlete_id,
                "name": athlete.get("displayName", ""),
                "position": (athlete.get("position") or {}).get("abbreviation"),
                "status": (athlete.get("status") or {}).get("type", "active"),
                "injuries": [
                    {
                        "status": injury.get("status"),
                        "detail": injury.get("detail"),
                        "type": injury.get("type"),
                    }
                    for injury in injuries
                ],
                "games_sample": statline["games"],
                "minutes": statline["minutes"] / games,
                "points": statline["points"] / games,
                "rebounds": statline["rebounds"] / games,
                "assists": statline["assists"] / games,
            }
        )
    return profiles


def _parse_minutes(value: str) -> float:
    if ":" in str(value):
        minutes, seconds = str(value).split(":", 1)
        return float(minutes) + (float(seconds) / 60.0)
    return _parse_float(value)


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
