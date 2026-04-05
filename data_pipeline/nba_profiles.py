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

    accum: dict[str, dict[str, Any]] = {}
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
                        {
                            "games": 0.0,
                            "starts": 0.0,
                            "minutes": 0.0,
                            "points": 0.0,
                            "rebounds": 0.0,
                            "assists": 0.0,
                            "turnovers": 0.0,
                            "fouls": 0.0,
                            "fga": 0.0,
                            "three_pa": 0.0,
                            "fta": 0.0,
                            "oreb": 0.0,
                            "dreb": 0.0,
                            "fantasy_scores": [],
                        },
                    )
                    minutes = _stat_from_labels(stats, label_idx, ("MIN",), parser=_parse_minutes)
                    points = _stat_from_labels(stats, label_idx, ("PTS",))
                    rebounds = _stat_from_labels(stats, label_idx, ("REB", "TREB"))
                    assists = _stat_from_labels(stats, label_idx, ("AST",))
                    turnovers = _stat_from_labels(stats, label_idx, ("TO", "TOV"))
                    fouls = _stat_from_labels(stats, label_idx, ("PF",))
                    fga = _attempts_from_labels(stats, label_idx, ("FGA", "FG-A", "FG"))
                    three_pa = _attempts_from_labels(stats, label_idx, ("3PA", "FG3A", "3PTA", "3PT-A", "3PT"))
                    fta = _attempts_from_labels(stats, label_idx, ("FTA", "FT-A", "FT"))
                    oreb = _stat_from_labels(stats, label_idx, ("OR", "OREB"))
                    dreb = _stat_from_labels(stats, label_idx, ("DR", "DREB"))
                    entry["games"] += 1.0
                    entry["starts"] += 1.0 if bool(athlete_block.get("starter")) else 0.0
                    entry["minutes"] += minutes
                    entry["points"] += points
                    entry["rebounds"] += rebounds
                    entry["assists"] += assists
                    entry["turnovers"] += turnovers
                    entry["fouls"] += fouls
                    entry["fga"] += fga
                    entry["three_pa"] += three_pa
                    entry["fta"] += fta
                    entry["oreb"] += oreb
                    entry["dreb"] += dreb
                    entry["fantasy_scores"].append(_draftkings_nba_box_score_fpts(points, rebounds, assists))

    profiles = []
    for athlete_id, athlete in roster_lookup.items():
        statline = accum.get(
            athlete_id,
            {
                "games": 0.0,
                "starts": 0.0,
                "minutes": 0.0,
                "points": 0.0,
                "rebounds": 0.0,
                "assists": 0.0,
                "turnovers": 0.0,
                "fouls": 0.0,
                "fga": 0.0,
                "three_pa": 0.0,
                "fta": 0.0,
                "oreb": 0.0,
                "dreb": 0.0,
                "fantasy_scores": [],
            },
        )
        games = max(statline["games"], 1.0)
        fantasy_scores = [float(score) for score in statline.get("fantasy_scores", [])]
        recent_fpts_avg = sum(fantasy_scores) / len(fantasy_scores) if fantasy_scores else 0.0
        if fantasy_scores:
            weights = list(range(1, len(fantasy_scores) + 1))
            weighted_total = sum(score * weight for score, weight in zip(fantasy_scores, weights))
            recent_fpts_weighted = weighted_total / float(sum(weights))
        else:
            recent_fpts_weighted = 0.0
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
                "starts": statline["starts"],
                "minutes": statline["minutes"] / games,
                "points": statline["points"] / games,
                "rebounds": statline["rebounds"] / games,
                "assists": statline["assists"] / games,
                "turnovers": statline["turnovers"] / games,
                "fouls": statline["fouls"] / games,
                "fga": statline["fga"] / games,
                "three_pa": statline["three_pa"] / games,
                "fta": statline["fta"] / games,
                "oreb": statline["oreb"] / games,
                "dreb": statline["dreb"] / games,
                "recent_fpts_avg": recent_fpts_avg,
                "recent_fpts_weighted": recent_fpts_weighted,
                "recent_form_delta": recent_fpts_weighted - recent_fpts_avg,
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


def _stat_from_labels(stats: list[Any], label_idx: dict[str, int], labels: tuple[str, ...], parser=_parse_float) -> float:
    for label in labels:
        if label in label_idx and label_idx[label] < len(stats):
            return parser(stats[label_idx[label]])
    return 0.0


def _attempts_from_labels(stats: list[Any], label_idx: dict[str, int], labels: tuple[str, ...]) -> float:
    for label in labels:
        if label in label_idx and label_idx[label] < len(stats):
            value = stats[label_idx[label]]
            if isinstance(value, str) and "-" in value:
                _made, attempts = value.split("-", 1)
                return _parse_float(attempts)
            return _parse_float(value)
    return 0.0


def _draftkings_nba_box_score_fpts(points: float, rebounds: float, assists: float) -> float:
    return float(points + (rebounds * 1.25) + (assists * 1.5))
