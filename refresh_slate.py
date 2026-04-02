import argparse
import io
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
from pypdf import PdfReader

from env_config import load_local_env
from data_pipeline import (
    SnapshotStore,
    build_batter_profile_payload,
    build_pitcher_profile_payload,
    fetch_nba_team_player_profiles,
)
from quantum_parlay_oracle import (
    canonical_team_code,
    clean_market_team_label,
    fetch_live_mlb_team_form,
    fetch_live_nba_team_form,
    load_live_legs,
)

load_local_env()


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
ESPN_NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
NBA_INJURY_REPORT_INDEX_URL = "https://official.nba.com/nba-injury-report-2025-26-season/"
FAST_TIMEOUT_SECONDS = 5
STANDARD_TIMEOUT_SECONDS = 12
PDF_TIMEOUT_SECONDS = 15
NBA_INJURY_PROBE_ATTEMPTS = 16
NBA_INJURY_REPORT_PDF_RE = re.compile(
    r"https://ak-static\.cms\.nba\.com/referee/injury/Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf"
)
NBA_PLAYER_STATUS_RE = re.compile(
    r"(?P<name>[A-Z][A-Za-z'. -]+?,\s*[A-Z][A-Za-z'. -]+?)\s+"
    r"(?P<status>Out|Questionable|Probable|Doubtful|Available)\s*"
)
MLB_TRANSACTIONS_URL = "https://statsapi.mlb.com/api/v1/transactions"


def fetch_last_mlb_lineup(team_id: int, date_str: str, lookback_days: int = 14) -> list[dict[str, Any]]:
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = (target_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date,
            "endDate": date_str,
        },
        timeout=20,
    )
    response.raise_for_status()
    games = []
    for date_block in response.json().get("dates", []):
        for game in date_block.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            official_date = str(game.get("officialDate") or "")
            if official_date and official_date >= date_str:
                continue
            games.append(game)
    games.sort(key=lambda item: str(item.get("officialDate") or ""), reverse=True)
    for game in games:
        game_pk = game.get("gamePk")
        if not game_pk:
            continue
        feed = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=20).json()
        teams = (feed.get("liveData", {}).get("boxscore", {}) or {}).get("teams", {})
        away_team_id = ((feed.get("gameData", {}).get("teams", {}) or {}).get("away", {}) or {}).get("id")
        side = "away" if away_team_id == team_id else "home"
        players = ((teams.get(side) or {}).get("players") or {})
        lineup = []
        for player in players.values():
            batting_order = str(player.get("battingOrder") or "").strip()
            if not batting_order:
                continue
            lineup.append(
                {
                    "id": player.get("person", {}).get("id"),
                    "fullName": player.get("person", {}).get("fullName", ""),
                    "battingOrder": batting_order,
                }
            )
        if lineup:
            lineup.sort(key=lambda item: int(str(item.get("battingOrder") or "999")))
            return lineup[:9]
    return []


def fetch_mlb_player_hand(player_id: int, hand_key: str = "pitchHand") -> str:
    response = requests.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}", timeout=20)
    response.raise_for_status()
    people = response.json().get("people", [])
    if not people:
        return "R"
    return str((people[0].get(hand_key) or {}).get("code") or "R")


def normalize_mlb_transaction(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_name": item.get("person", {}).get("fullName", ""),
        "date": item.get("date"),
        "effective_date": item.get("effectiveDate"),
        "type_code": item.get("typeCode"),
        "type_desc": item.get("typeDesc"),
        "description": item.get("description", ""),
    }


def fetch_mlb_team_roster_snapshot(team_id: int) -> dict[str, Any]:
    active = requests.get(
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
        params={"rosterType": "active"},
        timeout=20,
    ).json()
    full_roster = requests.get(
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
        params={"rosterType": "40Man"},
        timeout=20,
    ).json()
    active_players = []
    unavailable_players = []
    for item in active.get("roster", []):
        active_players.append(
            {
                "player_name": item.get("person", {}).get("fullName", ""),
                "player_id": item.get("person", {}).get("id"),
                "position": item.get("position", {}).get("abbreviation"),
                "status_code": item.get("status", {}).get("code"),
                "status_description": item.get("status", {}).get("description"),
            }
        )
    for item in full_roster.get("roster", []):
        if item.get("status", {}).get("code") == "A":
            continue
        unavailable_players.append(
            {
                "player_name": item.get("person", {}).get("fullName", ""),
                "player_id": item.get("person", {}).get("id"),
                "position": item.get("position", {}).get("abbreviation"),
                "status_code": item.get("status", {}).get("code"),
                "status_description": item.get("status", {}).get("description"),
            }
        )
    return {
        "active_players": active_players,
        "unavailable_players": unavailable_players,
    }


def fetch_mlb_team_transactions(team_id: int, date_str: str, days_back: int = 7) -> list[dict[str, Any]]:
    end_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = end_date - timedelta(days=days_back)
    response = requests.get(
        MLB_TRANSACTIONS_URL,
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return [normalize_mlb_transaction(item) for item in payload.get("transactions", [])]


def fetch_mlb_bullpen_snapshot(team_id: int, date_str: str, lookback_days: int = 3) -> dict[str, Any]:
    end_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = end_date - timedelta(days=lookback_days)
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        },
        timeout=20,
    )
    response.raise_for_status()
    appearances: dict[int, dict[str, Any]] = {}
    hand_cache: dict[int, str] = {}
    for date_block in response.json().get("dates", []):
        for game in date_block.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            game_pk = game.get("gamePk")
            feed = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=20).json()
            away_team = feed["gameData"]["teams"]["away"]["id"]
            side = "away" if away_team == team_id else "home"
            team_box = feed["liveData"]["boxscore"]["teams"][side]
            for pitcher_id in team_box.get("pitchers", []):
                player = team_box["players"].get(f"ID{pitcher_id}", {})
                pitching = player.get("stats", {}).get("pitching", {})
                if not pitching or int(pitching.get("gamesStarted", 0) or 0) > 0:
                    continue
                info = appearances.setdefault(
                    int(pitcher_id),
                    {
                        "player_id": int(pitcher_id),
                        "player_name": player.get("person", {}).get("fullName", ""),
                        "hand": hand_cache.setdefault(int(pitcher_id), fetch_mlb_player_hand(int(pitcher_id))),
                        "appearances": 0,
                        "pitches_last_3_days": 0,
                        "last_game_date": None,
                    },
                )
                info["appearances"] += 1
                info["pitches_last_3_days"] += int(pitching.get("numberOfPitches", 0) or 0)
                info["last_game_date"] = game.get("officialDate")
    relievers = sorted(
        appearances.values(),
        key=lambda item: (item["pitches_last_3_days"], item["appearances"]),
        reverse=True,
    )
    return {
        "relievers": relievers,
        "fatigue_score": sum(
            1.0
            for reliever in relievers
            if reliever["pitches_last_3_days"] >= 25
        ) + sum(
            0.5
            for reliever in relievers
            if 15 <= reliever["pitches_last_3_days"] < 25
        ),
    }


def nearest_hour_weather(latitude: float, longitude: float, game_time: str) -> dict[str, Any] | None:
    if not game_time:
        return None
    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
            "timezone": "UTC",
            "forecast_days": 2,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None
    target = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
    best_idx = min(
        range(len(times)),
        key=lambda idx: abs(datetime.fromisoformat(times[idx]).replace(tzinfo=target.tzinfo) - target),
    )
    return {
        "forecast_time": times[best_idx],
        "temperature_f": ((hourly.get("temperature_2m", [None])[best_idx] or 0.0) * 9.0 / 5.0) + 32.0,
        "wind_speed_mph": (hourly.get("wind_speed_10m", [None])[best_idx] or 0.0) * 0.621371,
        "wind_direction_degrees": hourly.get("wind_direction_10m", [None])[best_idx],
        "humidity_pct": hourly.get("relative_humidity_2m", [None])[best_idx],
    }


def fetch_mlb_game_contexts(date_str: str) -> list[dict[str, Any]]:
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,venue(location),lineups"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    contexts = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            away = game["teams"]["away"]["team"]
            home = game["teams"]["home"]["team"]
            matchup = f"{canonical_team_code(away['name'])}@{canonical_team_code(home['name'])}"
            venue = game.get("venue", {})
            location = venue.get("location", {})
            coordinates = location.get("defaultCoordinates", {})
            weather = None
            if coordinates.get("latitude") is not None and coordinates.get("longitude") is not None:
                try:
                    weather = nearest_hour_weather(
                        float(coordinates["latitude"]),
                        float(coordinates["longitude"]),
                        game.get("gameDate", ""),
                    )
                except Exception:
                    weather = None
            lineups = game.get("lineups") or {}
            away_players = list(lineups.get("awayPlayers", []) or [])
            home_players = list(lineups.get("homePlayers", []) or [])
            away_lineup_source = "confirmed"
            home_lineup_source = "confirmed"
            if len(away_players) < 9 and game["teams"]["away"]["team"].get("id"):
                away_players = fetch_last_mlb_lineup(int(game["teams"]["away"]["team"]["id"]), date_str)
                away_lineup_source = "last_fielded" if away_players else "missing"
            if len(home_players) < 9 and game["teams"]["home"]["team"].get("id"):
                home_players = fetch_last_mlb_lineup(int(game["teams"]["home"]["team"]["id"]), date_str)
                home_lineup_source = "last_fielded" if home_players else "missing"
            contexts.append(
                {
                    "game_pk": game.get("gamePk"),
                    "matchup": matchup,
                    "game_time": game.get("gameDate"),
                    "away_team_id": game["teams"]["away"]["team"].get("id"),
                    "home_team_id": game["teams"]["home"]["team"].get("id"),
                    "status": game.get("status", {}).get("detailedState", ""),
                    "venue": {
                        "name": venue.get("name"),
                        "city": location.get("city"),
                        "state": location.get("stateAbbrev") or location.get("state"),
                        "latitude": coordinates.get("latitude"),
                        "longitude": coordinates.get("longitude"),
                    },
                    "probable_pitchers": {
                        "away": game["teams"]["away"].get("probablePitcher", {}),
                        "home": game["teams"]["home"].get("probablePitcher", {}),
                    },
                    "lineups": {
                        "away": [player.get("fullName") for player in away_players],
                        "home": [player.get("fullName") for player in home_players],
                    },
                    "away_lineup_players": away_players,
                    "home_lineup_players": home_players,
                    "lineup_status": {
                        "away_confirmed": len(lineups.get("awayPlayers", [])) >= 9,
                        "home_confirmed": len(lineups.get("homePlayers", [])) >= 9,
                        "away_source": away_lineup_source,
                        "home_source": home_lineup_source,
                        "away_available": len(away_players) >= 9,
                        "home_available": len(home_players) >= 9,
                    },
                    "availability": {
                        "away": {},
                        "home": {},
                    },
                    "bullpen": {
                        "away": {},
                        "home": {},
                    },
                    "weather": weather,
                }
            )
    return contexts


def cache_mlb_matchup_profiles(store: SnapshotStore, date_str: str, contexts: list[dict[str, Any]]) -> int:
    season = int(date_str[:4])
    cached_count = 0
    for context in contexts:
        matchup_payload = {
            "matchup": context["matchup"],
            "season": season,
            "away_lineup": [],
            "home_lineup": [],
            "away_pitcher": None,
            "home_pitcher": None,
        }
        for side in ("away", "home"):
            lineup_key = f"{side}_lineup"
            player_objects = context.get(f"{side}_lineup_players", [])
            for lineup_index, player in enumerate(player_objects):
                existing = store.get_snapshot(
                    source="mlb_statsapi",
                    sport="mlb",
                    entity_type="player_profile",
                    entity_key=str(player["id"]),
                    as_of_date=date_str,
                )
                payload = existing["payload"] if existing is not None else build_batter_profile_payload(
                    int(player["id"]), player["fullName"], season, lineup_index
                )
                if existing is None:
                    store.upsert_snapshot(
                        source="mlb_statsapi",
                        sport="mlb",
                        entity_type="player_profile",
                        entity_key=str(player["id"]),
                        as_of_date=date_str,
                        payload=payload,
                        is_volatile=True,
                    )
                matchup_payload[lineup_key].append(payload)
                cached_count += 1
        for side in ("away", "home"):
            pitcher = context.get("probable_pitchers", {}).get(side) or {}
            if pitcher.get("id"):
                existing = store.get_snapshot(
                    source="mlb_statsapi",
                    sport="mlb",
                    entity_type="player_profile",
                    entity_key=str(pitcher["id"]),
                    as_of_date=date_str,
                )
                payload = existing["payload"] if existing is not None else build_pitcher_profile_payload(
                    int(pitcher["id"]), pitcher["fullName"], season
                )
                if existing is None:
                    store.upsert_snapshot(
                        source="mlb_statsapi",
                        sport="mlb",
                        entity_type="player_profile",
                        entity_key=str(pitcher["id"]),
                        as_of_date=date_str,
                        payload=payload,
                        is_volatile=True,
                    )
                matchup_payload[f"{side}_pitcher"] = payload
                cached_count += 1
        store.upsert_snapshot(
            source="mlb_refresh",
            sport="mlb",
            entity_type="matchup_profile",
            entity_key=context["matchup"],
            as_of_date=date_str,
            payload=matchup_payload,
            is_volatile=True,
        )
    return cached_count


def fetch_nba_game_contexts(date_str: str) -> list[dict[str, Any]]:
    response = requests.get(ESPN_NBA_SCOREBOARD_URL, params={"dates": date_str.replace("-", "")}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    contexts = []
    for event in payload.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        away = next((team for team in competitors if team.get("homeAway") == "away"), {})
        home = next((team for team in competitors if team.get("homeAway") == "home"), {})
        away_team = away.get("team", {})
        home_team = home.get("team", {})
        away_team_name = away_team.get("displayName", "")
        home_team_name = home_team.get("displayName", "")
        away_code = canonical_team_code(away_team_name) or clean_market_team_label(away_team.get("abbreviation", ""))
        home_code = canonical_team_code(home_team_name) or clean_market_team_label(home_team.get("abbreviation", ""))
        if not away_code or not home_code:
            continue
        contexts.append(
            {
                "game_id": event.get("id"),
                "matchup": f"{away_code}@{home_code}",
                "game_time": competition.get("date"),
                "status": event.get("status", {}).get("type", {}).get("description") or event.get("status", {}).get("type", {}).get("detail") or "",
                "venue": competition.get("venue", {}),
                "away_team_id": away.get("team", {}).get("id"),
                "home_team_id": home.get("team", {}).get("id"),
                "away_team_name": away_team_name or away_code,
                "home_team_name": home_team_name or home_code,
                "availability": {
                    "source": "pending_external_feed",
                    "away": [],
                    "home": [],
                },
            }
        )
    return contexts


def cache_nba_matchup_profiles(store: SnapshotStore, date_str: str, contexts: list[dict[str, Any]]) -> int:
    cached_count = 0
    for context in contexts:
        away_team_id = context.get("away_team_id")
        home_team_id = context.get("home_team_id")
        if not away_team_id or not home_team_id:
            continue
        existing = store.get_snapshot(
            source="nba_refresh",
            sport="nba",
            entity_type="matchup_profile",
            entity_key=context["matchup"],
            as_of_date=date_str,
        )
        existing_payload = existing["payload"] if existing is not None else {}
        try:
            away_profiles = fetch_nba_team_player_profiles(str(away_team_id), date_str)
        except Exception:
            away_profiles = existing_payload.get("away_profiles", [])
        try:
            home_profiles = fetch_nba_team_player_profiles(str(home_team_id), date_str)
        except Exception:
            home_profiles = existing_payload.get("home_profiles", [])
        if not away_profiles and not home_profiles:
            continue
        payload = {
            "matchup": context["matchup"],
            "away_profiles": away_profiles,
            "home_profiles": home_profiles,
        }
        store.upsert_snapshot(
            source="nba_refresh",
            sport="nba",
            entity_type="matchup_profile",
            entity_key=context["matchup"],
            as_of_date=date_str,
            payload=payload,
            is_volatile=True,
        )
        cached_count += len(away_profiles) + len(home_profiles)
    return cached_count


def latest_nba_injury_report_pdf_details(date_str: str) -> dict[str, Any] | None:
    try:
        html = requests.get(NBA_INJURY_REPORT_INDEX_URL, timeout=FAST_TIMEOUT_SECONDS).text
        exact_candidates = []
        fallback_candidates = []
        for match in NBA_INJURY_REPORT_PDF_RE.finditer(html):
            report_date, hour, minute, meridiem = match.groups()
            hour_24 = int(hour) % 12
            if meridiem == "PM":
                hour_24 += 12
            sort_key = (report_date, hour_24, int(minute))
            item = {
                "report_url": match.group(0),
                "report_date": report_date,
                "report_time": f"{hour}:{minute}{meridiem}",
                "is_stale": report_date != date_str,
            }
            if report_date == date_str:
                exact_candidates.append((sort_key, item))
            elif report_date < date_str:
                fallback_candidates.append((sort_key, item))
        if exact_candidates:
            exact_candidates.sort()
            return exact_candidates[-1][1]
        if fallback_candidates:
            fallback_candidates.sort()
            return fallback_candidates[-1][1]
    except Exception:
        pass
    probed_url = probe_nba_injury_report_pdf_url(date_str)
    if probed_url:
        return {
            "report_url": probed_url,
            "report_date": date_str,
            "report_time": None,
            "is_stale": False,
        }
    return None


def latest_nba_injury_report_pdf_url(date_str: str) -> str | None:
    details = latest_nba_injury_report_pdf_details(date_str)
    return str(details.get("report_url")) if details else None


def probe_nba_injury_report_pdf_url(date_str: str, max_attempts: int = NBA_INJURY_PROBE_ATTEMPTS) -> str | None:
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.now(UTC).date()
    if target_date.date() == today:
        cursor = datetime.now(UTC).replace(second=0, microsecond=0, tzinfo=None)
        cursor -= timedelta(minutes=cursor.minute % 15)
    else:
        cursor = target_date.replace(hour=23, minute=45)
    floor = target_date.replace(hour=0, minute=0)
    attempts = 0
    while cursor >= floor and attempts < max_attempts:
        hour_12 = cursor.strftime("%I")
        minute = cursor.strftime("%M")
        meridiem = cursor.strftime("%p")
        url = (
            f"https://ak-static.cms.nba.com/referee/injury/"
            f"Injury-Report_{date_str}_{hour_12}_{minute}{meridiem}.pdf"
        )
        try:
            response = requests.head(url, timeout=FAST_TIMEOUT_SECONDS, allow_redirects=True)
            if response.status_code == 200 and "pdf" in str(response.headers.get("content-type", "")).lower():
                return url
        except Exception:
            pass
        attempts += 1
        cursor -= timedelta(minutes=15)
    return None


def extract_pdf_text(url: str) -> str:
    content = requests.get(url, timeout=PDF_TIMEOUT_SECONDS).content
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def normalize_injury_report_text(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = re.sub(
        r"Injury Report:\s+\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}\s+(?:AM|PM)\s+Page\s+\d+\s+of\s+\d+\s+",
        " ",
        normalized,
    )
    normalized = normalized.replace(
        "Game Date Game Time Matchup Team Player Name Current Status Reason", " "
    )
    return re.sub(r"\s+", " ", normalized).strip()


def parse_team_player_statuses(section: str) -> list[dict[str, str]]:
    entries = []
    matches = list(NBA_PLAYER_STATUS_RE.finditer(section))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section)
        reason = section[start:end].strip(" -")
        entries.append(
            {
                "player_name": match.group("name").strip(),
                "status": match.group("status"),
                "reason": reason,
            }
        )
    return entries


def parse_nba_injury_report(text: str, contexts: list[dict[str, Any]], report_url: str) -> dict[str, dict[str, Any]]:
    normalized = normalize_injury_report_text(text)
    result: dict[str, dict[str, Any]] = {}
    ordered_contexts = []
    for context in contexts:
        matchup = context["matchup"]
        idx = normalized.find(matchup)
        if idx >= 0:
            ordered_contexts.append((idx, context))
    ordered_contexts.sort(key=lambda item: item[0])

    for position, (_, context) in enumerate(ordered_contexts):
        matchup = context["matchup"]
        start = normalized.find(matchup)
        if start < 0:
            continue
        end = len(normalized)
        if position + 1 < len(ordered_contexts):
            end = ordered_contexts[position + 1][0]
        segment = normalized[start + len(matchup) : end].strip()
        team_sections = {}
        team_specs = [
            ("away", context["away_team_name"]),
            ("home", context["home_team_name"]),
        ]
        team_positions = []
        for side, team_name in team_specs:
            team_idx = segment.find(team_name)
            if team_idx >= 0:
                team_positions.append((team_idx, side, team_name))
        team_positions.sort(key=lambda item: item[0])
        for team_pos, (team_idx, side, team_name) in enumerate(team_positions):
            team_end = len(segment)
            if team_pos + 1 < len(team_positions):
                team_end = team_positions[team_pos + 1][0]
            team_segment = segment[team_idx + len(team_name) : team_end].strip()
            not_submitted = team_segment.startswith("NOT YET SUBMITTED")
            if not_submitted:
                team_entries = []
            else:
                team_entries = parse_team_player_statuses(team_segment)
            team_sections[side] = {
                "team_name": team_name,
                "submitted": not not_submitted,
                "entries": team_entries,
            }
        result[matchup] = {
            "report_url": report_url,
            "availability": team_sections,
        }
    return result


def fetch_nba_injury_context(date_str: str, contexts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    report_url = latest_nba_injury_report_pdf_url(date_str)
    if report_url is None:
        return {}
    text = extract_pdf_text(report_url)
    return parse_nba_injury_report(text, contexts, report_url)


def fetch_nba_injury_context_details(date_str: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    report_details = latest_nba_injury_report_pdf_details(date_str)
    if report_details is None:
        return {
            "status": "missing",
            "message": "No NBA injury report PDF was found for this date.",
            "report_url": None,
            "report_date": None,
            "is_stale": False,
            "matched_matchups": 0,
            "expected_matchups": len(contexts),
            "submitted_teams": 0,
            "expected_teams": len(contexts) * 2,
            "parsed": {},
        }
    report_url = str(report_details["report_url"])
    try:
        text = extract_pdf_text(report_url)
    except Exception as exc:
        return {
            "status": "download_error",
            "message": f"Failed to download or parse the NBA injury report PDF: {exc}",
            "report_url": report_url,
            "report_date": report_details.get("report_date"),
            "is_stale": bool(report_details.get("is_stale")),
            "matched_matchups": 0,
            "expected_matchups": len(contexts),
            "submitted_teams": 0,
            "expected_teams": len(contexts) * 2,
            "parsed": {},
        }
    parsed = parse_nba_injury_report(text, contexts, report_url)
    submitted_teams = 0
    for matchup in parsed.values():
        availability = matchup.get("availability") or {}
        for side in ("away", "home"):
            if (availability.get(side) or {}).get("submitted"):
                submitted_teams += 1
    status = "ok" if parsed else "unmatched"
    if report_details.get("is_stale"):
        status = "stale_ok" if parsed else "stale_unmatched"
    if parsed:
        if report_details.get("is_stale"):
            message = (
                f"Using the latest prior NBA injury report from {report_details.get('report_date')} "
                "because the current slate date PDF was not available."
            )
        else:
            message = "NBA injury report fetched and parsed."
    else:
        if report_details.get("is_stale"):
            message = (
                f"Using the latest prior NBA injury report from {report_details.get('report_date')}, "
                "but no slate matchups were matched in the parsed PDF."
            )
        else:
            message = "NBA injury report was found, but no slate matchups were matched in the parsed PDF."
    return {
        "status": status,
        "message": message,
        "report_url": report_url,
        "report_date": report_details.get("report_date"),
        "is_stale": bool(report_details.get("is_stale")),
        "matched_matchups": len(parsed),
        "expected_matchups": len(contexts),
        "submitted_teams": submitted_teams,
        "expected_teams": len(contexts) * 2,
        "parsed": parsed,
    }


def build_nba_profile_availability_context(date_str: str, contexts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    derived: dict[str, dict[str, Any]] = {}
    for context in contexts:
        matchup = str(context.get("matchup") or "").strip()
        if not matchup:
            continue
        side_payloads: dict[str, dict[str, Any]] = {}
        for side, team_key in (("away", "away_team_id"), ("home", "home_team_id")):
            team_id = context.get(team_key)
            if not team_id:
                side_payloads[side] = {"submitted": False, "entries": []}
                continue
            try:
                profiles = fetch_nba_team_player_profiles(str(team_id), date_str)
            except Exception:
                profiles = []
            side_payloads[side] = {
                "submitted": bool(profiles),
                "entries": _nba_profile_availability_entries(profiles),
            }
        derived[matchup] = {
            "report_url": None,
            "availability": side_payloads,
        }
    return derived


def _nba_profile_availability_entries(team_profiles: list[dict[str, Any]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for profile in team_profiles or []:
        status = str(profile.get("status") or "").strip()
        normalized_status = status.lower()
        if normalized_status in {"", "active", "available"}:
            continue
        injuries = profile.get("injuries") or []
        reason = ""
        if injuries:
            first = injuries[0] or {}
            reason = str(first.get("detail") or first.get("type") or "").strip()
        entries.append(
            {
                "player_name": str(profile.get("name") or "").strip(),
                "status": status,
                "reason": reason,
            }
        )
    return entries


def merge_nba_availability_sources(
    parsed: dict[str, Any] | None,
    profile_fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = {
        "source": "pending_external_feed",
        "report_url": None,
        "away": [],
        "home": [],
        "away_submitted": False,
        "home_submitted": False,
    }
    if parsed:
        merged["source"] = "official_nba_injury_report_pdf"
        merged["report_url"] = parsed.get("report_url")
    if profile_fallback:
        merged["source"] = (
            "official_nba_injury_report_pdf+espn_team_profiles"
            if parsed
            else "espn_team_profiles"
        )
    for side in ("away", "home"):
        parsed_side = ((parsed or {}).get("availability") or {}).get(side) or {}
        fallback_side = ((profile_fallback or {}).get("availability") or {}).get(side) or {}
        if bool(parsed_side.get("submitted")):
            merged[side] = list(parsed_side.get("entries") or [])
            merged[f"{side}_submitted"] = True
        elif bool(fallback_side.get("submitted")):
            merged[side] = list(fallback_side.get("entries") or [])
            merged[f"{side}_submitted"] = True
    return merged


def refresh_slate(date_str: str, sport: str, db_path: str, kalshi_pages: int) -> dict:
    store = SnapshotStore(db_path)
    sports = ["mlb", "nba"] if sport == "both" else [sport]
    legs = []
    meta = {"games": 0, "kalshi_markets": 0}
    leg_refresh_error = None
    try:
        legs, meta = load_live_legs(date_str, sports=sports, kalshi_pages=kalshi_pages)
    except Exception as exc:
        leg_refresh_error = str(exc)
    context_count = 0
    player_profile_count = 0

    for selected_sport in sports:
        if selected_sport == "mlb":
            store.upsert_snapshot(
                source="mlb_statsapi",
                sport="mlb",
                entity_type="team_form",
                entity_key="daily",
                as_of_date=date_str,
                payload=fetch_live_mlb_team_form(date_str),
                is_volatile=True,
            )
            mlb_contexts = fetch_mlb_game_contexts(date_str)
            for context in mlb_contexts:
                away_team_id = context.get("away_team_id")
                home_team_id = context.get("home_team_id")
                if away_team_id:
                    context["availability"]["away"] = {
                        **fetch_mlb_team_roster_snapshot(int(away_team_id)),
                        "transactions": fetch_mlb_team_transactions(int(away_team_id), date_str),
                    }
                    context["bullpen"]["away"] = fetch_mlb_bullpen_snapshot(int(away_team_id), date_str)
                if home_team_id:
                    context["availability"]["home"] = {
                        **fetch_mlb_team_roster_snapshot(int(home_team_id)),
                        "transactions": fetch_mlb_team_transactions(int(home_team_id), date_str),
                    }
                    context["bullpen"]["home"] = fetch_mlb_bullpen_snapshot(int(home_team_id), date_str)
                context_count += 1
                store.upsert_snapshot(
                    source="mlb_refresh",
                    sport="mlb",
                    entity_type="game_context",
                    entity_key=context["matchup"],
                    as_of_date=date_str,
                    payload=context,
                    is_volatile=True,
                )
            player_profile_count += cache_mlb_matchup_profiles(store, date_str, mlb_contexts)
        if selected_sport == "nba":
            store.upsert_snapshot(
                source="nba_scoreboard",
                sport="nba",
                entity_type="team_form",
                entity_key="daily",
                as_of_date=date_str,
                payload=fetch_live_nba_team_form(date_str),
                is_volatile=True,
            )
            nba_contexts = fetch_nba_game_contexts(date_str)
            injury_context = {}
            profile_availability_context = {}
            injury_status = {
                "status": "missing",
                "message": "NBA injury refresh did not run.",
                "report_url": None,
                "matched_matchups": 0,
                "expected_matchups": len(nba_contexts),
                "submitted_teams": 0,
                "expected_teams": len(nba_contexts) * 2,
            }
            try:
                injury_details = fetch_nba_injury_context_details(date_str, nba_contexts)
                injury_context = injury_details.get("parsed", {})
                injury_status = {key: value for key, value in injury_details.items() if key != "parsed"}
            except Exception as exc:
                injury_context = {}
                injury_status = {
                    "status": "error",
                    "message": f"NBA injury refresh failed: {exc}",
                    "report_url": None,
                    "matched_matchups": 0,
                    "expected_matchups": len(nba_contexts),
                    "submitted_teams": 0,
                    "expected_teams": len(nba_contexts) * 2,
                }
            try:
                profile_availability_context = build_nba_profile_availability_context(date_str, nba_contexts)
            except Exception:
                profile_availability_context = {}
            for context in nba_contexts:
                parsed = injury_context.get(context["matchup"])
                profile_fallback = profile_availability_context.get(context["matchup"])
                merged_availability = merge_nba_availability_sources(parsed, profile_fallback)
                if merged_availability.get("source") != "pending_external_feed":
                    context["availability"] = merged_availability
                context_count += 1
                store.upsert_snapshot(
                    source="nba_refresh",
                    sport="nba",
                    entity_type="game_context",
                    entity_key=context["matchup"],
                    as_of_date=date_str,
                    payload=context,
                    is_volatile=True,
                )
            player_profile_count += cache_nba_matchup_profiles(store, date_str, nba_contexts)
            meta["nba_injury_report"] = injury_status

    store.upsert_snapshot(
        source="kalshi",
        sport=sport,
        entity_type="recognized_legs",
        entity_key="daily",
        as_of_date=date_str,
        payload={
            "legs": [leg.__dict__ for leg in legs],
            "meta": meta,
            "error": leg_refresh_error,
        },
        is_volatile=True,
    )
    return {
        "date": date_str,
        "sport": sport,
        "db_path": db_path,
        "recognized_legs": len(legs),
        "games": meta.get("games", 0),
        "kalshi_markets": meta.get("kalshi_markets", 0),
        "game_contexts": context_count,
        "player_profiles": player_profile_count,
        "leg_refresh_error": leg_refresh_error,
        "nba_injury_report": meta.get("nba_injury_report"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh volatile same-day slate data")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--sport", choices=["mlb", "nba", "both"], default="both")
    parser.add_argument("--db-path", default=SnapshotStore().db_path)
    parser.add_argument("--kalshi-pages", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = refresh_slate(args.date, args.sport, args.db_path, args.kalshi_pages)
    print(summary)


if __name__ == "__main__":
    main()
