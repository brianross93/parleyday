from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from player_name_utils import dfs_name_key


DEFAULT_API_DELAY_SECONDS = 0.6
BALL_HANDLING_TOV_KEYWORDS = (
    "lost ball",
    "lost the ball",
    "stripped",
    "fumble",
    "dribble",
    "palming",
    "double dribble",
    "discontinued dribble",
)
PASSING_TOV_KEYWORDS = (
    "bad pass",
    "pass",
    "intercept",
    "thrown away",
    "out-of-bounds",
)


def fetch_ball_handle_tracking_rows(
    season: str,
    *,
    team_abbrev: str | None = None,
    api_delay_seconds: float = DEFAULT_API_DELAY_SECONDS,
) -> list[dict[str, object]]:
    endpoints = _require_nba_api_endpoints()
    leaguedashptstats = endpoints["leaguedashptstats"]

    poss = leaguedashptstats.LeagueDashPtStats(
        pt_measure_type="Possessions",
        per_mode_simple="PerGame",
        player_or_team="Player",
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    time.sleep(api_delay_seconds)
    drives = leaguedashptstats.LeagueDashPtStats(
        pt_measure_type="Drives",
        per_mode_simple="PerGame",
        player_or_team="Player",
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    time.sleep(api_delay_seconds)

    drive_lookup = {
        _intish(record.get("PLAYER_ID")): record for record in drives.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for record in poss.to_dict(orient="records"):
        team_code = str(record.get("TEAM_ABBREVIATION") or "").upper()
        if team_abbrev and team_code != team_abbrev.upper():
            continue
        player_id = _intish(record.get("PLAYER_ID"))
        drive_record = drive_lookup.get(player_id, {})
        touches = _floatish(_first_present(record, "TOUCHES", "Touches", "touches"))
        drives_value = _floatish(_first_present(drive_record, "DRIVES", "Drives", "drives"))
        drive_tovs = _floatish(
            _first_present(
                drive_record,
                "DRIVE_TOV",
                "DRIVE_TOVS",
                "DRIVE_TOV_PER_GAME",
                "DriveTov",
                "drive_tov",
            )
        )
        rows.append(
            {
                "name": str(record.get("PLAYER_NAME") or ""),
                "name_key": dfs_name_key(str(record.get("PLAYER_NAME") or "")),
                "team_code": team_code,
                "attribute_name": "ball_handle",
                "source": "tracking",
                "features": {
                    "dribbles_per_touch": _floatish(
                        _first_present(
                            record,
                            "DRIBBLES_PER_TOUCH",
                            "AVG_DRIB_PER_TOUCH",
                            "DRIB_PER_TOUCH",
                        )
                    ),
                    "time_of_poss_per_touch": _floatish(
                        _first_present(
                            record,
                            "AVG_SEC_PER_TOUCH",
                            "AVG_TOUCH_TIME",
                            "TIME_OF_POSS",
                        )
                    ),
                    "drives_per_touch": drives_value / max(touches, 1.0),
                    "drive_tov_rate_inv": 1.0 - (drive_tovs / max(drives_value, 1.0)),
                },
            }
        )
    return rows


def fetch_passing_tracking_rows(
    season: str,
    *,
    team_abbrev: str | None = None,
    api_delay_seconds: float = DEFAULT_API_DELAY_SECONDS,
) -> list[dict[str, object]]:
    endpoints = _require_nba_api_endpoints()
    leaguedashptstats = endpoints["leaguedashptstats"]
    leaguedashplayerstats = endpoints["leaguedashplayerstats"]

    passing = leaguedashptstats.LeagueDashPtStats(
        pt_measure_type="Passing",
        per_mode_simple="PerGame",
        player_or_team="Player",
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    time.sleep(api_delay_seconds)
    advanced = leaguedashplayerstats.LeagueDashPlayerStats(
        per_mode_detailed="PerGame",
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]
    time.sleep(api_delay_seconds)
    advanced_lookup = {
        _intish(record.get("PLAYER_ID")): record for record in advanced.to_dict(orient="records")
    }

    rows: list[dict[str, object]] = []
    for record in passing.to_dict(orient="records"):
        team_code = str(record.get("TEAM_ABBREVIATION") or "").upper()
        if team_abbrev and team_code != team_abbrev.upper():
            continue
        player_id = _intish(record.get("PLAYER_ID"))
        advanced_record = advanced_lookup.get(player_id, {})
        features = {
            "passes_made": _floatish(record.get("PASSES_MADE")),
            "passes_received": _floatish(record.get("PASSES_RECEIVED")),
            "secondary_assists": _floatish(record.get("SECONDARY_AST")),
            "potential_assists": _floatish(record.get("POTENTIAL_AST")),
            "ast_points_created": _floatish(record.get("AST_POINTS_CREATED")),
            "ast_to_pass_pct": _floatish(record.get("AST_TO_PASS_PCT")),
            "usage_pct": _floatish(advanced_record.get("USG_PCT")),
            "ast_to_ratio": _floatish(advanced_record.get("AST_TO")),
        }
        for attribute_name in ("pass_vision", "pass_accuracy"):
            rows.append(
                {
                    "name": str(record.get("PLAYER_NAME") or ""),
                    "name_key": dfs_name_key(str(record.get("PLAYER_NAME") or "")),
                    "team_code": team_code,
                    "attribute_name": attribute_name,
                    "source": "tracking",
                    "features": dict(features),
                }
            )
    return rows


def fetch_shooting_tracking_rows(
    season: str,
    *,
    team_abbrev: str | None = None,
    api_delay_seconds: float = DEFAULT_API_DELAY_SECONDS,
) -> list[dict[str, object]]:
    endpoints = _require_nba_api_endpoints()
    leaguedashptstats = endpoints["leaguedashptstats"]

    catch_shoot = leaguedashptstats.LeagueDashPtStats(
        pt_measure_type="CatchShoot",
        per_mode_simple="PerGame",
        player_or_team="Player",
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    time.sleep(api_delay_seconds)
    pull_up = leaguedashptstats.LeagueDashPtStats(
        pt_measure_type="PullUpShot",
        per_mode_simple="PerGame",
        player_or_team="Player",
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    time.sleep(api_delay_seconds)
    pull_up_lookup = {
        _intish(record.get("PLAYER_ID")): record for record in pull_up.to_dict(orient="records")
    }

    rows: list[dict[str, object]] = []
    for record in catch_shoot.to_dict(orient="records"):
        team_code = str(record.get("TEAM_ABBREVIATION") or "").upper()
        if team_abbrev and team_code != team_abbrev.upper():
            continue
        player_id = _intish(record.get("PLAYER_ID"))
        pull_up_record = pull_up_lookup.get(player_id, {})
        rows.append(
            {
                "name": str(record.get("PLAYER_NAME") or ""),
                "name_key": dfs_name_key(str(record.get("PLAYER_NAME") or "")),
                "team_code": team_code,
                "attribute_name": "catch_shoot",
                "source": "tracking",
                "features": {
                    "catch_shoot_fga": _floatish(record.get("CATCH_SHOOT_FGA")),
                    "catch_shoot_fg_pct": _floatish(record.get("CATCH_SHOOT_FG_PCT")),
                    "catch_shoot_fg3a": _floatish(record.get("CATCH_SHOOT_FG3A")),
                    "catch_shoot_fg3_pct": _floatish(record.get("CATCH_SHOOT_FG3_PCT")),
                    "catch_shoot_efg_pct": _floatish(record.get("CATCH_SHOOT_EFG_PCT")),
                },
            }
        )
        rows.append(
            {
                "name": str(record.get("PLAYER_NAME") or ""),
                "name_key": dfs_name_key(str(record.get("PLAYER_NAME") or "")),
                "team_code": team_code,
                "attribute_name": "pullup_shooting",
                "source": "tracking",
                "features": {
                    "pullup_fga": _floatish(pull_up_record.get("PULL_UP_FGA")),
                    "pullup_fg_pct": _floatish(pull_up_record.get("PULL_UP_FG_PCT")),
                    "pullup_fg3a": _floatish(pull_up_record.get("PULL_UP_FG3A")),
                    "pullup_fg3_pct": _floatish(pull_up_record.get("PULL_UP_FG3_PCT")),
                    "pullup_efg_pct": _floatish(pull_up_record.get("PULL_UP_EFG_PCT")),
                },
            }
        )
    return rows


def fetch_ball_handle_pbp_rows(
    season: str,
    *,
    team_abbrev: str | None = None,
    max_games: int | None = None,
    min_classified_tovs: int = 5,
    api_delay_seconds: float = DEFAULT_API_DELAY_SECONDS,
) -> list[dict[str, object]]:
    return fetch_turnover_classification_rows(
        season,
        team_abbrev=team_abbrev,
        max_games=max_games,
        min_classified_tovs=min_classified_tovs,
        api_delay_seconds=api_delay_seconds,
        attribute_names=("ball_handle",),
    )


def fetch_turnover_classification_rows(
    season: str,
    *,
    team_abbrev: str | None = None,
    max_games: int | None = None,
    min_classified_tovs: int = 5,
    api_delay_seconds: float = DEFAULT_API_DELAY_SECONDS,
    attribute_names: tuple[str, ...] = ("ball_handle", "pass_vision", "pass_accuracy"),
) -> list[dict[str, object]]:
    endpoints = _require_nba_api_endpoints()
    leaguegamefinder = endpoints["leaguegamefinder"]
    playbyplayv3 = endpoints["playbyplayv3"]
    static_teams = endpoints["static_teams"]

    team_id = None
    if team_abbrev:
        matched = [team for team in static_teams.get_teams() if team["abbreviation"].upper() == team_abbrev.upper()]
        if matched:
            team_id = matched[0]["id"]
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        team_id_nullable=team_id,
    )
    games = finder.get_data_frames()[0]
    time.sleep(api_delay_seconds)
    game_ids = list(dict.fromkeys(games["GAME_ID"].tolist()))
    if max_games is not None:
        game_ids = game_ids[:max_games]

    counters: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"bh": 0, "pass": 0, "other": 0, "total": 0})
    for game_id in game_ids:
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, start_period=1, end_period=10)
            frame = pbp.play_by_play.get_data_frame()
        except Exception:
            try:
                frame = pbp.get_data_frames()[0]
            except Exception:
                time.sleep(api_delay_seconds)
                continue
        if frame.empty:
            time.sleep(api_delay_seconds)
            continue
        for record in frame.to_dict(orient="records"):
            action_type = str(record.get("actionType") or "").lower()
            description = str(record.get("description") or "")
            if "turnover" not in action_type and "turnover" not in description.lower():
                continue
            name = str(record.get("playerNameI") or record.get("playerName") or "")
            team_code = str(record.get("teamTricode") or "").upper()
            name_key = dfs_name_key(name)
            if not name_key or not team_code:
                continue
            bucket = classify_turnover_text(
                description=description,
                action_type=str(record.get("actionType") or ""),
                sub_type=str(record.get("subType") or ""),
            )
            entry = counters[(team_code, name_key)]
            entry["total"] += 1
            if bucket == "BALL_HANDLING":
                entry["bh"] += 1
            elif bucket == "PASSING":
                entry["pass"] += 1
            else:
                entry["other"] += 1
        time.sleep(api_delay_seconds)

    rows: list[dict[str, object]] = []
    for (team_code, name_key), counts in counters.items():
        total = counts["total"]
        if total < min_classified_tovs:
            continue
        features = {
            "bh_tov_rate_inv": 1.0 - (counts["bh"] / max(total, 1)),
            "pass_tov_rate_inv": 1.0 - (counts["pass"] / max(total, 1)),
            "classified_tovs": float(total),
        }
        for attribute_name in attribute_names:
            rows.append(
                {
                    "name_key": name_key,
                    "team_code": team_code,
                    "attribute_name": attribute_name,
                    "source": "pbp",
                    "features": dict(features),
                }
            )
    return rows


def classify_turnover_text(description: str, action_type: str = "", sub_type: str = "") -> str:
    text = f"{description} {action_type} {sub_type}".lower()
    for keyword in BALL_HANDLING_TOV_KEYWORDS:
        if keyword in text:
            return "BALL_HANDLING"
    for keyword in PASSING_TOV_KEYWORDS:
        if keyword in text:
            return "PASSING"
    return "OTHER"


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record.get(key)
        for record_key, value in record.items():
            if str(record_key).upper() == key.upper():
                return value
    return None


def _floatish(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _intish(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _require_nba_api_endpoints() -> dict[str, Any]:
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats, leaguegamefinder, playbyplayv3
        from nba_api.stats.static import teams as static_teams
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "nba_api is not installed. Run `python3 -m pip install nba_api` to fetch tracking or play-by-play attribute sources."
        ) from exc
    return {
        "leaguedashplayerstats": leaguedashplayerstats,
        "leaguedashptstats": leaguedashptstats,
        "leaguegamefinder": leaguegamefinder,
        "playbyplayv3": playbyplayv3,
        "static_teams": static_teams,
    }
