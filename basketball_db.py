from __future__ import annotations

import json
import sqlite3
import csv
import re
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import requests

from basketball_attribute_pipeline import build_attribute_scores
from data_pipeline.cache import DEFAULT_DB_PATH
from data_pipeline.nba_profiles import fetch_nba_team_player_profiles
from basketball_sim_schema import DefensiveRole, OffensiveRole, PlayerCondition, PlayerSimProfile, PlayerTraitProfile
from dfs_ingest import DraftKingsPlayer, DraftKingsSlate, draftkings_roster_slots_for_sport, parse_draftkings_salary_csv
from player_name_utils import dfs_name_key


CURRENT_SCHEMA_VERSION = 6
ESPN_NBA_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
CANONICAL_NBA_PLAYER_STATS_SOURCE = "stathead_player_season_dump"
BREF_TEAM_CODE_MAP = {
    "ATL": "ATL",
    "BKN": "BRK",
    "BOS": "BOS",
    "CHA": "CHO",
    "CHI": "CHI",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GS": "GSW",
    "HOU": "HOU",
    "IND": "IND",
    "LAC": "LAC",
    "LAL": "LAL",
    "MEM": "MEM",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NO": "NOP",
    "NY": "NYK",
    "OKC": "OKC",
    "ORL": "ORL",
    "PHI": "PHI",
    "PHX": "PHO",
    "POR": "POR",
    "SA": "SAS",
    "SAC": "SAC",
    "TOR": "TOR",
    "UTAH": "UTA",
    "WSH": "WAS",
}
REVERSE_BREF_TEAM_CODE_MAP = {value: key for key, value in BREF_TEAM_CODE_MAP.items()}
STATHEAD_HEADER_PREFIX = "Rk,Player,"


def ensure_basketball_schema(db_path: str = DEFAULT_DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        applied_versions = {
            int(row["version"])
            for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        for version, name, sql in _migrations():
            if version in applied_versions:
                continue
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, datetime.now(timezone.utc).isoformat()),
            )


def import_draftkings_slate_to_db(
    date_str: str,
    csv_path: str,
    *,
    sport: str = "nba",
    db_path: str = DEFAULT_DB_PATH,
) -> DraftKingsSlate:
    ensure_basketball_schema(db_path)
    slate = parse_draftkings_salary_csv(csv_path, sport=sport)
    imported_at = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row
        slate_id = _upsert_slate_row(conn, date_str=date_str, slate=slate, imported_at=imported_at)
        for player in slate.players:
            player_key = _player_key(player)
            conn.execute(
                """
                INSERT INTO basketball_players (
                    player_key, dk_player_id, name, name_key, sport, team_code,
                    default_positions_json, default_roster_positions_json,
                    avg_points_per_game, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_key) DO UPDATE SET
                    dk_player_id = excluded.dk_player_id,
                    name = excluded.name,
                    sport = excluded.sport,
                    team_code = excluded.team_code,
                    default_positions_json = excluded.default_positions_json,
                    default_roster_positions_json = excluded.default_roster_positions_json,
                    avg_points_per_game = excluded.avg_points_per_game,
                    updated_at = excluded.updated_at
                """,
                (
                    player_key,
                    player.player_id,
                    player.name,
                    dfs_name_key(player.name),
                    slate.sport,
                    player.team,
                    json.dumps(player.positions),
                    json.dumps(player.roster_positions),
                    float(player.avg_points_per_game),
                    imported_at,
                    imported_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO basketball_slate_players (
                    slate_id, player_key, dk_player_id, name, team_code, opponent_code, game_code,
                    start_time, salary, positions_json, roster_positions_json, avg_points_per_game,
                    raw_position, raw_game_info, is_active, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(slate_id, player_key) DO UPDATE SET
                    dk_player_id = excluded.dk_player_id,
                    name = excluded.name,
                    team_code = excluded.team_code,
                    opponent_code = excluded.opponent_code,
                    game_code = excluded.game_code,
                    start_time = excluded.start_time,
                    salary = excluded.salary,
                    positions_json = excluded.positions_json,
                    roster_positions_json = excluded.roster_positions_json,
                    avg_points_per_game = excluded.avg_points_per_game,
                    raw_position = excluded.raw_position,
                    raw_game_info = excluded.raw_game_info,
                    is_active = excluded.is_active,
                    imported_at = excluded.imported_at
                """,
                (
                    slate_id,
                    player_key,
                    player.player_id,
                    player.name,
                    player.team,
                    player.opponent,
                    player.game,
                    player.start_time,
                    int(player.salary),
                    json.dumps(player.positions),
                    json.dumps(player.roster_positions),
                    float(player.avg_points_per_game),
                    player.raw_position,
                    player.raw_game_info,
                    imported_at,
                ),
            )
    return load_draftkings_slate_from_db(date_str, sport=sport, db_path=db_path, source_path=str(Path(csv_path)))


def load_draftkings_slate_from_db(
    date_str: str,
    *,
    sport: str = "nba",
    db_path: str = DEFAULT_DB_PATH,
    source_path: str | None = None,
) -> DraftKingsSlate:
    ensure_basketball_schema(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT s.id, s.source_path, s.site, s.sport
            FROM basketball_slates s
            WHERE s.slate_date = ? AND s.sport = ?
        """
        params: list[object] = [date_str, sport]
        if source_path:
            query += " AND s.source_path = ?"
            params.append(source_path)
        query += " ORDER BY s.imported_at DESC LIMIT 1"
        slate_row = conn.execute(query, tuple(params)).fetchone()
        if slate_row is None:
            raise ValueError(f"No {sport} DraftKings slate found in DB for {date_str}")
        player_rows = conn.execute(
            """
            SELECT *
            FROM basketball_slate_players
            WHERE slate_id = ? AND is_active = 1
            ORDER BY salary DESC, name ASC
            """,
            (int(slate_row["id"]),),
        ).fetchall()
    players = tuple(_row_to_dk_player(row, sport=sport) for row in player_rows)
    return DraftKingsSlate(
        site=str(slate_row["site"]),
        sport=str(slate_row["sport"]),
        salary_cap=50000,
        roster_slots=draftkings_roster_slots_for_sport(sport),
        players=players,
        source_path=str(slate_row["source_path"]),
    )


def seed_nba_player_stats_from_espn(
    as_of_date: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    last_n_games: int = 82,
) -> int:
    ensure_basketball_schema(db_path)
    imported_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for team in fetch_espn_nba_teams():
        team_id = str(team.get("team_id") or "")
        if not team_id:
            continue
        profiles = fetch_nba_team_player_profiles(team_id, as_of_date, last_n_games=last_n_games)
        for profile in profiles:
            rows.append(
                {
                    "player_id": str(profile.get("player_id") or ""),
                    "name": str(profile.get("name") or ""),
                    "name_key": dfs_name_key(str(profile.get("name") or "")),
                    "team_code": str(team.get("team_code") or ""),
                    "team_id": team_id,
                    "position": str(profile.get("position") or ""),
                    "status": str(profile.get("status") or "active"),
                    "games_sample": float(profile.get("games_sample") or 0.0),
                    "starts": float(profile.get("starts") or 0.0),
                    "minutes": float(profile.get("minutes") or 0.0),
                    "points": float(profile.get("points") or 0.0),
                    "rebounds": float(profile.get("rebounds") or 0.0),
                    "assists": float(profile.get("assists") or 0.0),
                    "turnovers": float(profile.get("turnovers") or 0.0),
                    "fouls": float(profile.get("fouls") or 0.0),
                    "fga": float(profile.get("fga") or 0.0),
                    "three_pa": float(profile.get("three_pa") or 0.0),
                    "fta": float(profile.get("fta") or 0.0),
                    "oreb": float(profile.get("oreb") or 0.0),
                    "dreb": float(profile.get("dreb") or 0.0),
                    "recent_fpts_avg": float(profile.get("recent_fpts_avg") or 0.0),
                    "recent_fpts_weighted": float(profile.get("recent_fpts_weighted") or 0.0),
                    "recent_form_delta": float(profile.get("recent_form_delta") or 0.0),
                    "injuries_json": json.dumps(profile.get("injuries") or []),
                }
            )
    upsert_nba_player_stats(as_of_date, rows, db_path=db_path, imported_at=imported_at, source="espn_team_profiles")
    return len(rows)


def seed_nba_player_stats_from_bref(
    as_of_date: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    season_end_year: int | None = None,
) -> int:
    ensure_basketball_schema(db_path)
    imported_at = datetime.now(timezone.utc).isoformat()
    target_season_end_year = season_end_year or _season_end_year_for_date(as_of_date)
    rows: list[dict[str, object]] = []
    get_roster_stats = _load_bref_get_roster_stats()

    for team_code, bref_team_code in BREF_TEAM_CODE_MAP.items():
        frame = get_roster_stats(bref_team_code, target_season_end_year, data_format="PER_GAME")
        if frame is None or getattr(frame, "empty", True):
            continue
        for record in frame.to_dict(orient="records"):
            name = str(record.get("PLAYER") or "").strip()
            if not name:
                continue
            points = _float_from_record(record, "PTS")
            rebounds = _float_from_record(record, "TRB")
            assists = _float_from_record(record, "AST")
            steals = _float_from_record(record, "STL")
            blocks = _float_from_record(record, "BLK")
            turnovers = _float_from_record(record, "TOV")
            rows.append(
                {
                    "player_id": "",
                    "name": name,
                    "name_key": dfs_name_key(name),
                    "team_code": team_code,
                    "team_id": bref_team_code,
                    "position": str(record.get("POS") or ""),
                    "status": "active",
                    "games_sample": _float_from_record(record, "G"),
                    "starts": _float_from_record(record, "GS"),
                    "minutes": _float_from_record(record, "MP"),
                    "points": points,
                    "rebounds": rebounds,
                    "assists": assists,
                    "turnovers": turnovers,
                    "fouls": _float_from_record(record, "PF"),
                    "fga": _float_from_record(record, "FGA"),
                    "three_pa": _float_from_record(record, "3PA"),
                    "fta": _float_from_record(record, "FTA"),
                    "oreb": _float_from_record(record, "ORB"),
                    "dreb": _float_from_record(record, "DRB"),
                    "recent_fpts_avg": _estimated_fantasy_points(points, rebounds, assists, steals, blocks, turnovers),
                    "recent_fpts_weighted": _estimated_fantasy_points(points, rebounds, assists, steals, blocks, turnovers),
                    "recent_form_delta": 0.0,
                    "injuries_json": json.dumps([]),
                }
            )
    upsert_nba_player_stats(as_of_date, rows, db_path=db_path, imported_at=imported_at, source="basketball_reference_per_game")
    return len(rows)


def seed_nba_player_stats_from_stathead_dump(
    as_of_date: str,
    dump_path: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    ensure_basketball_schema(db_path)
    imported_at = datetime.now(timezone.utc).isoformat()
    path = Path(dump_path)
    rows = _parse_stathead_dump_rows(path)
    _replace_nba_player_stats_snapshot(as_of_date, db_path=db_path)
    upsert_nba_player_stats(as_of_date, rows, db_path=db_path, imported_at=imported_at, source="stathead_player_season_dump")
    return len(rows)


def seed_ball_handle_tracking_features(
    as_of_date: str,
    season: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    team_abbrev: str | None = None,
) -> int:
    from data_pipeline.nba_attribute_sources import fetch_ball_handle_tracking_rows

    rows = fetch_ball_handle_tracking_rows(season, team_abbrev=team_abbrev)
    return upsert_player_attribute_source_features(as_of_date, rows, db_path=db_path)


def seed_ball_handle_pbp_features(
    as_of_date: str,
    season: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    team_abbrev: str | None = None,
    max_games: int | None = None,
    min_classified_tovs: int = 5,
) -> int:
    from data_pipeline.nba_attribute_sources import fetch_ball_handle_pbp_rows

    rows = fetch_ball_handle_pbp_rows(
        season,
        team_abbrev=team_abbrev,
        max_games=max_games,
        min_classified_tovs=min_classified_tovs,
    )
    return upsert_player_attribute_source_features(as_of_date, rows, db_path=db_path)


def seed_turnover_classification_features(
    as_of_date: str,
    season: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    team_abbrev: str | None = None,
    max_games: int | None = None,
    min_classified_tovs: int = 5,
) -> int:
    from data_pipeline.nba_attribute_sources import fetch_turnover_classification_rows

    rows = fetch_turnover_classification_rows(
        season,
        team_abbrev=team_abbrev,
        max_games=max_games,
        min_classified_tovs=min_classified_tovs,
    )
    return upsert_player_attribute_source_features(as_of_date, rows, db_path=db_path)


def seed_turnover_classification_features_chunked(
    as_of_date: str,
    season: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    team_abbrevs: list[str] | None = None,
    max_games: int | None = None,
    min_classified_tovs: int = 5,
) -> dict[str, object]:
    try:
        from nba_api.stats.static import teams as static_teams
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "nba_api is not installed. Run `python3 -m pip install nba_api` to fetch play-by-play attribute sources."
        ) from exc

    selected_teams = team_abbrevs or sorted(
        str(team["abbreviation"]).upper()
        for team in static_teams.get_teams()
        if team.get("abbreviation")
    )
    imported_rows = 0
    completed: list[str] = []
    failed: dict[str, str] = {}
    for team_abbrev in selected_teams:
        try:
            imported_rows += seed_turnover_classification_features(
                as_of_date,
                season,
                db_path=db_path,
                team_abbrev=team_abbrev,
                max_games=max_games,
                min_classified_tovs=min_classified_tovs,
            )
            completed.append(team_abbrev)
        except Exception as exc:  # pragma: no cover - network dependent
            failed[team_abbrev] = str(exc)
    return {
        "teams_completed": completed,
        "teams_failed": failed,
        "rows_imported": imported_rows,
    }


def seed_passing_tracking_features(
    as_of_date: str,
    season: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    team_abbrev: str | None = None,
) -> int:
    from data_pipeline.nba_attribute_sources import fetch_passing_tracking_rows

    rows = fetch_passing_tracking_rows(season, team_abbrev=team_abbrev)
    return upsert_player_attribute_source_features(as_of_date, rows, db_path=db_path)


def seed_shooting_tracking_features(
    as_of_date: str,
    season: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    team_abbrev: str | None = None,
) -> int:
    from data_pipeline.nba_attribute_sources import fetch_shooting_tracking_rows

    rows = fetch_shooting_tracking_rows(season, team_abbrev=team_abbrev)
    return upsert_player_attribute_source_features(as_of_date, rows, db_path=db_path)


def seed_nba_player_season_history_from_nba_api(
    seasons: list[str],
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    ensure_basketball_schema(db_path)
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "nba_api is not installed. Run `python3 -m pip install nba_api` to fetch historical season data."
        ) from exc

    imported_at = datetime.now(timezone.utc).isoformat()
    total_rows = 0
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row
        for season in seasons:
            frame = leaguedashplayerstats.LeagueDashPlayerStats(
                per_mode_detailed="PerGame",
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Base",
            ).get_data_frames()[0]
            rows = frame.to_dict(orient="records")
            for record in rows:
                name = str(record.get("PLAYER_NAME") or "").strip()
                team_code = _normalize_external_team_code(str(record.get("TEAM_ABBREVIATION") or ""))
                name_key = dfs_name_key(name)
                if not name or not name_key or not team_code:
                    continue
                player_id = str(record.get("PLAYER_ID") or "")
                gp = float(record.get("GP") or 0.0)
                stats_row = {
                    "season": season,
                    "player_key": _profile_key(name_key, team_code),
                    "player_id": player_id,
                    "name": name,
                    "name_key": name_key,
                    "team_code": team_code,
                    "team_id": str(record.get("TEAM_ID") or ""),
                    "age": float(record.get("AGE") or 0.0),
                    "position": "",
                    "games_sample": gp,
                    "starts": 0.0,
                    "minutes": float(record.get("MIN") or 0.0) * gp,
                    "points": float(record.get("PTS") or 0.0) * gp,
                    "rebounds": float(record.get("REB") or 0.0) * gp,
                    "assists": float(record.get("AST") or 0.0) * gp,
                    "recent_fpts_avg": float(record.get("NBA_FANTASY_PTS") or 0.0),
                    "recent_fpts_weighted": float(record.get("NBA_FANTASY_PTS") or 0.0),
                    "recent_form_delta": 0.0,
                    "extra_stats_json": json.dumps(
                        {
                            "turnovers": float(record.get("TOV") or 0.0) * gp,
                            "fouls": float(record.get("PF") or 0.0) * gp,
                            "fga": float(record.get("FGA") or 0.0) * gp,
                            "three_pa": float(record.get("FG3A") or 0.0) * gp,
                            "fta": float(record.get("FTA") or 0.0) * gp,
                            "oreb": float(record.get("OREB") or 0.0) * gp,
                            "dreb": float(record.get("DREB") or 0.0) * gp,
                            "fg_pct": float(record.get("FG_PCT") or 0.0),
                            "fg3_pct": float(record.get("FG3_PCT") or 0.0),
                            "ft_pct": float(record.get("FT_PCT") or 0.0),
                            "stl": float(record.get("STL") or 0.0) * gp,
                            "blk": float(record.get("BLK") or 0.0) * gp,
                            "pfd": float(record.get("PFD") or 0.0) * gp,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "source": "nba_api_leaguedashplayerstats_base",
                    "imported_at": imported_at,
                }
                conn.execute(
                    """
                    INSERT INTO basketball_player_season_stats (
                        season, player_key, player_id, name, name_key, team_code, team_id, age, position,
                        games_sample, minutes, points, rebounds, assists,
                        recent_fpts_avg, recent_fpts_weighted, recent_form_delta,
                        extra_stats_json, source, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(season, name_key, team_code) DO UPDATE SET
                        player_key = excluded.player_key,
                        player_id = excluded.player_id,
                        name = excluded.name,
                        team_id = excluded.team_id,
                        age = excluded.age,
                        position = excluded.position,
                        games_sample = excluded.games_sample,
                        minutes = excluded.minutes,
                        points = excluded.points,
                        rebounds = excluded.rebounds,
                        assists = excluded.assists,
                        recent_fpts_avg = excluded.recent_fpts_avg,
                        recent_fpts_weighted = excluded.recent_fpts_weighted,
                        recent_form_delta = excluded.recent_form_delta,
                        extra_stats_json = excluded.extra_stats_json,
                        source = excluded.source,
                        imported_at = excluded.imported_at
                    """,
                    (
                        stats_row["season"],
                        stats_row["player_key"],
                        stats_row["player_id"],
                        stats_row["name"],
                        stats_row["name_key"],
                        stats_row["team_code"],
                        stats_row["team_id"],
                        stats_row["age"],
                        stats_row["position"],
                        stats_row["games_sample"],
                        stats_row["minutes"],
                        stats_row["points"],
                        stats_row["rebounds"],
                        stats_row["assists"],
                        stats_row["recent_fpts_avg"],
                        stats_row["recent_fpts_weighted"],
                        stats_row["recent_form_delta"],
                        stats_row["extra_stats_json"],
                        stats_row["source"],
                        stats_row["imported_at"],
                    ),
                )
                total_rows += 1
    return total_rows


def seed_nba_player_season_advanced_history_from_nba_api(
    seasons: list[str],
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    ensure_basketball_schema(db_path)
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "nba_api is not installed. Run `python3 -m pip install nba_api` to fetch historical advanced season data."
        ) from exc

    imported_at = datetime.now(timezone.utc).isoformat()
    total_rows = 0
    with closing(sqlite3.connect(db_path)) as conn, conn:
        for season in seasons:
            frame = leaguedashplayerstats.LeagueDashPlayerStats(
                per_mode_detailed="PerGame",
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Advanced",
            ).get_data_frames()[0]
            for record in frame.to_dict(orient="records"):
                name = str(record.get("PLAYER_NAME") or "").strip()
                team_code = _normalize_external_team_code(str(record.get("TEAM_ABBREVIATION") or ""))
                name_key = dfs_name_key(name)
                if not name or not name_key or not team_code:
                    continue
                metrics = {
                    "age": float(record.get("AGE") or 0.0),
                    "gp": float(record.get("GP") or 0.0),
                    "min": float(record.get("MIN") or 0.0),
                    "off_rating": float(record.get("OFF_RATING") or 0.0),
                    "def_rating": float(record.get("DEF_RATING") or 0.0),
                    "net_rating": float(record.get("NET_RATING") or 0.0),
                    "ast_pct": float(record.get("AST_PCT") or 0.0),
                    "ast_to": float(record.get("AST_TO") or 0.0),
                    "ast_ratio": float(record.get("AST_RATIO") or 0.0),
                    "oreb_pct": float(record.get("OREB_PCT") or 0.0),
                    "dreb_pct": float(record.get("DREB_PCT") or 0.0),
                    "reb_pct": float(record.get("REB_PCT") or 0.0),
                    "tm_tov_pct": float(record.get("TM_TOV_PCT") or 0.0),
                    "e_tov_pct": float(record.get("E_TOV_PCT") or 0.0),
                    "efg_pct": float(record.get("EFG_PCT") or 0.0),
                    "ts_pct": float(record.get("TS_PCT") or 0.0),
                    "usg_pct": float(record.get("USG_PCT") or 0.0),
                    "pace": float(record.get("PACE") or 0.0),
                    "pie": float(record.get("PIE") or 0.0),
                }
                conn.execute(
                    """
                    INSERT INTO basketball_player_season_advanced_stats (
                        season, player_key, player_id, name, name_key, team_code, team_id,
                        metrics_json, source, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(season, name_key, team_code) DO UPDATE SET
                        player_key = excluded.player_key,
                        player_id = excluded.player_id,
                        name = excluded.name,
                        team_id = excluded.team_id,
                        metrics_json = excluded.metrics_json,
                        source = excluded.source,
                        imported_at = excluded.imported_at
                    """,
                    (
                        season,
                        _profile_key(name_key, team_code),
                        str(record.get("PLAYER_ID") or ""),
                        name,
                        name_key,
                        team_code,
                        str(record.get("TEAM_ID") or ""),
                        json.dumps(metrics, separators=(",", ":"), sort_keys=True),
                        "nba_api_leaguedashplayerstats_advanced",
                        imported_at,
                    ),
                )
                total_rows += 1
    return total_rows


def generate_nba_player_profiles_for_season(
    season: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    ensure_basketball_schema(db_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM basketball_player_season_stats
            WHERE season = ?
            ORDER BY team_code, name_key
            """,
            (season,),
        ).fetchall()
        normalized_rows = [_season_stats_sqlite_row_to_dict(row) for row in rows]
        framework_scores = build_attribute_scores(normalized_rows)
        prior_lookup = _load_prior_profile_lookup(conn, season)
        advanced_lookup = _load_advanced_context_lookup(conn, season)
        count = 0
        for row, normalized in zip(rows, normalized_rows):
            row_key = (str(normalized.get("team_code") or ""), str(normalized.get("name_key") or ""))
            profile = _season_stats_row_to_profile(row, framework_scores.get(row_key, {}))
            profile = _apply_progression_blend(
                profile,
                age=float(row["age"] or 0.0),
                prior_context=prior_lookup.get(_profile_history_key(str(row["player_id"] or ""), str(row["name_key"] or ""))),
                advanced_context=advanced_lookup.get(_profile_history_key(str(row["player_id"] or ""), str(row["name_key"] or ""))),
            )
            conn.execute(
                """
                INSERT INTO basketball_player_season_profiles (
                    season, profile_key, player_key, player_id, name, name_key, team_code, age, position,
                    offensive_role, defensive_role, profile_version, traits_json, condition_json, source, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(season, name_key, team_code) DO UPDATE SET
                    profile_key = excluded.profile_key,
                    player_key = excluded.player_key,
                    player_id = excluded.player_id,
                    name = excluded.name,
                    age = excluded.age,
                    position = excluded.position,
                    offensive_role = excluded.offensive_role,
                    defensive_role = excluded.defensive_role,
                    profile_version = excluded.profile_version,
                    traits_json = excluded.traits_json,
                    condition_json = excluded.condition_json,
                    source = excluded.source,
                    generated_at = excluded.generated_at
                """,
                (
                    season,
                    _profile_key(str(row["name_key"]), str(row["team_code"])),
                    str(row["player_key"] or ""),
                    str(row["player_id"] or ""),
                    str(row["name"] or ""),
                    str(row["name_key"] or ""),
                    str(row["team_code"] or ""),
                    float(row["age"] or 0.0),
                    str(row["position"] or ""),
                    profile.offensive_role.value,
                    profile.defensive_role.value,
                    CURRENT_SCHEMA_VERSION,
                    json.dumps(_traits_dict(profile.traits), separators=(",", ":"), sort_keys=True),
                    json.dumps(_condition_dict(profile.condition), separators=(",", ":"), sort_keys=True),
                    str(row["source"] or ""),
                    generated_at,
                ),
            )
            count += 1
    return count


def upsert_nba_player_stats(
    as_of_date: str,
    rows: list[dict[str, object]],
    *,
    db_path: str = DEFAULT_DB_PATH,
    imported_at: str | None = None,
    source: str = "manual",
) -> int:
    ensure_basketball_schema(db_path)
    timestamp = imported_at or datetime.now(timezone.utc).isoformat()
    count = 0
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row
        for row in rows:
            name = str(row.get("name") or "")
            name_key = str(row.get("name_key") or dfs_name_key(name))
            team_code = str(row.get("team_code") or "")
            player_id = str(row.get("player_id") or "")
            if not name_key or not team_code:
                continue
            player_key = f"espn:{player_id}" if player_id else f"profile:{team_code}:{name_key}"
            conn.execute(
                """
                INSERT INTO basketball_players (
                    player_key, dk_player_id, name, name_key, sport, team_code,
                    default_positions_json, default_roster_positions_json,
                    avg_points_per_game, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_key) DO UPDATE SET
                    name = excluded.name,
                    name_key = excluded.name_key,
                    sport = excluded.sport,
                    team_code = excluded.team_code,
                    default_positions_json = excluded.default_positions_json,
                    default_roster_positions_json = excluded.default_roster_positions_json,
                    avg_points_per_game = excluded.avg_points_per_game,
                    updated_at = excluded.updated_at
                """,
                (
                    player_key,
                    None,
                    name,
                    name_key,
                    "nba",
                    team_code,
                    json.dumps(_position_tuple(str(row.get("position") or ""))),
                    json.dumps(_position_tuple(str(row.get("position") or ""))),
                    float(row.get("points") or 0.0),
                    timestamp,
                    timestamp,
                ),
            )
            conn.execute(
                """
                INSERT INTO basketball_player_stats (
                    as_of_date, player_key, player_id, name, name_key, team_code, team_id,
                    position, status, games_sample, minutes, points, rebounds, assists,
                    recent_fpts_avg, recent_fpts_weighted, recent_form_delta, injuries_json,
                    extra_stats_json,
                    source, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(as_of_date, name_key, team_code) DO UPDATE SET
                    player_key = excluded.player_key,
                    player_id = excluded.player_id,
                    name = excluded.name,
                    team_id = excluded.team_id,
                    position = excluded.position,
                    status = excluded.status,
                    games_sample = excluded.games_sample,
                    minutes = excluded.minutes,
                    points = excluded.points,
                    rebounds = excluded.rebounds,
                    assists = excluded.assists,
                    recent_fpts_avg = excluded.recent_fpts_avg,
                    recent_fpts_weighted = excluded.recent_fpts_weighted,
                    recent_form_delta = excluded.recent_form_delta,
                    injuries_json = excluded.injuries_json,
                    extra_stats_json = excluded.extra_stats_json,
                    source = excluded.source,
                    imported_at = excluded.imported_at
                """,
                (
                    as_of_date,
                    player_key,
                    player_id,
                    name,
                    name_key,
                    team_code,
                    str(row.get("team_id") or ""),
                    str(row.get("position") or ""),
                    str(row.get("status") or "active"),
                    float(row.get("games_sample") or 0.0),
                    float(row.get("minutes") or 0.0),
                    float(row.get("points") or 0.0),
                    float(row.get("rebounds") or 0.0),
                    float(row.get("assists") or 0.0),
                    float(row.get("recent_fpts_avg") or 0.0),
                    float(row.get("recent_fpts_weighted") or 0.0),
                    float(row.get("recent_form_delta") or 0.0),
                    str(row.get("injuries_json") or "[]"),
                    json.dumps(
                        {
                            "starts": float(row.get("starts") or 0.0),
                            "turnovers": float(row.get("turnovers") or 0.0),
                            "fouls": float(row.get("fouls") or 0.0),
                            "fga": float(row.get("fga") or 0.0),
                            "three_pa": float(row.get("three_pa") or 0.0),
                            "fta": float(row.get("fta") or 0.0),
                            "oreb": float(row.get("oreb") or 0.0),
                            "dreb": float(row.get("dreb") or 0.0),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    source,
                    timestamp,
                ),
            )
            count += 1
    return count


def upsert_player_attribute_source_features(
    as_of_date: str,
    rows: list[dict[str, object]],
    *,
    db_path: str = DEFAULT_DB_PATH,
    imported_at: str | None = None,
) -> int:
    ensure_basketball_schema(db_path)
    timestamp = imported_at or datetime.now(timezone.utc).isoformat()
    count = 0
    with closing(sqlite3.connect(db_path)) as conn, conn:
        for row in rows:
            name = str(row.get("name") or "")
            name_key = str(row.get("name_key") or dfs_name_key(name))
            team_code = _normalize_external_team_code(str(row.get("team_code") or ""))
            attribute_name = str(row.get("attribute_name") or "").strip()
            source = str(row.get("source") or "").strip()
            features = row.get("features")
            if not name_key or not team_code or not attribute_name or not source or not isinstance(features, dict):
                continue
            conn.execute(
                """
                INSERT INTO basketball_player_attribute_features (
                    as_of_date, name_key, team_code, attribute_name, source, features_json, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(as_of_date, name_key, team_code, attribute_name, source) DO UPDATE SET
                    features_json = excluded.features_json,
                    imported_at = excluded.imported_at
                """,
                (
                    as_of_date,
                    name_key,
                    team_code,
                    attribute_name,
                    source,
                    json.dumps(features, separators=(",", ":"), sort_keys=True),
                    timestamp,
                ),
            )
            count += 1
    return count


def _normalize_external_team_code(raw_team_code: str) -> str:
    code = str(raw_team_code or "").strip().upper()
    return REVERSE_BREF_TEAM_CODE_MAP.get(code, code)


def _replace_nba_player_stats_snapshot(as_of_date: str, *, db_path: str = DEFAULT_DB_PATH) -> None:
    ensure_basketball_schema(db_path)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM basketball_player_profiles WHERE as_of_date = ?", (as_of_date,))
        conn.execute("DELETE FROM basketball_player_stats WHERE as_of_date = ?", (as_of_date,))


def _load_bref_get_roster_stats():
    try:
        from basketball_reference_scraper.teams import get_roster_stats
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "basketball-reference-scraper is not installed. Run `python3 -m pip install basketball-reference-scraper` first."
        ) from exc
    return get_roster_stats


def _season_end_year_for_date(as_of_date: str) -> int:
    dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    return dt.year if dt.month <= 6 else dt.year + 1


def _float_from_record(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _estimated_fantasy_points(points: float, rebounds: float, assists: float, steals: float, blocks: float, turnovers: float) -> float:
    return (
        points
        + (rebounds * 1.25)
        + (assists * 1.5)
        + ((steals + blocks) * 2.0)
        - (turnovers * 0.5)
    )


def _parse_stathead_dump_rows(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    filtered_lines: list[str] = []
    header_line: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("--- When using SR data"):
            continue
        if line.startswith(STATHEAD_HEADER_PREFIX):
            if header_line is None:
                header_line = line
                filtered_lines.append(line)
            continue
        if re.match(r"^\d+,", line):
            filtered_lines.append(line)
    if header_line is None:
        raise ValueError(f"No Stathead CSV header found in {path}")
    reader = csv.DictReader(filtered_lines)
    rows: list[dict[str, object]] = []
    for record in reader:
        name = str(record.get("Player") or "").strip()
        if not name:
            continue
        team_code = _normalize_stathead_team_code(str(record.get("Team") or ""))
        if not team_code:
            continue
        points = _csv_float(record.get("PTS"))
        rebounds = _csv_float(record.get("TRB"))
        assists = _csv_float(record.get("AST"))
        steals = _csv_float(record.get("STL"))
        blocks = _csv_float(record.get("BLK"))
        turnovers = _csv_float(record.get("TOV"))
        player_id = str(record.get("Player-additional") or "").strip()
        rows.append(
            {
                "player_id": player_id,
                "name": name,
                "name_key": dfs_name_key(name),
                "team_code": team_code,
                "team_id": team_code,
                "position": _normalize_stat_position(str(record.get("Pos") or "")),
                "status": "active",
                "games_sample": _csv_float(record.get("G")),
                "starts": _csv_float(record.get("GS")),
                "minutes": _csv_float(record.get("MP")),
                "points": points,
                "rebounds": rebounds,
                "assists": assists,
                "turnovers": turnovers,
                "fouls": _csv_float(record.get("PF")),
                "fga": _csv_float(record.get("FGA")),
                "three_pa": _csv_float(record.get("3PA")),
                "fta": _csv_float(record.get("FTA")),
                "oreb": _csv_float(record.get("ORB")),
                "dreb": _csv_float(record.get("DRB")),
                "recent_fpts_avg": _estimated_fantasy_points(points, rebounds, assists, steals, blocks, turnovers),
                "recent_fpts_weighted": _estimated_fantasy_points(points, rebounds, assists, steals, blocks, turnovers),
                "recent_form_delta": 0.0,
                "injuries_json": json.dumps([]),
            }
        )
    return rows


def _csv_float(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_stathead_team_code(raw_team_code: str) -> str:
    code = str(raw_team_code or "").strip().upper()
    if not code:
        return ""
    if code in REVERSE_BREF_TEAM_CODE_MAP:
        return REVERSE_BREF_TEAM_CODE_MAP[code]
    if code in BREF_TEAM_CODE_MAP:
        return code
    for bref_code in sorted(REVERSE_BREF_TEAM_CODE_MAP.keys(), key=len, reverse=True):
        if code.endswith(bref_code):
            return REVERSE_BREF_TEAM_CODE_MAP[bref_code]
    return code


def generate_nba_player_profiles(as_of_date: str, *, db_path: str = DEFAULT_DB_PATH) -> int:
    ensure_basketball_schema(db_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row
        available_sources = {
            str(row["source"] or "")
            for row in conn.execute(
                """
                SELECT DISTINCT source
                FROM basketball_player_stats
                WHERE as_of_date = ?
                """,
                (as_of_date,),
            ).fetchall()
        }
        source_filter = CANONICAL_NBA_PLAYER_STATS_SOURCE if CANONICAL_NBA_PLAYER_STATS_SOURCE in available_sources else None
        rows = conn.execute(
            """
            SELECT *
            FROM basketball_player_stats
            WHERE as_of_date = ?
              AND (? IS NULL OR source = ?)
            """,
            (as_of_date, source_filter, source_filter),
        ).fetchall()
        supplemental_feature_rows = conn.execute(
            """
            SELECT as_of_date, name_key, team_code, attribute_name, source, features_json
            FROM basketball_player_attribute_features
            WHERE as_of_date = ?
            """,
            (as_of_date,),
        ).fetchall()
        supplemental_map = _group_attribute_source_features(supplemental_feature_rows)
        normalized_rows = [_stats_sqlite_row_to_dict(row, supplemental_map) for row in rows]
        framework_scores = build_attribute_scores(normalized_rows)
        current_season = _season_for_date(as_of_date)
        prior_lookup = _load_prior_profile_lookup(conn, current_season)
        advanced_lookup = _load_advanced_context_lookup(conn, current_season)
        count = 0
        for row, normalized in zip(rows, normalized_rows):
            row_key = (str(normalized.get("team_code") or ""), str(normalized.get("name_key") or ""))
            profile = _stats_row_to_profile(row, framework_scores.get(row_key, {}))
            profile = _apply_progression_blend(
                profile,
                age=_safe_float_sql(row["age"]) if "age" in row.keys() else 0.0,
                prior_context=prior_lookup.get(_profile_history_key(str(row["player_id"] or ""), str(row["name_key"] or ""))),
                advanced_context=advanced_lookup.get(_profile_history_key(str(row["player_id"] or ""), str(row["name_key"] or ""))),
            )
            conn.execute(
                """
                INSERT INTO basketball_player_profiles (
                    profile_key, as_of_date, player_key, player_id, name, name_key, team_code, position,
                    offensive_role, defensive_role, profile_version, traits_json, condition_json, source, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_key) DO UPDATE SET
                    as_of_date = excluded.as_of_date,
                    player_key = excluded.player_key,
                    player_id = excluded.player_id,
                    name = excluded.name,
                    team_code = excluded.team_code,
                    position = excluded.position,
                    offensive_role = excluded.offensive_role,
                    defensive_role = excluded.defensive_role,
                    profile_version = excluded.profile_version,
                    traits_json = excluded.traits_json,
                    condition_json = excluded.condition_json,
                    source = excluded.source,
                    generated_at = excluded.generated_at
                """,
                (
                    _profile_key(str(row["name_key"]), str(row["team_code"])),
                    as_of_date,
                    str(row["player_key"] or ""),
                    str(row["player_id"] or ""),
                    str(row["name"] or ""),
                    str(row["name_key"] or ""),
                    str(row["team_code"] or ""),
                    str(row["position"] or ""),
                    profile.offensive_role.value,
                    profile.defensive_role.value,
                    CURRENT_SCHEMA_VERSION,
                    json.dumps(_traits_dict(profile.traits), separators=(",", ":"), sort_keys=True),
                    json.dumps(_condition_dict(profile.condition), separators=(",", ":"), sort_keys=True),
                    source_filter or "season_stat_profile",
                    generated_at,
                ),
            )
            count += 1
    return count


def load_nba_sim_profiles_for_slate(
    as_of_date: str,
    slate: DraftKingsSlate,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[tuple[str, str], PlayerSimProfile]:
    ensure_basketball_schema(db_path)
    keys = {(player.team, dfs_name_key(player.name)) for player in slate.players if player.team and dfs_name_key(player.name)}
    if not keys:
        return {}
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM basketball_player_profiles
            WHERE as_of_date = ?
            """,
            (as_of_date,),
        ).fetchall()
    results: dict[tuple[str, str], PlayerSimProfile] = {}
    for row in rows:
        key = (str(row["team_code"] or ""), str(row["name_key"] or ""))
        if key not in keys:
            continue
        results[key] = _profile_row_to_sim_profile(row)
    return results


def list_nba_player_profiles(
    as_of_date: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    team_code: str | None = None,
    search: str | None = None,
) -> list[dict[str, object]]:
    ensure_basketball_schema(db_path)
    query = """
        SELECT
            s.as_of_date,
            s.player_key,
            s.player_id,
            s.name,
            s.name_key,
            s.team_code,
            s.team_id,
            s.position,
            s.status,
            s.games_sample,
            s.minutes,
            s.points,
            s.rebounds,
            s.assists,
            s.recent_fpts_avg,
            s.recent_fpts_weighted,
            s.recent_form_delta,
            s.injuries_json,
            s.extra_stats_json,
            p.offensive_role,
            p.defensive_role,
            p.profile_version,
            p.traits_json,
            p.condition_json
        FROM basketball_player_stats s
        LEFT JOIN basketball_player_profiles p
          ON p.as_of_date = s.as_of_date
         AND p.name_key = s.name_key
         AND p.team_code = s.team_code
        WHERE s.as_of_date = ?
    """
    params: list[object] = [as_of_date]
    if team_code:
        query += " AND s.team_code = ?"
        params.append(team_code.upper())
    if search:
        query += " AND s.name LIKE ?"
        params.append(f"%{search.strip()}%")
    query += " ORDER BY s.team_code ASC, s.minutes DESC, s.name ASC"

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, tuple(params)).fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        extra = json.loads(row["extra_stats_json"] or "{}")
        traits = _normalize_traits_payload(json.loads(row["traits_json"] or "{}"))
        condition = json.loads(row["condition_json"] or "{}")
        results.append(
            {
                "as_of_date": str(row["as_of_date"] or ""),
                "player_key": str(row["player_key"] or ""),
                "player_id": str(row["player_id"] or ""),
                "name": str(row["name"] or ""),
                "name_key": str(row["name_key"] or ""),
                "team_code": str(row["team_code"] or ""),
                "team_id": str(row["team_id"] or ""),
                "position": str(row["position"] or ""),
                "status": str(row["status"] or ""),
                "games_sample": float(row["games_sample"] or 0.0),
                "minutes": float(row["minutes"] or 0.0),
                "points": float(row["points"] or 0.0),
                "rebounds": float(row["rebounds"] or 0.0),
                "assists": float(row["assists"] or 0.0),
                "recent_fpts_avg": float(row["recent_fpts_avg"] or 0.0),
                "recent_fpts_weighted": float(row["recent_fpts_weighted"] or 0.0),
                "recent_form_delta": float(row["recent_form_delta"] or 0.0),
                "starts": float(extra.get("starts") or 0.0),
                "turnovers": float(extra.get("turnovers") or 0.0),
                "fouls": float(extra.get("fouls") or 0.0),
                "fga": float(extra.get("fga") or 0.0),
                "three_pa": float(extra.get("three_pa") or 0.0),
                "fta": float(extra.get("fta") or 0.0),
                "oreb": float(extra.get("oreb") or 0.0),
                "dreb": float(extra.get("dreb") or 0.0),
                "offensive_role": str(row["offensive_role"] or ""),
                "defensive_role": str(row["defensive_role"] or ""),
                "profile_version": int(row["profile_version"] or 0) if row["profile_version"] is not None else 0,
                "traits": traits,
                "condition": condition,
            }
        )
    return results


def list_nba_player_season_history(
    *,
    db_path: str = DEFAULT_DB_PATH,
    player_id: str | None = None,
    name_key: str | None = None,
    limit: int = 8,
) -> list[dict[str, object]]:
    ensure_basketball_schema(db_path)
    if not player_id and not name_key:
        return []
    query = """
        SELECT
            p.season,
            p.player_id,
            p.name,
            p.name_key,
            p.team_code,
            p.position,
            p.offensive_role,
            p.defensive_role,
            p.traits_json,
            s.games_sample,
            s.minutes,
            s.points,
            s.rebounds,
            s.assists,
            s.extra_stats_json,
            a.metrics_json
        FROM basketball_player_season_profiles p
        JOIN basketball_player_season_stats s
          ON s.season = p.season
         AND s.name_key = p.name_key
         AND s.team_code = p.team_code
        LEFT JOIN basketball_player_season_advanced_stats a
          ON a.season = p.season
         AND a.name_key = p.name_key
         AND a.team_code = p.team_code
        WHERE ((? <> '' AND p.player_id = ?) OR (? <> '' AND p.name_key = ?))
        ORDER BY p.season DESC
        LIMIT ?
    """
    params = (
        str(player_id or ""),
        str(player_id or ""),
        str(name_key or ""),
        str(name_key or ""),
        int(limit),
    )
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    history: list[dict[str, object]] = []
    for row in rows:
        traits = _normalize_traits_payload(json.loads(row["traits_json"] or "{}"))
        extra = json.loads(row["extra_stats_json"] or "{}")
        metrics = json.loads(row["metrics_json"] or "{}") if row["metrics_json"] else {}
        minutes = float(row["minutes"] or 0.0)
        games_sample = float(row["games_sample"] or 0.0)
        history.append(
            {
                "season": str(row["season"] or ""),
                "team_code": str(row["team_code"] or ""),
                "position": str(row["position"] or ""),
                "offensive_role": str(row["offensive_role"] or ""),
                "defensive_role": str(row["defensive_role"] or ""),
                "games_sample": games_sample,
                "minutes_per_game": minutes / max(games_sample, 1.0),
                "points_per_game": float(row["points"] or 0.0) / max(games_sample, 1.0),
                "rebounds_per_game": float(row["rebounds"] or 0.0) / max(games_sample, 1.0),
                "assists_per_game": float(row["assists"] or 0.0) / max(games_sample, 1.0),
                "fga_per_game": float(extra.get("fga") or 0.0) / max(games_sample, 1.0),
                "three_pa_per_game": float(extra.get("three_pa") or 0.0) / max(games_sample, 1.0),
                "fta_per_game": float(extra.get("fta") or 0.0) / max(games_sample, 1.0),
                "usg_pct": float(metrics.get("usg_pct") or 0.0),
                "ts_pct": float(metrics.get("ts_pct") or 0.0),
                "ast_pct": float(metrics.get("ast_pct") or 0.0),
                "pace": float(metrics.get("pace") or 0.0),
                "traits": traits,
            }
        )
    return history


def fetch_espn_nba_teams() -> list[dict[str, str]]:
    response = requests.get(ESPN_NBA_TEAMS_URL, timeout=20)
    response.raise_for_status()
    payload = response.json()
    teams = payload.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    results: list[dict[str, str]] = []
    for item in teams:
        team = item.get("team", {})
        results.append(
            {
                "team_id": str(team.get("id") or ""),
                "team_code": str(team.get("abbreviation") or ""),
                "team_name": str(team.get("displayName") or team.get("name") or ""),
            }
        )
    return results


def _upsert_slate_row(conn: sqlite3.Connection, *, date_str: str, slate: DraftKingsSlate, imported_at: str) -> int:
    conn.execute(
        """
        INSERT INTO basketball_slates (
            slate_date, sport, site, source_path, salary_cap, roster_slots_json, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slate_date, sport, site, source_path) DO UPDATE SET
            salary_cap = excluded.salary_cap,
            roster_slots_json = excluded.roster_slots_json,
            imported_at = excluded.imported_at
        """,
        (
            date_str,
            slate.sport,
            slate.site,
            slate.source_path,
            int(slate.salary_cap),
            json.dumps(slate.roster_slots),
            imported_at,
        ),
    )
    row = conn.execute(
        """
        SELECT id
        FROM basketball_slates
        WHERE slate_date = ? AND sport = ? AND site = ? AND source_path = ?
        """,
        (date_str, slate.sport, slate.site, slate.source_path),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to persist DraftKings slate row")
    return int(row["id"])


def _player_key(player: DraftKingsPlayer) -> str:
    return str(player.player_id or dfs_name_key(player.name))


def _row_to_dk_player(row: sqlite3.Row, *, sport: str) -> DraftKingsPlayer:
    return DraftKingsPlayer(
        player_id=str(row["dk_player_id"] or row["player_key"]),
        name=str(row["name"]),
        sport=sport,
        team=str(row["team_code"] or ""),
        opponent=str(row["opponent_code"] or ""),
        game=str(row["game_code"] or ""),
        start_time=str(row["start_time"]) if row["start_time"] else None,
        salary=int(row["salary"] or 0),
        positions=tuple(json.loads(row["positions_json"] or "[]")),
        roster_positions=tuple(json.loads(row["roster_positions_json"] or "[]")),
        avg_points_per_game=float(row["avg_points_per_game"] or 0.0),
        raw_position=str(row["raw_position"] or ""),
        raw_game_info=str(row["raw_game_info"] or ""),
    )


def _stats_row_to_profile(row: sqlite3.Row, framework_scores: dict[str, float] | None = None) -> PlayerSimProfile:
    framework_scores = framework_scores or {}
    position = _normalize_stat_position(str(row["position"] or ""))
    minutes = float(row["minutes"] or 0.0)
    points = float(row["points"] or 0.0)
    rebounds = float(row["rebounds"] or 0.0)
    assists = float(row["assists"] or 0.0)
    games_sample = float(row["games_sample"] or 0.0)
    recent_fpts_avg = float(row["recent_fpts_avg"] or 0.0)
    recent_fpts_weighted = float(row["recent_fpts_weighted"] or 0.0)
    recent_form_delta = float(row["recent_form_delta"] or 0.0)
    extra_stats = json.loads(row["extra_stats_json"] or "{}")
    starts = float(extra_stats.get("starts") or 0.0)
    turnovers = float(extra_stats.get("turnovers") or 0.0)
    fouls = float(extra_stats.get("fouls") or 0.0)
    fga = float(extra_stats.get("fga") or 0.0)
    three_pa = float(extra_stats.get("three_pa") or 0.0)
    fta = float(extra_stats.get("fta") or 0.0)
    oreb = float(extra_stats.get("oreb") or 0.0)
    dreb = float(extra_stats.get("dreb") or 0.0)
    minutes_per_game = minutes / max(games_sample, 1.0)
    rebounds_per_game = rebounds / max(games_sample, 1.0)
    assists_per_game = assists / max(games_sample, 1.0)

    ppm = points / max(minutes, 1.0)
    rpm = rebounds / max(minutes, 1.0)
    apm = assists / max(minutes, 1.0)
    topm = turnovers / max(minutes, 1.0)
    fgapm = fga / max(minutes, 1.0)
    three_pm = three_pa / max(minutes, 1.0)
    ftapm = fta / max(minutes, 1.0)
    oreb_pm = oreb / max(minutes, 1.0)
    dreb_pm = dreb / max(minutes, 1.0)
    foul_pm = fouls / max(minutes, 1.0)
    confidence = _bounded(min(games_sample / 24.0, 1.0), 0.25, 1.0)
    start_rate = _bounded(starts / max(games_sample, 1.0), 0.0, 1.0)
    form_ratio = recent_fpts_weighted / max(recent_fpts_avg, 1.0) if recent_fpts_avg > 0.0 else 1.0
    form_confidence = _bounded(0.5 + (recent_form_delta / max(recent_fpts_avg, 1.0)) * 0.25, 0.2, 0.9)
    shot_volume_signal = _bounded(fgapm / 0.55, 0.0, 1.0)
    three_volume_signal = _bounded(three_pm / 0.24, 0.0, 1.0)
    free_throw_pressure_signal = _bounded(ftapm / 0.28, 0.0, 1.0)
    creator_signal = _bounded(apm / 0.18, 0.0, 1.0)
    creation_load_signal = _bounded(assists_per_game / 9.5, 0.0, 1.0)
    turnover_burden_signal = _bounded(topm / 0.14, 0.0, 1.0)

    offensive_role = _infer_offensive_role_from_stats(position, points, assists)
    defensive_role = _infer_defensive_role_from_stats(position)
    movement_shooter_bonus = 0.12 if offensive_role == OffensiveRole.MOVEMENT_SHOOTER else 0.0
    creator_shooter_bonus = 0.1 if offensive_role in {OffensiveRole.PRIMARY_CREATOR, OffensiveRole.SECONDARY_CREATOR} else 0.0
    guard_shooting_bonus = 0.08 if position in {"PG", "SG", "SF", "G"} else 0.02
    ball_security_rating = framework_scores.get(
        "ball_handle",
        _to_rating((confidence * 0.45) + (min(1.0, apm * 1.7) * 0.22) + (max(0.0, 0.22 - (topm * 0.35))) + 0.12),
    )
    pullup_rating = framework_scores.get(
        "pullup_shooting",
        _to_rating(
            0.11
            + (ppm * 0.2)
            + (shot_volume_signal * 0.16)
            + (three_volume_signal * 0.16)
            + (free_throw_pressure_signal * 0.08)
            + (creator_signal * 0.08)
            + (creator_shooter_bonus * 0.8)
            + (guard_shooting_bonus * 0.85)
        ),
    )
    catch_rating = framework_scores.get(
        "catch_shoot",
        _to_rating(
            0.12
            + (three_volume_signal * 0.46)
            + (ppm * 0.12)
            + (shot_volume_signal * 0.08)
            + movement_shooter_bonus
            + (0.06 if position in {"SG", "SF", "PF", "F"} else 0.0)
        ),
    )
    finishing_rating = framework_scores.get(
        "finishing",
        _to_rating(
            0.08
            + (ppm * 0.18)
            + (shot_volume_signal * 0.10)
            + (free_throw_pressure_signal * 0.30)
            + (_bounded(oreb_pm / 0.10, 0.0, 1.0) * 0.14)
            + (0.06 if position in {"PF", "C", "F"} else 0.0)
            + (0.05 if offensive_role in {OffensiveRole.PRIMARY_CREATOR, OffensiveRole.SECONDARY_CREATOR, OffensiveRole.SLASHER} else 0.0)
        ),
    )
    pass_vision_rating = framework_scores.get(
        "pass_vision",
        _to_rating(
            0.08
            + (creation_load_signal * 0.52)
            + (creator_signal * 0.18)
            + (confidence * 0.10)
            + (0.08 if offensive_role in {OffensiveRole.PRIMARY_CREATOR, OffensiveRole.SECONDARY_CREATOR, OffensiveRole.POST_HUB} else 0.0)
            + (0.04 if position == "C" and assists_per_game >= 4.5 else 0.0)
        ),
    )
    pass_accuracy_rating = framework_scores.get(
        "pass_accuracy",
        _to_rating(
            0.12
            + (creation_load_signal * 0.34)
            + (creator_signal * 0.16)
            + (confidence * 0.12)
            + (_bounded(1.0 - turnover_burden_signal, 0.0, 1.0) * 0.14)
            + (0.05 if offensive_role in {OffensiveRole.PRIMARY_CREATOR, OffensiveRole.SECONDARY_CREATOR, OffensiveRole.POST_HUB} else 0.0)
        ),
    )
    screen_setting_rating = framework_scores.get("screen_setting", _position_screen_value(position, rebounds_per_game, start_rate))
    stamina_rating = framework_scores.get(
        "stamina",
        _to_rating(
            0.04
            + (_bounded(minutes_per_game / 38.0, 0.0, 1.0) * 0.62)
            + (start_rate * 0.18)
            + (confidence * 0.08)
            + (_bounded(games_sample / 82.0, 0.0, 1.0) * 0.08)
        ),
    )
    traits = PlayerTraitProfile(
        ball_security=ball_security_rating,
        separation=_to_rating((ppm * 0.46) + (min(1.0, fgapm * 0.65) * 0.2) + (min(1.0, apm * 1.15) * 0.14) + (confidence * 0.2)),
        burst=_to_rating((ppm * 0.56) + (min(1.0, ftapm * 0.85) * 0.16) + (0.12 if position in {"PG", "SG", "G"} else 0.02) + (confidence * 0.16)),
        speed=_to_rating(
            0.10
            + (ppm * 0.24)
            + (min(1.0, minutes_per_game / 36.0) * 0.10)
            + (min(1.0, fgapm * 0.45) * 0.10)
            + (min(1.0, ftapm * 0.55) * 0.12)
            + (0.16 if position in {"PG", "SG", "SF", "G"} else 0.06)
            + (confidence * 0.14)
        ),
        pullup_shooting=pullup_rating,
        catch_shoot=catch_rating,
        finishing=finishing_rating,
        pass_vision=pass_vision_rating,
        pass_accuracy=pass_accuracy_rating,
        decision_making=_to_rating((confidence * 0.42) + (min(1.0, apm * 1.0) * 0.24) + (max(0.0, 0.26 - (topm * 0.3))) + 0.1),
        screen_setting=screen_setting_rating,
        rebounding=_to_rating(((oreb_pm * 2.2) + (dreb_pm * 2.0)) / 2.0),
        free_throw_rating=_to_rating(0.72),
        ft_pct_raw=0.72,
        foul_drawing=_to_rating((ppm * 0.32) + (min(1.0, ftapm * 1.2) * 0.48) + (0.08 if position in {"PG", "SG", "SF", "G"} else 0.04)),
        containment=_position_perimeter_defense_value(position) + (1.0 if start_rate > 0.6 and position in {"SG", "SF", "G"} else 0.0),
        closeout=_position_closeout_value(position) + (0.8 if start_rate > 0.6 and position in {"SG", "SF", "G"} else 0.0),
        screen_nav=_position_screen_navigation_value(position) + (0.8 if start_rate > 0.6 and position in {"PG", "SG", "SF", "G"} else 0.0),
        interior_def=_position_interior_defense_value(position, rebounds_per_game, start_rate),
        rim_protect=_position_rim_protection_value(position, rebounds_per_game, start_rate),
        steal_pressure=_to_rating((confidence * 0.28) + (max(0.0, 0.22 - (foul_pm * 0.18))) + (0.42 if position in {"PG", "SG", "SF", "G"} else 0.18)),
        foul_discipline=_to_rating((confidence * 0.62) + max(0.0, 0.28 - (foul_pm * 0.22))),
        help_rotation=_to_rating((_position_interior_defense_value(position, rebounds_per_game, start_rate) / 20.0 * 0.52) + (_position_closeout_value(position) / 20.0 * 0.28) + (start_rate * 0.2)),
        stamina=stamina_rating,
        role_consistency=_to_rating((confidence * 0.7) + (start_rate * 0.3)),
        clutch=10.0,
        size=_position_size_value(position),
        reach=_position_reach_value(position),
    )
    condition = PlayerCondition(
        energy=_bounded(confidence, 0.25, 1.0),
        fatigue=_bounded(1.0 - (minutes / 34.0), 0.0, 0.75),
        foul_count=0,
        confidence=form_confidence if form_ratio else 0.5,
        minutes_played=0.0,
        available=str(row["status"] or "active").lower() != "out",
    )
    return PlayerSimProfile(
        player_id=str(row["player_id"] or row["name_key"]),
        name=str(row["name"] or ""),
        team_code=str(row["team_code"] or ""),
        positions=_position_tuple(position),
        offensive_role=offensive_role,
        defensive_role=defensive_role,
        traits=traits,
        condition=condition,
    )


def _stats_sqlite_row_to_dict(
    row: sqlite3.Row,
    supplemental_map: dict[tuple[str, str], dict[str, dict[str, dict[str, float]]]] | None = None,
) -> dict[str, object]:
    extra_stats = json.loads(row["extra_stats_json"] or "{}")
    row_key = (str(row["team_code"] or ""), str(row["name_key"] or ""))
    return {
        "team_code": str(row["team_code"] or ""),
        "name_key": str(row["name_key"] or ""),
        "position": _normalize_stat_position(str(row["position"] or "")),
        "games_sample": float(row["games_sample"] or 0.0),
        "minutes": float(row["minutes"] or 0.0),
        "points": float(row["points"] or 0.0),
        "rebounds": float(row["rebounds"] or 0.0),
        "assists": float(row["assists"] or 0.0),
        "recent_fpts_avg": float(row["recent_fpts_avg"] or 0.0),
        "recent_fpts_weighted": float(row["recent_fpts_weighted"] or 0.0),
        "recent_form_delta": float(row["recent_form_delta"] or 0.0),
        "extra_stats": {
            "starts": float(extra_stats.get("starts") or 0.0),
            "turnovers": float(extra_stats.get("turnovers") or 0.0),
            "fga": float(extra_stats.get("fga") or 0.0),
            "three_pa": float(extra_stats.get("three_pa") or 0.0),
            "fta": float(extra_stats.get("fta") or 0.0),
            "oreb": float(extra_stats.get("oreb") or 0.0),
            "dreb": float(extra_stats.get("dreb") or 0.0),
        },
        "supplemental_sources": (supplemental_map or {}).get(row_key, {}),
    }


def _season_stats_sqlite_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    extra_stats = json.loads(row["extra_stats_json"] or "{}")
    return {
        "team_code": str(row["team_code"] or ""),
        "name_key": str(row["name_key"] or ""),
        "position": _normalize_stat_position(str(row["position"] or "")),
        "games_sample": float(row["games_sample"] or 0.0),
        "minutes": float(row["minutes"] or 0.0),
        "points": float(row["points"] or 0.0),
        "rebounds": float(row["rebounds"] or 0.0),
        "assists": float(row["assists"] or 0.0),
        "recent_fpts_avg": float(row["recent_fpts_avg"] or 0.0),
        "recent_fpts_weighted": float(row["recent_fpts_weighted"] or 0.0),
        "recent_form_delta": float(row["recent_form_delta"] or 0.0),
        "extra_stats": {
            "starts": float(extra_stats.get("starts") or 0.0),
            "turnovers": float(extra_stats.get("turnovers") or 0.0),
            "fga": float(extra_stats.get("fga") or 0.0),
            "three_pa": float(extra_stats.get("three_pa") or 0.0),
            "fta": float(extra_stats.get("fta") or 0.0),
            "oreb": float(extra_stats.get("oreb") or 0.0),
            "dreb": float(extra_stats.get("dreb") or 0.0),
        },
    }


def _group_attribute_source_features(
    rows: list[sqlite3.Row],
) -> dict[tuple[str, str], dict[str, dict[str, dict[str, float]]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, float]]]] = {}
    for row in rows:
        row_key = (str(row["team_code"] or ""), str(row["name_key"] or ""))
        attribute_name = str(row["attribute_name"] or "")
        source = str(row["source"] or "")
        try:
            features = json.loads(row["features_json"] or "{}")
        except json.JSONDecodeError:
            features = {}
        if not isinstance(features, dict):
            continue
        feature_values = {
            str(key): float(value)
            for key, value in features.items()
            if _is_number_like(value)
        }
        grouped.setdefault(row_key, {}).setdefault(attribute_name, {})[source] = feature_values
    return grouped


def _is_number_like(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _profile_row_to_sim_profile(row: sqlite3.Row) -> PlayerSimProfile:
    traits = PlayerTraitProfile(**_normalize_traits_payload(json.loads(row["traits_json"] or "{}")))
    condition = PlayerCondition(**json.loads(row["condition_json"] or "{}"))
    return PlayerSimProfile(
        player_id=str(row["player_id"] or row["name_key"]),
        name=str(row["name"] or ""),
        team_code=str(row["team_code"] or ""),
        positions=_position_tuple(str(row["position"] or "")),
        offensive_role=OffensiveRole(str(row["offensive_role"])),
        defensive_role=DefensiveRole(str(row["defensive_role"])),
        traits=traits,
        condition=condition,
    )


def _season_stats_row_to_profile(row: sqlite3.Row, framework_scores: dict[str, float] | None = None) -> PlayerSimProfile:
    shim = {
        "player_id": row["player_id"],
        "name": row["name"],
        "team_code": row["team_code"],
        "position": row["position"],
        "status": "active",
        "games_sample": row["games_sample"],
        "minutes": row["minutes"],
        "points": row["points"],
        "rebounds": row["rebounds"],
        "assists": row["assists"],
        "recent_fpts_avg": row["recent_fpts_avg"],
        "recent_fpts_weighted": row["recent_fpts_weighted"],
        "recent_form_delta": row["recent_form_delta"],
        "extra_stats_json": row["extra_stats_json"],
    }
    return _stats_row_to_profile(shim, framework_scores)


def _normalize_traits_payload(raw_traits: dict[str, object]) -> dict[str, object]:
    traits = dict(raw_traits or {})
    if "rebounding" not in traits:
        oreb = float(traits.get("oreb") or 0.0) if _is_number_like(traits.get("oreb")) else 0.0
        dreb = float(traits.get("dreb") or 0.0) if _is_number_like(traits.get("dreb")) else 0.0
        if oreb or dreb:
            traits["rebounding"] = (oreb + dreb) / 2.0
    traits.pop("oreb", None)
    traits.pop("dreb", None)
    if "speed" not in traits:
        burst = float(traits.get("burst") or 10.0) if _is_number_like(traits.get("burst")) else 10.0
        separation = float(traits.get("separation") or 10.0) if _is_number_like(traits.get("separation")) else 10.0
        traits["speed"] = round(((burst * 0.55) + (separation * 0.45)), 2)
    return traits


def _traits_dict(traits: PlayerTraitProfile) -> dict[str, object]:
    return {
        field: getattr(traits, field)
        for field in traits.__dataclass_fields__
    }


def _condition_dict(condition: PlayerCondition) -> dict[str, object]:
    return {
        field: getattr(condition, field)
        for field in condition.__dataclass_fields__
    }


def _load_prior_profile_lookup(
    conn: sqlite3.Connection,
    season: str,
    *,
    lookback: int = 3,
) -> dict[str, dict[str, object]]:
    prior_seasons = _previous_seasons(season, lookback=lookback)
    if not prior_seasons:
        return {}
    placeholders = ",".join("?" for _ in prior_seasons)
    rows = conn.execute(
        f"""
        SELECT season, player_id, name_key, traits_json
        FROM basketball_player_season_profiles
        WHERE season IN ({placeholders})
        """,
        tuple(prior_seasons),
    ).fetchall()
    grouped: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for row in rows:
        key = _profile_history_key(str(row["player_id"] or ""), str(row["name_key"] or ""))
        try:
            traits_json = _normalize_traits_payload(json.loads(row["traits_json"] or "{}"))
        except json.JSONDecodeError:
            traits_json = {}
        if not isinstance(traits_json, dict):
            continue
        grouped.setdefault(key, []).append(
            (
                str(row["season"] or ""),
                {
                    str(name): float(value)
                    for name, value in traits_json.items()
                    if _is_number_like(value)
                },
            )
        )

    lookup: dict[str, dict[str, object]] = {}
    recency_weights = (1.0, 0.65, 0.4)
    for key, entries in grouped.items():
        ordered_entries = sorted(entries, key=lambda item: _season_start_year(item[0]), reverse=True)
        total_weight = 0.0
        blended: dict[str, float] = {}
        for idx, (_, trait_map) in enumerate(ordered_entries[:lookback]):
            weight = recency_weights[idx] if idx < len(recency_weights) else recency_weights[-1] * (0.75 ** (idx - len(recency_weights) + 1))
            total_weight += weight
            for trait_name, value in trait_map.items():
                if trait_name == "ft_pct_raw":
                    continue
                blended[trait_name] = blended.get(trait_name, 0.0) + (value * weight)
        if total_weight <= 0.0 or not blended:
            continue
        lookup[key] = {
            "count": min(len(ordered_entries), lookback),
            "traits": {
                trait_name: total / total_weight
                for trait_name, total in blended.items()
            },
        }
    return lookup


def _load_advanced_context_lookup(
    conn: sqlite3.Connection,
    season: str,
    *,
    lookback: int = 3,
) -> dict[str, dict[str, float]]:
    seasons = [season, *_previous_seasons(season, lookback=lookback)]
    placeholders = ",".join("?" for _ in seasons)
    rows = conn.execute(
        f"""
        SELECT season, player_id, name_key, metrics_json
        FROM basketball_player_season_advanced_stats
        WHERE season IN ({placeholders})
        """,
        tuple(seasons),
    ).fetchall()
    grouped: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for row in rows:
        key = _profile_history_key(str(row["player_id"] or ""), str(row["name_key"] or ""))
        try:
            metrics = json.loads(row["metrics_json"] or "{}")
        except json.JSONDecodeError:
            metrics = {}
        if not isinstance(metrics, dict):
            continue
        grouped.setdefault(key, []).append(
            (
                str(row["season"] or ""),
                {
                    str(metric): float(value)
                    for metric, value in metrics.items()
                    if _is_number_like(value)
                },
            )
        )
    lookup: dict[str, dict[str, float]] = {}
    for key, entries in grouped.items():
        ordered = sorted(entries, key=lambda item: _season_start_year(item[0]), reverse=True)
        current_metrics = ordered[0][1] if ordered else {}
        prior_entries = ordered[1 : 1 + lookback]
        prior_metrics: dict[str, float] = {}
        if prior_entries:
            for metric_name in {name for _, metrics in prior_entries for name in metrics.keys()}:
                values = [metrics.get(metric_name, 0.0) for _, metrics in prior_entries]
                prior_metrics[metric_name] = sum(values) / max(len(values), 1)
        lookup[key] = {
            "current_usg_pct": float(current_metrics.get("usg_pct") or 0.0),
            "current_ts_pct": float(current_metrics.get("ts_pct") or 0.0),
            "current_ast_pct": float(current_metrics.get("ast_pct") or 0.0),
            "current_pace": float(current_metrics.get("pace") or 0.0),
            "prior_usg_pct": float(prior_metrics.get("usg_pct") or 0.0),
            "prior_ts_pct": float(prior_metrics.get("ts_pct") or 0.0),
            "prior_ast_pct": float(prior_metrics.get("ast_pct") or 0.0),
            "prior_pace": float(prior_metrics.get("pace") or 0.0),
        }
    return lookup


def _trait_family(trait_name: str) -> str:
    mental_traits = {
        "pass_vision",
        "pass_accuracy",
        "decision_making",
        "help_rotation",
        "foul_discipline",
        "role_consistency",
        "clutch",
    }
    physical_traits = {
        "separation",
        "burst",
        "speed",
        "stamina",
        "size",
        "reach",
    }
    interior_traits = {"rebounding", "screen_setting", "foul_drawing"}
    defensive_traits = {
        "containment",
        "closeout",
        "screen_nav",
        "interior_def",
        "rim_protect",
        "steal_pressure",
    }
    if trait_name in mental_traits:
        return "mental"
    if trait_name in physical_traits:
        return "physical"
    if trait_name in defensive_traits:
        return "defensive"
    if trait_name in interior_traits:
        return "interior"
    return "technical"


def _prior_weight_for_trait(trait_name: str, age: float, prior_count: int) -> float:
    family = _trait_family(trait_name)
    base_weights = {
        "mental": 0.62,
        "technical": 0.45,
        "physical": 0.28,
        "defensive": 0.34,
        "interior": 0.40,
    }
    weight = base_weights.get(family, 0.4)
    availability_scale = min(1.0, (0.45 + (0.275 * max(0, prior_count - 1))))
    weight *= availability_scale
    if age <= 24.0:
        weight *= 0.78
    elif age >= 32.0 and family == "mental":
        weight = min(0.78, weight + 0.08)
    elif age >= 34.0 and family == "technical":
        weight = min(0.68, weight + 0.04)
    return _bounded(weight, 0.0, 0.95)


def _age_curve_delta(trait_name: str, age: float) -> float:
    if age <= 0.0:
        return 0.0
    family = _trait_family(trait_name)
    if family == "physical":
        delta = -max(0.0, age - 29.0) * 0.22
        if trait_name == "stamina":
            delta -= max(0.0, age - 31.0) * 0.10
        return delta
    if family == "defensive":
        return -max(0.0, age - 31.0) * 0.10
    if family == "interior":
        return -max(0.0, age - 32.0) * 0.08
    if family == "technical":
        if trait_name in {"pullup_shooting", "catch_shoot", "finishing"}:
            return -max(0.0, age - 34.0) * 0.05
        return 0.0
    if family == "mental":
        return -max(0.0, age - 36.0) * 0.03
    return 0.0


def _apply_progression_blend(
    profile: PlayerSimProfile,
    *,
    age: float,
    prior_context: dict[str, object] | None,
    advanced_context: dict[str, float] | None = None,
) -> PlayerSimProfile:
    if not prior_context:
        return profile
    prior_traits = prior_context.get("traits")
    if not isinstance(prior_traits, dict):
        return profile
    prior_count = int(prior_context.get("count") or 0)
    traits = _traits_dict(profile.traits)
    blended_traits: dict[str, object] = dict(traits)
    for trait_name, current_value in traits.items():
        if trait_name == "ft_pct_raw":
            continue
        prior_value = prior_traits.get(trait_name)
        if prior_value is None or not _is_number_like(prior_value):
            continue
        weight = _prior_weight_for_trait(trait_name, age, prior_count)
        if advanced_context:
            ast_delta = _bounded((advanced_context.get("prior_ast_pct", 0.0) - advanced_context.get("current_ast_pct", 0.0)) / 20.0, -0.15, 0.15)
            ts_delta = _bounded((advanced_context.get("prior_ts_pct", 0.0) - advanced_context.get("current_ts_pct", 0.0)) / 0.12, -0.12, 0.12)
            usg_delta = _bounded((advanced_context.get("prior_usg_pct", 0.0) - advanced_context.get("current_usg_pct", 0.0)) / 18.0, -0.12, 0.12)
            if trait_name in {"pass_vision", "pass_accuracy", "decision_making"}:
                weight = _bounded(weight + max(0.0, ast_delta) * 0.18, 0.0, 0.9)
            if trait_name in {"pullup_shooting", "catch_shoot", "finishing"}:
                weight = _bounded(weight + max(0.0, ts_delta + (usg_delta * 0.5)) * 0.12, 0.0, 0.85)
            if trait_name in {"speed", "burst", "stamina"} and age >= 31.0:
                pace_delta = _bounded((advanced_context.get("current_pace", 0.0) - advanced_context.get("prior_pace", 0.0)) / 20.0, -0.1, 0.1)
                weight = _bounded(weight - max(0.0, pace_delta) * 0.08, 0.0, 0.9)
        age_delta = _age_curve_delta(trait_name, age)
        blended_value = ((float(current_value) * (1.0 - weight)) + (float(prior_value) * weight)) + age_delta
        blended_traits[trait_name] = round(_bounded(blended_value, 1.0, 20.0), 2)
    return PlayerSimProfile(
        player_id=profile.player_id,
        name=profile.name,
        team_code=profile.team_code,
        positions=profile.positions,
        offensive_role=profile.offensive_role,
        defensive_role=profile.defensive_role,
        traits=PlayerTraitProfile(**blended_traits),
        condition=profile.condition,
    )


def _profile_key(name_key: str, team_code: str) -> str:
    return f"{team_code}:{name_key}"


def _profile_history_key(player_id: str, name_key: str) -> str:
    player_id = str(player_id or "").strip()
    return f"id:{player_id}" if player_id else f"name:{name_key}"


def _season_start_year(season: str) -> int:
    season_text = str(season or "").strip()
    if not season_text:
        return 0
    return int(season_text.split("-", 1)[0])


def _season_label_from_start_year(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _previous_seasons(season: str, *, lookback: int = 3) -> list[str]:
    start_year = _season_start_year(season)
    if start_year <= 0:
        return []
    return [
        _season_label_from_start_year(start_year - offset)
        for offset in range(1, lookback + 1)
        if (start_year - offset) > 0
    ]


def _season_for_date(as_of_date: str) -> str:
    date_text = str(as_of_date or "").strip()
    year, month, _ = [int(part) for part in date_text.split("-", 2)]
    start_year = year if month >= 9 else year - 1
    return _season_label_from_start_year(start_year)


def _safe_float_sql(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _position_tuple(position: str) -> tuple[str, ...]:
    pos = _normalize_stat_position(position)
    if pos in {"PG", "SG", "SF", "PF", "C"}:
        return (pos,)
    if pos == "G":
        return ("PG", "SG")
    if pos == "F":
        return ("SF", "PF")
    return ("UTIL",)


def _normalize_stat_position(position: str) -> str:
    pos = str(position or "").upper().strip()
    if not pos:
        return ""
    if pos in {"PG", "SG", "SF", "PF", "C"}:
        return pos
    if pos == "G":
        return "PG"
    if pos == "F":
        return "PF"
    hybrid_map = {
        "PG-SG": "PG",
        "SG-PG": "SG",
        "SF-PF": "SF",
        "PF-SF": "PF",
        "G-F": "SF",
        "F-G": "SF",
        "F-C": "PF",
        "C-F": "C",
    }
    if pos in hybrid_map:
        return hybrid_map[pos]
    for token in pos.replace("/", "-").split("-"):
        token = token.strip()
        if token in {"PG", "SG", "SF", "PF", "C"}:
            return token
        if token == "G":
            return "PG"
        if token == "F":
            return "PF"
    return pos


def _infer_offensive_role_from_stats(position: str, points: float, assists: float) -> OffensiveRole:
    if position in {"PG", "G"} and assists >= 6.0:
        return OffensiveRole.PRIMARY_CREATOR
    if position in {"PG", "SG", "G"} and assists >= 4.0:
        return OffensiveRole.SECONDARY_CREATOR
    if position == "C" and assists >= 4.0:
        return OffensiveRole.POST_HUB
    if position == "C":
        return OffensiveRole.ROLL_BIG
    if position in {"SG", "SF"} and points >= 18.0:
        return OffensiveRole.MOVEMENT_SHOOTER
    if position in {"SF", "PF", "F"}:
        return OffensiveRole.SLASHER
    return OffensiveRole.GLUE


def _infer_defensive_role_from_stats(position: str) -> DefensiveRole:
    if position == "C":
        return DefensiveRole.RIM_PROTECTOR
    if position in {"PF", "F"}:
        return DefensiveRole.HELPER
    if position in {"SF", "SG"}:
        return DefensiveRole.WING_STOPPER
    return DefensiveRole.POINT_OF_ATTACK


def _position_screen_value(position: str, rebounds_per_game: float, start_rate: float) -> float:
    base = 0.58 if position == "C" else 0.48 if position in {"PF", "F"} else 0.34 if position == "SF" else 0.22
    rebounding_signal = _bounded(rebounds_per_game / 10.5, 0.0, 1.0)
    starter_bonus = 0.08 if start_rate > 0.6 and position in {"PF", "C", "F"} else 0.0
    role_bonus = 0.04 if position in {"PF", "C", "F"} else 0.0
    return _to_rating(base + (rebounding_signal * 0.22) + starter_bonus + role_bonus)


def _position_perimeter_defense_value(position: str) -> float:
    base = 0.62 if position in {"PG", "SG", "SF", "G"} else 0.45
    return _to_rating(base)


def _position_closeout_value(position: str) -> float:
    return _bounded(_position_perimeter_defense_value(position) - 1.2, 1.0, 20.0)


def _position_screen_navigation_value(position: str) -> float:
    return _bounded(_position_perimeter_defense_value(position) - 2.0, 1.0, 20.0)


def _position_interior_defense_value(position: str, rebounds_per_game: float, start_rate: float) -> float:
    base = 0.58 if position == "C" else 0.42 if position in {"PF", "F"} else 0.30
    rebounding_signal = _bounded(rebounds_per_game / 11.0, 0.0, 1.0)
    starter_bonus = 0.08 if start_rate > 0.6 and position in {"PF", "C", "F"} else 0.0
    return _to_rating(base + (rebounding_signal * 0.20) + starter_bonus)


def _position_rim_protection_value(position: str, rebounds_per_game: float, start_rate: float) -> float:
    base = 0.54 if position == "C" else 0.30 if position in {"PF", "F"} else 0.16
    rebounding_signal = _bounded(rebounds_per_game / 11.5, 0.0, 1.0)
    anchor_bonus = 0.10 if start_rate > 0.7 and position == "C" else 0.04 if start_rate > 0.6 and position in {"PF", "F"} else 0.0
    return _to_rating(base + (rebounding_signal * 0.18) + anchor_bonus)


def _position_size_value(position: str) -> float:
    if position == "C":
        return 17.0
    if position in {"PF", "F"}:
        return 13.0
    if position == "SF":
        return 11.0
    return 8.0


def _position_reach_value(position: str) -> float:
    if position == "C":
        return 17.0
    if position in {"PF", "F", "SF"}:
        return 13.0
    return 10.0


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _to_rating(value: float) -> float:
    return _bounded(1.0 + (float(value) * 19.0), 1.0, 20.0)


def _migrations() -> list[tuple[int, str, str]]:
    return [
        (
            1,
            "basketball_slate_schema",
            """
            CREATE TABLE IF NOT EXISTS basketball_players (
                player_key TEXT PRIMARY KEY,
                dk_player_id TEXT,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                sport TEXT NOT NULL,
                team_code TEXT,
                default_positions_json TEXT NOT NULL,
                default_roster_positions_json TEXT NOT NULL,
                avg_points_per_game REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_players_name_key
            ON basketball_players (name_key);

            CREATE TABLE IF NOT EXISTS basketball_slates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slate_date TEXT NOT NULL,
                sport TEXT NOT NULL,
                site TEXT NOT NULL,
                source_path TEXT NOT NULL,
                salary_cap INTEGER NOT NULL,
                roster_slots_json TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                UNIQUE(slate_date, sport, site, source_path)
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_slates_lookup
            ON basketball_slates (slate_date, sport, imported_at);

            CREATE TABLE IF NOT EXISTS basketball_slate_players (
                slate_id INTEGER NOT NULL,
                player_key TEXT NOT NULL,
                dk_player_id TEXT,
                name TEXT NOT NULL,
                team_code TEXT NOT NULL,
                opponent_code TEXT,
                game_code TEXT,
                start_time TEXT,
                salary INTEGER NOT NULL,
                positions_json TEXT NOT NULL,
                roster_positions_json TEXT NOT NULL,
                avg_points_per_game REAL NOT NULL DEFAULT 0,
                raw_position TEXT,
                raw_game_info TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (slate_id, player_key),
                FOREIGN KEY (slate_id) REFERENCES basketball_slates(id) ON DELETE CASCADE,
                FOREIGN KEY (player_key) REFERENCES basketball_players(player_key) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_slate_players_game
            ON basketball_slate_players (slate_id, game_code, team_code);
            """,
        )
        ,
        (
            2,
            "basketball_profile_schema",
            """
            CREATE TABLE IF NOT EXISTS basketball_player_stats (
                as_of_date TEXT NOT NULL,
                player_key TEXT NOT NULL,
                player_id TEXT,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                team_code TEXT NOT NULL,
                team_id TEXT,
                position TEXT,
                status TEXT,
                games_sample REAL NOT NULL DEFAULT 0,
                minutes REAL NOT NULL DEFAULT 0,
                points REAL NOT NULL DEFAULT 0,
                rebounds REAL NOT NULL DEFAULT 0,
                assists REAL NOT NULL DEFAULT 0,
                recent_fpts_avg REAL NOT NULL DEFAULT 0,
                recent_fpts_weighted REAL NOT NULL DEFAULT 0,
                recent_form_delta REAL NOT NULL DEFAULT 0,
                injuries_json TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (as_of_date, name_key, team_code)
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_player_stats_lookup
            ON basketball_player_stats (as_of_date, team_code, name_key);

            CREATE TABLE IF NOT EXISTS basketball_player_profiles (
                profile_key TEXT PRIMARY KEY,
                as_of_date TEXT NOT NULL,
                player_key TEXT NOT NULL,
                player_id TEXT,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                team_code TEXT NOT NULL,
                position TEXT,
                offensive_role TEXT NOT NULL,
                defensive_role TEXT NOT NULL,
                profile_version INTEGER NOT NULL,
                traits_json TEXT NOT NULL,
                condition_json TEXT NOT NULL,
                source TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_player_profiles_lookup
            ON basketball_player_profiles (as_of_date, team_code, name_key);
            """,
        ),
        (
            3,
            "basketball_player_extra_stats",
            """
            ALTER TABLE basketball_player_stats
            ADD COLUMN extra_stats_json TEXT NOT NULL DEFAULT '{}';
            """,
        ),
        (
            4,
            "basketball_player_attribute_features",
            """
            CREATE TABLE IF NOT EXISTS basketball_player_attribute_features (
                as_of_date TEXT NOT NULL,
                name_key TEXT NOT NULL,
                team_code TEXT NOT NULL,
                attribute_name TEXT NOT NULL,
                source TEXT NOT NULL,
                features_json TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (as_of_date, name_key, team_code, attribute_name, source)
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_player_attribute_features_lookup
            ON basketball_player_attribute_features (as_of_date, attribute_name, source, team_code, name_key);
            """,
        ),
        (
            5,
            "basketball_player_season_history",
            """
            CREATE TABLE IF NOT EXISTS basketball_player_season_stats (
                season TEXT NOT NULL,
                player_key TEXT NOT NULL,
                player_id TEXT,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                team_code TEXT NOT NULL,
                team_id TEXT,
                age REAL NOT NULL DEFAULT 0,
                position TEXT,
                games_sample REAL NOT NULL DEFAULT 0,
                minutes REAL NOT NULL DEFAULT 0,
                points REAL NOT NULL DEFAULT 0,
                rebounds REAL NOT NULL DEFAULT 0,
                assists REAL NOT NULL DEFAULT 0,
                recent_fpts_avg REAL NOT NULL DEFAULT 0,
                recent_fpts_weighted REAL NOT NULL DEFAULT 0,
                recent_form_delta REAL NOT NULL DEFAULT 0,
                extra_stats_json TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (season, name_key, team_code)
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_player_season_stats_lookup
            ON basketball_player_season_stats (season, team_code, name_key);

            CREATE TABLE IF NOT EXISTS basketball_player_season_profiles (
                season TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                player_key TEXT NOT NULL,
                player_id TEXT,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                team_code TEXT NOT NULL,
                age REAL NOT NULL DEFAULT 0,
                position TEXT,
                offensive_role TEXT NOT NULL,
                defensive_role TEXT NOT NULL,
                profile_version INTEGER NOT NULL,
                traits_json TEXT NOT NULL,
                condition_json TEXT NOT NULL,
                source TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (season, name_key, team_code)
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_player_season_profiles_lookup
            ON basketball_player_season_profiles (season, team_code, name_key);
            """,
        ),
        (
            6,
            "basketball_player_season_advanced_history",
            """
            CREATE TABLE IF NOT EXISTS basketball_player_season_advanced_stats (
                season TEXT NOT NULL,
                player_key TEXT NOT NULL,
                player_id TEXT,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                team_code TEXT NOT NULL,
                team_id TEXT,
                metrics_json TEXT NOT NULL,
                source TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (season, name_key, team_code)
            );

            CREATE INDEX IF NOT EXISTS idx_basketball_player_season_advanced_lookup
            ON basketball_player_season_advanced_stats (season, team_code, name_key);
            """,
        ),
    ]
