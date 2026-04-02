from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from player_name_utils import dfs_name_key


DEFAULT_DFS_RESULTS_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "dfs_results.sqlite")


def ensure_dfs_results_schema(db_path: str = DEFAULT_DFS_RESULTS_DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dfs_builds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                slate_date TEXT NOT NULL,
                sport TEXT NOT NULL,
                request_mode TEXT NOT NULL,
                contest_type TEXT NOT NULL,
                salary_csv_path TEXT NOT NULL,
                input_label TEXT NOT NULL,
                preferred_salary_shape TEXT,
                max_players_per_game INTEGER,
                build_reasons_json TEXT NOT NULL,
                focus_players_json TEXT NOT NULL,
                fade_players_json TEXT NOT NULL,
                stack_targets_json TEXT NOT NULL,
                bring_back_targets_json TEXT NOT NULL,
                one_off_targets_json TEXT NOT NULL,
                avoid_chalk_json TEXT NOT NULL,
                game_boosts_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dfs_lineups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                build_id INTEGER NOT NULL,
                lineup_index INTEGER NOT NULL,
                family_label TEXT,
                family_core_json TEXT NOT NULL,
                salary_used INTEGER NOT NULL,
                salary_remaining INTEGER NOT NULL,
                median_fpts REAL NOT NULL,
                ceiling_fpts REAL NOT NULL,
                floor_fpts REAL NOT NULL,
                average_confidence REAL NOT NULL,
                availability_counts_json TEXT NOT NULL,
                focus_hits_json TEXT NOT NULL,
                fade_hits_json TEXT NOT NULL,
                primary_games_json TEXT NOT NULL,
                game_exposures_json TEXT NOT NULL,
                FOREIGN KEY(build_id) REFERENCES dfs_builds(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dfs_lineup_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lineup_id INTEGER NOT NULL,
                slot TEXT NOT NULL,
                player_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                player_name_key TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                game TEXT NOT NULL,
                salary INTEGER NOT NULL,
                positions_json TEXT NOT NULL,
                availability_status TEXT NOT NULL,
                availability_source TEXT NOT NULL,
                projected_median REAL NOT NULL,
                projected_ceiling REAL NOT NULL,
                is_focus INTEGER NOT NULL,
                is_fade INTEGER NOT NULL,
                FOREIGN KEY(lineup_id) REFERENCES dfs_lineups(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dfs_player_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                slate_date TEXT NOT NULL,
                player_name_key TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT,
                game TEXT,
                stats_json TEXT NOT NULL,
                dk_points REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(sport, slate_date, player_name_key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dfs_lineup_results (
                lineup_id INTEGER PRIMARY KEY,
                actual_points REAL NOT NULL,
                missing_players INTEGER NOT NULL,
                scored_at TEXT NOT NULL,
                scoring_source TEXT NOT NULL,
                FOREIGN KEY(lineup_id) REFERENCES dfs_lineups(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dfs_projected_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                build_id INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                player_name_key TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                game TEXT NOT NULL,
                salary INTEGER NOT NULL,
                positions_json TEXT NOT NULL,
                roster_positions_json TEXT NOT NULL,
                median_fpts REAL NOT NULL,
                ceiling_fpts REAL NOT NULL,
                floor_fpts REAL NOT NULL,
                volatility REAL NOT NULL,
                projection_confidence REAL NOT NULL,
                minutes REAL NOT NULL,
                points REAL NOT NULL,
                rebounds REAL NOT NULL,
                assists REAL NOT NULL,
                availability_status TEXT NOT NULL,
                availability_source TEXT NOT NULL,
                FOREIGN KEY(build_id) REFERENCES dfs_builds(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


def save_dfs_build(
    *,
    slate_date: str,
    result: Any,
    salary_csv_path: str,
    input_label: str = "",
    db_path: str = DEFAULT_DFS_RESULTS_DB_PATH,
) -> int:
    ensure_dfs_results_schema(db_path)
    payload = _as_mapping(result)
    created_at = _utcnow_iso()
    lineup_cards = list(payload.get("lineup_cards") or [])
    family_lookup = _family_lookup(payload.get("lineup_families") or [])
    projected_players = list(payload.get("projected_players") or [])

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO dfs_builds (
                created_at, slate_date, sport, request_mode, contest_type,
                salary_csv_path, input_label, preferred_salary_shape, max_players_per_game,
                build_reasons_json, focus_players_json, fade_players_json,
                stack_targets_json, bring_back_targets_json, one_off_targets_json,
                avoid_chalk_json, game_boosts_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                slate_date,
                str(payload.get("sport") or ""),
                str(payload.get("request_mode") or ""),
                str(payload.get("contest_type") or ""),
                salary_csv_path,
                input_label or os.path.basename(salary_csv_path),
                payload.get("preferred_salary_shape"),
                payload.get("max_players_per_game"),
                _json_dumps(payload.get("build_reasons") or []),
                _json_dumps(payload.get("focus_players") or []),
                _json_dumps(payload.get("fade_players") or []),
                _json_dumps(payload.get("stack_targets") or []),
                _json_dumps(payload.get("bring_back_targets") or []),
                _json_dumps(payload.get("one_off_targets") or []),
                _json_dumps(payload.get("avoid_chalk") or []),
                _json_dumps(payload.get("game_boosts") or {}),
            ),
        )
        build_id = int(cursor.lastrowid)

        for player in projected_players:
            player_map = _as_mapping(player)
            conn.execute(
                """
                INSERT INTO dfs_projected_players (
                    build_id, player_id, player_name, player_name_key, team, opponent, game,
                    salary, positions_json, roster_positions_json,
                    median_fpts, ceiling_fpts, floor_fpts, volatility, projection_confidence,
                    minutes, points, rebounds, assists,
                    availability_status, availability_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    build_id,
                    str(player_map.get("player_id") or player_map.get("name") or ""),
                    str(player_map.get("name") or ""),
                    str(player_map.get("player_name_key") or dfs_name_key(str(player_map.get("name") or ""))),
                    str(player_map.get("team") or ""),
                    str(player_map.get("opponent") or ""),
                    str(player_map.get("game") or ""),
                    int(player_map.get("salary") or 0),
                    _json_dumps(player_map.get("positions") or []),
                    _json_dumps(player_map.get("roster_positions") or []),
                    float(player_map.get("median_fpts") or 0.0),
                    float(player_map.get("ceiling_fpts") or 0.0),
                    float(player_map.get("floor_fpts") or 0.0),
                    float(player_map.get("volatility") or 0.0),
                    float(player_map.get("projection_confidence") or 0.0),
                    float(player_map.get("minutes") or 0.0),
                    float(player_map.get("points") or 0.0),
                    float(player_map.get("rebounds") or 0.0),
                    float(player_map.get("assists") or 0.0),
                    str(player_map.get("availability_status") or ""),
                    str(player_map.get("availability_source") or ""),
                ),
            )

        for lineup_index, card in enumerate(lineup_cards, start=1):
            card_map = _as_mapping(card)
            lineup_key = _lineup_key(card_map.get("slots") or [])
            family_meta = family_lookup.get(lineup_key, {})
            lineup_cursor = conn.execute(
                """
                INSERT INTO dfs_lineups (
                    build_id, lineup_index, family_label, family_core_json,
                    salary_used, salary_remaining, median_fpts, ceiling_fpts, floor_fpts,
                    average_confidence, availability_counts_json, focus_hits_json,
                    fade_hits_json, primary_games_json, game_exposures_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    build_id,
                    lineup_index,
                    family_meta.get("label"),
                    _json_dumps(family_meta.get("core_players") or []),
                    int(card_map.get("salary_used") or 0),
                    int(card_map.get("salary_remaining") or 0),
                    float(card_map.get("median_fpts") or 0.0),
                    float(card_map.get("ceiling_fpts") or 0.0),
                    float(card_map.get("floor_fpts") or 0.0),
                    float(card_map.get("average_confidence") or 0.0),
                    _json_dumps(card_map.get("availability_counts") or {}),
                    _json_dumps(card_map.get("focus_hits") or []),
                    _json_dumps(card_map.get("fade_hits") or []),
                    _json_dumps(card_map.get("primary_games") or []),
                    _json_dumps(card_map.get("game_exposures") or {}),
                ),
            )
            lineup_id = int(lineup_cursor.lastrowid)
            for slot in card_map.get("slots") or []:
                slot_map = _as_mapping(slot)
                conn.execute(
                    """
                    INSERT INTO dfs_lineup_players (
                        lineup_id, slot, player_id, player_name, player_name_key,
                        team, opponent, game, salary, positions_json,
                        availability_status, availability_source,
                        projected_median, projected_ceiling, is_focus, is_fade
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lineup_id,
                        str(slot_map.get("slot") or ""),
                        str(slot_map.get("player_id") or slot_map.get("name") or ""),
                        str(slot_map.get("name") or ""),
                        str(slot_map.get("player_name_key") or ""),
                        str(slot_map.get("team") or ""),
                        str(slot_map.get("opponent") or ""),
                        str(slot_map.get("game") or ""),
                        int(slot_map.get("salary") or 0),
                        _json_dumps(slot_map.get("positions") or []),
                        str(slot_map.get("availability_status") or ""),
                        str(slot_map.get("availability_source") or ""),
                        float(slot_map.get("median_fpts") or 0.0),
                        float(slot_map.get("ceiling_fpts") or 0.0),
                        1 if slot_map.get("is_focus") else 0,
                        1 if slot_map.get("is_fade") else 0,
                    ),
                )
        conn.commit()
    return build_id


def upsert_dfs_player_results(
    *,
    sport: str,
    slate_date: str,
    player_results: Iterable[Mapping[str, Any]],
    db_path: str = DEFAULT_DFS_RESULTS_DB_PATH,
    source: str = "manual",
) -> int:
    ensure_dfs_results_schema(db_path)
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        for item in player_results:
            name = str(item.get("player_name") or item.get("name") or "").strip()
            key = str(item.get("player_name_key") or "").strip()
            if not name or not key:
                continue
            conn.execute(
                """
                INSERT INTO dfs_player_results (
                    sport, slate_date, player_name_key, player_name, team, game,
                    stats_json, dk_points, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sport, slate_date, player_name_key) DO UPDATE SET
                    player_name=excluded.player_name,
                    team=excluded.team,
                    game=excluded.game,
                    stats_json=excluded.stats_json,
                    dk_points=excluded.dk_points,
                    source=excluded.source,
                    created_at=excluded.created_at
                """,
                (
                    sport,
                    slate_date,
                    key,
                    name,
                    item.get("team"),
                    item.get("game"),
                    _json_dumps(item.get("stats") or {}),
                    float(item.get("dk_points") or 0.0),
                    source,
                    _utcnow_iso(),
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def fetch_saved_build_summary(build_id: int, db_path: str = DEFAULT_DFS_RESULTS_DB_PATH) -> dict[str, Any] | None:
    ensure_dfs_results_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        build_row = conn.execute("SELECT * FROM dfs_builds WHERE id = ?", (build_id,)).fetchone()
        if build_row is None:
            return None
        lineup_rows = conn.execute(
            """
            SELECT l.*, r.actual_points, r.missing_players, r.scored_at, r.scoring_source
            FROM dfs_lineups l
            LEFT JOIN dfs_lineup_results r ON r.lineup_id = l.id
            WHERE l.build_id = ?
            ORDER BY l.lineup_index
            """,
            (build_id,),
        ).fetchall()
        projected_rows = conn.execute(
            """
            SELECT *
            FROM dfs_projected_players
            WHERE build_id = ?
            ORDER BY median_fpts DESC, salary DESC, player_name ASC
            """,
            (build_id,),
        ).fetchall()
    return {
        "build": dict(build_row),
        "lineups": [dict(row) for row in lineup_rows],
        "projected_players": [dict(row) for row in projected_rows],
    }


def _family_lookup(families: Iterable[Any]) -> dict[tuple[tuple[str, str], ...], dict[str, Any]]:
    lookup: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for family in families:
        family_map = _as_mapping(family)
        for card in family_map.get("lineup_cards") or []:
            card_map = _as_mapping(card)
            key = _lineup_key(card_map.get("slots") or [])
            if not key:
                continue
            lookup[key] = {
                "label": str(family_map.get("label") or ""),
                "core_players": list(family_map.get("core_players") or []),
            }
    return lookup


def _lineup_key(slots: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for slot in slots:
        slot_map = _as_mapping(slot)
        name_key = str(slot_map.get("player_name_key") or "").strip()
        slot_name = str(slot_map.get("slot") or "").strip()
        if not name_key:
            continue
        pairs.append((slot_name, name_key))
    return tuple(sorted(pairs))


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported DFS payload type: {type(value)!r}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
