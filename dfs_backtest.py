from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from dfs_results import DEFAULT_DFS_RESULTS_DB_PATH, ensure_dfs_results_schema, upsert_dfs_player_results
from player_name_utils import dfs_name_key


def draftkings_nba_actual_fpts(
    *,
    points: float = 0.0,
    rebounds: float = 0.0,
    assists: float = 0.0,
    steals: float = 0.0,
    blocks: float = 0.0,
    turnovers: float = 0.0,
) -> float:
    categories_10_plus = sum(
        1
        for value in (points, rebounds, assists, steals, blocks)
        if float(value or 0.0) >= 10.0
    )
    double_double_bonus = 1.5 if categories_10_plus >= 2 else 0.0
    triple_double_bonus = 3.0 if categories_10_plus >= 3 else 0.0
    return (
        float(points or 0.0)
        + (float(rebounds or 0.0) * 1.25)
        + (float(assists or 0.0) * 1.5)
        + (float(steals or 0.0) * 2.0)
        + (float(blocks or 0.0) * 2.0)
        - (float(turnovers or 0.0) * 0.5)
        + double_double_bonus
        + triple_double_bonus
    )


def draftkings_mlb_hitter_actual_fpts(
    *,
    singles: float = 0.0,
    doubles: float = 0.0,
    triples: float = 0.0,
    home_runs: float = 0.0,
    runs_batted_in: float = 0.0,
    runs: float = 0.0,
    walks: float = 0.0,
    hit_by_pitch: float = 0.0,
    stolen_bases: float = 0.0,
) -> float:
    return (
        (float(singles or 0.0) * 3.0)
        + (float(doubles or 0.0) * 5.0)
        + (float(triples or 0.0) * 8.0)
        + (float(home_runs or 0.0) * 10.0)
        + (float(runs_batted_in or 0.0) * 2.0)
        + (float(runs or 0.0) * 2.0)
        + ((float(walks or 0.0) + float(hit_by_pitch or 0.0)) * 2.0)
        + (float(stolen_bases or 0.0) * 5.0)
    )


def draftkings_mlb_pitcher_actual_fpts(
    *,
    innings_pitched: float = 0.0,
    strikeouts: float = 0.0,
    earned_runs: float = 0.0,
    hits_allowed: float = 0.0,
    walks_allowed: float = 0.0,
    hit_batters: float = 0.0,
    win: float = 0.0,
    complete_game: float = 0.0,
    shutout: float = 0.0,
    no_hitter: float = 0.0,
) -> float:
    return (
        (float(innings_pitched or 0.0) * 2.25)
        + (float(strikeouts or 0.0) * 2.0)
        - (float(earned_runs or 0.0) * 2.0)
        - ((float(hits_allowed or 0.0) + float(walks_allowed or 0.0) + float(hit_batters or 0.0)) * 0.6)
        + (float(win or 0.0) * 4.0)
        + (float(complete_game or 0.0) * 2.5)
        + (float(shutout or 0.0) * 2.5)
        + (float(no_hitter or 0.0) * 5.0)
    )


def normalize_dfs_player_result(
    *,
    sport: str,
    player_name: str,
    stats: Mapping[str, Any],
    team: str = "",
    game: str = "",
) -> dict[str, Any]:
    sport_key = str(sport or "").strip().lower()
    if sport_key == "nba":
        dk_points = draftkings_nba_actual_fpts(
            points=float(stats.get("points") or 0.0),
            rebounds=float(stats.get("rebounds") or 0.0),
            assists=float(stats.get("assists") or 0.0),
            steals=float(stats.get("steals") or 0.0),
            blocks=float(stats.get("blocks") or 0.0),
            turnovers=float(stats.get("turnovers") or 0.0),
        )
    elif sport_key == "mlb":
        is_pitcher = bool(stats.get("is_pitcher")) or str(stats.get("position") or "").upper() in {"P", "SP", "RP"}
        if is_pitcher:
            dk_points = draftkings_mlb_pitcher_actual_fpts(
                innings_pitched=float(stats.get("innings_pitched") or 0.0),
                strikeouts=float(stats.get("strikeouts") or 0.0),
                earned_runs=float(stats.get("earned_runs") or 0.0),
                hits_allowed=float(stats.get("hits_allowed") or 0.0),
                walks_allowed=float(stats.get("walks_allowed") or 0.0),
                hit_batters=float(stats.get("hit_batters") or 0.0),
                win=float(stats.get("win") or 0.0),
                complete_game=float(stats.get("complete_game") or 0.0),
                shutout=float(stats.get("shutout") or 0.0),
                no_hitter=float(stats.get("no_hitter") or 0.0),
            )
        else:
            dk_points = draftkings_mlb_hitter_actual_fpts(
                singles=float(stats.get("singles") or 0.0),
                doubles=float(stats.get("doubles") or 0.0),
                triples=float(stats.get("triples") or 0.0),
                home_runs=float(stats.get("home_runs") or 0.0),
                runs_batted_in=float(stats.get("runs_batted_in") or 0.0),
                runs=float(stats.get("runs") or 0.0),
                walks=float(stats.get("walks") or 0.0),
                hit_by_pitch=float(stats.get("hit_by_pitch") or 0.0),
                stolen_bases=float(stats.get("stolen_bases") or 0.0),
            )
    else:
        raise ValueError(f"Unsupported sport for DFS scoring: {sport}")
    return {
        "player_name": player_name,
        "player_name_key": dfs_name_key(player_name),
        "team": team,
        "game": game,
        "stats": dict(stats),
        "dk_points": dk_points,
    }


def upsert_normalized_player_results(
    *,
    sport: str,
    slate_date: str,
    raw_results: Iterable[Mapping[str, Any]],
    db_path: str = DEFAULT_DFS_RESULTS_DB_PATH,
    source: str = "manual",
) -> int:
    normalized = []
    for item in raw_results:
        player_name = str(item.get("player_name") or item.get("name") or "").strip()
        if not player_name:
            continue
        normalized.append(
            normalize_dfs_player_result(
                sport=sport,
                player_name=player_name,
                team=str(item.get("team") or ""),
                game=str(item.get("game") or ""),
                stats=item.get("stats") or {},
            )
        )
    return upsert_dfs_player_results(
        sport=sport,
        slate_date=slate_date,
        player_results=normalized,
        db_path=db_path,
        source=source,
    )


def score_saved_build(
    build_id: int,
    *,
    db_path: str = DEFAULT_DFS_RESULTS_DB_PATH,
    scoring_source: str = "local_box_score",
) -> list[dict[str, Any]]:
    ensure_dfs_results_schema(db_path)
    scored_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        build_row = conn.execute("SELECT * FROM dfs_builds WHERE id = ?", (build_id,)).fetchone()
        if build_row is None:
            raise ValueError(f"Unknown DFS build id: {build_id}")
        lineup_rows = conn.execute(
            "SELECT * FROM dfs_lineups WHERE build_id = ? ORDER BY lineup_index",
            (build_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for lineup_row in lineup_rows:
            player_rows = conn.execute(
                """
                SELECT p.*, r.dk_points
                FROM dfs_lineup_players p
                LEFT JOIN dfs_player_results r
                  ON r.sport = ?
                 AND r.slate_date = ?
                 AND r.player_name_key = p.player_name_key
                WHERE p.lineup_id = ?
                ORDER BY p.slot
                """,
                (build_row["sport"], build_row["slate_date"], lineup_row["id"]),
            ).fetchall()
            total = 0.0
            missing = 0
            for player_row in player_rows:
                dk_points = player_row["dk_points"]
                if dk_points is None:
                    missing += 1
                    continue
                total += float(dk_points)
            conn.execute(
                """
                INSERT INTO dfs_lineup_results (lineup_id, actual_points, missing_players, scored_at, scoring_source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(lineup_id) DO UPDATE SET
                    actual_points=excluded.actual_points,
                    missing_players=excluded.missing_players,
                    scored_at=excluded.scored_at,
                    scoring_source=excluded.scoring_source
                """,
                (lineup_row["id"], total, missing, scored_at, scoring_source),
            )
            results.append(
                {
                    "lineup_id": int(lineup_row["id"]),
                    "lineup_index": int(lineup_row["lineup_index"]),
                    "actual_points": total,
                    "missing_players": missing,
                }
            )
        conn.commit()
    return results
