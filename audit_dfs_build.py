from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from collections.abc import Iterable
from typing import Any

import requests

from dfs_backtest import draftkings_nba_actual_fpts, score_saved_build, upsert_normalized_player_results
from dfs_results import DEFAULT_DFS_RESULTS_DB_PATH
from player_name_utils import dfs_name_key


ESPN_NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_NBA_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def fetch_nba_actual_results_for_date(date_str: str) -> list[dict[str, Any]]:
    scoreboard = requests.get(ESPN_NBA_SCOREBOARD_URL, params={"dates": date_str.replace("-", "")}, timeout=20)
    scoreboard.raise_for_status()
    payload = scoreboard.json()
    raw_results: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            continue
        raw_results.extend(fetch_nba_actual_results_for_event(event_id))
    return raw_results


def fetch_nba_actual_results_for_event(event_id: str) -> list[dict[str, Any]]:
    summary = requests.get(ESPN_NBA_SUMMARY_URL, params={"event": event_id}, timeout=20)
    summary.raise_for_status()
    payload = summary.json()
    game = _espn_game_key(payload)
    rows: list[dict[str, Any]] = []
    for block in payload.get("boxscore", {}).get("players", []):
        team = str((block.get("team") or {}).get("abbreviation") or "").strip()
        for stat_group in block.get("statistics", []):
            labels = stat_group.get("labels", [])
            label_idx = {label: idx for idx, label in enumerate(labels)}
            for athlete in stat_group.get("athletes", []):
                name = str((athlete.get("athlete") or {}).get("displayName") or "").strip()
                stats = athlete.get("stats") or []
                if not name or not stats:
                    continue
                rows.append(
                    {
                        "name": name,
                        "team": team,
                        "game": game,
                        "stats": {
                            "points": _espn_stat(stats, label_idx, "PTS"),
                            "rebounds": _espn_stat(stats, label_idx, "REB"),
                            "assists": _espn_stat(stats, label_idx, "AST"),
                            "steals": _espn_stat(stats, label_idx, "STL"),
                            "blocks": _espn_stat(stats, label_idx, "BLK"),
                            "turnovers": _espn_stat(stats, label_idx, "TO"),
                        },
                    }
                )
    return rows


def audit_saved_nba_build(build_id: int, *, db_path: str = DEFAULT_DFS_RESULTS_DB_PATH) -> dict[str, Any]:
    build_row = _load_build_row(build_id, db_path=db_path)
    if build_row["sport"] != "nba":
        raise ValueError(f"Build {build_id} is not an NBA build")

    raw_results = fetch_nba_actual_results_for_date(build_row["slate_date"])
    upsert_normalized_player_results(
        sport="nba",
        slate_date=build_row["slate_date"],
        raw_results=raw_results,
        db_path=db_path,
        source="espn_summary",
    )
    scored_lineups = score_saved_build(build_id, db_path=db_path, scoring_source="espn_summary")

    projections = _load_saved_projected_players(build_id, db_path=db_path)
    projection_by_key = {row["player_name_key"]: row for row in projections}
    actual_by_key = {
        dfs_name_key(item["name"]): draftkings_nba_actual_fpts(**(item["stats"] or {}))
        for item in raw_results
    }

    lineup_rows = _load_lineup_player_rows(build_id, db_path=db_path)
    player_exposure = Counter(row["player_name_key"] for row in lineup_rows)
    lineup_count = max((row["lineup_index"] for row in lineup_rows), default=0)

    actual_lineups = {
        row["lineup_index"]: {
            "projected_median": row["projected_median_sum"],
            "actual_points": row["actual_points"],
            "players": row["players"],
        }
        for row in _load_lineup_audit_rows(build_id, db_path=db_path)
    }

    top_omissions: list[dict[str, Any]] = []
    for proj in projections:
        key = str(proj["player_name_key"])
        if player_exposure.get(key):
            continue
        actual = actual_by_key.get(key, 0.0)
        top_omissions.append(
            {
                "name": proj["player_name"],
                "team": proj["team"],
                "game": proj["game"],
                "salary": int(proj["salary"]),
                "projected_median": float(proj["median_fpts"]),
                "actual_points": actual,
                "actual_minus_projection": actual - float(proj["median_fpts"]),
                "value_x1000": (actual / int(proj["salary"]) * 1000.0) if int(proj["salary"]) else 0.0,
            }
        )
    top_omissions.sort(key=lambda item: (item["actual_minus_projection"], item["actual_points"]), reverse=True)

    survivor_rows: list[dict[str, Any]] = []
    for key, exposure in player_exposure.items():
        proj = projection_by_key.get(key)
        if proj is None:
            continue
        actual = actual_by_key.get(key, 0.0)
        survivor_rows.append(
            {
                "name": proj["player_name"],
                "salary": int(proj["salary"]),
                "projected_median": float(proj["median_fpts"]),
                "actual_points": actual,
                "actual_minus_projection": actual - float(proj["median_fpts"]),
                "exposure": exposure,
                "exposure_rate": (exposure / lineup_count) if lineup_count else 0.0,
            }
        )
    survivor_rows.sort(key=lambda item: (item["actual_minus_projection"], -item["exposure"]), reverse=False)

    return {
        "build": dict(build_row),
        "scored_lineups": scored_lineups,
        "actual_lineups": actual_lineups,
        "top_omissions": top_omissions,
        "survivors": survivor_rows,
    }


def print_nba_build_audit(report: dict[str, Any]) -> None:
    build = report["build"]
    print(f"Build #{build['id']} {build['sport']} {build['request_mode']} {build['contest_type']} {build['slate_date']}")
    print(f"Input: {build['salary_csv_path']}")
    print()
    print("Lineup results:")
    for lineup_index in sorted(report["actual_lineups"]):
        row = report["actual_lineups"][lineup_index]
        print(
            f"  L{lineup_index}: proj {row['projected_median']:.2f} -> actual {row['actual_points']:.2f} "
            f"players: {', '.join(row['players'])}"
        )
    print()
    print("Worst survivors:")
    for row in report["survivors"][:12]:
        print(
            f"  {row['name']:<22} exp {row['exposure']:<2} proj {row['projected_median']:.2f} "
            f"actual {row['actual_points']:.2f} diff {row['actual_minus_projection']:+.2f}"
        )
    print()
    print("Top omissions:")
    for row in report["top_omissions"][:15]:
        print(
            f"  {row['name']:<22} {row['game']:<7} proj {row['projected_median']:.2f} "
            f"actual {row['actual_points']:.2f} diff {row['actual_minus_projection']:+.2f} salary {row['salary']}"
        )


def _load_build_row(build_id: int, *, db_path: str) -> sqlite3.Row:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM dfs_builds WHERE id = ?", (build_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown DFS build id: {build_id}")
    return row


def _load_lineup_player_rows(build_id: int, *, db_path: str) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return list(
            conn.execute(
                """
                SELECT l.lineup_index, p.player_name, p.player_name_key, p.projected_median
                FROM dfs_lineup_players p
                JOIN dfs_lineups l ON l.id = p.lineup_id
                WHERE l.build_id = ?
                ORDER BY l.lineup_index, p.slot
                """,
                (build_id,),
            )
        )


def _load_saved_projected_players(build_id: int, *, db_path: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(
            conn.execute(
                """
                SELECT *
                FROM dfs_projected_players
                WHERE build_id = ?
                ORDER BY median_fpts DESC, salary DESC, player_name ASC
                """,
                (build_id,),
            )
        )
    if not rows:
        raise ValueError(f"Build {build_id} does not have a persisted projected slate")
    return [dict(row) for row in rows]


def _load_lineup_audit_rows(build_id: int, *, db_path: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(
            conn.execute(
                """
                SELECT
                    l.lineup_index,
                    COALESCE(SUM(p.projected_median), 0.0) AS projected_median_sum,
                    COALESCE(r.actual_points, 0.0) AS actual_points,
                    GROUP_CONCAT(p.player_name, '|') AS players
                FROM dfs_lineups l
                JOIN dfs_lineup_players p ON p.lineup_id = l.id
                LEFT JOIN dfs_lineup_results r ON r.lineup_id = l.id
                WHERE l.build_id = ?
                GROUP BY l.id, l.lineup_index, r.actual_points
                ORDER BY l.lineup_index
                """,
                (build_id,),
            )
        )
    return [
        {
            "lineup_index": int(row["lineup_index"]),
            "projected_median_sum": float(row["projected_median_sum"] or 0.0),
            "actual_points": float(row["actual_points"] or 0.0),
            "players": tuple(str(row["players"] or "").split("|")),
        }
        for row in rows
    ]


def _espn_stat(stats: list[Any], label_idx: dict[str, int], label: str) -> float:
    idx = label_idx.get(label)
    if idx is None or idx >= len(stats):
        return 0.0
    try:
        return float(stats[idx])
    except (TypeError, ValueError):
        return 0.0


def _espn_game_key(payload: dict[str, Any]) -> str:
    competitions: Iterable[dict[str, Any]] = payload.get("header", {}).get("competitions", []) or []
    if not competitions:
        return ""
    competitors = competitions[0].get("competitors", []) or []
    away = next((item for item in competitors if item.get("homeAway") == "away"), None)
    home = next((item for item in competitors if item.get("homeAway") == "home"), None)
    if away and home:
        return f"{away['team']['abbreviation']}@{home['team']['abbreviation']}"
    if len(competitors) == 2:
        return f"{competitors[0]['team']['abbreviation']}@{competitors[1]['team']['abbreviation']}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a saved DFS build against ESPN NBA box scores.")
    parser.add_argument("--build-id", type=int, required=True, help="Saved DFS build id from dfs_results.sqlite")
    parser.add_argument("--db-path", default=DEFAULT_DFS_RESULTS_DB_PATH)
    args = parser.parse_args()

    report = audit_saved_nba_build(args.build_id, db_path=args.db_path)
    print_nba_build_audit(report)


if __name__ == "__main__":
    main()
