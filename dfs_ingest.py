from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


CENTRAL_TZ = ZoneInfo("America/Chicago")
EASTERN_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DraftKingsPlayer:
    player_id: str
    name: str
    sport: str
    team: str
    opponent: str
    game: str
    start_time: str | None
    salary: int
    positions: tuple[str, ...]
    roster_positions: tuple[str, ...]
    avg_points_per_game: float
    raw_position: str
    raw_game_info: str


@dataclass(frozen=True)
class DraftKingsSlate:
    site: str
    sport: str
    salary_cap: int
    roster_slots: tuple[str, ...]
    players: tuple[DraftKingsPlayer, ...]
    source_path: str


NBA_CLASSIC_SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
MLB_CLASSIC_SLOTS = ("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_positions(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split("/") if part.strip())


def _parse_game_info(game_info: str, team: str) -> tuple[str, str, str | None]:
    raw = str(game_info or "").strip()
    if not raw:
        return "", "", None
    parts = raw.split()
    matchup = parts[0] if parts else ""
    opponent = ""
    if "@" in matchup:
        away, home = matchup.split("@", 1)
        opponent = home if team == away else away if team == home else ""
    start_time = None
    if len(parts) >= 3:
        datetime_text = f"{parts[1]} {parts[2]}"
        try:
            dt = datetime.strptime(datetime_text, "%m/%d/%Y %I:%M%p")
            if len(parts) >= 4 and parts[3] == "ET":
                dt = dt.replace(tzinfo=EASTERN_TZ).astimezone(CENTRAL_TZ)
            else:
                dt = dt.replace(tzinfo=CENTRAL_TZ)
            start_time = dt.isoformat()
        except ValueError:
            start_time = None
    return matchup, opponent, start_time


def infer_dk_sport(players: list[DraftKingsPlayer]) -> str:
    position_pool = {position for player in players for position in player.positions}
    roster_position_pool = {position for player in players for position in player.roster_positions}
    if {"P", "SP", "RP", "OF", "SS", "2B", "3B", "1B"} & (position_pool | roster_position_pool):
        return "mlb"
    if {"PG", "SG", "SF", "PF"} & (position_pool | roster_position_pool):
        return "nba"
    if "C" in position_pool or "C" in roster_position_pool:
        if {"OF", "SS", "2B", "3B", "1B", "P", "SP", "RP"} & (position_pool | roster_position_pool):
            return "mlb"
        return "nba"
    return "unknown"


def draftkings_roster_slots_for_sport(sport: str) -> tuple[str, ...]:
    normalized = str(sport or "").strip().lower()
    if normalized == "nba":
        return NBA_CLASSIC_SLOTS
    if normalized == "mlb":
        return MLB_CLASSIC_SLOTS
    return tuple()


def parse_draftkings_salary_csv(path: str | Path, *, sport: str | None = None) -> DraftKingsSlate:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    players: list[DraftKingsPlayer] = []
    for row in rows:
        team = str(row.get("TeamAbbrev") or "").strip()
        matchup, opponent, start_time = _parse_game_info(str(row.get("Game Info") or ""), team)
        players.append(
            DraftKingsPlayer(
                player_id=str(row.get("ID") or "").strip(),
                name=str(row.get("Name") or "").strip(),
                sport=str(sport or "").strip().lower(),
                team=team,
                opponent=opponent,
                game=matchup,
                start_time=start_time,
                salary=_safe_int(row.get("Salary")),
                positions=_parse_positions(str(row.get("Position") or "")),
                roster_positions=_parse_positions(str(row.get("Roster Position") or "")),
                avg_points_per_game=_safe_float(row.get("AvgPointsPerGame")),
                raw_position=str(row.get("Position") or "").strip(),
                raw_game_info=str(row.get("Game Info") or "").strip(),
            )
        )
    inferred_sport = str(sport or infer_dk_sport(players)).strip().lower()
    normalized_players = tuple(
        DraftKingsPlayer(
            player_id=player.player_id,
            name=player.name,
            sport=inferred_sport,
            team=player.team,
            opponent=player.opponent,
            game=player.game,
            start_time=player.start_time,
            salary=player.salary,
            positions=player.positions,
            roster_positions=player.roster_positions,
            avg_points_per_game=player.avg_points_per_game,
            raw_position=player.raw_position,
            raw_game_info=player.raw_game_info,
        )
        for player in players
    )
    return DraftKingsSlate(
        site="draftkings",
        sport=inferred_sport,
        salary_cap=50000,
        roster_slots=draftkings_roster_slots_for_sport(inferred_sport),
        players=normalized_players,
        source_path=str(csv_path),
    )
