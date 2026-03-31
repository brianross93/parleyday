from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class OptimizedLineupSolution:
    player_indices: tuple[int, ...]
    slot_assignments: tuple[tuple[str, int], ...]
    objective_score: float


@dataclass(frozen=True)
class CSVPlayerProjection:
    name: str
    positions: tuple[str, ...]
    salary: int
    projected: float
    team: str = ""
    game: str = ""


@dataclass(frozen=True)
class RosterConfig:
    sport: str
    slots: tuple[str, ...]


ROSTER_CONFIGS: dict[str, RosterConfig] = {
    "nba": RosterConfig(sport="nba", slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")),
    "mlb": RosterConfig(sport="mlb", slots=("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF")),
    "nfl": RosterConfig(sport="nfl", slots=("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST")),
}


def solve_dfs_lineups(
    *,
    player_count: int,
    slot_names: Sequence[str],
    salary_cap: int,
    salaries: Sequence[int],
    player_scores: Sequence[float],
    eligibility_fn: Callable[[int, str], bool],
    lineups_to_generate: int,
    max_players_per_game: int | None = None,
    game_keys: Sequence[str] | None = None,
    game_countable_fn: Callable[[int], bool] | None = None,
    objective_noise_scale: float = 0.0,
    rng_seed: int = 0,
    required_player_indices: Sequence[int] | None = None,
    max_exposure: float | None = None,
) -> list[OptimizedLineupSolution]:
    if player_count <= 0 or not slot_names:
        return []

    var_index: dict[tuple[int, int], int] = {}
    objective: list[float] = []
    salaries_vector: list[float] = []
    next_var = 0
    for player_idx in range(player_count):
        for slot_idx, slot_name in enumerate(slot_names):
            if not eligibility_fn(player_idx, slot_name):
                continue
            var_index[(player_idx, slot_idx)] = next_var
            objective.append(-float(player_scores[player_idx]))
            salaries_vector.append(float(salaries[player_idx]))
            next_var += 1
    if not var_index:
        return []

    num_vars = next_var
    constraints: list[LinearConstraint] = []
    player_rows: list[np.ndarray] = []

    for slot_idx in range(len(slot_names)):
        row = np.zeros(num_vars)
        for player_idx in range(player_count):
            var = var_index.get((player_idx, slot_idx))
            if var is not None:
                row[var] = 1.0
        constraints.append(LinearConstraint(row, 1.0, 1.0))

    for player_idx in range(player_count):
        row = np.zeros(num_vars)
        for slot_idx in range(len(slot_names)):
            var = var_index.get((player_idx, slot_idx))
            if var is not None:
                row[var] = 1.0
        player_rows.append(row)
        constraints.append(LinearConstraint(row, 0.0, 1.0))

    for player_idx in sorted(set(int(idx) for idx in (required_player_indices or []) if 0 <= int(idx) < player_count)):
        constraints.append(LinearConstraint(player_rows[player_idx], 1.0, 1.0))

    constraints.append(LinearConstraint(np.array(salaries_vector), -np.inf, float(salary_cap)))

    if max_players_per_game and game_keys:
        countable = game_countable_fn or (lambda _idx: True)
        unique_games = sorted({str(game_keys[idx] or "").strip() for idx in range(player_count) if str(game_keys[idx] or "").strip()})
        for game in unique_games:
            row = np.zeros(num_vars)
            for player_idx in range(player_count):
                if not countable(player_idx):
                    continue
                if str(game_keys[player_idx] or "").strip() != game:
                    continue
                row += player_rows[player_idx]
            constraints.append(LinearConstraint(row, -np.inf, float(max_players_per_game)))

    bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
    integrality = np.ones(num_vars, dtype=int)
    c = np.array(objective)
    rng = np.random.default_rng(rng_seed)

    solutions: list[OptimizedLineupSolution] = []
    no_good_rows: list[np.ndarray] = []
    lineup_size = len(slot_names)
    exposure_counts = [0] * player_count
    max_player_appearances: int | None = None
    if max_exposure is not None:
        bounded = max(0.0, min(1.0, float(max_exposure)))
        max_player_appearances = max(1, int(math.floor(bounded * max(1, int(lineups_to_generate)))))

    for _ in range(max(1, int(lineups_to_generate))):
        active_constraints = constraints + [LinearConstraint(row, -np.inf, float(lineup_size - 1)) for row in no_good_rows]
        if max_player_appearances is not None:
            for player_idx, row in enumerate(player_rows):
                if exposure_counts[player_idx] >= max_player_appearances:
                    active_constraints.append(LinearConstraint(row, 0.0, 0.0))
        solve_c = c
        if objective_noise_scale > 0:
            solve_c = c + rng.normal(0.0, float(objective_noise_scale), size=num_vars)
        result = milp(c=solve_c, constraints=active_constraints, bounds=bounds, integrality=integrality)
        if result.status != 0 or result.x is None:
            break
        selected = np.where(result.x > 0.5)[0]
        player_indices = sorted({player_idx for (player_idx, _slot_idx), var in var_index.items() if var in selected})
        if len(player_indices) != lineup_size:
            break
        slot_assignments: list[tuple[str, int]] = []
        player_set_row = np.zeros(num_vars)
        for (player_idx, slot_idx), var in var_index.items():
            if var in selected:
                slot_assignments.append((slot_names[slot_idx], player_idx))
        for player_idx in player_indices:
            player_set_row += player_rows[player_idx]
        slot_assignments.sort(key=lambda item: (slot_names.index(item[0]), item[1]))
        solutions.append(
            OptimizedLineupSolution(
                player_indices=tuple(player_indices),
                slot_assignments=tuple(slot_assignments),
                objective_score=float(-result.fun),
            )
        )
        no_good_rows.append(player_set_row)
        for player_idx in player_indices:
            exposure_counts[player_idx] += 1

    return solutions


def player_eligible_for_slot(positions: Sequence[str], slot: str, sport: str) -> bool:
    pos = {str(item).strip().upper() for item in positions if str(item).strip()}
    slot_key = str(slot).strip().upper()
    sport_key = str(sport).strip().lower()
    if sport_key == "nba":
        if slot_key == "UTIL":
            return bool(pos)
        if slot_key == "G":
            return bool(pos & {"PG", "SG"})
        if slot_key == "F":
            return bool(pos & {"SF", "PF"})
        return slot_key in pos
    if sport_key == "mlb":
        if slot_key == "P":
            return bool(pos & {"P", "SP", "RP"})
        return slot_key in pos
    if sport_key == "nfl":
        if slot_key == "FLEX":
            return bool(pos & {"RB", "WR", "TE"})
        if slot_key == "DST":
            return bool(pos & {"DST", "DEF", "D/ST"})
        return slot_key in pos
    return False


def load_csv_player_pool(csv_path: str | Path, sport: str) -> list[CSVPlayerProjection]:
    sport_key = str(sport).strip().lower()
    if sport_key not in ROSTER_CONFIGS:
        raise ValueError(f"Unsupported sport: {sport}")
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    header_map = {_normalize_header(name): name for name in fieldnames}
    name_key = _require_header(header_map, ("name", "player", "playername"))
    position_key = _require_header(header_map, ("position", "positions", "pos", "rosterposition", "roster_position"))
    salary_key = _require_header(header_map, ("salary", "cost", "price"))
    projection_key = _require_header(header_map, ("projected", "projection", "proj", "fpts", "fantasypoints", "fantasy_points"))
    team_key = _resolve_header(header_map, ("team", "teamabbrev", "team_abbrev", "teamabbr"))
    game_key = _resolve_header(header_map, ("game", "gameinfo", "game_info", "matchup"))

    players: list[CSVPlayerProjection] = []
    for row in rows:
        raw_name = str(row.get(name_key, "")).strip()
        if not raw_name:
            continue
        positions = _parse_positions(str(row.get(position_key, "")))
        if not positions:
            continue
        salary = _coerce_int(row.get(salary_key))
        projected = _coerce_float(row.get(projection_key))
        if salary <= 0:
            continue
        players.append(
            CSVPlayerProjection(
                name=raw_name,
                positions=positions,
                salary=salary,
                projected=projected,
                team=str(row.get(team_key, "")).strip() if team_key else "",
                game=str(row.get(game_key, "")).strip() if game_key else "",
            )
        )
    return players


def optimize_csv_lineups(
    *,
    players: Sequence[CSVPlayerProjection],
    sport: str,
    top: int,
    salary_cap: int,
    max_exposure: float | None = None,
    objective_noise_scale: float = 0.0,
) -> list[OptimizedLineupSolution]:
    config = ROSTER_CONFIGS[str(sport).strip().lower()]
    return solve_dfs_lineups(
        player_count=len(players),
        slot_names=config.slots,
        salary_cap=salary_cap,
        salaries=[player.salary for player in players],
        player_scores=[player.projected for player in players],
        eligibility_fn=lambda idx, slot: player_eligible_for_slot(players[idx].positions, slot, config.sport),
        lineups_to_generate=top,
        objective_noise_scale=objective_noise_scale,
        max_exposure=max_exposure,
        game_keys=[player.game for player in players],
    )


def summarize_exposure(players: Sequence[CSVPlayerProjection], solutions: Sequence[OptimizedLineupSolution]) -> list[tuple[str, int, float]]:
    if not solutions:
        return []
    counts: dict[int, int] = {}
    for solution in solutions:
        for player_idx in solution.player_indices:
            counts[player_idx] = counts.get(player_idx, 0) + 1
    total = len(solutions)
    exposure = [(players[idx].name, count, count / total) for idx, count in counts.items()]
    exposure.sort(key=lambda item: (-item[1], item[0]))
    return exposure


def export_lineups_csv(
    output_path: str | Path,
    *,
    players: Sequence[CSVPlayerProjection],
    solutions: Sequence[OptimizedLineupSolution],
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Lineup", "Slot", "Name", "Positions", "Salary", "Projected", "Team", "Game"),
        )
        writer.writeheader()
        for lineup_index, solution in enumerate(solutions, start=1):
            for slot_name, player_idx in solution.slot_assignments:
                player = players[player_idx]
                writer.writerow(
                    {
                        "Lineup": lineup_index,
                        "Slot": slot_name,
                        "Name": player.name,
                        "Positions": "/".join(player.positions),
                        "Salary": player.salary,
                        "Projected": f"{player.projected:.2f}",
                        "Team": player.team,
                        "Game": player.game,
                    }
                )


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _resolve_header(header_map: dict[str, str], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        key = _normalize_header(alias)
        if key in header_map:
            return header_map[key]
    return None


def _require_header(header_map: dict[str, str], aliases: Sequence[str]) -> str:
    resolved = _resolve_header(header_map, aliases)
    if resolved is None:
        raise ValueError(f"Missing required CSV column matching one of: {', '.join(aliases)}")
    return resolved


def _parse_positions(value: str) -> tuple[str, ...]:
    parts = [part.strip().upper() for part in re.split(r"[/,;|]", value or "") if part.strip()]
    return tuple(dict.fromkeys(parts))


def _coerce_int(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _demo_players(sport: str) -> list[CSVPlayerProjection]:
    sport_key = sport.lower()
    if sport_key == "nba":
        rows = [
            ("Guard A", ("PG",), 9000, 48.0), ("Guard B", ("SG",), 7800, 39.0), ("Wing A", ("SF",), 7200, 37.0),
            ("Forward A", ("PF",), 8100, 42.0), ("Center A", ("C",), 8300, 41.0), ("Combo G", ("PG", "SG"), 6400, 31.0),
            ("Combo F", ("SF", "PF"), 5800, 28.0), ("Value G", ("SG",), 4200, 22.0), ("Value F", ("PF",), 3900, 20.5),
            ("Value C", ("C",), 3600, 18.5), ("Bench Wing", ("SF",), 3400, 17.0), ("Bench Guard", ("PG",), 3200, 16.0),
        ]
    elif sport_key == "mlb":
        rows = [
            ("Pitcher A", ("P",), 9800, 24.0), ("Pitcher B", ("P",), 8600, 20.0), ("Pitcher C", ("P",), 7300, 17.0),
            ("Catcher A", ("C",), 3200, 9.5), ("First A", ("1B",), 5100, 12.0), ("Second A", ("2B",), 4700, 11.2),
            ("Third A", ("3B",), 4900, 11.6), ("Short A", ("SS",), 5300, 12.4), ("Outfield A", ("OF",), 5400, 12.2),
            ("Outfield B", ("OF",), 4300, 10.4), ("Outfield C", ("OF",), 3900, 9.8), ("Outfield D", ("OF",), 3000, 7.5),
        ]
    else:
        rows = [
            ("QB A", ("QB",), 7200, 23.0), ("RB A", ("RB",), 7600, 20.0), ("RB B", ("RB",), 6500, 17.0),
            ("WR A", ("WR",), 8100, 22.0), ("WR B", ("WR",), 6900, 18.0), ("WR C", ("WR",), 5500, 14.0),
            ("TE A", ("TE",), 4800, 11.0), ("FLEX A", ("RB", "WR"), 4200, 10.5), ("DST A", ("DST",), 3000, 8.0),
            ("WR D", ("WR",), 3600, 9.0), ("RB C", ("RB",), 4100, 10.0),
        ]
    return [CSVPlayerProjection(name=name, positions=positions, salary=salary, projected=projected) for name, positions, salary, projected in rows]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve DFS lineups from a projection CSV.")
    parser.add_argument("--csv", dest="csv_path", help="Input CSV with Name, Position, Salary, Projected columns.")
    parser.add_argument("--sport", choices=sorted(ROSTER_CONFIGS), default="nba")
    parser.add_argument("--top", type=int, default=20, help="Number of lineups to generate.")
    parser.add_argument("--salary-cap", type=int, default=50000)
    parser.add_argument("--max-exposure", type=float, default=None, help="Max player exposure as a decimal, e.g. 0.6")
    parser.add_argument("--export", dest="export_path", help="Optional CSV path for solved lineups.")
    parser.add_argument("--demo", action="store_true", help="Use a built-in demo player pool.")
    parser.add_argument("--noise", type=float, default=0.0, help="Optional objective noise for diversification.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.demo and not args.csv_path:
        parser.error("Provide --csv or use --demo.")
    players = _demo_players(args.sport) if args.demo else load_csv_player_pool(args.csv_path, args.sport)
    if not players:
        print("No players loaded.", file=sys.stderr)
        return 1
    solutions = optimize_csv_lineups(
        players=players,
        sport=args.sport,
        top=args.top,
        salary_cap=args.salary_cap,
        max_exposure=args.max_exposure,
        objective_noise_scale=args.noise,
    )
    if not solutions:
        print("No valid lineups found.", file=sys.stderr)
        return 1
    for lineup_index, solution in enumerate(solutions, start=1):
        salary_used = sum(players[idx].salary for idx in solution.player_indices)
        print(f"Lineup {lineup_index}: score={solution.objective_score:.2f} salary={salary_used}")
        for slot_name, player_idx in solution.slot_assignments:
            player = players[player_idx]
            print(f"  {slot_name:>4}  {player.name}  ({'/'.join(player.positions)})  salary={player.salary} proj={player.projected:.2f}")
    print("\nExposure")
    for name, count, exposure in summarize_exposure(players, solutions):
        print(f"  {name}: {count}/{len(solutions)} ({exposure:.0%})")
    if args.export_path:
        export_lineups_csv(args.export_path, players=players, solutions=solutions)
        print(f"\nExported lineups to {args.export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
