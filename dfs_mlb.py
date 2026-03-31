from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from dfs_optimizer import solve_dfs_lineups
from dfs_ingest import DraftKingsPlayer, DraftKingsSlate
from player_name_utils import dfs_name_key
from quantum_parlay_oracle import (
    expected_mlb_runs,
    load_game_context_snapshot,
    load_matchup_profile_snapshot,
    load_team_form_snapshot,
)


@dataclass(frozen=True)
class DraftKingsMLBProjection:
    player_id: str
    name: str
    team: str
    opponent: str
    salary: int
    positions: tuple[str, ...]
    roster_positions: tuple[str, ...]
    game: str
    median_fpts: float
    ceiling_fpts: float
    floor_fpts: float
    volatility: float
    projection_confidence: float
    plate_appearances: float
    innings_pitched: float
    hits: float
    home_runs: float
    stolen_bases: float
    strikeouts: float
    runs_allowed: float
    availability_status: str
    availability_source: str


@dataclass(frozen=True)
class DraftKingsMLBLineup:
    players: tuple[DraftKingsMLBProjection, ...]
    salary_used: int
    median_fpts: float
    ceiling_fpts: float
    floor_fpts: float
    average_confidence: float
    unknown_count: int


MLB_CLASSIC_SLOTS = ("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF")


def draftkings_mlb_hitter_fpts(
    *,
    singles: float,
    doubles: float,
    triples: float,
    home_runs: float,
    runs_batted_in: float,
    runs: float,
    walks_hbp: float,
    stolen_bases: float,
) -> float:
    return float(
        (singles * 3.0)
        + (doubles * 5.0)
        + (triples * 8.0)
        + (home_runs * 10.0)
        + (runs_batted_in * 2.0)
        + (runs * 2.0)
        + (walks_hbp * 2.0)
        + (stolen_bases * 5.0)
    )


def draftkings_mlb_pitcher_fpts(
    *,
    innings_pitched: float,
    strikeouts: float,
    win_prob: float,
    earned_runs: float,
    hits_allowed: float,
    walks_allowed: float,
) -> float:
    return float(
        (innings_pitched * 2.25)
        + (strikeouts * 2.0)
        + (win_prob * 4.0)
        - (earned_runs * 2.0)
        - (hits_allowed * 0.6)
        - (walks_allowed * 0.6)
    )


def build_mlb_dk_projections(date_str: str, slate: DraftKingsSlate) -> list[DraftKingsMLBProjection]:
    if slate.sport != "mlb":
        return []
    team_form = load_team_form_snapshot(date_str, "mlb")
    game_cache: dict[str, dict[str, DraftKingsMLBProjection]] = {}
    game_context_cache: dict[str, dict | None] = {}
    projections: list[DraftKingsMLBProjection] = []
    for player in slate.players:
        game = str(player.game or "")
        if not game or "@" not in game:
            continue
        if game not in game_cache:
            game_cache[game] = _build_game_projection_lookup(date_str, game, team_form)
        if game not in game_context_cache:
            game_context_cache[game] = load_game_context_snapshot(date_str, "mlb", game)
        lookup = game_cache.get(game, {})
        projected = lookup.get(dfs_name_key(player.name))
        if projected is None:
            projected = _fallback_projection(
                player,
                probable_name=_probable_pitcher_name_for_player(player, game_context_cache.get(game)),
            )
        if projected.availability_status == "out":
            continue
        projections.append(projected)
    return sorted(projections, key=lambda item: (item.median_fpts, -item.salary), reverse=True)


def attach_mlb_salary_metadata(
    slate: DraftKingsSlate,
    projections: Iterable[DraftKingsMLBProjection],
) -> list[DraftKingsMLBProjection]:
    by_name = {dfs_name_key(player.name): player for player in slate.players}
    enriched: list[DraftKingsMLBProjection] = []
    for projection in projections:
        source = by_name.get(dfs_name_key(projection.name))
        if source is None:
            enriched.append(projection)
            continue
        enriched.append(
            DraftKingsMLBProjection(
                player_id=source.player_id,
                name=projection.name,
                team=projection.team or source.team,
                opponent=projection.opponent or source.opponent,
                salary=source.salary,
                positions=source.positions,
                roster_positions=source.roster_positions,
                game=projection.game or source.game,
                median_fpts=projection.median_fpts,
                ceiling_fpts=projection.ceiling_fpts,
                floor_fpts=projection.floor_fpts,
                volatility=projection.volatility,
                projection_confidence=projection.projection_confidence,
                plate_appearances=projection.plate_appearances,
                innings_pitched=projection.innings_pitched,
                hits=projection.hits,
                home_runs=projection.home_runs,
                stolen_bases=projection.stolen_bases,
                strikeouts=projection.strikeouts,
                runs_allowed=projection.runs_allowed,
                availability_status=projection.availability_status,
                availability_source=projection.availability_source,
            )
        )
    return enriched


def optimize_mlb_classic_lineups(
    projections: list[DraftKingsMLBProjection],
    salary_cap: int = 50000,
    max_candidates: int | None = None,
    limit: int = 10,
    contest_type: str = "cash",
    focus_players: set[str] | None = None,
    fade_players: set[str] | None = None,
    game_boosts: dict[str, float] | None = None,
    stack_targets: set[str] | None = None,
    bring_back_targets: set[str] | None = None,
    one_off_targets: set[str] | None = None,
    max_players_per_game: int | None = None,
    preferred_salary_shape: str | None = None,
    objective_noise_scale: float = 0.0,
    max_exposure: float | None = None,
    locked_players: set[str] | None = None,
) -> list[DraftKingsMLBLineup]:
    focus_keys = {dfs_name_key(name) for name in (focus_players or set()) if dfs_name_key(name)}
    fade_keys = {dfs_name_key(name) for name in (fade_players or set()) if dfs_name_key(name)}
    stack_keys = {dfs_name_key(name) for name in (stack_targets or set()) if dfs_name_key(name)}
    bring_back_keys = {dfs_name_key(name) for name in (bring_back_targets or set()) if dfs_name_key(name)}
    one_off_keys = {dfs_name_key(name) for name in (one_off_targets or set()) if dfs_name_key(name)}
    locked_keys = {dfs_name_key(name) for name in (locked_players or set()) if dfs_name_key(name)}
    environment_boosts = dict(game_boosts or {})
    eligible = [item for item in projections if item.availability_status != "out"]
    ranked_eligible = sorted(
        eligible,
        key=lambda item: (
            _lineup_pool_score(
                item, contest_type, focus_keys, fade_keys, environment_boosts, stack_keys, bring_back_keys, one_off_keys
            ),
            item.ceiling_fpts,
            -item.salary,
        ),
        reverse=True,
    )
    if max_candidates is None:
        candidate_pool = ranked_eligible
    else:
        value_slice = max(14, max_candidates // 2)
        salary_slice = max(10, max_candidates // 4)
        ranked_by_median = ranked_eligible[:value_slice]
        ranked_by_value = sorted(
            eligible,
            key=lambda item: (
                _lineup_pool_score(
                    item, contest_type, focus_keys, fade_keys, environment_boosts, stack_keys, bring_back_keys, one_off_keys
                )
                / max(item.salary, 1),
                item.ceiling_fpts,
            ),
            reverse=True,
        )[: value_slice]
        ranked_by_salary_relief = sorted(
            eligible,
            key=lambda item: (
                item.salary,
                -_lineup_pool_score(
                    item, contest_type, focus_keys, fade_keys, environment_boosts, stack_keys, bring_back_keys, one_off_keys
                ),
            ),
        )[:salary_slice]
        ranked_by_position_coverage: list[DraftKingsMLBProjection] = []
        for slot_name, cap in (("P", 10), ("C", 4), ("1B", 5), ("2B", 5), ("3B", 5), ("SS", 5), ("OF", 12)):
            slot_players = [
                item
                for item in eligible
                if _mlb_can_fill_slot(item, slot_name)
            ]
            slot_players.sort(
                key=lambda item: _lineup_pool_score(
                    item, contest_type, focus_keys, fade_keys, environment_boosts, stack_keys, bring_back_keys, one_off_keys
                ),
                reverse=True,
            )
            ranked_by_position_coverage.extend(slot_players[:cap])
        candidate_pool = list(
            {
                item.player_id or item.name: item
                for item in (ranked_by_median + ranked_by_value + ranked_by_salary_relief + ranked_by_position_coverage)
            }.values()
        )
        if contest_type in {"single_entry_gpp", "large_field_gpp"}:
            core_keys = {item.player_id or item.name for item in candidate_pool}
            remaining_pool = [item for item in eligible if (item.player_id or item.name) not in core_keys]
            tail_size = min(max(6, max_candidates // 8), len(remaining_pool))
            if tail_size > 0:
                rng = random.Random(f"mlb:{contest_type}:{pool_signature(candidate_pool)}:{salary_cap}:{limit}")
                candidate_pool.extend(rng.sample(remaining_pool, tail_size))
    lineups: list[DraftKingsMLBLineup] = []
    player_scores = [
        _lineup_pool_score(item, contest_type, focus_keys, fade_keys, environment_boosts, stack_keys, bring_back_keys, one_off_keys)
        for item in candidate_pool
    ]
    required_player_indices = [
        idx for idx, item in enumerate(candidate_pool)
        if dfs_name_key(item.name) in locked_keys
    ]
    solved = solve_dfs_lineups(
        player_count=len(candidate_pool),
        slot_names=MLB_CLASSIC_SLOTS,
        salary_cap=salary_cap,
        salaries=[item.salary for item in candidate_pool],
        player_scores=player_scores,
        eligibility_fn=lambda player_idx, slot_name: _mlb_can_fill_slot(candidate_pool[player_idx], slot_name),
        lineups_to_generate=limit,
        max_players_per_game=max_players_per_game,
        game_keys=[item.game for item in candidate_pool],
        game_countable_fn=lambda player_idx: "P" not in set(candidate_pool[player_idx].positions),
        objective_noise_scale=objective_noise_scale,
        rng_seed=29,
        required_player_indices=required_player_indices,
        max_exposure=max_exposure,
    )
    for solved_lineup in solved:
        lineup_players = tuple(candidate_pool[idx] for idx in solved_lineup.player_indices)
        salary_used = sum(player.salary for player in lineup_players)
        median = sum(player.median_fpts for player in lineup_players)
        ceiling = sum(player.ceiling_fpts for player in lineup_players)
        floor = sum(player.floor_fpts for player in lineup_players)
        average_confidence = sum(player.projection_confidence for player in lineup_players) / len(lineup_players)
        unknown_count = sum(1 for player in lineup_players if player.availability_status == "unknown")
        lineups.append(
            DraftKingsMLBLineup(
                players=tuple(sorted(lineup_players, key=lambda item: item.salary, reverse=True)),
                salary_used=salary_used,
                median_fpts=median,
                ceiling_fpts=ceiling,
                floor_fpts=floor,
                average_confidence=average_confidence,
                unknown_count=unknown_count,
            )
        )
    lineups.sort(
        key=lambda item: _lineup_rank_key(
            item, contest_type, salary_cap, environment_boosts, preferred_salary_shape, stack_keys, bring_back_keys, one_off_keys
        ),
        reverse=True,
    )
    return lineups[:limit]


def _build_game_projection_lookup(date_str: str, game: str, team_form: dict[str, dict]) -> dict[str, DraftKingsMLBProjection]:
    away_code, home_code = game.split("@", 1)
    game_context = load_game_context_snapshot(date_str, "mlb", game) or {}
    payload = load_matchup_profile_snapshot(date_str, game) or {}
    away_runs, home_runs = expected_mlb_runs(away_code, home_code, team_form, game_context)
    results: dict[str, DraftKingsMLBProjection] = {}
    weather = game_context.get("weather") or {}
    weather_boost = 1.0
    if float(weather.get("temperature_f") or 0.0) >= 82.0:
        weather_boost += 0.05
    if float(weather.get("wind_speed_mph") or 0.0) >= 12.0:
        weather_boost += 0.04
    for side, offense_code, defense_code, lineup_key, pitcher_key, target_runs in (
        ("away", away_code, home_code, "away_lineup", "home_pitcher", away_runs),
        ("home", home_code, away_code, "home_lineup", "away_pitcher", home_runs),
    ):
        lineup_status = game_context.get("lineup_status") or {}
        lineups = game_context.get("lineups") or {}
        lineup_names = set(lineups.get(side, []) or [])
        lineup_source = str(lineup_status.get(f"{side}_source") or "missing")
        lineup_available = bool(lineup_status.get(f"{side}_available"))
        opponent_pitcher = payload.get(pitcher_key) or {}
        pitcher_contact_factor = 1.0 + max(0.0, float(opponent_pitcher.get("single_rate", 0.155)) - 0.155) * 1.8
        pitcher_power_factor = 1.0 + max(0.0, float(opponent_pitcher.get("home_run_rate", 0.032)) - 0.032) * 3.5
        pitcher_k_factor = 1.0 + max(0.0, float(opponent_pitcher.get("strikeout_rate", 0.225)) - 0.225) * 1.7
        team_env = max(0.78, min(1.35, (target_runs / 4.3) * weather_boost))
        for batter in payload.get(lineup_key) or []:
            name = str(batter.get("name") or "").strip()
            if not name:
                continue
            split_factor = float(
                batter.get("vs_left_factor", 1.0)
                if str(opponent_pitcher.get("hand") or "R").upper() == "L"
                else batter.get("vs_right_factor", 1.0)
            )
            pa_share = float(batter.get("pa_share", 0.11) or 0.11)
            expected_pa = max(3.1, min(5.2, 38.0 * pa_share))
            single_rate = float(batter.get("single_rate", 0.155) or 0.155) * split_factor * pitcher_contact_factor * team_env
            double_rate = float(batter.get("double_rate", 0.045) or 0.045) * split_factor * (0.96 + (team_env - 1.0) * 0.5)
            triple_rate = float(batter.get("triple_rate", 0.005) or 0.005) * split_factor
            home_run_rate = float(batter.get("home_run_rate", 0.032) or 0.032) * split_factor * pitcher_power_factor * team_env
            walk_rate = float(batter.get("walk_rate", 0.085) or 0.085)
            hbp_rate = float(batter.get("hbp_rate", 0.010) or 0.010)
            strikeout_rate = float(batter.get("strikeout_rate", 0.225) or 0.225) * pitcher_k_factor
            in_lineup = name in lineup_names or not lineup_available
            status = "active" if in_lineup else "unknown"
            availability_source = f"lineup_{lineup_source}" if lineup_source else "lineup"

            singles = expected_pa * max(0.0, min(single_rate, 0.28))
            doubles = expected_pa * max(0.0, min(double_rate, 0.12))
            triples = expected_pa * max(0.0, min(triple_rate, 0.02))
            home_runs = expected_pa * max(0.0, min(home_run_rate, 0.11))
            walks_hbp = expected_pa * max(0.0, min(walk_rate + hbp_rate, 0.24))
            hits = singles + doubles + triples + home_runs
            runs = max(0.15, (hits + walks_hbp) * (target_runs / 9.5))
            rbi = max(0.15, (hits + home_runs) * (target_runs / 9.0))
            stolen_bases = expected_pa * max(0.0, min(((float(batter.get("speed_factor", 1.0) or 1.0) - 1.0) * 0.065), 0.10))
            median = draftkings_mlb_hitter_fpts(
                singles=singles,
                doubles=doubles,
                triples=triples,
                home_runs=home_runs,
                runs_batted_in=rbi,
                runs=runs,
                walks_hbp=walks_hbp,
                stolen_bases=stolen_bases,
            )
            volatility = min(0.6, 0.28 + (home_runs * 1.3) + (stolen_bases * 0.9) + (0.06 if strikeout_rate > 0.24 else 0.0))
            confidence = 0.72 if lineup_source == "confirmed" else 0.58 if lineup_source == "last_fielded" else 0.45
            results[dfs_name_key(name)] = DraftKingsMLBProjection(
                player_id="",
                name=name,
                team=offense_code,
                opponent=defense_code,
                salary=0,
                positions=tuple(),
                roster_positions=tuple(),
                game=game,
                median_fpts=median,
                ceiling_fpts=median * (1.0 + volatility * 0.95),
                floor_fpts=max(0.0, median * (1.0 - volatility * 0.6)),
                volatility=volatility,
                projection_confidence=confidence,
                plate_appearances=expected_pa,
                innings_pitched=0.0,
                hits=hits,
                home_runs=home_runs,
                stolen_bases=stolen_bases,
                strikeouts=0.0,
                runs_allowed=0.0,
                availability_status=status,
                availability_source=availability_source,
            )
    for side, pitcher_key, team_code, opponent_code, own_runs, opp_runs in (
        ("away", "away_pitcher", away_code, home_code, away_runs, home_runs),
        ("home", "home_pitcher", home_code, away_code, home_runs, away_runs),
    ):
        pitcher = payload.get(pitcher_key) or {}
        name = str(pitcher.get("name") or "").strip()
        if not name:
            continue
        opponent_lineup = payload.get("home_lineup" if side == "away" else "away_lineup") or []
        opponent_k_factor = sum(float(item.get("strikeout_rate", 0.225) or 0.225) for item in opponent_lineup[:9]) / max(len(opponent_lineup[:9]), 1)
        innings = min(max(float(pitcher.get("fatigue_start", 80) or 80) / 15.5, 4.8), 7.1)
        batters_faced = innings * 4.2
        strikeouts = batters_faced * float(pitcher.get("strikeout_rate", 0.225) or 0.225) * (0.92 + (opponent_k_factor / 0.225) * 0.08)
        hits_allowed = batters_faced * (
            float(pitcher.get("single_rate", 0.155) or 0.155)
            + float(pitcher.get("double_rate", 0.045) or 0.045)
            + float(pitcher.get("triple_rate", 0.005) or 0.005)
            + float(pitcher.get("home_run_rate", 0.032) or 0.032)
        )
        walks_allowed = batters_faced * (
            float(pitcher.get("walk_rate", 0.085) or 0.085) + float(pitcher.get("hbp_rate", 0.010) or 0.010)
        )
        earned_runs = max(0.4, opp_runs * 0.72)
        run_diff = own_runs - opp_runs
        win_prob = 1.0 / (1.0 + math.exp(-(run_diff * 0.75)))
        status = "active"
        context_pitcher = (game_context.get("probable_pitchers") or {}).get(side) or {}
        if str(context_pitcher.get("fullName") or "").strip() and str(context_pitcher.get("fullName") or "").strip() != name:
            status = "unknown"
        median = draftkings_mlb_pitcher_fpts(
            innings_pitched=innings,
            strikeouts=strikeouts,
            win_prob=win_prob,
            earned_runs=earned_runs,
            hits_allowed=hits_allowed,
            walks_allowed=walks_allowed,
        )
        volatility = min(0.52, 0.24 + (strikeouts / 18.0) + (earned_runs / 12.0))
        confidence = 0.76 if status == "active" else 0.52
        results[dfs_name_key(name)] = DraftKingsMLBProjection(
            player_id="",
            name=name,
            team=team_code,
            opponent=opponent_code,
            salary=0,
            positions=tuple(),
            roster_positions=tuple(),
            game=game,
            median_fpts=median,
            ceiling_fpts=median * (1.0 + volatility),
            floor_fpts=max(0.0, median * (1.0 - volatility * 0.65)),
            volatility=volatility,
            projection_confidence=confidence,
            plate_appearances=0.0,
            innings_pitched=innings,
            hits=0.0,
            home_runs=0.0,
            stolen_bases=0.0,
            strikeouts=strikeouts,
            runs_allowed=earned_runs,
            availability_status=status,
            availability_source="probable_pitcher",
        )
    return results


def _fallback_projection(
    player: DraftKingsPlayer,
    *,
    probable_name: str | None = None,
) -> DraftKingsMLBProjection:
    avg = float(player.avg_points_per_game)
    positions = set(player.positions) | set(player.roster_positions)
    is_pitcher = bool(positions & {"P", "SP", "RP"})
    volatility = 0.36 if is_pitcher else 0.44
    availability_status = "unknown"
    availability_source = "fallback"
    projection_confidence = 0.4
    if is_pitcher:
        probable_key = dfs_name_key(probable_name or "")
        if probable_key:
            if probable_key == dfs_name_key(player.name):
                availability_status = "active"
                availability_source = "salary_pool_probable_pitcher"
                projection_confidence = 0.44
            else:
                availability_status = "out"
                availability_source = "not_probable_pitcher"
        else:
            availability_status = "active"
            availability_source = "salary_pool_pitcher"
    return DraftKingsMLBProjection(
        player_id=player.player_id,
        name=player.name,
        team=player.team,
        opponent=player.opponent,
        salary=player.salary,
        positions=player.positions,
        roster_positions=player.roster_positions,
        game=player.game,
        median_fpts=avg,
        ceiling_fpts=avg * 1.32,
        floor_fpts=max(0.0, avg * 0.64),
        volatility=volatility,
        projection_confidence=projection_confidence,
        plate_appearances=0.0,
        innings_pitched=0.0,
        hits=0.0,
        home_runs=0.0,
        stolen_bases=0.0,
        strikeouts=0.0,
        runs_allowed=0.0,
        availability_status=availability_status,
        availability_source=availability_source,
    )


def _probable_pitcher_name_for_player(player: DraftKingsPlayer, game_context: dict | None) -> str | None:
    positions = set(player.positions) | set(player.roster_positions)
    if not (positions & {"P", "SP", "RP"}):
        return None
    game = str(player.game or "")
    if "@" not in game:
        return None
    away_team, home_team = game.split("@", 1)
    probable_pitchers = (game_context or {}).get("probable_pitchers") or {}
    if player.team == away_team:
        probable = probable_pitchers.get("away") or {}
    elif player.team == home_team:
        probable = probable_pitchers.get("home") or {}
    else:
        return None
    name = str(probable.get("fullName") or "").strip()
    return name or None


def _lineup_pool_score(
    player: DraftKingsMLBProjection,
    contest_type: str,
    focus_keys: set[str],
    fade_keys: set[str],
    game_boosts: dict[str, float],
    stack_keys: set[str],
    bring_back_keys: set[str],
    one_off_keys: set[str],
) -> float:
    name_key = dfs_name_key(player.name)
    focus_bonus = 6.0 if name_key in focus_keys else 0.0
    fade_penalty = 8.0 if name_key in fade_keys else 0.0
    stack_bonus = 4.0 if name_key in stack_keys else 0.0
    bring_back_bonus = 2.4 if name_key in bring_back_keys else 0.0
    one_off_bonus = 1.8 if name_key in one_off_keys else 0.0
    base_conf = player.projection_confidence * 8.0
    environment_bonus = float(game_boosts.get(player.game, 0.0) or 0.0)
    is_pitcher = "P" in set(player.positions)
    if contest_type == "large_field_gpp":
        return (
            player.ceiling_fpts
            + (player.volatility * 8.0)
            + base_conf
            + focus_bonus
            + stack_bonus
            + bring_back_bonus
            + one_off_bonus
            + (environment_bonus * (4.0 if is_pitcher else 10.0))
            - fade_penalty
        )
    if contest_type == "single_entry_gpp":
        return (
            ((player.median_fpts * 0.72) + (player.ceiling_fpts * 0.42))
            + base_conf
            + focus_bonus
            + (stack_bonus * 0.8)
            + (bring_back_bonus * 0.7)
            + one_off_bonus
            + (environment_bonus * (2.2 if is_pitcher else 7.5))
            - fade_penalty
        )
    return (
        player.median_fpts
        + (player.floor_fpts * 0.32)
        + base_conf
        + focus_bonus
        + (stack_bonus * 0.2)
        + (bring_back_bonus * 0.15)
        + (one_off_bonus * 0.35)
        + (environment_bonus * (0.8 if is_pitcher else 3.5))
        - fade_penalty
    )


def _lineup_rank_key(
    lineup: DraftKingsMLBLineup,
    contest_type: str,
    salary_cap: int,
    game_boosts: dict[str, float],
    preferred_salary_shape: str | None,
    stack_keys: set[str],
    bring_back_keys: set[str],
    one_off_keys: set[str],
) -> tuple[float, float, float, float]:
    confidence_bonus = lineup.average_confidence * 10.0
    environment_bonus = _lineup_environment_bonus(lineup, contest_type, salary_cap, game_boosts)
    salary_shape_bonus = _lineup_salary_shape_bonus(lineup, salary_cap, preferred_salary_shape)
    guidance_bonus = _lineup_guidance_bonus(lineup, contest_type, stack_keys, bring_back_keys, one_off_keys)
    if contest_type == "large_field_gpp":
        return (
            lineup.ceiling_fpts + confidence_bonus + environment_bonus + salary_shape_bonus + guidance_bonus,
            lineup.median_fpts,
            lineup.floor_fpts,
            -lineup.salary_used,
        )
    if contest_type == "single_entry_gpp":
        blended = (lineup.median_fpts * 0.64) + (lineup.ceiling_fpts * 0.44)
        return (
            blended + confidence_bonus + environment_bonus + salary_shape_bonus + guidance_bonus,
            lineup.ceiling_fpts,
            lineup.floor_fpts,
            -lineup.salary_used,
        )
    blended = lineup.median_fpts + (lineup.floor_fpts * 0.34)
    return (
        blended + confidence_bonus + environment_bonus + salary_shape_bonus + guidance_bonus,
        lineup.median_fpts,
        lineup.ceiling_fpts,
        -lineup.salary_used,
    )


def _lineup_environment_bonus(
    lineup: DraftKingsMLBLineup,
    contest_type: str,
    salary_cap: int,
    game_boosts: dict[str, float],
) -> float:
    if not game_boosts:
        return 0.0
    game_counts: dict[str, int] = {}
    base_bonus = 0.0
    for player in lineup.players:
        if not player.game:
            continue
        game_counts[player.game] = game_counts.get(player.game, 0) + 1
        if "P" not in set(player.positions):
            base_bonus += float(game_boosts.get(player.game, 0.0) or 0.0)
    stack_bonus = sum(max(count - 1, 0) * float(game_boosts.get(game, 0.0) or 0.0) for game, count in game_counts.items())
    salary_remaining = max(0, salary_cap - lineup.salary_used)
    if contest_type == "large_field_gpp":
        return (base_bonus * 1.15) + (stack_bonus * 1.8) + (min(salary_remaining, 1600) / 260.0)
    if contest_type == "single_entry_gpp":
        return (base_bonus * 0.85) + (stack_bonus * 1.1) + (min(salary_remaining, 1100) / 420.0)
    concentration_penalty = sum(max(count - 4, 0) * 1.0 for count in game_counts.values())
    return (base_bonus * 0.45) - concentration_penalty


def _lineup_salary_shape_bonus(lineup: DraftKingsMLBLineup, salary_cap: int, preferred_salary_shape: str | None) -> float:
    shape = str(preferred_salary_shape or "").strip().lower()
    salary_remaining = max(0, salary_cap - lineup.salary_used)
    salaries = [player.salary for player in lineup.players]
    if not salaries:
        return 0.0
    avg_salary = sum(salaries) / len(salaries)
    variance = sum((salary - avg_salary) ** 2 for salary in salaries) / len(salaries)
    std_dev = variance ** 0.5
    if shape == "balanced":
        return max(0.0, 8.0 - (std_dev / 600.0)) - (salary_remaining / 900.0)
    if shape == "stars_and_scrubs":
        high_salary_count = sum(1 for salary in salaries if salary >= 5600)
        punt_count = sum(1 for salary in salaries if salary <= 2800)
        return (high_salary_count * 2.2) + (punt_count * 1.4) + (std_dev / 1350.0)
    if shape == "leave_salary":
        return min(salary_remaining, 1800) / 220.0
    return 0.0


def _lineup_guidance_bonus(
    lineup: DraftKingsMLBLineup,
    contest_type: str,
    stack_keys: set[str],
    bring_back_keys: set[str],
    one_off_keys: set[str],
) -> float:
    players_by_game: dict[str, list[DraftKingsMLBProjection]] = {}
    stack_hits = 0
    bring_back_hits = 0
    one_off_hits = 0
    for player in lineup.players:
        key = dfs_name_key(player.name)
        if key in stack_keys:
            stack_hits += 1
        if key in bring_back_keys:
            bring_back_hits += 1
        if key in one_off_keys:
            one_off_hits += 1
        if player.game:
            players_by_game.setdefault(player.game, []).append(player)
    game_correlation_bonus = 0.0
    for players in players_by_game.values():
        hitters = [player for player in players if "P" not in set(player.positions)]
        teams = {player.team for player in hitters if player.team}
        if len(hitters) >= 3 and len(teams) >= 1:
            game_correlation_bonus += 1.0
        if len(teams) >= 2 and any(dfs_name_key(player.name) in bring_back_keys for player in hitters):
            game_correlation_bonus += 1.2
    if contest_type == "large_field_gpp":
        return (stack_hits * 2.2) + (bring_back_hits * 1.6) + (one_off_hits * 0.9) + game_correlation_bonus
    if contest_type == "single_entry_gpp":
        return (stack_hits * 1.5) + (bring_back_hits * 0.9) + (one_off_hits * 0.7) + (game_correlation_bonus * 0.8)
    return (stack_hits * 0.25) + (bring_back_hits * 0.2) + (one_off_hits * 0.45)


def _is_valid_mlb_classic_lineup(players: tuple[DraftKingsMLBProjection, ...]) -> bool:
    if len(players) != 10:
        return False
    return _can_fill_slots(players, list(MLB_CLASSIC_SLOTS))


def _can_fill_slots(players: Sequence[DraftKingsMLBProjection], required_slots: list[str]) -> bool:
    slots = list(required_slots)

    def assign(slot_index: int, used: set[int]) -> bool:
        if slot_index >= len(slots):
            return True
        slot = slots[slot_index]
        for idx, player in enumerate(players):
            if idx in used:
                continue
            if not _mlb_can_fill_slot(player, slot):
                continue
            used.add(idx)
            if assign(slot_index + 1, used):
                return True
            used.remove(idx)
        return False

    return assign(0, set())


def _lineup_game_count_exceeds(players: tuple[DraftKingsMLBProjection, ...], max_players_per_game: int) -> bool:
    counts: dict[str, int] = {}
    for player in players:
        if not player.game or "P" in set(player.positions):
            continue
        counts[player.game] = counts.get(player.game, 0) + 1
        if counts[player.game] > max_players_per_game:
            return True
    return False


def _mlb_can_fill_slot(player: DraftKingsMLBProjection, slot_name: str) -> bool:
    positions = set(player.positions) | set(player.roster_positions)
    if slot_name == "P":
        return bool(positions & {"P", "SP", "RP"})
    return slot_name in positions


def pool_signature(players: list[DraftKingsMLBProjection]) -> str:
    return "|".join(sorted((player.player_id or player.name) for player in players[:16]))
