from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from dfs_ingest import parse_draftkings_salary_csv
from dfs_thesis_engine import build_mlb_dfs_analysis, build_nba_dfs_analysis
from dfs_mlb import (
    DraftKingsMLBLineup,
    DraftKingsMLBProjection,
    attach_mlb_salary_metadata,
    build_mlb_dk_projections,
    optimize_mlb_classic_lineups,
)
from dfs_nba import (
    DraftKingsLineup,
    DraftKingsNBAProjection,
    attach_salary_metadata,
    build_nba_dk_projections,
    optimize_nba_classic_lineups,
)
from player_name_utils import dfs_name_key


@dataclass(frozen=True)
class ContestBuildConfig:
    request_mode: str
    contest_type: str
    max_ranked_theses: int
    max_candidates: int | None
    limit: int
    include_leans: bool
    allow_candidate_fades: bool
    objective_noise_scale: float
    max_exposure: float | None
    cash_quality_floor: float | None


@dataclass(frozen=True)
class DFSLineupSlot:
    slot: str
    player_id: str
    player_name_key: str
    name: str
    team: str
    opponent: str
    game: str
    salary: int
    median_fpts: float
    ceiling_fpts: float
    availability_status: str
    availability_source: str
    is_focus: bool
    is_fade: bool
    positions: tuple[str, ...]


@dataclass(frozen=True)
class DFSLineupCard:
    sport: str
    contest_type: str
    request_mode: str
    salary_used: int
    salary_remaining: int
    median_fpts: float
    ceiling_fpts: float
    floor_fpts: float
    average_confidence: float
    availability_counts: dict[str, int]
    unknown_count: int
    focus_hits: tuple[str, ...]
    fade_hits: tuple[str, ...]
    primary_games: tuple[str, ...]
    game_exposures: dict[str, int]
    slots: tuple[DFSLineupSlot, ...]


@dataclass(frozen=True)
class DFSLineupFamily:
    label: str
    core_players: tuple[str, ...]
    lineup_cards: tuple[DFSLineupCard, ...]


@dataclass(frozen=True)
class ThesisDrivenDFSResult:
    sport: str
    request_mode: str
    contest_type: str
    focus_players: tuple[str, ...]
    fade_players: tuple[str, ...]
    stack_targets: tuple[str, ...]
    bring_back_targets: tuple[str, ...]
    one_off_targets: tuple[str, ...]
    avoid_chalk: tuple[str, ...]
    max_players_per_game: int | None
    preferred_salary_shape: str | None
    build_reasons: tuple[str, ...]
    game_boosts: dict[str, float]
    lineups: tuple[DraftKingsLineup, ...]
    lineup_cards: tuple[DFSLineupCard, ...]
    best_overall_lineup: DFSLineupCard | None = None
    best_value_lineup: DFSLineupCard | None = None
    lineup_families: tuple[DFSLineupFamily, ...] = ()


DEFAULT_CONTEST_CONFIGS: dict[str, ContestBuildConfig] = {
    "cash": ContestBuildConfig(
        request_mode="cash",
        contest_type="cash",
        max_ranked_theses=2,
        max_candidates=None,
        limit=10,
        include_leans=False,
        allow_candidate_fades=False,
        objective_noise_scale=0.0,
        max_exposure=0.34,
        cash_quality_floor=0.92,
    ),
    "head_to_head": ContestBuildConfig(
        request_mode="head_to_head",
        contest_type="cash",
        max_ranked_theses=2,
        max_candidates=None,
        limit=10,
        include_leans=False,
        allow_candidate_fades=False,
        objective_noise_scale=0.0,
        max_exposure=0.34,
        cash_quality_floor=0.92,
    ),
    "single_entry_gpp": ContestBuildConfig(
        request_mode="single_entry_gpp",
        contest_type="single_entry_gpp",
        max_ranked_theses=3,
        max_candidates=None,
        limit=8,
        include_leans=True,
        allow_candidate_fades=True,
        objective_noise_scale=0.015,
        max_exposure=0.5,
        cash_quality_floor=None,
    ),
    "large_field_gpp": ContestBuildConfig(
        request_mode="large_field_gpp",
        contest_type="large_field_gpp",
        max_ranked_theses=4,
        max_candidates=None,
        limit=10,
        include_leans=True,
        allow_candidate_fades=True,
        objective_noise_scale=0.03,
        max_exposure=0.5,
        cash_quality_floor=None,
    ),
    "tournament": ContestBuildConfig(
        request_mode="tournament",
        contest_type="large_field_gpp",
        max_ranked_theses=4,
        max_candidates=None,
        limit=10,
        include_leans=True,
        allow_candidate_fades=True,
        objective_noise_scale=0.03,
        max_exposure=0.5,
        cash_quality_floor=None,
    ),
}

NBA_CLASSIC_SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
MLB_CLASSIC_SLOTS = ("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF")


def extract_nba_thesis_signals(
    result: dict[str, Any],
    *,
    contest_type: str = "cash",
) -> tuple[set[str], set[str], tuple[str, ...]]:
    config = DEFAULT_CONTEST_CONFIGS.get(contest_type, DEFAULT_CONTEST_CONFIGS["cash"])
    focus_players: list[str] = []
    fade_players: list[str] = []
    build_reasons: list[str] = []

    thesis_lookup: dict[str, dict[str, Any]] = {}
    for bucket_name in ("theses", "intuition_theses"):
        for thesis in result.get(bucket_name) or []:
            thesis_id = str(thesis.get("thesis_id") or "").strip()
            if thesis_id:
                thesis_lookup[thesis_id] = thesis

    candidate_lookup = {
        str(item.get("thesis_id") or "").strip(): item
        for item in (result.get("thesis_candidates") or [])
        if str(item.get("thesis_id") or "").strip()
    }

    ranked_items = list((result.get("thesis_judgment") or {}).get("ranked_theses") or [])
    if ranked_items:
        for item in ranked_items:
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict == "pass":
                continue
            if verdict in {"lean", "conditional"} and not config.include_leans:
                continue
            thesis_id = str(item.get("thesis_id") or "").strip()
            if not thesis_id:
                continue
            thesis = thesis_lookup.get(thesis_id, {})
            candidate_bundle = candidate_lookup.get(thesis_id, {})
            _merge_focus_players(focus_players, thesis.get("focus_players") or [])
            _merge_focus_players(focus_players, _candidate_leg_players(item.get("best_candidate") or {}))
            _merge_focus_players(focus_players, _candidate_leg_players(candidate_bundle.get("best_candidate") or {}))
            _merge_focus_players(fade_players, thesis.get("fade_players") or [])
            if config.allow_candidate_fades:
                _merge_focus_players(fade_players, _negative_candidate_players(candidate_bundle))
            summary = str(item.get("summary") or thesis.get("summary") or "").strip()
            if summary:
                build_reasons.append(summary)
            if len(build_reasons) >= config.max_ranked_theses:
                break
    else:
        theses = [item for item in (result.get("theses") or []) if item.get("sport") == "nba"]
        ranked = sorted(
            theses,
            key=lambda item: float(item.get("verified_confidence", item.get("confidence", 0.0))),
            reverse=True,
        )
        for thesis in ranked[: config.max_ranked_theses]:
            _merge_focus_players(focus_players, thesis.get("focus_players") or [])
            _merge_focus_players(fade_players, thesis.get("fade_players") or [])
            summary = str(thesis.get("summary") or "").strip()
            if summary:
                build_reasons.append(summary)

    return set(focus_players), set(fade_players), tuple(build_reasons[: config.max_ranked_theses])


def extract_mlb_thesis_signals(
    result: dict[str, Any],
    *,
    contest_type: str = "cash",
) -> tuple[set[str], set[str], tuple[str, ...]]:
    config = DEFAULT_CONTEST_CONFIGS.get(contest_type, DEFAULT_CONTEST_CONFIGS["cash"])
    focus_players: list[str] = []
    fade_players: list[str] = []
    build_reasons: list[str] = []
    thesis_lookup: dict[str, dict[str, Any]] = {}
    for bucket_name in ("theses", "intuition_theses"):
        for thesis in result.get(bucket_name) or []:
            if thesis.get("sport") != "mlb":
                continue
            thesis_id = str(thesis.get("thesis_id") or "").strip()
            if thesis_id:
                thesis_lookup[thesis_id] = thesis
    candidate_lookup = {
        str(item.get("thesis_id") or "").strip(): item
        for item in (result.get("thesis_candidates") or [])
        if str(item.get("thesis_id") or "").strip()
    }
    ranked_items = list((result.get("thesis_judgment") or {}).get("ranked_theses") or [])
    if ranked_items:
        for item in ranked_items:
            thesis_id = str(item.get("thesis_id") or "").strip()
            thesis = thesis_lookup.get(thesis_id)
            if not thesis:
                continue
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict == "pass":
                continue
            if verdict in {"lean", "conditional"} and not config.include_leans:
                continue
            candidate_bundle = candidate_lookup.get(thesis_id, {})
            _merge_focus_players(focus_players, thesis.get("focus_players") or [])
            _merge_focus_players(focus_players, _candidate_leg_players(item.get("best_candidate") or {}))
            _merge_focus_players(focus_players, _candidate_leg_players(candidate_bundle.get("best_candidate") or {}))
            _merge_focus_players(fade_players, thesis.get("fade_players") or [])
            if config.allow_candidate_fades:
                _merge_focus_players(fade_players, _negative_candidate_players(candidate_bundle))
            summary = str(item.get("summary") or thesis.get("summary") or "").strip()
            if summary:
                build_reasons.append(summary)
            if len(build_reasons) >= config.max_ranked_theses:
                break
    else:
        theses = [item for item in (result.get("theses") or []) if item.get("sport") == "mlb"]
        ranked = sorted(theses, key=lambda item: float(item.get("verified_confidence", item.get("confidence", 0.0))), reverse=True)
        for thesis in ranked[: config.max_ranked_theses]:
            _merge_focus_players(focus_players, thesis.get("focus_players") or [])
            _merge_focus_players(fade_players, thesis.get("fade_players") or [])
            summary = str(thesis.get("summary") or "").strip()
            if summary:
                build_reasons.append(summary)
    return set(focus_players), set(fade_players), tuple(build_reasons[: config.max_ranked_theses])


def extract_nba_environment_scores(
    result: dict[str, Any],
    *,
    contest_type: str = "cash",
) -> dict[str, float]:
    return _extract_environment_scores(result, sport="nba", contest_type=contest_type)


def extract_mlb_environment_scores(
    result: dict[str, Any],
    *,
    contest_type: str = "cash",
) -> dict[str, float]:
    return _extract_environment_scores(result, sport="mlb", contest_type=contest_type)


def _extract_environment_scores(
    result: dict[str, Any],
    *,
    sport: str,
    contest_type: str = "cash",
) -> dict[str, float]:
    config = DEFAULT_CONTEST_CONFIGS.get(contest_type, DEFAULT_CONTEST_CONFIGS["cash"])
    scores: dict[str, float] = {}
    thesis_lookup: dict[str, dict[str, Any]] = {}
    for bucket_name in ("theses", "intuition_theses"):
        for thesis in result.get(bucket_name) or []:
            if thesis.get("sport") != sport:
                continue
            thesis_id = str(thesis.get("thesis_id") or "").strip()
            if thesis_id:
                thesis_lookup[thesis_id] = thesis
    ranked_items = list((result.get("thesis_judgment") or {}).get("ranked_theses") or [])
    if ranked_items:
        for item in ranked_items:
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict == "pass":
                continue
            if verdict in {"lean", "conditional"} and not config.include_leans:
                continue
            thesis = thesis_lookup.get(str(item.get("thesis_id") or "").strip(), {})
            games = tuple(thesis.get("games") or item.get("games") or ())
            if not games:
                continue
            confidence = float(item.get("confidence", thesis.get("verified_confidence", thesis.get("confidence", 0.0))) or 0.0)
            verdict_weight = 1.0 if verdict == "back" else 0.5 if verdict == "conditional" else 0.65
            for game in games:
                game_key = str(game or "").strip()
                if not game_key:
                    continue
                scores[game_key] = scores.get(game_key, 0.0) + (confidence * verdict_weight)
    else:
        for thesis in thesis_lookup.values():
            confidence = float(thesis.get("verified_confidence", thesis.get("confidence", 0.0)) or 0.0)
            for game in thesis.get("games") or []:
                game_key = str(game or "").strip()
                if game_key:
                    scores[game_key] = scores.get(game_key, 0.0) + confidence
    return scores


def extract_nba_dfs_guidance(
    result: dict[str, Any],
    *,
    contest_type: str = "cash",
) -> dict[str, Any]:
    return _extract_dfs_guidance(result, contest_type=contest_type)


def extract_mlb_dfs_guidance(
    result: dict[str, Any],
    *,
    contest_type: str = "cash",
) -> dict[str, Any]:
    return _extract_dfs_guidance(result, contest_type=contest_type)


def _extract_dfs_guidance(
    result: dict[str, Any],
    *,
    contest_type: str = "cash",
) -> dict[str, Any]:
    config = DEFAULT_CONTEST_CONFIGS.get(contest_type, DEFAULT_CONTEST_CONFIGS["cash"])
    stack_targets: list[str] = []
    bring_back_targets: list[str] = []
    one_off_targets: list[str] = []
    avoid_chalk: list[str] = []
    max_players_from_game: list[int] = []
    preferred_shapes: list[str] = []

    ranked_items = list((result.get("thesis_judgment") or {}).get("ranked_theses") or [])
    for item in ranked_items:
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict == "pass":
            continue
        if verdict in {"lean", "conditional"} and not config.include_leans:
            continue
        guidance = item.get("dfs_guidance") or {}
        _merge_focus_players(stack_targets, guidance.get("stack_targets") or [])
        _merge_focus_players(bring_back_targets, guidance.get("bring_back_targets") or [])
        _merge_focus_players(one_off_targets, guidance.get("one_off_targets") or [])
        _merge_focus_players(avoid_chalk, guidance.get("avoid_chalk") or [])
        max_game = guidance.get("max_players_from_game")
        if max_game is not None:
            try:
                max_players_from_game.append(int(max_game))
            except (TypeError, ValueError):
                pass
        shape = str(guidance.get("preferred_salary_shape") or "").strip().lower()
        if shape in {"balanced", "stars_and_scrubs", "leave_salary"}:
            preferred_shapes.append(shape)

    preferred_salary_shape = preferred_shapes[0] if preferred_shapes else None
    max_players_per_game = None
    if max_players_from_game:
        if config.request_mode == "head_to_head":
            max_players_per_game = min(max_players_from_game)
        else:
            max_players_per_game = max(max_players_from_game)

    return {
        "stack_targets": tuple(sorted(dict.fromkeys(stack_targets))),
        "bring_back_targets": tuple(sorted(dict.fromkeys(bring_back_targets))),
        "one_off_targets": tuple(sorted(dict.fromkeys(one_off_targets))),
        "avoid_chalk": tuple(sorted(dict.fromkeys(avoid_chalk))),
        "max_players_per_game": max_players_per_game,
        "preferred_salary_shape": preferred_salary_shape,
    }


def build_nba_contest_lineups(
    *,
    date_str: str,
    salary_csv_path: str,
    contest_type: str = "cash",
    oracle_result: dict[str, Any] | None = None,
) -> ThesisDrivenDFSResult:
    config = DEFAULT_CONTEST_CONFIGS.get(contest_type, DEFAULT_CONTEST_CONFIGS["cash"])
    slate = parse_draftkings_salary_csv(salary_csv_path, sport="nba")
    projections = attach_salary_metadata(slate, build_nba_dk_projections(date_str, slate))
    if oracle_result is None:
        analysis = build_nba_dfs_analysis(projections, contest_type=config.contest_type)
        focus_players = set(analysis.focus_players)
        fade_players = set(analysis.fade_players)
        build_reasons = analysis.build_reasons
        dfs_guidance = {
            "stack_targets": analysis.stack_targets,
            "bring_back_targets": analysis.bring_back_targets,
            "one_off_targets": analysis.one_off_targets,
            "avoid_chalk": analysis.avoid_chalk,
            "max_players_per_game": analysis.max_players_per_game,
            "preferred_salary_shape": analysis.preferred_salary_shape,
        }
        game_boosts = analysis.game_boosts
    else:
        oracle_result = _filter_oracle_result_for_dfs_slate(slate, oracle_result)
        focus_players, fade_players, build_reasons = extract_nba_thesis_signals(
            oracle_result,
            contest_type=config.request_mode,
        )
        dfs_guidance = extract_nba_dfs_guidance(oracle_result, contest_type=config.request_mode)
        game_boosts = extract_nba_environment_scores(oracle_result, contest_type=config.request_mode)
        focus_players, fade_players, dfs_guidance, game_boosts = _scope_dfs_inputs_to_player_pool(
            projections,
            focus_players=focus_players,
            fade_players=fade_players,
            dfs_guidance=dfs_guidance,
            game_boosts=game_boosts,
        )
    fade_union = set(fade_players) | set(dfs_guidance["avoid_chalk"])
    lineups = _build_nba_hedged_portfolio_lineups(
        projections=projections,
        salary_cap=slate.salary_cap,
        max_candidates=config.max_candidates,
        limit=config.limit,
        contest_type=config.contest_type,
        focus_players=focus_players,
        fade_players=fade_union,
        game_boosts=game_boosts,
        dfs_guidance=dfs_guidance,
        objective_noise_scale=config.objective_noise_scale,
        max_exposure=config.max_exposure,
    )
    best_overall_lineup = _build_best_overall_nba_lineup_card(
        projections=projections,
        request_mode=config.request_mode,
        salary_cap=slate.salary_cap,
        focus_players=focus_players,
        fade_players=fade_union,
        game_boosts=game_boosts,
        dfs_guidance=dfs_guidance,
    )
    best_value_lineup = _build_best_value_nba_lineup_card(
        projections=projections,
        request_mode=config.request_mode,
        salary_cap=slate.salary_cap,
        focus_players=focus_players,
        fade_players=fade_union,
        game_boosts=game_boosts,
        dfs_guidance=dfs_guidance,
    )
    if config.contest_type == "cash":
        lineups = _rerank_cash_lineups_robustly(lineups, game_boosts=game_boosts)
        lineups = _apply_cash_quality_floor(
            lineups,
            game_boosts=game_boosts,
            quality_floor=config.cash_quality_floor,
        )
    lineup_cards = tuple(
        _build_lineup_card(
            lineup,
            sport="nba",
            slots=NBA_CLASSIC_SLOTS,
            request_mode=config.request_mode,
            contest_type=config.contest_type,
            salary_cap=slate.salary_cap,
            focus_players=focus_players,
            fade_players=fade_union,
        )
        for lineup in lineups
    )
    lineup_families = _build_nba_lineup_families(
        projections=projections,
        request_mode=config.request_mode,
        contest_type=config.contest_type,
        salary_cap=slate.salary_cap,
        limit=config.limit,
        focus_players=focus_players,
        fade_players=fade_union,
        game_boosts=game_boosts,
        dfs_guidance=dfs_guidance,
        objective_noise_scale=config.objective_noise_scale,
    )
    return ThesisDrivenDFSResult(
        sport="nba",
        request_mode=config.request_mode,
        contest_type=config.contest_type,
        focus_players=tuple(sorted(focus_players)),
        fade_players=tuple(sorted(fade_union)),
        stack_targets=tuple(sorted(dfs_guidance["stack_targets"])),
        bring_back_targets=tuple(sorted(dfs_guidance["bring_back_targets"])),
        one_off_targets=tuple(sorted(dfs_guidance["one_off_targets"])),
        avoid_chalk=tuple(sorted(dfs_guidance["avoid_chalk"])),
        max_players_per_game=dfs_guidance["max_players_per_game"],
        preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
        build_reasons=build_reasons,
        game_boosts=dict(sorted(game_boosts.items(), key=lambda item: item[1], reverse=True)),
        lineups=tuple(lineups),
        lineup_cards=lineup_cards,
        best_overall_lineup=best_overall_lineup,
        best_value_lineup=best_value_lineup,
        lineup_families=lineup_families,
    )


def build_mlb_contest_lineups(
    *,
    date_str: str,
    salary_csv_path: str,
    contest_type: str = "cash",
    oracle_result: dict[str, Any] | None = None,
) -> ThesisDrivenDFSResult:
    config = DEFAULT_CONTEST_CONFIGS.get(contest_type, DEFAULT_CONTEST_CONFIGS["cash"])
    slate = parse_draftkings_salary_csv(salary_csv_path, sport="mlb")
    projections = attach_mlb_salary_metadata(slate, build_mlb_dk_projections(date_str, slate))
    if oracle_result is None:
        analysis = build_mlb_dfs_analysis(projections, contest_type=config.contest_type)
        focus_players = set(analysis.focus_players)
        fade_players = set(analysis.fade_players)
        build_reasons = analysis.build_reasons
        dfs_guidance = {
            "stack_targets": analysis.stack_targets,
            "bring_back_targets": analysis.bring_back_targets,
            "one_off_targets": analysis.one_off_targets,
            "avoid_chalk": analysis.avoid_chalk,
            "max_players_per_game": analysis.max_players_per_game,
            "preferred_salary_shape": analysis.preferred_salary_shape,
        }
        game_boosts = analysis.game_boosts
    else:
        oracle_result = _filter_oracle_result_for_dfs_slate(slate, oracle_result)
        focus_players, fade_players, build_reasons = extract_mlb_thesis_signals(oracle_result, contest_type=config.request_mode)
        dfs_guidance = extract_mlb_dfs_guidance(oracle_result, contest_type=config.request_mode)
        game_boosts = extract_mlb_environment_scores(oracle_result, contest_type=config.request_mode)
        focus_players, fade_players, dfs_guidance, game_boosts = _scope_dfs_inputs_to_player_pool(
            projections,
            focus_players=focus_players,
            fade_players=fade_players,
            dfs_guidance=dfs_guidance,
            game_boosts=game_boosts,
        )
    lineups = optimize_mlb_classic_lineups(
        projections,
        salary_cap=slate.salary_cap,
        max_candidates=config.max_candidates,
        limit=config.limit,
        contest_type=config.contest_type,
        focus_players=focus_players,
        fade_players=set(fade_players) | set(dfs_guidance["avoid_chalk"]),
        game_boosts=game_boosts,
        stack_targets=set(dfs_guidance["stack_targets"]),
        bring_back_targets=set(dfs_guidance["bring_back_targets"]),
        one_off_targets=set(dfs_guidance["one_off_targets"]),
        max_players_per_game=dfs_guidance["max_players_per_game"],
        preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
        objective_noise_scale=config.objective_noise_scale,
        max_exposure=config.max_exposure,
    )
    best_overall_lineup = _build_best_overall_mlb_lineup_card(
        projections=projections,
        request_mode=config.request_mode,
        salary_cap=slate.salary_cap,
        focus_players=focus_players,
        fade_players=set(fade_players) | set(dfs_guidance["avoid_chalk"]),
        game_boosts=game_boosts,
        dfs_guidance=dfs_guidance,
    )
    best_value_lineup = _build_best_value_mlb_lineup_card(
        projections=projections,
        request_mode=config.request_mode,
        salary_cap=slate.salary_cap,
        focus_players=focus_players,
        fade_players=set(fade_players) | set(dfs_guidance["avoid_chalk"]),
        game_boosts=game_boosts,
        dfs_guidance=dfs_guidance,
    )
    if config.contest_type == "cash":
        lineups = _rerank_cash_lineups_robustly(lineups, game_boosts=game_boosts)
        lineups = _apply_cash_quality_floor(
            lineups,
            game_boosts=game_boosts,
            quality_floor=config.cash_quality_floor,
        )
    lineup_cards = tuple(
        _build_lineup_card(
            lineup,
            sport="mlb",
            slots=MLB_CLASSIC_SLOTS,
            request_mode=config.request_mode,
            contest_type=config.contest_type,
            salary_cap=slate.salary_cap,
            focus_players=focus_players,
            fade_players=set(fade_players) | set(dfs_guidance["avoid_chalk"]),
        )
        for lineup in lineups
    )
    return ThesisDrivenDFSResult(
        sport="mlb",
        request_mode=config.request_mode,
        contest_type=config.contest_type,
        focus_players=tuple(sorted(focus_players)),
        fade_players=tuple(sorted(set(fade_players) | set(dfs_guidance["avoid_chalk"]))),
        stack_targets=tuple(sorted(dfs_guidance["stack_targets"])),
        bring_back_targets=tuple(sorted(dfs_guidance["bring_back_targets"])),
        one_off_targets=tuple(sorted(dfs_guidance["one_off_targets"])),
        avoid_chalk=tuple(sorted(dfs_guidance["avoid_chalk"])),
        max_players_per_game=dfs_guidance["max_players_per_game"],
        preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
        build_reasons=build_reasons,
        game_boosts=dict(sorted(game_boosts.items(), key=lambda item: item[1], reverse=True)),
        lineups=tuple(lineups),
        lineup_cards=lineup_cards,
        best_overall_lineup=best_overall_lineup,
        best_value_lineup=best_value_lineup,
    )


def _build_nba_lineup_families(
    *,
    projections: list[DraftKingsNBAProjection],
    request_mode: str,
    contest_type: str,
    salary_cap: int,
    limit: int,
    focus_players: set[str],
    fade_players: set[str],
    game_boosts: dict[str, float],
    dfs_guidance: dict[str, Any],
    objective_noise_scale: float,
) -> tuple[DFSLineupFamily, ...]:
    if contest_type != "large_field_gpp":
        return ()

    projection_lookup = {dfs_name_key(player.name): player for player in projections if dfs_name_key(player.name)}
    ranked = sorted(
        projections,
        key=lambda player: (
            player.median_fpts / max(player.salary, 1),
            player.ceiling_fpts,
            player.median_fpts,
        ),
        reverse=True,
    )[:12]
    pair_candidates: list[tuple[float, tuple[str, str]]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for idx, left in enumerate(ranked):
        for right in ranked[idx + 1 :]:
            pair = tuple(sorted((left.name, right.name)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            left_key = dfs_name_key(left.name)
            right_key = dfs_name_key(right.name)
            left_projection = projection_lookup.get(left_key)
            right_projection = projection_lookup.get(right_key)
            correlation_bonus = 0.0
            if left_projection and right_projection and left_projection.game and left_projection.game == right_projection.game:
                game_boost = float(game_boosts.get(left_projection.game, 0.0))
                correlation_bonus += 1.5 + (game_boost * 1.2)
                if left_projection.team != right_projection.team:
                    correlation_bonus += 0.8
            score = (
                (left.median_fpts / max(left.salary, 1)) * 1000.0
                + (right.median_fpts / max(right.salary, 1)) * 1000.0
                + ((left.ceiling_fpts + right.ceiling_fpts) * 0.08)
                + correlation_bonus
            )
            pair_candidates.append((score, pair))
    pair_candidates.sort(reverse=True)

    families: list[DFSLineupFamily] = []
    used_pairs: set[tuple[str, str]] = set()
    for _score, pair in pair_candidates:
        if pair in used_pairs:
            continue
        family_lineups = optimize_nba_classic_lineups(
            projections,
            salary_cap=salary_cap,
            limit=min(3, max(2, limit // 3)),
            contest_type=contest_type,
            focus_players=set(focus_players) | set(pair),
            fade_players=fade_players,
            game_boosts=game_boosts,
            stack_targets=set(dfs_guidance["stack_targets"]),
            bring_back_targets=set(dfs_guidance["bring_back_targets"]),
            one_off_targets=set(dfs_guidance["one_off_targets"]),
            max_players_per_game=dfs_guidance["max_players_per_game"],
            preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
            objective_noise_scale=max(objective_noise_scale, 0.02),
            max_exposure=1.0,
            locked_players=set(pair),
        )
        if not family_lineups:
            continue
        cards = tuple(
            _build_lineup_card(
                lineup,
                sport="nba",
                slots=NBA_CLASSIC_SLOTS,
                request_mode=request_mode,
                contest_type=contest_type,
                salary_cap=salary_cap,
                focus_players=set(focus_players) | set(pair),
                fade_players=fade_players,
            )
            for lineup in family_lineups
        )
        families.append(
            DFSLineupFamily(
                label=f"{pair[0]} + {pair[1]}",
                core_players=pair,
                lineup_cards=cards,
            )
        )
        used_pairs.add(pair)
        if len(families) >= 4:
            break
    return tuple(families)


def _build_best_overall_nba_lineup_card(
    *,
    projections: list[DraftKingsNBAProjection],
    request_mode: str,
    salary_cap: int,
    focus_players: set[str],
    fade_players: set[str],
    game_boosts: dict[str, float],
    dfs_guidance: dict[str, Any],
) -> DFSLineupCard | None:
    weighted = [
        DraftKingsNBAProjection(
            player_id=player.player_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            salary=player.salary,
            positions=player.positions,
            roster_positions=player.roster_positions,
            game=player.game,
            median_fpts=player.median_fpts,
            ceiling_fpts=0.0,
            floor_fpts=0.0,
            volatility=0.0,
            projection_confidence=0.0,
            minutes=player.minutes,
            points=player.points,
            rebounds=player.rebounds,
            assists=player.assists,
            availability_status=player.availability_status,
            availability_source=player.availability_source,
        )
        for player in projections
    ]
    best_lineups = optimize_nba_classic_lineups(
        weighted,
        salary_cap=salary_cap,
        max_candidates=None,
        limit=1,
        contest_type="cash",
        focus_players=set(),
        fade_players=set(),
        game_boosts={},
        stack_targets=set(),
        bring_back_targets=set(),
        one_off_targets=set(),
        max_players_per_game=None,
        preferred_salary_shape=None,
        objective_noise_scale=0.0,
        max_exposure=1.0,
    )
    if not best_lineups:
        return None
    original_lookup = {player.player_id: player for player in projections}
    players = tuple(original_lookup.get(player.player_id, player) for player in best_lineups[0].players)
    rebuilt = DraftKingsLineup(
        players=players,
        salary_used=sum(player.salary for player in players),
        median_fpts=sum(player.median_fpts for player in players),
        ceiling_fpts=sum(player.ceiling_fpts for player in players),
        floor_fpts=sum(player.floor_fpts for player in players),
        average_confidence=sum(player.projection_confidence for player in players) / max(len(players), 1),
        unknown_count=sum(1 for player in players if player.availability_status == "unknown"),
    )
    return _build_lineup_card(
        rebuilt,
        sport="nba",
        slots=NBA_CLASSIC_SLOTS,
        request_mode=request_mode,
        contest_type="best_overall_points",
        salary_cap=salary_cap,
        focus_players=focus_players,
        fade_players=fade_players,
    )


def _build_best_value_nba_lineup_card(
    *,
    projections: list[DraftKingsNBAProjection],
    request_mode: str,
    salary_cap: int,
    focus_players: set[str],
    fade_players: set[str],
    game_boosts: dict[str, float],
    dfs_guidance: dict[str, Any],
) -> DFSLineupCard | None:
    weighted = [
        DraftKingsNBAProjection(
            player_id=player.player_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            salary=player.salary,
            positions=player.positions,
            roster_positions=player.roster_positions,
            game=player.game,
            median_fpts=(player.median_fpts / max(player.salary, 1)) * 1000.0,
            ceiling_fpts=(player.ceiling_fpts / max(player.salary, 1)) * 1000.0,
            floor_fpts=(player.floor_fpts / max(player.salary, 1)) * 1000.0,
            volatility=player.volatility,
            projection_confidence=player.projection_confidence,
            minutes=player.minutes,
            points=player.points,
            rebounds=player.rebounds,
            assists=player.assists,
            availability_status=player.availability_status,
            availability_source=player.availability_source,
        )
        for player in projections
    ]
    lineup = optimize_nba_classic_lineups(
        weighted,
        salary_cap=salary_cap,
        max_candidates=None,
        limit=1,
        contest_type="cash",
        focus_players=focus_players,
        fade_players=fade_players,
        game_boosts=game_boosts,
        stack_targets=set(dfs_guidance["stack_targets"]),
        bring_back_targets=set(dfs_guidance["bring_back_targets"]),
        one_off_targets=set(dfs_guidance["one_off_targets"]),
        max_players_per_game=dfs_guidance["max_players_per_game"],
        preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
        objective_noise_scale=0.0,
        max_exposure=1.0,
    )
    if not lineup:
        return None
    original_lookup = {player.player_id: player for player in projections}
    players = tuple(original_lookup.get(player.player_id, player) for player in lineup[0].players)
    rebuilt = DraftKingsLineup(
        players=players,
        salary_used=sum(player.salary for player in players),
        median_fpts=sum(player.median_fpts for player in players),
        ceiling_fpts=sum(player.ceiling_fpts for player in players),
        floor_fpts=sum(player.floor_fpts for player in players),
        average_confidence=sum(player.projection_confidence for player in players) / max(len(players), 1),
        unknown_count=sum(1 for player in players if player.availability_status == "unknown"),
    )
    return _build_lineup_card(
        rebuilt,
        sport="nba",
        slots=NBA_CLASSIC_SLOTS,
        request_mode=request_mode,
        contest_type="best_value",
        salary_cap=salary_cap,
        focus_players=focus_players,
        fade_players=fade_players,
    )


def _build_nba_hedged_portfolio_lineups(
    *,
    projections: list[DraftKingsNBAProjection],
    salary_cap: int,
    max_candidates: int | None,
    limit: int,
    contest_type: str,
    focus_players: set[str],
    fade_players: set[str],
    game_boosts: dict[str, float],
    dfs_guidance: dict[str, Any],
    objective_noise_scale: float,
    max_exposure: float | None,
) -> list[DraftKingsLineup]:
    anchor_candidates = _select_nba_anchor_candidates(
        projections,
        contest_type=contest_type,
        focus_players=focus_players,
    )
    if not anchor_candidates:
        return optimize_nba_classic_lineups(
            projections,
            salary_cap=salary_cap,
            max_candidates=max_candidates,
            limit=limit,
            contest_type=contest_type,
            focus_players=focus_players,
            fade_players=fade_players,
            game_boosts=game_boosts,
            stack_targets=set(dfs_guidance["stack_targets"]),
            bring_back_targets=set(dfs_guidance["bring_back_targets"]),
            one_off_targets=set(dfs_guidance["one_off_targets"]),
            max_players_per_game=dfs_guidance["max_players_per_game"],
            preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
            objective_noise_scale=objective_noise_scale,
            max_exposure=max_exposure,
        )

    lineups: list[DraftKingsLineup] = []
    seen_sets: set[tuple[str, ...]] = set()
    exposure_counts: dict[str, int] = {}
    max_player_appearances: int | None = None
    if max_exposure is not None:
        bounded = max(0.0, min(1.0, float(max_exposure)))
        max_player_appearances = max(1, int(bounded * max(1, int(limit))))
    normalized_contest = str(contest_type or "").strip().lower()
    if normalized_contest == "cash":
        primary_anchors = anchor_candidates[: min(len(anchor_candidates), limit)]
        family_limit = 1
    else:
        anchor_count = min(len(anchor_candidates), max(4, min(limit, 6)))
        if anchor_count <= 0:
            anchor_count = 1
        primary_anchors = anchor_candidates[:anchor_count]
        family_limit = max(1, limit // anchor_count)

    for anchor in primary_anchors:
        generated = 0
        attempts = 0
        while generated < family_limit and len(lineups) < limit and attempts < max(4, family_limit * 3):
            attempts += 1
            excluded_players = {
                name
                for name, count in exposure_counts.items()
                if max_player_appearances is not None and count >= max_player_appearances and dfs_name_key(name) != dfs_name_key(anchor)
            }
            family_lineups = optimize_nba_classic_lineups(
                projections,
                salary_cap=salary_cap,
                max_candidates=max_candidates,
                limit=1,
                contest_type=contest_type,
                focus_players=set(focus_players) | {anchor},
                fade_players=fade_players,
                game_boosts=game_boosts,
                stack_targets=set(dfs_guidance["stack_targets"]),
                bring_back_targets=set(dfs_guidance["bring_back_targets"]),
                one_off_targets=set(dfs_guidance["one_off_targets"]),
                max_players_per_game=dfs_guidance["max_players_per_game"],
                preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
                objective_noise_scale=max(objective_noise_scale, 0.01),
                max_exposure=1.0,
                locked_players={anchor},
                excluded_players=excluded_players,
            )
            if not family_lineups:
                break
            lineup = family_lineups[0]
            lineup_key = tuple(sorted(player.player_id or player.name for player in lineup.players))
            if lineup_key in seen_sets:
                break
            seen_sets.add(lineup_key)
            lineups.append(lineup)
            generated += 1
            for player in lineup.players:
                exposure_counts[player.name] = exposure_counts.get(player.name, 0) + 1
            if len(lineups) >= limit:
                return lineups[:limit]

    if len(lineups) < limit:
        attempts = 0
        while len(lineups) < limit and attempts < max(8, limit * 3):
            attempts += 1
            excluded_players = {
                name
                for name, count in exposure_counts.items()
                if max_player_appearances is not None and count >= max_player_appearances
            }
            fallback_lineups = optimize_nba_classic_lineups(
                projections,
                salary_cap=salary_cap,
                max_candidates=max_candidates,
                limit=1,
                contest_type=contest_type,
                focus_players=focus_players,
                fade_players=fade_players,
                game_boosts=game_boosts,
                stack_targets=set(dfs_guidance["stack_targets"]),
                bring_back_targets=set(dfs_guidance["bring_back_targets"]),
                one_off_targets=set(dfs_guidance["one_off_targets"]),
                max_players_per_game=dfs_guidance["max_players_per_game"],
                preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
                objective_noise_scale=objective_noise_scale,
                max_exposure=1.0,
                excluded_players=excluded_players,
            )
            if not fallback_lineups:
                break
            lineup = fallback_lineups[0]
            lineup_key = tuple(sorted(player.player_id or player.name for player in lineup.players))
            if lineup_key in seen_sets:
                break
            seen_sets.add(lineup_key)
            lineups.append(lineup)
            for player in lineup.players:
                exposure_counts[player.name] = exposure_counts.get(player.name, 0) + 1
    return lineups[:limit]


def _build_best_overall_mlb_lineup_card(
    *,
    projections: list[DraftKingsMLBProjection],
    request_mode: str,
    salary_cap: int,
    focus_players: set[str],
    fade_players: set[str],
    game_boosts: dict[str, float],
    dfs_guidance: dict[str, Any],
) -> DFSLineupCard | None:
    weighted = [
        DraftKingsMLBProjection(
            player_id=player.player_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            salary=player.salary,
            positions=player.positions,
            roster_positions=player.roster_positions,
            game=player.game,
            median_fpts=player.median_fpts,
            ceiling_fpts=0.0,
            floor_fpts=0.0,
            volatility=0.0,
            projection_confidence=0.0,
            plate_appearances=player.plate_appearances,
            innings_pitched=player.innings_pitched,
            hits=player.hits,
            home_runs=player.home_runs,
            stolen_bases=player.stolen_bases,
            strikeouts=player.strikeouts,
            runs_allowed=player.runs_allowed,
            availability_status=player.availability_status,
            availability_source=player.availability_source,
        )
        for player in projections
    ]
    best_lineups = optimize_mlb_classic_lineups(
        weighted,
        salary_cap=salary_cap,
        max_candidates=None,
        limit=1,
        contest_type="cash",
        focus_players=set(),
        fade_players=set(),
        game_boosts={},
        stack_targets=set(),
        bring_back_targets=set(),
        one_off_targets=set(),
        max_players_per_game=None,
        preferred_salary_shape=None,
        objective_noise_scale=0.0,
        max_exposure=1.0,
    )
    if not best_lineups:
        return None
    original_lookup = {player.player_id: player for player in projections}
    players = tuple(original_lookup.get(player.player_id, player) for player in best_lineups[0].players)
    rebuilt = DraftKingsMLBLineup(
        players=players,
        salary_used=sum(player.salary for player in players),
        median_fpts=sum(player.median_fpts for player in players),
        ceiling_fpts=sum(player.ceiling_fpts for player in players),
        floor_fpts=sum(player.floor_fpts for player in players),
        average_confidence=sum(player.projection_confidence for player in players) / max(len(players), 1),
        unknown_count=sum(1 for player in players if player.availability_status == "unknown"),
    )
    return _build_lineup_card(
        rebuilt,
        sport="mlb",
        slots=MLB_CLASSIC_SLOTS,
        request_mode=request_mode,
        contest_type="best_overall_points",
        salary_cap=salary_cap,
        focus_players=focus_players,
        fade_players=fade_players,
    )


def _build_best_value_mlb_lineup_card(
    *,
    projections: list[DraftKingsMLBProjection],
    request_mode: str,
    salary_cap: int,
    focus_players: set[str],
    fade_players: set[str],
    game_boosts: dict[str, float],
    dfs_guidance: dict[str, Any],
) -> DFSLineupCard | None:
    weighted = [
        DraftKingsMLBProjection(
            player_id=player.player_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            salary=player.salary,
            positions=player.positions,
            roster_positions=player.roster_positions,
            game=player.game,
            median_fpts=(player.median_fpts / max(player.salary, 1)) * 1000.0,
            ceiling_fpts=(player.ceiling_fpts / max(player.salary, 1)) * 1000.0,
            floor_fpts=(player.floor_fpts / max(player.salary, 1)) * 1000.0,
            volatility=player.volatility,
            projection_confidence=player.projection_confidence,
            plate_appearances=player.plate_appearances,
            innings_pitched=player.innings_pitched,
            hits=player.hits,
            home_runs=player.home_runs,
            stolen_bases=player.stolen_bases,
            strikeouts=player.strikeouts,
            runs_allowed=player.runs_allowed,
            availability_status=player.availability_status,
            availability_source=player.availability_source,
        )
        for player in projections
    ]
    lineup = optimize_mlb_classic_lineups(
        weighted,
        salary_cap=salary_cap,
        max_candidates=None,
        limit=1,
        contest_type="cash",
        focus_players=focus_players,
        fade_players=fade_players,
        game_boosts=game_boosts,
        stack_targets=set(dfs_guidance["stack_targets"]),
        bring_back_targets=set(dfs_guidance["bring_back_targets"]),
        one_off_targets=set(dfs_guidance["one_off_targets"]),
        max_players_per_game=dfs_guidance["max_players_per_game"],
        preferred_salary_shape=dfs_guidance["preferred_salary_shape"],
        objective_noise_scale=0.0,
        max_exposure=1.0,
    )
    if not lineup:
        return None
    original_lookup = {player.player_id: player for player in projections}
    players = tuple(original_lookup.get(player.player_id, player) for player in lineup[0].players)
    rebuilt = DraftKingsMLBLineup(
        players=players,
        salary_used=sum(player.salary for player in players),
        median_fpts=sum(player.median_fpts for player in players),
        ceiling_fpts=sum(player.ceiling_fpts for player in players),
        floor_fpts=sum(player.floor_fpts for player in players),
        average_confidence=sum(player.projection_confidence for player in players) / max(len(players), 1),
        unknown_count=sum(1 for player in players if player.availability_status == "unknown"),
    )
    return _build_lineup_card(
        rebuilt,
        sport="mlb",
        slots=MLB_CLASSIC_SLOTS,
        request_mode=request_mode,
        contest_type="best_value",
        salary_cap=salary_cap,
        focus_players=focus_players,
        fade_players=fade_players,
    )


def _select_nba_anchor_candidates(
    projections: list[DraftKingsNBAProjection],
    *,
    contest_type: str,
    focus_players: set[str],
) -> tuple[str, ...]:
    focus_keys = {dfs_name_key(name) for name in focus_players if dfs_name_key(name)}
    ranked = sorted(
        [
            (
                player,
                float(player.median_fpts)
                + (float(player.ceiling_fpts) * 0.22)
                + ((float(player.median_fpts) / max(float(player.salary), 1.0)) * 1400.0)
                + (4.0 if dfs_name_key(player.name) in focus_keys else 0.0),
            )
            for player in projections
            if player.availability_status != "out"
            and (player.salary >= 7800 or player.median_fpts >= 38.0)
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    target_count = 6 if contest_type == "cash" else 8
    cutoff_score = ranked[target_count - 1][1] if len(ranked) >= target_count else (ranked[-1][1] if ranked else 0.0)
    threshold_score = max(cutoff_score, (ranked[0][1] - 8.0) if ranked else 0.0)
    if contest_type != "cash":
        anchors: list[str] = []
        for player, score in ranked:
            name = str(player.name or "").strip()
            if not name:
                continue
            if score < threshold_score and len(anchors) >= target_count:
                continue
            anchors.append(name)
            if len(anchors) >= target_count:
                break
        return tuple(anchors)

    def anchor_cluster(player: DraftKingsNBAProjection) -> str:
        positions = set(player.positions)
        if positions & {"PG", "SG"}:
            return "backcourt"
        if positions & {"C"}:
            return "center"
        if positions & {"PF"}:
            return "frontcourt"
        return "wing"

    selected: list[str] = []
    selected_keys: set[str] = set()

    def add_player(player: DraftKingsNBAProjection) -> None:
        key = dfs_name_key(player.name)
        if not key or key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(player.name)

    seeds: dict[str, tuple[DraftKingsNBAProjection, float]] = {}
    for player, score in ranked:
        cluster = anchor_cluster(player)
        seeds.setdefault(cluster, (player, score))
    for cluster_name in ("center", "backcourt", "wing", "frontcourt"):
        seed = seeds.get(cluster_name)
        if seed:
            add_player(seed[0])

    for cluster_name in ("center", "backcourt", "wing", "frontcourt"):
        seed = seeds.get(cluster_name)
        if not seed:
            continue
        seed_player, seed_score = seed
        seed_salary = seed_player.salary
        alternate_candidates: list[tuple[int, float, DraftKingsNBAProjection]] = []
        for player, score in ranked:
            if anchor_cluster(player) != cluster_name:
                continue
            key = dfs_name_key(player.name)
            if not key or key in selected_keys:
                continue
            if abs(player.salary - seed_salary) <= 1200 and score >= (seed_score - 8.5):
                alternate_candidates.append((abs(player.salary - seed_salary), -score, player))
        alternate = min(alternate_candidates, default=None)
        if alternate:
            add_player(alternate[2])
        if len(selected) >= target_count:
            break

    for player, _score in ranked:
        if len(selected) >= target_count:
            break
        add_player(player)
    return tuple(selected[:target_count])


def _cash_lineup_robust_score(
    lineup: DraftKingsLineup | DraftKingsMLBLineup,
    *,
    game_boosts: dict[str, float],
) -> float:
    players = tuple(getattr(lineup, "players", ()) or ())
    games = {str(getattr(player, "game", "") or "").strip() for player in players if str(getattr(player, "game", "") or "").strip()}
    avg_game_boost = sum(float(game_boosts.get(game, 0.0) or 0.0) for game in games) / max(len(games), 1)
    downside_state = (float(lineup.floor_fpts) * 0.85) + (float(lineup.median_fpts) * 0.15)
    base_state = float(lineup.median_fpts)
    upside_state = float(lineup.median_fpts) + ((float(lineup.ceiling_fpts) - float(lineup.median_fpts)) * 0.35)
    robustness = (downside_state * 0.45) + (base_state * 0.4) + (upside_state * 0.15)
    confidence = float(getattr(lineup, "average_confidence", 0.0) or 0.0) * 3.0
    return robustness + (avg_game_boost * 2.5) + confidence


def _rerank_cash_lineups_robustly(
    lineups: list[DraftKingsLineup] | list[DraftKingsMLBLineup],
    *,
    game_boosts: dict[str, float],
):
    return sorted(
        lineups,
        key=lambda lineup: (
            _cash_lineup_robust_score(lineup, game_boosts=game_boosts),
            float(lineup.median_fpts),
            float(lineup.floor_fpts),
            -int(lineup.salary_used),
        ),
        reverse=True,
    )


def _apply_cash_quality_floor(
    lineups: list[DraftKingsLineup] | list[DraftKingsMLBLineup],
    *,
    game_boosts: dict[str, float],
    quality_floor: float | None,
):
    if not lineups or quality_floor is None:
        return lineups
    top_score = _cash_lineup_robust_score(lineups[0], game_boosts=game_boosts)
    threshold = top_score * float(quality_floor)
    filtered = [lineup for lineup in lineups if _cash_lineup_robust_score(lineup, game_boosts=game_boosts) >= threshold]
    return filtered or lineups[:1]


def _merge_focus_players(target: list[str], players: list[str]) -> None:
    seen = {dfs_name_key(name) for name in target}
    for player in players:
        key = dfs_name_key(str(player or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        target.append(str(player))


def _filter_oracle_result_for_dfs_slate(slate, result: dict[str, Any]) -> dict[str, Any]:
    filtered = deepcopy(result or {})
    slate_sport = str(slate.sport or "").strip().lower()
    slate_games = {str(player.game or "").strip() for player in slate.players if str(player.game or "").strip()}
    slate_player_keys = {dfs_name_key(player.name) for player in slate.players if dfs_name_key(player.name)}

    def leg_allowed(leg: dict[str, Any]) -> bool:
        if not isinstance(leg, dict):
            return False
        leg_sport = str(leg.get("sport") or slate_sport).strip().lower()
        if leg_sport and leg_sport != slate_sport:
            return False
        game = str(leg.get("game") or "").strip()
        if slate_games and game and game not in slate_games:
            return False
        player = str(leg.get("player") or leg.get("player_name") or leg.get("subject") or "").strip()
        if slate_player_keys and player and dfs_name_key(player) not in slate_player_keys:
            return False
        return True

    filtered["theses"] = [
        thesis
        for thesis in (filtered.get("theses") or [])
        if str(thesis.get("sport") or "").strip().lower() == slate_sport
        and (
            not slate_games
            or not (thesis.get("games") or [])
            or any(str(game or "").strip() in slate_games for game in (thesis.get("games") or []))
        )
    ]
    filtered["intuition_theses"] = [
        thesis
        for thesis in (filtered.get("intuition_theses") or [])
        if str(thesis.get("sport") or "").strip().lower() == slate_sport
        and (
            not slate_games
            or not (thesis.get("games") or [])
            or any(str(game or "").strip() in slate_games for game in (thesis.get("games") or []))
        )
    ]

    kept_thesis_ids = {
        str(thesis.get("thesis_id") or "").strip()
        for thesis in (filtered.get("theses") or []) + (filtered.get("intuition_theses") or [])
        if str(thesis.get("thesis_id") or "").strip()
    }

    filtered_candidates: list[dict[str, Any]] = []
    for bundle in filtered.get("thesis_candidates") or []:
        thesis_id = str(bundle.get("thesis_id") or "").strip()
        if thesis_id and thesis_id not in kept_thesis_ids:
            continue
        new_bundle = deepcopy(bundle)
        kept_candidates: list[dict[str, Any]] = []
        for candidate in bundle.get("candidates") or []:
            legs = [leg for leg in (candidate.get("legs") or []) if leg_allowed(leg)]
            if not legs:
                continue
            new_candidate = deepcopy(candidate)
            new_candidate["legs"] = legs
            kept_candidates.append(new_candidate)
        if not kept_candidates and bundle.get("best_candidate"):
            best_legs = [leg for leg in ((bundle.get("best_candidate") or {}).get("legs") or []) if leg_allowed(leg)]
            if best_legs:
                new_bundle["best_candidate"] = {**deepcopy(bundle.get("best_candidate") or {}), "legs": best_legs}
                kept_candidates = [new_bundle["best_candidate"]]
        if kept_candidates:
            new_bundle["candidates"] = kept_candidates
            if not new_bundle.get("best_candidate"):
                new_bundle["best_candidate"] = kept_candidates[0]
            filtered_candidates.append(new_bundle)
    filtered["thesis_candidates"] = filtered_candidates

    candidate_lookup = {str(item.get("thesis_id") or "").strip(): item for item in filtered_candidates}
    judgment = deepcopy(filtered.get("thesis_judgment") or {})
    ranked: list[dict[str, Any]] = []
    for item in judgment.get("ranked_theses") or []:
        thesis_id = str(item.get("thesis_id") or "").strip()
        if thesis_id and thesis_id not in kept_thesis_ids:
            continue
        guidance = item.get("dfs_guidance") or {}
        item_copy = deepcopy(item)
        item_copy["dfs_guidance"] = {
            **guidance,
            "stack_targets": [name for name in (guidance.get("stack_targets") or []) if not slate_player_keys or dfs_name_key(name) in slate_player_keys],
            "bring_back_targets": [name for name in (guidance.get("bring_back_targets") or []) if not slate_player_keys or dfs_name_key(name) in slate_player_keys],
            "one_off_targets": [name for name in (guidance.get("one_off_targets") or []) if not slate_player_keys or dfs_name_key(name) in slate_player_keys],
            "avoid_chalk": [name for name in (guidance.get("avoid_chalk") or []) if not slate_player_keys or dfs_name_key(name) in slate_player_keys],
        }
        best_candidate = item.get("best_candidate") or candidate_lookup.get(thesis_id, {}).get("best_candidate") or {}
        best_legs = [leg for leg in (best_candidate.get("legs") or []) if leg_allowed(leg)]
        item_copy["best_candidate"] = {**deepcopy(best_candidate), "legs": best_legs} if best_legs else None
        ranked.append(item_copy)
    judgment["ranked_theses"] = ranked
    filtered["thesis_judgment"] = judgment
    return filtered


def _scope_dfs_inputs_to_player_pool(
    players,
    *,
    focus_players: set[str],
    fade_players: set[str],
    dfs_guidance: dict[str, Any],
    game_boosts: dict[str, float],
) -> tuple[set[str], set[str], dict[str, Any], dict[str, float]]:
    pool_player_keys = {dfs_name_key(player.name) for player in players if dfs_name_key(player.name)}
    pool_games = {str(player.game or "").strip() for player in players if str(player.game or "").strip()}

    def _filter_names(values) -> tuple[str, ...]:
        kept: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            name = str(value or "").strip()
            key = dfs_name_key(name)
            if not key or key not in pool_player_keys or key in seen:
                continue
            kept.append(name)
            seen.add(key)
        return tuple(kept)

    scoped_guidance = {
        "stack_targets": _filter_names(dfs_guidance.get("stack_targets") or []),
        "bring_back_targets": _filter_names(dfs_guidance.get("bring_back_targets") or []),
        "one_off_targets": _filter_names(dfs_guidance.get("one_off_targets") or []),
        "avoid_chalk": _filter_names(dfs_guidance.get("avoid_chalk") or []),
        "max_players_per_game": dfs_guidance.get("max_players_per_game"),
        "preferred_salary_shape": dfs_guidance.get("preferred_salary_shape"),
    }
    scoped_game_boosts = {
        str(game): float(score)
        for game, score in (game_boosts or {}).items()
        if str(game or "").strip() in pool_games
    }
    return (
        {name for name in focus_players if dfs_name_key(name) in pool_player_keys},
        {name for name in fade_players if dfs_name_key(name) in pool_player_keys},
        scoped_guidance,
        scoped_game_boosts,
    )


def _candidate_leg_players(candidate: dict[str, Any]) -> list[str]:
    players: list[str] = []
    for leg in candidate.get("legs", []) or []:
        player = str(leg.get("player") or leg.get("player_name") or leg.get("subject") or "").strip()
        if player and dfs_name_key(player):
            players.append(player)
    return players


def _negative_candidate_players(candidate_bundle: dict[str, Any]) -> list[str]:
    players: list[str] = []
    for candidate in candidate_bundle.get("candidates", []) or []:
        if float(candidate.get("average_edge", 0.0) or 0.0) >= 0:
            continue
        players.extend(_candidate_leg_players(candidate))
    return players


def _build_lineup_card(
    lineup: DraftKingsLineup | DraftKingsMLBLineup,
    *,
    sport: str,
    slots: tuple[str, ...],
    request_mode: str,
    contest_type: str,
    salary_cap: int,
    focus_players: set[str],
    fade_players: set[str],
) -> DFSLineupCard:
    assigned = _assign_dfs_slots(lineup.players, slots)
    focus_keys = {dfs_name_key(name) for name in focus_players if dfs_name_key(name)}
    fade_keys = {dfs_name_key(name) for name in fade_players if dfs_name_key(name)}
    counts: dict[str, int] = {}
    game_exposures: dict[str, int] = {}
    primary_games: list[str] = []
    seen_games: set[str] = set()
    slots: list[DFSLineupSlot] = []
    focus_hits: list[str] = []
    fade_hits: list[str] = []
    for slot_name, player in assigned:
        counts[player.availability_status] = counts.get(player.availability_status, 0) + 1
        if player.game:
            game_exposures[player.game] = game_exposures.get(player.game, 0) + 1
        if player.game and player.game not in seen_games:
            seen_games.add(player.game)
            primary_games.append(player.game)
        key = dfs_name_key(player.name)
        if key in focus_keys:
            focus_hits.append(player.name)
        if key in fade_keys:
            fade_hits.append(player.name)
        slots.append(
            DFSLineupSlot(
                slot=slot_name,
                player_id=player.player_id,
                player_name_key=key or "",
                name=player.name,
                team=player.team,
                opponent=player.opponent,
                game=player.game,
                salary=player.salary,
                median_fpts=player.median_fpts,
                ceiling_fpts=player.ceiling_fpts,
                availability_status=player.availability_status,
                availability_source=player.availability_source,
                is_focus=key in focus_keys,
                is_fade=key in fade_keys,
                positions=tuple(player.positions),
            )
        )
    return DFSLineupCard(
        sport=sport,
        contest_type=contest_type,
        request_mode=request_mode,
        salary_used=lineup.salary_used,
        salary_remaining=max(0, salary_cap - lineup.salary_used),
        median_fpts=lineup.median_fpts,
        ceiling_fpts=lineup.ceiling_fpts,
        floor_fpts=lineup.floor_fpts,
        average_confidence=lineup.average_confidence,
        availability_counts=counts,
        unknown_count=lineup.unknown_count,
        focus_hits=tuple(dict.fromkeys(focus_hits)),
        fade_hits=tuple(dict.fromkeys(fade_hits)),
        primary_games=tuple(primary_games),
        game_exposures=game_exposures,
        slots=tuple(slots),
    )


def _assign_dfs_slots(players, slots: tuple[str, ...]):
    used: set[int] = set()
    assigned: list[tuple[str, Any]] = []

    def assign(slot_index: int) -> bool:
        if slot_index >= len(slots):
            return True
        slot = slots[slot_index]
        candidates = sorted(
            (
                idx
                for idx, player in enumerate(players)
                if idx not in used and _player_can_fill_slot(player, slot)
            ),
            key=lambda idx: (_slot_flex_penalty(players[idx], slot), -players[idx].median_fpts, players[idx].salary),
        )
        for idx in candidates:
            used.add(idx)
            assigned.append((slot, players[idx]))
            if assign(slot_index + 1):
                return True
            assigned.pop()
            used.remove(idx)
        return False

    if not assign(0):
        raise ValueError("Could not assign DFS slots for lineup")
    return tuple(assigned)


def _player_can_fill_slot(player: DraftKingsNBAProjection | DraftKingsMLBProjection, slot: str) -> bool:
    positions = set(player.positions) | set(getattr(player, "roster_positions", tuple()) or tuple())
    if slot == "P":
        return bool(positions & {"P", "SP", "RP"})
    if slot in {"C", "1B", "2B", "3B", "SS", "OF"}:
        return slot in positions
    if slot == "UTIL":
        return bool(positions)
    if slot == "G":
        return bool(positions & {"PG", "SG"})
    if slot == "F":
        return bool(positions & {"SF", "PF"})
    return slot in positions


def _slot_flex_penalty(player: DraftKingsNBAProjection | DraftKingsMLBProjection, slot: str) -> int:
    positions = set(player.positions) | set(getattr(player, "roster_positions", tuple()) or tuple())
    if slot == "P":
        return 0 if positions <= {"P", "SP", "RP"} and (positions & {"P", "SP", "RP"}) else 1
    if slot in {"C", "1B", "2B", "3B", "SS", "OF"}:
        return 0 if slot in positions and len(positions) == 1 else 1
    if slot in {"PG", "SG", "SF", "PF", "C"}:
        return 0 if slot in positions and len(positions) == 1 else 1
    if slot == "G":
        return 1 if positions <= {"PG", "SG"} else 2
    if slot == "F":
        return 1 if positions <= {"SF", "PF"} else 2
    return 3
