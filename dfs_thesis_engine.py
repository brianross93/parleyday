from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from player_name_utils import dfs_name_key


@dataclass(frozen=True)
class DFSAnalysis:
    focus_players: tuple[str, ...]
    fade_players: tuple[str, ...]
    build_reasons: tuple[str, ...]
    game_boosts: dict[str, float]
    stack_targets: tuple[str, ...]
    bring_back_targets: tuple[str, ...]
    one_off_targets: tuple[str, ...]
    avoid_chalk: tuple[str, ...]
    max_players_per_game: int | None
    preferred_salary_shape: str | None


def build_nba_dfs_analysis(projections: list[Any], *, contest_type: str) -> DFSAnalysis:
    active = [player for player in projections if str(getattr(player, "availability_status", "")).lower() != "out"]
    if not active:
        return DFSAnalysis((), (), (), {}, (), (), (), (), None, None)

    game_scores = _nba_game_scores(active, contest_type=contest_type)
    ranked_games = sorted(game_scores.items(), key=lambda item: item[1], reverse=True)
    game_boosts = _normalize_game_scores(ranked_games, ceiling=1.8, keep=4)

    ranked_players = sorted(active, key=lambda player: _nba_player_score(player, contest_type=contest_type), reverse=True)
    focus_players = _top_unique_players(ranked_players, limit=4 if contest_type != "cash" else 3)

    expensive_pool = [
        player for player in active
        if float(getattr(player, "salary", 0) or 0) >= 7000
    ]
    fades_ranked = sorted(expensive_pool, key=lambda player: _nba_fade_score(player))
    fade_players = _top_unique_players(fades_ranked, limit=3)

    best_game = ranked_games[0][0] if ranked_games else ""
    stack_targets: tuple[str, ...] = ()
    bring_back_targets: tuple[str, ...] = ()
    one_off_targets: tuple[str, ...] = ()
    if contest_type != "cash" and best_game:
        stack_targets, bring_back_targets = _nba_stack_targets(active, best_game)
        focus_keys = {dfs_name_key(name) for name in focus_players}
        top_one_offs = [
            player.name
            for player in ranked_players
            if player.game != best_game and dfs_name_key(player.name) in focus_keys
        ]
        one_off_targets = tuple(dict.fromkeys(top_one_offs[:2]))

    value_leaders = sorted(active, key=lambda player: _value_per_k(player), reverse=True)
    top_values = [player.name for player in value_leaders[:2]]
    top_raw = sorted(active, key=lambda player: float(getattr(player, "median_fpts", 0.0) or 0.0), reverse=True)
    top_raw_names = [player.name for player in top_raw[:2]]
    build_reasons = []
    if best_game:
        build_reasons.append(
            f"{best_game} is the strongest DFS environment on this slate, with the best mix of raw projection and point-per-dollar depth."
        )
    if top_values:
        build_reasons.append(
            f"Best salary relief sits with {', '.join(top_values)}, giving this slate its cleanest value backbone."
        )
    if top_raw_names:
        build_reasons.append(
            f"Highest raw fantasy ceilings come from {', '.join(top_raw_names)}, so lineups still need exposure to true spend-up anchors."
        )

    preferred_salary_shape = _nba_salary_shape(active, contest_type=contest_type)
    max_players_per_game = 3 if contest_type == "cash" else 4
    avoid_chalk = tuple(name for name in fade_players[:2])

    return DFSAnalysis(
        focus_players=focus_players,
        fade_players=fade_players,
        build_reasons=tuple(build_reasons[:3]),
        game_boosts=game_boosts,
        stack_targets=stack_targets,
        bring_back_targets=bring_back_targets,
        one_off_targets=one_off_targets,
        avoid_chalk=avoid_chalk,
        max_players_per_game=max_players_per_game,
        preferred_salary_shape=preferred_salary_shape,
    )


def build_mlb_dfs_analysis(projections: list[Any], *, contest_type: str) -> DFSAnalysis:
    active = [player for player in projections if str(getattr(player, "availability_status", "")).lower() != "out"]
    if not active:
        return DFSAnalysis((), (), (), {}, (), (), (), (), None, None)

    game_scores = _mlb_game_scores(active, contest_type=contest_type)
    ranked_games = sorted(game_scores.items(), key=lambda item: item[1], reverse=True)
    game_boosts = _normalize_game_scores(ranked_games, ceiling=1.6, keep=5)

    pitchers = [player for player in active if "P" in set(getattr(player, "positions", ()))]
    hitters = [player for player in active if "P" not in set(getattr(player, "positions", ()))]
    ranked_pitchers = sorted(pitchers, key=lambda player: _mlb_pitcher_score(player, contest_type=contest_type), reverse=True)
    ranked_hitters = sorted(hitters, key=lambda player: _mlb_hitter_score(player, contest_type=contest_type), reverse=True)
    focus_players = tuple(dict.fromkeys([player.name for player in ranked_pitchers[:2] + ranked_hitters[:2]]))

    expensive_pitchers = [player for player in pitchers if float(getattr(player, "salary", 0) or 0) >= 8500]
    fade_players = tuple(
        dict.fromkeys(
            player.name
            for player in sorted(expensive_pitchers, key=lambda player: _value_per_k(player))[:2]
        )
    )

    best_game = ranked_games[0][0] if ranked_games else ""
    stack_targets: tuple[str, ...] = ()
    if best_game and contest_type != "cash":
        best_hitters = [player.name for player in ranked_hitters if player.game == best_game][:3]
        stack_targets = tuple(dict.fromkeys(best_hitters))

    build_reasons = []
    if ranked_pitchers:
        build_reasons.append(
            f"Pitching is anchored by {ranked_pitchers[0].name}, who carries the best combination of median stability and salary-adjusted strikeout upside."
        )
    if best_game:
        build_reasons.append(
            f"{best_game} is the best hitter environment on the slate, so that game deserves the strongest stack attention."
        )
    if ranked_hitters:
        build_reasons.append(
            f"Best bat values center on {', '.join(player.name for player in ranked_hitters[:2])}, giving the slate its cleanest salary relief."
        )

    return DFSAnalysis(
        focus_players=focus_players,
        fade_players=fade_players,
        build_reasons=tuple(build_reasons[:3]),
        game_boosts=game_boosts,
        stack_targets=stack_targets,
        bring_back_targets=(),
        one_off_targets=tuple(player.name for player in ranked_hitters[:2]),
        avoid_chalk=fade_players,
        max_players_per_game=4 if contest_type == "cash" else 5,
        preferred_salary_shape="balanced" if contest_type == "cash" else "stars_and_scrubs",
    )


def _nba_player_score(player: Any, *, contest_type: str) -> float:
    median = float(getattr(player, "median_fpts", 0.0) or 0.0)
    floor = float(getattr(player, "floor_fpts", 0.0) or 0.0)
    ceiling = float(getattr(player, "ceiling_fpts", 0.0) or 0.0)
    volatility = float(getattr(player, "volatility", 0.0) or 0.0)
    confidence = float(getattr(player, "projection_confidence", 0.0) or 0.0)
    value = _value_per_k(player)
    if contest_type == "cash":
        return median + (floor * 0.35) + (value * 6.5) + (confidence * 4.0)
    return ceiling + (median * 0.25) + (value * 5.0) + (volatility * 6.0)


def _nba_fade_score(player: Any) -> float:
    median = float(getattr(player, "median_fpts", 0.0) or 0.0)
    floor = float(getattr(player, "floor_fpts", 0.0) or 0.0)
    salary = float(getattr(player, "salary", 0.0) or 0.0)
    return _value_per_k(player) + ((floor / max(salary, 1.0)) * 1000.0) + (median / max(salary, 1.0) * 250.0)


def _nba_game_scores(players: list[Any], *, contest_type: str) -> dict[str, float]:
    grouped: dict[str, list[Any]] = {}
    for player in players:
        game = str(getattr(player, "game", "") or "").strip()
        if not game:
            continue
        grouped.setdefault(game, []).append(player)
    scores: dict[str, float] = {}
    for game, game_players in grouped.items():
        ranked = sorted(game_players, key=lambda player: _nba_player_score(player, contest_type=contest_type), reverse=True)
        top = ranked[:6]
        if not top:
            continue
        raw = sum(float(getattr(player, "median_fpts", 0.0) or 0.0) for player in top[:4])
        value = sum(_value_per_k(player) for player in top[:4])
        ceiling = sum(float(getattr(player, "ceiling_fpts", 0.0) or 0.0) for player in top[:2])
        scores[game] = (raw * 0.035) + (value * 0.45) + (ceiling * 0.015)
    return scores


def _nba_stack_targets(players: list[Any], best_game: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    in_game = [player for player in players if player.game == best_game]
    teams: dict[str, list[Any]] = {}
    for player in in_game:
        teams.setdefault(str(getattr(player, "team", "") or ""), []).append(player)
    ranked_teams = sorted(
        teams.items(),
        key=lambda item: sum(_nba_player_score(player, contest_type="large_field_gpp") for player in item[1][:3]),
        reverse=True,
    )
    if len(ranked_teams) < 2:
        return (), ()
    stack_team, stack_players = ranked_teams[0]
    bring_team, bring_players = ranked_teams[1]
    stack_targets = tuple(player.name for player in sorted(stack_players, key=lambda player: _nba_player_score(player, contest_type="large_field_gpp"), reverse=True)[:2])
    bring_back_targets = tuple(player.name for player in sorted(bring_players, key=lambda player: _nba_player_score(player, contest_type="large_field_gpp"), reverse=True)[:2])
    return stack_targets, bring_back_targets


def _nba_salary_shape(players: list[Any], *, contest_type: str) -> str:
    cheap_values = sum(1 for player in players if float(getattr(player, "salary", 0) or 0) <= 4500 and _value_per_k(player) >= 5.0)
    expensive_anchors = sum(1 for player in players if float(getattr(player, "salary", 0) or 0) >= 9000 and float(getattr(player, "median_fpts", 0.0) or 0.0) >= 40.0)
    if contest_type != "cash" and cheap_values >= 4 and expensive_anchors >= 2:
        return "stars_and_scrubs"
    return "balanced" if contest_type == "cash" else "leave_salary"


def _mlb_pitcher_score(player: Any, *, contest_type: str) -> float:
    median = float(getattr(player, "median_fpts", 0.0) or 0.0)
    ceiling = float(getattr(player, "ceiling_fpts", 0.0) or 0.0)
    return median + (_value_per_k(player) * 4.0) if contest_type == "cash" else ceiling + (_value_per_k(player) * 3.0)


def _mlb_hitter_score(player: Any, *, contest_type: str) -> float:
    median = float(getattr(player, "median_fpts", 0.0) or 0.0)
    ceiling = float(getattr(player, "ceiling_fpts", 0.0) or 0.0)
    volatility = float(getattr(player, "volatility", 0.0) or 0.0)
    value = _value_per_k(player)
    return median + (value * 4.5) if contest_type == "cash" else ceiling + (value * 3.5) + (volatility * 4.0)


def _mlb_game_scores(players: list[Any], *, contest_type: str) -> dict[str, float]:
    hitters = [player for player in players if "P" not in set(getattr(player, "positions", ()))]
    grouped: dict[str, list[Any]] = {}
    for player in hitters:
        game = str(getattr(player, "game", "") or "").strip()
        if not game:
            continue
        grouped.setdefault(game, []).append(player)
    scores: dict[str, float] = {}
    for game, game_hitters in grouped.items():
        ranked = sorted(game_hitters, key=lambda player: _mlb_hitter_score(player, contest_type=contest_type), reverse=True)
        top = ranked[:5]
        if not top:
            continue
        raw = sum(float(getattr(player, "median_fpts", 0.0) or 0.0) for player in top[:3])
        value = sum(_value_per_k(player) for player in top[:4])
        scores[game] = (raw * 0.05) + (value * 0.55)
    return scores


def _top_unique_players(players: list[Any], *, limit: int) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for player in players:
        name = str(getattr(player, "name", "") or "").strip()
        key = dfs_name_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= limit:
            break
    return tuple(names)


def _normalize_game_scores(ranked_games: list[tuple[str, float]], *, ceiling: float, keep: int) -> dict[str, float]:
    if not ranked_games:
        return {}
    top = ranked_games[:keep]
    max_score = max(score for _game, score in top) or 1.0
    return {
        game: round((score / max_score) * ceiling, 2)
        for game, score in top
    }


def _value_per_k(player: Any) -> float:
    salary = float(getattr(player, "salary", 0.0) or 0.0)
    median = float(getattr(player, "median_fpts", 0.0) or 0.0)
    return (median / max(salary, 1.0)) * 1000.0
