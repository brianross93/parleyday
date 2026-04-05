from __future__ import annotations

from basketball_sim_schema import RotationPlan

NBA_FOUL_OUT = 6


def select_lineup(
    rotation: RotationPlan,
    minutes_played: dict[str, float],
    current_lineup: tuple[str, ...],
    stint_minutes: dict[str, float],
    game_seconds_remaining: float,
    foul_counts: dict[str, int] | None = None,
) -> tuple[str, ...]:
    target_minutes = rotation.target_minutes or {player_id: 24.0 for player_id in rotation.starters}
    max_stint = rotation.max_stint_minutes or {player_id: 10.0 for player_id in target_minutes}
    closing_group = rotation.closing_group or rotation.starters
    player_pool = tuple(dict.fromkeys(rotation.starters + tuple(target_minutes.keys())))
    if len(player_pool) <= 5:
        return tuple(player_pool[:5])

    late_game = game_seconds_remaining <= (6.0 * 60.0)
    period, seconds_remaining_in_period = _game_phase(game_seconds_remaining)
    opening_window = period in {1, 3} and seconds_remaining_in_period >= (8.0 * 60.0)
    closing_window = period in {2, 4} and seconds_remaining_in_period <= (4.0 * 60.0)
    bench_window = (
        (period == 1 and seconds_remaining_in_period <= (6.0 * 60.0))
        or (period == 2 and seconds_remaining_in_period >= (8.0 * 60.0))
        or (period == 3 and seconds_remaining_in_period <= (3.5 * 60.0))
        or (period == 4 and seconds_remaining_in_period >= (9.0 * 60.0))
    )
    star_core = tuple(
        player_id
        for player_id, _ in sorted(target_minutes.items(), key=lambda item: item[1], reverse=True)[:2]
    )
    scored_players: list[tuple[float, str]] = []
    score_by_player: dict[str, float] = {}
    foul_counts = foul_counts or {}
    disqualified_players = {
        player_id
        for player_id in player_pool
        if foul_counts.get(player_id, 0) >= NBA_FOUL_OUT
    }
    for player_id in player_pool:
        player_fouls = foul_counts.get(player_id, 0)
        if player_id in disqualified_players:
            continue
        played = minutes_played.get(player_id, 0.0)
        target = target_minutes.get(player_id, 18.0)
        remaining_need = target - played
        continuity_bonus = 2.0 if player_id in current_lineup else 0.0
        closing_bonus = 5.0 if late_game and player_id in closing_group else 0.0
        starter_bonus = 1.5 if (not late_game and player_id in rotation.starters) else 0.0
        opening_bonus = 5.0 if (opening_window and player_id in rotation.starters) else 0.0
        close_half_bonus = 4.0 if (closing_window and player_id in closing_group) else 0.0
        bench_bonus = 2.8 if (bench_window and player_id not in rotation.starters) else 0.0
        stagger_bonus = 1.8 if (bench_window and player_id in star_core) else 0.0
        over_stint_penalty = 8.0 if stint_minutes.get(player_id, 0.0) >= max_stint.get(player_id, 10.0) else 0.0
        foul_penalty = 100.0 if player_fouls >= 5 else 0.0
        caution_penalty = 3.0 if (not late_game and player_fouls == 4) else 0.0
        extra_rest_penalty = 2.5 if (bench_window and player_id in rotation.starters and player_id not in star_core) else 0.0
        score = (
            remaining_need
            + continuity_bonus
            + closing_bonus
            + starter_bonus
            + opening_bonus
            + close_half_bonus
            + bench_bonus
            + stagger_bonus
            - over_stint_penalty
            - foul_penalty
            - caution_penalty
            - extra_rest_penalty
        )
        scored_players.append((score, player_id))
        score_by_player[player_id] = score

    scored_players.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen: list[str] = []
    used = set()
    available_players = [player_id for _, player_id in scored_players]

    def add_player(player_id: str | None) -> None:
        if not player_id or player_id in used or player_id in disqualified_players:
            return
        used.add(player_id)
        chosen.append(player_id)

    for starter_id in rotation.starters:
        if len(chosen) >= 5:
            break
        if late_game or closing_window:
            if starter_id in closing_group:
                add_player(starter_id)
            continue

        starter_score = score_by_player.get(starter_id, float("-inf"))
        starter_max_stint = max_stint.get(starter_id, 10.0)
        starter_over_stint = stint_minutes.get(starter_id, 0.0) >= starter_max_stint
        starter_in_foul_trouble = foul_counts.get(starter_id, 0) >= 4 and not late_game
        should_seek_backup = (
            starter_over_stint
            or starter_in_foul_trouble
            or (bench_window and starter_id not in star_core)
        )
        if should_seek_backup:
            for backup_id in rotation.backup_priority.get(starter_id, ()):
                if backup_id in used:
                    continue
                if backup_id in disqualified_players:
                    continue
                if score_by_player.get(backup_id, float("-inf")) >= (starter_score - 3.0):
                    add_player(backup_id)
                    break
        if starter_id not in used and (opening_window or starter_id in star_core or not bench_window or starter_score > 0.0):
            add_player(starter_id)

    if late_game or closing_window:
        for player_id in closing_group:
            add_player(player_id)

    for player_id in available_players:
        if len(chosen) >= 5:
            break
        add_player(player_id)

    if len(chosen) < 5:
        for player_id in player_pool:
            if foul_counts.get(player_id, 0) >= NBA_FOUL_OUT:
                continue
            if player_id not in chosen:
                chosen.append(player_id)
            if len(chosen) == 5:
                break
    return tuple(chosen[:5])


def _game_phase(game_seconds_remaining: float) -> tuple[int, float]:
    elapsed = (48.0 * 60.0) - game_seconds_remaining
    period = min(4, int(elapsed // (12.0 * 60.0)) + 1)
    period_elapsed = elapsed % (12.0 * 60.0)
    return period, max(0.0, (12.0 * 60.0) - period_elapsed)


def update_stint_minutes(
    current_lineup: tuple[str, ...],
    next_lineup: tuple[str, ...],
    stint_minutes: dict[str, float],
    added_minutes: float,
) -> dict[str, float]:
    updated = dict(stint_minutes)
    current_set = set(current_lineup)
    next_set = set(next_lineup)
    for player_id in next_set:
        if player_id in current_set:
            updated[player_id] = updated.get(player_id, 0.0) + added_minutes
        else:
            updated[player_id] = added_minutes
    for player_id in current_set - next_set:
        updated[player_id] = 0.0
    return updated
