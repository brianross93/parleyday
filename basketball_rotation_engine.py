from __future__ import annotations

from basketball_sim_schema import RotationPlan


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
    scored_players: list[tuple[float, str]] = []
    foul_counts = foul_counts or {}
    for player_id in player_pool:
        played = minutes_played.get(player_id, 0.0)
        target = target_minutes.get(player_id, 18.0)
        remaining_need = target - played
        continuity_bonus = 2.0 if player_id in current_lineup else 0.0
        closing_bonus = 5.0 if late_game and player_id in closing_group else 0.0
        starter_bonus = 1.5 if (not late_game and player_id in rotation.starters) else 0.0
        over_stint_penalty = 8.0 if stint_minutes.get(player_id, 0.0) >= max_stint.get(player_id, 10.0) else 0.0
        foul_penalty = 100.0 if foul_counts.get(player_id, 0) >= 5 else 0.0
        caution_penalty = 3.0 if (not late_game and foul_counts.get(player_id, 0) == 4) else 0.0
        score = remaining_need + continuity_bonus + closing_bonus + starter_bonus - over_stint_penalty - foul_penalty - caution_penalty
        scored_players.append((score, player_id))

    scored_players.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = [player_id for _, player_id in scored_players[:5]]
    if len(chosen) < 5:
        for player_id in player_pool:
            if player_id not in chosen:
                chosen.append(player_id)
            if len(chosen) == 5:
                break
    return tuple(chosen[:5])


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
