from basketball_rotation_engine import select_lineup, update_stint_minutes
from basketball_sim_schema import RotationPlan


def test_select_lineup_prefers_players_with_remaining_minute_need() -> None:
    rotation = RotationPlan(
        starters=("a", "b", "c", "d", "e"),
        closing_group=("a", "b", "c", "d", "e"),
        target_minutes={
            "a": 34.0,
            "b": 34.0,
            "c": 34.0,
            "d": 34.0,
            "e": 34.0,
            "f": 24.0,
            "g": 22.0,
        },
        max_stint_minutes={player_id: 8.0 for player_id in ("a", "b", "c", "d", "e", "f", "g")},
    )
    lineup = select_lineup(
        rotation,
        minutes_played={"a": 30.0, "b": 30.0, "c": 30.0, "d": 30.0, "e": 30.0, "f": 8.0, "g": 6.0},
        current_lineup=("a", "b", "c", "d", "e"),
        stint_minutes={"a": 9.0, "b": 9.0, "c": 9.0, "d": 9.0, "e": 9.0},
        game_seconds_remaining=20.0 * 60.0,
    )
    assert "f" in lineup
    assert "g" in lineup


def test_update_stint_minutes_resets_subbed_out_players() -> None:
    updated = update_stint_minutes(
        current_lineup=("a", "b", "c", "d", "e"),
        next_lineup=("a", "b", "c", "d", "f"),
        stint_minutes={"a": 6.0, "b": 6.0, "c": 6.0, "d": 6.0, "e": 6.0},
        added_minutes=0.5,
    )
    assert updated["a"] == 6.5
    assert updated["f"] == 0.5
    assert updated["e"] == 0.0


def test_select_lineup_benches_five_foul_player_when_alternatives_exist() -> None:
    rotation = RotationPlan(
        starters=("a", "b", "c", "d", "e"),
        closing_group=("a", "b", "c", "d", "e"),
        target_minutes={key: 30.0 for key in ("a", "b", "c", "d", "e", "f")},
        max_stint_minutes={key: 8.0 for key in ("a", "b", "c", "d", "e", "f")},
    )
    lineup = select_lineup(
        rotation,
        minutes_played={key: 20.0 for key in ("a", "b", "c", "d", "e", "f")},
        current_lineup=("a", "b", "c", "d", "e"),
        stint_minutes={key: 2.0 for key in ("a", "b", "c", "d", "e")},
        game_seconds_remaining=18.0 * 60.0,
        foul_counts={"a": 5},
    )
    assert "a" not in lineup
    assert "f" in lineup
