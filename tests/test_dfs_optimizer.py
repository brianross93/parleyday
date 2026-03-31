from pathlib import Path

from dfs_optimizer import load_csv_player_pool, solve_dfs_lineups


def test_no_good_cuts_exclude_repeated_player_sets() -> None:
    slot_names = ("A", "B")
    salaries = (1, 1, 1)
    scores = (10.0, 9.0, 8.0)

    def eligible(player_idx: int, slot_name: str) -> bool:
        if player_idx == 0:
            return True
        if player_idx == 1:
            return slot_name in {"A", "B"}
        if player_idx == 2:
            return slot_name == "B"
        return False

    solutions = solve_dfs_lineups(
        player_count=3,
        slot_names=slot_names,
        salary_cap=10,
        salaries=salaries,
        player_scores=scores,
        eligibility_fn=eligible,
        lineups_to_generate=3,
    )

    assert len(solutions) == 3
    assert solutions[0].player_indices == (0, 1)
    assert solutions[1].player_indices == (0, 2)
    assert solutions[2].player_indices == (1, 2)
    assert len({solution.player_indices for solution in solutions}) == len(solutions)


def test_required_player_indices_force_locked_core() -> None:
    solutions = solve_dfs_lineups(
        player_count=4,
        slot_names=("A", "B"),
        salary_cap=10,
        salaries=(1, 1, 1, 1),
        player_scores=(10.0, 9.0, 8.0, 7.0),
        eligibility_fn=lambda player_idx, slot_name: True,
        lineups_to_generate=2,
        required_player_indices=(2,),
    )

    assert solutions
    assert all(2 in solution.player_indices for solution in solutions)


def test_max_exposure_caps_player_reuse() -> None:
    solutions = solve_dfs_lineups(
        player_count=4,
        slot_names=("A", "B"),
        salary_cap=10,
        salaries=(1, 1, 1, 1),
        player_scores=(10.0, 9.0, 8.0, 7.0),
        eligibility_fn=lambda player_idx, slot_name: True,
        lineups_to_generate=3,
        max_exposure=2 / 3,
    )

    appearances = sum(0 in solution.player_indices for solution in solutions)
    assert appearances <= 2


def test_load_csv_player_pool_accepts_flexible_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "players.csv"
    csv_path.write_text(
        "Player Name,Pos,Cost,Proj,TeamAbbrev,Game Info\n"
        "Nikola Jokic,C,11000,58.4,DEN,DEN@LAL\n"
        "Jamal Murray,PG/SG,7600,39.1,DEN,DEN@LAL\n",
        encoding="utf-8",
    )

    players = load_csv_player_pool(csv_path, "nba")

    assert len(players) == 2
    assert players[0].name == "Nikola Jokic"
    assert players[1].positions == ("PG", "SG")
    assert players[1].projected == 39.1
