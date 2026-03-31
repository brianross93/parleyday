from __future__ import annotations

from pathlib import Path

from dfs_backtest import (
    draftkings_mlb_hitter_actual_fpts,
    draftkings_mlb_pitcher_actual_fpts,
    draftkings_nba_actual_fpts,
    score_saved_build,
    upsert_normalized_player_results,
)
from dfs_nba import DraftKingsLineup, DraftKingsNBAProjection
from dfs_results import fetch_saved_build_summary, save_dfs_build
from dfs_strategy import DFSLineupCard, DFSLineupFamily, DFSLineupSlot, ThesisDrivenDFSResult
from player_name_utils import dfs_name_key


def _nba_projection(
    name: str,
    team: str,
    opponent: str,
    salary: int,
    positions: tuple[str, ...],
    *,
    game: str = "NYK@HOU",
    median: float = 30.0,
    ceiling: float = 40.0,
    floor: float = 20.0,
) -> DraftKingsNBAProjection:
    return DraftKingsNBAProjection(
        player_id=name.lower().replace(" ", "_"),
        name=name,
        team=team,
        opponent=opponent,
        salary=salary,
        positions=positions,
        roster_positions=positions + ("UTIL",),
        game=game,
        median_fpts=median,
        ceiling_fpts=ceiling,
        floor_fpts=floor,
        volatility=8.0,
        projection_confidence=0.7,
        minutes=34.0,
        points=20.0,
        rebounds=8.0,
        assists=5.0,
        availability_status="active",
        availability_source="live_profile",
    )


def test_save_and_score_dfs_build_round_trip(tmp_path: Path) -> None:
    db_path = str(tmp_path / "dfs_results.sqlite")
    player_a = _nba_projection("Alpha Guard", "HOU", "NYK", 8000, ("PG",), median=42.0, ceiling=54.0)
    player_b = _nba_projection("Beta Big", "NYK", "HOU", 7600, ("C",), median=38.0, ceiling=49.0)
    lineup = DraftKingsLineup(
        players=(player_a, player_b),
        salary_used=15600,
        median_fpts=80.0,
        ceiling_fpts=103.0,
        floor_fpts=58.0,
        average_confidence=0.7,
        unknown_count=0,
    )
    card = DFSLineupCard(
        sport="nba",
        contest_type="cash",
        request_mode="head_to_head",
        salary_used=15600,
        salary_remaining=34400,
        median_fpts=80.0,
        ceiling_fpts=103.0,
        floor_fpts=58.0,
        average_confidence=0.7,
        availability_counts={"active": 2},
        unknown_count=0,
        focus_hits=("Alpha Guard",),
        fade_hits=(),
        primary_games=("NYK@HOU",),
        game_exposures={"NYK@HOU": 2},
        slots=(
            DFSLineupSlot(
                slot="PG",
                player_id=player_a.player_id,
                player_name_key=dfs_name_key(player_a.name),
                name=player_a.name,
                team=player_a.team,
                opponent=player_a.opponent,
                game=player_a.game,
                salary=player_a.salary,
                median_fpts=player_a.median_fpts,
                ceiling_fpts=player_a.ceiling_fpts,
                availability_status=player_a.availability_status,
                availability_source=player_a.availability_source,
                is_focus=True,
                is_fade=False,
                positions=player_a.positions,
            ),
            DFSLineupSlot(
                slot="C",
                player_id=player_b.player_id,
                player_name_key=dfs_name_key(player_b.name),
                name=player_b.name,
                team=player_b.team,
                opponent=player_b.opponent,
                game=player_b.game,
                salary=player_b.salary,
                median_fpts=player_b.median_fpts,
                ceiling_fpts=player_b.ceiling_fpts,
                availability_status=player_b.availability_status,
                availability_source=player_b.availability_source,
                is_focus=False,
                is_fade=False,
                positions=player_b.positions,
            ),
        ),
    )
    result = ThesisDrivenDFSResult(
        sport="nba",
        request_mode="head_to_head",
        contest_type="cash",
        focus_players=("Alpha Guard",),
        fade_players=(),
        stack_targets=(),
        bring_back_targets=(),
        one_off_targets=(),
        avoid_chalk=(),
        max_players_per_game=3,
        preferred_salary_shape="balanced",
        build_reasons=("Test build",),
        game_boosts={"NYK@HOU": 1.25},
        lineups=(lineup,),
        lineup_cards=(card,),
        lineup_families=(
            DFSLineupFamily(
                label="Alpha Guard core",
                core_players=("Alpha Guard",),
                lineup_cards=(card,),
            ),
        ),
    )
    build_id = save_dfs_build(
        slate_date="2026-03-31",
        result=result,
        salary_csv_path=r"C:\fake\DKSalaries.csv",
        input_label="DKSalaries.csv",
        db_path=db_path,
    )

    summary = fetch_saved_build_summary(build_id, db_path=db_path)
    assert summary is not None
    assert summary["build"]["sport"] == "nba"
    assert summary["lineups"][0]["family_label"] == "Alpha Guard core"

    inserted = upsert_normalized_player_results(
        sport="nba",
        slate_date="2026-03-31",
        raw_results=[
            {"player_name": "Alpha Guard", "stats": {"points": 24, "rebounds": 10, "assists": 8, "steals": 2, "blocks": 1, "turnovers": 3}},
            {"player_name": "Beta Big", "stats": {"points": 18, "rebounds": 12, "assists": 2, "steals": 0, "blocks": 2, "turnovers": 1}},
        ],
        db_path=db_path,
        source="test_box_score",
    )
    assert inserted == 2

    scored = score_saved_build(build_id, db_path=db_path, scoring_source="test_box_score")
    assert len(scored) == 1
    expected = draftkings_nba_actual_fpts(points=24, rebounds=10, assists=8, steals=2, blocks=1, turnovers=3) + draftkings_nba_actual_fpts(points=18, rebounds=12, assists=2, steals=0, blocks=2, turnovers=1)
    assert scored[0]["actual_points"] == expected
    assert scored[0]["missing_players"] == 0


def test_draftkings_actual_scoring_formulas() -> None:
    nba_points = draftkings_nba_actual_fpts(points=10, rebounds=10, assists=10, steals=1, blocks=0, turnovers=2)
    assert nba_points == 10 + 12.5 + 15 + 2 - 1 + 1.5 + 3

    hitter_points = draftkings_mlb_hitter_actual_fpts(
        singles=1,
        doubles=1,
        triples=0,
        home_runs=1,
        runs_batted_in=3,
        runs=2,
        walks=1,
        hit_by_pitch=0,
        stolen_bases=1,
    )
    assert hitter_points == 3 + 5 + 10 + 6 + 4 + 2 + 5

    pitcher_points = draftkings_mlb_pitcher_actual_fpts(
        innings_pitched=6.0,
        strikeouts=8,
        earned_runs=2,
        hits_allowed=5,
        walks_allowed=1,
        hit_batters=0,
        win=1,
        complete_game=0,
        shutout=0,
        no_hitter=0,
    )
    assert pitcher_points == (6.0 * 2.25) + 16 - 4 - (6 * 0.6) + 4
