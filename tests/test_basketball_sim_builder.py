from pathlib import Path

from basketball_db import import_draftkings_slate_to_db, load_draftkings_slate_from_db
from dfs_ingest import DraftKingsPlayer, DraftKingsSlate
from dfs_nba import DraftKingsNBAProjection

import basketball_sim_builder as sim_builder
from basketball_sim_schema import DefensiveCoverage, OffensiveRole, PlayFamily


def _slate() -> DraftKingsSlate:
    players = []
    for idx, (team, opp, game) in enumerate(
        [
            ("AAA", "BBB", "AAA@BBB"),
            ("AAA", "BBB", "AAA@BBB"),
            ("AAA", "BBB", "AAA@BBB"),
            ("AAA", "BBB", "AAA@BBB"),
            ("AAA", "BBB", "AAA@BBB"),
            ("BBB", "AAA", "AAA@BBB"),
            ("BBB", "AAA", "AAA@BBB"),
            ("BBB", "AAA", "AAA@BBB"),
            ("BBB", "AAA", "AAA@BBB"),
            ("BBB", "AAA", "AAA@BBB"),
        ],
        start=1,
    ):
        players.append(
            DraftKingsPlayer(
                player_id=str(idx),
                name=f"{team} Player {idx}",
                sport="nba",
                team=team,
                opponent=opp,
                game=game,
                start_time=None,
                salary=5000 + (idx * 100),
                positions=("PG",) if idx % 5 == 1 else ("C",) if idx % 5 == 0 else ("SF",),
                roster_positions=("UTIL",),
                avg_points_per_game=25.0,
                raw_position="PG",
                raw_game_info=game,
            )
        )
    return DraftKingsSlate(
        site="draftkings",
        sport="nba",
        salary_cap=50000,
        roster_slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        players=tuple(players),
        source_path="test.csv",
    )


def _projection(player: DraftKingsPlayer, median: float, assists: float, rebounds: float, minutes: float) -> DraftKingsNBAProjection:
    return DraftKingsNBAProjection(
        player_id=player.player_id,
        name=player.name,
        team=player.team,
        opponent=player.opponent,
        salary=player.salary,
        positions=player.positions,
        roster_positions=player.roster_positions,
        game=player.game,
        median_fpts=median,
        ceiling_fpts=median * 1.25,
        floor_fpts=median * 0.75,
        volatility=0.3,
        projection_confidence=0.7,
        minutes=minutes,
        points=median * 0.55,
        rebounds=rebounds,
        assists=assists,
        availability_status="active",
        availability_source="profile",
        recent_games_sample=8.0,
        recent_minutes_avg=minutes,
        participation_rate=1.0,
        role_stability=0.8,
        recent_fpts_avg=median,
        recent_fpts_weighted=median + 2.0,
        recent_form_delta=2.0,
    )


def test_build_nba_sim_inputs_groups_players_into_games(monkeypatch) -> None:
    slate = _slate()
    projection_map = [
        _projection(player, 30.0 + idx, 6.0 if player.team == "AAA" else 3.0, 8.0, 30.0)
        for idx, player in enumerate(slate.players)
    ]
    monkeypatch.setattr(sim_builder, "build_nba_dk_projections", lambda date_str, slate_arg: projection_map)
    monkeypatch.setattr(
        sim_builder,
        "load_nba_matchup_features",
        lambda date_str, teams: {
            "AAA": {"recent_pace": 100.0, "opp_orb_rate_allowed": 0.29},
            "BBB": {"recent_pace": 98.0, "opp_orb_rate_allowed": 0.27},
        },
    )
    results = sim_builder.build_nba_sim_inputs("2026-04-02", slate)
    assert len(results) == 1
    game_input = results[0]
    assert game_input.game_id == "AAA@BBB"
    assert game_input.home_team_code == "BBB"
    assert game_input.away_team_code == "AAA"
    assert len(game_input.players) == 10
    assert PlayFamily.HIGH_PICK_AND_ROLL in game_input.home_tactics.play_family_weights
    assert DefensiveCoverage.DROP in game_input.away_tactics.coverage_weights


def test_projection_to_sim_profile_infers_creator_role() -> None:
    slate = _slate()
    player = slate.players[0]
    projection = _projection(player, 42.0, 8.0, 5.0, 34.0)
    profile = sim_builder._projection_to_sim_profile(projection)
    assert profile.offensive_role == OffensiveRole.PRIMARY_CREATOR
    assert profile.traits.pass_vision >= 1.0
    assert profile.traits.offensive_load >= 1.0
    assert profile.condition.available


def test_draftkings_slate_round_trips_through_sqlite(tmp_path: Path) -> None:
    csv_path = tmp_path / "DKSalaries.csv"
    db_path = tmp_path / "parleyday.sqlite"
    csv_path.write_text(
        "\n".join(
            [
                "Position,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame",
                "PG,Alpha Guard,1,PG/G/UTIL,8400,AAA@BBB 04/02/2026 7:00PM ET,AAA,42.5",
                "SG,Alpha Wing,2,SG/G/UTIL,7600,AAA@BBB 04/02/2026 7:00PM ET,AAA,35.0",
                "SF,Alpha Forward,3,SF/F/UTIL,7100,AAA@BBB 04/02/2026 7:00PM ET,AAA,33.2",
                "PF,Alpha Big,4,PF/F/UTIL,6900,AAA@BBB 04/02/2026 7:00PM ET,AAA,31.5",
                "C,Alpha Center,5,C/UTIL,6700,AAA@BBB 04/02/2026 7:00PM ET,AAA,30.1",
                "PG,Beta Guard,6,PG/G/UTIL,8300,AAA@BBB 04/02/2026 7:00PM ET,BBB,41.8",
                "SG,Beta Wing,7,SG/G/UTIL,7500,AAA@BBB 04/02/2026 7:00PM ET,BBB,34.0",
                "SF,Beta Forward,8,SF/F/UTIL,7000,AAA@BBB 04/02/2026 7:00PM ET,BBB,32.0",
                "PF,Beta Big,9,PF/F/UTIL,6800,AAA@BBB 04/02/2026 7:00PM ET,BBB,30.2",
                "C,Beta Center,10,C/UTIL,6600,AAA@BBB 04/02/2026 7:00PM ET,BBB,29.9",
            ]
        ),
        encoding="utf-8",
    )
    imported = import_draftkings_slate_to_db("2026-04-02", str(csv_path), db_path=str(db_path))
    loaded = load_draftkings_slate_from_db("2026-04-02", db_path=str(db_path), source_path=str(csv_path))

    assert imported.sport == "nba"
    assert len(loaded.players) == 10
    assert loaded.players[0].game == "AAA@BBB"
    assert {player.team for player in loaded.players} == {"AAA", "BBB"}


def test_build_nba_sim_inputs_from_csv_uses_sqlite_slate(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "DKSalaries.csv"
    db_path = tmp_path / "parleyday.sqlite"
    csv_path.write_text(
        "\n".join(
            [
                "Position,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame",
                "PG,AAA One,1,PG/G/UTIL,8400,AAA@BBB 04/02/2026 7:00PM ET,AAA,42.5",
                "SG,AAA Two,2,SG/G/UTIL,7600,AAA@BBB 04/02/2026 7:00PM ET,AAA,35.0",
                "SF,AAA Three,3,SF/F/UTIL,7100,AAA@BBB 04/02/2026 7:00PM ET,AAA,33.2",
                "PF,AAA Four,4,PF/F/UTIL,6900,AAA@BBB 04/02/2026 7:00PM ET,AAA,31.5",
                "C,AAA Five,5,C/UTIL,6700,AAA@BBB 04/02/2026 7:00PM ET,AAA,30.1",
                "PG,BBB One,6,PG/G/UTIL,8300,AAA@BBB 04/02/2026 7:00PM ET,BBB,41.8",
                "SG,BBB Two,7,SG/G/UTIL,7500,AAA@BBB 04/02/2026 7:00PM ET,BBB,34.0",
                "SF,BBB Three,8,SF/F/UTIL,7000,AAA@BBB 04/02/2026 7:00PM ET,BBB,32.0",
                "PF,BBB Four,9,PF/F/UTIL,6800,AAA@BBB 04/02/2026 7:00PM ET,BBB,30.2",
                "C,BBB Five,10,C/UTIL,6600,AAA@BBB 04/02/2026 7:00PM ET,BBB,29.9",
            ]
        ),
        encoding="utf-8",
    )

    imported_slate = import_draftkings_slate_to_db("2026-04-02", str(csv_path), db_path=str(db_path))
    projection_map = [
        _projection(player, 30.0 + idx, 6.0 if player.team == "AAA" else 3.0, 8.0, 30.0)
        for idx, player in enumerate(imported_slate.players)
    ]
    monkeypatch.setattr(sim_builder, "build_nba_dk_projections", lambda date_str, slate_arg: projection_map)
    monkeypatch.setattr(
        sim_builder,
        "load_nba_matchup_features",
        lambda date_str, teams: {
            "AAA": {"recent_pace": 100.0, "opp_orb_rate_allowed": 0.29},
            "BBB": {"recent_pace": 98.0, "opp_orb_rate_allowed": 0.27},
        },
    )

    results = sim_builder.build_nba_sim_inputs_from_dk_csv("2026-04-02", str(csv_path), db_path=str(db_path))
    assert len(results) == 1
    assert results[0].game_id == "AAA@BBB"
    assert len(results[0].players) == 10
