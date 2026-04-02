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
