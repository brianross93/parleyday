from dfs_ingest import DraftKingsPlayer, DraftKingsSlate
from dfs_nba import DraftKingsNBAProjection

import basketball_sim_builder as sim_builder
from basketball_game_engine import simulate_game, simulate_games


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


def _sim_inputs(monkeypatch):
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
    return sim_builder.build_nba_sim_inputs("2026-04-02", slate)


def test_simulate_game_returns_coherent_result(monkeypatch) -> None:
    sim_inputs = _sim_inputs(monkeypatch)
    result = simulate_game(sim_inputs[0], rng_seed=7)
    assert result.game_id == "AAA@BBB"
    assert result.possession_count >= 150
    assert result.home_score >= 0
    assert result.away_score >= 0
    assert len(result.event_log) >= result.possession_count
    assert len(result.player_box_scores) == 10
    assert {box.team_code for box in result.team_box_scores} == {"AAA", "BBB"}
    assert max(player.minutes for player in result.player_box_scores) <= 48.0
    assert sum(1 for player in result.player_box_scores if player.minutes > 0.0) > 5


def test_possessions_reset_shot_clock_on_change(monkeypatch) -> None:
    sim_inputs = _sim_inputs(monkeypatch)
    result = simulate_game(sim_inputs[0], rng_seed=7)
    assert result.possessions
    for possession in result.possessions:
        if possession.entry_type.value == "oreb":
            assert possession.start_shot_clock == 14.0
        else:
            assert possession.start_shot_clock == 24.0


def test_simulate_games_runs_multiple_inputs(monkeypatch) -> None:
    sim_inputs = _sim_inputs(monkeypatch)
    results = simulate_games(sim_inputs, rng_seed=11)
    assert len(results) == 1
    assert results[0].game_id == "AAA@BBB"
