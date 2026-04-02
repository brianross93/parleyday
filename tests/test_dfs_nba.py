from dfs_ingest import DraftKingsSlate
from dfs_nba import (
    DraftKingsNBAProjection,
    _apply_matchup_feature_adjustments,
    _apply_availability_discount,
    _fallback_projection,
    _fallback_status_from_injury_context,
    _is_valid_nba_classic_lineup,
    _recent_form_bonus,
    attach_salary_metadata,
    draftkings_nba_fpts,
    optimize_nba_classic_lineups,
)
from player_name_utils import dfs_name_key


def test_draftkings_nba_scoring_baseline() -> None:
    assert draftkings_nba_fpts(20, 10, 5) == 40.0


def test_attach_salary_metadata_merges_player_pool_fields() -> None:
    from dfs_ingest import DraftKingsPlayer

    slate = DraftKingsSlate(
        site="draftkings",
        sport="nba",
        salary_cap=50000,
        roster_slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        source_path="test.csv",
        players=(
            DraftKingsPlayer(
                player_id="1",
                name="Player A",
                sport="nba",
                team="AAA",
                opponent="BBB",
                game="AAA@BBB",
                start_time=None,
                salary=7200,
                positions=("PG",),
                roster_positions=("PG", "G", "UTIL"),
                avg_points_per_game=34.0,
                raw_position="PG",
                raw_game_info="AAA@BBB",
            ),
        ),
    )
    projections = [
        DraftKingsNBAProjection(
            player_id="",
            name="Player A",
            team="AAA",
            opponent="BBB",
            salary=0,
            positions=tuple(),
            roster_positions=tuple(),
            game="AAA@BBB",
            median_fpts=35.0,
            ceiling_fpts=48.0,
            floor_fpts=25.0,
            volatility=0.3,
            projection_confidence=0.7,
            minutes=34.0,
            points=22.0,
            rebounds=5.0,
            assists=6.0,
            availability_status="active",
            availability_source="profile",
            recent_games_sample=8.0,
            recent_minutes_avg=34.0,
            participation_rate=1.0,
            role_stability=1.0,
            recent_fpts_avg=34.0,
            recent_fpts_weighted=36.0,
            recent_form_delta=2.0,
        )
    ]

    merged = attach_salary_metadata(slate, projections)

    assert merged[0].player_id == "1"
    assert merged[0].salary == 7200
    assert merged[0].positions == ("PG",)


def test_is_valid_nba_classic_lineup_accepts_full_positional_spread() -> None:
    players = (
        _player("PG1", 7000, ("PG",)),
        _player("SG1", 7000, ("SG",)),
        _player("SF1", 7000, ("SF",)),
        _player("PF1", 7000, ("PF",)),
        _player("C1", 7000, ("C",)),
        _player("G1", 6000, ("PG", "SG")),
        _player("F1", 6000, ("SF", "PF")),
        _player("U1", 5000, ("PG", "SG")),
    )
    assert _is_valid_nba_classic_lineup(players)


def test_optimize_nba_classic_lineups_returns_legal_lineup() -> None:
    pool = [
        _player("PG1", 6200, ("PG",), median=42),
        _player("SG1", 6100, ("SG",), median=40),
        _player("SF1", 6000, ("SF",), median=39),
        _player("PF1", 5900, ("PF",), median=38),
        _player("C1", 5800, ("C",), median=37),
        _player("G1", 5700, ("PG", "SG"), median=36),
        _player("F1", 5600, ("SF", "PF"), median=35),
        _player("U1", 5500, ("PG", "SG"), median=34),
        _player("EXPENSIVE", 12000, ("C",), median=39),
    ]

    lineups = optimize_nba_classic_lineups(pool, salary_cap=50000, max_candidates=9, limit=3)

    assert lineups
    assert lineups[0].salary_used <= 50000
    assert len(lineups[0].players) == 8
    assert _is_valid_nba_classic_lineup(lineups[0].players)
    assert lineups[0].unknown_count == 0


def test_apply_availability_discount_reduces_day_to_day_projection() -> None:
    adjusted = _apply_availability_discount(32.0, 18.0, 8.0, 3.0, "day-to-day")
    assert adjusted["minutes"] < 32.0
    assert adjusted["points"] < 18.0


def test_apply_matchup_feature_adjustments_boosts_relevant_stats_by_position() -> None:
    base = {"minutes": 32.0, "points": 18.0, "rebounds": 8.0, "assists": 6.0}
    team_features = {"recent_pace": 101.0}
    opponent_features = {
        "recent_pace": 103.0,
        "opp_3pa_allowed_pg": 41.0,
        "opp_orb_rate_allowed": 0.31,
        "opp_guard_ast_allowed_pg": 12.0,
        "opp_center_reb_allowed_pg": 15.0,
    }

    guard = _apply_matchup_feature_adjustments(
        base,
        team_code="AAA",
        opponent_code="BBB",
        position="G",
        team_features=team_features,
        opponent_features=opponent_features,
    )
    center = _apply_matchup_feature_adjustments(
        base,
        team_code="AAA",
        opponent_code="BBB",
        position="C",
        team_features=team_features,
        opponent_features=opponent_features,
    )

    assert guard["assists"] > base["assists"]
    assert guard["points"] > base["points"]
    assert center["rebounds"] > base["rebounds"]


def test_dfs_name_key_collapses_initials() -> None:
    assert dfs_name_key("P.J. Washington") == dfs_name_key("PJ Washington")


def test_optimize_nba_classic_lineups_excludes_out_players() -> None:
    pool = [
        _player("PG1", 6200, ("PG",), median=42),
        _player("SG1", 6100, ("SG",), median=40),
        _player("SF1", 6000, ("SF",), median=39),
        _player("PF1", 5900, ("PF",), median=38),
        _player("C1", 5800, ("C",), median=37),
        _player("G1", 5700, ("PG", "SG"), median=36),
        _player("F1", 5600, ("SF", "PF"), median=35),
        _player("U1", 5500, ("PG", "SG"), median=34),
    ]
    out_player = DraftKingsNBAProjection(
        player_id="OUT1",
        name="OUT1",
        team="AAA",
        opponent="BBB",
        salary=3000,
        positions=("PG",),
        roster_positions=("PG", "G", "UTIL"),
        game="AAA@BBB",
        median_fpts=60.0,
        ceiling_fpts=70.0,
        floor_fpts=50.0,
        volatility=0.1,
        projection_confidence=0.0,
        minutes=0.0,
        points=0.0,
        rebounds=0.0,
        assists=0.0,
        availability_status="out",
        availability_source="profile",
        recent_games_sample=0.0,
        recent_minutes_avg=0.0,
        participation_rate=0.0,
        role_stability=0.0,
        recent_fpts_avg=0.0,
        recent_fpts_weighted=0.0,
        recent_form_delta=0.0,
    )
    lineups = optimize_nba_classic_lineups(pool + [out_player], salary_cap=50000, max_candidates=10, limit=3)
    assert lineups
    assert all(all(player.name != "OUT1" for player in lineup.players) for lineup in lineups)


def test_optimize_nba_classic_lineups_keeps_unknown_players_eligible() -> None:
    pool = [
        _player("PG1", 6200, ("PG",), median=42),
        _player("SG1", 6100, ("SG",), median=40),
        _player("SF1", 6000, ("SF",), median=39),
        _player("PF1", 5900, ("PF",), median=38),
        _player("C1", 5800, ("C",), median=37),
        _player("G1", 5700, ("PG", "SG"), median=36),
        _player("F1", 5600, ("SF", "PF"), median=35),
        _player("U1", 5500, ("PG", "SG"), median=34),
    ]
    unknown_player = _player("UNKNOWN1", 3000, ("PG",), median=45, status="unknown")
    lineups = optimize_nba_classic_lineups(pool + [unknown_player], salary_cap=50000, max_candidates=10, limit=3, contest_type="large_field_gpp")
    assert lineups
    assert any(any(player.name == "UNKNOWN1" for player in lineup.players) for lineup in lineups)
    assert any(lineup.unknown_count > 0 for lineup in lineups)


def test_optimize_nba_classic_lineups_cash_filters_low_role_punts() -> None:
    core = [
        _player("PG1", 6200, ("PG",), median=42),
        _player("SG1", 6100, ("SG",), median=40),
        _player("SF1", 6000, ("SF",), median=39),
        _player("PF1", 5900, ("PF",), median=38),
        _player("C1", 5800, ("C",), median=37),
        _player("G1", 5700, ("PG", "SG"), median=36),
        _player("F1", 5600, ("SF", "PF"), median=35),
        _player("U1", 5500, ("PG", "SG"), median=34),
    ]
    low_role = _player(
        "LOW_ROLE",
        3200,
        ("SF",),
        median=20,
        recent_games_sample=2.0,
        recent_minutes_avg=11.0,
        participation_rate=0.25,
        role_stability=0.09,
    )
    lineups = optimize_nba_classic_lineups(core + [low_role], salary_cap=50000, max_candidates=9, limit=3, contest_type="cash")
    assert lineups
    assert all(all(player.name != "LOW_ROLE" for player in lineup.players) for lineup in lineups)


def test_recent_form_bonus_prefers_hotter_recent_run() -> None:
    hot = _player("HOT", 5600, ("SG",), median=30.0)
    cold = _player("COLD", 5600, ("SG",), median=30.0)
    hot = DraftKingsNBAProjection(**{**hot.__dict__, "recent_fpts_avg": 28.0, "recent_fpts_weighted": 31.5, "recent_form_delta": 3.5})
    cold = DraftKingsNBAProjection(**{**cold.__dict__, "recent_fpts_avg": 28.0, "recent_fpts_weighted": 24.5, "recent_form_delta": -3.5})
    assert _recent_form_bonus(hot, "cash") > 0.0
    assert _recent_form_bonus(cold, "cash") < 0.0


def test_fallback_status_defaults_to_active_when_team_submitted_and_player_not_listed() -> None:
    from dfs_ingest import DraftKingsPlayer

    player = DraftKingsPlayer(
        player_id="1",
        name="Cade Cunningham",
        sport="nba",
        team="DET",
        opponent="TOR",
        game="TOR@DET",
        start_time=None,
        salary=10700,
        positions=("PG",),
        roster_positions=("PG", "G", "UTIL"),
        avg_points_per_game=50.2,
        raw_position="PG",
        raw_game_info="TOR@DET",
    )

    status = _fallback_status_from_injury_context(
        player,
        {
            "availability": {
                "home": {"submitted": True, "entries": []},
                "away": {"submitted": False, "entries": []},
            }
        },
    )

    projection = _fallback_projection(player, fallback_status=status)

    assert projection.availability_status == "active"
    assert projection.availability_source == "injury_report"


def test_fallback_status_marks_listed_out_players_as_out() -> None:
    from dfs_ingest import DraftKingsPlayer

    player = DraftKingsPlayer(
        player_id="2",
        name="Giannis Antetokounmpo",
        sport="nba",
        team="MIL",
        opponent="DAL",
        game="DAL@MIL",
        start_time=None,
        salary=11000,
        positions=("PF",),
        roster_positions=("PF", "F", "UTIL"),
        avg_points_per_game=50.6,
        raw_position="PF",
        raw_game_info="DAL@MIL",
    )

    status = _fallback_status_from_injury_context(
        player,
        {
            "availability": {
                "home": {
                    "submitted": True,
                    "entries": [{"player_name": "Antetokounmpo, Giannis", "status": "Out"}],
                },
                "away": {"submitted": False, "entries": []},
            }
        },
    )

    projection = _fallback_projection(player, fallback_status=status)

    assert projection.availability_status == "out"
    assert projection.availability_source == "injury_report"


def _player(
    name: str,
    salary: int,
    positions: tuple[str, ...],
    *,
    median: float = 30.0,
    status: str = "active",
    recent_games_sample: float = 8.0,
    recent_minutes_avg: float = 32.0,
    participation_rate: float = 1.0,
    role_stability: float = 1.0,
    recent_fpts_avg: float = 30.0,
    recent_fpts_weighted: float = 30.0,
    recent_form_delta: float = 0.0,
) -> DraftKingsNBAProjection:
    return DraftKingsNBAProjection(
        player_id=name,
        name=name,
        team="AAA",
        opponent="BBB",
        salary=salary,
        positions=positions,
        roster_positions=positions,
        game="AAA@BBB",
        median_fpts=median,
        ceiling_fpts=median * 1.25,
        floor_fpts=median * 0.75,
        volatility=0.3,
        projection_confidence=0.7,
        minutes=32.0,
        points=20.0,
        rebounds=5.0,
        assists=5.0,
        availability_status=status,
        availability_source="profile",
        recent_games_sample=recent_games_sample,
        recent_minutes_avg=recent_minutes_avg,
        participation_rate=participation_rate,
        role_stability=role_stability,
        recent_fpts_avg=recent_fpts_avg,
        recent_fpts_weighted=recent_fpts_weighted,
        recent_form_delta=recent_form_delta,
    )
