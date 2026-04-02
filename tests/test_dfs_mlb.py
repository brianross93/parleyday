from dfs_ingest import DraftKingsPlayer
from dfs_mlb import (
    DraftKingsMLBProjection,
    _fallback_projection,
    _is_valid_mlb_classic_lineup,
    draftkings_mlb_hitter_fpts,
    draftkings_mlb_pitcher_fpts,
    optimize_mlb_classic_lineups,
)


def test_draftkings_mlb_hitter_scoring_baseline() -> None:
    score = draftkings_mlb_hitter_fpts(
        singles=1,
        doubles=1,
        triples=0,
        home_runs=1,
        runs_batted_in=2,
        runs=2,
        walks_hbp=1,
        stolen_bases=1,
    )
    assert score == 33.0


def test_draftkings_mlb_pitcher_scoring_baseline() -> None:
    score = draftkings_mlb_pitcher_fpts(
        innings_pitched=6.0,
        strikeouts=7.0,
        win_prob=0.5,
        earned_runs=2.0,
        hits_allowed=5.0,
        walks_allowed=2.0,
    )
    assert round(score, 2) == 21.3


def test_is_valid_mlb_classic_lineup_accepts_full_positional_spread() -> None:
    lineup = (
        _player("P1", 9500, ("P",)),
        _player("P2", 9100, ("P",)),
        _player("C1", 3200, ("C",)),
        _player("1B1", 4300, ("1B",)),
        _player("2B1", 3900, ("2B",)),
        _player("3B1", 4100, ("3B",)),
        _player("SS1", 4000, ("SS",)),
        _player("OF1", 4800, ("OF",)),
        _player("OF2", 4500, ("OF",)),
        _player("OF3", 4300, ("OF",)),
    )
    assert _is_valid_mlb_classic_lineup(lineup)


def test_optimize_mlb_classic_lineups_returns_legal_lineup() -> None:
    pool = [
        _player("P1", 9500, ("P",), median=24),
        _player("P2", 9100, ("P",), median=22),
        _player("C1", 3200, ("C",), median=8),
        _player("1B1", 4300, ("1B",), median=10),
        _player("2B1", 3900, ("2B",), median=9),
        _player("3B1", 4100, ("3B",), median=9.5),
        _player("SS1", 4000, ("SS",), median=9),
        _player("OF1", 4800, ("OF",), median=11),
        _player("OF2", 4500, ("OF",), median=10),
        _player("OF3", 4300, ("OF",), median=9.5),
        _player("OF4", 2700, ("OF",), median=5.5),
    ]

    lineups = optimize_mlb_classic_lineups(pool, salary_cap=50000, max_candidates=16, limit=3)

    assert lineups
    assert lineups[0].salary_used <= 50000
    assert len(lineups[0].players) == 10
    assert _is_valid_mlb_classic_lineup(lineups[0].players)


def test_fallback_projection_treats_salary_pool_pitchers_as_active() -> None:
    pitcher = DraftKingsPlayer(
        player_id="p1",
        name="Garrett Crochet",
        sport="mlb",
        team="BOS",
        opponent="HOU",
        game="BOS@HOU",
        start_time=None,
        salary=9500,
        positions=("SP",),
        roster_positions=("P",),
        avg_points_per_game=25.5,
        raw_position="SP",
        raw_game_info="BOS@HOU",
    )

    projection = _fallback_projection(pitcher)

    assert projection.availability_status == "active"
    assert projection.availability_source == "salary_pool_pitcher"


def test_fallback_projection_rejects_non_probable_pitchers_when_probable_known() -> None:
    pitcher = DraftKingsPlayer(
        player_id="p1",
        name="Garrett Crochet",
        sport="mlb",
        team="BOS",
        opponent="HOU",
        game="BOS@HOU",
        start_time=None,
        salary=9500,
        positions=("SP",),
        roster_positions=("P",),
        avg_points_per_game=25.5,
        raw_position="SP",
        raw_game_info="BOS@HOU",
    )

    projection = _fallback_projection(pitcher, probable_name="Brayan Bello")

    assert projection.availability_status == "out"
    assert projection.availability_source == "not_probable_pitcher"


def test_fallback_projection_keeps_matching_probable_pitcher_active() -> None:
    pitcher = DraftKingsPlayer(
        player_id="p1",
        name="Ryan Feltner",
        sport="mlb",
        team="COL",
        opponent="TOR",
        game="COL@TOR",
        start_time=None,
        salary=6200,
        positions=("SP",),
        roster_positions=("P",),
        avg_points_per_game=9.88,
        raw_position="SP",
        raw_game_info="COL@TOR",
    )

    projection = _fallback_projection(pitcher, probable_name="Ryan Feltner")

    assert projection.availability_status == "active"
    assert projection.availability_source == "salary_pool_probable_pitcher"


def _player(
    name: str,
    salary: int,
    positions: tuple[str, ...],
    *,
    median: float = 8.0,
    status: str = "active",
) -> DraftKingsMLBProjection:
    return DraftKingsMLBProjection(
        player_id=name,
        name=name,
        team="AAA",
        opponent="BBB",
        salary=salary,
        positions=positions,
        roster_positions=positions,
        game="AAA@BBB",
        median_fpts=median,
        ceiling_fpts=median * 1.4,
        floor_fpts=median * 0.6,
        volatility=0.4,
        projection_confidence=0.7,
        plate_appearances=4.0 if "P" not in positions else 0.0,
        innings_pitched=6.0 if "P" in positions else 0.0,
        hits=1.0 if "P" not in positions else 0.0,
        home_runs=0.2 if "P" not in positions else 0.0,
        stolen_bases=0.1 if "P" not in positions else 0.0,
        strikeouts=6.0 if "P" in positions else 0.0,
        runs_allowed=2.0 if "P" in positions else 0.0,
        availability_status=status,
        availability_source="profile",
    )
