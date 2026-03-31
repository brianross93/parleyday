from dfs_ingest import DraftKingsPlayer, DraftKingsSlate
from dfs_thesis_engine import build_nba_dfs_analysis
from dfs_mlb import DraftKingsMLBProjection
from dfs_nba import DraftKingsLineup, DraftKingsNBAProjection
from dfs_strategy import (
    _build_nba_hedged_portfolio_lineups,
    _apply_cash_quality_floor,
    _filter_oracle_result_for_dfs_slate,
    build_mlb_contest_lineups,
    build_nba_contest_lineups,
    extract_mlb_dfs_guidance,
    extract_mlb_environment_scores,
    extract_nba_dfs_guidance,
    extract_nba_environment_scores,
    extract_nba_thesis_signals,
)


def test_extract_nba_thesis_signals_uses_ranked_judgment_and_candidates() -> None:
    oracle_result = {
        "theses": [
            {
                "thesis_id": "t1",
                "sport": "nba",
                "summary": "Back Washington rebounds.",
                "focus_players": ["P.J. Washington"],
                "fade_players": ["Julius Randle"],
            },
            {
                "thesis_id": "t2",
                "sport": "nba",
                "summary": "Lean Bam rebounds.",
                "focus_players": ["Bam Adebayo"],
                "fade_players": ["Tyrese Maxey"],
            },
        ],
        "thesis_candidates": [
            {
                "thesis_id": "t1",
                "best_candidate": {"legs": [{"player": "P.J. Washington"}]},
                "candidates": [],
            },
            {
                "thesis_id": "t2",
                "best_candidate": {"legs": [{"player": "Bam Adebayo"}]},
                "candidates": [],
            },
        ],
        "thesis_judgment": {
            "ranked_theses": [
                {"thesis_id": "t1", "verdict": "back", "summary": "Washington is the cleanest play."},
                {"thesis_id": "t2", "verdict": "lean", "summary": "Bam is secondary."},
            ]
        },
    }

    cash_focus, cash_fades, cash_reasons = extract_nba_thesis_signals(oracle_result, contest_type="cash")
    gpp_focus, gpp_fades, gpp_reasons = extract_nba_thesis_signals(oracle_result, contest_type="single_entry_gpp")

    assert "P.J. Washington" in cash_focus
    assert "Julius Randle" in cash_fades
    assert "Bam Adebayo" not in cash_focus
    assert cash_reasons == ("Washington is the cleanest play.",)

    assert "Bam Adebayo" in gpp_focus
    assert "Tyrese Maxey" in gpp_fades
    assert len(gpp_reasons) == 2


def test_extract_nba_environment_scores_respects_mode() -> None:
    oracle_result = {
        "theses": [
            {"thesis_id": "t1", "sport": "nba", "games": ["MIN@DAL"], "confidence": 0.8},
            {"thesis_id": "t2", "sport": "nba", "games": ["PHI@MIA"], "confidence": 0.6},
        ],
        "thesis_judgment": {
            "ranked_theses": [
                {"thesis_id": "t1", "verdict": "back", "confidence": 0.8},
                {"thesis_id": "t2", "verdict": "lean", "confidence": 0.6},
            ]
        },
    }

    h2h_scores = extract_nba_environment_scores(oracle_result, contest_type="head_to_head")
    tournament_scores = extract_nba_environment_scores(oracle_result, contest_type="tournament")

    assert h2h_scores == {"MIN@DAL": 0.8}
    assert tournament_scores["MIN@DAL"] == 0.8
    assert tournament_scores["PHI@MIA"] > 0.0


def test_extract_nba_environment_scores_includes_conditional_for_tournament_only() -> None:
    oracle_result = {
        "theses": [
            {"thesis_id": "t1", "sport": "nba", "games": ["MIN@DAL"], "confidence": 0.8},
            {"thesis_id": "t2", "sport": "nba", "games": ["PHI@MIA"], "confidence": 0.6},
        ],
        "thesis_judgment": {
            "ranked_theses": [
                {"thesis_id": "t1", "verdict": "back", "confidence": 0.8},
                {"thesis_id": "t2", "verdict": "conditional", "confidence": 0.6},
            ]
        },
    }

    h2h_scores = extract_nba_environment_scores(oracle_result, contest_type="head_to_head")
    tournament_scores = extract_nba_environment_scores(oracle_result, contest_type="tournament")

    assert h2h_scores == {"MIN@DAL": 0.8}
    assert tournament_scores["MIN@DAL"] == 0.8
    assert tournament_scores["PHI@MIA"] == 0.3


def test_extract_nba_dfs_guidance_reads_structured_guidance() -> None:
    oracle_result = {
        "thesis_judgment": {
            "ranked_theses": [
                {
                    "thesis_id": "t1",
                    "verdict": "back",
                    "dfs_guidance": {
                        "stack_targets": ["P.J. Washington"],
                        "bring_back_targets": ["Anthony Edwards"],
                        "one_off_targets": ["Bam Adebayo"],
                        "avoid_chalk": ["Julius Randle"],
                        "max_players_from_game": 3,
                        "preferred_salary_shape": "leave_salary",
                    },
                },
                {
                    "thesis_id": "t2",
                    "verdict": "lean",
                    "dfs_guidance": {
                        "stack_targets": ["Bam Adebayo"],
                    },
                },
            ]
        }
    }

    h2h_guidance = extract_nba_dfs_guidance(oracle_result, contest_type="head_to_head")
    tournament_guidance = extract_nba_dfs_guidance(oracle_result, contest_type="tournament")

    assert h2h_guidance["stack_targets"] == ("P.J. Washington",)
    assert h2h_guidance["preferred_salary_shape"] == "leave_salary"
    assert h2h_guidance["max_players_per_game"] == 3
    assert tournament_guidance["stack_targets"] == ("Bam Adebayo", "P.J. Washington")


def test_extract_mlb_guidance_and_environment_use_mlb_aliases() -> None:
    oracle_result = {
        "theses": [
            {"thesis_id": "m1", "sport": "mlb", "games": ["NYM@STL"], "confidence": 0.7},
        ],
        "thesis_judgment": {
            "ranked_theses": [
                {
                    "thesis_id": "m1",
                    "verdict": "back",
                    "confidence": 0.7,
                    "dfs_guidance": {
                        "stack_targets": ["Francisco Lindor"],
                        "preferred_salary_shape": "stars_and_scrubs",
                    },
                }
            ]
        },
    }

    guidance = extract_mlb_dfs_guidance(oracle_result, contest_type="tournament")
    environment = extract_mlb_environment_scores(oracle_result, contest_type="tournament")

    assert guidance["stack_targets"] == ("Francisco Lindor",)
    assert guidance["preferred_salary_shape"] == "stars_and_scrubs"
    assert environment == {"NYM@STL": 0.7}


def test_filter_oracle_result_for_dfs_slate_drops_off_slate_games_and_players() -> None:
    slate = DraftKingsSlate(
        site="draftkings",
        sport="nba",
        salary_cap=50000,
        roster_slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        source_path="test.csv",
        players=(
            DraftKingsPlayer("1", "Ziaire Williams", "nba", "BKN", "CHA", "CHA@BKN", None, 5200, ("SG", "SF"), ("SG", "SF", "F", "G", "UTIL"), 18.0, "SG/SF", "CHA@BKN"),
            DraftKingsPlayer("2", "Paolo Banchero", "nba", "ORL", "PHX", "PHX@ORL", None, 9500, ("PF",), ("PF", "F", "UTIL"), 42.0, "PF", "PHX@ORL"),
        ),
    )
    oracle_result = {
        "theses": [
            {"thesis_id": "keep", "sport": "nba", "games": ["CHA@BKN"], "focus_players": ["Ziaire Williams"]},
            {"thesis_id": "drop_game", "sport": "nba", "games": ["CLE@LAL"], "focus_players": ["Evan Mobley"]},
            {"thesis_id": "drop_sport", "sport": "mlb", "games": ["NYM@STL"], "focus_players": ["Francisco Lindor"]},
        ],
        "intuition_theses": [],
        "thesis_candidates": [
            {"thesis_id": "keep", "candidates": [{"legs": [{"sport": "nba", "game": "CHA@BKN", "player": "Ziaire Williams"}]}]},
            {"thesis_id": "drop_game", "candidates": [{"legs": [{"sport": "nba", "game": "CLE@LAL", "player": "Evan Mobley"}]}]},
        ],
        "thesis_judgment": {
            "ranked_theses": [
                {
                    "thesis_id": "keep",
                    "verdict": "back",
                    "dfs_guidance": {
                        "stack_targets": ["Ziaire Williams"],
                        "bring_back_targets": ["Paolo Banchero"],
                        "one_off_targets": ["Evan Mobley"],
                    },
                },
                {
                    "thesis_id": "drop_game",
                    "verdict": "back",
                    "dfs_guidance": {"stack_targets": ["Evan Mobley"]},
                },
            ]
        },
    }

    filtered = _filter_oracle_result_for_dfs_slate(slate, oracle_result)

    assert [item["thesis_id"] for item in filtered["theses"]] == ["keep"]
    assert [item["thesis_id"] for item in filtered["thesis_candidates"]] == ["keep"]
    assert [item["thesis_id"] for item in filtered["thesis_judgment"]["ranked_theses"]] == ["keep"]
    guidance = filtered["thesis_judgment"]["ranked_theses"][0]["dfs_guidance"]
    assert guidance["stack_targets"] == ["Ziaire Williams"]
    assert guidance["bring_back_targets"] == ["Paolo Banchero"]
    assert guidance["one_off_targets"] == []


def test_build_nba_contest_lineups_surfaces_availability(monkeypatch) -> None:
    slate = DraftKingsSlate(
        site="draftkings",
        sport="nba",
        salary_cap=50000,
        roster_slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        source_path="test.csv",
        players=(
            DraftKingsPlayer("1", "P.J. Washington", "nba", "DAL", "MIN", "MIN@DAL", None, 7200, ("PF",), ("PF", "F", "UTIL"), 35.0, "PF", "MIN@DAL"),
            DraftKingsPlayer("2", "Bam Adebayo", "nba", "MIA", "PHI", "PHI@MIA", None, 8600, ("C",), ("C", "UTIL"), 42.0, "C", "PHI@MIA"),
        ),
    )
    projections = [
        _projection("P.J. Washington", "DAL", "MIN", 7200, ("PF",), median=39.0, status="questionable"),
        _projection("Bam Adebayo", "MIA", "PHI", 8600, ("C",), median=44.0, status="active"),
        _projection("Guard A", "DAL", "MIN", 6000, ("PG",), median=35.0),
        _projection("Guard B", "MIA", "PHI", 5900, ("SG",), median=34.0),
        _projection("Wing A", "DAL", "MIN", 5700, ("SF",), median=33.0),
        _projection("Big A", "MIA", "PHI", 5500, ("PF",), median=32.0),
        _projection("Guard Flex", "DAL", "MIN", 5200, ("PG", "SG"), median=31.0),
        _projection("Forward Flex", "MIA", "PHI", 5100, ("SF", "PF"), median=30.0),
        _projection("Util A", "DAL", "MIN", 5000, ("SG",), median=29.0, status="unknown"),
    ]
    oracle_result = {
        "theses": [
            {
                "thesis_id": "t1",
                "sport": "nba",
                "summary": "Back Washington rebounds.",
                "focus_players": ["P.J. Washington"],
                "fade_players": ["Fade Guy"],
            }
        ],
        "thesis_candidates": [
            {"thesis_id": "t1", "best_candidate": {"legs": [{"player": "P.J. Washington"}]}, "candidates": []}
        ],
        "thesis_judgment": {
            "ranked_theses": [
                {
                    "thesis_id": "t1",
                    "verdict": "back",
                    "summary": "Washington is the cleanest play.",
                    "dfs_guidance": {
                        "stack_targets": ["P.J. Washington"],
                        "bring_back_targets": ["Bam Adebayo"],
                        "one_off_targets": ["Guard A"],
                        "avoid_chalk": ["Fade Guy"],
                        "max_players_from_game": 4,
                        "preferred_salary_shape": "balanced",
                    },
                }
            ]
        },
    }

    monkeypatch.setattr("dfs_strategy.parse_draftkings_salary_csv", lambda *args, **kwargs: slate)
    monkeypatch.setattr("dfs_strategy.build_nba_dk_projections", lambda *args, **kwargs: projections)

    result = build_nba_contest_lineups(
        date_str="2026-03-31",
        salary_csv_path="ignored.csv",
        contest_type="head_to_head",
        oracle_result=oracle_result,
    )

    assert result.lineup_cards
    assert result.best_overall_lineup is not None
    assert result.best_value_lineup is not None
    card = result.lineup_cards[0]
    assert result.request_mode == "head_to_head"
    assert result.contest_type == "cash"
    assert result.stack_targets == ("P.J. Washington",)
    assert result.bring_back_targets == ("Bam Adebayo",)
    assert result.one_off_targets == ()
    assert result.max_players_per_game == 4
    assert result.preferred_salary_shape == "balanced"
    assert card.salary_remaining == 50000 - card.salary_used
    assert card.request_mode == "head_to_head"
    assert "P.J. Washington" in card.focus_hits
    assert card.availability_counts["questionable"] == 1
    assert any(slot.name == "P.J. Washington" and slot.availability_status == "questionable" for slot in card.slots)
    assert len(card.slots) == 8
    assert all(slot.availability_status for slot in card.slots)
    assert len(result.best_overall_lineup.slots) == 8
    assert len(result.best_value_lineup.slots) == 8
    assert card.game_exposures


def test_build_nba_contest_lineups_scopes_focus_and_games_to_slate(monkeypatch) -> None:
    slate = DraftKingsSlate(
        site="draftkings",
        sport="nba",
        salary_cap=50000,
        roster_slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        source_path="test.csv",
        players=(
            DraftKingsPlayer("1", "Ziaire Williams", "nba", "BKN", "CHA", "CHA@BKN", None, 5200, ("SG", "SF"), ("SG", "SF", "F", "G", "UTIL"), 18.0, "SG/SF", "CHA@BKN"),
            DraftKingsPlayer("2", "Paolo Banchero", "nba", "ORL", "PHX", "PHX@ORL", None, 9500, ("PF",), ("PF", "F", "UTIL"), 42.0, "PF", "PHX@ORL"),
        ),
    )
    projections = [
        _projection("Ziaire Williams", "BKN", "CHA", 5200, ("SG", "SF"), median=22.0),
        _projection("Paolo Banchero", "ORL", "PHX", 9500, ("PF",), median=40.0),
        _projection("Guard A", "BKN", "CHA", 6000, ("PG",), median=35.0),
        _projection("Guard B", "ORL", "PHX", 5900, ("SG",), median=34.0),
        _projection("Wing A", "BKN", "CHA", 5700, ("SF",), median=33.0),
        _projection("Big A", "ORL", "PHX", 5500, ("C",), median=32.0),
        _projection("Guard Flex", "BKN", "CHA", 5200, ("PG", "SG"), median=31.0),
        _projection("Forward Flex", "ORL", "PHX", 5100, ("SF", "PF"), median=30.0),
    ]
    oracle_result = {
        "theses": [
            {
                "thesis_id": "t1",
                "sport": "nba",
                "summary": "Back Ziaire.",
                "focus_players": ["Ziaire Williams", "Evan Mobley"],
                "fade_players": ["Paolo Banchero", "LeBron James"],
            }
        ],
        "thesis_judgment": {
            "ranked_theses": [
                {
                    "thesis_id": "t1",
                    "verdict": "back",
                    "summary": "Back Ziaire.",
                    "confidence": 0.7,
                    "dfs_guidance": {
                        "stack_targets": ["Ziaire Williams", "Evan Mobley"],
                        "bring_back_targets": ["Paolo Banchero", "LeBron James"],
                        "one_off_targets": ["Guard A"],
                        "avoid_chalk": ["LeBron James"],
                    },
                }
            ]
        },
    }

    monkeypatch.setattr("dfs_strategy.parse_draftkings_salary_csv", lambda *args, **kwargs: slate)
    monkeypatch.setattr("dfs_strategy.build_nba_dk_projections", lambda *args, **kwargs: projections)

    result = build_nba_contest_lineups(
        date_str="2026-03-31",
        salary_csv_path="ignored.csv",
        contest_type="tournament",
        oracle_result=oracle_result,
    )

    assert result.focus_players == ("Ziaire Williams",)
    assert result.fade_players == ("Paolo Banchero",)
    assert result.stack_targets == ("Ziaire Williams",)
    assert result.bring_back_targets == ("Paolo Banchero",)
    assert result.one_off_targets == ()
    assert result.lineup_families
    assert len(result.lineup_families[0].core_players) == 2


def test_build_mlb_contest_lineups_smoke(monkeypatch) -> None:
    slate = DraftKingsSlate(
        site="draftkings",
        sport="mlb",
        salary_cap=50000,
        roster_slots=("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"),
        source_path="test.csv",
        players=tuple(),
    )
    projections = [
        _mlb_player("P1", 9500, ("P",), median=24),
        _mlb_player("P2", 8500, ("P",), median=22),
        _mlb_player("C1", 3200, ("C",), median=8),
        _mlb_player("1B1", 3800, ("1B",), median=10),
        _mlb_player("2B1", 3900, ("2B",), median=9),
        _mlb_player("3B1", 4100, ("3B",), median=9.5),
        _mlb_player("SS1", 4000, ("SS",), median=9),
        _mlb_player("OF1", 4800, ("OF",), median=11),
        _mlb_player("OF2", 4500, ("OF",), median=10),
        _mlb_player("OF3", 3700, ("OF",), median=9.5),
    ]
    oracle_result = {
        "theses": [
            {
                "thesis_id": "mlb_1",
                "sport": "mlb",
                "games": ["AAA@BBB"],
                "summary": "Warm weather hitting environment.",
                "focus_players": ["OF1"],
                "fade_players": ["P2"],
            }
        ],
        "thesis_candidates": [{"thesis_id": "mlb_1", "best_candidate": {"legs": [{"player": "OF1"}]}, "candidates": []}],
        "thesis_judgment": {
            "ranked_theses": [
                {
                    "thesis_id": "mlb_1",
                    "verdict": "back",
                    "summary": "OF1 is the cleanest stack anchor.",
                    "confidence": 0.74,
                        "dfs_guidance": {
                            "stack_targets": ["OF1"],
                            "max_players_from_game": 8,
                            "preferred_salary_shape": "stars_and_scrubs",
                        },
                    }
                ]
        },
    }
    monkeypatch.setattr("dfs_strategy.parse_draftkings_salary_csv", lambda *args, **kwargs: slate)
    monkeypatch.setattr("dfs_strategy.build_mlb_dk_projections", lambda *args, **kwargs: projections)
    monkeypatch.setattr("dfs_strategy.attach_mlb_salary_metadata", lambda *args, **kwargs: projections)

    result = build_mlb_contest_lineups(
        date_str="2026-03-31",
        salary_csv_path="ignored.csv",
        contest_type="tournament",
        oracle_result=oracle_result,
    )

    assert result.sport == "mlb"
    assert result.request_mode == "tournament"
    assert result.lineup_cards
    assert result.stack_targets == ("OF1",)
    assert result.preferred_salary_shape == "stars_and_scrubs"


def test_build_mlb_contest_lineups_renders_pitchers_with_sp_eligibility(monkeypatch) -> None:
    slate = DraftKingsSlate(
        site="draftkings",
        sport="mlb",
        salary_cap=50000,
        roster_slots=("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"),
        source_path="test.csv",
        players=tuple(),
    )
    projections = [
        _mlb_player("SP1", 9500, ("SP",), median=24, roster_positions=("P",)),
        _mlb_player("SP2", 8500, ("SP",), median=22, roster_positions=("P",)),
        _mlb_player("C1", 3200, ("C",), median=8),
        _mlb_player("1B1", 3800, ("1B",), median=10),
        _mlb_player("2B1", 3900, ("2B",), median=9),
        _mlb_player("3B1", 4100, ("3B",), median=9.5),
        _mlb_player("SS1", 4000, ("SS",), median=9),
        _mlb_player("OF1", 4800, ("OF",), median=11),
        _mlb_player("OF2", 4500, ("OF",), median=10),
        _mlb_player("OF3", 3700, ("OF",), median=9.5),
    ]
    oracle_result = {
        "theses": [{"thesis_id": "mlb_1", "sport": "mlb", "games": ["AAA@BBB"], "summary": "Playable stack.", "focus_players": ["OF1"], "fade_players": []}],
        "thesis_candidates": [{"thesis_id": "mlb_1", "best_candidate": {"legs": [{"player": "OF1"}]}, "candidates": []}],
        "thesis_judgment": {"ranked_theses": [{"thesis_id": "mlb_1", "verdict": "back", "summary": "OF1 is live."}]},
    }
    monkeypatch.setattr("dfs_strategy.parse_draftkings_salary_csv", lambda *args, **kwargs: slate)
    monkeypatch.setattr("dfs_strategy.build_mlb_dk_projections", lambda *args, **kwargs: projections)
    monkeypatch.setattr("dfs_strategy.attach_mlb_salary_metadata", lambda *args, **kwargs: projections)

    result = build_mlb_contest_lineups(
        date_str="2026-03-31",
        salary_csv_path="ignored.csv",
        contest_type="tournament",
        oracle_result=oracle_result,
    )

    assert result.lineup_cards
    pitcher_slots = [slot for slot in result.lineup_cards[0].slots if slot.slot == "P"]
    assert len(pitcher_slots) == 2
    assert {slot.name for slot in pitcher_slots} == {"SP1", "SP2"}


def test_build_nba_contest_lineups_uses_dfs_native_analysis_without_oracle(monkeypatch) -> None:
    slate = DraftKingsSlate(
        site="draftkings",
        sport="nba",
        salary_cap=50000,
        roster_slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        source_path="test.csv",
        players=(
            DraftKingsPlayer("1", "Paolo Banchero", "nba", "ORL", "PHX", "PHX@ORL", None, 9500, ("PF",), ("PF", "F", "UTIL"), 42.0, "PF", "PHX@ORL"),
            DraftKingsPlayer("2", "Coby White", "nba", "CHA", "BKN", "CHA@BKN", None, 5200, ("SG",), ("SG", "G", "UTIL"), 29.0, "SG", "CHA@BKN"),
        ),
    )
    projections = [
        _projection("Paolo Banchero", "ORL", "PHX", 9500, ("PF",), median=42.0),
        _projection("Coby White", "CHA", "BKN", 5200, ("SG",), median=30.0),
        _projection("Guard A", "ORL", "PHX", 6100, ("PG",), median=35.0),
        _projection("Wing A", "CHA", "BKN", 5600, ("SF",), median=32.0),
        _projection("Center A", "ORL", "PHX", 5800, ("C",), median=31.0),
        _projection("Guard B", "CHA", "BKN", 5400, ("PG", "SG"), median=29.0),
        _projection("Forward B", "ORL", "PHX", 5000, ("SF", "PF"), median=28.0),
        _projection("Util A", "CHA", "BKN", 4800, ("SG",), median=27.0),
        _projection("Guard C", "ORL", "PHX", 4700, ("PG",), median=26.5),
        _projection("Wing B", "CHA", "BKN", 4600, ("SF",), median=26.0),
        _projection("Center B", "ORL", "PHX", 4500, ("C",), median=25.5),
        _projection("Forward C", "CHA", "BKN", 4400, ("PF",), median=25.0),
        _projection("Combo G", "ORL", "PHX", 4300, ("PG", "SG"), median=24.0),
        _projection("Combo F", "CHA", "BKN", 4200, ("SF", "PF"), median=23.5),
        _projection("Util B", "ORL", "PHX", 4100, ("SG",), median=23.0),
        _projection("Util C", "CHA", "BKN", 4000, ("PF",), median=22.5),
        _projection("Guard D", "NYK", "HOU", 4500, ("PG",), median=26.0),
        _projection("Wing C", "HOU", "NYK", 4700, ("SF",), median=27.0),
        _projection("Center C", "NYK", "HOU", 4900, ("C",), median=28.0),
        _projection("Forward D", "HOU", "NYK", 4600, ("PF",), median=26.5),
    ]

    monkeypatch.setattr("dfs_strategy.parse_draftkings_salary_csv", lambda *args, **kwargs: slate)
    monkeypatch.setattr("dfs_strategy.build_nba_dk_projections", lambda *args, **kwargs: projections)

    result = build_nba_contest_lineups(
        date_str="2026-03-31",
        salary_csv_path="ignored.csv",
        contest_type="head_to_head",
    )

    assert result.lineup_cards
    assert result.build_reasons
    assert result.game_boosts
    assert result.focus_players
    assert result.best_overall_lineup is not None
    assert result.best_value_lineup is not None


def test_build_nba_dfs_analysis_produces_focus_and_game_boosts() -> None:
    projections = [
        _projection("Paolo Banchero", "ORL", "PHX", 9500, ("PF",), median=42.9),
        _projection("Franz Wagner", "ORL", "PHX", 7500, ("SF",), median=36.6),
        _projection("Devin Booker", "PHX", "ORL", 9400, ("SG",), median=40.1),
        _projection("Coby White", "CHA", "BKN", 5200, ("SG",), median=29.6),
        _projection("Miles Bridges", "CHA", "BKN", 5900, ("PF",), median=31.6),
        _projection("Nic Claxton", "BKN", "CHA", 5700, ("C",), median=29.4),
    ]

    analysis = build_nba_dfs_analysis(projections, contest_type="cash")

    assert analysis.build_reasons
    assert analysis.focus_players
    assert analysis.game_boosts


def test_build_nba_contest_lineups_hedges_across_alternate_spend_up_anchors(monkeypatch) -> None:
    slate = DraftKingsSlate(
        site="draftkings",
        sport="nba",
        salary_cap=50000,
        roster_slots=("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"),
        source_path="test.csv",
        players=tuple(),
    )
    projections = [
        _projection("Alperen Sengun", "HOU", "NYK", 9200, ("C",), median=44.0),
        _projection("Karl-Anthony Towns", "NYK", "HOU", 9200, ("C",), median=43.5),
        _projection("Devin Booker", "PHX", "ORL", 9400, ("SG",), median=43.0),
        _projection("Paolo Banchero", "ORL", "PHX", 9500, ("PF",), median=42.5),
        _projection("Coby White", "CHA", "BKN", 5200, ("SG",), median=30.0),
        _projection("Miles Bridges", "CHA", "BKN", 5900, ("PF",), median=31.0),
        _projection("Mikal Bridges", "NYK", "HOU", 5500, ("SF",), median=30.5),
        _projection("Amen Thompson", "HOU", "NYK", 8100, ("PG",), median=39.0),
        _projection("Nic Claxton", "BKN", "CHA", 5700, ("C",), median=29.0),
        _projection("Desmond Bane", "PHX", "ORL", 7400, ("SG",), median=34.0),
        _projection("Josh Hart", "NYK", "HOU", 6900, ("SG",), median=32.0),
        _projection("Tari Eason", "HOU", "NYK", 4400, ("PF",), median=24.0),
        _projection("Ryan Kalkbrenner", "CHA", "BKN", 3200, ("C",), median=19.0),
        _projection("Jordan Goodwin", "PHX", "ORL", 4400, ("PG",), median=22.0),
        _projection("OG Anunoby", "NYK", "HOU", 6500, ("SF", "PF"), median=31.0),
        _projection("Kon Knueppel", "CHA", "BKN", 6700, ("SG",), median=33.0),
        _projection("Royce O'Neale", "PHX", "ORL", 4800, ("SF", "PF"), median=24.0),
        _projection("Jabari Smith Jr.", "HOU", "NYK", 6300, ("PF", "C"), median=31.0),
    ]

    monkeypatch.setattr("dfs_strategy.parse_draftkings_salary_csv", lambda *args, **kwargs: slate)
    monkeypatch.setattr("dfs_strategy.build_nba_dk_projections", lambda *args, **kwargs: projections)

    result = build_nba_contest_lineups(
        date_str="2026-03-31",
        salary_csv_path="ignored.csv",
        contest_type="head_to_head",
    )

    lineup_names = [{slot.name for slot in lineup.slots} for lineup in result.lineup_cards]
    assert any("Alperen Sengun" in names for names in lineup_names)
    assert any("Karl-Anthony Towns" in names or "Devin Booker" in names for names in lineup_names)


def test_build_nba_hedged_portfolio_lineups_applies_global_exposure_across_families(monkeypatch) -> None:
    projections = [
        _projection("Anchor A", "AAA", "BBB", 9200, ("C",), median=44.0),
        _projection("Anchor B", "CCC", "DDD", 9200, ("PF",), median=43.5),
        _projection("Anchor C", "EEE", "FFF", 9100, ("SF",), median=43.0),
        _projection("Value Core", "AAA", "BBB", 3500, ("SG",), median=22.0),
        _projection("Pivot Value 1", "CCC", "DDD", 3600, ("SG",), median=21.5),
        _projection("Pivot Value 2", "EEE", "FFF", 3700, ("SG",), median=21.0),
    ]

    def fake_optimize(
        projections,
        *,
        limit,
        locked_players=None,
        excluded_players=None,
        **kwargs,
    ):
        excluded = set(excluded_players or set())
        anchor = sorted(locked_players or {"Anchor A"})[0]
        players = [next(player for player in projections if player.name == anchor)]
        value_name = "Value Core"
        if value_name in excluded:
            for alt in ("Pivot Value 1", "Pivot Value 2"):
                if alt not in excluded:
                    value_name = alt
                    break
        players.append(next(player for player in projections if player.name == value_name))
        lineup = DraftKingsLineup(
            players=tuple(players),
            salary_used=sum(player.salary for player in players),
            median_fpts=sum(player.median_fpts for player in players),
            ceiling_fpts=sum(player.ceiling_fpts for player in players),
            floor_fpts=sum(player.floor_fpts for player in players),
            average_confidence=0.7,
            unknown_count=0,
        )
        return [lineup for _ in range(limit)]

    monkeypatch.setattr("dfs_strategy.optimize_nba_classic_lineups", fake_optimize)

    result = _build_nba_hedged_portfolio_lineups(
        projections=projections,
        salary_cap=50000,
        max_candidates=None,
        limit=3,
        contest_type="cash",
        focus_players=set(),
        fade_players=set(),
        game_boosts={},
        dfs_guidance={
            "stack_targets": (),
            "bring_back_targets": (),
            "one_off_targets": (),
            "max_players_per_game": None,
            "preferred_salary_shape": "balanced",
        },
        objective_noise_scale=0.0,
        max_exposure=0.34,
    )

    value_core_exposure = sum(1 for lineup in result for player in lineup.players if player.name == "Value Core")
    assert value_core_exposure == 1


def test_apply_cash_quality_floor_drops_weak_tail() -> None:
    strong = DraftKingsLineup(
        players=tuple(),
        salary_used=50000,
        median_fpts=200.0,
        ceiling_fpts=250.0,
        floor_fpts=160.0,
        average_confidence=0.6,
        unknown_count=0,
    )
    weak = DraftKingsLineup(
        players=tuple(),
        salary_used=50000,
        median_fpts=160.0,
        ceiling_fpts=190.0,
        floor_fpts=120.0,
        average_confidence=0.4,
        unknown_count=0,
    )
    filtered = _apply_cash_quality_floor([strong, weak], game_boosts={}, quality_floor=0.92)
    assert filtered == [strong]


def _projection(
    name: str,
    team: str,
    opponent: str,
    salary: int,
    positions: tuple[str, ...],
    *,
    median: float,
    status: str = "active",
) -> DraftKingsNBAProjection:
    return DraftKingsNBAProjection(
        player_id=name,
        name=name,
        team=team,
        opponent=opponent,
        salary=salary,
        positions=positions,
        roster_positions=positions,
        game=f"{opponent}@{team}" if team else "",
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
    )


def _mlb_player(
    name: str,
    salary: int,
    positions: tuple[str, ...],
    *,
    median: float,
    roster_positions: tuple[str, ...] | None = None,
) -> DraftKingsMLBProjection:
    return DraftKingsMLBProjection(
        player_id=name,
        name=name,
        team="AAA",
        opponent="BBB",
        salary=salary,
        positions=positions,
        roster_positions=roster_positions or positions,
        game="AAA@BBB",
        median_fpts=median,
        ceiling_fpts=median * 1.35,
        floor_fpts=median * 0.62,
        volatility=0.4,
        projection_confidence=0.7,
        plate_appearances=0.0 if "P" in positions else 4.1,
        innings_pitched=6.0 if "P" in positions else 0.0,
        hits=0.0 if "P" in positions else 1.1,
        home_runs=0.0 if "P" in positions else 0.18,
        stolen_bases=0.0 if "P" in positions else 0.08,
        strikeouts=6.5 if "P" in positions else 0.0,
        runs_allowed=2.4 if "P" in positions else 0.0,
        availability_status="active",
        availability_source="profile",
    )
