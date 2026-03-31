import unittest
from unittest.mock import patch

from llm_judge import (
    _extract_response_text,
    _extract_json_payload,
    _extract_web_sources,
    build_chat_prompt,
    build_intuition_thesis_prompt,
    build_thesis_judge_prompt,
    build_thesis_verification_prompt,
    build_judgment_payload,
    extract_judgment_games,
    generate_chat_reply,
    generate_intuition_theses,
    generate_slate_judgment,
    generate_thesis_judgment,
    generate_verified_theses,
)


class LLMJudgeTests(unittest.TestCase):
    def test_extract_response_text_falls_back_to_model_dump(self) -> None:
        class DummyResponse:
            output_text = ""

            @staticmethod
            def model_dump() -> dict:
                return {
                    "output": [
                        {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Recovered from nested payload.",
                                }
                            ]
                        }
                    ]
                }

        self.assertEqual(_extract_response_text(DummyResponse()), "Recovered from nested payload.")

    def test_extract_web_sources_reads_sources_from_response_payload(self) -> None:
        class DummyResponse:
            @staticmethod
            def model_dump() -> dict:
                return {
                    "output": [
                        {
                            "type": "web_search_call",
                            "action": {
                                "sources": [
                                    {"title": "Official source", "url": "https://example.com/official"},
                                    {"title": "Official source", "url": "https://example.com/official"},
                                ]
                            },
                        }
                    ]
                }

        self.assertEqual(
            _extract_web_sources(DummyResponse()),
            [{"title": "Official source", "url": "https://example.com/official"}],
        )

    def test_extract_json_payload_handles_fenced_json(self) -> None:
        payload = _extract_json_payload(
            """```json
            {"theses":[{"summary":"Interesting spot","sport":"nba","games":["DET@OKC"],"confidence":0.4}]}
            ```"""
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["theses"][0]["summary"], "Interesting spot")

    def test_extract_judgment_games_prefers_ranked_recommendations_and_dedupes(self) -> None:
        result = {
            "config": {"sport": "both", "date": "2026-03-29"},
            "meta": {"games": 6},
            "tier_parlays": [
                {
                    "legs": [
                        {"sport": "nba", "game": "DET@OKC"},
                        {"sport": "nba", "game": "PHX@MEM"},
                    ]
                },
                {
                    "legs": [
                        {"sport": "nba", "game": "DET@OKC"},
                        {"sport": "mlb", "game": "CHC@STL"},
                    ]
                },
            ],
            "top_legs": [
                {"sport": "mlb", "game": "CHC@STL"},
                {"sport": "mlb", "game": "NYY@BOS"},
            ],
            "fades": [
                {"sport": "mlb", "game": "TEX@BAL"},
            ],
            "moonshot": {"sport": "mlb", "game": "ATH@ATL"},
        }

        self.assertEqual(
            extract_judgment_games(result),
            [
                ("nba", "DET@OKC"),
                ("nba", "PHX@MEM"),
                ("mlb", "CHC@STL"),
                ("mlb", "NYY@BOS"),
                ("mlb", "TEX@BAL"),
                ("mlb", "ATH@ATL"),
            ],
        )

    def test_extract_judgment_games_expands_to_full_slate_coverage(self) -> None:
        result = {
            "config": {"sport": "both", "date": "2026-03-29"},
            "meta": {"games": 12},
            "tier_parlays": [
                {"legs": [{"sport": "mlb", "game": f"G{i}@H{i}"} for i in range(4)]},
                {"legs": [{"sport": "mlb", "game": f"G{i}@H{i}"} for i in range(4, 8)]},
                {"legs": [{"sport": "nba", "game": f"N{i}@M{i}"} for i in range(2)]},
            ],
            "top_legs": [{"sport": "mlb", "game": f"T{i}@U{i}"} for i in range(4)],
            "fades": [{"sport": "mlb", "game": f"F{i}@Z{i}"} for i in range(2)],
            "moonshot": {"sport": "mlb", "game": "ATH@ATL"},
        }

        self.assertEqual(len(extract_judgment_games(result)), 16)

    @patch("llm_judge.OpenAI")
    def test_generate_slate_judgment_enables_web_search_tool_when_requested(self, openai_cls) -> None:
        calls = []

        class DummyResponse:
            output_text = "Call:\nSmall stake.\nRecommended Action:\nTake one lean.\nWhy:\nContext.\nMain Risks:\nVariance.\nPasses:\nNone."

            @staticmethod
            def model_dump() -> dict:
                return {
                    "output": [
                        {
                            "type": "web_search_call",
                            "action": {"sources": [{"title": "NBA report", "url": "https://example.com/nba"}]},
                        }
                    ]
                }

        class DummyClient:
            class DummyResponses:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return DummyResponse()

            def __init__(self):
                self.responses = self.DummyResponses()

        openai_cls.return_value = DummyClient()
        result = {
            "config": {"date": "2026-03-29", "sport": "nba", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 2, "pricing_summary": {"simulation": 2}},
            "tier_parlays": [],
            "top_legs": [],
            "fades": [],
            "moonshot": None,
        }

        judgment = generate_slate_judgment(result, api_key="test-key", web_search=True)

        self.assertEqual(judgment["status"], "ok")
        self.assertTrue(judgment["web_search_enabled"])
        self.assertEqual(judgment["sources"], [{"title": "NBA report", "url": "https://example.com/nba"}])
        self.assertEqual(calls[0]["tools"][0]["type"], "web_search")
        self.assertEqual(calls[0]["tool_choice"], "auto")
        self.assertIn("web_search_call.action.sources", calls[0]["include"])

    @patch("llm_judge.OpenAI")
    def test_generate_chat_reply_omits_web_search_tool_when_disabled(self, openai_cls) -> None:
        calls = []

        class DummyResponse:
            output_text = "Answer."

        class DummyResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return DummyResponse()

        class DummyClient:
            def __init__(self):
                self.responses = DummyResponses()

        openai_cls.return_value = DummyClient()
        result = {
            "config": {"date": "2026-03-29", "sport": "nba", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 2, "pricing_summary": {"simulation": 2}},
            "tier_parlays": [],
            "top_legs": [],
            "fades": [],
            "moonshot": None,
        }

        reply = generate_chat_reply(result, "What changed?", api_key="test-key", web_search=False)

        self.assertEqual(reply["status"], "ok")
        self.assertFalse(reply["web_search_enabled"])
        self.assertNotIn("tools", calls[0])

    def test_build_chat_prompt_includes_question_and_history(self) -> None:
        result = {
            "config": {"sport": "nba", "date": "2026-03-29", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "tier_parlays": [],
            "top_legs": [{"sport": "nba", "label": "Detroit ML", "game": "DET@OKC", "category": "ml", "activation": 0.49, "implied_prob": 0.13, "score_delta": 0.36, "trust_score": 0.62, "pricing_label": "Monte Carlo", "notes": "Monte Carlo"}],
            "fades": [],
            "moonshot": None,
        }

        prompt = build_chat_prompt(
            result,
            "Why is Detroit not a stronger recommendation?",
            history=[{"role": "user", "content": "What is the best overall parlay?"}],
        )

        self.assertIn("Why is Detroit not a stronger recommendation?", prompt)
        self.assertIn("User: What is the best overall parlay?", prompt)
        self.assertIn('"game": "DET@OKC"', prompt)

    def test_build_intuition_thesis_prompt_includes_structured_theses(self) -> None:
        result = {
            "config": {"sport": "nba", "date": "2026-03-29", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "tier_parlays": [],
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"type": "thin_rotation", "summary": "Rotation is compressed."}],
            "intuition_theses": [],
        }

        prompt = build_intuition_thesis_prompt(result)

        self.assertIn('"structured_theses"', prompt)
        self.assertIn("Rotation is compressed.", prompt)

    def test_build_judgment_payload_includes_thesis_candidates(self) -> None:
        result = {
            "config": {"sport": "nba", "date": "2026-03-29", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"thesis_id": "structured_1", "type": "thin_rotation", "summary": "Rotation is compressed."}],
            "intuition_theses": [],
            "thesis_candidates": [
                {
                    "thesis_id": "structured_1",
                    "type": "thin_rotation",
                    "summary": "Rotation is compressed.",
                    "candidates": [
                        {
                            "actual_size": 2,
                            "payout_estimate": 12.4,
                            "model_joint_prob": 0.18,
                            "market_joint_prob": 0.11,
                            "expression_score": 5.2,
                            "legs": [
                                {"sport": "nba", "label": "Detroit ML", "game": "DET@OKC", "category": "ml", "activation": 0.49, "implied_prob": 0.13, "score_delta": 0.36, "pricing_label": "Monte Carlo", "notes": "Monte Carlo"},
                            ],
                        }
                    ],
                }
            ],
        }

        payload = build_judgment_payload(result)

        self.assertIn("thesis_candidates", payload)
        self.assertEqual(payload["thesis_candidates"][0]["thesis_id"], "structured_1")
        self.assertEqual(payload["thesis_candidates"][0]["candidates"][0]["actual_size"], 2)

    def test_build_thesis_judge_prompt_includes_thesis_candidates(self) -> None:
        result = {
            "config": {"sport": "nba", "date": "2026-03-29", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"thesis_id": "structured_1", "type": "thin_rotation", "summary": "Rotation is compressed."}],
            "intuition_theses": [],
            "thesis_candidates": [{"thesis_id": "structured_1", "summary": "Rotation is compressed.", "candidates": []}],
        }

        prompt = build_thesis_judge_prompt(result)

        self.assertIn('"thesis_candidates"', prompt)
        self.assertIn("Rotation is compressed.", prompt)

    def test_build_thesis_verification_prompt_includes_both_thesis_lanes(self) -> None:
        result = {
            "config": {"sport": "nba", "date": "2026-03-29", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "tier_parlays": [],
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"thesis_id": "structured_1", "type": "thin_rotation", "summary": "Rotation is compressed."}],
            "intuition_theses": [{"thesis_id": "intuition_1", "type": "market_weirdness", "summary": "Something feels stale."}],
        }

        prompt = build_thesis_verification_prompt(result)

        self.assertIn('"structured_theses"', prompt)
        self.assertIn('"intuition_theses"', prompt)
        self.assertIn("Something feels stale.", prompt)

    @patch("llm_judge.OpenAI")
    def test_generate_thesis_judgment_returns_ranked_theses(self, openai_cls) -> None:
        class DummyResponse:
            output_text = """{
                "call": "Keep one thesis alive.",
                "portfolio_note": "Treat this as a selective slate.",
                "ranked_theses": [
                    {
                        "thesis_id": "structured_1",
                        "verdict": "lean",
                        "confidence": 0.61,
                        "best_candidate_index": 0,
                        "reason": "This is the cleanest expression.",
                        "risks": ["Lineup uncertainty"],
                        "kill_conditions": ["Starter scratch"],
                        "dfs_guidance": {
                            "stack_targets": ["P.J. Washington"],
                            "bring_back_targets": ["Anthony Edwards"],
                            "one_off_targets": ["Bam Adebayo"],
                            "avoid_chalk": ["Julius Randle"],
                            "max_players_from_game": 3,
                            "preferred_salary_shape": "leave_salary"
                        }
                    }
                ]
            }"""

            @staticmethod
            def model_dump() -> dict:
                return {"output": []}

        class DummyClient:
            class DummyResponses:
                @staticmethod
                def create(**kwargs):
                    return DummyResponse()

            def __init__(self):
                self.responses = self.DummyResponses()

        openai_cls.return_value = DummyClient()
        result = {
            "config": {"date": "2026-03-29", "sport": "mlb", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"thesis_id": "structured_1", "type": "run_environment", "summary": "Warm weather game."}],
            "intuition_theses": [],
            "thesis_candidates": [
                {
                    "thesis_id": "structured_1",
                    "candidates": [
                        {
                            "actual_size": 2,
                            "payout_estimate": 12.4,
                            "model_joint_prob": 0.18,
                            "market_joint_prob": 0.11,
                            "legs": [{"sport": "mlb", "label": "Over", "game": "PIT@CIN", "category": "total", "activation": 0.6, "implied_prob": 0.5, "score_delta": 0.1, "pricing_label": "Monte Carlo", "notes": ""}],
                        }
                    ],
                }
            ],
        }

        judgment = generate_thesis_judgment(result, api_key="test-key", web_search=False)

        self.assertEqual(judgment["status"], "ok")
        self.assertEqual(judgment["call"], "Keep one thesis alive.")
        self.assertEqual(judgment["ranked_theses"][0]["verdict"], "lean")
        self.assertIsNotNone(judgment["ranked_theses"][0]["best_candidate"])
        self.assertEqual(judgment["ranked_theses"][0]["dfs_guidance"]["stack_targets"], ["P.J. Washington"])
        self.assertEqual(judgment["ranked_theses"][0]["dfs_guidance"]["preferred_salary_shape"], "leave_salary")

    @patch("llm_judge.OpenAI")
    def test_generate_thesis_judgment_preserves_conditional_verdict(self, openai_cls) -> None:
        class DummyResponse:
            output_text = """{
                "call": "Keep one thesis ready.",
                "portfolio_note": "Wait for the trigger, but keep the build ready.",
                "ranked_theses": [
                    {
                        "thesis_id": "structured_1",
                        "verdict": "conditional",
                        "confidence": 0.57,
                        "best_candidate_index": 0,
                        "reason": "The thesis is usable after one more confirmation.",
                        "risks": ["Late news"],
                        "kill_conditions": ["Starter scratch"]
                    }
                ]
            }"""

            @staticmethod
            def model_dump() -> dict:
                return {"output": []}

        class DummyClient:
            class DummyResponses:
                @staticmethod
                def create(**kwargs):
                    return DummyResponse()

            def __init__(self):
                self.responses = self.DummyResponses()

        openai_cls.return_value = DummyClient()
        result = {
            "config": {"date": "2026-03-29", "sport": "nba", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"thesis_id": "structured_1", "type": "thin_rotation", "summary": "Rotation is compressed."}],
            "intuition_theses": [],
            "thesis_candidates": [{"thesis_id": "structured_1", "candidates": [{"legs": []}]}],
        }

        judgment = generate_thesis_judgment(result, api_key="test-key", web_search=False)

        self.assertEqual(judgment["status"], "ok")
        self.assertEqual(judgment["ranked_theses"][0]["verdict"], "conditional")

    @patch("llm_judge.OpenAI")
    def test_generate_intuition_theses_returns_structured_items(self, openai_cls) -> None:
        calls = []

        class DummyResponse:
            output_text = """{
                "theses": [
                    {
                        "type": "market_weirdness",
                        "sport": "mlb",
                        "games": ["PIT@CIN"],
                        "summary": "This prop cluster feels stale relative to the rest of the board.",
                        "supporting_facts": ["Odd low price", "Fallback pricing"],
                        "missing_information": ["Confirmed lineups"],
                        "verification_targets": ["official lineups"],
                        "confidence": 0.43,
                        "kill_conditions": ["lineup scratch"],
                        "candidate_leg_types": ["prop"]
                    }
                ]
            }"""

            @staticmethod
            def model_dump() -> dict:
                return {"output": []}

        class DummyClient:
            class DummyResponses:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return DummyResponse()

            def __init__(self):
                self.responses = self.DummyResponses()

        openai_cls.return_value = DummyClient()
        result = {
            "config": {"date": "2026-03-29", "sport": "mlb", "slate_mode": "auto", "score_source": "implied"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"market": 4}},
            "tier_parlays": [],
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"type": "run_environment", "summary": "Warm weather game."}],
        }

        intuition = generate_intuition_theses(result, api_key="test-key", web_search=False)

        self.assertEqual(intuition["status"], "ok")
        self.assertEqual(len(intuition["theses"]), 1)
        self.assertEqual(intuition["theses"][0]["source"], "intuition")
        self.assertEqual(intuition["theses"][0]["type"], "market_weirdness")
        self.assertNotIn("tools", calls[0])

    @patch("llm_judge.OpenAI")
    def test_generate_verified_theses_returns_status_updates(self, openai_cls) -> None:
        calls = []

        class DummyResponse:
            output_text = """{
                "verified_theses": [
                    {
                        "thesis_id": "structured_1",
                        "verification_status": "partially_verified",
                        "updated_confidence": 0.58,
                        "verification_notes": ["Starter matchup still holds", "Lineups remain unconfirmed"],
                        "sources": [{"title": "Official lineup page", "url": "https://example.com/lineups"}]
                    }
                ]
            }"""

            @staticmethod
            def model_dump() -> dict:
                return {"output": []}

        class DummyClient:
            class DummyResponses:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return DummyResponse()

            def __init__(self):
                self.responses = self.DummyResponses()

        openai_cls.return_value = DummyClient()
        result = {
            "config": {"date": "2026-03-29", "sport": "mlb", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"games": 2, "kalshi_markets": 4, "pricing_summary": {"simulation": 4}},
            "tier_parlays": [],
            "top_legs": [],
            "fades": [],
            "moonshot": None,
            "theses": [{"thesis_id": "structured_1", "type": "run_environment", "summary": "Warm weather game."}],
            "intuition_theses": [],
        }

        verification = generate_verified_theses(result, api_key="test-key", web_search=False)

        self.assertEqual(verification["status"], "ok")
        self.assertEqual(len(verification["verified_theses"]), 1)
        self.assertEqual(verification["verified_theses"][0]["verification_status"], "partially_verified")
        self.assertEqual(verification["verified_theses"][0]["updated_confidence"], 0.58)
        self.assertEqual(
            verification["verified_theses"][0]["sources"],
            [{"title": "Official lineup page", "url": "https://example.com/lineups"}],
        )
        self.assertNotIn("tools", calls[0])

    @patch("llm_judge.load_nba_matchup_profile_snapshot")
    @patch("llm_judge.load_matchup_profile_snapshot")
    @patch("llm_judge.load_game_context_snapshot")
    def test_build_judgment_payload_includes_cached_roster_context(
        self,
        load_game_context_snapshot,
        load_matchup_profile_snapshot,
        load_nba_matchup_profile_snapshot,
    ) -> None:
        def game_context_side_effect(date_str: str, sport: str, matchup: str):
            if sport == "nba":
                return {
                    "matchup": matchup,
                    "status": "Scheduled",
                    "game_time": "2026-03-29T23:00:00Z",
                    "availability": {
                        "source": "official_nba_injury_report_pdf",
                        "away": [{"player_name": "Giannis Antetokounmpo", "status": "Out"}],
                        "home": [{"player_name": "Kawhi Leonard", "status": "Questionable"}],
                        "away_submitted": True,
                        "home_submitted": True,
                    },
                }
            return {
                "matchup": matchup,
                "status": "Scheduled",
                "game_time": "2026-03-29T19:10:00Z",
                "probable_pitchers": {
                    "away": {"fullName": "Justin Steele"},
                    "home": {"fullName": "Sonny Gray"},
                },
                "lineup_status": {"away_confirmed": True, "home_confirmed": False},
                "lineups": {
                    "away": ["One", "Two", "Three"],
                    "home": ["A", "B", "C"],
                },
                "availability": {
                    "away": {
                        "unavailable_players": [
                            {"player_name": "Cody Bellinger", "status_description": "Injured List"}
                        ],
                        "transactions": [{"description": "Placed player on injured list"}],
                    },
                    "home": {"unavailable_players": [], "transactions": []},
                },
                "bullpen": {
                    "away": {
                        "fatigue_score": 1.5,
                        "relievers": [{"player_name": "Lefty Reliever", "hand": "L", "pitches_last_3_days": 27}],
                    },
                    "home": {"fatigue_score": 0.5, "relievers": []},
                },
                "weather": {"temperature_f": 61.0, "wind_speed_mph": 12.5, "humidity_pct": 44},
                "venue": {"name": "Busch Stadium"},
            }

        load_game_context_snapshot.side_effect = game_context_side_effect
        load_matchup_profile_snapshot.return_value = {"away_lineup": [1] * 9, "home_lineup": [1] * 9}
        load_nba_matchup_profile_snapshot.return_value = {
            "away_profiles": [
                {"name": "Giannis Antetokounmpo", "minutes_per_game": 35.2, "points_per_game": 30.1}
            ],
            "home_profiles": [
                {"name": "Kawhi Leonard", "minutes_per_game": 33.0, "points_per_game": 25.0}
            ],
        }

        result = {
            "config": {"date": "2026-03-29", "sport": "both", "slate_mode": "auto", "score_source": "sim"},
            "meta": {"entropy_source": "Live matchup simulation", "kalshi_markets": 10, "games": 4, "pricing_summary": {"simulation": 8}},
            "refresh": {"game_contexts": 4, "player_profiles": 42},
            "tier_parlays": [
                {
                    "key": "best",
                    "label": "Best Overall",
                    "actual_size": 2,
                    "payout_estimate": 4.2,
                    "model_joint_prob": 0.32,
                    "market_joint_prob": 0.21,
                    "average_edge": 0.11,
                    "average_trust": 0.88,
                    "legs": [
                        {
                            "sport": "nba",
                            "label": "Bucks ML",
                            "game": "MIL@LAC",
                            "category": "ml",
                            "activation": 0.49,
                            "implied_prob": 0.31,
                            "score_delta": 0.18,
                            "trust_score": 0.72,
                            "pricing_label": "Monte Carlo",
                            "notes": "Monte Carlo",
                        },
                        {
                            "sport": "mlb",
                            "label": "Cubs ML",
                            "game": "CHC@STL",
                            "category": "ml",
                            "activation": 0.57,
                            "implied_prob": 0.46,
                            "score_delta": 0.11,
                            "trust_score": 0.84,
                            "pricing_label": "Monte Carlo",
                            "notes": "Monte Carlo",
                        },
                    ],
                }
            ],
            "top_legs": [],
            "fades": [],
            "moonshot": None,
        }

        payload = build_judgment_payload(result)

        self.assertEqual(len(payload["game_contexts"]), 2)
        nba_context = next(item for item in payload["game_contexts"] if item["sport"] == "nba")
        mlb_context = next(item for item in payload["game_contexts"] if item["sport"] == "mlb")

        self.assertEqual(nba_context["away"]["outs"], ["Giannis Antetokounmpo"])
        self.assertEqual(nba_context["home"]["questionable"], ["Kawhi Leonard"])
        self.assertEqual(mlb_context["probable_pitchers"]["away"], "Justin Steele")
        self.assertTrue(mlb_context["away"]["lineup_confirmed"])
        self.assertEqual(mlb_context["away"]["unavailable"][0]["name"], "Cody Bellinger")
        self.assertEqual(mlb_context["away"]["relievers"][0]["hand"], "L")


if __name__ == "__main__":
    unittest.main()
