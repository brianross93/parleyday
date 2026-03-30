import unittest
from unittest.mock import patch

from llm_judge import _extract_response_text, build_chat_prompt, build_judgment_payload, extract_judgment_games


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

    def test_extract_judgment_games_prefers_ranked_recommendations_and_dedupes(self) -> None:
        result = {
            "config": {"sport": "both", "date": "2026-03-29"},
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
        }

        self.assertEqual(
            extract_judgment_games(result),
            [("nba", "DET@OKC"), ("nba", "PHX@MEM"), ("mlb", "CHC@STL"), ("mlb", "NYY@BOS")],
        )

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
