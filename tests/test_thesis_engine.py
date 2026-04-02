import unittest

import numpy as np

from quantum_parlay_oracle import Leg
from thesis_engine import build_structured_theses


class ThesisEngineTests(unittest.TestCase):
    def test_build_structured_theses_emits_nba_thin_rotation_thesis(self) -> None:
        legs = [Leg(0, "Detroit ML", "ml", "DET@OKC", 0.22, "notes", "nba")]

        def load_game_context(date_str: str, sport: str, matchup: str):
            self.assertEqual((date_str, sport, matchup), ("2026-03-30", "nba", "DET@OKC"))
            return {
                "availability": {
                    "away": [
                        {"player_name": "A", "status": "Out"},
                        {"player_name": "B", "status": "Doubtful"},
                    ],
                    "away_submitted": True,
                }
            }

        def load_matchup_profile(date_str: str, matchup: str):
            return None

        def load_nba_matchup_profile(date_str: str, matchup: str):
            return {"away_profiles": [{"name": f"P{i}"} for i in range(8)]}

        theses = build_structured_theses(
            date_str="2026-03-30",
            legs=legs,
            activation=np.array([0.41], dtype=np.float64),
            pricing_details={0: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"}},
            load_game_context=load_game_context,
            load_matchup_profile=load_matchup_profile,
            load_nba_matchup_profile=load_nba_matchup_profile,
        )

        thin = next(thesis for thesis in theses if thesis["type"] == "thin_rotation")
        self.assertEqual(thin["sport"], "nba")
        self.assertIn("Estimated playable rotation 6", thin["supporting_facts"])

    def test_build_structured_theses_emits_mlb_bullpen_and_environment_theses(self) -> None:
        legs = [Leg(0, "PIT@CIN O5.5", "total", "PIT@CIN", 0.73, "notes", "mlb")]

        def load_game_context(date_str: str, sport: str, matchup: str):
            return {
                "bullpen": {
                    "away": {"fatigue_score": 1.4},
                    "home": {"fatigue_score": 0.8},
                },
                "weather": {"temperature_f": 82, "wind_speed_mph": 13},
                "lineup_status": {"away_confirmed": False, "home_confirmed": False},
                "probable_pitchers": {
                    "away": {"fullName": "Away Starter"},
                    "home": {"fullName": "Home Starter"},
                },
            }

        def load_matchup_profile(date_str: str, matchup: str):
            return {"away_lineup": list(range(9)), "home_lineup": list(range(9))}

        def load_nba_matchup_profile(date_str: str, matchup: str):
            return None

        theses = build_structured_theses(
            date_str="2026-03-30",
            legs=legs,
            activation=np.array([0.79], dtype=np.float64),
            pricing_details={0: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"}},
            load_game_context=load_game_context,
            load_matchup_profile=load_matchup_profile,
            load_nba_matchup_profile=load_nba_matchup_profile,
        )

        thesis_types = {thesis["type"] for thesis in theses}
        self.assertIn("bullpen_exhaustion", thesis_types)
        self.assertIn("run_environment", thesis_types)

    def test_build_structured_theses_emits_model_market_divergence(self) -> None:
        legs = [Leg(0, "Dallas ML", "ml", "MIN@DAL", 0.27, "notes", "nba")]

        theses = build_structured_theses(
            date_str="2026-03-30",
            legs=legs,
            activation=np.array([0.49], dtype=np.float64),
            pricing_details={0: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"}},
            load_game_context=lambda *_: None,
            load_matchup_profile=lambda *_: None,
            load_nba_matchup_profile=lambda *_: None,
        )

        divergence = next(thesis for thesis in theses if thesis["type"] == "model_market_divergence")
        self.assertEqual(divergence["games"], ["MIN@DAL"])
        self.assertIn("Dallas ML", divergence["supporting_facts"][0])


if __name__ == "__main__":
    unittest.main()
