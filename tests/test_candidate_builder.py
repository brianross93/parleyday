import unittest

import numpy as np

from candidate_builder import build_thesis_candidates
from quantum_parlay_oracle import Leg


class CandidateBuilderTests(unittest.TestCase):
    def test_build_thesis_candidates_creates_natural_expression(self) -> None:
        legs = [
            Leg(0, "PIT@CIN O5.5", "total", "PIT@CIN", 0.72, "notes", "mlb", entry_price=0.74),
            Leg(1, "Elly De La Cruz O 1 H", "prop", "PIT@CIN", 0.38, "notes", "mlb", entry_price=0.40),
            Leg(2, "Oneil Cruz O 1 H", "prop", "PIT@CIN", 0.36, "notes", "mlb", entry_price=0.38),
            Leg(3, "Royals ML", "ml", "MIN@KC", 0.55, "notes", "mlb", entry_price=0.57),
        ]
        theses = [
            {
                "thesis_id": "run_env_1",
                "type": "run_environment",
                "source": "structured",
                "sport": "mlb",
                "games": ["PIT@CIN"],
                "summary": "Offense-friendly run environment.",
                "confidence": 0.61,
                "candidate_leg_types": ["total", "hits", "hr"],
                "kill_conditions": ["pitching change"],
                "verification_status": "partially_verified",
            }
        ]
        activation = np.array([0.79, 0.41, 0.39, 0.55], dtype=np.float64)
        co_activation = np.outer(activation, activation)
        np.fill_diagonal(co_activation, activation)
        pricing_details = {
            0: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"},
            1: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"},
            2: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"},
            3: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"},
        }

        results = build_thesis_candidates(
            theses=theses,
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            pricing_details=pricing_details,
        )

        self.assertEqual(len(results), 1)
        thesis_result = results[0]
        self.assertIsNotNone(thesis_result["best_candidate"])
        self.assertTrue(thesis_result["best_candidate"]["legs"])
        self.assertEqual({leg["game"] for leg in thesis_result["best_candidate"]["legs"]}, {"PIT@CIN"})
        self.assertGreater(thesis_result["best_candidate"]["payout_estimate"], 1.0)
        self.assertTrue(thesis_result["supporting_legs"])
        self.assertEqual(thesis_result["supporting_legs"][0]["game"], "PIT@CIN")

    def test_build_thesis_candidates_respects_candidate_leg_types(self) -> None:
        legs = [
            Leg(0, "Team A ML", "ml", "A@B", 0.44, "notes", "nba", entry_price=0.46),
            Leg(1, "Star Guard O 30 PTS", "prop", "A@B", 0.52, "notes", "nba", entry_price=0.54),
            Leg(2, "A@B O228.5", "total", "A@B", 0.49, "notes", "nba", entry_price=0.51),
        ]
        theses = [
            {
                "thesis_id": "rotation_1",
                "type": "thin_rotation",
                "source": "structured",
                "sport": "nba",
                "games": ["A@B"],
                "summary": "Usage concentration thesis.",
                "confidence": 0.57,
                "candidate_leg_types": ["points", "rebounds", "assists", "prop"],
                "kill_conditions": [],
            }
        ]
        activation = np.array([0.47, 0.61, 0.50], dtype=np.float64)
        co_activation = np.outer(activation, activation)
        np.fill_diagonal(co_activation, activation)

        results = build_thesis_candidates(
            theses=theses,
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            pricing_details={},
        )

        self.assertEqual(len(results), 1)
        best_candidate = results[0]["best_candidate"]
        self.assertIsNotNone(best_candidate)
        labels = {leg["label"] for leg in best_candidate["legs"]}
        self.assertIn("Star Guard O 30 PTS", labels)
        self.assertNotIn("Team A ML", labels)

    def test_build_thesis_candidates_uses_intent_pools_for_supporting_legs(self) -> None:
        legs = [
            Leg(0, "Game O8.5", "total", "A@B", 0.61, "notes", "mlb", entry_price=0.63),
            Leg(1, "Slugger O 1 H", "prop", "A@B", 0.42, "notes", "mlb", entry_price=0.44),
            Leg(2, "Slugger O 1 HR", "prop", "A@B", 0.11, "notes", "mlb", entry_price=0.13),
            Leg(3, "Other Game O8.5", "total", "C@D", 0.75, "notes", "mlb", entry_price=0.77),
        ]
        theses = [
            {
                "thesis_id": "run_env_2",
                "type": "run_environment",
                "source": "structured",
                "sport": "mlb",
                "games": ["A@B"],
                "summary": "Run environment supports totals and hits.",
                "confidence": 0.63,
                "candidate_leg_types": ["total", "hits", "hr"],
                "kill_conditions": [],
            }
        ]
        activation = np.array([0.66, 0.48, 0.14, 0.80], dtype=np.float64)
        co_activation = np.outer(activation, activation)
        np.fill_diagonal(co_activation, activation)

        results = build_thesis_candidates(
            theses=theses,
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            pricing_details={},
        )

        supporting = results[0]["supporting_legs"]
        labels = {leg["label"] for leg in supporting}
        intents = {leg["intent_type"] for leg in supporting}
        self.assertIn("Game O8.5", labels)
        self.assertIn("Slugger O 1 H", labels)
        self.assertIn("Slugger O 1 HR", labels)
        self.assertNotIn("Other Game O8.5", labels)
        self.assertIn("total", intents)
        self.assertIn("hits", intents)
        self.assertIn("hr", intents)

    def test_build_thesis_candidates_prefers_focus_player_expression(self) -> None:
        legs = [
            Leg(0, "Bam Adebayo O 10 REB", "prop", "PHI@MIA", 0.55, "notes", "nba", entry_price=0.57),
            Leg(1, "Paul George O 2 REB", "prop", "PHI@MIA", 0.895, "notes", "nba", entry_price=0.96),
            Leg(2, "Tyrese Maxey O 20 PTS", "prop", "PHI@MIA", 0.87, "notes", "nba", entry_price=0.88),
        ]
        theses = [
            {
                "thesis_id": "phi_mia_rebound_control",
                "type": "model_market_divergence",
                "source": "intuition",
                "sport": "nba",
                "games": ["PHI@MIA"],
                "summary": "Miami interior control thesis.",
                "confidence": 0.53,
                "candidate_leg_types": ["rebounds", "prop"],
                "focus_players": ["Bam Adebayo"],
                "fade_players": ["Tyrese Maxey"],
                "focus_stats": ["rebounds"],
                "kill_conditions": [],
            }
        ]
        activation = np.array([0.851, 0.980, 0.502], dtype=np.float64)
        co_activation = np.outer(activation, activation)
        np.fill_diagonal(co_activation, activation)
        pricing_details = {
            0: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"},
            1: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"},
            2: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"},
        }

        results = build_thesis_candidates(
            theses=theses,
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            pricing_details=pricing_details,
        )

        self.assertEqual(len(results), 1)
        best_candidate = results[0]["best_candidate"]
        self.assertIsNotNone(best_candidate)
        labels = {leg["label"] for leg in best_candidate["legs"]}
        self.assertIn("Bam Adebayo O 10 REB", labels)
        self.assertNotIn("Tyrese Maxey O 20 PTS", labels)


if __name__ == "__main__":
    unittest.main()
