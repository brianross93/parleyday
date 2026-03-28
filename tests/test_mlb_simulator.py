import unittest

from monte_carlo.mlb import MLBGameConfig, MLBGameSimulator, build_demo_mlb_matchup


class MLBSimulatorTests(unittest.TestCase):
    def test_simulator_returns_distributions_and_edges(self) -> None:
        away, home, props = build_demo_mlb_matchup()
        simulator = MLBGameSimulator(MLBGameConfig(n_simulations=250, random_seed=11))

        result = simulator.simulate_game(away=away, home=home, market_props=props)
        edges = simulator.evaluate_edges(result, props)

        self.assertEqual(len(result.away_scores), 250)
        self.assertEqual(len(result.home_scores), 250)
        self.assertIn(("Juan Soto", "hits"), result.player_props)
        self.assertTrue(edges)
        self.assertTrue(all(0.0 <= edge.sim_probability <= 1.0 for edge in edges))


if __name__ == "__main__":
    unittest.main()
