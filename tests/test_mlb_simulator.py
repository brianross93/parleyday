import unittest

from monte_carlo.mlb import BatterProfile, MLBGameConfig, MLBGameSimulator, PitcherProfile, TeamContext, build_demo_mlb_matchup


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

    def test_simulator_hands_game_to_bullpen_when_starter_is_spent(self) -> None:
        lineup = tuple(
            BatterProfile(
                player_id=str(idx),
                name=f"Batter {idx}",
                hand="R",
                pa_share=0.11,
                strikeout_rate=0.18,
                walk_rate=0.08,
                hbp_rate=0.01,
                single_rate=0.16,
                double_rate=0.05,
                triple_rate=0.01,
                home_run_rate=0.03,
            )
            for idx in range(1, 10)
        )
        tired_starter = PitcherProfile(
            player_id="SP1",
            name="Tired Starter",
            hand="R",
            strikeout_rate=0.18,
            walk_rate=0.09,
            hbp_rate=0.01,
            single_rate=0.17,
            double_rate=0.05,
            triple_rate=0.01,
            home_run_rate=0.04,
            fatigue_start=8,
            fatigue_full=12,
        )
        reliever = PitcherProfile(
            player_id="RP1",
            name="Bridge Reliever",
            hand="R",
            strikeout_rate=0.24,
            walk_rate=0.08,
            hbp_rate=0.01,
            single_rate=0.14,
            double_rate=0.04,
            triple_rate=0.005,
            home_run_rate=0.025,
            fatigue_start=18,
            fatigue_full=28,
        )
        away = TeamContext(team_code="AWY", lineup=lineup, starter=tired_starter, bullpen=(reliever,))
        home = TeamContext(team_code="HME", lineup=lineup, starter=tired_starter, bullpen=(reliever,))

        simulator = MLBGameSimulator(MLBGameConfig(n_simulations=150, random_seed=17))
        result = simulator.simulate_game(away=away, home=home)

        self.assertIn(("Bridge Reliever", "hits_allowed"), result.player_props)
        self.assertGreater(result.player_props[("Bridge Reliever", "hits_allowed")].mean, 0.0)
        self.assertLess(result.player_props[("Tired Starter", "hits_allowed")].mean, 12.0)

    def test_platoon_split_factor_changes_hit_outcomes_by_pitcher_hand(self) -> None:
        split_batter = BatterProfile(
            player_id="1",
            name="Split Bat",
            hand="L",
            pa_share=0.12,
            strikeout_rate=0.18,
            walk_rate=0.08,
            hbp_rate=0.01,
            single_rate=0.17,
            double_rate=0.05,
            triple_rate=0.01,
            home_run_rate=0.04,
            vs_left_factor=0.86,
            vs_right_factor=1.14,
        )
        lineup = tuple(
            split_batter if idx == 0 else BatterProfile(
                player_id=str(idx + 1),
                name=f"Filler {idx + 1}",
                hand="R",
                pa_share=0.11,
                strikeout_rate=0.21,
                walk_rate=0.07,
                hbp_rate=0.01,
                single_rate=0.15,
                double_rate=0.04,
                triple_rate=0.005,
                home_run_rate=0.03,
            )
            for idx in range(9)
        )
        righty = PitcherProfile("RSP", "Righty", "R", 0.21, 0.08, 0.01, 0.15, 0.04, 0.005, 0.03, 120, 140)
        lefty = PitcherProfile("LSP", "Lefty", "L", 0.21, 0.08, 0.01, 0.15, 0.04, 0.005, 0.03, 120, 140)
        offense = TeamContext(team_code="OFF", lineup=lineup, starter=righty)
        defense_right = TeamContext(team_code="RGT", lineup=lineup, starter=righty)
        defense_left = TeamContext(team_code="LFT", lineup=lineup, starter=lefty)

        simulator = MLBGameSimulator(MLBGameConfig(n_simulations=500, random_seed=23))
        vs_right = simulator.simulate_game(away=offense, home=defense_right)
        simulator = MLBGameSimulator(MLBGameConfig(n_simulations=500, random_seed=23))
        vs_left = simulator.simulate_game(away=offense, home=defense_left)

        self.assertGreater(vs_right.player_props[("Split Bat", "hits")].mean, vs_left.player_props[("Split Bat", "hits")].mean)


if __name__ == "__main__":
    unittest.main()
