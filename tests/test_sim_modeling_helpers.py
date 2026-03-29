import unittest

from quantum_parlay_oracle import (
    calibrate_mlb_offense_factor,
    direct_activation_and_coactivation,
    leg_moneyline_side,
    nba_stat_dispersion,
    parse_mlb_prop_label,
    parse_nba_prop_label,
    partial_state_score,
    project_nba_player_means,
    run_oracle,
    sample_nba_stat_over_probability,
    simulate_live_mlb_leg_probabilities,
    summarize_from_scores,
    summarize_results,
)
import numpy as np
from quantum_parlay_oracle import Leg, QuantumEntropySource, StaticEntropySource
from unittest.mock import patch


class SimModelingHelperTests(unittest.TestCase):
    def test_parse_prop_labels(self) -> None:
        self.assertEqual(parse_mlb_prop_label("Aaron Judge O 2 H"), ("Aaron Judge", "hits", 1.5))
        self.assertEqual(parse_mlb_prop_label("Paul Skenes O 8 K"), ("Paul Skenes", "strikeouts", 7.5))
        self.assertEqual(parse_nba_prop_label("Jayson Tatum O 27 PTS"), ("Jayson Tatum", "points", 26.5))
        self.assertEqual(parse_nba_prop_label("Nikola Jokic O 11 REB"), ("Nikola Jokic", "rebounds", 10.5))

    def test_moneyline_side_matches_full_team_labels(self) -> None:
        self.assertEqual(
            leg_moneyline_side(Leg(0, "Chicago ML", "ml", "CHI@MEM", 0.63, "Chicago Bulls at Memphis Grizzlies", "nba")),
            "CHI",
        )
        self.assertEqual(
            leg_moneyline_side(Leg(1, "Phoenix ML", "ml", "UTA@PHX", 0.92, "Utah Jazz at Phoenix Suns", "nba")),
            "PHX",
        )

    def test_calibrate_mlb_offense_factor_is_damped_and_clipped(self) -> None:
        self.assertAlmostEqual(calibrate_mlb_offense_factor(4.5, 4.5), 1.0)
        self.assertLess(calibrate_mlb_offense_factor(2.0, 8.0), 0.75)
        self.assertGreater(calibrate_mlb_offense_factor(8.0, 2.0), 1.0)
        self.assertLessEqual(calibrate_mlb_offense_factor(20.0, 1.0), 1.35)

    def test_project_nba_player_means_redistributes_usage_from_out_players(self) -> None:
        profiles = [
            {"name": "Star Player", "minutes": 35.0, "points": 30.0, "rebounds": 8.0, "assists": 7.0},
            {"name": "Starter Wing", "minutes": 32.0, "points": 18.0, "rebounds": 5.0, "assists": 4.0},
            {"name": "Bench Guard", "minutes": 18.0, "points": 8.0, "rebounds": 2.0, "assists": 3.0},
            {"name": "Center", "minutes": 30.0, "points": 14.0, "rebounds": 10.0, "assists": 2.0},
            {"name": "Forward", "minutes": 28.0, "points": 12.0, "rebounds": 6.0, "assists": 2.0},
        ]
        healthy = project_nba_player_means(profiles, team_total=112.0, availability_entries=[])
        with_star_out = project_nba_player_means(
            profiles,
            team_total=112.0,
            availability_entries=[{"player_name": "Star Player", "status": "Out"}],
        )

        self.assertNotIn("Star Player", with_star_out)
        self.assertGreater(with_star_out["Starter Wing"]["points"], healthy["Starter Wing"]["points"])
        self.assertGreater(with_star_out["Bench Guard"]["minutes"], healthy["Bench Guard"]["minutes"])
        self.assertLessEqual(sum(player["minutes"] for player in with_star_out.values()), 240.0 + 1e-6)

    def test_nba_dispersion_reflects_uncertainty(self) -> None:
        stable = nba_stat_dispersion(
            stat="points",
            mean=24.0,
            minutes=35.0,
            games_sample=8.0,
            status="active",
        )
        uncertain = nba_stat_dispersion(
            stat="points",
            mean=24.0,
            minutes=22.0,
            games_sample=2.0,
            status="questionable",
        )
        self.assertGreater(uncertain, stable)

    def test_nba_over_probability_gains_tail_with_more_volatility(self) -> None:
        seed = 7
        low_vol = sample_nba_stat_over_probability(
            mean=28.0,
            line=31.5,
            stat="points",
            minutes=36.0,
            games_sample=10.0,
            status="active",
            rng=np.random.default_rng(seed),
            n_samples=4000,
        )
        high_vol = sample_nba_stat_over_probability(
            mean=28.0,
            line=31.5,
            stat="points",
            minutes=20.0,
            games_sample=2.0,
            status="questionable",
            rng=np.random.default_rng(seed),
            n_samples=4000,
        )
        self.assertGreater(high_vol, low_vol)

    def test_partial_state_score_handles_empty_future_frontier(self) -> None:
        legs = [Leg(0, "Team A ML", "ml", "AAA@BBB", 0.58, "notes", "mlb")]
        activation = np.array([0.6], dtype=np.float64)
        co_activation = np.array([[0.36]], dtype=np.float64)

        score = partial_state_score(
            parlay=[0],
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            available_sports={"mlb"},
            target_size=2,
        )

        self.assertTrue(np.isfinite(score))

    def test_summarize_results_handles_zero_samples(self) -> None:
        legs = [Leg(0, "Team A ML", "ml", "AAA@BBB", 0.58, "notes", "mlb")]
        samples = np.zeros((0, 1), dtype=np.float64)

        result = summarize_results(
            legs=legs,
            samples=samples,
            qrng=QuantumEntropySource(n_bytes=1024, fallback=True),
            slate_mode="cached",
            loader_meta={"target_date": "2026-03-28", "games": 1, "pricing_summary": {"market": 1}},
        )

        self.assertEqual(result["summary"]["samples_collected"], 0)
        self.assertEqual(result["top_legs"], [])
        self.assertEqual(result["tier_parlays"], [])

    def test_direct_activation_handles_empty_legs(self) -> None:
        activation, co_activation, pricing = direct_activation_and_coactivation([], "implied")
        self.assertEqual(activation.shape, (0,))
        self.assertEqual(co_activation.shape, (0, 0))
        self.assertEqual(pricing, {})

    def test_summarize_from_scores_handles_empty_legs(self) -> None:
        result = summarize_from_scores(
            legs=[],
            activation=np.zeros(0, dtype=np.float64),
            co_activation=np.zeros((0, 0), dtype=np.float64),
            entropy_source=StaticEntropySource("Direct market scoring"),
            slate_mode="cached",
            loader_meta={"games": 0, "recognized_legs": 0},
        )
        self.assertEqual(result["top_legs"], [])
        self.assertEqual(result["tier_parlays"], [])

    def test_run_oracle_handles_empty_leg_slate(self) -> None:
        with patch("quantum_parlay_oracle.load_legs", return_value=([], "cached", {"games": 0, "recognized_legs": 0})):
            result = run_oracle(date_str="2026-03-29", sport="both", slate_mode="auto", score_source="sim")
        self.assertEqual(result["meta"]["entropy_source"], "No recognized legs")
        self.assertEqual(result["top_legs"], [])

    def test_live_mlb_sim_falls_back_when_lineups_are_missing(self) -> None:
        with patch(
            "quantum_parlay_oracle.load_matchup_profile_snapshot",
            return_value={"away_lineup": [], "home_lineup": []},
        ):
            probabilities, reason = simulate_live_mlb_leg_probabilities(
                "CHC@STL",
                "2026-03-29",
                [Leg(0, "Pete Crow-Armstrong O 1 H", "prop", "CHC@STL", 0.5, "notes", "mlb")],
            )
        self.assertEqual(probabilities, {})
        self.assertEqual(reason, "Lineup not confirmed")


if __name__ == "__main__":
    unittest.main()
