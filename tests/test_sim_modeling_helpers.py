import unittest

from quantum_parlay_oracle import (
    calibrate_mlb_offense_factor,
    nba_stat_dispersion,
    parse_mlb_prop_label,
    parse_nba_prop_label,
    project_nba_player_means,
    sample_nba_stat_over_probability,
)
import numpy as np


class SimModelingHelperTests(unittest.TestCase):
    def test_parse_prop_labels(self) -> None:
        self.assertEqual(parse_mlb_prop_label("Aaron Judge O 2 H"), ("Aaron Judge", "hits", 1.5))
        self.assertEqual(parse_mlb_prop_label("Paul Skenes O 8 K"), ("Paul Skenes", "strikeouts", 7.5))
        self.assertEqual(parse_nba_prop_label("Jayson Tatum O 27 PTS"), ("Jayson Tatum", "points", 26.5))
        self.assertEqual(parse_nba_prop_label("Nikola Jokic O 11 REB"), ("Nikola Jokic", "rebounds", 10.5))

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


if __name__ == "__main__":
    unittest.main()
