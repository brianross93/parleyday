import unittest

from monte_carlo.nba import NBAGameConfig, NBAGameSimulator, NBAPlayerProfile, NBATeamContext
from quantum_parlay_oracle import (
    build_live_nba_team_context,
    build_tiered_parlays,
    calibrate_mlb_offense_factor,
    direct_activation_and_coactivation,
    fetch_live_nba_team_form,
    leg_moneyline_side,
    nba_team_form_is_usable,
    nba_stat_dispersion,
    parse_mlb_prop_label,
    parse_nba_prop_label,
    partial_state_score,
    project_nba_player_means,
    run_oracle,
    sample_nba_stat_over_probability,
    simulate_live_mlb_leg_probabilities,
    simulate_live_nba_leg_probabilities,
    structured_threshold_label,
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

    def test_structured_threshold_label_handles_integer_floor_strike(self) -> None:
        market = {
            "title": "Draymond Green: rebounds?",
            "yes_sub_title": "Draymond Green",
            "floor_strike": 2.0,
            "rules_primary": "",
        }
        self.assertEqual(structured_threshold_label(market, "REB", "Rebounds"), "Draymond Green O 3 REB")

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

    def test_build_live_nba_team_context_uses_projected_minutes(self) -> None:
        context = build_live_nba_team_context(
            "MIL",
            [
                {"name": "Guard One", "minutes": 34.0, "points": 24.0, "rebounds": 5.0, "assists": 7.0, "games_sample": 8.0},
                {"name": "Wing Two", "minutes": 32.0, "points": 18.0, "rebounds": 6.0, "assists": 4.0, "games_sample": 8.0},
                {"name": "Big Three", "minutes": 30.0, "points": 16.0, "rebounds": 10.0, "assists": 2.0, "games_sample": 8.0},
            ],
            114.0,
            [{"player_name": "Guard One", "status": "Questionable"}],
        )
        self.assertIsNotNone(context)
        self.assertEqual(context.code, "MIL")
        self.assertAlmostEqual(context.expected_points, 114.0)
        self.assertTrue(any(player.minutes > 0 for player in context.players))

    def test_nba_possession_simulator_generates_player_distributions(self) -> None:
        away = NBATeamContext(
            code="LAC",
            expected_points=113.0,
            players=[
                NBAPlayerProfile("Guard A", 35.0, 26.0, 5.0, 7.0, games_sample=8.0),
                NBAPlayerProfile("Wing A", 34.0, 20.0, 6.0, 4.0, games_sample=8.0),
                NBAPlayerProfile("Big A", 32.0, 18.0, 11.0, 3.0, games_sample=8.0),
                NBAPlayerProfile("Guard B", 28.0, 12.0, 3.0, 5.0, games_sample=8.0),
                NBAPlayerProfile("Wing B", 26.0, 10.0, 5.0, 2.0, games_sample=8.0),
            ],
        )
        home = NBATeamContext(
            code="MIL",
            expected_points=111.0,
            players=[
                NBAPlayerProfile("Guard C", 36.0, 27.0, 4.0, 8.0, games_sample=8.0),
                NBAPlayerProfile("Wing C", 33.0, 19.0, 7.0, 4.0, games_sample=8.0),
                NBAPlayerProfile("Big C", 31.0, 17.0, 10.0, 3.0, games_sample=8.0),
                NBAPlayerProfile("Guard D", 28.0, 11.0, 4.0, 4.0, games_sample=8.0),
                NBAPlayerProfile("Wing D", 25.0, 9.0, 5.0, 2.0, games_sample=8.0),
            ],
        )
        simulator = NBAGameSimulator(NBAGameConfig(n_simulations=200, random_seed=11))
        result = simulator.simulate_game(
            away=away,
            home=home,
            tracked_props={("Guard A", "points"), ("Big C", "rebounds"), ("Guard C", "assists")},
        )
        self.assertEqual(len(result.away_scores), 200)
        self.assertIn(("Guard A", "points"), result.player_props)
        self.assertGreater(result.player_props[("Guard A", "points")].mean, 10.0)
        self.assertGreater(result.player_props[("Big C", "rebounds")].mean, 4.0)
        self.assertGreater(result.player_props[("Guard C", "assists")].mean, 2.0)

    def test_live_nba_sim_skips_players_not_in_matchup(self) -> None:
        with patch(
            "quantum_parlay_oracle.load_nba_matchup_profile_snapshot",
            return_value={
                "away_profiles": [
                    {"name": "Guard A", "minutes": 35.0, "points": 26.0, "rebounds": 5.0, "assists": 7.0, "games_sample": 8.0}
                ],
                "home_profiles": [
                    {"name": "Guard B", "minutes": 34.0, "points": 24.0, "rebounds": 4.0, "assists": 8.0, "games_sample": 8.0}
                ],
            },
        ), patch("quantum_parlay_oracle.load_game_context_snapshot", return_value={"availability": {"away": [], "home": []}}), patch(
            "quantum_parlay_oracle.load_team_form_snapshot",
            return_value={"LAC": {"net_rating_proxy": 1.0}, "MIL": {"net_rating_proxy": 0.0}},
        ):
            probabilities = simulate_live_nba_leg_probabilities(
                "LAC@MIL",
                "2026-03-29",
                [Leg(0, "Someone Else O 5 AST", "prop", "LAC@MIL", 0.5, "notes", "nba")],
            )
        self.assertEqual(probabilities, {})

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

    def test_fetch_live_nba_team_form_uses_date_specific_scoreboard_records(self) -> None:
        mock_games = [
            type(
                "Game",
                (),
                {
                    "away_code": "LAC",
                    "home_code": "MIL",
                    "away_detail": "42-30",
                    "home_detail": "38-34",
                },
            )()
        ]
        with patch("quantum_parlay_oracle.fetch_nba_schedule", return_value=mock_games):
            form = fetch_live_nba_team_form("2026-03-29")
        self.assertEqual(form["LAC"]["games_played"], 72)
        self.assertGreater(form["LAC"]["net_rating_proxy"], 0.0)
        self.assertEqual(form["MIL"]["games_played"], 72)

    def test_nba_team_form_is_usable_rejects_placeholder_snapshot(self) -> None:
        self.assertFalse(
            nba_team_form_is_usable(
                {
                    "LAC": {"games_played": 1, "win_pct": 0.0, "net_rating_proxy": -12.0},
                    "MIL": {"games_played": 1, "win_pct": 0.0, "net_rating_proxy": -12.0},
                }
            )
        )
        self.assertTrue(
            nba_team_form_is_usable(
                {
                    "LAC": {"games_played": 74, "win_pct": 0.51, "net_rating_proxy": 0.3},
                    "MIL": {"games_played": 73, "win_pct": 0.39, "net_rating_proxy": -2.4},
                }
            )
        )

    def test_tiered_parlays_require_positive_edge_and_target_payout(self) -> None:
        legs = [
            Leg(0, "A ML", "ml", "A@B", 0.78, "notes", "nba"),
            Leg(1, "C ML", "ml", "C@D", 0.76, "notes", "nba"),
            Leg(2, "E ML", "ml", "E@F", 0.74, "notes", "nba"),
            Leg(3, "G ML", "ml", "G@H", 0.52, "notes", "nba"),
            Leg(4, "I ML", "ml", "I@J", 0.50, "notes", "nba"),
            Leg(5, "K ML", "ml", "K@L", 0.48, "notes", "nba"),
            Leg(6, "M ML", "ml", "M@N", 0.46, "notes", "nba"),
            Leg(7, "O ML", "ml", "O@P", 0.44, "notes", "nba"),
        ]
        activation = np.array([0.81, 0.80, 0.77, 0.60, 0.57, 0.53, 0.50, 0.47], dtype=np.float64)
        co_activation = np.outer(activation, activation)
        np.fill_diagonal(co_activation, activation)
        pricing_details = {
            idx: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"} for idx in range(len(legs))
        }

        tiers = build_tiered_parlays(
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            pricing_details=pricing_details,
        )

        cash = next(tier for tier in tiers if tier["key"] == "cash")
        self.assertTrue(cash["legs"])
        self.assertTrue(all(leg["score_delta"] > 0 for leg in cash["legs"]))
        self.assertGreaterEqual(cash["payout_estimate"], cash["target_payout_min"])
        self.assertLessEqual(cash["payout_estimate"], cash["target_payout_max"])

    def test_cash_tier_avoids_same_game_stacks(self) -> None:
        legs = [
            Leg(0, "AAA ML", "ml", "AAA@BBB", 0.72, "notes", "mlb"),
            Leg(1, "AAA@BBB O5.5", "total", "AAA@BBB", 0.72, "notes", "mlb"),
            Leg(2, "CCC ML", "ml", "CCC@DDD", 0.74, "notes", "mlb"),
            Leg(3, "EEE ML", "ml", "EEE@FFF", 0.73, "notes", "mlb"),
            Leg(4, "GGG ML", "ml", "GGG@HHH", 0.71, "notes", "mlb"),
        ]
        activation = np.array([0.84, 0.83, 0.82, 0.81, 0.79], dtype=np.float64)
        co_activation = np.outer(activation, activation)
        np.fill_diagonal(co_activation, activation)
        pricing_details = {
            idx: {"pricing_source": "simulation", "pricing_label": "Monte Carlo"} for idx in range(len(legs))
        }
        tiers = build_tiered_parlays(
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            pricing_details=pricing_details,
        )
        cash = next(tier for tier in tiers if tier["key"] == "cash")
        games = [leg["game"] for leg in cash["legs"]]
        self.assertEqual(len(games), len(set(games)))


if __name__ == "__main__":
    unittest.main()
