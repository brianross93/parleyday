import unittest

from data_pipeline.mlb_profiles import team_context_from_cached_payload
from quantum_parlay_oracle import expected_mlb_runs


class MLBContextAdjustmentTests(unittest.TestCase):
    def test_team_context_from_cached_payload_falls_back_when_pitcher_missing(self) -> None:
        context = team_context_from_cached_payload(
            "NYY",
            lineup_payload=[
                {
                    "player_id": "1",
                    "name": "Aaron Judge",
                    "hand": "R",
                    "pa_share": 0.13,
                    "strikeout_rate": 0.24,
                    "walk_rate": 0.12,
                    "hbp_rate": 0.01,
                    "single_rate": 0.15,
                    "double_rate": 0.05,
                    "triple_rate": 0.002,
                    "home_run_rate": 0.07,
                    "speed_factor": 1.0,
                    "vs_left_factor": 1.04,
                    "vs_right_factor": 1.02,
                }
            ],
            pitcher_payload=None,
        )

        self.assertEqual(context.starter.name, "NYY Starter")
        self.assertEqual(context.starter.player_id, "NYY-sp-fallback")
        self.assertAlmostEqual(context.starter.strikeout_rate, 0.225)

    def test_expected_mlb_runs_responds_to_bullpen_fatigue_and_pitcher_availability(self) -> None:
        team_form = {
            "NYY": {
                "runs_scored_pg": 4.9,
                "runs_allowed_pg": 4.1,
                "recent_win_pct": 0.6,
            },
            "BOS": {
                "runs_scored_pg": 4.4,
                "runs_allowed_pg": 4.6,
                "recent_win_pct": 0.45,
            },
        }
        base_away, base_home = expected_mlb_runs("NYY", "BOS", team_form, None)
        context = {
            "lineup_status": {"away_confirmed": True, "home_confirmed": True},
            "lineups": {"away": ["Aaron Judge"], "home": ["Rafael Devers"]},
            "probable_pitchers": {
                "away": {"fullName": "Gerrit Cole"},
                "home": {"fullName": "Tanner Houck"},
            },
            "availability": {
                "away": {"unavailable_players": [{"player_name": "Gerrit Cole"}]},
                "home": {"unavailable_players": []},
            },
            "bullpen": {
                "away": {"fatigue_score": 3.0},
                "home": {"fatigue_score": 0.0},
            },
            "weather": {},
        }
        adj_away, adj_home = expected_mlb_runs("NYY", "BOS", team_form, context)

        self.assertGreater(adj_home, base_home)
        self.assertGreaterEqual(adj_away, base_away)


if __name__ == "__main__":
    unittest.main()
