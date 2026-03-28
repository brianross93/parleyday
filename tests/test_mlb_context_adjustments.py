import unittest

from quantum_parlay_oracle import expected_mlb_runs


class MLBContextAdjustmentTests(unittest.TestCase):
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
        self.assertEqual(adj_away, base_away)


if __name__ == "__main__":
    unittest.main()
