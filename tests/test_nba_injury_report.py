import unittest

from refresh_slate import merge_nba_availability_sources, parse_nba_injury_report


class NBAInjuryReportTests(unittest.TestCase):
    def test_parse_nba_injury_report_by_matchup_and_team(self) -> None:
        contexts = [
            {
                "matchup": "SAS@MIL",
                "away_team_name": "San Antonio Spurs",
                "home_team_name": "Milwaukee Bucks",
            },
            {
                "matchup": "DET@MIN",
                "away_team_name": "Detroit Pistons",
                "home_team_name": "Minnesota Timberwolves",
            },
        ]
        text = (
            "Injury Report: 03/28/26 01:30 PM Page 1 of 5 Game Date Game Time Matchup Team Player Name Current Status Reason "
            "03/28/2026 03:00 (ET) SAS@MIL San Antonio Spurs Ingram, Harrison Out G League - Two-Way "
            "Milwaukee Bucks Antetokounmpo, Giannis Out Injury/Illness - Left Knee; Hyperextension; Bone Bruise "
            "Turner, Myles Available Injury/Illness - N/a; Illness "
            "05:30 (ET) DET@MIN Detroit Pistons Cunningham, Cade Out Injury/Illness - Left Lung; Pneumothorax "
            "Minnesota Timberwolves Edwards, Anthony Out Injury/Illness - Right Knee; Patellofemoral Pain Syndrome"
        )

        parsed = parse_nba_injury_report(text, contexts, "https://example.com/report.pdf")

        self.assertIn("SAS@MIL", parsed)
        self.assertEqual(parsed["SAS@MIL"]["availability"]["away"]["entries"][0]["player_name"], "Ingram, Harrison")
        self.assertEqual(parsed["SAS@MIL"]["availability"]["home"]["entries"][0]["status"], "Out")
        self.assertEqual(parsed["DET@MIN"]["availability"]["home"]["entries"][0]["player_name"], "Edwards, Anthony")

    def test_merge_nba_availability_sources_uses_profile_fallback_when_pdf_missing(self) -> None:
        fallback = {
            "availability": {
                "away": {"submitted": True, "entries": [{"player_name": "Player A", "status": "Out", "reason": "Ankle"}]},
                "home": {"submitted": True, "entries": []},
            }
        }
        merged = merge_nba_availability_sources(None, fallback)
        self.assertEqual(merged["source"], "espn_team_profiles")
        self.assertTrue(merged["away_submitted"])
        self.assertEqual(merged["away"][0]["player_name"], "Player A")

    def test_merge_nba_availability_sources_uses_fallback_only_for_unsubmitted_pdf_side(self) -> None:
        parsed = {
            "report_url": "https://example.com/report.pdf",
            "availability": {
                "away": {"submitted": False, "entries": []},
                "home": {"submitted": True, "entries": [{"player_name": "Player B", "status": "Questionable", "reason": "Knee"}]},
            },
        }
        fallback = {
            "availability": {
                "away": {"submitted": True, "entries": [{"player_name": "Player A", "status": "Out", "reason": "Ankle"}]},
                "home": {"submitted": True, "entries": [{"player_name": "Player C", "status": "Out", "reason": "Rest"}]},
            }
        }
        merged = merge_nba_availability_sources(parsed, fallback)
        self.assertEqual(merged["source"], "official_nba_injury_report_pdf+espn_team_profiles")
        self.assertEqual(merged["away"][0]["player_name"], "Player A")
        self.assertEqual(merged["home"][0]["player_name"], "Player B")
        self.assertTrue(merged["away_submitted"])
        self.assertTrue(merged["home_submitted"])


if __name__ == "__main__":
    unittest.main()
