import unittest

from refresh_slate import parse_nba_injury_report


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


if __name__ == "__main__":
    unittest.main()
