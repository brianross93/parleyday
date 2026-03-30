import unittest
from unittest.mock import Mock, patch

from refresh_slate import (
    fetch_nba_injury_context_details,
    latest_nba_injury_report_pdf_details,
    probe_nba_injury_report_pdf_url,
)


class RefreshResilienceTests(unittest.TestCase):
    @patch("refresh_slate.requests.get")
    def test_latest_nba_injury_report_pdf_details_falls_back_to_prior_report(self, mock_get: Mock) -> None:
        mock_get.return_value.text = """
        https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-03-29_08_00PM.pdf
        https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-03-29_01_00PM.pdf
        """

        details = latest_nba_injury_report_pdf_details("2026-03-30")

        self.assertIsNotNone(details)
        self.assertEqual(details["report_date"], "2026-03-29")
        self.assertTrue(details["is_stale"])
        self.assertIn("08_00PM", details["report_url"])

    @patch("refresh_slate.requests.head")
    def test_probe_nba_injury_report_pdf_url_respects_attempt_limit(self, mock_head: Mock) -> None:
        mock_head.side_effect = RuntimeError("timeout")

        result = probe_nba_injury_report_pdf_url("2026-03-27", max_attempts=3)

        self.assertIsNone(result)
        self.assertEqual(mock_head.call_count, 3)

    @patch("refresh_slate.parse_nba_injury_report")
    @patch("refresh_slate.extract_pdf_text")
    @patch("refresh_slate.latest_nba_injury_report_pdf_details")
    def test_fetch_nba_injury_context_details_reports_match_status(
        self,
        mock_latest: Mock,
        mock_extract: Mock,
        mock_parse: Mock,
    ) -> None:
        mock_latest.return_value = {
            "report_url": "https://example.com/report.pdf",
            "report_date": "2026-03-30",
            "report_time": "01:30PM",
            "is_stale": False,
        }
        mock_extract.return_value = "pdf text"
        mock_parse.return_value = {
            "DET@OKC": {
                "availability": {
                    "away": {"submitted": True, "entries": []},
                    "home": {"submitted": False, "entries": []},
                }
            }
        }

        details = fetch_nba_injury_context_details("2026-03-30", [{"matchup": "DET@OKC"}])

        self.assertEqual(details["status"], "ok")
        self.assertEqual(details["report_url"], "https://example.com/report.pdf")
        self.assertEqual(details["matched_matchups"], 1)
        self.assertEqual(details["submitted_teams"], 1)
        self.assertIn("parsed", details)

    @patch("refresh_slate.parse_nba_injury_report")
    @patch("refresh_slate.extract_pdf_text")
    @patch("refresh_slate.latest_nba_injury_report_pdf_details")
    def test_fetch_nba_injury_context_details_marks_stale_fallback(
        self,
        mock_latest: Mock,
        mock_extract: Mock,
        mock_parse: Mock,
    ) -> None:
        mock_latest.return_value = {
            "report_url": "https://example.com/report.pdf",
            "report_date": "2026-03-29",
            "report_time": "08:00PM",
            "is_stale": True,
        }
        mock_extract.return_value = "pdf text"
        mock_parse.return_value = {
            "DET@OKC": {
                "availability": {
                    "away": {"submitted": True, "entries": []},
                    "home": {"submitted": True, "entries": []},
                }
            }
        }

        details = fetch_nba_injury_context_details("2026-03-30", [{"matchup": "DET@OKC"}])

        self.assertEqual(details["status"], "stale_ok")
        self.assertTrue(details["is_stale"])
        self.assertEqual(details["report_date"], "2026-03-29")
        self.assertIn("latest prior NBA injury report", details["message"])


if __name__ == "__main__":
    unittest.main()
