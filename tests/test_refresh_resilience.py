import unittest
from unittest.mock import Mock, patch

from refresh_slate import probe_nba_injury_report_pdf_url


class RefreshResilienceTests(unittest.TestCase):
    @patch("refresh_slate.requests.head")
    def test_probe_nba_injury_report_pdf_url_respects_attempt_limit(self, mock_head: Mock) -> None:
        mock_head.side_effect = RuntimeError("timeout")

        result = probe_nba_injury_report_pdf_url("2026-03-27", max_attempts=3)

        self.assertIsNone(result)
        self.assertEqual(mock_head.call_count, 3)


if __name__ == "__main__":
    unittest.main()
