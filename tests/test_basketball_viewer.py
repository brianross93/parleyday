from __future__ import annotations

import unittest

from basketball_viewer import build_possession_view_payload
from dashboard_app import app


class BasketballViewerTests(unittest.TestCase):
    def test_payload_builds_for_transition_entry(self) -> None:
        payload = build_possession_view_payload(
            view_mode="single",
            data_mode="calibration",
            date="",
            matchup="",
            csv_path="",
            seed=7,
            play_family="high_pick_and_roll",
            coverage="drop",
            entry_type="transition",
            entry_source="live_turnover_break",
            offense_team="HOM",
        )
        self.assertTrue(payload["events"])
        self.assertEqual(payload["events"][0]["event_type"], "advance")
        self.assertIn("transition_entry", payload["events"][0]["notes"] or "")
        self.assertEqual(payload["summary"]["entry_source"], "live_turnover_break")

    def test_payload_accepts_enum_names(self) -> None:
        payload = build_possession_view_payload(
            view_mode="single",
            data_mode="calibration",
            date="",
            matchup="",
            csv_path="",
            seed=7,
            play_family="HIGH_PICK_AND_ROLL",
            coverage="DROP",
            entry_type="TRANSITION",
            entry_source="LIVE_TURNOVER_BREAK",
            offense_team="HOM",
        )
        self.assertEqual(payload["form"]["play_family"], "HIGH_PICK_AND_ROLL")
        self.assertTrue(payload["events"])

    def test_payload_supports_handoff_and_ice(self) -> None:
        payload = build_possession_view_payload(
            view_mode="single",
            data_mode="calibration",
            date="",
            matchup="",
            csv_path="",
            seed=9,
            play_family="handoff",
            coverage="ice",
            entry_type="normal",
            entry_source="dead_ball",
            offense_team="HOM",
        )
        self.assertTrue(payload["events"])

    def test_payload_supports_double_drag_and_hedge(self) -> None:
        payload = build_possession_view_payload(
            view_mode="single",
            data_mode="calibration",
            date="",
            matchup="",
            csv_path="",
            seed=11,
            play_family="double_drag",
            coverage="hedge",
            entry_type="normal",
            entry_source="dead_ball",
            offense_team="HOM",
        )
        self.assertTrue(payload["events"])

    def test_game_mode_builds_beats(self) -> None:
        payload = build_possession_view_payload(
            view_mode="game",
            data_mode="calibration",
            date="",
            matchup="",
            csv_path="",
            seed=7,
            play_family="high_pick_and_roll",
            coverage="drop",
            entry_type="normal",
            entry_source="dead_ball",
            offense_team="HOM",
        )
        self.assertTrue(payload["match"]["beats"])
        self.assertEqual(payload["form"]["view_mode"], "game")

    def test_viewer_route_renders(self) -> None:
        client = app.test_client()
        response = client.get("/basketball-viewer")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Basketball Match View", response.data)


if __name__ == "__main__":
    unittest.main()
