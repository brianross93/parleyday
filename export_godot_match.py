from __future__ import annotations

import json
from pathlib import Path
from datetime import date, timedelta

from basketball_viewer import build_possession_view_payload
from refresh_slate import fetch_nba_game_contexts


DEFAULT_OUT = Path("/Users/brianross/Desktop/basketball-sim-godot/data/latest_game.json")


def _default_live_game() -> tuple[str, str] | None:
    for offset in range(0, 3):
        slate_date = (date.today() - timedelta(days=offset)).isoformat()
        try:
            games = fetch_nba_game_contexts(slate_date)
        except Exception:
            continue
        if games:
            return slate_date, str(games[0]["matchup"])
    return None


def export_match_payload(out_path: Path = DEFAULT_OUT, *, date_str: str | None = None, matchup: str | None = None) -> Path:
    live_game = (date_str, matchup) if date_str and matchup else _default_live_game()
    if live_game is not None:
        payload = build_possession_view_payload(
            view_mode="game",
            data_mode="live",
            date=live_game[0],
            matchup=live_game[1],
            csv_path="",
            seed=7,
            play_family="high_pick_and_roll",
            coverage="drop",
            entry_type="normal",
            entry_source="dead_ball",
            offense_team=live_game[1].split("@", 1)[1],
        )
    else:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    target = export_match_payload()
    print(target)
