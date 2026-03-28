import os
import tempfile
import unittest
from unittest.mock import patch

from data_pipeline.cache import SnapshotStore
from quantum_parlay_oracle import load_legs


class CachedLegFallbackTests(unittest.TestCase):
    def test_load_legs_uses_cached_recognized_legs_when_live_fetch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cache.sqlite")
            store = SnapshotStore(db_path)
            store.upsert_snapshot(
                source="kalshi",
                sport="nba",
                entity_type="recognized_legs",
                entity_key="daily",
                as_of_date="2026-03-28",
                payload={
                    "legs": [
                        {
                            "id": 0,
                            "label": "MIL ML",
                            "category": "ml",
                            "game": "SA@MIL",
                            "implied_prob": 0.62,
                            "notes": "cached",
                            "sport": "nba",
                        }
                    ],
                    "meta": {"games": 6, "kalshi_markets": 10},
                },
                is_volatile=True,
            )
            with patch.dict(os.environ, {"PARLEYDAY_DB_PATH": db_path}, clear=False):
                with patch("quantum_parlay_oracle.load_live_legs", side_effect=RuntimeError("boom")):
                    legs, resolved_mode, meta = load_legs("2026-03-28", "auto", ["nba"], 1)

        self.assertEqual(resolved_mode, "cached")
        self.assertEqual(len(legs), 1)
        self.assertEqual(legs[0].label, "MIL ML")
        self.assertEqual(meta["source"], "cache")


if __name__ == "__main__":
    unittest.main()
