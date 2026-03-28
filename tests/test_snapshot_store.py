import os
import tempfile
import unittest

from data_pipeline.cache import SnapshotStore


class SnapshotStoreTests(unittest.TestCase):
    def test_upsert_and_get_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cache.sqlite")
            store = SnapshotStore(db_path)
            store.upsert_snapshot(
                source="test",
                sport="mlb",
                entity_type="team_form",
                entity_key="daily",
                as_of_date="2026-03-28",
                payload={"NYY": {"win_pct": 0.6}},
            )

            snapshot = store.get_snapshot(
                source="test",
                sport="mlb",
                entity_type="team_form",
                entity_key="daily",
                as_of_date="2026-03-28",
            )

            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot["payload"]["NYY"]["win_pct"], 0.6)


if __name__ == "__main__":
    unittest.main()
