from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "parleyday.sqlite")


class SnapshotStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    is_volatile INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    UNIQUE(source, sport, entity_type, entity_key, as_of_date)
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_lookup
                ON snapshots (sport, entity_type, entity_key, as_of_date);
                """
            )

    def upsert_snapshot(
        self,
        *,
        source: str,
        sport: str,
        entity_type: str,
        entity_key: str,
        as_of_date: str,
        payload: Any,
        is_volatile: bool = False,
        fetched_at: str | None = None,
    ) -> None:
        timestamp = fetched_at or datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO snapshots (
                    source, sport, entity_type, entity_key, as_of_date, fetched_at, is_volatile, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, sport, entity_type, entity_key, as_of_date)
                DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    is_volatile = excluded.is_volatile,
                    payload_json = excluded.payload_json
                """,
                (
                    source,
                    sport,
                    entity_type,
                    entity_key,
                    as_of_date,
                    timestamp,
                    int(is_volatile),
                    encoded,
                ),
            )

    def get_snapshot(
        self,
        *,
        source: str,
        sport: str,
        entity_type: str,
        entity_key: str,
        as_of_date: str,
        max_age_hours: float | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT source, sport, entity_type, entity_key, as_of_date, fetched_at, is_volatile, payload_json
                FROM snapshots
                WHERE source = ? AND sport = ? AND entity_type = ? AND entity_key = ? AND as_of_date = ?
                """,
                (source, sport, entity_type, entity_key, as_of_date),
            ).fetchone()
        if row is None:
            return None
        fetched_at = datetime.fromisoformat(row["fetched_at"])
        if max_age_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            if fetched_at < cutoff:
                return None
        return {
            "source": row["source"],
            "sport": row["sport"],
            "entity_type": row["entity_type"],
            "entity_key": row["entity_key"],
            "as_of_date": row["as_of_date"],
            "fetched_at": row["fetched_at"],
            "is_volatile": bool(row["is_volatile"]),
            "payload": json.loads(row["payload_json"]),
        }

    def list_snapshots(self, as_of_date: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT source, sport, entity_type, entity_key, as_of_date, fetched_at, is_volatile
            FROM snapshots
        """
        params: tuple[Any, ...] = ()
        if as_of_date is not None:
            query += " WHERE as_of_date = ?"
            params = (as_of_date,)
        query += " ORDER BY as_of_date DESC, sport, entity_type, entity_key"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
