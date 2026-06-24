"""Structured audit log backed by SQLite.

Every attribution decision — submission, appeal — is persisted as a
row in the `audit_log` table. Entries are linked via `content_id` so
the appeal for a given submission can be retrieved alongside the
original classification.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


class AuditLog:
    """Thin wrapper around a single SQLite file.

    The schema is deliberately simple: every event is one row with a
    JSON `payload` column for the full event record. The structured
    columns (`content_id`, `event`, `timestamp`, `status`) duplicate
    fields from the payload so the most common filters are indexable.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'classified',
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_content ON audit_log(content_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status)"
            )

    def append(
        self,
        *,
        content_id: str,
        event: str,
        payload: dict[str, Any],
        status: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Append a new entry and return the canonical record."""
        ts = timestamp or _utc_now_iso()
        record_status = status if status is not None else payload.get("status", "classified")
        record = {
            "content_id": content_id,
            "event": event,
            "timestamp": ts,
            "status": record_status,
            **payload,
        }
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (content_id, event, timestamp, status, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content_id, event, ts, record_status, json.dumps(record)),
            )
        return record

    def update_status(self, content_id: str, new_status: str) -> int:
        """Update `status` on every existing row for content_id. Returns row count."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT id, payload FROM audit_log WHERE content_id = ? ORDER BY id ASC",
                (content_id,),
            )
            rows = cursor.fetchall()
            for row in rows:
                payload = json.loads(row["payload"])
                if payload.get("status") == new_status:
                    continue
                payload["status"] = new_status
                conn.execute(
                    "UPDATE audit_log SET status = ?, payload = ? WHERE id = ?",
                    (new_status, json.dumps(payload), row["id"]),
                )
            return len(rows)

    def get_entries(
        self,
        *,
        limit: int = 100,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent log entries as structured JSON."""
        with self._conn() as conn:
            if status:
                cursor = conn.execute(
                    """
                    SELECT payload FROM audit_log
                    WHERE status = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT payload FROM audit_log ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            return [json.loads(row["payload"]) for row in cursor.fetchall()]

    def get_by_content_id(self, content_id: str) -> list[dict[str, Any]]:
        """Return all entries for a given content_id, oldest first."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT payload FROM audit_log WHERE content_id = ? ORDER BY id ASC",
                (content_id,),
            )
            return [json.loads(row["payload"]) for row in cursor.fetchall()]


_log: AuditLog | None = None


def init_log(db_path: str) -> AuditLog:
    global _log
    _log = AuditLog(db_path)
    return _log


def get_log() -> AuditLog:
    if _log is None:
        raise RuntimeError("Audit log not initialized; call init_log() first.")
    return _log


__all__ = ["AuditLog", "init_log", "get_log"]
