"""SQLite persistence for inference monitoring."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB_PATH = Path("inference_monitoring.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inference_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    filename TEXT NOT NULL,
    prediction TEXT NOT NULL,
    probability REAL NOT NULL,
    latency_ms REAL NOT NULL,
    model_version TEXT NOT NULL,
    endpoint TEXT NOT NULL DEFAULT 'predict'
);
"""


@dataclass(frozen=True)
class InferenceRecord:
    """Persisted inference event for monitoring."""

    id: int
    timestamp: str
    filename: str
    prediction: str
    probability: float
    latency_ms: float
    model_version: str
    endpoint: str = "predict"


class InferenceStore:
    """Persist and query inference events."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_CREATE_TABLE_SQL)
            connection.commit()

    def save(
        self,
        *,
        filename: str,
        prediction: str,
        probability: float,
        latency_ms: float,
        model_version: str,
        endpoint: str = "predict",
        timestamp: str | None = None,
    ) -> int:
        """Persist one inference and return its row id."""
        event_time = timestamp or datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO inference_logs (
                    timestamp, filename, prediction, probability,
                    latency_ms, model_version, endpoint
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_time,
                    filename,
                    prediction,
                    probability,
                    latency_ms,
                    model_version,
                    endpoint,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_recent(self, limit: int = 100) -> list[InferenceRecord]:
        """Return the most recent inference events."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, filename, prediction, probability,
                       latency_ms, model_version, endpoint
                FROM inference_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            InferenceRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                filename=row["filename"],
                prediction=row["prediction"],
                probability=row["probability"],
                latency_ms=row["latency_ms"],
                model_version=row["model_version"],
                endpoint=row["endpoint"],
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM inference_logs").fetchone()
        return int(row["total"])
