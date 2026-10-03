import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id    TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    model_version TEXT NOT NULL,
    label         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    entropy       REAL NOT NULL,
    latency_ms    REAL NOT NULL,
    features      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY REFERENCES predictions(request_id),
    ts         REAL NOT NULL,
    true_label TEXT NOT NULL
);
"""


class PredictionStore:
    """Append-only log of every prediction and of any ground-truth label that
    arrives later. SQLite keeps it a single file that survives restarts."""

    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def log_prediction(self, request_id, model_version, label, confidence, entropy, latency_ms, features):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, time.time(), model_version, label, confidence, entropy,
                 latency_ms, json.dumps(features)),
            )

    def add_feedback(self, request_id, true_label):
        """Returns the predicted label for the request, or None if the id is unknown.
        Re-sending feedback for the same request overwrites the earlier label."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT label FROM predictions WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?, ?, ?)", (request_id, time.time(), true_label))
            return row[0]

    def recent_predictions(self, limit):
        """The newest `limit` predictions, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, confidence, entropy, features FROM predictions "
                "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"label": r[0], "confidence": r[1], "entropy": r[2], "features": json.loads(r[3])}
                for r in reversed(rows)]

    def recent_labelled(self, limit):
        """(predicted, true) pairs for the newest `limit` predictions that have feedback."""
        with self._lock:
            return self._conn.execute(
                "SELECT p.label, f.true_label FROM predictions p JOIN feedback f USING (request_id) "
                "ORDER BY p.ts DESC LIMIT ?", (limit,)).fetchall()

    def counts(self):
        with self._lock:
            n_pred = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            n_fb = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {"predictions": n_pred, "labelled": n_fb}

    def close(self):
        self._conn.close()
