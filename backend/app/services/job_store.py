"""
BharatSR — Job/Result Store (SQLite)
Lightweight storage for async job tracking and result caching.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class JobStore:
    """SQLite-backed job and result storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                model_id TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                result_path TEXT,
                metrics_json TEXT,
                error_message TEXT,
                inference_time_s REAL
            )
        """)
        conn.commit()
        conn.close()

    def create_job(self, model_id: str = "srcnn") -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())[:8]
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO jobs (job_id, status, model_id, created_at) VALUES (?, ?, ?, ?)",
            (job_id, "pending", model_id, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        return job_id

    def update_job(self, job_id: str, status: str, result_path: str = None,
                   metrics: dict = None, error: str = None,
                   inference_time: float = None):
        """Update job status and results."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """UPDATE jobs SET status=?, completed_at=?, result_path=?,
               metrics_json=?, error_message=?, inference_time_s=?
               WHERE job_id=?""",
            (status, datetime.utcnow().isoformat(), result_path,
             json.dumps(metrics) if metrics else None, error,
             inference_time, job_id)
        )
        conn.commit()
        conn.close()

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        conn.close()
        if row is None:
            return None
        result = dict(row)
        if result.get("metrics_json"):
            result["metrics"] = json.loads(result["metrics_json"])
        else:
            result["metrics"] = None
        return result
