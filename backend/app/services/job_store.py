"""
BharatSR — Job/Result Store (SQLite Prototype)
Lightweight storage for asynchronous inference tracking and result caching.

NOTE: This SQLite + background worker mechanism is designed as a prototype/hackathon
deployment architecture for single-node evaluation. Production deployment would use
Celery/Redis or cloud message queues.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any


class JobStore:
    """
    SQLite-backed job and result storage with concurrency locking,
    progress tracking, cancellation, and cleanup.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables and apply incremental schema migrations if needed."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    model_id TEXT,
                    progress_pct INTEGER DEFAULT 0,
                    is_cancelled INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    result_path TEXT,
                    metrics_json TEXT,
                    error_message TEXT,
                    inference_time_s REAL
                )
            """)
            conn.commit()

            # Ensure columns exist if migrating from earlier schema
            cursor = conn.execute("PRAGMA table_info(jobs)")
            cols = [col["name"] for col in cursor.fetchall()]
            if "progress_pct" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN progress_pct INTEGER DEFAULT 0")
            if "is_cancelled" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN is_cancelled INTEGER DEFAULT 0")
            conn.commit()
            conn.close()

    def create_job(self, model_id: str = "rcan") -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO jobs (job_id, status, model_id, progress_pct, is_cancelled, created_at)
                   VALUES (?, ?, ?, 0, 0, ?)""",
                (job_id, "pending", model_id, datetime.utcnow().isoformat())
            )
            conn.commit()
            conn.close()
        return job_id

    def update_progress(self, job_id: str, progress_pct: int):
        """Update job progress percentage (0-100)."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE jobs SET progress_pct=? WHERE job_id=?",
                (int(progress_pct), job_id)
            )
            conn.commit()
            conn.close()

    def update_job(self, job_id: str, status: str, result_path: str = None,
                   metrics: dict = None, error: str = None,
                   inference_time: float = None, progress_pct: int = 100):
        """Update job status and final results."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE jobs SET status=?, completed_at=?, result_path=?,
                   metrics_json=?, error_message=?, inference_time_s=?, progress_pct=?
                   WHERE job_id=?""",
                (status, datetime.utcnow().isoformat(), result_path,
                 json.dumps(metrics) if metrics else None, error,
                 inference_time, progress_pct, job_id)
            )
            conn.commit()
            conn.close()

    def cancel_job(self, job_id: str) -> bool:
        """Mark a job as cancelled if not already completed."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["status"] in ("completed", "failed"):
                conn.close()
                return False
            conn.execute(
                "UPDATE jobs SET status='cancelled', is_cancelled=1, completed_at=? WHERE job_id=?",
                (datetime.utcnow().isoformat(), job_id)
            )
            conn.commit()
            conn.close()
            return True

    def is_job_cancelled(self, job_id: str) -> bool:
        """Check if job cancellation was requested."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT is_cancelled FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            conn.close()
            return bool(row and row["is_cancelled"])

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job details."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            conn.close()
        if row is None:
            return None
        result = dict(row)
        result["is_cancelled"] = bool(result.get("is_cancelled", 0))
        if result.get("metrics_json"):
            try:
                result["metrics"] = json.loads(result["metrics_json"])
            except Exception:
                result["metrics"] = None
        else:
            result["metrics"] = None
        return result

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List most recent jobs."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT job_id, status, model_id, progress_pct, is_cancelled,
                          created_at, completed_at, inference_time_s, error_message
                   FROM jobs ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]

    def cleanup_expired_jobs(self, max_age_hours: int = 24, runs_dir: Optional[Path] = None) -> int:
        """
        Delete jobs older than max_age_hours and cleanup associated payload files.
        Returns number of deleted job records.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT job_id, result_path FROM jobs WHERE created_at < ?", (cutoff,)).fetchall()
            deleted_count = len(rows)
            for row in rows:
                # Remove result files if exist
                if row["result_path"]:
                    p = Path(row["result_path"])
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass
                if runs_dir:
                    npz_file = runs_dir / f"{row['job_id']}.npz"
                    if npz_file.exists():
                        try:
                            npz_file.unlink()
                        except Exception:
                            pass
            conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
            conn.commit()
            conn.close()
        return deleted_count
