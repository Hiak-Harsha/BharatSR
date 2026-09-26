"""
BharatSR — Job/Result Store (Persistent SQLite Engine)
Thread-safe persistent storage for asynchronous inference tracking,
progress monitoring, cancellation, and TTL cleanup.

Design Decision:
Intentional single-node SQLite + ThreadPoolExecutor architecture.
Maintains a persistent connection guarded by an exclusive re-entrant lock,
eliminating connection open/close overhead while ensuring ACID durability.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.job_store")


class JobStore:
    """
    SQLite-backed job and result storage with a persistent connection,
    thread-safe concurrency locking, progress tracking, cancellation, and cleanup.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self):
        """Closes the persistent SQLite connection cleanly."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception as e:
                    logger.warning(f"Error closing SQLite connection: {e}")
                self._conn = None


    def _init_db(self):
        """Create tables and apply incremental schema migrations if needed."""
        with self._lock:
            self._conn.execute("""
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
            self._conn.commit()

            cursor = self._conn.execute("PRAGMA table_info(jobs)")
            cols = [col["name"] for col in cursor.fetchall()]
            if "progress_pct" not in cols:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN progress_pct INTEGER DEFAULT 0")
            if "is_cancelled" not in cols:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN is_cancelled INTEGER DEFAULT 0")
            self._conn.commit()

    def create_job(self, model_id: str = "rcan") -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._conn.execute(
                """INSERT INTO jobs (job_id, status, model_id, progress_pct, is_cancelled, created_at)
                   VALUES (?, ?, ?, 0, 0, ?)""",
                (job_id, "pending", model_id, datetime.utcnow().isoformat()),
            )
            self._conn.commit()
        return job_id

    def update_progress(self, job_id: str, progress_pct: int):
        """Update job progress percentage (0-100)."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET progress_pct=? WHERE job_id=?",
                (int(progress_pct), job_id),
            )
            self._conn.commit()

    def update_job(
        self,
        job_id: str,
        status: str,
        result_path: str = None,
        metrics: dict = None,
        error: str = None,
        inference_time: float = None,
        progress_pct: int = 100,
    ):
        """Update job status and final results."""
        completed_at = datetime.utcnow().isoformat() if status in ("completed", "failed", "cancelled") else None
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status=?,
                   completed_at=COALESCE(?, completed_at),
                   result_path=?,
                   metrics_json=?, error_message=?, inference_time_s=?, progress_pct=?
                   WHERE job_id=?""",
                (
                    status,
                    completed_at,
                    result_path,
                    json.dumps(metrics) if metrics else None,
                    error,
                    inference_time,
                    progress_pct,
                    job_id,
                ),
            )
            self._conn.commit()

    def cancel_job(self, job_id: str) -> bool:
        """Mark a job as cancelled if not already completed."""
        with self._lock:
            row = self._conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["status"] in ("completed", "failed"):
                return False
            self._conn.execute(
                "UPDATE jobs SET status='cancelled', is_cancelled=1, completed_at=? WHERE job_id=?",
                (datetime.utcnow().isoformat(), job_id),
            )
            self._conn.commit()
            return True

    def is_job_cancelled(self, job_id: str) -> bool:
        """Check if job cancellation was requested."""
        with self._lock:
            row = self._conn.execute("SELECT is_cancelled FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return bool(row and row["is_cancelled"])

    def is_cancelled(self, job_id: str) -> bool:
        """Alias for is_job_cancelled."""
        return self.is_job_cancelled(job_id)

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job details."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["is_cancelled"] = bool(result.get("is_cancelled", 0))
        if result.get("metrics_json"):
            try:
                result["metrics"] = json.loads(result["metrics_json"])
            except Exception as e:
                logger.warning(f"Failed to parse metrics_json for job {job_id}: {e}")
                result["metrics"] = None

        else:
            result["metrics"] = None
        return result

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List most recent jobs."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT job_id, status, model_id, progress_pct, is_cancelled,
                          created_at, completed_at, inference_time_s, error_message
                   FROM jobs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def cleanup_expired_jobs(self, max_age_hours: int = 24, runs_dir: Optional[Path] = None) -> int:
        """
        Delete jobs older than max_age_hours and cleanup associated payload files.
        Returns number of deleted job records.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id, result_path FROM jobs WHERE created_at < ?", (cutoff,)
            ).fetchall()
            deleted_count = len(rows)
            for row in rows:
                if row["result_path"]:
                    p = Path(row["result_path"])
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to unlink expired result file {p}: {e}")
                if runs_dir:
                    npz_file = runs_dir / f"{row['job_id']}.npz"
                    if npz_file.exists():
                        try:
                            npz_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to unlink expired artifact {npz_file}: {e}")

            self._conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
            self._conn.commit()
        return deleted_count
