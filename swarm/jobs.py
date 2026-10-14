"""Batch job queue: leases with fencing tokens, retries and idempotent submission.

A job is a list of tasks in the OpenAI Batch format (``custom_id``, ``url``,
``body``). Agents pull tasks for the model they serve; each lease bumps the
task's fencing token and only a completion carrying the current token is
accepted, so a result is committed exactly once even if a lease expired and
the task was handed to another node meanwhile.

    queued --lease--> leased --complete--> succeeded
                        |  \\--fail / lease expired--> queued (after backoff)
                        \\--attempts exhausted--> failed
    queued | leased --cancel--> cancelled

SQLite with one connection and a lock: every state change is a short
single-writer transaction, which is plenty for a LAN of Macs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Callable

LEASE_S = 30.0
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
ENDPOINTS = ("/v1/chat/completions", "/v1/completions")
TERMINAL = ("succeeded", "failed", "cancelled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE,
    metadata TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    custom_id TEXT NOT NULL,
    model TEXT NOT NULL,
    url TEXT NOT NULL,
    body TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    token INTEGER NOT NULL DEFAULT 0,
    owner TEXT,
    lease_expiry REAL,
    available_at REAL NOT NULL,
    result TEXT,
    error TEXT,
    finished_by TEXT,
    updated_at REAL NOT NULL,
    UNIQUE (job_id, custom_id)
);
CREATE INDEX IF NOT EXISTS tasks_queue ON tasks (state, model, available_at, id);
CREATE INDEX IF NOT EXISTS tasks_job ON tasks (job_id, id);
"""


class JobError(ValueError):
    pass


class JobStore:
    def __init__(self, path: str = ":memory:", clock: Callable[[], float] = time.time,
                 lease_s: float = LEASE_S):
        self.clock = clock
        self.lease_s = lease_s
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def _tx(self):
        return _Tx(self)

    # -- clients ------------------------------------------------------------
    def submit(self, tasks: list[dict], model: str | None = None, max_attempts: int = MAX_ATTEMPTS,
               idempotency_key: str | None = None, metadata: dict | None = None) -> tuple[str, bool]:
        """Create a job; returns (job_id, created). A repeated idempotency key returns the old job."""
        rows, seen = [], set()
        for i, t in enumerate(tasks):
            custom_id = str(t.get("custom_id") or f"task-{i}")
            if custom_id in seen:
                raise JobError(f"duplicate custom_id {custom_id!r}")
            seen.add(custom_id)
            url = t.get("url", "/v1/chat/completions")
            if url not in ENDPOINTS:
                raise JobError(f"{custom_id}: unsupported url {url!r}; use one of {ENDPOINTS}")
            body = t.get("body")
            if not isinstance(body, dict):
                raise JobError(f"{custom_id}: body must be an object")
            body = {**body, "stream": False}
            task_model = model or body.get("model")
            if not task_model:
                raise JobError(f"{custom_id}: no model given")
            rows.append((custom_id, task_model, url, json.dumps({**body, "model": task_model})))
        if not rows:
            raise JobError("a job needs at least one task")
        if max_attempts < 1:
            raise JobError("max_attempts must be >= 1")

        now = self.clock()
        with self._tx() as db:
            if idempotency_key:
                row = db.execute("SELECT id FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row:
                    return row["id"], False
            job_id = "job_" + uuid.uuid4().hex[:16]
            db.execute(
                "INSERT INTO jobs (id, idempotency_key, metadata, created_at) VALUES (?,?,?,?)",
                (job_id, idempotency_key, json.dumps(metadata or {}), now),
            )
            db.executemany(
                "INSERT INTO tasks (job_id, custom_id, model, url, body, max_attempts, available_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(job_id, *r, max_attempts, now, now) for r in rows],
            )
        return job_id, True

    def job(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                return None
            counts = dict(self.db.execute(
                "SELECT state, COUNT(*) FROM tasks WHERE job_id=? GROUP BY state", (job_id,)
            ).fetchall())
        total = sum(counts.values())
        done = sum(counts.get(s, 0) for s in TERMINAL)
        return {
            "id": job["id"],
            "object": "batch",
            "status": "completed" if done == total else "in_progress",
            "created_at": job["created_at"],
            "metadata": json.loads(job["metadata"]),
            "request_counts": {
                "total": total,
                "completed": counts.get("succeeded", 0),
                "failed": counts.get("failed", 0),
                "cancelled": counts.get("cancelled", 0),
                "queued": counts.get("queued", 0),
                "running": counts.get("leased", 0),
            },
        }

    def jobs(self, limit: int = 20) -> list[dict]:
        with self.lock:
            ids = [r[0] for r in self.db.execute(
                "SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            )]
        return [self.job(i) for i in ids]

    def results(self, job_id: str) -> list[dict]:
        """One line per task in submission order, in the OpenAI Batch output format."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id, custom_id, state, result, error, finished_by, attempts FROM tasks"
                " WHERE job_id=? ORDER BY id", (job_id,)
            ).fetchall()
        out = []
        for r in rows:
            line = {"id": f"task_{r['id']}", "custom_id": r["custom_id"], "status": r["state"],
                    "node": r["finished_by"], "attempts": r["attempts"], "response": None, "error": None}
            if r["state"] == "succeeded":
                line["response"] = {"status_code": 200, "body": json.loads(r["result"])}
            elif r["error"]:
                line["error"] = {"message": r["error"]}
            out.append(line)
        return out

    def cancel(self, job_id: str) -> int:
        """Cancel tasks not finished yet. A running task's late result is rejected by its token."""
        with self._tx() as db:
            cur = db.execute(
                "UPDATE tasks SET state='cancelled', token=token+1, owner=NULL, lease_expiry=NULL,"
                " updated_at=? WHERE job_id=? AND state IN ('queued','leased')",
                (self.clock(), job_id),
            )
            return cur.rowcount

    # -- agents -------------------------------------------------------------
    def lease(self, owner: str, model: str, limit: int = 1) -> list[dict]:
        now = self.clock()
        with self._tx() as db:
            self._reclaim(db, now)
            rows = db.execute(
                "SELECT id, url, body FROM tasks WHERE state='queued' AND model=? AND available_at<=?"
                " ORDER BY available_at, id LIMIT ?", (model, now, limit)
            ).fetchall()
            leases = []
            for r in rows:
                token = db.execute(
                    "UPDATE tasks SET state='leased', owner=?, token=token+1, attempts=attempts+1,"
                    " lease_expiry=?, updated_at=? WHERE id=? RETURNING token",
                    (owner, now + self.lease_s, now, r["id"]),
                ).fetchone()[0]
                leases.append({"task_id": r["id"], "token": token, "url": r["url"],
                               "body": json.loads(r["body"]), "lease_s": self.lease_s})
        return leases

    def renew(self, owner: str, leases: list[dict]) -> list[int]:
        """Extend the given leases; returns the task ids this owner no longer holds."""
        now = self.clock()
        lost = []
        with self._tx() as db:
            for l in leases:
                cur = db.execute(
                    "UPDATE tasks SET lease_expiry=?, updated_at=? WHERE id=? AND token=? AND owner=?"
                    " AND state='leased'",
                    (now + self.lease_s, now, l["task_id"], l["token"], owner),
                )
                if cur.rowcount == 0:
                    lost.append(l["task_id"])
        return lost

    def complete(self, task_id: int, token: int, owner: str, result: dict) -> bool:
        """Commit a result if the token is still current; a stale completion is dropped."""
        with self._tx() as db:
            cur = db.execute(
                "UPDATE tasks SET state='succeeded', result=?, error=NULL, finished_by=?, owner=NULL,"
                " lease_expiry=NULL, updated_at=? WHERE id=? AND token=? AND state='leased'",
                (json.dumps(result), owner, self.clock(), task_id, token),
            )
            return cur.rowcount == 1

    def fail(self, task_id: int, token: int, owner: str, error: str, retryable: bool = True) -> bool:
        now = self.clock()
        with self._tx() as db:
            row = db.execute(
                "SELECT attempts, max_attempts FROM tasks WHERE id=? AND token=? AND state='leased'",
                (task_id, token),
            ).fetchone()
            if row is None:
                return False
            self._retry_or_fail(db, task_id, row["attempts"], row["max_attempts"], error,
                                retryable, now, owner)
            return True

    def model_backlog(self) -> dict[str, int]:
        with self.lock:
            return dict(self.db.execute(
                "SELECT model, COUNT(*) FROM tasks WHERE state='queued' GROUP BY model"
            ).fetchall())

    # -- internals ----------------------------------------------------------
    def _reclaim(self, db, now: float) -> None:
        for r in db.execute(
            "SELECT id, attempts, max_attempts, owner FROM tasks WHERE state='leased' AND lease_expiry<?",
            (now,),
        ).fetchall():
            self._retry_or_fail(db, r["id"], r["attempts"], r["max_attempts"],
                                f"lease expired on {r['owner']}", True, now, None)

    def _retry_or_fail(self, db, task_id, attempts, max_attempts, error, retryable, now, owner):
        if retryable and attempts < max_attempts:
            delay = min(BACKOFF_BASE_S * 2 ** (attempts - 1), BACKOFF_MAX_S)
            db.execute(
                "UPDATE tasks SET state='queued', token=token+1, owner=NULL, lease_expiry=NULL,"
                " available_at=?, error=?, updated_at=? WHERE id=?",
                (now + delay, error, now, task_id),
            )
        else:
            db.execute(
                "UPDATE tasks SET state='failed', token=token+1, owner=NULL, lease_expiry=NULL,"
                " error=?, finished_by=?, updated_at=? WHERE id=?",
                (error, owner, now, task_id),
            )


class _Tx:
    """``with store._tx() as db``: lock + BEGIN IMMEDIATE ... COMMIT/ROLLBACK."""

    def __init__(self, store: JobStore):
        self.store = store

    def __enter__(self):
        self.store.lock.acquire()
        self.store.db.execute("BEGIN IMMEDIATE")
        return self.store.db

    def __exit__(self, exc_type, *_):
        try:
            self.store.db.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.store.lock.release()
