"""API keys for the coordinator's OpenAI endpoints, with per-key usage.

While no key exists the API is open, as before. Once a key has been created,
every ``/v1/*`` request needs ``Authorization: Bearer <key>`` (or must come
from an administrator, see ``swarm.coordinator``). Only a SHA-256 of each key
is stored; the key itself is shown once, when it is created.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from typing import Callable

PREFIX = "sk-swarm-"

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    hint TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    last_used_at REAL,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    revoked_at REAL
);
"""


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class KeyStore:
    def __init__(self, path: str = ":memory:", clock: Callable[[], float] = time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self._active: dict[str, str] = {}  # hash -> id, for revocation-free lookups
        self._load()

    def _load(self) -> None:
        with self.lock:
            rows = self.db.execute("SELECT id, hash FROM api_keys WHERE revoked_at IS NULL").fetchall()
        self._active = {r["hash"]: r["id"] for r in rows}

    @property
    def required(self) -> bool:
        """Whether the API needs a key: true as soon as one active key exists."""
        return bool(self._active)

    def create(self, name: str) -> tuple[dict, str]:
        secret = PREFIX + secrets.token_hex(24)
        key_id = "key_" + secrets.token_hex(6)
        with self.lock:
            self.db.execute(
                "INSERT INTO api_keys (id, name, hint, hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (key_id, name.strip() or key_id, secret[:len(PREFIX) + 4] + "…" + secret[-4:],
                 _digest(secret), self.clock()),
            )
        self._load()
        return self.get(key_id), secret

    def revoke(self, key_id: str) -> bool:
        with self.lock:
            cur = self.db.execute(
                "UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL", (self.clock(), key_id)
            )
        self._load()
        return cur.rowcount == 1

    def verify(self, secret: str | None) -> str | None:
        """The key id for a valid, unrevoked key, else None."""
        return self._active.get(_digest(secret)) if secret else None

    def record(self, key_id: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE api_keys SET requests=requests+1, prompt_tokens=prompt_tokens+?, "
                "completion_tokens=completion_tokens+?, last_used_at=? WHERE id=?",
                (prompt_tokens, completion_tokens, self.clock(), key_id),
            )

    def get(self, key_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        return self._public(row) if row else None

    def list(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        return [self._public(r) for r in rows]

    @staticmethod
    def _public(row) -> dict:
        d = dict(row)
        d.pop("hash")
        d["active"] = d["revoked_at"] is None
        return d
