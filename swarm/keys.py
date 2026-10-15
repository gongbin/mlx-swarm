"""Bearer credentials with usage counters: API keys and node join tokens.

- API keys (``KeyStore(path)``) guard the OpenAI endpoints. While none exists
  the API is open, as before; once one has been created, every ``/v1/*``
  request needs ``Authorization: Bearer sk-swarm-...``.
- Node tokens (``KeyStore(path, **NODE_TOKENS)``) guard ``/internal/*``, which
  agents use, in the same way: open until the first token exists.

Administrators pass both (see ``swarm.coordinator``). Only a SHA-256 of each
secret is stored; the secret itself is shown once, when it is created.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from typing import Callable

PREFIX = "sk-swarm-"
NODE_TOKENS = {"table": "node_tokens", "prefix": "nt-swarm-", "id_prefix": "node_"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
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
    def __init__(self, path: str = ":memory:", clock: Callable[[], float] = time.time,
                 table: str = "api_keys", prefix: str = PREFIX, id_prefix: str = "key_"):
        self.clock = clock
        self.table, self.prefix, self.id_prefix = table, prefix, id_prefix
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA.replace("{table}", table))
        self._active: dict[str, str] = {}  # hash -> id, for revocation-free lookups
        self._load()

    def _load(self) -> None:
        with self.lock:
            rows = self.db.execute(f"SELECT id, hash FROM {self.table} WHERE revoked_at IS NULL").fetchall()
        self._active = {r["hash"]: r["id"] for r in rows}

    @property
    def required(self) -> bool:
        """Whether the API needs a key: true as soon as one active key exists."""
        return bool(self._active)

    def create(self, name: str) -> tuple[dict, str]:
        secret = self.prefix + secrets.token_hex(24)
        key_id = self.id_prefix + secrets.token_hex(6)
        with self.lock:
            self.db.execute(
                f"INSERT INTO {self.table} (id, name, hint, hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (key_id, name.strip() or key_id, secret[:len(self.prefix) + 4] + "…" + secret[-4:],
                 _digest(secret), self.clock()),
            )
        self._load()
        return self.get(key_id), secret

    def revoke(self, key_id: str) -> bool:
        with self.lock:
            cur = self.db.execute(
                f"UPDATE {self.table} SET revoked_at=? WHERE id=? AND revoked_at IS NULL", (self.clock(), key_id)
            )
        self._load()
        return cur.rowcount == 1

    def verify(self, secret: str | None) -> str | None:
        """The key id for a valid, unrevoked key, else None."""
        return self._active.get(_digest(secret)) if secret else None

    def record(self, key_id: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self.lock:
            self.db.execute(
                f"UPDATE {self.table} SET requests=requests+1, prompt_tokens=prompt_tokens+?, "
                "completion_tokens=completion_tokens+?, last_used_at=? WHERE id=?",
                (prompt_tokens, completion_tokens, self.clock(), key_id),
            )

    def get(self, key_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute(f"SELECT * FROM {self.table} WHERE id=?", (key_id,)).fetchone()
        return self._public(row) if row else None

    def list(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute(f"SELECT * FROM {self.table} ORDER BY created_at DESC").fetchall()
        return [self._public(r) for r in rows]

    @staticmethod
    def _public(row) -> dict:
        d = dict(row)
        d.pop("hash")
        d["active"] = d["revoked_at"] is None
        return d
