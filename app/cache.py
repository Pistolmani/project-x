"""SQLite key-value cache (BACKEND.md §6): query embeddings, search results and LLM responses.

The free LLM tiers allow tens to hundreds of requests a day, so repeated questions (the eval re-runs the
same 50 across configs; demo users ask the demo questions) must not spend quota twice. Values are JSON.
Search results depend on the corpus, so their keys include a corpus generation that every document
write bumps; old entries are then simply never hit again.
"""

import hashlib
import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


def make_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class Cache:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by FastAPI's worker threads, serialized by a lock.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, key TEXT, value TEXT, created REAL,"
                             " PRIMARY KEY (ns, key))")
            self._db.execute("CREATE TABLE IF NOT EXISTS generation (ns TEXT PRIMARY KEY, n INTEGER)")
        self.stats: Counter[str] = Counter()  # "<ns>:hit" / "<ns>:miss"

    def get(self, ns: str, key: str) -> Any | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE ns = ? AND key = ?", (ns, key)).fetchone()
        self.stats[f"{ns}:{'hit' if row else 'miss'}"] += 1
        return json.loads(row[0]) if row else None

    def set(self, ns: str, key: str, value: Any) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?, ?)",
                             (ns, key, json.dumps(value, default=str), time.time()))

    def generation(self, ns: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT n FROM generation WHERE ns = ?", (ns,)).fetchone()
        return row[0] if row else 0

    def bump(self, ns: str) -> int:
        with self._lock, self._db:
            self._db.execute("INSERT INTO generation VALUES (?, 1) ON CONFLICT(ns) DO UPDATE SET n = n + 1", (ns,))
            return self._db.execute("SELECT n FROM generation WHERE ns = ?", (ns,)).fetchone()[0]


class CachedEmbedder:
    """Caches `embed_query` of a dense or sparse embedder (sparse returns (indices, values))."""

    def __init__(self, embedder, cache: Cache, ns: str):
        self.embedder = embedder
        self.cache = cache
        self.ns = ns

    def embed_query(self, text: str):
        key = make_key(text)
        hit = self.cache.get(self.ns, key)
        if hit is not None:
            return tuple(hit) if isinstance(hit[0], list) else hit
        value = self.embedder.embed_query(text)
        self.cache.set(self.ns, key, value)
        return value

    def __getattr__(self, name):  # embed_documents, dim, ... pass straight through
        return getattr(self.embedder, name)
