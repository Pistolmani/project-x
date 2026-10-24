"""Add / update / delete documents at runtime (BACKEND.md §3.5–§3.6). Behind POST/PUT/DELETE /documents.

    load (format-specific) -> render -> exact dedup by content hash -> chunk -> embed -> upsert

Update is crash-safe by ordering: the new version's chunks are written first (they have their own point
IDs), then the registry, then the old version's chunks are deleted. A crash in between leaves the old
version, or both, searchable; never neither. Every write bumps the cache's corpus generation so cached
search results cannot outlive the corpus they came from.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.cache import Cache
from app.ingestion.chunking import ChunkingStrategy, chunk
from app.ingestion.document import render
from app.ingestion.loaders import LoadedDocument, load
from app.ingestion.text import content_hash
from app.store.qdrant_store import NUM_CHUNKS_PREFIX, ChunkPoint, QdrantStore


class DocumentNotFound(KeyError):
    pass


@dataclass
class IngestResult:
    doc_id: str
    status: str  # "created" | "updated" | "duplicate"
    version: int
    title: str
    num_chunks: dict[str, int]
    duplicate_of: str | None = None


class IngestionService:
    def __init__(self, store: QdrantStore, dense, sparse, tokenizer, settings,
                 strategies: list[ChunkingStrategy] | None = None, cache: Cache | None = None):  # fmt: skip
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.tokenizer = tokenizer
        self.settings = settings
        # Only the deployed strategy by default: the cloud index holds one strategy (EVALUATION.md §7.1).
        self.strategies = strategies or [ChunkingStrategy.FIXED]
        self.cache = cache

    def add(self, data: bytes, filename: str) -> IngestResult:
        doc = load(data, filename)
        text = render(doc.sections).text
        sha = content_hash(text)
        existing = self.store.find_by_hash(sha)
        if existing is not None:
            return IngestResult(existing["doc_id"], "duplicate", existing.get("version", 1), existing["title"],
                                _counts(existing), duplicate_of=existing["doc_id"])  # fmt: skip
        doc_id = str(uuid.uuid4())
        counts = self._index(doc_id, doc, version=1)
        now = _now()
        self.store.upsert_documents([_record(doc_id, doc, sha, 1, counts, created=now, updated=now)])
        self._changed()
        return IngestResult(doc_id, "created", 1, doc.title, counts)

    def update(self, doc_id: str, data: bytes, filename: str) -> IngestResult:
        current = self.store.get_document(doc_id)
        if current is None:
            raise DocumentNotFound(doc_id)
        doc = load(data, filename)
        sha = content_hash(render(doc.sections).text)
        if sha == current["sha256"]:
            return IngestResult(doc_id, "duplicate", current.get("version", 1), current["title"], _counts(current),
                                duplicate_of=doc_id)  # fmt: skip
        version = current.get("version", 1) + 1
        counts = self._index(doc_id, doc, version)  # 1. new chunks, alongside the old ones
        self.store.upsert_documents([_record(doc_id, doc, sha, version, counts,  # 2. registry points at them
                                             created=current.get("created_at"), updated=_now())])  # fmt: skip
        self.store.delete_old_versions(doc_id, keep_version=version)  # 3. only now drop the old ones
        self._changed()
        return IngestResult(doc_id, "updated", version, doc.title, counts)

    def delete(self, doc_id: str) -> None:
        if self.store.get_document(doc_id) is None:
            raise DocumentNotFound(doc_id)
        self.store.delete_document(doc_id)
        self._changed()

    def _index(self, doc_id: str, doc: LoadedDocument, version: int) -> dict[str, int]:
        structured = render(doc.sections)
        counts = {}
        for strategy in self.strategies:
            chunks = chunk(structured, self.tokenizer, strategy, self.settings.chunk_size, self.settings.chunk_overlap)
            texts = [c.text for c in chunks]
            points = [
                ChunkPoint(doc_id, strategy.value, c.index, vec, sp, {
                    "doc_id": doc_id, "chunk_id": f"{doc_id}:{strategy.value}:{c.index}", "text": c.text,
                    "title": doc.title, "source": doc.filename, "format": doc.format, "chunk_index": c.index,
                    "chunking_strategy": strategy.value, "start_char": c.start, "end_char": c.end,
                    "n_tokens": c.n_tokens, "version": version, "sha256": content_hash(c.text),
                    "created_at": _now()}, version)
                for c, vec, sp in zip(chunks, self.dense.embed_documents(texts), self.sparse.embed_documents(texts))
            ]  # fmt: skip
            self.store.upsert_chunks(points)
            counts[strategy.value] = len(points)
        return counts

    def _changed(self) -> None:
        if self.cache is not None:
            self.cache.bump("corpus")


def _record(doc_id, doc: LoadedDocument, sha, version, counts, created, updated) -> dict:
    return {**{NUM_CHUNKS_PREFIX + s: n for s, n in counts.items()}, "doc_id": doc_id, "filename": doc.filename,
            "format": doc.format, "title": doc.title, "source": doc.filename, "sha256": sha, "version": version,
            "metadata": doc.metadata, "created_at": created, "updated_at": updated}  # fmt: skip


def _counts(record: dict) -> dict[str, int]:
    return {k.removeprefix(NUM_CHUNKS_PREFIX): v for k, v in record.items() if k.startswith(NUM_CHUNKS_PREFIX)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
