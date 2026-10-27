"""Qdrant collections and the operations ingestion needs (DATABASE.md §2–§7).

Two collections linked by `doc_id`:
- `chunks`    one point per chunk per chunking strategy; named `dense` + `sparse` vectors.
- `documents` one vectorless point per document: the registry used for listing, versions and dedup.

Memory layout (DATABASE.md §7): int8-quantized dense vectors stay in RAM for search; the float32
originals, the payload and the sparse index live on disk, so ~300k chunks fit the 1 GB free tier.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "sparse"
NUM_CHUNKS_PREFIX = "num_chunks_"  # registry payload key per strategy, e.g. num_chunks_recursive

# Fixed namespace so IDs are deterministic: re-running ingestion overwrites instead of duplicating.
_NAMESPACE = uuid.UUID("6f1c2b4e-3d5a-4c7e-9b8f-2a1d0e9c7b54")


def stable_id(key: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, key))


def chunk_point_id(doc_id: str, strategy: str, index: int, version: int = 1) -> str:
    # The strategy is part of the key: both strategies' chunks of one document share the collection.
    # From version 2 the version is too, so an update can insert its chunks before deleting the old ones.
    key = f"{doc_id}:{strategy}:{index}" if version == 1 else f"{doc_id}:{strategy}:{index}:v{version}"
    return stable_id(key)


@dataclass(frozen=True)
class ChunkPoint:
    doc_id: str
    strategy: str
    index: int
    dense: list[float]
    sparse: tuple[list[int], list[float]]
    payload: dict[str, Any]
    version: int = 1


class QdrantStore:
    def __init__(self, client: QdrantClient, chunks: str = "chunks", documents: str = "documents"):
        self.client = client
        self.chunks = chunks
        self.documents = documents

    @classmethod
    def from_settings(cls, settings) -> "QdrantStore":
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        client = QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=60)
        return cls(client, settings.chunks_collection, settings.documents_collection)

    # --- schema -------------------------------------------------------------------------------

    def ensure_collections(self, dense_dim: int) -> None:
        if not self.client.collection_exists(self.chunks):
            self.client.create_collection(
                self.chunks,
                vectors_config={
                    DENSE: models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE, on_disk=True
                    )
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True),
                        modifier=models.Modifier.IDF,
                    )
                },
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, always_ram=True
                    )
                ),
                on_disk_payload=True,
            )
            for name, schema in [
                ("doc_id", models.PayloadSchemaType.KEYWORD),
                ("sha256", models.PayloadSchemaType.KEYWORD),
                ("chunking_strategy", models.PayloadSchemaType.KEYWORD),
                ("version", models.PayloadSchemaType.INTEGER),
            ]:
                self.client.create_payload_index(self.chunks, name, schema)

        if not self.client.collection_exists(self.documents):
            self.client.create_collection(self.documents, vectors_config={})
            for name in ("doc_id", "sha256", "format"):
                self.client.create_payload_index(
                    self.documents, name, models.PayloadSchemaType.KEYWORD
                )

    # --- documents registry -------------------------------------------------------------------

    def chunk_counts_by_hash(self) -> dict[str, dict[str, int]]:
        """sha256 -> {strategy: chunk count} for everything already indexed. Dedup is per
        (content, strategy), so indexing a second chunking strategy is not mistaken for a duplicate."""
        result: dict[str, dict[str, int]] = defaultdict(dict)
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.documents, limit=1000, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                result[p.payload["sha256"]].update(
                    (key.removeprefix(NUM_CHUNKS_PREFIX), value)
                    for key, value in p.payload.items()
                    if key.startswith(NUM_CHUNKS_PREFIX)
                )
            if offset is None:
                return result

    def upsert_documents(self, records: Iterable[dict[str, Any]]) -> None:
        """Create or fully replace registry points (each record needs `doc_id`)."""
        points = [models.PointStruct(id=r["doc_id"], vector={}, payload=r) for r in records]
        if points:
            self.client.upsert(self.documents, points=points)

    def record_chunk_count(self, doc_id: str, strategy: str, n_chunks: int) -> None:
        """Merge one strategy's chunk count into an existing registry point. A flat key per strategy
        (not a nested dict) because set_payload merges top-level keys, while upsert would replace."""
        self.client.set_payload(
            self.documents, payload={NUM_CHUNKS_PREFIX + strategy: n_chunks}, points=[doc_id]
        )

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        found = self.client.retrieve(self.documents, ids=[doc_id], with_payload=True)
        return found[0].payload if found else None

    # --- chunks -------------------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[ChunkPoint]) -> None:
        points = [
            models.PointStruct(
                id=chunk_point_id(c.doc_id, c.strategy, c.index, c.version),
                vector={
                    DENSE: c.dense,
                    SPARSE: models.SparseVector(indices=c.sparse[0], values=c.sparse[1]),
                },
                payload=c.payload,
            )
            for c in chunks
        ]
        if points:
            self.client.upsert(self.chunks, points=points, wait=True)

    def search_lists(
        self,
        dense: list[float] | None,
        sparse: tuple[list[int], list[float]] | None,
        query_filter: models.Filter | None,
        limit: int,
    ) -> dict[str, list[models.ScoredPoint]]:
        """Run the dense and/or sparse search as two separate ranked lists in one round trip.
        Fusion is NOT delegated to Qdrant: the backend fuses them with its own RRF."""
        requests, names = [], []
        if dense is not None:
            names.append(DENSE)
            requests.append(
                models.QueryRequest(
                    query=dense,
                    using=DENSE,
                    filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    # Search the int8 copies in RAM, then rescore the best with the float32 originals.
                    params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)
                    ),
                )
            )
        if sparse is not None:
            names.append(SPARSE)
            requests.append(
                models.QueryRequest(
                    query=models.SparseVector(indices=sparse[0], values=sparse[1]),
                    using=SPARSE,
                    filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            )
        if not requests:
            return {}
        responses = self.client.query_batch_points(self.chunks, requests=requests)
        return {name: response.points for name, response in zip(names, responses)}

    def get_chunks(self, doc_id: str, strategy: str, limit: int = 1000) -> list[models.Record]:
        """One document's chunks of one strategy, in reading order."""
        flt = models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="chunking_strategy", match=models.MatchValue(value=strategy)),
        ])
        points, _ = self.client.scroll(self.chunks, scroll_filter=flt, limit=limit, with_payload=True)
        return sorted(points, key=lambda p: p.payload["chunk_index"])

    def find_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """The registry record of a document with this content hash, if any (exact dedup)."""
        flt = models.Filter(must=[models.FieldCondition(key="sha256", match=models.MatchValue(value=sha256))])
        points, _ = self.client.scroll(self.documents, scroll_filter=flt, limit=1, with_payload=True)
        return points[0].payload if points else None

    def list_documents(
        self,
        limit: int = 50,
        offset: str | None = None,
        fmt: str | None = None,
        exclude_format: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        flt = _format_filter(fmt, exclude_format)
        points, next_offset = self.client.scroll(
            self.documents, scroll_filter=flt, limit=limit, offset=offset, with_payload=True
        )
        return [p.payload for p in points], (str(next_offset) if next_offset is not None else None)

    def delete_old_versions(self, doc_id: str, keep_version: int) -> None:
        """Remove a document's chunks from every version except `keep_version` (the update's last step)."""
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))],
            must_not=[models.FieldCondition(key="version", match=models.MatchValue(value=keep_version))],
        )
        self.client.delete(self.chunks, points_selector=models.FilterSelector(filter=flt))

    def delete_document(self, doc_id: str) -> None:
        by_doc = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        self.client.delete(self.chunks, points_selector=models.FilterSelector(filter=by_doc))
        self.client.delete(self.documents, points_selector=models.PointIdsList(points=[doc_id]))

    def count(self, collection: str, strategy: str | None = None) -> int:
        flt = None
        if strategy:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunking_strategy", match=models.MatchValue(value=strategy)
                    )
                ]
            )
        return self.client.count(collection, count_filter=flt, exact=True).count


def _format_filter(fmt: str | None, exclude_format: str | None) -> models.Filter | None:
    must = [models.FieldCondition(key="format", match=models.MatchValue(value=fmt))] if fmt else []
    must_not = (
        [models.FieldCondition(key="format", match=models.MatchValue(value=exclude_format))]
        if exclude_format
        else []
    )
    return models.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None
