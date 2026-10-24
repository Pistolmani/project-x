import pytest
from qdrant_client import QdrantClient

from app.store.qdrant_store import ChunkPoint, QdrantStore, chunk_point_id, stable_id

DIM = 4


def test_ids_are_deterministic_and_strategy_specific():
    assert stable_id("wiki:Albedo") == stable_id("wiki:Albedo")
    doc = stable_id("wiki:Albedo")
    assert chunk_point_id(doc, "fixed", 0) != chunk_point_id(doc, "recursive", 0)


@pytest.fixture
def store():
    s = QdrantStore(QdrantClient(":memory:"))
    s.ensure_collections(DIM)
    return s


def _point(doc_id, strategy, index):
    payload = {"doc_id": doc_id, "chunking_strategy": strategy, "chunk_index": index, "version": 1}
    return ChunkPoint(doc_id, strategy, index, [0.1, 0.2, 0.3, 0.4], ([1, 7], [0.5, 1.0]), payload)


def test_schema_is_created_once_and_documents_collection_is_vectorless(store):
    store.ensure_collections(DIM)  # idempotent
    assert store.client.get_collection("documents").config.params.vectors == {}


def test_upsert_is_idempotent_and_counts_per_strategy(store):
    doc = stable_id("wiki:A")
    points = [_point(doc, "fixed", i) for i in range(3)] + [_point(doc, "recursive", 0)]
    store.upsert_chunks(points)
    store.upsert_chunks(points)  # re-run: same IDs, no duplicates
    assert store.count("chunks") == 4
    assert store.count("chunks", "fixed") == 3


def test_registry_tracks_chunk_counts_per_strategy(store):
    doc = stable_id("wiki:A")
    store.upsert_documents([{"doc_id": doc, "sha256": "abc", "title": "A", "num_chunks_fixed": 3}])
    store.record_chunk_count(doc, "recursive", 2)  # merges, does not wipe num_chunks_fixed
    assert store.chunk_counts_by_hash() == {"abc": {"fixed": 3, "recursive": 2}}
    assert store.get_document(doc)["title"] == "A"


def test_delete_document_removes_chunks_and_registry_point(store):
    a, b = stable_id("wiki:A"), stable_id("wiki:B")
    store.upsert_chunks([_point(a, "fixed", 0), _point(a, "fixed", 1), _point(b, "fixed", 0)])
    store.upsert_documents([{"doc_id": a, "sha256": "a"}, {"doc_id": b, "sha256": "b"}])
    store.delete_document(a)
    assert store.count("chunks") == 1
    assert store.get_document(a) is None
    assert store.get_document(b) is not None
