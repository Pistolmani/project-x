from qdrant_client import QdrantClient

from ingest.copy_to_cloud import copy_chunks, copy_registry
from app.store.qdrant_store import ChunkPoint, QdrantStore, stable_id


def _store():
    s = QdrantStore(QdrantClient(":memory:"))
    s.ensure_collections(3)
    return s


def test_copies_only_the_chosen_strategy_and_its_chunk_count():
    src, dst = _store(), _store()
    doc_id = stable_id("wiki:A")
    src.upsert_chunks([ChunkPoint(doc_id, strategy, i, [1.0, 0.0, float(i)], ([1, 2], [0.5, 0.5]),
                                  {"doc_id": doc_id, "chunking_strategy": strategy, "text": f"t{i}"})
                       for strategy, n in [("fixed", 3), ("recursive", 5)] for i in range(n)])  # fmt: skip
    src.upsert_documents([{"doc_id": doc_id, "sha256": "h", "num_chunks_fixed": 3, "num_chunks_recursive": 5}])

    assert copy_chunks(src, dst, "fixed", batch=2) == 3
    assert copy_registry(src, dst, "fixed", batch=2) == 1
    assert dst.count(dst.chunks) == dst.count(dst.chunks, "fixed") == 3
    assert dst.get_document(doc_id) == {"doc_id": doc_id, "sha256": "h", "num_chunks_fixed": 3}
    point = dst.client.retrieve(dst.chunks, ids=[src.client.scroll(src.chunks, limit=1)[0][0].id],
                                with_vectors=True)  # fmt: skip
    assert point and point[0].vector["sparse"].indices == [1, 2]
