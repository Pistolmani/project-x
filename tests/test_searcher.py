"""Wiring tests for the Searcher with tiny fake embedders and an in-memory Qdrant (no models)."""

import zlib

import pytest
from qdrant_client import QdrantClient

from app.retrieval.searcher import SearchParams, Searcher, cap_per_doc
from app.retrieval.candidate import Candidate
from app.store.qdrant_store import ChunkPoint, QdrantStore, stable_id

VOCAB = ["river", "delta", "magazine", "women", "radio", "station", "bridge", "war"]


class FakeDense:
    """Bag-of-vocabulary vector: enough for cosine search to behave sensibly."""

    def embed_query(self, text):
        lowered = text.lower()
        return [1.0 if w in lowered else 0.01 for w in VOCAB]


class FakeSparse:
    def embed_query(self, text):
        ids = sorted({zlib.crc32(w.encode()) % 10_000 for w in text.lower().split()})
        return ids, [1.0] * len(ids)


class CountingCrossEncoder:
    """Prefers chunks mentioning 'station', and records how many texts it was asked to score."""

    def __init__(self):
        self.seen = 0

    def score(self, query, texts):
        self.seen += len(texts)
        return [5.0 if "station" in t else -5.0 for t in texts]


DOCS = {
    "River": ["The river delta has islands.", "The delta floods in spring."],
    "Magazine": ["The magazine for women started in 1844.", "The magazine closed later."],
    "Radio": ["The radio station broadcasts music.", "The station opened in 2001."],
}


@pytest.fixture
def store():
    s = QdrantStore(QdrantClient(":memory:"))
    s.ensure_collections(len(VOCAB))
    dense, sparse = FakeDense(), FakeSparse()
    points = []
    for strategy in ("recursive", "fixed"):
        for title, chunks in DOCS.items():
            doc_id = stable_id(f"wiki:{title}")
            for i, text in enumerate(chunks):
                payload = {"doc_id": doc_id, "title": title, "text": text, "n_tokens": 30,
                           "chunking_strategy": strategy, "chunk_index": i,
                           "start_char": i * 100, "end_char": i * 100 + 50}  # fmt: skip
                points.append(ChunkPoint(doc_id, strategy, i, dense.embed_query(text),
                                         sparse.embed_query(text), payload))  # fmt: skip
    s.upsert_chunks(points)
    return s


def test_hybrid_search_returns_traced_candidates_of_the_default_strategy(store):
    result = Searcher(store, FakeDense(), FakeSparse()).search(
        "river delta", SearchParams(cross_encoder=False, max_per_doc=None)
    )
    assert result.results[0].title == "River"
    assert all(c.payload["chunking_strategy"] == "fixed" for c in result.results)
    top = result.results[0].trace()
    assert top["dense"]["rank"] is not None and top["rrf"]["score"] > 0
    assert set(result.timings_ms) >= {"embed", "qdrant", "fusion", "feature_rerank"}


def test_dense_only_mode_never_queries_sparse(store):
    class ExplodingSparse:
        def embed_query(self, text):
            raise AssertionError("sparse must not be used in dense mode")

    result = Searcher(store, FakeDense(), ExplodingSparse()).search(
        "river", SearchParams(mode="dense", cross_encoder=False)
    )
    assert result.stage_sizes["sparse"] == 0
    assert all(c.sparse_rank is None for c in result.results)


def test_cross_encoder_scores_only_top_n_and_decides_the_order(store):
    ce = CountingCrossEncoder()
    result = Searcher(store, FakeDense(), FakeSparse(), ce).search(
        "magazine", SearchParams(top_n=3, top_k=3, max_per_doc=None)
    )
    assert ce.seen == 3
    assert result.top_ce_score is not None
    scores = [c.ce_score for c in result.results]
    assert scores == sorted(scores, reverse=True)


def test_stages_can_be_switched_off(store):
    ce = CountingCrossEncoder()
    result = Searcher(store, FakeDense(), FakeSparse(), ce).search(
        "radio", SearchParams(feature_rerank=False, cross_encoder=False)
    )
    assert ce.seen == 0
    assert all(c.feature_score is None and c.ce_score is None for c in result.results)


def test_doc_filter_scopes_the_search(store):
    radio = stable_id("wiki:Radio")
    result = Searcher(store, FakeDense(), FakeSparse()).search(
        "river delta magazine", SearchParams(doc_ids=[radio], cross_encoder=False)
    )
    assert result.results and all(c.doc_id == radio for c in result.results)


def test_cap_per_doc_keeps_evidence_from_several_articles():
    cands = [Candidate(str(i), {"doc_id": d}) for i, d in enumerate("aaabbc")]
    assert [c.doc_id for c in cap_per_doc(cands, 2, 10)] == ["a", "a", "b", "b", "c"]
    assert [c.doc_id for c in cap_per_doc(cands, 1, 2)] == ["a", "b"]
    assert len(cap_per_doc(cands, None, 4)) == 4
