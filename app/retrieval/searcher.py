"""The retrieval pipeline behind `search_documents()` and `POST /search`.

    query -> dense + sparse lists (Qdrant) -> weighted RRF -> feature re-ranker -> cross-encoder
          -> per-document cap -> top-k

Every stage can be switched off through SearchParams, which is how the evaluation compares
configurations (hybrid vs dense-only, with/without each re-ranker) on the same code path.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from qdrant_client import models

from app.ingestion.chunking import ChunkingStrategy
from app.retrieval.candidate import Candidate
from app.retrieval.features import FeatureReranker
from app.retrieval.fusion import weighted_rrf
from app.store.qdrant_store import DENSE, SPARSE, QdrantStore


class SearchParams(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    w_dense: float = Field(1.0, ge=0)
    w_sparse: float = Field(1.0, ge=0)
    rrf_k: int = Field(60, gt=0)
    candidates: int = Field(50, gt=0)  # per initial list
    feature_rerank: bool = True
    feature_weights: dict[str, float] | None = None
    cross_encoder: bool = True
    top_n: int = Field(20, gt=0)  # how many survivors the cross-encoder scores
    top_k: int = Field(8, gt=0)  # returned to the caller / the LLM
    max_per_doc: int | None = Field(2, gt=0)  # None = no cap
    doc_ids: list[str] | None = None  # optional scope filter


@dataclass
class SearchResult:
    query: str
    params: SearchParams
    results: list[Candidate]
    stage_sizes: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_ce_score(self) -> float | None:
        """Best cross-encoder score: what the relevance floor (Phase 2b) will be applied to."""
        scores = [c.ce_score for c in self.results if c.ce_score is not None]
        return max(scores) if scores else None


class Searcher:
    def __init__(self, store: QdrantStore, dense, sparse, cross_encoder=None):
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.cross_encoder = cross_encoder

    def search(self, query: str, params: SearchParams | None = None) -> SearchResult:
        params = params or SearchParams()
        timings: dict[str, float] = {}

        with _timed(timings, "embed"):
            dense_vec = self.dense.embed_query(query) if params.mode != "sparse" else None
            sparse_vec = self.sparse.embed_query(query) if params.mode != "dense" else None
        with _timed(timings, "qdrant"):
            lists = self.store.search_lists(dense_vec, sparse_vec, self._filter(params), params.candidates)

        with _timed(timings, "fusion"):
            candidates = self._fuse(lists, params)
        sizes = {"dense": len(lists.get(DENSE, [])), "sparse": len(lists.get(SPARSE, [])),
                 "fused": len(candidates)}  # fmt: skip

        if params.feature_rerank:
            with _timed(timings, "feature_rerank"):
                candidates = FeatureReranker(params.feature_weights).rerank(query, candidates)

        if params.cross_encoder and self.cross_encoder is not None:
            with _timed(timings, "cross_encoder"):
                head = candidates[: params.top_n]
                for c, s in zip(head, self.cross_encoder.score(query, [c.text for c in head])):
                    c.ce_score = s
                candidates = sorted(head, key=lambda c: -c.ce_score)
            sizes["cross_encoder"] = len(candidates)

        results = cap_per_doc(candidates, params.max_per_doc, params.top_k)
        sizes["returned"] = len(results)
        return SearchResult(query, params, results, sizes, timings)

    @staticmethod
    def _filter(params: SearchParams) -> models.Filter:
        must = [models.FieldCondition(key="chunking_strategy",
                                      match=models.MatchValue(value=params.strategy.value))]  # fmt: skip
        if params.doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=params.doc_ids)))
        return models.Filter(must=must)

    @staticmethod
    def _fuse(lists: dict[str, list[models.ScoredPoint]], params: SearchParams) -> list[Candidate]:
        by_id: dict[str, Candidate] = {}
        for name, points in lists.items():
            for rank, p in enumerate(points, start=1):
                c = by_id.setdefault(str(p.id), Candidate(str(p.id), p.payload or {}))
                if name == DENSE:
                    c.dense_rank, c.dense_score = rank, p.score
                else:
                    c.sparse_rank, c.sparse_score = rank, p.score
        fused = weighted_rrf(
            {name: [str(p.id) for p in points] for name, points in lists.items()},
            weights={DENSE: params.w_dense, SPARSE: params.w_sparse},
            k=params.rrf_k,
        )
        ordered = []
        for point_id, score, contributions in fused:
            c = by_id[point_id]
            c.rrf_score, c.rrf_contributions = score, contributions
            ordered.append(c)
        return ordered


def cap_per_doc(candidates: list[Candidate], max_per_doc: int | None, top_k: int) -> list[Candidate]:
    """Keep at most `max_per_doc` chunks per article so one long article cannot fill the top-k;
    multi-hop questions need evidence from two different articles."""
    kept: list[Candidate] = []
    per_doc: dict[str, int] = {}
    for c in candidates:
        if max_per_doc is not None and per_doc.get(c.doc_id, 0) >= max_per_doc:
            continue
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
        kept.append(c)
        if len(kept) == top_k:
            break
    return kept


@contextmanager
def _timed(timings: dict[str, float], stage: str):
    """Record the wall time of a `with` block in milliseconds under `stage`."""
    start = time.perf_counter()
    yield
    timings[stage] = round((time.perf_counter() - start) * 1000, 1)
