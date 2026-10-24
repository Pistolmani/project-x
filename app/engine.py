"""Builds the engine once per process: store, models, searcher, agent. The API, the UI's backend and the
evaluation scripts all use this, so they run the same configuration."""

from dataclasses import dataclass

from app.agent.agent import Agent
from app.cache import Cache, CachedEmbedder
from app.config import Settings, get_settings
from app.llm.cached import CachedLLM
from app.llm.fallback import build_llm
from app.retrieval.searcher import Searcher
from app.store.qdrant_store import QdrantStore


@dataclass
class Engine:
    settings: Settings
    store: QdrantStore
    dense: object
    sparse: object
    searcher: Searcher
    llm: CachedLLM
    agent: Agent
    cache: Cache


def build_engine(settings: Settings | None = None, cross_encoder: bool = True, cache: bool = True) -> Engine:
    from app.ingestion.embedding import DenseEmbedder, SparseEmbedder
    from app.retrieval.cross_encoder import CrossEncoderReranker

    settings = settings or get_settings()
    store = QdrantStore.from_settings(settings)
    kv = Cache(settings.cache_path if cache else ":memory:")
    dense = CachedEmbedder(DenseEmbedder(settings.dense_model), kv, "embed_dense")
    sparse = CachedEmbedder(SparseEmbedder(settings.sparse_model), kv, "embed_sparse")
    searcher = Searcher(store, dense, sparse, CrossEncoderReranker() if cross_encoder else None,
                        kv if cache else None)  # fmt: skip
    llm = CachedLLM(build_llm(settings), kv)
    verifier_llm = CachedLLM(build_llm(settings, [settings.gemini_model]), kv, "llm_verifier")  # one JSON call
    agent = Agent(llm, searcher, store, verifier_llm)
    return Engine(settings, store, dense, sparse, searcher, llm, agent, kv)
