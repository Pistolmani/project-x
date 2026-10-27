"""Builds the engine once per process: store, models, searcher, agent. The API, the UI's backend and the
evaluation scripts all use this, so they run the same configuration."""

from dataclasses import dataclass

from app.agent.agent import Agent
from app.agent.guard import Guard, GroqGuardModel
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
    backend = settings.model_backend
    dense = CachedEmbedder(DenseEmbedder(settings.dense_model, backend=backend), kv, "embed_dense")
    sparse = CachedEmbedder(SparseEmbedder(settings.sparse_model), kv, "embed_sparse")
    searcher = Searcher(store, dense, sparse, CrossEncoderReranker(backend=backend) if cross_encoder else None,
                        kv if cache else None)
    llm = CachedLLM(build_llm(settings), kv)
    verifier_llm = CachedLLM(build_llm(settings, [settings.gemini_model]), kv, "llm_verifier")  # one JSON call
    agent = Agent(llm, searcher, store, verifier_llm, build_guard(settings, kv))
    return Engine(settings, store, dense, sparse, searcher, llm, agent, kv)


def build_guard(settings: Settings, cache: Cache | None = None) -> Guard:
    """Without a Groq key the guard still runs its plain-code checks (length, patterns, prompt leaks)."""
    if settings.groq_api_key is None:
        return Guard(max_question_chars=settings.max_question_chars)
    key = settings.groq_api_key.get_secret_value()
    return Guard(GroqGuardModel(key, settings.guard_model, cache), GroqGuardModel(key, settings.guard_detector_model, cache),
                 settings.max_question_chars)
