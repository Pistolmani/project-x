"""The HTTP surface with an in-memory Qdrant, fake embedders and a scripted LLM (no models, no network)."""

import json
import re
import zlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.agent.agent import Agent
from app.api.main import create_app
from app.cache import Cache
from app.config import Settings
from app.ingestion.service import IngestionService
from app.llm.base import LLMError, LLMResponse, LLMUnavailable, ToolCall
from app.retrieval.searcher import Searcher
from app.store.qdrant_store import QdrantStore

VOCAB = ["jaws", "shark", "film", "spielberg", "1975", "river", "delta"]


class Dense:
    def embed_query(self, text):
        return [1.0 if w in text.lower() else 0.01 for w in VOCAB]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class Sparse:
    def embed_query(self, text):
        ids = sorted({zlib.crc32(w.encode()) % 10_000 for w in re.findall(r"\w+", text.lower())})
        return ids, [1.0] * len(ids)

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class Tokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class LLM:
    """Searches once, then answers citing S1; verifier calls say supported."""

    name = "scripted"

    def __init__(self, fail=False):
        self.fail = fail

    def can_replay(self, messages):
        return True

    def chat(self, system, messages, tools=None, response_schema=None):
        if self.fail:
            raise LLMUnavailable([LLMError("gemini", "quota", retryable=True, status=429)])
        if response_schema:
            return LLMResponse(json.dumps({"supported": True, "unsupported_claims": [], "conflict": False,
                                           "conflict_sides": [], "explanation": ""}), [], "scripted", "m")  # fmt: skip
        if not any(m.role == "tool" for m in messages):
            return LLMResponse("", [ToolCall("search_documents", {"query": messages[0].text})], "scripted", "m")
        return LLMResponse("", [ToolCall("submit_answer", {"status": "answered", "answer": "Spielberg [S1].",
                                                           "citations": ["S1"]})], "scripted", "m")  # fmt: skip


def make_client(fail_llm=False):
    store = QdrantStore(QdrantClient(":memory:"))
    store.ensure_collections(len(VOCAB))
    cache = Cache()
    settings = Settings(chunk_size=40, chunk_overlap=5, _env_file=None)
    searcher = Searcher(store, Dense(), Sparse(), cache=cache)
    llm = LLM(fail_llm)
    engine = SimpleNamespace(store=store, searcher=searcher, agent=Agent(llm, searcher, store), cache=cache,
                             llm=SimpleNamespace(llm=SimpleNamespace(providers=[llm])), settings=settings)  # fmt: skip
    ingestion = IngestionService(store, Dense(), Sparse(), Tokenizer(), settings, cache=cache)
    return TestClient(create_app(engine, ingestion))


JAWS = b"# Jaws\n\nJaws is a 1975 shark film directed by Steven Spielberg.\n"


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def test_document_lifecycle(client):
    created = client.post("/documents", files={"file": ("jaws.md", JAWS)})
    assert created.status_code == 201
    doc_id = created.json()["doc_id"]
    assert client.post("/documents", files={"file": ("copy.md", JAWS)}).json()["status"] == "duplicate"

    listed = client.get("/documents").json()["documents"]
    assert [d["title"] for d in listed] == ["Jaws"]
    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["document"]["format"] == "markdown" and "Spielberg" in detail["chunks"][0]["text"]

    updated = client.put(f"/documents/{doc_id}", files={"file": ("jaws.md", JAWS + b"\nIt was a hit.\n")})
    assert updated.json()["version"] == 2
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").json()["error"]["code"] == "document_not_found"


def test_bad_upload_is_a_422_with_a_reason(client):
    response = client.post("/documents", files={"file": ("notes.docx", b"x")})
    assert response.status_code == 422 and "unsupported" in response.json()["error"]["message"]


def test_search_returns_ranked_hits_with_traces(client):
    client.post("/documents", files={"file": ("jaws.md", JAWS)})
    body = client.post("/search", json={"query": "Spielberg shark film", "params": {"cross_encoder": False}}).json()
    assert body["results"][0]["title"] == "Jaws" and body["results"][0]["trace"]["rrf"]["score"] > 0


def test_ask_answers_with_citations(client):
    client.post("/documents", files={"file": ("jaws.md", JAWS)})
    body = client.post("/ask", json={"question": "Who directed Jaws?", "params": {"relevance_floor": None}}).json()
    assert body["status"] == "answered" and body["citations"][0]["title"] == "Jaws"


def test_ask_during_an_llm_outage_is_a_503_with_a_structured_body():
    with make_client(fail_llm=True) as client:
        response = client.post("/ask", json={"question": "Who directed Jaws?"})
    assert response.status_code == 503
    assert response.json()["status"] == "error" and response.json()["answer"] == ""


def test_health_reports_store_and_providers(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 0 and body["llm_providers"] == ["scripted"]


def test_request_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "params": {"top_k": 0}}).status_code == 422
