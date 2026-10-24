"""Add / update / delete against an in-memory Qdrant with tiny fake embedders and tokenizer."""

import re

import pytest
from qdrant_client import QdrantClient

from app.cache import Cache
from app.config import Settings
from app.ingestion.service import DocumentNotFound, IngestionService
from app.store.qdrant_store import QdrantStore


class WhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0, 0.5] for t in texts]


class FakeSparse:
    def embed_documents(self, texts):
        return [([1, 2], [1.0, 1.0]) for _ in texts]


@pytest.fixture
def service():
    store = QdrantStore(QdrantClient(":memory:"))
    store.ensure_collections(3)
    settings = Settings(chunk_size=8, chunk_overlap=2, _env_file=None)
    return IngestionService(store, FakeEmbedder(), FakeSparse(), WhitespaceTokenizer(), settings, cache=Cache())


DOC_V1 = b"# Jaws\n\nJaws is a 1975 film directed by Steven Spielberg about a great white shark.\n"
DOC_V2 = b"# Jaws\n\nJaws is a 1975 thriller by Steven Spielberg.\n"


def test_add_indexes_chunks_and_registers_the_document(service):
    result = service.add(DOC_V1, "jaws.md")
    assert result.status == "created" and result.version == 1 and result.num_chunks["fixed"] >= 2
    record = service.store.get_document(result.doc_id)
    assert record["title"] == "Jaws" and record["format"] == "markdown" and record["num_chunks_fixed"] == result.num_chunks["fixed"]
    assert service.store.count(service.store.chunks) == result.num_chunks["fixed"]
    assert service.cache.generation("corpus") == 1


def test_identical_content_is_a_duplicate_even_with_different_formatting(service):
    first = service.add(DOC_V1, "jaws.md")
    again = service.add(DOC_V1.replace(b"\n\n", b"\n\n\n"), "copy.md")
    assert again.status == "duplicate" and again.duplicate_of == first.doc_id
    assert service.store.count(service.store.documents) == 1


def test_update_bumps_the_version_and_leaves_only_the_new_chunks(service):
    doc_id = service.add(DOC_V1, "jaws.md").doc_id
    updated = service.update(doc_id, DOC_V2, "jaws.md")
    assert updated.status == "updated" and updated.version == 2
    chunks = service.store.get_chunks(doc_id, "fixed")
    assert {c.payload["version"] for c in chunks} == {2} and len(chunks) == updated.num_chunks["fixed"]
    assert "thriller" in " ".join(c.payload["text"] for c in chunks)
    assert service.store.get_document(doc_id)["version"] == 2


def test_update_with_unchanged_content_does_nothing(service):
    doc_id = service.add(DOC_V1, "jaws.md").doc_id
    assert service.update(doc_id, DOC_V1, "jaws.md").status == "duplicate"
    assert service.store.get_document(doc_id)["version"] == 1


def test_delete_removes_chunks_and_registry(service):
    doc_id = service.add(DOC_V1, "jaws.md").doc_id
    service.delete(doc_id)
    assert service.store.count(service.store.chunks) == 0 and service.store.get_document(doc_id) is None
    with pytest.raises(DocumentNotFound):
        service.delete(doc_id)
    with pytest.raises(DocumentNotFound):
        service.update(doc_id, DOC_V2, "jaws.md")


def test_tag_is_recorded_and_survives_an_update(service):
    doc_id = service.add(DOC_V1, "jaws.md", tag="eval_planted").doc_id
    assert service.store.get_document(doc_id)["tag"] == "eval_planted"
    service.update(doc_id, DOC_V2, "jaws.md")
    assert service.store.get_document(doc_id)["tag"] == "eval_planted"
