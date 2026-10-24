import pytest

from app.ingestion.chunking import ChunkingStrategy, chunk
from app.ingestion.document import Section, render

pytestmark = pytest.mark.integration  # downloads the bge tokenizer once


def _sentence(i: int) -> str:
    return f"Sentence number {i} talks about the river delta and its many small islands."


@pytest.fixture
def doc():
    paragraphs = [" ".join(_sentence(i * 10 + j) for j in range(6)) for i in range(4)]
    return render(
        [
            Section(None, [paragraphs[0]]),
            Section("Geography", paragraphs[1:3]),
            Section("History", [paragraphs[3]]),
        ]
    )


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_chunks_are_exact_substrings_within_the_cap(doc, tokenizer, strategy):
    chunks = chunk(doc, tokenizer, strategy, size=64, overlap=8)
    assert chunks
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.text == doc.text[c.start : c.end]
        assert 0 < c.n_tokens <= 64


def test_fixed_windows_overlap(doc, tokenizer):
    chunks = chunk(doc, tokenizer, ChunkingStrategy.FIXED, size=64, overlap=8)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # consecutive windows share text


def test_fixed_rejects_overlap_not_smaller_than_size(doc, tokenizer):
    with pytest.raises(ValueError):
        chunk(doc, tokenizer, ChunkingStrategy.FIXED, size=32, overlap=32)


def test_recursive_never_crosses_a_section_boundary(doc, tokenizer):
    chunks = chunk(doc, tokenizer, ChunkingStrategy.RECURSIVE, size=64)
    boundaries = [s.heading[0] for s in doc.sections if s.heading]
    for c in chunks:
        assert not any(c.start < b < c.end for b in boundaries)


def test_recursive_keeps_every_sentence_whole(doc, tokenizer):
    # Each sentence is well under the cap, so none may be split across chunks.
    chunks = chunk(doc, tokenizer, ChunkingStrategy.RECURSIVE, size=64)
    for i in range(40):
        start = doc.text.find(_sentence(i))
        if start == -1:
            continue
        end = start + len(_sentence(i))
        assert any(c.start <= start and end <= c.end for c in chunks), _sentence(i)


def test_recursive_packs_small_paragraphs_together(tokenizer):
    doc = render([Section(None, ["Short one.", "Short two.", "Short three."])])
    chunks = chunk(doc, tokenizer, ChunkingStrategy.RECURSIVE, size=64)
    assert [c.text for c in chunks] == ["Short one.\n\nShort two.\n\nShort three."]


def test_recursive_hard_splits_only_an_oversized_sentence(tokenizer):
    giant = "word " * 200 + "end."
    doc = render([Section(None, ["Before.", giant.strip()])])
    chunks = chunk(doc, tokenizer, ChunkingStrategy.RECURSIVE, size=64)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 64 for c in chunks)
