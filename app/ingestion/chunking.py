"""The two chunking strategies compared in the evaluation (decision D9).

Both work on a StructuredText and return chunks that are exact substrings of its text, sized in
tokens of the dense embedding model's own tokenizer, so the cap is exactly what the model sees.

A. fixed     — slide a window of `size` tokens with `overlap`, ignoring structure (can cut sentences).
B. recursive — pack whole units (heading -> paragraph -> sentence) up to `size` tokens, never crossing
               a section boundary; only a single over-long sentence is ever cut, by token windows.
"""

from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum

from app.ingestion.document import StructuredText
from app.ingestion.text import Span, sentence_spans


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str
    n_tokens: int


class TokenOffsets:
    """Character offsets of every token in one text, so any span's token count is two bisects."""

    def __init__(self, tokenizer, text: str):
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        self.starts = [s for s, _ in encoding["offset_mapping"]]
        self.ends = [e for _, e in encoding["offset_mapping"]]

    def __len__(self) -> int:
        return len(self.starts)

    def count(self, span: Span) -> int:
        return self.first_at_or_after(span[1]) - self.first_at_or_after(span[0])

    def first_at_or_after(self, char: int) -> int:
        return bisect_left(self.starts, char)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 10**9  # we only count tokens here; silences the >512 warning
    return tokenizer


def chunk(
    doc: StructuredText,
    tokenizer,
    strategy: ChunkingStrategy,
    size: int = 256,
    overlap: int = 32,
) -> list[Chunk]:
    offsets = TokenOffsets(tokenizer, doc.text)
    if strategy is ChunkingStrategy.FIXED:
        spans = _fixed_spans(offsets, 0, len(offsets), size, overlap)
    else:
        spans = _recursive_spans(doc, offsets, size)
    return [
        Chunk(i, start, end, doc.text[start:end], offsets.count((start, end)))
        for i, (start, end) in enumerate(spans)
    ]


def _fixed_spans(offsets: TokenOffsets, first: int, last: int, size: int, overlap: int) -> list[Span]:
    """Token windows over tokens[first:last], as character spans."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    spans: list[Span] = []
    step = size - overlap
    for i in range(first, last, step):
        j = min(i + size, last)
        spans.append((offsets.starts[i], offsets.ends[j - 1]))
        if j == last:
            break
    return spans


def _recursive_spans(doc: StructuredText, offsets: TokenOffsets, size: int) -> list[Span]:
    spans: list[Span] = []
    for section in doc.sections:
        atoms: list[Span] = [section.heading] if section.heading else []
        for paragraph in section.paragraphs:
            atoms.extend(_split_to_fit(doc.text, paragraph, offsets, size))
        spans.extend(_pack(atoms, offsets, size))
    return spans


def _split_to_fit(text: str, span: Span, offsets: TokenOffsets, size: int) -> list[Span]:
    """paragraph -> sentences -> token windows, descending only as far as needed to fit `size`."""
    if offsets.count(span) <= size:
        return [span]
    pieces: list[Span] = []
    for sentence in sentence_spans(text, *span):
        if offsets.count(sentence) <= size:
            pieces.append(sentence)
        else:  # a single sentence longer than the cap: the only place recursive cuts mid-sentence
            first = offsets.first_at_or_after(sentence[0])
            last = offsets.first_at_or_after(sentence[1])
            pieces.extend(_fixed_spans(offsets, first, last, size, overlap=0))
    return pieces


def _pack(atoms: list[Span], offsets: TokenOffsets, size: int) -> list[Span]:
    """Greedily merge consecutive atoms while the merged span stays within `size` tokens."""
    packed: list[Span] = []
    for atom in atoms:
        if packed and offsets.count((packed[-1][0], atom[1])) <= size:
            packed[-1] = (packed[-1][0], atom[1])
        else:
            packed.append(atom)
    return packed
