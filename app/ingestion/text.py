"""Small text utilities shared by every loader: normalization, content hashing, sentence spans."""

import hashlib
import re
import unicodedata

Span = tuple[int, int]  # [start, end) character offsets into a document's text


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


def content_hash(text: str) -> str:
    """sha256 of the text with whitespace collapsed, so formatting-only differences still dedup."""
    collapsed = " ".join(normalize(text).split())
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


# A sentence ends at . ! or ? (plus any closing quotes/brackets), followed by whitespace and then
# something that looks like a sentence start. Deliberately simple: abbreviations such as "U.S. Army"
# can cause an extra split, which only makes a recursive chunk boundary slightly early.
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*(\s+)(?=[\"“(\[]?[A-Z0-9])")


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Sentence spans inside text[start:end], as absolute offsets into `text`."""
    end = len(text) if end is None else end
    spans: list[Span] = []
    cursor = start
    for match in _BOUNDARY.finditer(text, start, end):
        stop = match.start(1)  # the sentence ends before the whitespace
        if stop > cursor:
            spans.append((cursor, stop))
        cursor = match.end(1)
    if cursor < end and text[cursor:end].strip():
        spans.append((cursor, end))
    return spans
