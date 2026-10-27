"""How source passages are shown (FRONTEND.md §3): display only, the agent's sources are unchanged.

Fixed-size chunks are 256-token windows that overlap by 32 tokens and ignore sentence boundaries, so two
cited neighbours of one article repeat each other and start or end mid-sentence. Here they are merged
back into one excerpt (exact, since chunk text == article text[start_char:end_char]), cut edges get "…",
and the text is shown literally: line breaks kept, Markdown characters escaped.
"""

import re
from dataclasses import dataclass, field
from typing import Any

SENTENCE_END = (".", "!", "?", '"', "”", "'", ")")
_MARKDOWN = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~$])")


@dataclass
class Passage:
    doc_id: str
    title: str
    text: str
    source_ids: list[str]
    chunk_indices: list[int] = field(default_factory=list)
    start_char: int | None = None
    end_char: int | None = None
    best_ce: float | None = None
    suspicious: str | None = None  # instruction-like phrase found in one of its chunks


def group_citations(citations: list[dict[str, Any]]) -> list[Passage]:
    """One passage per run of overlapping or touching chunks of the same article, in the order the
    agent first cited them. Citations without offsets stay passages of their own."""
    passages: list[Passage] = []
    for c in citations:
        start, end = c.get("start_char"), c.get("end_char")
        new = Passage(c.get("doc_id", ""), c.get("title") or c.get("doc_id", ""), c.get("text", ""),
                      [c["source_id"]], [c["chunk_index"]] if c.get("chunk_index") is not None else [],
                      start, end, c.get("ce_score"), c.get("suspicious"))
        target = next((p for p in passages if _joinable(p, new)), None)
        if target is None:
            passages.append(new)
        else:
            _merge(target, new)
    return passages


def _joinable(a: Passage, b: Passage) -> bool:
    if a.doc_id != b.doc_id or None in (a.start_char, a.end_char, b.start_char, b.end_char):
        return False
    return b.start_char <= a.end_char and a.start_char <= b.end_char


def _merge(a: Passage, b: Passage) -> None:
    first, second = (a, b) if a.start_char <= b.start_char else (b, a)
    if second.end_char <= first.end_char:
        text = first.text  # one contains the other
    else:
        text = first.text + second.text[first.end_char - second.start_char:]
    a.text, a.start_char, a.end_char = text, first.start_char, max(a.end_char, b.end_char)
    a.source_ids += b.source_ids
    a.chunk_indices = sorted(set(a.chunk_indices + b.chunk_indices))
    scores = [s for s in (a.best_ce, b.best_ce) if s is not None]
    a.best_ce = max(scores) if scores else None
    a.suspicious = a.suspicious or b.suspicious


def excerpt(text: str, start_char: int | None) -> str:
    """Mark the edges where a chunk cut the article: "…" before text that is not the article's start,
    and after text that does not end a sentence."""
    text = text.strip()
    if start_char:
        text = "…" + text
    if text and not text.endswith(SENTENCE_END):
        text += "…"
    return text


def as_plain_markdown(text: str) -> str:
    """Article text rendered literally by st.markdown: Markdown characters escaped, every line break kept
    (a hard break per line, a blank line between paragraphs)."""
    lines = [_MARKDOWN.sub(r"\\\1", line) for line in text.split("\n")]
    return "  \n".join(lines)


def label(p: Passage) -> str:
    chunks = ""
    if p.chunk_indices:
        lo, hi = p.chunk_indices[0], p.chunk_indices[-1]
        chunks = f" · chunk #{lo}" if lo == hi else f" · chunks #{lo}–{hi}"
    score = ""
    if p.best_ce is not None:
        score = f" · {'best ' if len(p.source_ids) > 1 else ''}CE score {p.best_ce:.2f}"
    return f"[{', '.join(p.source_ids)}] {p.title}{chunks}{score}"
