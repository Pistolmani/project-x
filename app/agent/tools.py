"""The tools the LLM can call (BACKEND.md §5.1), and the per-question evidence they collect.

Every chunk the model sees gets a short source handle (S1, S2, ...) the first time it appears. The model
cites handles, not UUIDs: short ids are copied reliably, and a citation can only be valid if the handle
was actually shown, which is what the citation check (gates.py) relies on.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.guard import find_injection
from app.agent.prompts import PASSAGE_WARNING
from app.llm.base import ToolCall, ToolSpec
from app.retrieval.searcher import SearchParams, Searcher
from app.store.qdrant_store import QdrantStore

SEARCH = ToolSpec(
    "search_documents",
    "Hybrid search over the document collection. Returns the most relevant passages, each with a source id "
    "(S1, S2, ...) to cite. Search again with a refined query when a passage names something you must look up "
    "next (multi-hop questions).",
    {"type": "object", "properties": {
        "query": {"type": "string", "description": "A focused search query, e.g. one sub-question."}},
     "required": ["query"]},
)  # fmt: skip
GET_DOCUMENT = ToolSpec(
    "get_document",
    "Read more of the article a source comes from, in order, when a passage is cut off or more context is needed.",
    {"type": "object", "properties": {
        "source_id": {"type": "string", "description": "A source id you were shown, e.g. S3."}},
     "required": ["source_id"]},
)  # fmt: skip
SUBMIT = ToolSpec(
    "submit_answer",
    "Finish. Call exactly once, after investigating.",
    {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient", "conflict"],
                   "description": "answered = the sources support an answer; insufficient = they do not; "
                                  "conflict = sources disagree on the answer."},
        "answer": {"type": "string", "description": "The answer with [S#] citations after each claim; for "
                   "insufficient, say what is missing; for conflict, state what each source says."},
        "citations": {"type": "array", "items": {"type": "string"},
                      "description": "Every source id the answer relies on."}},
     "required": ["status", "answer", "citations"]},
)  # fmt: skip


@dataclass
class Source:
    handle: str
    point_id: str
    doc_id: str
    title: str
    text: str
    ce_score: float | None  # cross-encoder score for the query that surfaced it (None if read via get_document)
    query: str | None
    payload: dict[str, Any] = field(repr=False, default_factory=dict)
    suspicious: str | None = None  # an instruction-like phrase found in the text (guard.find_injection)


class Toolbox:
    """Executes tool calls for one question and remembers every source shown to the model."""

    def __init__(self, searcher: Searcher, store: QdrantStore, params: SearchParams,
                 document_chars: int = 6000):  # fmt: skip
        self.searcher = searcher
        self.store = store
        self.params = params
        self.document_chars = document_chars
        self.sources: dict[str, Source] = {}  # handle -> source
        self._by_point: dict[str, str] = {}  # point id -> handle
        self.searches: list[dict[str, Any]] = []  # per search: query, top cross-encoder score

    def specs(self) -> list[ToolSpec]:
        return [SEARCH, GET_DOCUMENT, SUBMIT]

    def run(self, call: ToolCall) -> dict[str, Any]:
        """Tool result for the model. Bad calls return an `error` the model can react to, never raise."""
        if call.args_error:
            return {"error": f"{call.args_error}. Call {call.name} again with valid JSON arguments."}
        try:
            if call.name == SEARCH.name:
                return self.search(_required_str(call.args, "query"))
            if call.name == GET_DOCUMENT.name:
                return self.get_document(_required_str(call.args, "source_id"))
        except ValueError as e:
            return {"error": str(e)}
        return {"error": f"unknown tool {call.name!r}; available: search_documents, get_document, submit_answer"}

    def search(self, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        result = self.searcher.search(query, self.params)
        self.searches.append({"query": query, "top_ce_score": result.top_ce_score, "n": len(result.results),
                              "ms": round((time.perf_counter() - start) * 1000, 1)})  # fmt: skip
        shown = [self._register(c.point_id, c.payload, c.ce_score, query) for c in result.results]
        results: dict[str, Any] = {"results": [self._show(s) for s in shown]}
        if not shown:
            results["note"] = "no passages found"
        return results

    def get_document(self, handle: str) -> dict[str, Any]:
        source = self.sources.get(handle.strip().upper())
        if source is None:
            raise ValueError(f"unknown source_id {handle!r}; use an id from a search result")
        shown, used, truncated = [], 0, False
        for record in self.store.get_chunks(source.doc_id, self.params.strategy.value):
            text = record.payload["text"]
            if used + len(text) > self.document_chars and shown:
                truncated = True
                break
            used += len(text)
            existing = self._by_point.get(str(record.id))
            shown.append(self.sources[existing] if existing else
                         self._register(str(record.id), record.payload, None, None))  # fmt: skip
        return {"title": source.title, "passages": [self._show(s) for s in shown], "truncated": truncated}

    def best_ce_score(self, handles: list[str] | None = None) -> float | None:
        """Best cross-encoder score among the given sources (all searched sources if None)."""
        chosen = self.sources.values() if handles is None else (self.sources[h] for h in handles)
        scores = [s.ce_score for s in chosen if s.ce_score is not None]
        return max(scores) if scores else None

    def _register(self, point_id: str, payload: dict, ce_score: float | None, query: str | None) -> Source:
        handle = self._by_point.get(point_id)
        if handle is not None:
            source = self.sources[handle]
            if ce_score is not None and (source.ce_score is None or ce_score > source.ce_score):
                source.ce_score, source.query = ce_score, query
            return source
        handle = f"S{len(self.sources) + 1}"
        self._by_point[point_id] = handle
        text = payload.get("text", "")
        self.sources[handle] = Source(handle, point_id, payload.get("doc_id", ""), payload.get("title", ""),
                                      text, ce_score, query, payload, find_injection(text))  # fmt: skip
        return self.sources[handle]

    @staticmethod
    def _show(source: Source) -> dict[str, str]:
        # Passage text is untrusted data; the system prompt says instructions inside it are never followed.
        # It travels as a JSON string, so a passage cannot fake the end of its own result or another source.
        shown = {"source_id": source.handle, "title": source.title, "text": source.text}
        if source.suspicious:
            shown["warning"] = PASSAGE_WARNING
        return shown


def _required_str(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty argument {key!r}")
    return value.strip()
