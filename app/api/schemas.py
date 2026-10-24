"""Request/response models for the HTTP API (BACKEND.md §9)."""

from typing import Any

from pydantic import BaseModel, Field

from app.agent.agent import AgentParams
from app.retrieval.searcher import SearchParams


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    params: SearchParams = Field(default_factory=SearchParams)


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    chunk_index: int | None
    start_char: int | None
    end_char: int | None
    trace: dict[str, Any]  # every ranking stage's rank/score: the "why is this ranked here" view


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    stage_sizes: dict[str, int]
    timings_ms: dict[str, float]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    params: AgentParams = Field(default_factory=AgentParams)


class IngestResponse(BaseModel):
    doc_id: str
    status: str
    version: int
    title: str
    num_chunks: dict[str, int]
    duplicate_of: str | None = None


class DocumentList(BaseModel):
    documents: list[dict[str, Any]]
    next_offset: str | None


class ErrorBody(BaseModel):
    code: str
    message: str
