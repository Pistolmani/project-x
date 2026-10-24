"""A retrieved chunk plus the record of how every ranking stage scored it.

Keeping per-stage ranks and scores on the candidate is what makes the pipeline explainable: the UI
trace view and the "why did chunk X outrank chunk Y?" question both read from `trace()`.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    point_id: str
    payload: dict[str, Any]
    dense_rank: int | None = None  # 1-based rank in the dense list, None if not retrieved there
    dense_score: float | None = None  # cosine similarity
    sparse_rank: int | None = None
    sparse_score: float | None = None  # BM25 score (Qdrant applies IDF)
    rrf_score: float = 0.0
    rrf_contributions: dict[str, float] = field(default_factory=dict)  # list name -> w/(k+rank)
    features: dict[str, float] = field(default_factory=dict)
    feature_score: float | None = None
    ce_score: float | None = None  # raw cross-encoder logit: the relevance-floor signal

    @property
    def doc_id(self) -> str:
        return self.payload["doc_id"]

    @property
    def title(self) -> str:
        return self.payload.get("title", "")

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def n_tokens(self) -> int:
        return self.payload.get("n_tokens", 0)

    def trace(self) -> dict[str, Any]:
        return {
            "chunk_id": self.payload.get("chunk_id"),
            "doc_id": self.doc_id,
            "title": self.title,
            "dense": {"rank": self.dense_rank, "score": self.dense_score},
            "sparse": {"rank": self.sparse_rank, "score": self.sparse_score},
            "rrf": {"score": self.rrf_score, "contributions": self.rrf_contributions},
            "features": {"score": self.feature_score, **self.features},
            "cross_encoder": self.ce_score,
        }
