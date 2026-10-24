"""Anti-hallucination gates (BACKEND.md §5.4). Each is a separate, testable step so the eval can measure
it: citation check, retrieval gate (relevance floor), and the verifier (groundedness + contradiction)."""

import json
import re
from dataclasses import dataclass, field

from app.agent.prompts import VERIFIER, VERIFIER_SCHEMA, verifier_input
from app.agent.tools import Toolbox
from app.llm.base import LLM, LLMUnavailable, Message

CITATION = re.compile(r"\[(S\d+)\]")


def check_citations(answer: str, submitted: list[str], known: set[str]) -> tuple[list[str], list[str]]:
    """(valid, invalid) handles, in first-mention order. A citation is valid only if that source was
    actually shown to the model; anything else is a fabricated reference and is dropped."""
    mentioned = [h.strip().upper().strip("[]") for h in submitted] + CITATION.findall(answer)
    ordered = list(dict.fromkeys(h for h in mentioned if h))
    return [h for h in ordered if h in known], [h for h in ordered if h not in known]


def passes_relevance_floor(best_ce_score: float | None, floor: float | None) -> bool:
    """Retrieval gate on the raw cross-encoder score, never on RRF (rank-only, so it always has a top)."""
    if floor is None:
        return True
    return best_ce_score is not None and best_ce_score >= floor


@dataclass
class Verdict:
    supported: bool
    conflict: bool
    unsupported_claims: list[str] = field(default_factory=list)
    conflict_sides: list[dict] = field(default_factory=list)
    explanation: str = ""


class Verifier:
    """One LLM call that checks the draft against the evidence: is every claim supported, and do the
    passages contradict each other on the asked-about fact?"""

    def __init__(self, llm: LLM, max_passages: int = 8):
        self.llm = llm
        self.max_passages = max_passages

    def evidence(self, toolbox: Toolbox, cited: list[str]) -> list[str]:
        """The cited sources, then the best-scoring other retrieved sources. A planted conflicting passage
        is often retrieved but not cited (the model picked one side), so the verifier must see it too."""
        others = sorted((s for h, s in toolbox.sources.items() if h not in cited and s.ce_score is not None),
                        key=lambda s: -s.ce_score)  # fmt: skip
        return (cited + [s.handle for s in others])[: max(self.max_passages, len(cited))]

    def check(self, question: str, answer: str, toolbox: Toolbox, cited: list[str]) -> tuple[Verdict | None, dict]:
        handles = self.evidence(toolbox, cited)
        passages = [(h, toolbox.sources[h].title, toolbox.sources[h].text) for h in handles]
        try:
            response = self.llm.chat(VERIFIER, [Message("user", verifier_input(question, answer, passages))],
                                     response_schema=VERIFIER_SCHEMA)  # fmt: skip
        except LLMUnavailable as e:
            return None, {"error": str(e), "passages": handles}
        info = {"passages": handles, "provider": response.provider, "usage": vars(response.usage)}
        try:
            data = json.loads(response.text)
            verdict = Verdict(bool(data["supported"]), bool(data["conflict"]), list(data.get("unsupported_claims", [])),
                              [s for s in data.get("conflict_sides", []) if s.get("source_id") in toolbox.sources],
                              str(data.get("explanation", "")))  # fmt: skip
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            return None, info | {"error": f"unparseable verdict: {e}"}
        return verdict, info
