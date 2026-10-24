"""The agentic RAG loop behind `POST /ask` (BACKEND.md §5.2).

    plan -> search (repeatedly, refining) -> read -> submit_answer
         -> citation check -> retrieval gate -> verifier (groundedness + contradiction) -> answer / refuse

The LLM decides how to investigate; the gates after it are plain code, so a refusal or a surfaced conflict
can be traced to the exact check that produced it (`AgentAnswer.gates`).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.agent import prompts
from app.agent.gates import Verifier, check_citations, passes_relevance_floor
from app.agent.tools import SUBMIT, Toolbox
from app.llm.base import LLM, LLMUnavailable, Message
from app.retrieval.searcher import SearchParams, Searcher
from app.store.qdrant_store import QdrantStore

Status = Literal["answered", "insufficient", "conflict", "error"]

REFUSAL = "I don't have enough information in the documents to answer this."


class AgentParams(BaseModel):
    search: SearchParams = Field(default_factory=SearchParams)
    agentic: bool = True  # False = single-shot (one search with the raw question, then submit_answer)
    max_iterations: int = Field(5, gt=0)  # investigator LLM calls before the answer is forced
    max_tool_calls: int = Field(8, gt=0)  # search/get_document budget
    # Cross-encoder logit. evaluation.calibrate_floor: 3.92 passes 95% of answerable tuning questions and
    # blocks 71% of held-out (unanswerable) ones; None disables the gate.
    relevance_floor: float | None = 3.92
    verify: bool = True  # groundedness + contradiction check (one extra LLM call)


@dataclass
class Citation:
    source_id: str
    chunk_id: str
    doc_id: str
    title: str
    text: str
    ce_score: float | None


@dataclass
class AgentAnswer:
    question: str
    status: Status
    answer: str
    citations: list[Citation] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)  # every query the agent ran
    # Every source shown to the model (cited or not): what the eval's "retrieval miss" bucket checks.
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)  # what each check decided, and why
    trace: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0})
    error: str | None = None
    ms: float = 0.0


class Agent:
    def __init__(self, llm: LLM, searcher: Searcher, store: QdrantStore, verifier_llm: LLM | None = None):
        self.llm = llm
        self.verifier_llm = verifier_llm or llm  # a different model spreads the free-tier quota
        self.searcher = searcher
        self.store = store

    def ask(
        self,
        question: str,
        params: AgentParams | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentAnswer:
        params = params or AgentParams()
        emit = on_event or (lambda _: None)
        start = time.perf_counter()
        toolbox = Toolbox(self.searcher, self.store, params.search)
        out = AgentAnswer(question, "error", "")
        try:
            if params.agentic:
                submitted = self._investigate(question, params, toolbox, out, emit)
            else:
                submitted = self._single_shot(question, toolbox, out, emit)
            self._finalize(question, submitted, params, toolbox, out, emit)
        except LLMUnavailable as e:
            out.status, out.answer, out.error = "error", "", f"LLM unavailable: {e}"
        except (UnexpectedResponse, ResponseHandlingException) as e:
            out.status, out.answer, out.error = "error", "", f"vector store unavailable: {e}"
        out.searched = [s["query"] for s in toolbox.searches]
        out.retrieved = [{"source_id": s.handle, "doc_id": s.doc_id, "title": s.title, "ce_score": s.ce_score}
                         for s in toolbox.sources.values()]  # fmt: skip
        out.ms = round((time.perf_counter() - start) * 1000, 1)
        return out

    # --- the loop -------------------------------------------------------------------------------

    def _investigate(self, question: str, params: AgentParams, toolbox: Toolbox, out: AgentAnswer,
                     emit: Callable[[dict[str, Any]], None] = lambda _: None) -> dict:  # fmt: skip
        messages = [Message("user", question)]
        tool_calls = 0
        for iteration in range(1, params.max_iterations + 1):
            emit({"type": "progress", "stage": "thinking", "iteration": iteration,
                  "message": "planning…" if iteration == 1 else f"thinking (step {iteration})…"})  # fmt: skip
            response = self._chat(prompts.INVESTIGATOR, messages, toolbox.specs(), out)
            messages.append(response.as_message())
            if not response.tool_calls:
                messages.append(Message("user", prompts.NO_TOOL_CALLED))
                continue
            submit = next((c for c in response.tool_calls if c.name == SUBMIT.name), None)
            if submit is not None:
                parsed = _parse_submit(submit.args)
                if isinstance(parsed, dict):
                    return parsed
                # Every call in the turn needs a result, or the provider rejects the next request.
                for call in response.tool_calls:
                    error = parsed if call is submit else "not run: submit_answer must be called on its own"
                    messages.append(Message("tool", tool_call_id=call.id, tool_name=call.name,
                                            result={"error": error}))  # fmt: skip
                continue
            for call in response.tool_calls:
                if tool_calls >= params.max_tool_calls:
                    result = {"error": prompts.SUBMIT_NOW}
                else:
                    tool_calls += 1
                    result = self._run_tool(call, toolbox, out, emit)
                messages.append(Message("tool", tool_call_id=call.id, tool_name=call.name, result=result))

        # Out of iterations: one last call that can only submit.
        emit({"type": "progress", "stage": "submitting", "message": "drafting final answer…"})
        messages.append(Message("user", prompts.SUBMIT_NOW))
        response = self._chat(prompts.INVESTIGATOR, messages, [SUBMIT], out)
        submit = next((c for c in response.tool_calls if c.name == SUBMIT.name), None)
        parsed = _parse_submit(submit.args) if submit else "no answer submitted"
        if isinstance(parsed, dict):
            return parsed
        out.gates["loop"] = {"passed": False, "reason": f"agent did not conclude: {parsed}"}
        return {"status": "insufficient", "answer": "", "citations": []}

    def _single_shot(self, question: str, toolbox: Toolbox, out: AgentAnswer,
                     emit: Callable[[dict[str, Any]], None] = lambda _: None) -> dict:  # fmt: skip
        """Non-agentic RAG: one search with the raw question, then one LLM call forced to submit_answer."""
        from app.agent.tools import SEARCH
        from app.llm.base import ToolCall

        result = self._run_tool(ToolCall(SEARCH.name, {"query": question}), toolbox, out, emit)
        passages = "\n\n".join(f"[{p['source_id']}] ({p['title']})\n{p['text']}" for p in result.get("results", []))
        prompt = f"Question: {question}\n\nPassages:\n{passages or '(no passages found)'}\n\n{prompts.SINGLE_SHOT}"
        emit({"type": "progress", "stage": "submitting", "message": "drafting answer…"})
        response = self._chat(prompts.INVESTIGATOR, [Message("user", prompt)], [SUBMIT], out)
        submit = next((c for c in response.tool_calls if c.name == SUBMIT.name), None)
        parsed = _parse_submit(submit.args) if submit else "no answer submitted"
        if isinstance(parsed, dict):
            return parsed
        out.gates["loop"] = {"passed": False, "reason": f"single-shot did not conclude: {parsed}"}
        return {"status": "insufficient", "answer": "", "citations": []}

    def _run_tool(self, call, toolbox: Toolbox, out: AgentAnswer,
                  emit: Callable[[dict[str, Any]], None]) -> dict:  # fmt: skip
        if call.name == "search_documents":
            q = call.args.get("query", "") if isinstance(call.args, dict) else ""
            emit({"type": "progress", "stage": "searching", "query": q, "message": f"searching: {q}"})
        elif call.name == "get_document":
            sid = call.args.get("source_id", "") if isinstance(call.args, dict) else ""
            emit({"type": "progress", "stage": "reading", "source_id": sid, "message": f"reading document {sid}…"})
        step_start = time.perf_counter()
        result = toolbox.run(call)
        n_res = len(result.get("results", result.get("passages", [])))
        out.trace.append({"step": "tool", "tool": call.name, "args": call.args, "results": n_res,
                          "error": result.get("error"),
                          "ms": round((time.perf_counter() - step_start) * 1000, 1)})  # fmt: skip
        if call.name == "search_documents" and not result.get("error"):
            emit({"type": "progress", "stage": "found", "count": n_res,
                  "message": f"found {n_res} passage{'s' if n_res != 1 else ''}"})  # fmt: skip
        return result

    def _question_ce_score(self, question: str, texts: list[str]) -> float | None:
        cross_encoder = getattr(self.searcher, "cross_encoder", None)
        if cross_encoder is None or not texts:
            return None
        return max(cross_encoder.score(question, texts))

    def _chat(self, system, messages, tools, out: AgentAnswer):
        step_start = time.perf_counter()
        response = self.llm.chat(system, messages, tools)
        out.usage["llm_calls"] += 1
        out.usage["input_tokens"] += response.usage.input_tokens
        out.usage["output_tokens"] += response.usage.output_tokens
        out.trace.append({"step": "llm", "provider": response.provider, "model": response.model,
                          "tool_calls": [c.name for c in response.tool_calls],
                          "ms": round((time.perf_counter() - step_start) * 1000, 1)})  # fmt: skip
        return response

    # --- the gates ------------------------------------------------------------------------------

    def _finalize(self, question, submitted: dict, params: AgentParams, toolbox: Toolbox, out: AgentAnswer,
                  emit: Callable[[dict[str, Any]], None] = lambda _: None) -> None:  # fmt: skip
        status, answer = submitted["status"], submitted["answer"].strip()
        cited, invalid = check_citations(answer, submitted["citations"], set(toolbox.sources))
        out.gates["citations"] = {"valid": cited, "invalid": invalid, "passed": bool(cited) or status != "answered"}

        if status == "answered" and not cited:
            status, answer = "insufficient", REFUSAL
        elif status == "answered":
            best_subquery = toolbox.best_ce_score(cited)
            if best_subquery is None:  # cited only passages read via get_document: use the article's searches
                best_subquery = toolbox.best_ce_score()
            # The floor was calibrated on the user's question, but each passage was scored against the
            # agent's own sub-query, and keyword-style sub-queries score lower even on the right passage.
            # So the cited passages are also scored against the original question; the better score counts.
            best_question = self._question_ce_score(question, [toolbox.sources[h].text for h in cited])
            scores = [x for x in (best_subquery, best_question) if x is not None]
            best = max(scores) if scores else None
            passed = passes_relevance_floor(best, params.relevance_floor)
            out.gates["retrieval"] = {"best_ce_score": best, "subquery_ce_score": best_subquery,
                                      "question_ce_score": best_question, "floor": params.relevance_floor,
                                      "passed": passed}  # fmt: skip
            if not passed:
                status, answer = "insufficient", REFUSAL

        if status == "answered" and params.verify:
            emit({"type": "progress", "stage": "verifying", "message": "verifying…"})
            verdict, info = Verifier(self.verifier_llm).check(question, answer, toolbox, cited)
            out.usage["llm_calls"] += 1
            out.usage["input_tokens"] += info.get("usage", {}).get("input_tokens", 0)
            out.usage["output_tokens"] += info.get("usage", {}).get("output_tokens", 0)
            out.gates["verifier"] = info | ({"supported": verdict.supported, "conflict": verdict.conflict,
                                             "unsupported_claims": verdict.unsupported_claims,
                                             "explanation": verdict.explanation} if verdict else {})  # fmt: skip
            if verdict and verdict.conflict and len({s["source_id"] for s in verdict.conflict_sides}) >= 2:
                status = "conflict"
                answer = "The sources disagree: " + "; ".join(
                    f"{s['claim']} [{s['source_id']}]" for s in verdict.conflict_sides) + "."  # fmt: skip
                cited = list(dict.fromkeys(s["source_id"] for s in verdict.conflict_sides))
            elif verdict and not verdict.supported:
                status, answer = "insufficient", REFUSAL

        if status == "insufficient" and answer != REFUSAL:
            answer = f"{REFUSAL} {answer}".strip()
        out.status, out.answer = status, answer
        out.citations = [_citation(toolbox, h) for h in cited]


def _parse_submit(args: dict) -> dict | str:
    """Validated submit_answer arguments, or an error message for the model."""
    status = args.get("status")
    if status not in ("answered", "insufficient", "conflict"):
        return "status must be one of: answered, insufficient, conflict"
    if not isinstance(args.get("answer"), str):
        return "answer must be a string"
    citations = args.get("citations") or []
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        return "citations must be a list of source ids such as [\"S1\"]"
    return {"status": status, "answer": args["answer"], "citations": citations}


def _citation(toolbox: Toolbox, handle: str) -> Citation:
    s = toolbox.sources[handle]
    return Citation(handle, s.point_id, s.doc_id, s.title, s.text, s.ce_score)
