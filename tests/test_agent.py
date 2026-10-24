"""The agent loop and its gates, driven by a scripted LLM (no network, no models)."""

import json
from types import SimpleNamespace

from app.agent.agent import REFUSAL, Agent, AgentParams
from app.agent.gates import check_citations
from app.llm.base import LLMError, LLMResponse, LLMUnavailable, ToolCall
from app.retrieval.candidate import Candidate
from app.retrieval.searcher import SearchResult

PASSAGES = {  # query keyword -> (doc, chunk text, cross-encoder score)
    "jaws": [("Jaws (film)", "Jaws is a 1975 film directed by Steven Spielberg.", 6.0)],
    "spielberg": [("Steven Spielberg", "Steven Spielberg was born in 1946 in Cincinnati.", 7.0),
                  ("Spielberg fan page", "Spielberg was born in 1952.", 2.0)],
    "weather": [("Rain", "Rain is liquid water.", -6.0)],
}  # fmt: skip


class FakeSearcher:
    def search(self, query, params):
        hits = next((v for k, v in PASSAGES.items() if k in query.lower()), [])
        results = []
        for title, text, ce in hits:
            c = Candidate(f"pt-{title}", {"doc_id": f"doc-{title}", "title": title, "text": text, "chunk_index": 0})
            c.ce_score = ce
            results.append(c)
        return SearchResult(query, params, results)


class FakeStore:
    def get_chunks(self, doc_id, strategy, limit=1000):
        return [SimpleNamespace(id=f"{doc_id}-{i}", payload={"doc_id": doc_id, "title": "T", "text": f"part {i}",
                                                             "chunk_index": i}) for i in range(3)]  # fmt: skip


class ScriptedLLM:
    """Investigator turns come from `turns`; verifier calls (they pass a response_schema) from `verdicts`."""

    name = "scripted"

    def __init__(self, turns, verdicts=()):
        self.turns, self.verdicts, self.seen = list(turns), list(verdicts), []

    def can_replay(self, messages):
        return True

    def chat(self, system, messages, tools=None, response_schema=None):
        self.seen.append(list(messages))
        if response_schema:
            return LLMResponse(json.dumps(self.verdicts.pop(0)), [], "scripted", "m")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        calls = turn if isinstance(turn, list) else []
        return LLMResponse(turn if isinstance(turn, str) else "", calls, "scripted", "m")


def search(q):
    return [ToolCall("search_documents", {"query": q})]


def submit(status, answer, citations):
    return [ToolCall("submit_answer", {"status": status, "answer": answer, "citations": citations})]


SUPPORTED = {"supported": True, "unsupported_claims": [], "conflict": False, "conflict_sides": [], "explanation": ""}


def ask(llm, question="When was the director of Jaws born?", **params):
    return Agent(llm, FakeSearcher(), FakeStore()).ask(question, AgentParams(**params))


def test_multi_hop_answer_with_citations_from_two_searches():
    llm = ScriptedLLM([search("Jaws director"), search("Steven Spielberg birth"),
                       submit("answered", "Jaws was directed by Spielberg [S1], born in 1946 [S2].", ["S1", "S2"])],
                      [SUPPORTED])  # fmt: skip
    out = ask(llm)
    assert out.status == "answered"
    assert [c.title for c in out.citations] == ["Jaws (film)", "Steven Spielberg"]
    assert out.searched == ["Jaws director", "Steven Spielberg birth"]
    assert out.gates["retrieval"]["passed"] and out.usage["llm_calls"] == 4


def test_citations_of_sources_never_shown_are_dropped():
    assert check_citations("x [S1] y [S9]", ["S1", "s7"], {"S1", "S2"}) == (["S1"], ["S7", "S9"])


def test_an_answer_citing_only_unknown_sources_is_refused():
    out = ask(ScriptedLLM([search("Jaws"), submit("answered", "It was 1946 [S5].", ["S5"])]))
    assert out.status == "insufficient" and out.answer == REFUSAL
    assert out.gates["citations"]["invalid"] == ["S5"]


def test_retrieval_gate_refuses_when_nothing_relevant_was_found():
    llm = ScriptedLLM([search("weather"), submit("answered", "Rain is water [S1].", ["S1"])])
    out = ask(llm, relevance_floor=0.0)
    assert out.status == "insufficient" and not out.gates["retrieval"]["passed"]
    assert len(llm.seen) == 2  # refused before the verifier was called


def test_verifier_surfaces_a_conflict_the_model_missed():
    conflict = {"supported": True, "unsupported_claims": [], "conflict": True, "explanation": "two years",
                "conflict_sides": [{"claim": "born in 1946", "source_id": "S1"},
                                   {"claim": "born in 1952", "source_id": "S2"}]}  # fmt: skip
    llm = ScriptedLLM([search("Spielberg"), submit("answered", "Born in 1946 [S1].", ["S1"])], [conflict])
    out = ask(llm)
    assert out.status == "conflict"
    assert [c.source_id for c in out.citations] == ["S1", "S2"]
    assert "1946" in out.answer and "1952" in out.answer


def test_verifier_sees_retrieved_but_uncited_passages():
    llm = ScriptedLLM([search("Spielberg"), submit("answered", "Born in 1946 [S1].", ["S1"])], [SUPPORTED])
    out = ask(llm)
    assert out.gates["verifier"]["passages"] == ["S1", "S2"]


def test_unsupported_draft_is_refused():
    unsupported = SUPPORTED | {"supported": False, "unsupported_claims": ["born in Ohio"]}
    llm = ScriptedLLM([search("Spielberg"), submit("answered", "Born in Ohio [S1].", ["S1"])], [unsupported])
    assert ask(llm).status == "insufficient"


def test_model_refusal_is_kept_and_prefixed():
    out = ask(ScriptedLLM([search("weather"), submit("insufficient", "No birth year found.", [])]))
    assert out.status == "insufficient" and out.answer == f"{REFUSAL} No birth year found."


def test_tool_budget_and_iteration_cap_force_a_final_answer():
    llm = ScriptedLLM([search("Jaws")] * 3 + [submit("answered", "Spielberg [S1].", ["S1"])], [SUPPORTED])
    out = ask(llm, max_iterations=3, max_tool_calls=2)
    tool_steps = [s for s in out.trace if s["step"] == "tool"]
    assert len(tool_steps) == 2 and out.status == "answered"
    last_tool_result = [m for m in llm.seen[-2] if m.role == "tool"][-1].result
    assert "search budget" in last_tool_result["error"]


def test_malformed_arguments_and_plain_text_are_fed_back_to_the_model():
    bad = [ToolCall("search_documents", {}, args_error="arguments are not valid JSON")]
    llm = ScriptedLLM([bad, "I think it is 1946.", search("Jaws"), submit("answered", "Spielberg [S1].", ["S1"])],
                      [SUPPORTED])  # fmt: skip
    out = ask(llm)
    assert out.status == "answered"
    history = llm.seen[2]
    assert "not valid JSON" in history[2].result["error"]
    assert history[-1].role == "user" and "submit_answer" in history[-1].text


def test_invalid_submit_gets_an_error_for_every_call_in_the_turn():
    turn = search("Jaws") + [ToolCall("submit_answer", {"status": "maybe", "answer": "x", "citations": []})]
    llm = ScriptedLLM([turn, search("Jaws"), submit("answered", "Spielberg [S1].", ["S1"])], [SUPPORTED])
    ask(llm)
    results = [m for m in llm.seen[1] if m.role == "tool"]
    assert len(results) == 2 and all("error" in m.result for m in results)


def test_get_document_reads_the_article_and_rejects_unknown_ids():
    llm = ScriptedLLM([search("Jaws"), [ToolCall("get_document", {"source_id": "S9"})],
                       [ToolCall("get_document", {"source_id": "s1"})],
                       submit("answered", "It is in part 2 [S4].", ["S4"])], [SUPPORTED])  # fmt: skip
    out = ask(llm)
    assert "unknown source_id" in [m for m in llm.seen[2] if m.role == "tool"][-1].result["error"]
    assert [p["source_id"] for p in [m for m in llm.seen[3] if m.role == "tool"][-1].result["passages"]] == ["S2", "S3", "S4"]
    assert out.status == "answered" and out.gates["retrieval"]["best_ce_score"] == 6.0


def test_single_shot_mode_searches_once_and_runs_gates():
    llm = ScriptedLLM([submit("answered", "Directed by Spielberg [S1].", ["S1"])], [SUPPORTED])
    events = []
    out = Agent(llm, FakeSearcher(), FakeStore()).ask(
        "Who directed Jaws?", AgentParams(agentic=False), on_event=events.append
    )
    assert out.status == "answered" and out.searched == ["Who directed Jaws?"]
    assert [e["stage"] for e in events] == ["searching", "found", "submitting", "verifying"]
    assert out.usage["llm_calls"] == 2  # one single-shot answer + one verifier


def test_llm_outage_returns_a_structured_error_not_an_answer():
    out = ask(ScriptedLLM([LLMUnavailable([LLMError("gemini", "down", retryable=False)])]))
    assert out.status == "error" and out.answer == "" and "LLM unavailable" in out.error


def test_answer_lists_every_retrieved_source_not_only_the_cited_ones():
    llm = ScriptedLLM([search("Spielberg"), submit("answered", "Born in 1946 [S1].", ["S1"])], [SUPPORTED])
    out = ask(llm)
    assert [r["title"] for r in out.retrieved] == ["Steven Spielberg", "Spielberg fan page"]
    assert [c.source_id for c in out.citations] == ["S1"]


def test_retrieval_gate_also_scores_cited_passages_against_the_original_question():
    class QuestionAwareCE:
        def score(self, query, texts):
            return [6.5 if query.startswith("When was") else -1.0 for _ in texts]

    searcher = FakeSearcher()
    searcher.cross_encoder = QuestionAwareCE()
    llm = ScriptedLLM([search("weather"), submit("answered", "Rain is water [S1].", ["S1"])], [SUPPORTED])
    out = Agent(llm, searcher, FakeStore()).ask("When was rain discovered?", AgentParams(relevance_floor=3.92))
    gate = out.gates["retrieval"]
    assert gate["subquery_ce_score"] == -6.0 and gate["question_ce_score"] == 6.5 and gate["passed"]
    assert out.status == "answered"
