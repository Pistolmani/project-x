"""Fallback policy and message conversion, without network calls."""

import json
from types import SimpleNamespace

import pytest
from google.genai import types

from app.llm.base import LLMError, LLMResponse, LLMUnavailable, Message, ToolCall
from app.llm.fallback import FallbackLLM
from app.llm.gemini import GeminiLLM, to_contents
from app.llm.groq import _parse_call, to_openai


class Scripted:
    """A provider that replays a list of outcomes: an LLMError is raised, anything else returned."""

    def __init__(self, name, outcomes, replays=True):
        self.name, self.outcomes, self.replays, self.calls = name, list(outcomes), replays, 0

    def can_replay(self, messages):
        return self.replays

    def chat(self, system, messages, tools=None, response_schema=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, LLMError):
            raise outcome
        return LLMResponse(outcome, [], self.name, "m")


def rate_limited(name):
    return LLMError(name, "429", retryable=True, status=429)


def test_retries_a_rate_limit_with_backoff_then_succeeds():
    sleeps = []
    gemini = Scripted("gemini", [rate_limited("gemini"), "ok"])
    r = FallbackLLM([gemini], retries=2, backoff_s=1, sleep=sleeps.append).chat("s", [])
    assert r.text == "ok" and gemini.calls == 2 and sleeps == [1]


def test_falls_back_to_groq_once_gemini_exhausts_its_retries():
    sleeps = []
    gemini = Scripted("gemini", [rate_limited("gemini")] * 3)
    groq = Scripted("groq", ["from groq"])
    r = FallbackLLM([gemini, groq], retries=2, backoff_s=1, sleep=sleeps.append).chat("s", [])
    assert r.provider == "groq" and gemini.calls == 3 and sleeps == [1, 2]


def test_waits_as_long_as_the_provider_asks_within_the_limit():
    sleeps = []
    slow = LLMError("gemini", "429", retryable=True, status=429, retry_after=46)
    gemini = Scripted("gemini", [slow, "ok"])
    FallbackLLM([gemini], max_wait_s=60, sleep=sleeps.append).chat("s", [])
    assert sleeps == [46]


def test_a_wait_longer_than_the_limit_moves_on_without_sleeping():
    sleeps = []
    gemini = Scripted("gemini", [LLMError("gemini", "429", retryable=True, status=429, retry_after=3600)])
    groq = Scripted("groq", ["ok"])
    assert FallbackLLM([gemini, groq], max_wait_s=60, sleep=sleeps.append).chat("s", []).provider == "groq"
    assert sleeps == [] and gemini.calls == 1


def test_non_retryable_error_moves_on_immediately():
    gemini = Scripted("gemini", [LLMError("gemini", "bad request", retryable=False, status=400)])
    groq = Scripted("groq", ["ok"])
    FallbackLLM([gemini, groq], sleep=lambda s: None).chat("s", [])
    assert gemini.calls == 1


def test_all_providers_failing_raises_with_every_error():
    gemini = Scripted("gemini", [LLMError("gemini", "down", retryable=False)])
    groq = Scripted("groq", [LLMError("groq", "down", retryable=False)])
    with pytest.raises(LLMUnavailable) as e:
        FallbackLLM([gemini, groq], sleep=lambda s: None).chat("s", [])
    assert [err.provider for err in e.value.errors] == ["gemini", "groq"]


def test_a_provider_that_cannot_replay_the_history_is_skipped():
    gemini = Scripted("gemini", ["never"], replays=False)
    groq = Scripted("groq", ["ok"])
    assert FallbackLLM([gemini, groq]).chat("s", []).provider == "groq" and gemini.calls == 0


def test_gemini_refuses_histories_continued_by_another_provider():
    gemini = GeminiLLM("key", "flash", client=object())
    assert gemini.can_replay([Message("user", "q"), Message("assistant", "a", provider="gemini/flash")])
    assert not gemini.can_replay([Message("assistant", "a", provider="groq")])
    assert not gemini.can_replay([Message("assistant", "a", provider="gemini/flash-lite")])


HISTORY = [
    Message("user", "q"),
    Message("assistant", "", [ToolCall("search", {"query": "a"}, id="c1"), ToolCall("search", {"query": "b"}, id="c2")],
            provider="groq"),
    Message("tool", tool_call_id="c1", tool_name="search", result={"chunks": []}),
    Message("tool", tool_call_id="c2", tool_name="search", result={"chunks": [1]}),
]


def test_gemini_groups_parallel_tool_results_into_one_turn():
    contents = to_contents(HISTORY)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [p.function_response.id for p in contents[2].parts] == ["c1", "c2"]


def test_gemini_replays_its_own_native_turn_verbatim():
    native = types.Content(role="model", parts=[types.Part(text="x", thought_signature=b"sig")])
    assert to_contents([Message("assistant", "x", provider="gemini", native=native)])[0] is native


def test_openai_format_carries_tool_call_ids_and_json_results():
    out = to_openai("sys", HISTORY)
    assert out[0] == {"role": "system", "content": "sys"}
    assert [c["id"] for c in out[2]["tool_calls"]] == ["c1", "c2"]
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"chunks": []})}


def test_malformed_groq_arguments_are_flagged_not_raised():
    call = SimpleNamespace(id="c", function=SimpleNamespace(name="search", arguments="{query: oops"))
    parsed = _parse_call(call)
    assert parsed.args == {} and "not valid JSON" in parsed.args_error


def test_compact_openai_format_shortens_only_older_tool_results():
    long = "x" * 1000
    history = [Message("user", "q")]
    for n in range(3):
        history += [Message("assistant", "", [ToolCall("search", {"query": str(n)}, id=f"c{n}")], provider="groq"),
                    Message("tool", tool_call_id=f"c{n}", tool_name="search",
                            result={"results": [{"source_id": f"S{n}", "title": "T", "text": long}]})]
    def lengths(keep_full):
        return [len(json.loads(m["content"])["results"][0]["text"]) for m in to_openai("s", history, keep_full)
                if m["role"] == "tool"]

    assert lengths(2) == [304, 1000, 1000] and lengths(0) == [304, 304, 304]
    assert all(m["role"] != "tool" or len(m["content"]) > 1000 for m in to_openai("s", history))
