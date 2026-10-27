from google.genai import types

from app.cache import Cache, CachedEmbedder, make_key
from app.llm.base import LLMResponse, Message, ToolCall, Usage
from app.llm.cached import CachedLLM
from app.retrieval.candidate import Candidate
from app.retrieval.searcher import SearchParams, SearchResult, Searcher


def test_get_set_and_generations(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    assert cache.get("ns", "k") is None
    cache.set("ns", "k", {"a": [1, 2]})
    assert Cache(tmp_path / "c.sqlite").get("ns", "k") == {"a": [1, 2]}  # persisted
    assert cache.get("ns", "k") == {"a": [1, 2]}
    assert cache.generation("corpus") == 0 and cache.bump("corpus") == 1 and cache.bump("corpus") == 2
    assert cache.stats["ns:hit"] == 1 and cache.stats["ns:miss"] == 1


def test_cached_embedder_calls_the_model_once_and_restores_sparse_tuples():
    class Counting:
        calls = 0

        def embed_query(self, text):
            self.calls += 1
            return [1, 2], [0.5, 0.5]

    inner = Counting()
    sparse = CachedEmbedder(inner, Cache(), "sparse")
    assert sparse.embed_query("q") == sparse.embed_query("q") == ([1, 2], [0.5, 0.5])
    assert inner.calls == 1


class CountingSearcher(Searcher):
    def __init__(self, cache):
        super().__init__(store=None, dense=None, sparse=None, cache=cache)
        self.calls = 0

    def _search(self, query, params):
        self.calls += 1
        c = Candidate("p1", {"doc_id": "d", "title": "T", "text": "x"}, dense_rank=1, ce_score=4.0)
        return SearchResult(query, params, [c], {"returned": 1}, {})


def test_search_cache_hits_until_the_corpus_changes():
    cache = Cache()
    searcher = CountingSearcher(cache)
    first = searcher.search("q")
    again = searcher.search("q")
    assert searcher.calls == 1 and again.results[0].ce_score == 4.0 and again.results[0] is not first.results[0]
    searcher.search("q", SearchParams(top_k=3))  # different parameters, different entry
    cache.bump("corpus")
    searcher.search("q")
    assert searcher.calls == 3


class OnceLLM:
    name = "gemini/m"

    def __init__(self):
        self.calls = 0

    def can_replay(self, messages):
        return True

    def chat(self, system, messages, tools=None, response_schema=None):
        self.calls += 1
        native = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
            id="c1", name="search_documents", args={"query": "a"}), thought_signature=b"\x01sig")])
        return LLMResponse("", [ToolCall("search_documents", {"query": "a"}, id="c1")], self.name, "m",
                           Usage(10, 2), native)


def test_llm_cache_replays_tool_calls_and_the_native_turn():
    inner = OnceLLM()
    llm = CachedLLM(inner, Cache())
    first = llm.chat("sys", [Message("user", "q")])
    hit = llm.chat("sys", [Message("user", "q")])
    assert inner.calls == 1
    assert hit.tool_calls[0].id == "c1" and hit.tool_calls[0].args == {"query": "a"}
    assert hit.native.parts[0].thought_signature == b"\x01sig" and hit.usage.input_tokens == 10
    llm.chat("sys", [Message("user", "another question")])
    assert inner.calls == 2 and first.provider == hit.provider


def test_keys_are_order_independent_for_dicts():
    assert make_key({"a": 1, "b": 2}) == make_key({"b": 2, "a": 1}) != make_key({"a": 2, "b": 1})
