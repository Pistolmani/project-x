"""Caches LLM responses (BACKEND.md §6) keyed by the full request, so a repeated question, or the eval
re-running the same questions, costs no quota. A hit replays the stored response exactly, including
Gemini's native turn, so the conversation can continue on the provider that produced it."""

from dataclasses import asdict

from app.cache import Cache, make_key
from app.llm.base import LLM, LLMResponse, Message, ToolCall, Usage


class CachedLLM:
    def __init__(self, llm: LLM, cache: Cache, ns: str = "llm"):
        self.llm = llm
        self.cache = cache
        self.ns = ns
        self.name = getattr(llm, "name", "llm")

    def can_replay(self, messages: list[Message]) -> bool:
        return self.llm.can_replay(messages)

    def chat(self, system, messages, tools=None, response_schema=None) -> LLMResponse:
        key = make_key(system, [_neutral(m) for m in messages], [asdict(t) for t in tools or []], response_schema)
        hit = self.cache.get(self.ns, key)
        if hit is not None:
            return _restore(hit)
        response = self.llm.chat(system, messages, tools, response_schema)
        self.cache.set(self.ns, key, _store(response))
        return response


def _neutral(m: Message) -> dict:
    # The native copy is left out: it carries opaque signatures that differ between identical answers.
    return {"role": m.role, "text": m.text, "calls": [(c.name, c.args) for c in m.tool_calls],
            "tool": m.tool_name, "result": m.result, "provider": m.provider}  # fmt: skip


def _store(r: LLMResponse) -> dict:
    native = r.native.model_dump(mode="json", exclude_none=True) if r.native is not None else None
    return {"text": r.text, "tool_calls": [asdict(c) for c in r.tool_calls], "provider": r.provider,
            "model": r.model, "usage": asdict(r.usage), "native": native}  # fmt: skip


def _restore(d: dict) -> LLMResponse:
    native = None
    if d["native"] is not None:
        from google.genai import types

        native = types.Content.model_validate(d["native"])
    return LLMResponse(d["text"], [ToolCall(**c) for c in d["tool_calls"]], d["provider"], d["model"],
                       Usage(**d["usage"]), native)  # fmt: skip
