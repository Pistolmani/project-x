"""Provider-neutral chat types (BACKEND.md §5).

The agent talks to `LLM.chat()` only, so Gemini and Groq are interchangeable behind it. Messages are
kept in a neutral form; an adapter may also attach its own `native` copy of a turn (Gemini needs its
thought signatures replayed exactly), which is why a provider can refuse a history it cannot replay.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments object


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    args_error: str | None = None  # set when the model produced unparseable arguments


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant turns
    tool_call_id: str | None = None  # tool turns: which call this answers
    tool_name: str | None = None
    result: dict[str, Any] | None = None  # tool turns: the tool's output
    provider: str | None = None  # assistant turns: who produced it
    native: Any = None  # assistant turns: the provider's own copy, for exact replay


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    native: Any = None

    def as_message(self) -> Message:
        return Message("assistant", self.text, self.tool_calls, provider=self.provider, native=self.native)


class LLMError(Exception):
    """A failed provider call. `retryable` = worth trying again (rate limit, timeout, 5xx)."""

    def __init__(self, provider: str, message: str, retryable: bool, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after  # seconds the provider asked us to wait, when it said


class LLMUnavailable(Exception):
    """Every provider failed; the API turns this into a structured error, never an invented answer."""

    def __init__(self, errors: list[LLMError]):
        super().__init__("; ".join(str(e) for e in errors) or "no LLM provider configured")
        self.errors = errors


class LLM(Protocol):
    name: str

    def can_replay(self, messages: list[Message]) -> bool: ...

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
