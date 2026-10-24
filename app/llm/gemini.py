"""Gemini adapter (primary LLM, D3), on the official `google-genai` SDK."""

from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from app.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiLLM:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: float = 30, client=None):
        self.model = model
        # The SDK's own retries are off: retry and fallback policy live in FallbackLLM.
        self.client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )

    def can_replay(self, messages: list[Message]) -> bool:
        # Gemini rejects tool-call turns without the thought signatures it issued, so a history that
        # another provider continued cannot be handed back to it.
        return all(m.provider in (None, self.name) for m in messages if m.role == "assistant")

    def chat(self, system, messages, tools=None, response_schema=None) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if tools:
            config.tools = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description,
                                          parameters_json_schema=t.parameters) for t in tools])]  # fmt: skip
        if response_schema:
            config.response_mime_type = "application/json"
            config.response_json_schema = response_schema
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=to_contents(messages), config=config
            )
        except errors.APIError as e:
            raise LLMError(self.name, str(e)[:300], e.code in RETRYABLE_STATUS, e.code) from e
        except httpx.TimeoutException as e:
            raise LLMError(self.name, "timeout", retryable=True) from e
        except httpx.TransportError as e:
            raise LLMError(self.name, f"connection error: {e}", retryable=True) from e

        if not response.candidates or response.candidates[0].content is None:
            reason = response.prompt_feedback.block_reason if response.prompt_feedback else "empty"
            raise LLMError(self.name, f"no candidate returned ({reason})", retryable=False)
        content = response.candidates[0].content
        parts = content.parts or []
        calls = [ToolCall(p.function_call.name, dict(p.function_call.args or {}),
                          **({"id": p.function_call.id} if p.function_call.id else {}))
                 for p in parts if p.function_call]  # fmt: skip
        text = "".join(p.text for p in parts if p.text and not p.thought)
        meta = response.usage_metadata
        usage = Usage(meta.prompt_token_count or 0, meta.candidates_token_count or 0) if meta else Usage()
        return LLMResponse(text, calls, self.name, self.model, usage, native=content)


def to_contents(messages: list[Message]) -> list[types.Content]:
    """Neutral messages -> Gemini contents. Consecutive tool results become one user turn, which is
    how Gemini expects the answers to a turn with several function calls."""
    contents: list[types.Content] = []
    for m in messages:
        if m.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m.text)]))
        elif m.role == "assistant":
            if isinstance(m.native, types.Content):
                contents.append(m.native)  # exact replay, thought signatures included
                continue
            parts = [types.Part.from_text(text=m.text)] if m.text else []
            parts += [types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args))
                      for c in m.tool_calls]  # fmt: skip
            contents.append(types.Content(role="model", parts=parts))
        else:
            part = types.Part(function_response=types.FunctionResponse(
                id=m.tool_call_id, name=m.tool_name, response=_as_object(m.result)))  # fmt: skip
            if contents and contents[-1].role == "user" and contents[-1].parts[0].function_response:
                contents[-1].parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
    return contents


def _as_object(result: Any) -> dict[str, Any]:
    return result if isinstance(result, dict) else {"result": result}
