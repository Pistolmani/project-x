"""Groq adapter (fallback LLM, A2): OpenAI-style chat completions with tool calls."""

import json

import groq

from app.llm.base import LLMError, LLMResponse, Message, ToolCall, Usage


class GroqLLM:
    name = "groq"

    def __init__(self, api_key: str, model: str, timeout_s: float = 30, client=None):
        self.model = model
        self.client = client or groq.Groq(api_key=api_key, timeout=timeout_s, max_retries=0)

    def can_replay(self, messages: list[Message]) -> bool:
        return True  # rebuilt from the neutral form, so any history works

    def chat(self, system, messages, tools=None, response_schema=None) -> LLMResponse:
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]  # fmt: skip
        if response_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "schema": response_schema}}  # fmt: skip
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=to_openai(system, messages), **kwargs
            )
        except (groq.RateLimitError, groq.InternalServerError) as e:
            raise LLMError(self.name, str(e)[:300], retryable=True, status=e.status_code) from e
        except groq.APITimeoutError as e:
            raise LLMError(self.name, "timeout", retryable=True) from e
        except groq.APIConnectionError as e:
            raise LLMError(self.name, f"connection error: {e}", retryable=True) from e
        except groq.APIStatusError as e:
            raise LLMError(self.name, str(e)[:300], retryable=False, status=e.status_code) from e

        message = response.choices[0].message
        calls = [_parse_call(c) for c in message.tool_calls or []]
        usage = Usage(response.usage.prompt_tokens, response.usage.completion_tokens) if response.usage else Usage()
        return LLMResponse(message.content or "", calls, self.name, self.model, usage)


def _parse_call(call) -> ToolCall:
    try:
        args, error = json.loads(call.function.arguments or "{}"), None
        if not isinstance(args, dict):
            args, error = {}, "arguments must be a JSON object"
    except json.JSONDecodeError as e:
        args, error = {}, f"arguments are not valid JSON: {e.msg}"
    return ToolCall(call.function.name, args, id=call.id, args_error=error)


def to_openai(system: str, messages: list[Message]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            entry = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                entry["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                    "name": c.name, "arguments": json.dumps(c.args)}} for c in m.tool_calls]  # fmt: skip
            out.append(entry)
        else:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": json.dumps(m.result)})
    return out
