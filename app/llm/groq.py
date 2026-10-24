"""Groq adapter (fallback LLM, A2): OpenAI-style chat completions with tool calls."""

import json
import re

import groq

from app.llm.base import LLMError, LLMResponse, Message, ToolCall, Usage


MAX_COMPLETION_TOKENS = 2048


class GroqLLM:
    name = "groq"

    def __init__(self, api_key: str, model: str, timeout_s: float = 30, client=None):
        self.model = model
        self.client = client or groq.Groq(api_key=api_key, timeout=timeout_s, max_retries=0)

    def can_replay(self, messages: list[Message]) -> bool:
        return True  # rebuilt from the neutral form, so any history works

    def chat(self, system, messages, tools=None, response_schema=None) -> LLMResponse:
        # Groq counts prompt + max_completion_tokens against the 8k tokens/minute free tier, and gpt-oss's
        # default completion budget alone nearly fills it. Answers here are short (reasoning included).
        kwargs: dict = {"max_completion_tokens": MAX_COMPLETION_TOKENS}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]  # fmt: skip
        if response_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "schema": response_schema}}  # fmt: skip
        try:
            # 413: the whole request exceeds the free tier's 8k tokens/minute. Long investigations that fell
            # back here mid-way get there, so retry with search results shortened: first the older ones,
            # then all of them.
            for level, keep_full in enumerate([None, KEEP_FULL_TOOL_RESULTS, 0]):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=to_openai(system, messages, keep_full=keep_full), **kwargs
                    )
                    break
                except groq.APIStatusError as e:
                    if e.status_code != 413 or level == 2:
                        raise
        except (groq.RateLimitError, groq.InternalServerError) as e:
            raise LLMError(self.name, str(e)[:200], True, e.status_code, _retry_after(e)) from e
        except groq.APITimeoutError as e:
            raise LLMError(self.name, "timeout", retryable=True) from e
        except groq.APIConnectionError as e:
            raise LLMError(self.name, f"connection error: {e}", retryable=True) from e
        except groq.APIStatusError as e:
            # gpt-oss occasionally emits a call to a tool that does not exist ("tool_use_failed"): a
            # generation glitch, so worth another try.
            glitch = e.status_code == 400 and "tool_use_failed" in str(e)
            raise LLMError(self.name, str(e)[:400], retryable=glitch, status=e.status_code) from e

        message = response.choices[0].message
        calls = [_parse_call(c) for c in message.tool_calls or []]
        usage = Usage(response.usage.prompt_tokens, response.usage.completion_tokens) if response.usage else Usage()
        return LLMResponse(message.content or "", calls, self.name, self.model, usage)


def _retry_after(e: groq.APIStatusError) -> float | None:
    header = e.response.headers.get("retry-after") if e.response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in ([\d.]+)s", str(e))
    return float(match.group(1)) if match else None


def _parse_call(call) -> ToolCall:
    try:
        args, error = json.loads(call.function.arguments or "{}"), None
        if not isinstance(args, dict):
            args, error = {}, "arguments must be a JSON object"
    except json.JSONDecodeError as e:
        args, error = {}, f"arguments are not valid JSON: {e.msg}"
    return ToolCall(call.function.name, args, id=call.id, args_error=error)


KEEP_FULL_TOOL_RESULTS = 2
COMPACT_PASSAGE_CHARS = 300


def to_openai(system: str, messages: list[Message], keep_full: int | None = None) -> list[dict]:
    """Neutral messages -> OpenAI chat format. With `keep_full` set, all but the last `keep_full` tool
    results keep only the first COMPACT_PASSAGE_CHARS characters of each passage (source ids and titles
    stay, so earlier citations remain valid)."""
    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    if keep_full is None:
        shorten = set()
    else:
        shorten = set(tool_indices[: len(tool_indices) - keep_full])
    out = [{"role": "system", "content": system}]
    for i, m in enumerate(messages):
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            entry = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                entry["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                    "name": c.name, "arguments": json.dumps(c.args)}} for c in m.tool_calls]  # fmt: skip
            out.append(entry)
        else:
            result = _shortened(m.result) if i in shorten else m.result
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": json.dumps(result)})
    return out


def _shortened(result):
    if not isinstance(result, dict):
        return result
    short = dict(result)
    for key in ("results", "passages"):
        if isinstance(short.get(key), list):
            short[key] = [p | {"text": p["text"][:COMPACT_PASSAGE_CHARS] + " ..."}
                          if isinstance(p, dict) and len(p.get("text", "")) > COMPACT_PASSAGE_CHARS else p
                          for p in short[key]]  # fmt: skip
    return short
