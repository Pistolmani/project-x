"""Retry + provider fallback (BACKEND.md §5.5): Gemini first, Groq when Gemini throttles or fails."""

import logging
import time
from typing import Callable

from app.llm.base import LLM, LLMError, LLMResponse, LLMUnavailable, Message

log = logging.getLogger(__name__)


class FallbackLLM:
    """Tries each provider in order. A retryable error (429, 5xx, timeout) is retried after the delay the
    provider asked for (free tiers say "retry in 46s"), or an exponential backoff if it gave none, as long
    as the wait is at most `max_wait_s`; a longer wait or any other error moves on to the next provider.
    Providers that cannot replay the conversation so far are skipped, so a question that fell back to a
    different model stays there."""

    name = "fallback"

    def __init__(self, providers: list[LLM], retries: int = 2, backoff_s: float = 1.0, max_wait_s: float = 60,
                 sleep: Callable[[float], None] = time.sleep):  # fmt: skip
        self.providers = providers
        self.retries = retries
        self.backoff_s = backoff_s
        self.max_wait_s = max_wait_s
        self.sleep = sleep

    def can_replay(self, messages: list[Message]) -> bool:
        return any(p.can_replay(messages) for p in self.providers)

    def chat(self, system, messages, tools=None, response_schema=None) -> LLMResponse:
        failures: list[LLMError] = []
        for provider in self.providers:
            if not provider.can_replay(messages):
                continue
            for attempt in range(self.retries + 1):
                try:
                    return provider.chat(system, messages, tools, response_schema)
                except LLMError as e:
                    log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                    failures.append(e)
                    if not e.retryable or attempt == self.retries:
                        break
                    delay = e.retry_after if e.retry_after is not None else self.backoff_s * 2**attempt
                    if delay > self.max_wait_s:
                        break
                    self.sleep(delay)
        raise LLMUnavailable(failures)


def build_llm(settings, gemini_models: list[str] | None = None) -> FallbackLLM:
    """The configured provider chain; a provider without an API key is simply left out. Each Gemini model
    has its own free-tier quota, so listing two doubles the requests per minute before Groq is needed."""
    from app.llm.gemini import GeminiLLM
    from app.llm.groq import GroqLLM

    providers: list[LLM] = []
    if settings.gemini_api_key:
        key = settings.gemini_api_key.get_secret_value()
        for model in gemini_models or [settings.gemini_model, settings.gemini_fallback_model]:
            providers.append(GeminiLLM(key, model, settings.llm_timeout_s))
    if settings.groq_api_key:
        providers.append(GroqLLM(settings.groq_api_key.get_secret_value(), settings.groq_model,
                                 settings.llm_timeout_s))  # fmt: skip
    return FallbackLLM(providers, retries=settings.llm_retries, max_wait_s=settings.llm_max_wait_s)
