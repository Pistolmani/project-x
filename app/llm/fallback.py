"""Retry + provider fallback (BACKEND.md §5.5): Gemini first, Groq when Gemini throttles or fails."""

import logging
import time
from typing import Callable

from app.llm.base import LLM, LLMError, LLMResponse, LLMUnavailable, Message

log = logging.getLogger(__name__)


class FallbackLLM:
    """Tries each provider in order. Retryable errors (429, 5xx, timeouts) are retried with
    exponential backoff before moving on; any other error moves on at once. Providers that cannot
    replay the conversation so far are skipped, so once a question falls back to Groq it stays there."""

    name = "fallback"

    def __init__(self, providers: list[LLM], retries: int = 2, backoff_s: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):  # fmt: skip
        self.providers = providers
        self.retries = retries
        self.backoff_s = backoff_s
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
                    self.sleep(self.backoff_s * 2**attempt)
        raise LLMUnavailable(failures)


def build_llm(settings) -> FallbackLLM:
    """The configured provider chain; a provider without an API key is simply left out."""
    from app.llm.gemini import GeminiLLM
    from app.llm.groq import GroqLLM

    providers: list[LLM] = []
    if settings.gemini_api_key:
        providers.append(GeminiLLM(settings.gemini_api_key.get_secret_value(), settings.gemini_model,
                                   settings.llm_timeout_s))  # fmt: skip
    if settings.groq_api_key:
        providers.append(GroqLLM(settings.groq_api_key.get_secret_value(), settings.groq_model,
                                 settings.llm_timeout_s))  # fmt: skip
    return FallbackLLM(providers, retries=settings.llm_retries)
