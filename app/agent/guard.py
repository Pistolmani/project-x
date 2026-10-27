"""Guardrails against hijacking and harmful output (BACKEND.md §5.5). Three layers around the agent:

    question -> input guard -> [agent + gates] -> output guard -> user
                               passages scanned for planted instructions

- Input guard: a policy classifier (gpt-oss-safeguard on Groq) reads the question against INPUT_POLICY
  (prompt injection, jailbreak, harmful requests). A blocked question never reaches the search or the LLM.
- Passage scan: plain patterns flag passages that contain instructions aimed at an AI. Flagged passages are
  still shown to the model (they may hold the answer), with a warning that they are data.
- Output guard: everything the model wrote that the user will read is checked for a leaked system prompt
  (verbatim overlap, plain code) and against OUTPUT_POLICY by the same classifier.

The classifier's free tier is small (2,000 tokens/min), so when it is unreachable the input guard falls
back to Prompt Guard 2 (a small injection detector) plus the patterns, and the output guard keeps only text
that already passed the grounding gates. Every decision is recorded in AgentAnswer.gates["guard"].
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field

from app.agent import prompts
from app.cache import Cache, make_key

# Instructions aimed at an AI, as they appear in injection attempts. Each needs wording addressed to a model,
# so ordinary prose does not match ("a disregard for the rules", "You are now entering Free Derry", a prison
# "jailbreak"). On the 50k-article corpus they flag 1 article (results/guardrails.json, corpus_scan); a flag
# only adds a warning, the passage is still used.
_GAP = r"[^.\n]{0,40}?"
_WHICH = r"(\s+(all|any|of|your|the|my|previous|prior|above|earlier|preceding|system's|these))+"
INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    rf"\b(ignore|disregard|forget|override){_WHICH}\s+(instructions?|rules|prompts?|guidelines|directions)\b",
    r"\b(system|developer)\s+(prompt|message|notice|instructions?)\b",
    r"\b(reveal|print|show|repeat|output|disclose)\s+(me\s+)?(your|the)\s+(system\s+|hidden\s+|initial\s+)?"
    r"(prompt|instructions)\b",
    r"\b(you are now|act as|pretend to be)\s+(an?\s+)?(unrestricted|uncensored|unfiltered|jailbroken|evil)\b",
    r"\bfrom now on,? you (are|will|must)\b",
    r"\b(developer mode|do anything now|DAN mode)\b",
    r"\b(note|message|notice) (to|for) the (ai|assistant|language model|llm|model)\b",
    rf"\b(answer|reply|respond)\b{_GAP}\b(every|all|each|any) questions?\b{_GAP}\bwith\b",
    rf"\bdo not\b{_GAP}\b(cite|mention|tell)\b{_GAP}\b(sources?|the user)\b",
]]  # fmt: skip
LEAK_NGRAM = 10  # this many consecutive words of a prompt in an answer = the prompt leaked
PROMPT_GUARD_THRESHOLD = 0.9  # Prompt Guard 2 probability that the text is an attack


def find_injection(text: str) -> str | None:
    """The first instruction-like phrase in `text`, or None."""
    for pattern in INJECTION_PATTERNS:
        if m := pattern.search(text):
            return m.group(0)
    return None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


_SECRET_PROMPTS = (prompts.INVESTIGATOR, prompts.VERIFIER, prompts.INPUT_POLICY, prompts.OUTPUT_POLICY)
_PROMPT_NGRAMS = set().union(*(_ngrams(_words(p), LEAK_NGRAM) for p in _SECRET_PROMPTS))


def leaks_prompt(text: str) -> bool:
    return bool(_ngrams(_words(text), LEAK_NGRAM) & _PROMPT_NGRAMS)


@dataclass
class GuardVerdict:
    allowed: bool
    category: str = "none"  # none | injection | harmful | prompt_leak | too_long
    reason: str = ""
    checks: list[str] = field(default_factory=list)  # what ran, in order
    error: str | None = None  # the classifier was unreachable: the decision came from the fallbacks

    def as_gate(self) -> dict:
        return asdict(self) | {"passed": self.allowed}


class GuardUnavailable(Exception):
    pass


class GroqGuardModel:
    """One Groq model used as a classifier. Results are cached: the eval repeats the same texts, and the
    free tier of the policy model is tiny."""

    def __init__(self, api_key: str, model: str, cache: Cache | None = None, timeout_s: float = 15,
                 max_wait_s: float = 20, client=None):  # fmt: skip
        import groq

        self.model = model
        self.cache = cache
        self.max_wait_s = max_wait_s
        self.client = client or groq.Groq(api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(self, system: str | None, text: str, max_tokens: int | None = None) -> str:
        key = make_key(self.model, system, text)
        if self.cache is not None and (hit := self.cache.get("guard", key)) is not None:
            return hit
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        kwargs = {"max_completion_tokens": max_tokens} if max_tokens else {}
        content = self._call(messages, kwargs)
        if self.cache is not None:
            self.cache.set("guard", key, content)
        return content

    def _call(self, messages: list[dict], kwargs: dict) -> str:
        import groq

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
                return response.choices[0].message.content or ""
            except groq.RateLimitError as e:
                wait = _retry_after(e)
                if attempt == 0 and wait is not None and wait <= self.max_wait_s:
                    time.sleep(wait)
                    continue
                raise GuardUnavailable(f"{self.model}: rate limited") from e
            except groq.APIError as e:
                raise GuardUnavailable(f"{self.model}: {type(e).__name__}") from e
        raise GuardUnavailable(f"{self.model}: rate limited")


def _retry_after(e) -> float | None:
    header = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
    try:
        return float(header) if header else None
    except ValueError:
        return None


class Guard:
    def __init__(self, classifier: GroqGuardModel | None = None, detector: GroqGuardModel | None = None,
                 max_question_chars: int = 1000):  # fmt: skip
        self.classifier = classifier  # policy model: injection + harmful, question and answer
        self.detector = detector  # Prompt Guard 2: injection only, the fallback for questions
        self.max_question_chars = max_question_chars

    def check_question(self, question: str) -> GuardVerdict:
        if len(question) > self.max_question_chars:
            return GuardVerdict(False, "too_long", f"questions are limited to {self.max_question_chars} characters",
                                ["length"])  # fmt: skip
        checks = ["length"]
        error = None  # set only when a configured classifier could not be reached
        if self.classifier is not None:
            checks.append("policy_classifier")
            try:
                return self._classify(prompts.INPUT_POLICY, question, checks)
            except GuardUnavailable as e:
                error = str(e)
        # Fallback: injection only (harmful requests still meet the grounding gates and the output guard).
        if self.detector is not None:
            checks.append("prompt_guard")
            try:
                score = float(self.detector.complete(None, question))
                if score >= PROMPT_GUARD_THRESHOLD:
                    return GuardVerdict(False, "injection", f"Prompt Guard score {score:.3f}", checks, error)
            except (GuardUnavailable, ValueError) as e:
                error = "; ".join(x for x in (error, str(e)) if x)
        checks.append("patterns")
        if phrase := find_injection(question):
            return GuardVerdict(False, "injection", f"instruction-like phrase: {phrase!r}", checks, error)
        return GuardVerdict(True, checks=checks, error=error)

    def check_answer(self, question: str, text: str) -> GuardVerdict:
        """`text` is what the model wrote that the user would read (without the fixed refusal sentence)."""
        checks = ["prompt_leak"]
        if leaks_prompt(text):
            return GuardVerdict(False, "prompt_leak", "the answer repeats the system's instructions", checks)
        if self.classifier is None:
            return GuardVerdict(True, checks=checks)
        checks.append("policy_classifier")
        try:
            return self._classify(prompts.OUTPUT_POLICY, prompts.output_guard_input(question, text), checks)
        except GuardUnavailable as e:
            return GuardVerdict(True, checks=checks, error=str(e))

    def _classify(self, policy: str, text: str, checks: list[str]) -> GuardVerdict:
        raw = self.classifier.complete(policy, text, max_tokens=prompts.POLICY_MAX_TOKENS)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
            violation = int(data["violation"]) == 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise GuardUnavailable(f"unparseable classifier verdict: {raw[:80]!r}") from e
        category = str(data.get("category") or "harmful") if violation else "none"
        return GuardVerdict(not violation, category, str(data.get("rationale", ""))[:300], checks)
