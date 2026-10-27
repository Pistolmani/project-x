"""In-process metrics behind `GET /metrics` (BACKEND.md §9): what the service did since it started.

Counters and recent latencies only, kept in memory: the free host restarts the process when it sleeps,
so the numbers describe the current run. Per-request detail (every search, LLM step and gate) is already
in each answer's `trace` and `gates`; this is the aggregate view over all requests.
"""

import math
import threading
import time
from collections import Counter, defaultdict, deque
from typing import Any


def percentile(values, q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))]


class Metrics:
    def __init__(self, window: int = 500, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.window = window
        self._lock = threading.Lock()
        self.requests: Counter[str] = Counter()  # "POST /ask" -> count
        self.errors: Counter[str] = Counter()  # route -> 5xx responses
        self.latency_ms: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))  # route -> recent
        self.answers: Counter[str] = Counter()  # answer status
        self.answer_ms: deque = deque(maxlen=window)
        self.llm_calls: Counter[str] = Counter()  # "provider/model" -> calls
        self.tokens: Counter[str] = Counter()  # input / output
        self.guard_blocks: Counter[str] = Counter()  # "input:injection", "output:harmful", ...
        self.guard_fallbacks = 0  # the policy classifier was unreachable

    def record_request(self, route: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests[route] += 1
            self.latency_ms[route].append(ms)
            if status >= 500:
                self.errors[route] += 1

    def record_answer(self, answer: dict[str, Any]) -> None:
        """One finished /ask or /ask/stream answer (as the JSON the API returns)."""
        with self._lock:
            self.answers[answer.get("status", "unknown")] += 1
            self.answer_ms.append(answer.get("ms", 0.0))
            for step in answer.get("trace") or []:
                if step.get("step") == "llm":
                    self.llm_calls[f"{step.get('provider')}/{step.get('model')}"] += 1
            usage = answer.get("usage") or {}
            self.tokens["input"] += usage.get("input_tokens", 0)
            self.tokens["output"] += usage.get("output_tokens", 0)
            for side, gate in ((answer.get("gates") or {}).get("guard") or {}).items():
                if not gate.get("passed"):
                    self.guard_blocks[f"{side}:{gate.get('category')}"] += 1
                if gate.get("error"):
                    self.guard_fallbacks += 1

    def snapshot(self, cache_stats: Counter | dict | None = None) -> dict[str, Any]:
        with self._lock:
            routes = {route: {"count": n, "errors_5xx": self.errors[route],
                              "p50_ms": percentile(self.latency_ms[route], 50),
                              "p95_ms": percentile(self.latency_ms[route], 95)}
                      for route, n in sorted(self.requests.items())}  # fmt: skip
            answers = {"total": sum(self.answers.values()), "by_status": dict(self.answers),
                       "p50_ms": percentile(self.answer_ms, 50), "p95_ms": percentile(self.answer_ms, 95),
                       "llm_calls": dict(self.llm_calls), "tokens": dict(self.tokens)}  # fmt: skip
            guard = {"blocked": dict(self.guard_blocks), "classifier_fallbacks": self.guard_fallbacks}
        return {"uptime_s": round(self.clock() - self.started), "latency_window": self.window,
                "requests": routes, "answers": answers, "guard": guard, "cache": _cache_rates(cache_stats or {})}  # fmt: skip


def _cache_rates(stats) -> dict[str, dict[str, Any]]:
    """{"search:hit": 3, "search:miss": 1} -> {"search": {"hits": 3, "misses": 1, "hit_rate": 0.75}}"""
    out: dict[str, dict[str, Any]] = {}
    for key, n in stats.items():
        ns, _, kind = key.rpartition(":")
        out.setdefault(ns, {"hits": 0, "misses": 0})["hits" if kind == "hit" else "misses"] += n
    for v in out.values():
        total = v["hits"] + v["misses"]
        v["hit_rate"] = round(v["hits"] / total, 3) if total else None
    return dict(sorted(out.items()))
