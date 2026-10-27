"""Aggregate metrics (app/metrics.py) and the GET /metrics endpoint."""

from app.metrics import Metrics, percentile
from tests.test_api import make_client


def test_percentile_is_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(range(1, 101), 95) == 95


def test_answers_count_outcomes_providers_tokens_and_guard_blocks():
    m = Metrics(clock=iter([0.0, 90.0]).__next__)
    m.record_answer({"status": "answered", "ms": 2000.0, "usage": {"input_tokens": 100, "output_tokens": 10},
                     "trace": [{"step": "llm", "provider": "gemini/lite", "model": "lite"}, {"step": "tool"},
                               {"step": "llm", "provider": "groq", "model": "oss"}],
                     "gates": {"guard": {"input": {"passed": True}, "output": {"passed": True, "error": "rate limited"}}}})
    m.record_answer({"status": "blocked", "ms": 300.0, "gates": {"guard": {"input": {"passed": False, "category": "injection"}}}})
    snap = m.snapshot({"search:hit": 3, "search:miss": 1, "llm:miss": 2})
    assert snap["uptime_s"] == 90
    assert snap["answers"]["by_status"] == {"answered": 1, "blocked": 1}
    assert snap["answers"]["llm_calls"] == {"gemini/lite": 1, "groq/oss": 1}
    assert snap["answers"]["tokens"] == {"input": 100, "output": 10}
    assert snap["guard"] == {"blocked": {"input:injection": 1}, "classifier_fallbacks": 1}
    assert snap["cache"]["search"] == {"hits": 3, "misses": 1, "hit_rate": 0.75}
    assert snap["cache"]["llm"]["hit_rate"] == 0.0


def test_metrics_endpoint_groups_requests_by_route_template():
    with make_client() as c:
        doc = c.post("/documents", files={"file": ("jaws.md", b"# Jaws\n\nJaws is a 1975 shark film.\n")}).json()
        c.get(f"/documents/{doc['doc_id']}")
        c.get("/documents/no-such-id")
        c.post("/ask", json={"question": "Who directed Jaws?"})
        snap = c.get("/metrics").json()
    routes = snap["requests"]
    assert routes["GET /documents/{doc_id}"]["count"] == 2  # one entry for every id
    assert routes["POST /documents"]["count"] == 1 and routes["POST /ask"]["p50_ms"] is not None
    assert snap["answers"]["total"] == 1 and snap["answers"]["llm_calls"] == {"scripted/m": 2}
