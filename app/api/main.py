"""FastAPI app (BACKEND.md §9): the one surface the UI and the evaluation suite both drive.

    uvicorn app.api.main:app --port 8000

Endpoints are plain `def`, so FastAPI runs them in its thread pool and the blocking model/Qdrant/LLM
calls never stall the event loop. Every failure is a JSON body {"error": {"code", "message"}}.
"""

import json
import logging
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.api.schemas import (AskRequest, DocumentList, IngestResponse, SearchHit, SearchRequest,
                             SearchResponse)
from app.ingestion.loaders import LoaderError
from app.ingestion.service import DocumentNotFound
from app.metrics import Metrics

log = logging.getLogger(__name__)

DEMO_TAG = "demo_upload"  # what the public demo tags uploads with; only these may be changed there


class DemoForbidden(Exception):
    pass


class UploadTooLarge(Exception):
    pass


class AskLimited(Exception):
    pass


class HourlyLimit:
    """Sliding one-hour window over all callers. The UI reaches the API over localhost, so every visitor
    shares one client IP and a per-IP limit would mean nothing; this caps the shared LLM quota instead."""

    def __init__(self, per_hour: int, clock=time.monotonic):
        self.per_hour, self.clock = per_hour, clock
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def take(self) -> None:
        if self.per_hour <= 0:
            return
        with self.lock:
            now = self.clock()
            while self.calls and now - self.calls[0] >= 3600:
                self.calls.popleft()
            if len(self.calls) >= self.per_hour:
                raise AskLimited(int(3600 - (now - self.calls[0])) + 1)
            self.calls.append(now)


def create_app(engine=None, ingestion=None) -> FastAPI:
    """Pass `engine`/`ingestion` to skip model loading (tests); otherwise they are built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            from app.engine import build_engine
            from app.ingestion.chunking import load_tokenizer
            from app.ingestion.service import IngestionService

            built = build_engine()
            app.state.engine = built
            app.state.ingestion = IngestionService(built.store, built.dense, built.sparse,
                                                   load_tokenizer(built.settings.dense_model), built.settings,
                                                   cache=built.cache)
        else:
            app.state.engine, app.state.ingestion = engine, ingestion
        app.state.ask_limit = HourlyLimit(_settings(app).ask_limit_per_hour)
        yield

    app = FastAPI(title="Investigation Engine", version="0.2", lifespan=lifespan)
    app.state.metrics = Metrics()

    @app.middleware("http")
    async def _measure(request: Request, call_next):
        start, status = time.perf_counter(), 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # The route template, not the raw path, so /documents/{doc_id} is one entry, not one per id.
            route = request.scope.get("route")
            label = f"{request.method} {getattr(route, 'path', 'unmatched')}"
            request.app.state.metrics.record_request(label, status, round((time.perf_counter() - start) * 1000, 1))

    def error(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

    @app.exception_handler(LoaderError)
    def _loader_error(request: Request, e: LoaderError):
        return error(422, "unprocessable_document", str(e))

    @app.exception_handler(DocumentNotFound)
    def _not_found(request: Request, e: DocumentNotFound):
        return error(404, "document_not_found", f"no document {e.args[0]}")

    @app.exception_handler(DemoForbidden)
    def _forbidden(request: Request, e: DemoForbidden):
        return error(403, "demo_read_only", "the public demo only changes documents uploaded there; "
                                            "the corpus and the evaluation documents are read-only")

    @app.exception_handler(UploadTooLarge)
    def _too_large(request: Request, e: UploadTooLarge):
        return error(413, "upload_too_large", f"uploads are limited to {e.args[0]:g} MB")

    @app.exception_handler(AskLimited)
    def _limited(request: Request, e: AskLimited):
        return JSONResponse(status_code=429, headers={"Retry-After": str(e.args[0])},
                            content={"error": {"code": "ask_limit", "message": "the demo's hourly question limit "
                                               f"is used up (it protects the free LLM quota); try again in "
                                               f"{e.args[0] // 60 + 1} min"}})

    @app.exception_handler(UnexpectedResponse)
    @app.exception_handler(ResponseHandlingException)
    def _store_down(request: Request, e: Exception):
        log.exception("vector store error")
        return error(503, "vector_store_unavailable", "the vector store did not respond; try again shortly")

    @app.get("/health")
    def health(request: Request):
        engine = request.app.state.engine
        try:
            documents = engine.store.count(engine.store.documents)
            store_ok = True
        except (UnexpectedResponse, ResponseHandlingException):
            documents, store_ok = None, False
        return {"status": "ok" if store_ok else "degraded", "vector_store": store_ok, "documents": documents,
                "llm_providers": [p.name for p in getattr(engine.llm.llm, "providers", [])],
                "cache": dict(engine.cache.stats)}

    @app.get("/metrics")
    def metrics(request: Request):
        return request.app.state.metrics.snapshot(request.app.state.engine.cache.stats)

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request):
        result = request.app.state.engine.searcher.search(body.query, body.params)
        hits = [SearchHit(chunk_id=c.point_id, doc_id=c.doc_id, title=c.title, text=c.text,
                          chunk_index=c.payload.get("chunk_index"), start_char=c.payload.get("start_char"),
                          end_char=c.payload.get("end_char"), trace=c.trace()) for c in result.results]
        return SearchResponse(query=body.query, results=hits, stage_sizes=result.stage_sizes,
                              timings_ms=result.timings_ms)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        request.app.state.ask_limit.take()
        _force_guard(request, body)
        answer = request.app.state.engine.agent.ask(body.question, body.params)
        content = _jsonable(asdict(answer))
        _record_answer(request.app, content)
        # An LLM or store outage is a 503 that still carries the structured answer (status "error"),
        # never a fabricated reply.
        return JSONResponse(status_code=503 if answer.status == "error" else 200, content=content)

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request):
        """Server-Sent Events stream: live progress events while the agent works, then the final answer."""
        request.app.state.ask_limit.take()
        _force_guard(request, body)
        agent = request.app.state.engine.agent
        events: queue.Queue[dict | None] = queue.Queue()

        def run():
            try:
                answer = _jsonable(asdict(agent.ask(body.question, body.params, on_event=events.put)))
                _record_answer(request.app, answer)
                events.put({"type": "answer", "answer": answer})
            except Exception as e:  # anything ask() does not turn into status "error" itself
                log.exception("ask/stream failed")
                events.put({"type": "error", "code": "internal_error",
                            "message": f"the investigation failed: {type(e).__name__}"})
            finally:
                events.put(None)

        threading.Thread(target=run, daemon=True).start()

        def generate():
            while (ev := events.get()) is not None:
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/documents", response_model=IngestResponse, status_code=201)
    def add_document(request: Request, file: UploadFile = File(...), tag: str | None = Form(None)):
        if _settings(request.app).demo_mode:
            tag = DEMO_TAG  # visitors cannot label an upload as corpus or eval data
        result = request.app.state.ingestion.add(_read(request, file), file.filename or "upload.txt", tag=tag)
        if result.status == "duplicate":
            return JSONResponse(status_code=200, content=asdict(result))
        return IngestResponse(**asdict(result))

    @app.get("/documents", response_model=DocumentList)
    def list_documents(
        request: Request,
        limit: int = Query(50, ge=1, le=500),
        offset: str | None = None,
        format: str | None = None,
        exclude_format: str | None = None,
    ):
        documents, next_offset = request.app.state.engine.store.list_documents(
            limit, offset, fmt=format, exclude_format=exclude_format
        )
        return DocumentList(documents=documents, next_offset=next_offset)

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str, request: Request, strategy: str = "fixed"):
        store = request.app.state.engine.store
        record = store.get_document(doc_id)
        if record is None:
            raise DocumentNotFound(doc_id)
        chunks = [{"chunk_id": str(r.id), **{k: r.payload.get(k) for k in
                   ("chunk_index", "text", "start_char", "end_char", "n_tokens", "version")}}
                  for r in store.get_chunks(doc_id, strategy)]
        return {"document": record, "chunks": chunks}

    @app.put("/documents/{doc_id}", response_model=IngestResponse)
    def update_document(doc_id: str, request: Request, file: UploadFile = File(...)):
        _check_mutable(request, doc_id)
        result = request.app.state.ingestion.update(doc_id, _read(request, file), file.filename or "upload.txt")
        return IngestResponse(**asdict(result))

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, request: Request):
        _check_mutable(request, doc_id)
        request.app.state.ingestion.delete(doc_id)

    @app.get("/eval/results")
    def eval_results(request: Request):
        settings = getattr(request.app.state.engine, "settings", None)
        results_dir = getattr(settings, "results_dir", None) or (Path(__file__).resolve().parents[2] / "results")
        out = {}
        if Path(results_dir).is_dir():
            for path in sorted(Path(results_dir).glob("*.json")):
                out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        return out

    return app


def _settings(app: FastAPI):
    from app.config import get_settings

    return getattr(app.state.engine, "settings", None) or get_settings()


def _read(request: Request, file: UploadFile) -> bytes:
    limit = _settings(request.app).max_upload_mb
    data = file.file.read(int(limit * 1024 * 1024) + 1)
    if len(data) > limit * 1024 * 1024:
        raise UploadTooLarge(limit)
    return data


def _record_answer(app: FastAPI, answer: dict) -> None:
    """Count the answer in /metrics and log one JSON line (without the question text)."""
    app.state.metrics.record_answer(answer)
    providers = sorted({f"{s.get('provider')}/{s.get('model')}" for s in answer.get("trace") or [] if s.get("step") == "llm"})
    guard = (answer.get("gates") or {}).get("guard") or {}
    log.info("ask %s", json.dumps({
        "status": answer.get("status"), "ms": answer.get("ms"), "question_chars": len(answer.get("question", "")),
        "llm_calls": (answer.get("usage") or {}).get("llm_calls"), "providers": providers,
        "citations": len(answer.get("citations") or []),
        "guard_blocked": [f"{k}:{v.get('category')}" for k, v in guard.items() if not v.get("passed")],
        "error": answer.get("error")}))


def _force_guard(request: Request, body) -> None:
    if _settings(request.app).demo_mode:
        body.params.guard = True


def _check_mutable(request: Request, doc_id: str) -> None:
    if not _settings(request.app).demo_mode:
        return
    record = request.app.state.engine.store.get_document(doc_id)
    if record is None:
        raise DocumentNotFound(doc_id)
    if record.get("tag") != DEMO_TAG:
        raise DemoForbidden(doc_id)


def _jsonable(value):
    """Floats like -inf from score fields are not valid JSON; map them to None."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


app = create_app()
