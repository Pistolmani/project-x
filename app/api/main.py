"""FastAPI app (BACKEND.md §9): the one surface the UI and the evaluation suite both drive.

    uvicorn app.api.main:app --port 8000

Endpoints are plain `def`, so FastAPI runs them in its thread pool and the blocking model/Qdrant/LLM
calls never stall the event loop. Every failure is a JSON body {"error": {"code", "message"}}.
"""

import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.api.schemas import (AskRequest, DocumentList, IngestResponse, SearchHit, SearchRequest,
                             SearchResponse)  # fmt: skip
from app.ingestion.loaders import LoaderError
from app.ingestion.service import DocumentNotFound

log = logging.getLogger(__name__)


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
                                                   cache=built.cache)  # fmt: skip
        else:
            app.state.engine, app.state.ingestion = engine, ingestion
        yield

    app = FastAPI(title="Investigation Engine", version="0.2", lifespan=lifespan)

    def error(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

    @app.exception_handler(LoaderError)
    def _loader_error(request: Request, e: LoaderError):
        return error(422, "unprocessable_document", str(e))

    @app.exception_handler(DocumentNotFound)
    def _not_found(request: Request, e: DocumentNotFound):
        return error(404, "document_not_found", f"no document {e.args[0]}")

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
                "cache": dict(engine.cache.stats)}  # fmt: skip

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request):
        result = request.app.state.engine.searcher.search(body.query, body.params)
        hits = [SearchHit(chunk_id=c.point_id, doc_id=c.doc_id, title=c.title, text=c.text,
                          chunk_index=c.payload.get("chunk_index"), start_char=c.payload.get("start_char"),
                          end_char=c.payload.get("end_char"), trace=c.trace()) for c in result.results]  # fmt: skip
        return SearchResponse(query=body.query, results=hits, stage_sizes=result.stage_sizes,
                              timings_ms=result.timings_ms)  # fmt: skip

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        answer = request.app.state.engine.agent.ask(body.question, body.params)
        # An LLM or store outage is a 503 that still carries the structured answer (status "error"),
        # never a fabricated reply.
        return JSONResponse(status_code=503 if answer.status == "error" else 200, content=_jsonable(asdict(answer)))

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request):
        """Server-Sent Events stream: live progress events while the agent works, then the final answer."""
        agent = request.app.state.engine.agent
        events: queue.Queue[dict | None] = queue.Queue()

        def run():
            try:
                answer = agent.ask(body.question, body.params, on_event=events.put)
                events.put({"type": "answer", "answer": _jsonable(asdict(answer))})
            except Exception as e:  # anything ask() does not turn into status "error" itself
                log.exception("ask/stream failed")
                events.put({"type": "error", "code": "internal_error",
                            "message": f"the investigation failed: {type(e).__name__}"})  # fmt: skip
            finally:
                events.put(None)

        threading.Thread(target=run, daemon=True).start()

        def generate():
            while (ev := events.get()) is not None:
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/documents", response_model=IngestResponse, status_code=201)
    def add_document(request: Request, file: UploadFile = File(...)):
        result = request.app.state.ingestion.add(file.file.read(), file.filename or "upload.txt")
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
                  for r in store.get_chunks(doc_id, strategy)]  # fmt: skip
        return {"document": record, "chunks": chunks}

    @app.put("/documents/{doc_id}", response_model=IngestResponse)
    def update_document(doc_id: str, request: Request, file: UploadFile = File(...)):
        result = request.app.state.ingestion.update(doc_id, file.file.read(), file.filename or "upload.txt")
        return IngestResponse(**asdict(result))

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, request: Request):
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
