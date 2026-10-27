"""Performance on the full index (EVALUATION-REPORT.md §7): search latency per stage and embedding throughput.

    python -m evaluation.benchmark                      # torch (GPU if present) and onnx (CPU), 100 questions
    python -m evaluation.benchmark --backends onnx --n 50

No LLM calls and no writes: every search runs the full retrieval pipeline (embed -> Qdrant -> RRF ->
feature re-ranker -> cross-encoder -> cap) against the configured index with the cache off, one question
at a time, after a short warm-up. "onnx" is what the live demo runs (CPU only); "torch" is the local
evaluation setup. Qdrant time includes the network round trip from this machine to the cluster.
Embedding throughput (dense + BM25 over real corpus chunks) is the dominant cost of ingesting a document.
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone

from app.config import get_settings
from app.metrics import percentile
from app.retrieval.searcher import SearchParams, Searcher
from app.store.qdrant_store import QdrantStore

STAGES = ["embed", "qdrant", "fusion", "feature_rerank", "cross_encoder", "total"]
WARMUP = 3


def summarize(values: list[float]) -> dict[str, float | None]:
    return {"p50": percentile(values, 50), "p95": percentile(values, 95),
            "mean": round(sum(values) / len(values), 1) if values else None}


def search_latency(backend: str, questions: list[str]) -> dict:
    from app.ingestion.embedding import DenseEmbedder, SparseEmbedder
    from app.retrieval.cross_encoder import CrossEncoderReranker

    settings = get_settings()
    dense = DenseEmbedder(settings.dense_model, backend=backend)
    searcher = Searcher(QdrantStore.from_settings(settings), dense, SparseEmbedder(settings.sparse_model),
                        CrossEncoderReranker(backend=backend))  # cache off: every search does the full work
    for q in questions[:WARMUP]:
        searcher.search(q, SearchParams())
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    for q in questions:
        start = time.perf_counter()
        result = searcher.search(q, SearchParams())
        per_stage["total"].append(round((time.perf_counter() - start) * 1000, 1))
        for stage, ms in result.timings_ms.items():
            per_stage[stage].append(ms)
    device = getattr(dense, "device", "cpu")
    return {"backend": backend, "device": device, "questions": len(questions),
            "ms": {stage: summarize(v) for stage, v in per_stage.items() if v}}


def embedding_throughput(backend: str, texts: list[str]) -> dict:
    from app.ingestion.embedding import DenseEmbedder, SparseEmbedder

    settings = get_settings()
    dense, sparse = DenseEmbedder(settings.dense_model, backend=backend), SparseEmbedder(settings.sparse_model)
    dense.embed_documents(texts[:8])  # warm-up
    start = time.perf_counter()
    dense.embed_documents(texts)
    sparse.embed_documents(texts)
    seconds = time.perf_counter() - start
    return {"backend": backend, "chunks": len(texts), "seconds": round(seconds, 1),
            "chunks_per_s": round(len(texts) / seconds, 1)}


def corpus_chunks(n: int, seed: int = 7) -> list[str]:
    """`n` chunk-sized (~1,000 character) windows from random corpus articles."""
    path = get_settings().data_dir / "corpus" / "docs.jsonl"
    docs = [json.loads(line)["text"] for line in path.open(encoding="utf-8")]
    rng = random.Random(seed)
    texts = []
    for text in rng.sample(docs, n):
        start = rng.randint(0, max(0, len(text) - 1000))
        texts.append(text[start : start + 1000])
    return texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", default="torch,onnx")
    p.add_argument("--n", type=int, default=100, help="questions from the tuning set")
    p.add_argument("--chunks", type=int, default=1000, help="chunks for the embedding throughput test")
    args = p.parse_args()

    settings = get_settings()
    sets = json.loads((settings.data_dir / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = [q["question"] for q in sets["tuning"][: args.n]]
    texts = corpus_chunks(args.chunks)
    store = QdrantStore.from_settings(settings)
    result = {"what": "search latency per stage and embedding throughput (evaluation/benchmark.py)",
              "date": datetime.now(timezone.utc).date().isoformat(),
              "index": {"documents": store.count(store.documents), "chunks": store.count(store.chunks)},
              "search": [], "embedding": []}
    for backend in args.backends.split(","):
        print(f"[{backend}] search latency over {len(questions)} questions…")
        result["search"].append(search_latency(backend, questions))
        print(f"[{backend}] embedding {len(texts)} chunks…")
        result["embedding"].append(embedding_throughput(backend, texts))
    path = settings.results_dir / "benchmark.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for s in result["search"]:
        print(s["backend"], s["device"], {k: v["p50"] for k, v in s["ms"].items()})
    for e in result["embedding"]:
        print(e)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
