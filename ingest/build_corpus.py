"""Build the corpus and eval sets, then chunk, embed and index them into Qdrant.

    python -m ingest.build_corpus --stage build  --target 50000 --seed 42
    python -m ingest.build_corpus --stage index  --strategies recursive,fixed

Stage `build` (CPU): select questions + titles, load articles, swap leads, absence-check the
unanswerable set, and write data/corpus/docs.jsonl, data/eval/questions.json, data/manifest.json.
Stage `index` (GPU if present): chunk every document with each strategy, embed dense + sparse, and
upsert to Qdrant. Re-running is safe: IDs are deterministic and (content, strategy) pairs that are
already indexed are skipped. The two stages are separate so a failed upload never forces a rebuild.
"""

import argparse
import json
import statistics
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from itertools import islice
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import ChunkingStrategy, chunk, load_tokenizer
from app.ingestion.document import SectionSpans, StructuredText
from app.ingestion.text import content_hash
from app.store.qdrant_store import NUM_CHUNKS_PREFIX, ChunkPoint, QdrantStore, stable_id
from ingest import hotpot, wiki
from ingest.select import Selection, SelectionConfig, select, trim_to_target


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    paths = _Paths(settings.data_dir)
    if args.stage in ("build", "all"):
        build(args, paths)
    if args.stage in ("index", "all"):
        strategies = [ChunkingStrategy(s) for s in args.strategies.split(",")]
        index(paths, settings, strategies, args.batch_docs)


# --- stage 1: build ---------------------------------------------------------------------------


def build(args: argparse.Namespace, paths: "_Paths") -> None:
    started = time.time()
    config = SelectionConfig(
        seed=args.seed, target_docs=args.target, retrieval=args.retrieval, tuning=args.tuning
    )

    print("Loading HotpotQA ...")
    questions, paragraphs = hotpot.load()
    print(f"  {len(questions):,} questions, {len(paragraphs):,} distinct articles")

    print("Downloading / locating Wikipedia shards (~11.6 GB the first time) ...")
    files = wiki.download(paths.wiki)
    wiki_titles = wiki.read_titles(files)
    available = set(paragraphs) & wiki_titles
    unresolved = len(paragraphs) - len(available)
    print(f"  {len(available):,} HotpotQA articles found in {wiki.WIKI_CONFIG}; {unresolved:,} unresolved")

    sel = select(questions, available, config)
    print(f"Selected {len(sel.corpus_titles):,} titles ({len(sel.held_out)} held out). Reading articles ...")
    articles = wiki.read_articles(files, set(sel.corpus_titles))

    docs, duplicates = _dedup([
        wiki.build_doc(stable_id(f"wiki:{t}"), t, articles[t][0], articles[t][1], paragraphs[t])
        for t in sel.corpus_titles
    ])  # fmt: skip
    # The fill was over-selected by a margin; trim back so dedup drops never leave us under target.
    keep = set(trim_to_target([d.title for d in docs], sel.required_titles, config.target_docs))
    docs = [d for d in docs if d.title in keep]
    if len(docs) < config.target_docs:
        raise RuntimeError(f"only {len(docs):,} documents, target {config.target_docs:,}")
    unanswerable, failed_absence = _absence_check(sel, docs, config.unanswerable)

    paths.write_docs(docs)
    paths.write_json(paths.questions, _eval_sets(sel, unanswerable, docs))
    paths.write_json(
        paths.manifest,
        {
            "created_at": _now(),
            "selection_config": asdict(config),
            "sources": {"hotpotqa": f"{hotpot.HOTPOT_REPO}/{hotpot.HOTPOT_CONFIG}",
                        "wikipedia": f"{wiki.WIKI_REPO}/{wiki.WIKI_CONFIG}"},
            "counts": {
                "documents": len(docs),
                "hotpot_articles": len(paragraphs),
                "unresolved_titles": unresolved,
                "lead_replaced": sum(d.lead_replaced for d in docs),
                "exact_duplicates_dropped": len(duplicates),
            },
            "duplicate_titles": duplicates,
            "held_out_titles": sorted(sel.held_out),
            "unanswerable_failed_absence_check": failed_absence,
            "documents": [
                {"doc_id": d.doc_id, "title": d.title, "sha256": d.sha256, "lead_replaced": d.lead_replaced}
                for d in docs
            ],
        },
    )  # fmt: skip
    print(f"Build done in {time.time() - started:.0f}s: {len(docs):,} docs -> {paths.docs}")


def _dedup(docs: list[wiki.CorpusDoc]) -> tuple[list[wiki.CorpusDoc], list[str]]:
    """Exact dedup by content hash (DATABASE.md §4); the first title wins."""
    seen: set[str] = set()
    kept, dropped = [], []
    for d in docs:
        if d.sha256 in seen:
            dropped.append(d.title)
        else:
            seen.add(d.sha256)
            kept.append(d)
    return kept, dropped


def _absence_check(sel: Selection, docs: list[wiki.CorpusDoc], needed: int):
    """Keep unanswerable candidates whose gold answer appears in no remaining document."""
    lowered = [d.structured.text.lower() for d in docs]
    kept, failed = [], []
    for q in sel.unanswerable_candidates:
        if any(q.answer.lower() in text for text in lowered):
            failed.append({"id": q.id, "answer": q.answer})
        elif len(kept) < needed:
            kept.append(q)
    if len(kept) < needed:
        raise RuntimeError(f"only {len(kept)} unanswerable questions passed the absence check")
    return kept, failed


def _eval_sets(sel: Selection, unanswerable, docs: list[wiki.CorpusDoc]) -> dict:
    by_title = {d.title: d for d in docs}

    def record(q: hotpot.Question, expected: str) -> dict:
        gold = []
        for title, sent_id in q.supporting:
            doc = by_title.get(title)
            in_range = doc is not None and sent_id < len(doc.gold_sentence_spans)
            gold.append({
                "title": title,
                "sentence_index": sent_id,
                "doc_id": doc.doc_id if doc else None,  # None: held out on purpose
                "span": doc.gold_sentence_spans[sent_id] if in_range else None,
            })  # fmt: skip
        return {"id": q.id, "question": q.question, "answer": q.answer, "hotpot_type": q.type,
                "level": q.level, "expected_behavior": expected, "gold": gold}  # fmt: skip

    return {
        "behavioral": {
            "simple": [record(q, "answer_with_citation") for q in sel.simple],
            "multihop": [record(q, "answer_with_all_hops_cited") for q in sel.multihop],
            "unanswerable": [record(q, "refuse") for q in unanswerable],
            # Conflicting documents are injected in the evaluation phase; these questions are
            # chosen now so that their gold articles are guaranteed to be in the corpus.
            "contradictory": [record(q, "surface_conflict") for q in sel.contradictory],
        },
        "retrieval": [record(q, "retrieve_gold") for q in sel.retrieval],
        "tuning": [record(q, "retrieve_gold") for q in sel.tuning],
    }


# --- stage 2: index ---------------------------------------------------------------------------


def index(paths: "_Paths", settings, strategies: list[ChunkingStrategy], batch_docs: int) -> None:
    from tqdm import tqdm

    from app.ingestion.embedding import DenseEmbedder, SparseEmbedder

    # Documents are streamed from docs.jsonl one batch at a time, never all in memory: the full
    # corpus as Python objects is well over 1 GB, and a run was killed for low system memory.
    n_docs = paths.count_docs()
    tokenizer = load_tokenizer(settings.dense_model)
    dense = DenseEmbedder(settings.dense_model)
    sparse = SparseEmbedder(settings.sparse_model)
    store = QdrantStore.from_settings(settings)
    store.ensure_collections(dense.dim)
    print(f"Indexing {n_docs:,} docs into {settings.qdrant_url} (dense on {dense.device})")

    counts = store.chunk_counts_by_hash()  # what a previous (possibly interrupted) run finished

    stats: dict = {"documents": n_docs, "doc_tokens": [], "strategies": {}}
    for strategy in strategies:
        done = sum(strategy.value in per_strategy for per_strategy in counts.values())
        todo = (d for d in paths.iter_docs() if strategy.value not in counts.get(d["sha256"], {}))
        n_chunks, chunk_tokens, started = 0, [], time.time()
        total_batches = -(-(n_docs - done) // batch_docs)  # ceiling division
        for batch in tqdm(_batched(todo, batch_docs), total=total_batches, desc=strategy.value, unit="batch"):
            pending: list[tuple[dict, object]] = []
            for d in batch:
                chunks = chunk(_structured(d), tokenizer, strategy, settings.chunk_size, settings.chunk_overlap)
                pending.extend((d, c) for c in chunks)
                counts[d["sha256"]][strategy.value] = len(chunks)
            texts = [c.text for _, c in pending]
            points = [
                ChunkPoint(d["doc_id"], strategy.value, c.index, vec, sp, _chunk_payload(d, c, strategy))
                for (d, c), vec, sp in zip(pending, dense.embed_documents(texts), sparse.embed_documents(texts))
            ]
            store.upsert_chunks(points)
            # Registry written only after its chunks landed: "indexed" in the registry is never a lie,
            # so an interrupted run resumes exactly where it stopped.
            store.upsert_documents(_document_record(d, counts[d["sha256"]]) for d in batch)
            n_chunks += len(points)
            chunk_tokens.extend(c.n_tokens for _, c in pending)
        stats["strategies"][strategy.value] = {
            "chunks_indexed": n_chunks,
            "chunks_in_collection": store.count(store.chunks, strategy.value),
            "mean_chunk_tokens": round(statistics.fmean(chunk_tokens), 1) if chunk_tokens else None,
            "seconds": round(time.time() - started),
        }  # fmt: skip

    doc_tokens = [len(tokenizer(d["text"], add_special_tokens=False)["input_ids"]) for d in paths.iter_docs()]
    stats["doc_tokens"] = _summary(doc_tokens)
    paths.write_json(paths.stats, stats)
    print(json.dumps(stats, indent=2))


def _chunk_payload(d: dict, c, strategy: ChunkingStrategy) -> dict:
    return {
        "doc_id": d["doc_id"],
        "chunk_id": f"{d['doc_id']}:{strategy.value}:{c.index}",
        "text": c.text,
        "title": d["title"],
        "source": d["source"],
        "format": "wikipedia",
        "chunk_index": c.index,
        "chunking_strategy": strategy.value,
        "start_char": c.start,  # with end_char: exact position in the document, for the
        "end_char": c.end,  #      chunk-level gold-sentence metric
        "n_tokens": c.n_tokens,
        "version": 1,
        "sha256": content_hash(c.text),
        "created_at": _now(),
    }


def _document_record(d: dict, chunk_counts: dict[str, int]) -> dict:
    now = _now()
    return {
        **{NUM_CHUNKS_PREFIX + strategy: n for strategy, n in chunk_counts.items()},
        "doc_id": d["doc_id"],
        "filename": None,
        "format": "wikipedia",
        "title": d["title"],
        "source": d["source"],
        "sha256": d["sha256"],
        "lead_replaced": d["lead_replaced"],
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }


def _structured(d: dict) -> StructuredText:
    return StructuredText(
        d["text"],
        [SectionSpans(tuple(s["heading"]) if s["heading"] else None,
                      [tuple(p) for p in s["paragraphs"]]) for s in d["sections"]],
    )  # fmt: skip


def _summary(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 1),
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
        "total": sum(ordered),
    }


# --- files ------------------------------------------------------------------------------------


class _Paths:
    def __init__(self, data_dir: Path):
        self.wiki = data_dir.parent / ".cache" / "wikipedia"
        self.docs = data_dir / "corpus" / "docs.jsonl"
        self.questions = data_dir / "eval" / "questions.json"
        self.manifest = data_dir / "manifest.json"
        self.stats = data_dir / "index_stats.json"

    def write_docs(self, docs: list[wiki.CorpusDoc]) -> None:
        self.docs.parent.mkdir(parents=True, exist_ok=True)
        with self.docs.open("w", encoding="utf-8") as f:
            for d in docs:
                row = {
                    "doc_id": d.doc_id, "title": d.title, "source": d.source, "sha256": d.sha256,
                    "lead_replaced": d.lead_replaced, "text": d.structured.text,
                    "sections": [{"heading": s.heading, "paragraphs": s.paragraphs}
                                 for s in d.structured.sections],
                    "gold_sentence_spans": d.gold_sentence_spans,
                }  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def iter_docs(self) -> Iterator[dict]:
        with self.docs.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def count_docs(self) -> int:
        with self.docs.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    @staticmethod
    def write_json(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _batched(items: Iterable[dict], size: int) -> Iterator[list[dict]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["build", "index", "all"], default="all")
    p.add_argument("--target", type=int, default=50_000, help="corpus size in documents")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieval", type=int, default=300, help="retrieval-eval question count")
    p.add_argument("--tuning", type=int, default=100, help="weight-tuning question count")
    p.add_argument("--strategies", default="recursive,fixed")
    p.add_argument("--batch-docs", type=int, default=64, help="documents per embed/upsert batch")
    return p.parse_args()


if __name__ == "__main__":
    main()
