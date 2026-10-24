"""Quick retrieval comparison on real data (not the full Phase 4 evaluation).

    python -m evaluation.quick_retrieval --set tuning
    python -m evaluation.quick_retrieval --set retrieval --strategy fixed

Runs every question of a set through four configurations, from plain dense search to the full
pipeline, and prints Recall@K, MRR and gold-sentence recall. Tune on `tuning`; report `retrieval`.
"""

import argparse
import json
import statistics

from app.config import get_settings
from app.ingestion.chunking import ChunkingStrategy
from app.retrieval.searcher import SearchParams, Searcher
from evaluation.retrieval_metrics import gold_sentence_recall, recall_at_k, reciprocal_rank

CONFIGS = {
    "dense only": {"mode": "dense", "feature_rerank": False, "cross_encoder": False},
    "hybrid (RRF)": {"mode": "hybrid", "feature_rerank": False, "cross_encoder": False},
    "hybrid + features": {"mode": "hybrid", "feature_rerank": True, "cross_encoder": False},
    "full pipeline": {"mode": "hybrid", "feature_rerank": True, "cross_encoder": True},
}
KS = (1, 5, 10, 20)


def evaluate(searcher: Searcher, questions: list[dict], params: SearchParams) -> dict[str, float]:
    rows: dict[str, list[float]] = {f"R@{k}": [] for k in KS} | {"MRR": [], "GoldSent@10": [], "ms": []}
    for q in questions:
        gold_docs = {g["doc_id"] for g in q["gold"] if g["doc_id"]}
        gold_sentences = [(g["doc_id"], *g["span"]) for g in q["gold"] if g["doc_id"] and g["span"]]
        result = searcher.search(q["question"], params)
        ranked = [(c.doc_id, c.payload["start_char"], c.payload["end_char"]) for c in result.results]
        for k in KS:
            rows[f"R@{k}"].append(recall_at_k(ranked, gold_docs, k))
        rows["MRR"].append(reciprocal_rank(ranked, gold_docs))
        if gold_sentences:
            rows["GoldSent@10"].append(gold_sentence_recall(ranked, gold_sentences, 10))
        rows["ms"].append(sum(result.timings_ms.values()))
    return {name: statistics.fmean(values) for name, values in rows.items() if values}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["tuning", "retrieval"], default="tuning")
    p.add_argument("--strategy", choices=[s.value for s in ChunkingStrategy], default="recursive")
    p.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated config names")
    args = p.parse_args()

    from app.ingestion.embedding import DenseEmbedder, SparseEmbedder
    from app.retrieval.cross_encoder import CrossEncoderReranker
    from app.store.qdrant_store import QdrantStore

    settings = get_settings()
    questions = json.loads((settings.data_dir / "eval" / "questions.json").read_text(encoding="utf-8"))[args.set]
    searcher = Searcher(
        QdrantStore.from_settings(settings),
        DenseEmbedder(settings.dense_model),
        SparseEmbedder(settings.sparse_model),
        CrossEncoderReranker(),
    )
    print(f"{len(questions)} {args.set} questions, chunking={args.strategy}, qdrant={settings.qdrant_url}\n")
    header = None
    for name in args.configs.split(","):
        # top_n=40: the cross-encoder must score more than 20, or the per-article cap would leave
        # the full pipeline with fewer than 20 results and an artificially low R@20.
        params = SearchParams(strategy=args.strategy, top_k=20, top_n=40, **CONFIGS[name])
        scores = evaluate(searcher, questions, params)
        if header is None:
            header = list(scores)
            print(f"{'config':<20}" + "".join(f"{h:>12}" for h in header))
        print(f"{name:<20}" + "".join(f"{scores[h]:>12.3f}" for h in header))


if __name__ == "__main__":
    main()
