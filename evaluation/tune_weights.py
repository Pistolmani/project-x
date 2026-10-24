"""Tune the RRF and feature-re-ranker weights on the TUNING set only (never the retrieval eval set).

    python -m evaluation.tune_weights --strategy recursive

1. Retrieve once per tuning question (hybrid, re-rankers off, every fused candidate) and compute the
   features. Retrieval is the only expensive part, so it happens exactly once.
2. Coordinate ascent over small grids, entirely offline: RRF is recomputed from the stored dense/sparse
   ranks and the features are re-weighted, with no further Qdrant or model calls.
   Objective: recall of gold articles in the top 20 after the per-article cap — exactly the list the
   cross-encoder will see, so this is the re-ranker's real job. Tie-break: MRR.
3. Report before/after on the tuning set, plus the retrieval set as a held-out check that the gain is
   not overfitting (the retrieval set is only *reported*, never used to choose a value).
"""

import argparse
import json
from dataclasses import dataclass

from app.config import get_settings
from app.retrieval.candidate import Candidate
from app.retrieval.features import DEFAULT_WEIGHTS, FeatureReranker, featurize, set_fusion_prior
from app.retrieval.fusion import weighted_rrf
from app.retrieval.searcher import SearchParams, Searcher, cap_per_doc
from evaluation.retrieval_metrics import recall_at_k, reciprocal_rank

WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
RRF_K_GRID = [10, 30, 60, 100]
TOP_N = 20  # the cross-encoder's input size
PASSES = 3


@dataclass
class Prepared:
    gold: set[str]
    candidates: list[Candidate]


def prepare(searcher: Searcher, questions: list[dict], strategy: str) -> list[Prepared]:
    params = SearchParams(strategy=strategy, mode="hybrid", feature_rerank=False,
                          cross_encoder=False, top_k=1000, max_per_doc=None)  # fmt: skip
    prepared = []
    for q in questions:
        result = searcher.search(q["question"], params)
        featurize(q["question"], result.results)
        prepared.append(Prepared({g["doc_id"] for g in q["gold"] if g["doc_id"]}, result.results))
    return prepared


def evaluate(prepared: list[Prepared], config: dict) -> tuple[float, float]:
    """(mean recall@TOP_N, MRR) for one weighting."""
    reranker = FeatureReranker(config["features"])
    recalls, rrs = [], []
    for p in prepared:
        dense = sorted((c for c in p.candidates if c.dense_rank), key=lambda c: c.dense_rank)
        sparse = sorted((c for c in p.candidates if c.sparse_rank), key=lambda c: c.sparse_rank)
        fused = weighted_rrf(
            {"dense": [c.point_id for c in dense], "sparse": [c.point_id for c in sparse]},
            {"dense": config["w_dense"], "sparse": config["w_sparse"]},
            k=config["rrf_k"],
        )
        scores = {point_id: score for point_id, score, _ in fused}
        for c in p.candidates:
            c.rrf_score = scores.get(c.point_id, 0.0)
        set_fusion_prior(p.candidates)
        ranked = cap_per_doc(reranker.score(p.candidates), 2, TOP_N)
        hits = [(c.doc_id, 0, 0) for c in ranked]
        recalls.append(recall_at_k(hits, p.gold, TOP_N))
        rrs.append(reciprocal_rank(hits, p.gold))
    return sum(recalls) / len(recalls), sum(rrs) / len(rrs)


def coordinate_ascent(prepared: list[Prepared], start: dict) -> dict:
    best = json.loads(json.dumps(start))  # deep copy
    best_score = evaluate(prepared, best)
    knobs = [("w_dense", WEIGHT_GRID), ("w_sparse", WEIGHT_GRID), ("rrf_k", RRF_K_GRID)]
    knobs += [(f"features.{name}", WEIGHT_GRID) for name in DEFAULT_WEIGHTS]
    for _ in range(PASSES):
        improved = False
        for knob, grid in knobs:
            for value in grid:
                trial = json.loads(json.dumps(best))
                _set(trial, knob, value)
                score = evaluate(prepared, trial)
                if (round(score[0], 6), score[1]) > (round(best_score[0], 6), best_score[1]):
                    best, best_score, improved = trial, score, True
        if not improved:
            break
    return best


def _set(config: dict, knob: str, value) -> None:
    if knob.startswith("features."):
        config["features"][knob.split(".", 1)[1]] = value
    else:
        config[knob] = value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=["recursive", "fixed"], default="fixed")
    args = p.parse_args()

    from app.ingestion.embedding import DenseEmbedder, SparseEmbedder
    from app.store.qdrant_store import QdrantStore

    settings = get_settings()
    sets = json.loads((settings.data_dir / "eval" / "questions.json").read_text(encoding="utf-8"))
    searcher = Searcher(QdrantStore.from_settings(settings), DenseEmbedder(settings.dense_model),
                        SparseEmbedder(settings.sparse_model))  # fmt: skip

    print(f"Retrieving candidates for {len(sets['tuning'])} tuning + {len(sets['retrieval'])} held-out questions ...")
    tuning = prepare(searcher, sets["tuning"], args.strategy)
    heldout = prepare(searcher, sets["retrieval"], args.strategy)

    start = {"w_dense": 1.0, "w_sparse": 1.0, "rrf_k": 60, "features": dict(DEFAULT_WEIGHTS)}
    no_features = {**start, "features": {n: 0.0 for n in DEFAULT_WEIGHTS} | {"fusion_prior": 1.0}}
    tuned = coordinate_ascent(tuning, start)

    report = {"strategy": args.strategy, "objective": f"recall@{TOP_N} after per-article cap (tie: MRR)",
              "tuned": tuned, "results": {}}  # fmt: skip
    print(f"\n{'config':<28}{'tuning R@20':>13}{'tuning MRR':>12}{'held-out R@20':>15}{'held-out MRR':>14}")
    for name, cfg in [("RRF only (no features)", no_features), ("default weights", start), ("tuned weights", tuned)]:
        t, h = evaluate(tuning, cfg), evaluate(heldout, cfg)
        report["results"][name] = {"tuning": t, "heldout": h}
        print(f"{name:<28}{t[0]:>13.3f}{t[1]:>12.3f}{h[0]:>15.3f}{h[1]:>14.3f}")

    out = settings.data_dir / f"tuned_weights_{args.strategy}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTuned: {json.dumps(tuned)}\n-> {out}")


if __name__ == "__main__":
    main()
