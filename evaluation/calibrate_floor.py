"""Calibrate the retrieval gate's relevance floor (BACKEND.md §4, "Relevance floor").

    python -m evaluation.calibrate_floor

The floor is a threshold on the best raw cross-encoder score of a search. To pick it without touching
the evaluation sets:
- positives = the TUNING questions (their gold articles are indexed),
- negatives = the unanswerable SPARES: bridge questions whose gold articles were held out of the corpus,
  minus the 10 used in the behavioral set.
The two distributions overlap, so the floor is a trade-off. The gate is a backstop: the investigator's
own refusal and the verifier come after it. So the chosen floor is the highest one that still lets
TARGET_PASS of answerable questions through; the balanced-accuracy optimum is reported for comparison.
Note the agent applies it to the score of the passages it cites, found by its own
sub-queries, so this calibration on the raw question is an approximation; the /ask eval measures the result.
"""

import json

from app.engine import build_engine
from ingest import hotpot

TARGET_PASS = 0.95


def spare_questions(manifest: dict, eval_ids: set[str]) -> list[hotpot.Question]:
    held_out = set(manifest["held_out_titles"])
    questions, _ = hotpot.load(("validation",))
    return [q for q in questions if q.type == "bridge" and q.gold_titles and q.gold_titles <= held_out
            and q.id not in eval_ids]  # fmt: skip


def best_threshold(positives: list[float], negatives: list[float]) -> tuple[float, float]:
    """(threshold, balanced accuracy); a score passes when score >= threshold."""
    best = (0.0, -1.0)
    for t in sorted(set(positives + negatives)):
        tpr = sum(s >= t for s in positives) / len(positives)
        tnr = sum(s < t for s in negatives) / len(negatives)
        if (tpr + tnr) / 2 > best[1]:
            best = (t, (tpr + tnr) / 2)
    return best


def high_recall_threshold(positives: list[float], target: float) -> float:
    """Highest threshold that passes at least `target` of the positives."""
    ranked = sorted(positives)
    return ranked[int((1 - target) * len(ranked))]


def main() -> None:
    engine = build_engine()
    data_dir = engine.settings.data_dir
    sets = json.loads((data_dir / "eval" / "questions.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    spares = spare_questions(manifest, {q["id"] for q in sets["behavioral"]["unanswerable"]})
    print(f"{len(sets['tuning'])} positives (tuning set), {len(spares)} negatives (unanswerable spares)")

    def top_score(question: str) -> float:
        score = engine.searcher.search(question).top_ce_score
        return score if score is not None else float("-inf")

    positives = [top_score(q["question"]) for q in sets["tuning"]]
    negatives = [top_score(q.question) for q in spares]

    def rates(t):
        return sum(s >= t for s in positives) / len(positives), sum(s < t for s in negatives) / len(negatives)

    balanced_floor, balanced = best_threshold(positives, negatives)
    floor = high_recall_threshold(positives, TARGET_PASS)
    pass_rate, block_rate = rates(floor)

    def quantiles(xs):
        xs = sorted(xs)
        return [round(xs[int(p * (len(xs) - 1))], 2) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]

    print(f"positives p10/25/50/75/90: {quantiles(positives)}")
    print(f"negatives p10/25/50/75/90: {quantiles(negatives)}")
    b_pass, b_block = rates(balanced_floor)
    print(f"balanced optimum {balanced_floor:.2f}: passes {b_pass:.0%} of answerable, blocks {b_block:.0%} "
          f"of unanswerable (balanced accuracy {balanced:.3f})")
    print(f"chosen floor {floor:.2f} (>= {TARGET_PASS:.0%} pass): passes {pass_rate:.0%}, blocks {block_rate:.0%}")
    out = data_dir / "relevance_floor.json"
    out.write_text(json.dumps({"floor": floor, "target_pass": TARGET_PASS, "pass_rate": pass_rate,
                               "block_rate": block_rate, "balanced_floor": balanced_floor,
                               "balanced_accuracy": balanced, "positives": positives, "negatives": negatives},
                              indent=2), encoding="utf-8")  # fmt: skip
    print(f"-> {out}")


if __name__ == "__main__":
    main()
