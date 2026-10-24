"""Custom weighted Reciprocal Rank Fusion — custom ranking component #1 (D8).

    score(d) = Σ over lists L containing d of  w_L / (k + rank_L(d))      (rank is 1-based)

Why ranks and not scores: dense cosine (~0..1) and BM25 (unbounded) live on different scales, so
adding them lets one dominate arbitrarily. RRF only uses positions, so it is scale-free. `k` damps the
gap between top ranks (k=60, from the original RRF paper); the weights shift trust between semantic
and lexical evidence and are tuned on the tuning set, never on the eval set.

Qdrant has a built-in RRF; it is deliberately not used, because this is the graded custom component.
"""

from math import inf


def weighted_rrf(
    ranked: dict[str, list[str]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuse ranked ID lists. Returns (id, score, {list: contribution}) best-first.

    An item found by only one list still scores (from that list alone). Ties are broken by the best
    rank the item reached in any list, then by ID, so results are deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    weights = weights or {}
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")

    contributions: dict[str, dict[str, float]] = {}
    best_rank: dict[str, int] = {}
    for name, ids in ranked.items():
        weight = weights.get(name, 1.0)
        seen: set[str] = set()
        for rank, item in enumerate(ids, start=1):
            if item in seen:  # a duplicate inside one list counts once, at its best rank
                continue
            seen.add(item)
            contributions.setdefault(item, {})[name] = weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, inf), rank)

    fused = [(item, sum(parts.values()), parts) for item, parts in contributions.items()]
    fused.sort(key=lambda t: (-t[1], best_rank[t[0]], t[0]))
    return fused
