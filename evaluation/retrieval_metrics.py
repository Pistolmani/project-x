"""Retrieval metrics (EVALUATION.md §4.1–4.2). Pure functions over plain data.

`ranked` is the search output in order: one (doc_id, start_char, end_char) per returned chunk.
Recall@K and MRR are at **document** granularity (a chunk counts for its article); the gold-sentence
metric is at **chunk** granularity and is what tells the two chunking strategies apart.
"""

Hit = tuple[str, int, int]  # (doc_id, start_char, end_char) of one returned chunk


def recall_at_k(ranked: list[Hit], gold_docs: set[str], k: int) -> float:
    """Share of the gold supporting documents found among the top-k returned chunks."""
    if not gold_docs:
        raise ValueError("no gold documents")
    found = {doc for doc, _, _ in ranked[:k]}
    return len(found & gold_docs) / len(gold_docs)


def reciprocal_rank(ranked: list[Hit], gold_docs: set[str]) -> float:
    """1 / rank of the first chunk from a gold document (0 if none); averaged over questions = MRR."""
    for rank, (doc, _, _) in enumerate(ranked, start=1):
        if doc in gold_docs:
            return 1.0 / rank
    return 0.0


def gold_sentence_recall(ranked: list[Hit], gold_sentences: list[Hit], k: int) -> float:
    """Share of gold sentences that sit *entirely inside* one of the top-k chunks.

    Exact because chunks and gold sentences carry character offsets into the same document text:
    a fixed window that cuts a gold sentence in half does not count, even though Recall@K by
    document would."""
    if not gold_sentences:
        raise ValueError("no gold sentences")
    top = ranked[:k]
    contained = sum(
        any(doc == g_doc and start <= g_start and g_end <= end for doc, start, end in top)
        for g_doc, g_start, g_end in gold_sentences
    )
    return contained / len(gold_sentences)
