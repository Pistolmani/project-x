import pytest

from evaluation.retrieval_metrics import gold_sentence_recall, recall_at_k, reciprocal_rank

RANKED = [("x", 0, 100), ("a", 0, 50), ("b", 40, 120), ("a", 50, 90)]


def test_recall_at_k_counts_documents_not_chunks():
    assert recall_at_k(RANKED, {"a", "b"}, 1) == 0.0
    assert recall_at_k(RANKED, {"a", "b"}, 2) == 0.5
    assert recall_at_k(RANKED, {"a", "b"}, 10) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)


def test_reciprocal_rank_of_first_gold_chunk():
    assert reciprocal_rank(RANKED, {"a"}) == 0.5
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_gold_sentence_must_be_entirely_inside_one_chunk():
    inside = ("a", 10, 40)  # within ("a", 0, 50)
    straddling = ("a", 45, 60)  # cut by the boundary at 50: neither chunk holds it whole
    assert gold_sentence_recall(RANKED, [inside], 10) == 1.0
    assert gold_sentence_recall(RANKED, [straddling], 10) == 0.0
    assert gold_sentence_recall(RANKED, [inside, straddling], 10) == 0.5
    assert gold_sentence_recall(RANKED, [inside], 1) == 0.0  # outside the top-k
