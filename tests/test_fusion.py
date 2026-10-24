import pytest

from app.retrieval.fusion import weighted_rrf


def test_matches_hand_computed_scores():
    fused = weighted_rrf({"dense": ["a", "b", "c"], "sparse": ["b", "d"]}, k=60)
    scores = {item: score for item, score, _ in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)  # 2nd in dense, 1st in sparse
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 62)  # only in one list: still scores
    assert [item for item, _, _ in fused] == ["b", "a", "d", "c"]


def test_contributions_explain_the_score():
    item, score, parts = weighted_rrf({"dense": ["x"], "sparse": ["x"]}, k=60)[0]
    assert item == "x"
    assert parts == pytest.approx({"dense": 1 / 61, "sparse": 1 / 61})
    assert score == pytest.approx(sum(parts.values()))


def test_weights_shift_the_winner():
    lists = {"dense": ["sem", "lex"], "sparse": ["lex", "sem"]}
    assert weighted_rrf(lists, {"dense": 2.0, "sparse": 1.0})[0][0] == "sem"
    assert weighted_rrf(lists, {"dense": 1.0, "sparse": 2.0})[0][0] == "lex"


def test_zero_weight_ignores_a_list():
    fused = weighted_rrf({"dense": ["a"], "sparse": ["b"]}, {"dense": 1.0, "sparse": 0.0})
    assert fused[0][0] == "a"
    assert dict((i, s) for i, s, _ in fused)["b"] == 0.0


def test_ties_are_deterministic():
    # a and b both get exactly 1/(k+1); the tie falls back to best rank, then ID.
    fused = weighted_rrf({"dense": ["b"], "sparse": ["a"]})
    assert [item for item, _, _ in fused] == ["a", "b"]


def test_duplicate_within_a_list_counts_once():
    fused = weighted_rrf({"dense": ["a", "a", "b"]}, k=60)
    assert dict((i, s) for i, s, _ in fused)["a"] == pytest.approx(1 / 61)


def test_smaller_k_widens_the_gap_between_ranks():
    def gap(k):
        s = {i: v for i, v, _ in weighted_rrf({"dense": ["a", "b"]}, k=k)}
        return s["a"] - s["b"]

    assert gap(1) > gap(60)


def test_empty_and_invalid_inputs():
    assert weighted_rrf({}) == []
    assert weighted_rrf({"dense": []}) == []
    with pytest.raises(ValueError):
        weighted_rrf({"dense": ["a"]}, k=0)
    with pytest.raises(ValueError):
        weighted_rrf({"dense": ["a"]}, {"dense": -1.0})
