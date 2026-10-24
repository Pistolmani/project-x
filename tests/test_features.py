import pytest

from app.retrieval.candidate import Candidate
from app.retrieval.features import (
    FeatureReranker,
    answer_type,
    content_terms,
    entity_overlap,
    exact_phrase,
    expected_answer_type,
    length_ok,
    query_entities,
    term_coverage,
    title_match,
    words,
)

QUERY = "Which magazine was started first Arthur's Magazine or First for Women?"


def test_query_entities_finds_names_and_skips_the_question_word():
    assert query_entities(QUERY) == ["Arthur's Magazine", "First for Women"]
    assert query_entities('Who wrote "The Old Man and the Sea"?') == ["The Old Man and the Sea"]
    assert query_entities("what is the capital of france") == []


@pytest.mark.parametrize(
    "query, expected",
    [
        ("How many members does the band have?", "number"),
        ("When was the bridge built?", "year"),
        ("In what year did the war end?", "year"),
        ("Who directed the film?", "name"),
        ("Which magazine was started first?", None),
    ],
)
def test_expected_answer_type(query, expected):
    assert expected_answer_type(query) == expected


def test_title_match_is_full_when_the_question_names_the_article():
    terms = content_terms(QUERY)
    assert title_match(terms, "Arthur's Magazine") == 1.0
    assert title_match(terms, "Radio City (Indian radio station)") == 0.0


def test_term_coverage():
    q = content_terms("river delta islands")
    assert term_coverage(q, content_terms("The river delta has many islands.")) == 1.0
    assert term_coverage(q, content_terms("A river.")) == pytest.approx(1 / 3)
    assert term_coverage(set(), {"x"}) == 0.0


def test_entity_overlap_is_case_insensitive_share():
    entities = ["Arthur's Magazine", "First for Women"]
    assert entity_overlap(entities, "arthur's magazine was founded in 1844") == 0.5
    assert entity_overlap([], "anything") == 0.0


def test_exact_phrase_rewards_longer_verbatim_matches():
    q = words("the old man and the sea")
    assert exact_phrase(q, words("He wrote The Old Man and the Sea in Cuba")) == 1.0
    assert 0 < exact_phrase(q, words("an old man walked")) < 1.0  # only "old man"
    assert exact_phrase(q, words("sea man old")) == 0.0
    assert exact_phrase(["single"], ["single"]) == 0.0  # needs at least two words


def test_answer_type_and_length_features():
    assert answer_type("year", "It was founded in 1844.") == 1.0
    assert answer_type("year", "It was founded long ago.") == 0.0
    assert answer_type("name", "It was directed by Steven Spielberg.") == 1.0
    assert answer_type(None, "anything 1844") == 0.0
    assert length_ok(5) == 0.25 and length_ok(400) == 1.0


def _cand(pid, title, text, rrf, n_tokens=100):
    c = Candidate(pid, {"doc_id": pid, "title": title, "text": text, "n_tokens": n_tokens})
    c.rrf_score = rrf
    return c


def test_reranker_writes_explainable_features_in_range():
    cands = [_cand("a", "Arthur's Magazine", "Arthur's Magazine was an American monthly founded in 1844.", 0.03),
             _cand("b", "Radio City", "Radio City is India's first private FM radio station.", 0.02)]  # fmt: skip
    ranked = FeatureReranker().rerank(QUERY, cands)
    assert ranked[0].point_id == "a"
    for c in ranked:
        assert set(c.features) == {"fusion_prior", "title_match", "term_coverage", "entity_overlap",
                                   "exact_phrase", "answer_type", "length_ok"}  # fmt: skip
        assert all(0.0 <= v <= 1.0 for v in c.features.values())


def test_features_can_overturn_the_fused_order():
    weak_fusion_right_entity = _cand("a", "First for Women", "First for Women is a woman's magazine.", 0.010)
    strong_fusion_off_topic = _cand("b", "Women's Wear", "A magazine started first in Paris.", 0.030)
    ranked = FeatureReranker().rerank(QUERY, [strong_fusion_off_topic, weak_fusion_right_entity])
    assert ranked[0].point_id == "a"
    # ...and with features silenced, the fused order wins again.
    only_fusion = {name: 0.0 for name in ranked[0].features} | {"fusion_prior": 1.0}
    assert FeatureReranker(only_fusion).rerank(QUERY, ranked)[0].point_id == "b"


def test_unknown_feature_weight_is_rejected():
    with pytest.raises(ValueError):
        FeatureReranker({"recency": 1.0})
