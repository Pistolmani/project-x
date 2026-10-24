import pytest

from ingest.hotpot import Question
from ingest.select import SelectionConfig, is_distinctive_answer, select, trim_to_target


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("yes", False),
        ("No", False),
        ("1990", False),
        ("July 4, 1776", False),
        ("American", False),  # one common word: appears in thousands of articles
        ("Arthur's Magazine", True),
        ("Ferdinand Magellan", True),
        ("the red one", False),  # no capital letter: not a named entity
    ],
)
def test_is_distinctive_answer(answer, expected):
    assert is_distinctive_answer(answer) is expected


def _q(i, *, split="validation", type_="bridge", level="hard", answer="Some Person", gold=None):
    gold = gold or (f"G{i}a", f"G{i}b")
    context = tuple(gold) + tuple(f"D{i}_{k}" for k in range(8))
    return Question(f"q{i}", split, f"question {i}?", answer, type_, level,
                    tuple((t, 0) for t in gold), context)  # fmt: skip


def _pool():
    qs = [_q(i) for i in range(40)]
    qs += [_q(100 + i, split="train", level="easy") for i in range(10)]
    qs.append(_q(999, answer="yes"))  # never eligible for unanswerable
    return qs


CONFIG = SelectionConfig(seed=1, target_docs=1000, simple=3, multihop=3, unanswerable=2,
                         unanswerable_spares=1, contradictory=2, retrieval=5, tuning=4)  # fmt: skip


def _available(questions):
    return {t for q in questions for t in q.context_titles} | {f"extra{i}" for i in range(2000)}


def test_held_out_articles_never_reach_the_corpus_or_other_questions():
    qs = _pool()
    # q0 shares a gold article with whichever question becomes unanswerable, in some seeds.
    sel = select(qs, _available(qs), CONFIG)
    assert sel.held_out
    assert not sel.held_out & set(sel.corpus_titles)
    for q in sel.simple + sel.multihop + sel.contradictory + sel.retrieval + sel.tuning:
        assert not q.gold_titles & sel.held_out


def test_question_sharing_a_held_out_article_is_excluded():
    shared = ("Shared", "Other")
    qs = [_q(1, gold=shared), _q(2, gold=shared)] + _pool()[2:]
    config = SelectionConfig(seed=0, target_docs=1000, simple=1, multihop=1, unanswerable=1,
                             unanswerable_spares=0, contradictory=1, retrieval=1, tuning=1)  # fmt: skip
    for seed in range(20):
        sel = select(qs, _available(qs), SelectionConfig(**{**config.__dict__, "seed": seed}))
        picked = sel.simple + sel.multihop + sel.contradictory + sel.retrieval
        if "Shared" in sel.held_out:
            assert all("Shared" not in q.gold_titles for q in picked)


def test_unanswerable_rules_and_no_question_used_twice():
    qs = _pool()
    sel = select(qs, _available(qs), CONFIG)
    for q in sel.unanswerable_candidates:
        assert q.split == "validation" and q.type == "bridge" and is_distinctive_answer(q.answer)
    assert all(q.level == "easy" for q in sel.simple)
    ids = [q.id for q in sel.simple + sel.multihop + sel.contradictory + sel.retrieval
           + sel.tuning + sel.unanswerable_candidates]  # fmt: skip
    assert len(ids) == len(set(ids))  # in particular: tuning never overlaps the retrieval eval set
    assert len(sel.tuning) == CONFIG.tuning


def test_corpus_has_required_titles_then_fills_past_target_deterministically():
    qs = _pool()
    a = select(qs, _available(qs), CONFIG)
    b = select(qs, _available(qs), CONFIG)
    assert a.corpus_titles == b.corpus_titles  # same seed, same corpus
    assert len(a.corpus_titles) == CONFIG.target_docs + CONFIG.fill_margin == len(set(a.corpus_titles))
    for q in a.retrieval:
        assert set(q.context_titles) <= a.required_titles <= set(a.corpus_titles)


def test_trim_to_target_replaces_dedup_drops_with_fill_and_keeps_required():
    titles = ["r1", "r2", "r3", "f1", "f2", "f3", "f4"]
    after_dedup = [t for t in titles if t not in {"r2", "f1"}]  # two duplicates dropped
    trimmed = trim_to_target(after_dedup, required={"r1", "r2", "r3"}, target=5)
    assert trimmed == ["r1", "r3", "f2", "f3", "f4"]  # exactly the target, required first


def test_missing_gold_article_makes_question_unusable():
    qs = _pool()
    available = _available(qs) - {"G0a"}
    sel = select(qs, available, CONFIG)
    picked = sel.simple + sel.multihop + sel.contradictory + sel.retrieval + sel.unanswerable_candidates
    assert "q0" not in {q.id for q in picked}
