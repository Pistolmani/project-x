"""Choose the eval questions and the ~50k corpus titles, in the order DATASET.md §3 requires.

Order matters because holding out an article for one test case can silently break another:
1. unanswerable candidates first; their gold articles become the held-out set (never ingested);
2. every other eval question is drawn only from questions that touch no held-out article;
3. the corpus = all articles the kept questions reference, then a seeded fill up to the target.

The unanswerable *absence check* (gold answer string absent from the final corpus) needs the article
texts, so it runs later in build_corpus; that is why we draw spare unanswerable candidates here.
Pure functions over plain data, so the rules are unit-testable without any downloads.
"""

import random
import re
from dataclasses import dataclass, field

from ingest.hotpot import Question


@dataclass(frozen=True)
class SelectionConfig:
    seed: int = 42
    target_docs: int = 50_000
    simple: int = 15
    multihop: int = 15
    unanswerable: int = 10
    # Extra candidates, since many fail the absence check: 11 of 30 failed on a 2k-doc sample, and the
    # failure rate grows with corpus size (more articles that might mention the answer).
    unanswerable_spares: int = 60
    contradictory: int = 10
    retrieval: int = 300
    tuning: int = 100  # for tuning ranking weights; never scored, so the eval set stays unseen
    # Extra fill titles so exact-dedup drops can be replaced and the corpus still reaches the target
    # (the brief requires >= 50,000 documents; a 50k run dropped 2 duplicates).
    fill_margin: int = 200


@dataclass
class Selection:
    simple: list[Question] = field(default_factory=list)
    multihop: list[Question] = field(default_factory=list)
    unanswerable_candidates: list[Question] = field(default_factory=list)
    contradictory: list[Question] = field(default_factory=list)
    retrieval: list[Question] = field(default_factory=list)
    tuning: list[Question] = field(default_factory=list)
    held_out: set[str] = field(default_factory=set)
    # Required titles first, then the seeded fill (target + margin); trim with `trim_to_target`.
    corpus_titles: list[str] = field(default_factory=list)
    required_titles: set[str] = field(default_factory=set)


_DIGIT = re.compile(r"\d")


def is_distinctive_answer(answer: str) -> bool:
    """Answers where "this string appears nowhere in the corpus" actually proves something:
    multi-word, capitalized, no digits. Rejects yes/no, years, numbers and one-word answers
    like "American" that occur in thousands of articles (DATASET.md §6)."""
    a = answer.strip()
    return (
        a.lower() not in {"yes", "no"}
        and len(a.split()) >= 2
        and any(ch.isupper() for ch in a)
        and not _DIGIT.search(a)
    )


def select(questions: list[Question], available: set[str], config: SelectionConfig) -> Selection:
    """`available` = titles that exist in both HotpotQA (2017 paragraph) and the 2023 dump."""
    rng = random.Random(config.seed)
    # Gold articles must exist; a missing (renamed) distractor just isn't in the corpus.
    usable = [q for q in questions if q.gold_titles <= available]
    rng.shuffle(usable)
    sel = Selection()
    used_ids: set[str] = set()

    def take(n: int, predicate) -> list[Question]:
        picked = []
        for q in usable:
            if len(picked) == n:
                break
            if q.id not in used_ids and not (q.gold_titles & sel.held_out) and predicate(q):
                picked.append(q)
                used_ids.add(q.id)
        if len(picked) < n:
            raise ValueError(f"only {len(picked)} questions match, needed {n}")
        return picked

    # 1. Unanswerable first: hard bridge questions with distinctive answers (dev split, like multi-hop).
    sel.unanswerable_candidates = take(
        config.unanswerable + config.unanswerable_spares,
        lambda q: q.split == "validation" and q.type == "bridge" and is_distinctive_answer(q.answer),
    )
    for q in sel.unanswerable_candidates:
        sel.held_out |= q.gold_titles

    # 2. Everything else avoids held-out articles (enforced inside take()).
    sel.simple = take(config.simple, lambda q: q.level == "easy")
    sel.multihop = take(config.multihop, lambda q: q.split == "validation" and q.type == "bridge")
    sel.contradictory = take(config.contradictory, lambda q: q.split == "validation")
    sel.retrieval = take(config.retrieval, lambda q: q.split == "validation")
    # Same distribution as the retrieval set but disjoint from it: ranking weights are tuned
    # here, so reported retrieval metrics come from questions the weights never saw.
    sel.tuning = take(config.tuning, lambda q: q.split == "validation")

    # 3. Corpus: every article the kept questions reference (incl. the unanswerable questions'
    #    distractors), minus held-out, then a seeded fill from the remaining available titles.
    kept = (
        sel.simple + sel.multihop + sel.contradictory + sel.retrieval + sel.tuning
        + sel.unanswerable_candidates
    )  # fmt: skip
    required = sorted(({t for q in kept for t in q.context_titles} & available) - sel.held_out)
    if len(required) > config.target_docs:
        raise ValueError(f"{len(required)} required articles exceed target {config.target_docs}")
    pool = sorted(available - sel.held_out - set(required))
    fill = rng.sample(pool, min(len(pool), config.target_docs - len(required) + config.fill_margin))
    sel.corpus_titles = required + fill
    sel.required_titles = set(required)
    return sel


def trim_to_target(titles: list[str], required: set[str], target: int) -> list[str]:
    """Called after dedup has removed some titles: keep every surviving required title, then fill
    titles in their seeded order until the target is reached exactly."""
    kept = [t for t in titles if t in required]
    fill = [t for t in titles if t not in required]
    return kept + fill[: max(0, target - len(kept))]
