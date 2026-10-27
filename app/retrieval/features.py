"""Custom feature-based re-ranker — custom ranking component #2 (D8).

Every feature is a cheap, explainable number in [0, 1] built from signals this corpus actually has
(BACKEND.md §4[2]). The score is a weighted sum, so "why did X beat Y" is answered by comparing two
small dicts. No recency / source authority: the corpus is undated single-source Wikipedia, and a
recency signal would even promote the injected contradictory documents.

Text handling is deliberately plain (lowercase + regex words + a small stopword list): no NLP library,
nothing to download, and every rule can be explained in one sentence.
"""

import re

from app.retrieval.candidate import Candidate

DEFAULT_WEIGHTS: dict[str, float] = {
    "fusion_prior": 1.0,  # keep the fused order as the baseline the features refine
    "title_match": 0.6,
    "term_coverage": 0.6,
    "entity_overlap": 0.8,
    "exact_phrase": 0.4,
    "answer_type": 0.3,
    "length_ok": 0.3,
}

STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been being do does did "
    "what which who whom whose when where why how that this these those it its as than then there "
    "their they he she his her him we you your i me my our not no yes also into about after before "
    "both either same more most many much any each other such only own so too very can will would "
    "should could may might has have had".split()
)
QUESTION_WORDS = frozenset("which what who whom whose when where why how are is was were did do does in".split())
# Lower-case words allowed inside a capitalized name: "First for Women", "Bank of America".
NAME_CONNECTORS = frozenset("of for the and de da del della di du la le van von y".split())

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER = re.compile(r"\b\d[\d,.]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b", re.I)
_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+")
SHORT_CHUNK_TOKENS = 20


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_terms(text: str) -> set[str]:
    return {w for w in words(text) if w not in STOPWORDS}


def query_entities(query: str) -> list[str]:
    """Quoted strings plus runs of capitalized words, e.g. "Arthur's Magazine", "First for Women".
    A capitalized question word at the start ("Which") is not an entity."""
    entities = [m.strip() for m in _QUOTED.findall(query)]
    run: list[str] = []
    tokens = re.findall(r"[\w'’.-]+", _QUOTED.sub(" ", query))
    for i, token in enumerate(tokens + [""]):
        capitalized = token[:1].isupper() and not (i == 0 and token.lower() in QUESTION_WORDS)
        if capitalized:
            run.append(token)
        elif run and token.lower() in NAME_CONNECTORS and tokens[i + 1 : i + 2] and tokens[i + 1][:1].isupper():
            run.append(token)
        else:
            if run:
                entities.append(" ".join(run).strip(".-"))
            run = []
    return [e for e in dict.fromkeys(entities) if e]


def expected_answer_type(query: str) -> str | None:
    q = query.lower()
    if re.search(r"\bhow (many|much)\b|\bwhat number\b", q):
        return "number"
    if re.search(r"\bwhen\b|\bwhat year\b|\bwhich year\b|\bwhat decade\b", q):
        return "year"
    if re.search(r"\bwho\b|\bwhom\b|\bwhose\b", q):
        return "name"
    return None


# --- the features -----------------------------------------------------------------------------


def title_match(query_terms: set[str], title: str) -> float:
    """Share of the title's content words that the query mentions: 1.0 when the question names the
    article ("...Arthur's Magazine..." vs title "Arthur's Magazine")."""
    title_terms = content_terms(title)
    return len(title_terms & query_terms) / len(title_terms) if title_terms else 0.0


def term_coverage(query_terms: set[str], chunk_terms: set[str]) -> float:
    """Share of the query's content words present in the chunk; a one-word lucky match scores low."""
    return len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0.0


def entity_overlap(entities: list[str], chunk_text: str) -> float:
    """Share of the query's named entities that appear verbatim (case-insensitive) in the chunk."""
    if not entities:
        return 0.0
    lowered = chunk_text.lower()
    return sum(e.lower() in lowered for e in entities) / len(entities)


def exact_phrase(query_words: list[str], chunk_words: list[str], max_n: int = 6) -> float:
    """Length of the longest query phrase (>= 2 words) found verbatim in the chunk, scaled to [0, 1]:
    0 for no shared bigram, 1 for a shared phrase of `max_n` words (or the whole short query)."""
    cap = min(len(query_words), max_n)
    if cap < 2:
        return 0.0
    chunk_ngrams = {n: set(zip(*(chunk_words[i:] for i in range(n)))) for n in range(2, cap + 1)}
    longest = 0
    for n in range(2, cap + 1):
        query_ngrams = set(zip(*(query_words[i:] for i in range(n))))
        if query_ngrams & chunk_ngrams[n]:
            longest = n
        else:
            break  # no n-gram match means no longer match either
    return (longest - 1) / (cap - 1) if longest else 0.0


def answer_type(expected: str | None, chunk_text: str) -> float:
    """1.0 if the chunk contains something of the type the question asks for (a year for "when", a
    number for "how many", a person-like name for "who"). Constant across candidates when the
    question implies no type, so it then has no effect on the order."""
    pattern = {"year": _YEAR, "number": _NUMBER, "name": _NAME}.get(expected or "")
    return 1.0 if pattern and pattern.search(chunk_text) else 0.0


def length_ok(n_tokens: int) -> float:
    """Light penalty for scraps (a heading plus a few words) that can match keywords by accident."""
    return min(1.0, n_tokens / SHORT_CHUNK_TOKENS)


# --- the re-ranker ----------------------------------------------------------------------------


def featurize(query: str, candidates: list[Candidate]) -> None:
    """Compute every feature for every candidate (writes `c.features`). Separate from weighting so
    the tuner can compute features once and try many weightings cheaply."""
    query_terms = content_terms(query)
    query_words = words(query)
    entities = query_entities(query)
    expected = expected_answer_type(query)
    for c in candidates:
        c.features = {
            "fusion_prior": 0.0,  # set by set_fusion_prior: depends on the whole candidate set
            "title_match": title_match(query_terms, c.title),
            "term_coverage": term_coverage(query_terms, content_terms(c.text)),
            "entity_overlap": entity_overlap(entities, f"{c.title}\n{c.text}"),
            "exact_phrase": exact_phrase(query_words, words(c.text)),
            "answer_type": answer_type(expected, c.text),
            "length_ok": length_ok(c.n_tokens),
        }
    set_fusion_prior(candidates)


def set_fusion_prior(candidates: list[Candidate]) -> None:
    """RRF score min-max scaled within this candidate set, so it is comparable to the [0, 1] features."""
    if not candidates:
        return
    lo = min(c.rrf_score for c in candidates)
    hi = max(c.rrf_score for c in candidates)
    for c in candidates:
        c.features["fusion_prior"] = (c.rrf_score - lo) / (hi - lo) if hi > lo else 1.0


class FeatureReranker:
    def __init__(self, weights: dict[str, float] | None = None):
        unknown = set(weights or {}) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def score(self, candidates: list[Candidate]) -> list[Candidate]:
        """Weight already-computed features (writes `feature_score`); best-first. Ties keep the
        incoming (fused) order, since Python's sort is stable."""
        for c in candidates:
            c.feature_score = sum(self.weights[name] * value for name, value in c.features.items())
        return sorted(candidates, key=lambda c: -c.feature_score)

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        featurize(query, candidates)
        return self.score(candidates)
