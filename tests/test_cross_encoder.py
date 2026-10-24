import pytest

pytestmark = pytest.mark.integration  # downloads bge-reranker-base (~1.1 GB) once


def test_relevant_passage_outscores_irrelevant_one_with_raw_logits():
    from app.retrieval.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    relevant, irrelevant = reranker.score(
        "When was Arthur's Magazine founded?",
        [
            "Arthur's Magazine was an American literary periodical founded in 1844.",
            "Radio City is India's first private FM radio station.",
        ],
    )
    assert relevant > irrelevant
    assert irrelevant < 0 < relevant  # raw logits, not probabilities squashed into [0, 1]
