import pytest

import app  # noqa: F401  (sets HF_HOME before any model library is imported)


@pytest.fixture(scope="session")
def tokenizer():
    from app.config import get_settings
    from app.ingestion.chunking import load_tokenizer

    return load_tokenizer(get_settings().dense_model)
