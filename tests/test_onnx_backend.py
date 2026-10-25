"""The ONNX backend must work where torch is not installed (Streamlit Cloud) and match the torch backend."""

import subprocess
import sys

import pytest

from app import PROJECT_ROOT

NO_TORCH = """
import sys
sys.modules["torch"] = None  # any `import torch` now raises ImportError, as on the slim deployment
sys.modules["sentence_transformers"] = None
import fastembed, fastembed.rerank.cross_encoder as fce

class FakeEmbedding:
    def __init__(self, name, **options): assert options == {'enable_cpu_mem_arena': False}
    def embed(self, texts, batch_size=None): return [[3.0, 4.0] for _ in texts]

class FakeCrossEncoder:
    def __init__(self, name, **options): assert options == {'enable_cpu_mem_arena': False}
    def rerank(self, query, texts, batch_size=None): return [float(len(t)) for t in texts]

fastembed.TextEmbedding = FakeEmbedding
fce.TextCrossEncoder = FakeCrossEncoder
from app.ingestion.embedding import DenseEmbedder
from app.retrieval.cross_encoder import CrossEncoderReranker
dense = DenseEmbedder("m", backend="onnx")
assert dense.batch_size == 16
assert dense.dim == 2 and [round(x, 6) for x in dense.embed_query("q")] == [0.6, 0.8]  # normalized
assert CrossEncoderReranker(backend="onnx").score("q", ["ab", "abcd"]) == [2.0, 4.0]
print("ok")
"""


def test_onnx_backend_never_imports_torch():
    out = subprocess.run([sys.executable, "-c", NO_TORCH], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-2000:]


@pytest.mark.integration  # loads both model versions (~1.3 GB, cached after the first run)
def test_onnx_and_torch_backends_agree():
    import numpy as np

    from app.ingestion.embedding import DenseEmbedder
    from app.retrieval.cross_encoder import CrossEncoderReranker

    texts = ["Who directed Jaws?", "Arthur's Magazine was an American literary periodical founded in 1844."]
    a = np.array(DenseEmbedder("BAAI/bge-small-en-v1.5", device="cpu").embed_documents(texts))
    b = np.array(DenseEmbedder("BAAI/bge-small-en-v1.5", backend="onnx").embed_documents(texts))
    assert (a * b).sum(1).min() > 0.999

    query, passages = "When was Arthur's Magazine founded?", texts
    torch_scores = CrossEncoderReranker(device="cpu").score(query, passages)
    onnx_scores = CrossEncoderReranker(backend="onnx").score(query, passages)
    assert max(abs(x - y) for x, y in zip(torch_scores, onnx_scores)) < 0.2
