"""Dense (bge) and sparse (BM25) embedders: the two halves of hybrid search (D3, D7).

Heavy libraries are imported inside the constructors so importing this module stays cheap.
"""

# bge v1.5 recommends this prefix for short retrieval queries; passages are embedded as-is.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseEmbedder:
    """`backend="torch"` (sentence-transformers, GPU for bulk ingestion) or `"onnx"` (FastEmbed, CPU-only
    deployments without torch). Both give the same normalized vectors (results/onnx_parity.json), so
    queries embedded either way search the same index."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 128, backend: str = "torch"):
        self.backend = backend
        self.batch_size = batch_size
        if backend == "onnx":
            from fastembed import TextEmbedding

            self.model = TextEmbedding(model_name)
            self._dim = len(next(iter(self.model.embed(["dimension probe"]))))
            return
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()  # fp16 on GPU: ~2x faster, no measurable retrieval loss for bge-small
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "onnx":
            import numpy as np

            vectors = np.array(list(self.model.embed(texts, batch_size=self.batch_size)), dtype="float32")
            return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).tolist()
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([BGE_QUERY_INSTRUCTION + text])[0]


class SparseEmbedder:
    """BM25 term weights from FastEmbed. IDF is applied by Qdrant at query time (Modifier.IDF),
    so document vectors stay valid as the corpus grows."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [(e.indices.tolist(), e.values.tolist()) for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        e = next(iter(self.model.query_embed(text)))
        return e.indices.tolist(), e.values.tolist()
