"""Dense (bge) and sparse (BM25) embedders: the two halves of hybrid search (D3, D7).

Heavy libraries are imported inside the constructors so importing this module stays cheap.
"""

# bge v1.5 recommends this prefix for short retrieval queries; passages are embedded as-is.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 128):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()  # fp16 on GPU: ~2x faster, no measurable retrieval loss for bge-small
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
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
