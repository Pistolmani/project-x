"""`bge-reranker-base` cross-encoder: the expensive precision pass on the short list (A2, bonus).

It reads query and chunk *together*, so it is far more accurate than comparing two separate embeddings,
but it costs one transformer pass per pair; that is why it only sees the top-N survivors.
Scores are the raw logits (no sigmoid): the relevance floor that decides "insufficient evidence" is a
threshold on this value, calibrated on real scores in Phase 2b (BACKEND.md §4, relevance floor).
"""


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        import torch
        from sentence_transformers import CrossEncoder

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        if self.device == "cuda":
            self.model.model.half()
        self.batch_size = batch_size
        self._identity = torch.nn.Identity()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, t) for t in texts],
            batch_size=self.batch_size,
            activation_fn=self._identity,  # raw logits, not sigmoid probabilities
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
