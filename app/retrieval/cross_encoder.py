"""`bge-reranker-base` cross-encoder: the expensive precision pass on the short list (A2, bonus).

It reads query and chunk *together*, so it is far more accurate than comparing two separate embeddings,
but it costs one transformer pass per pair; that is why it only sees the top-N survivors.
Scores are the raw logits (no sigmoid): the relevance floor that decides "insufficient evidence" is a
threshold on this value, calibrated on real scores (`evaluation/calibrate_floor.py`, BACKEND.md §4).
"""


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int = 512,
        backend: str = "torch",
    ):
        """`backend="onnx"` runs the same model through FastEmbed without torch (CPU deployments); its
        logits stay within ~0.13 of torch's (results/onnx_parity.json)."""
        self.backend = backend
        # ONNX: small batches keep onnxruntime's memory arena ~330 MB lower at the same speed (2.7 GB host)
        self.batch_size = batch_size or (8 if backend == "onnx" else 32)
        if backend == "onnx":
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            from app.ingestion.embedding import ONNX_OPTIONS

            self.model = TextCrossEncoder(model_name, **ONNX_OPTIONS)  # returns raw logits, truncates at 512 tokens
            return
        import torch
        from sentence_transformers import CrossEncoder

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        if self.device == "cuda":
            self.model.model.half()
        self._identity = torch.nn.Identity()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self.backend == "onnx":
            from app.ingestion.embedding import INFERENCE_LOCK

            with INFERENCE_LOCK:
                return [float(s) for s in self.model.rerank(query, texts, batch_size=self.batch_size)]
        scores = self.model.predict(
            [(query, t) for t in texts],
            batch_size=self.batch_size,
            activation_fn=self._identity,  # raw logits, not sigmoid probabilities
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
