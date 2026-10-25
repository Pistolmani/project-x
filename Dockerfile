# Optional: runs the app on any Docker host (the live demo is on Streamlit Community Cloud, SETUP.md §5).
# FastAPI on 127.0.0.1:8000 (internal) + Streamlit on 8501 (public). The index lives in Qdrant Cloud; the
# image holds code, the ONNX models and the tracked results/ only. Secrets come in as environment variables.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    MODEL_BACKEND=onnx \
    DEMO_MODE=true \
    ASK_LIMIT_PER_HOUR=60 \
    MAX_UPLOAD_MB=10
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the models into the image so a cold start does not download ~1.2 GB
COPY --chown=user app/__init__.py app/config.py app/
COPY --chown=user app/ingestion/__init__.py app/ingestion/embedding.py app/ingestion/
COPY --chown=user app/retrieval/__init__.py app/retrieval/cross_encoder.py app/retrieval/
RUN python -c "from app.ingestion.embedding import DenseEmbedder, SparseEmbedder; \
from app.retrieval.cross_encoder import CrossEncoderReranker; \
DenseEmbedder('BAAI/bge-small-en-v1.5', backend='onnx'); SparseEmbedder('Qdrant/bm25'); \
CrossEncoderReranker(backend='onnx')"

COPY --chown=user . .
RUN mkdir -p data

EXPOSE 8501
CMD ["bash", "docker/start.sh"]
