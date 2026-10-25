# Hugging Face Docker Space (SETUP.md §5): FastAPI on 127.0.0.1:8000 (internal) + Streamlit on 8501 (public).
# The index lives in Qdrant Cloud; the image holds code, models and the tracked results/ only.
FROM python:3.11-slim

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    DEMO_MODE=true \
    ASK_LIMIT_PER_HOUR=60 \
    MAX_UPLOAD_MB=10
WORKDIR /home/user/app

# CPU-only torch first, so requirements.txt does not pull the multi-GB CUDA build
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Bake the three models into the image so a cold start does not download ~1.3 GB
COPY --chown=user app/__init__.py app/config.py app/
COPY --chown=user app/ingestion/__init__.py app/ingestion/embedding.py app/ingestion/
COPY --chown=user app/retrieval/__init__.py app/retrieval/cross_encoder.py app/retrieval/
RUN python -c "from app.ingestion.embedding import DenseEmbedder, SparseEmbedder; \
from app.retrieval.cross_encoder import CrossEncoderReranker; \
DenseEmbedder('BAAI/bge-small-en-v1.5'); SparseEmbedder('Qdrant/bm25'); CrossEncoderReranker()"

COPY --chown=user . .
RUN mkdir -p data

EXPOSE 8501
CMD ["bash", "docker/start.sh"]
