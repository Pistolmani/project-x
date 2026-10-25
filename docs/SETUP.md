# SETUP — Local Dev, Offline Ingestion & Deployment

> Run it locally, ingest the corpus offline, and deploy to Streamlit Community Cloud. Covers the **free-tier
> limits and secret handling the brief explicitly requires**. Decision IDs refer to `GRILL-ME.md`.

---

## 1. Prerequisites & accounts (all free tier)

- **Qdrant Cloud** — a free cluster (vectors + document registry, D2/D6).
- **Google AI Studio** — a Gemini API key (D3/A2).
- **Groq** — an API key (fallback LLM, A2).
- **Streamlit Community Cloud** — hosts the live demo (sign in with GitHub). See §5 for why not HF Spaces.
- **Hugging Face** — only for downloading HotpotQA, the `wikimedia/wikipedia` dataset and the models.
- Python 3.11+.

---

## 2. Secrets & config (A5 — "managed properly")

Never commit secrets. Use `pydantic-settings` reading a local **`.env`** (git-ignored); on Streamlit Cloud,
the app's **Secrets** box (same variable names; top-level secrets become environment variables). Committed
templates without values: `.env.example` and `.streamlit/secrets.toml.example`.

```
QDRANT_URL=...
QDRANT_API_KEY=...
GEMINI_API_KEY=...
GROQ_API_KEY=...
APP_PORT=8000          # FastAPI
STREAMLIT_PORT=8501
```

---

## 3. Local development

```
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt                # runtime (requirements.txt) + torch, eval, tests
uvicorn app.api.main:app --reload --port 8000      # backend (http://localhost:8000)
streamlit run ui/streamlit_app.py --server.port 8501         # UI (separate terminal, talks to API_URL=http://localhost:8000)
python -m pytest                                   # the A7 test suite (147 tests; -m integration for model tests)
```

---

## 4. Offline bulk ingestion (do NOT run on the app host — review fix)

Embedding ~50k long articles on the free host's CPU is far too slow. Run ingestion **locally** (or on a
free Colab GPU) into a local Qdrant, then copy the chosen strategy to Qdrant Cloud; the deployed app
only *queries*.

```
# Local Qdrant for development (data persists in ./qdrant_storage, git-ignored)
docker run -d --name qdrant -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant

# Stage 1 (CPU): select eval questions + ~50k titles, load articles, swap leads, absence-check.
#   Writes data/corpus/docs.jsonl, data/eval/questions.json, data/manifest.json.
python -m ingest.build_corpus --stage build --target 50000 --seed 42

# Stage 2 (GPU if present): chunk with each strategy, embed dense + sparse, upsert to QDRANT_URL.
#   Resumable: already-indexed (content, strategy) pairs are skipped.
python -m ingest.build_corpus --stage index --strategies recursive,fixed

# Compare the strategies locally (EVALUATION.md §7.1), then deploy only the winner.
python -m evaluation.quick_retrieval --set tuning --strategy fixed
python -m evaluation.tune_weights --strategy fixed

# Stage 3: point .env at the cloud cluster (QDRANT_URL + QDRANT_API_KEY), then copy the stored vectors
#   from local Qdrant (nothing is re-embedded). Idempotent; --start-offset resumes a cut-off run.
python -m ingest.copy_to_cloud --strategy fixed --source-url http://localhost:6333
```

The first `build` downloads HotpotQA (~360 MB) and the English Wikipedia shards (~11.6 GB) into
`.cache/` inside the project. On the author's machine (RTX 3060 Laptop, 6 GB) dense embedding runs on
the GPU; on a CPU-only machine it works but is much slower.

Details (article/label handling, 50k selection, sizing) are in `DATASET.md`.

---

## 5. Deploy to Streamlit Community Cloud (A4)

**Why not Hugging Face Spaces (the original plan, D4):** since 2026, Docker and Gradio Spaces on the free
`cpu-basic` hardware need a paid plan (PRO); free accounts only get Gradio on ZeroGPU. Streamlit Community
Cloud is free, deploys straight from the GitHub repo, and redeploys on every push.

**Constraints and how the app fits them:**

| Streamlit Cloud | The app's answer |
|---|---|
| Runs only `streamlit run <file>` — one process | `EMBEDDED_API=true`: the entry script starts the FastAPI backend in a background thread on `127.0.0.1:8000` (`ui/embedded_api.py`). The UI still talks to it over HTTP, so it stays a thin client, exactly as with two processes. |
| ~2.7 GB RAM, CPU only | `MODEL_BACKEND=onnx`: bge-small, BM25 and bge-reranker-base run through FastEmbed/ONNX, so torch is not installed (`requirements.txt` is the slim runtime; `requirements-dev.txt` adds torch for local GPU ingestion). Measured peak: **2.13 GB** with Streamlit + API + three cold `/ask` calls (re-ranker batch 8; batch 32 peaked at 2.46 GB). |
| Same results as the evaluated system | ONNX vs torch on the cloud index: identical dense retrieval (dense-only R@10 0.795 for both), full pipeline R@10 0.920 / MRR 0.956 vs 0.915 / 0.953 in the report; re-ranker logits within ~0.13. Details in `results/onnx_parity.json`. |
| Public: anyone can use it | `DEMO_MODE=true`: visitors can only update/delete documents they uploaded there (tag `demo_upload`); the corpus and the eval's planted documents return 403. `ASK_LIMIT_PER_HOUR=60` caps `/ask` across all visitors (they share one localhost client IP) to protect the free LLM quota; uploads capped at 10 MB. |

**Deploy (once):**
1. share.streamlit.io → sign in with GitHub → **Create app** → repo `Pistolmani/project-x`, branch `main`,
   main file `ui/streamlit_app.py`.
2. **Advanced settings → Secrets:** paste `.streamlit/secrets.toml.example` with the four keys filled in.
3. Deploy. First start: dependency install (~1–2 min with uv), then the models download (~1.2 GB) and load
   behind a "Starting the backend" spinner. Later restarts reuse the downloaded models until the container is
   recycled.

The deployed app runs on Python 3.14 (Streamlit Cloud's default); the pinned requirements install and run
there. Locally the project uses 3.11.

**Other hosts:** the `Dockerfile` (+ `docker/start.sh`) runs the same app on any Docker host: API on
127.0.0.1:8000, Streamlit on 8501, ONNX models baked into the image.

---

## 6. Free-tier limits & mitigations (required by the brief — review fix)

| Limit | Reality | Mitigation |
|-------|---------|------------|
| **Streamlit app sleeps when idle** | Community Cloud apps without traffic for 12 h go to sleep; the next visitor wakes them (container start + model load, about a minute or more). | Accept for a demo; open the app before an interview or review. |
| **Qdrant free cluster suspends, then deletes** | Unused free clusters are **suspended after 1 week** and **deleted after 4 weeks** if not reactivated. | Periodic query to keep it active through the eval/interview window; if lost, **rebuild from the seeded manifest** with the idempotent ingest script (`DATABASE.md` §8). |
| **Gemini free-tier rate limits** | The eval makes >1,000 LLM calls. | LLM-free retrieval eval, single-config `/ask`, caching, Groq fallback (`EVALUATION.md` §7, `BACKEND.md` §8). |
| **Qdrant RAM (1 GB) and disk (4 GB)** | ~490k recursive chunks; float32 vectors in RAM alone would nearly fill it, and both strategies (~3.9 GB est.) are too close to the disk limit. | int8 vectors in RAM, **originals + payload + sparse index on disk**; compare chunking **locally at 50k**, push **only the winner** to the cloud (`DATABASE.md` §7). |
| **Host CPU for embedding** | Too slow for 50k docs. | Offline/Colab ingestion (§4); the host only embeds queries and visitors' uploads. |
| **Host RAM (~2.7 GB) and CPU latency** | Re-ranking 20 passages per search on 2 shared CPU cores; a multi-hop `/ask` takes ~30–60 s locally on CPU. | ONNX models, re-ranker batch 8 (§5); progress streams to the UI while the agent works. |
| **Shared LLM quota on a public URL** | One visitor could spend the day's free Gemini/Groq requests. | `ASK_LIMIT_PER_HOUR` (60) plus the answer cache. |

---

## 7. Persistence boundary (recap)

Durable state = **Qdrant Cloud** (vectors + `documents` registry). The app host is **stateless compute**; a
rebuild reconnects to the same collections and loses nothing (`DATABASE.md` §1).

---

## 8. Deliverables checklist

- [ ] GitHub repo · [ ] live Space URL · [ ] README (architecture + rationale) · [ ] Evaluation Report
      (post-run) · [ ] AI Usage Report (`AI-USAGE.md`).
