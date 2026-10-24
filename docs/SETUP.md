# SETUP — Local Dev, Offline Ingestion & Deployment

> Run it locally, ingest the corpus offline, and deploy to Hugging Face Spaces. Covers the **free-tier
> limits and secret handling the brief explicitly requires**. Decision IDs refer to `GRILL-ME.md`.

---

## 1. Prerequisites & accounts (all free tier)

- **Qdrant Cloud** — a free cluster (vectors + document registry, D2/D6).
- **Google AI Studio** — a Gemini API key (D3/A2).
- **Groq** — an API key (fallback LLM, A2).
- **Hugging Face** — account for the Space (D4) and the `wikimedia/wikipedia` dataset.
- Python 3.11+.

---

## 2. Secrets & config (A5 — "managed properly")

Never commit secrets. Use `pydantic-settings` reading a local **`.env`** (git-ignored); on the Space, use
**Space secrets** (same variable names). A committed `.env.example` documents the keys without values.

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
pip install -r requirements.txt
uvicorn app.api.main:app --reload --port 8000      # backend (http://localhost:8000)
streamlit run ui/app.py --server.port 8501         # UI (separate terminal, talks to API_URL=http://localhost:8000)
python -m pytest                                   # the A7 test suite (125 backend + UI AppTest tests)
```

---

## 4. Offline bulk ingestion (do NOT run on the Space — review fix)

Embedding ~50k long articles on the Space's free CPU is far too slow. Run ingestion **locally** (or on a
free Colab GPU) into a local Qdrant, then copy the chosen strategy to Qdrant Cloud; the deployed Space
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

## 5. Deploy to Hugging Face Spaces (Docker — A4)

A **Docker Space** with one entrypoint runs **both** processes:

```dockerfile
# Dockerfile (sketch)
CMD ["bash", "-lc", "uvicorn app.api.main:app --port 8000 & streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"]
```

A Space routes public traffic to **one port only**, declared in the YAML header of the Space's
`README.md`. Point it at Streamlit; FastAPI on 8000 stays internal (reached by Streamlit over localhost),
which is exactly the intended topology:

```yaml
---
sdk: docker
app_port: 8501
---
```

Set the four secrets in **Space → Settings → Secrets**. The Space connects to the Qdrant cluster populated
in §4 — no data is embedded on the Space itself.

---

## 6. Free-tier limits & mitigations (required by the brief — review fix)

| Limit | Reality | Mitigation |
|-------|---------|------------|
| **HF Space sleeps when idle** | Free Spaces pause after inactivity; first hit cold-starts (model download + load). | Accept for a demo; optionally a lightweight keep-alive ping; document the first-request delay. |
| **Qdrant free cluster suspends, then deletes** | Unused free clusters are **suspended after 1 week** and **deleted after 4 weeks** if not reactivated. | Periodic query to keep it active through the eval/interview window; if lost, **rebuild from the seeded manifest** with the idempotent ingest script (`DATABASE.md` §8). |
| **Gemini free-tier rate limits** | The eval makes >1,000 LLM calls. | LLM-free retrieval eval, single-config `/ask`, caching, Groq fallback (`EVALUATION.md` §7, `BACKEND.md` §8). |
| **Qdrant RAM (1 GB) and disk (4 GB)** | ~490k recursive chunks; float32 vectors in RAM alone would nearly fill it, and both strategies (~3.9 GB est.) are too close to the disk limit. | int8 vectors in RAM, **originals + payload + sparse index on disk**; compare chunking **locally at 50k**, push **only the winner** to the cloud (`DATABASE.md` §7). |
| **Space CPU for embedding** | Too slow for 50k docs. | Offline/Colab ingestion (§4). |

---

## 7. Persistence boundary (recap)

Durable state = **Qdrant Cloud** (vectors + `documents` registry). The Space is **stateless compute**; a
rebuild reconnects to the same collections and loses nothing (`DATABASE.md` §1).

---

## 8. Deliverables checklist

- [ ] GitHub repo · [ ] live Space URL · [ ] README (architecture + rationale) · [ ] Evaluation Report
      (post-run) · [ ] AI Usage Report (`AI-USAGE.md`).
