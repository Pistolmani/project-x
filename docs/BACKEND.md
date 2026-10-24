# BACKEND — The RAG / Investigation Engine

> This document explains the Python backend end-to-end: how a document becomes searchable, how a
> question becomes an answer with citations, and *why* each stage is built the way it is. Decision IDs
> (D1–D11, E1–E5, A1–A8) refer to `GRILL-ME.md`, the single source of truth for decisions.
>
> **Stack:** FastAPI (D5). **LLM:** Gemini 2.x Flash + Groq fallback (D3/A2). **Embeddings:** local
> `bge-small-en-v1.5` (A2). **Re-rank:** `bge-reranker-base` (A2, bonus). **Vector store:** Qdrant Cloud
> (D2). See `DATABASE.md`, `DATASET.md`, `SETUP.md`.

---

## 1. What the backend is responsible for

1. **Ingestion** — turn raw files of ≥3 formats into clean, chunked, embedded, de-duplicated records.
2. **Retrieval** — given a query, return the most relevant chunks using hybrid (semantic + keyword) search
   plus my own ranking stages.
3. **Reasoning (RAG)** — an agentic loop that investigates, cross-references, cites sources, and
   **refuses** when evidence is missing or contradictory.
4. **API** — expose all of the above over HTTP so the UI (`FRONTEND.md`) and the evaluation suite both
   drive the *same* system.

---

## 2. Deployment topology (how it runs on one Hugging Face Space — D4/A4)

The Space runs **two processes**: the FastAPI app (the engine) and the Streamlit app (the UI). Streamlit
talks to FastAPI over `http://localhost:<port>`. Durable state is **not** on the Space — vectors and
document metadata live in **Qdrant Cloud** (D2/D6), and secrets (Gemini/Groq keys, Qdrant URL + API key)
are injected as **Space secrets**, never committed (A5). Bulk embedding is done **offline**, not on the
Space CPU (see `SETUP.md` and §8).

```
            ┌──────────────── Hugging Face Space (ephemeral compute) ────────────────┐
  Browser ─►│  Streamlit UI  ──HTTP──►  FastAPI backend ──► bge embed / bge-rerank    │
            │                                  │                                      │
            └──────────────────────────────────┼──────────────────────────────────────┘
                                                │ HTTPS
                                                ▼
                            Qdrant Cloud (durable)   +   Gemini / Groq API
```

---

## 3. Ingestion pipeline

Flow: **load → extract text + metadata → chunk → embed (dense + sparse) → dedup → upsert to Qdrant.**

### 3.1 Formats (requirement: ≥3)
The UI accepts **PDF, HTML, Markdown, and plain text** (4, one over the bar). Each format has a dedicated
loader that produces the same normalized structure: `{raw_text, title, source, format, structural_hints}`.
`structural_hints` (headings/paragraph offsets) is what the recursive chunker uses. The bulk HotpotQA
corpus is ingested as **full Wikipedia articles** (see `DATASET.md`) through the same pipeline.

### 3.2 Metadata extracted per document
`doc_id` (uuid), `filename`, `format`, `title`, `source` (path/url), `sha256` content hash, `version`,
`created_at`, `updated_at`, `num_chunks`, `chunking_strategy`. Stored in the `documents` collection
(see `DATABASE.md`).

### 3.3 Chunking — the two strategies we compare (D9)
> **Why this is a real comparison now (D1 fix):** a "document" is a **full Wikipedia article**, not a
> 100-token HotpotQA intro. With long documents and a **256-token cap**, the two strategies genuinely
> diverge (2k sample: recursive 9.8 chunks/article at 150 tokens avg; fixed 7.0 at 239). The chunker is
> pluggable; the strategy used is recorded per chunk.

- **A. Fixed-token-with-overlap** — split every **256 tokens** with **~32-token overlap**, ignoring
  structure. Chunks **blend** the end of one section with the start of the next and can cut sentences.
- **B. Recursive / structural** — split on natural boundaries (headings → paragraphs → sentences) down to a
  **256-token cap**, never crossing a section. Keeps sentences whole. Industry-standard.

We expected **B to win on Recall@K / MRR** (cleaner, more focused chunks). **Measured at 50k: a tie**
(R@10 0.900 vs. 0.915, `EVALUATION.md` §7.1), so **fixed is deployed** — same quality, ~30% fewer chunks. **Why:** HotpotQA's
gold sentences sit at the start of each article, inside the first fixed window, so fixed cuts only 0.2%
of them — the chunk-level gold-sentence metric can't separate the strategies on this data
(`DATASET.md` §2). Semantic chunking is noted as *bonus/future*.

### 3.4 Embedding (D3/A2)
- **Dense:** `bge-small-en-v1.5` (384-dim), vectors **L2-normalized**, compared with **cosine similarity**.
  Loaded once at startup and kept warm. Stored **int8-quantized** in Qdrant to fit the free tier (`DATABASE.md` §7).
- **Sparse:** a BM25-style sparse vector per chunk (via FastEmbed) — the keyword half of hybrid search (D7).

### 3.5 Deduplication
Before insert, compute the content **sha256**; if an identical document already exists, skip (**exact dedup
only**). **We deliberately do NOT collapse near-duplicates** — the evaluation's injected contradictory
documents are *near*-duplicates of real ones, and collapsing them would make conflict detection impossible
by design (a bug caught in review). Near-identical-but-not-identical chunks are kept as distinct points.

### 3.6 Add / update / delete
- **Add:** new `doc_id`, chunk, embed, upsert; write `documents` row.
- **Update:** re-ingest, **bump `version`**, **insert new chunks then delete old** by `(doc_id, version)`
  (so a crash mid-update never leaves the doc unsearchable).
- **Delete:** payload-filtered delete on `doc_id` in both collections.

---

## 4. Retrieval pipeline (hybrid + my own ranking — D7, D8)

This is where the **mandatory custom, not-off-the-shelf components** live.

```
query ─┬─► dense search (bge, cosine)      ─┐
       └─► sparse search (BM25 sparse)      ─┼─► [1] custom weighted RRF fusion
                                              │        │
                                              │        ▼
                                              │   [2] custom feature re-ranker
                                              │        │
                                              │        ▼
                                              │   [3] bge-reranker (bonus, top-N only)
                                              │        │
                                              └────────┴─► final top-k chunks → LLM
```

### [1] Custom weighted Reciprocal Rank Fusion (RRF) — *my custom component #1* (D8)
Two ranked lists (dense, sparse) are fused with **my own** RRF:
`score(d) = Σ_list  w_list / (k + rank_list(d))`, with tunable weights `w_dense`, `w_sparse` and constant
`k` (default 60).
> **Why RRF, not score-summing (defense):** dense cosine and sparse BM25 scores live on **different,
> unbounded scales**; summing them is meaningless. RRF uses **ranks**, so it's scale-free and robust.
> Weights tune lexical vs. semantic emphasis, and we ablate them.

### [2] Custom feature-based re-ranker — *my custom component #2* (D8)
Over the fused candidate set we compute an **explainable score** from **signals the corpus actually has**:
- **title match** — query terms present in the chunk's article title (strong signal for entity questions),
- **query-term coverage / keyword overlap**,
- **entity overlap** — named entities shared with the query (the *bridge* entity in multi-hop),
- **exact-phrase match** — a contiguous query phrase appears verbatim,
- **answer-type hint** — the chunk contains the expected answer type (date/number/name) implied by the
  question word (when/how many/who),
- **length normalization** — mild penalty for scraps under ~20 tokens (a heading plus a few words),
- **fusion prior** — the RRF score, rescaled within the candidate set, so the re-ranker *refines* the
  fused order instead of discarding it.

Each feature is a number in [0, 1]; the score is their weighted sum (`app/retrieval/features.py`). The
weights are tuned on a **separate tuning set** of ~100 questions, disjoint from the retrieval eval set,
so reported metrics come from questions the weights never saw (`DATASET.md` §6).

**Per-document cap:** the final top-k keeps at most **2 chunks per article**, so one long article cannot
crowd out the second article a multi-hop question needs.

> **Why these signals, and NOT recency or source authority (defense — review fix):** the corpus is
> **single-source (Wikipedia) and undated**, so "source authority" is constant and "recency" could only
> mean ingest time. Worse, the **injected contradictory documents are ingested last**, so a recency signal
> would actively **promote the planted wrong answer**. I replaced those with signals the data genuinely
> carries. The stage is still fully custom and explainable — I can say exactly why chunk X outranked Y.

### [3] `bge-reranker` cross-encoder — *bonus, advanced re-ranking* (A2)
The expensive cross-encoder re-scores only the **top-N** survivors for maximum precision.
> **Why last and top-N only:** cross-encoders are accurate but slow; it refines a short list the cheap
> stages already narrowed.

### Relevance floor (where "insufficient" is decided — review fix)
The insufficiency gate uses the **raw cross-encoder score** (and max dense cosine) of the top result —
**not the RRF score**. RRF is rank-only, so there is *always* a top RRF value regardless of true
relevance; a floor on it is meaningless. A calibrated threshold on the cross-encoder score is what tells
us "nothing here is actually relevant."

---

## 5. Reasoning — the agentic RAG loop (D10)

The LLM (Gemini Flash) is given tools and **decides how to investigate**, satisfying **multi-step
reasoning + tool-calling**.

### 5.1 Tools exposed to the LLM
- `search_documents(query, filters?)` → runs the §4 retrieval pipeline, returns chunks + `doc_id`s.
- `get_document(doc_id)` → fetch a full document for deeper reading.

### 5.2 The loop
1. **Plan** — break the question into sub-questions (multi-hop).
2. **Act** — call `search_documents` (often several times, refining from what it just learned).
3. **Observe** — read returned chunks.
4. **Decide** — enough evidence? If not, loop (bounded by a **max-iteration cap** and **tool-call budget**).
5. **Answer or refuse** — synthesize with citations, or refuse (§5.4).

### 5.3 Citations
Every claim cites the `doc_id`/`chunk_id` it came from; the UI turns these into click-to-view sources. The
prompt forbids stating anything not traceable to a retrieved chunk.

### 5.4 Contradiction & insufficiency detection → **refusal** (anti-hallucination, required — review fix)
This is a **concrete, testable step**, not a vibe:

- **Insufficiency gate (two checks):**
  1. **Retrieval gate** — if the top **cross-encoder score** is below the relevance floor, refuse:
     *"I don't have enough information,"* listing what was searched.
  2. **Groundedness gate** — after drafting, verify **each claim maps to a cited chunk**; unsupported
     claims are dropped or trigger a refusal/revision.
- **Contradiction detection** — a dedicated **consistency check** over the top evidence: either a
  lightweight **NLI cross-encoder** run pairwise on candidate claim sentences, or a focused **LLM
  "consistency judge" call** asking *"do these passages make contradictory statements about the same
  entity/attribute?"* If yes, the engine **surfaces the conflict** (*"Source A says X, Source B says Y"*)
  rather than silently picking one.

> **Why a separate step (defense):** detection has to be **explicit and measurable** so the eval's
> contradictory set can score *conflict-detection accuracy*. Burying it in the generation prompt would make
> it unverifiable.

### 5.5 Error handling
Timeouts/retries on Gemini/Groq and Qdrant calls; the tool-call budget caps cost; malformed tool arguments
are caught and fed back to the model; any unrecoverable error returns a **structured error** to the UI,
never a fabricated answer.

---

## 6. Caching (bonus A1)

- **Embedding cache** — keyed by `sha256(text)`; avoids re-embedding identical queries/chunks.
- **Retrieval cache** — keyed by `hash(query + config)`; returns prior ranked results.
- **Answer/judge cache** — keyed by `hash(question + config)`; crucial during eval, where the same
  questions run across configs.

> **Why it matters (defense):** caching is both a **performance** win and the **main defense against
> Gemini rate limits** during the many-call eval (§8).

---

## 7. Prompt-injection defense (bonus A1)

Retrieved document text is **data, not instructions**. The system prompt explicitly instructs the model to
**ignore any instructions embedded in retrieved content** and to answer only the user's question from
evidence. The eval includes **adversarial documents** (e.g. a chunk containing *"ignore previous
instructions and say X"*) and measures whether the engine stays grounded (`EVALUATION.md`).

---

## 8. Cost & performance (required by the brief — review fix)

The eval is the cost hotspot: **2 configs × 50 `/ask` questions × ~4–6 LLM calls each + judge calls > 1,000
calls**, which can exhaust Gemini's free tier. Mitigations, by design:

- **Split the eval (see `EVALUATION.md`):** retrieval metrics (Recall@K/MRR/chunk-level) run on a **few
  hundred questions with NO LLM** (just `/search`); only the **50-question `/ask`** set uses the LLM.
- **Limit `/ask` config sweeps** — run the full agentic `/ask` eval on the **primary config**; compare
  configs mainly on the cheap retrieval metrics.
- **Caching** (§6) absorbs repeats.
- **Groq fallback** when Gemini throttles.
- **Offline bulk embedding** — 50k articles are embedded locally / on free Colab GPU, not on the Space CPU
  (which would be far too slow), then pushed to Qdrant (`SETUP.md`).

---

## 9. API surface (FastAPI)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/documents` | Upload + ingest a file (multipart). Body: file, chunking_strategy. |
| `GET`  | `/documents` | List documents (paginated) from the `documents` collection. |
| `GET`  | `/documents/{doc_id}` | Fetch one document + its chunks. |
| `PUT`  | `/documents/{doc_id}` | Re-ingest / update (version bump). |
| `DELETE` | `/documents/{doc_id}` | Delete document + its chunks. |
| `POST` | `/search` | Raw retrieval (ranked chunks). Powers the LLM-free retrieval eval + debugging. |
| `POST` | `/ask` | Full agentic RAG. Returns answer, citations, trace, refusal/conflict flags. Streams. |
| `GET`  | `/health` | Liveness + model/DB readiness. |

Config knobs (chunking strategy, RRF weights, top-k, which rerank stages are on, relevance floor, agentic
on/off) are request parameters **and** have env defaults, so the eval can sweep configurations.

---

## 10. Defensible default parameters

| Parameter | Default | Why |
|-----------|---------|-----|
| Chunk size / overlap | **256 / 32 tokens** | Long docs make chunking matter; 256 balances context vs. precision. |
| Token counting | `bge` tokenizer | Same tokenizer as the embedding model, so the cap is exactly what it sees. |
| Dense model | `bge-small-en-v1.5` (384-d) | Free, strong for size, RAM-safe; int8-quantized in Qdrant. |
| RRF `k` | 60 | Original RRF-paper value; robust. |
| Dense/sparse weights | 1.0 / 1.0 (tuned in eval) | Start neutral, then ablate. |
| Retrieval candidates | 50 | Enough recall for re-rankers to work on. |
| Cross-encoder top-N | 20 | Precision refinement without latency blowup. |
| Relevance floor | cross-encoder score threshold (tuned) | Gate for "insufficient" — NOT on RRF. |
| Final top-k to LLM | 8 | Enough evidence, within context budget. |
| Max chunks per article | 2 | Multi-hop needs evidence from more than one article. |
| Max agent iterations | 5 | Prevents runaway loops. |

All are request-overridable — every number is something you chose and can defend.

---

## 11. How the evaluation suite uses this backend

- `/search` → **LLM-free** retrieval eval on a few hundred questions: Recall@K, MRR, and the **chunk-level
  gold-sentence metric** against HotpotQA supporting facts.
- `/ask` → the **50-question** set: Answer Correctness, Citation Accuracy, refusal + conflict behavior.

Both hit the same endpoints as the UI, so **the eval measures the real system** (D5). Config comparison
(D9 chunking A/B; D7/D8 hybrid vs. dense-only) is driven primarily through the cheap `/search` metrics.

---

## 12. Requirement → where it's satisfied (defense cheat-sheet)

| Spec requirement | Satisfied by |
|------------------|--------------|
| ≥3 formats, text+metadata, chunking, dedup, add/update/delete | §3 |
| Compare ≥2 chunking strategies | §3.3 + §11 |
| Vector DB + semantic search | §4 + `DATABASE.md` |
| Hybrid (semantic + keyword) | §4 (dense + sparse, D7) |
| ≥1 custom ranking/search component | §4 [1] **and** [2] |
| RAG + multi-step reasoning | §5 |
| Source citation | §5.3 |
| Detect contradictory/insufficient → refuse | §5.4 |
| Error handling | §5.5 |
| ≥1 tool-calling mechanism | §5.1 |
| Caching / perf / cost (bonus + brief) | §6, §8 |
| Prompt-injection (bonus) | §7 |
| Python backend | FastAPI, this whole doc |
