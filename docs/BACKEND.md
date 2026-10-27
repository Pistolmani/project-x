# BACKEND — The RAG / Investigation Engine

> This document explains the Python backend end-to-end: how a document becomes searchable, how a
> question becomes an answer with citations, and *why* each stage is built the way it is. Decision IDs
> (D1–D11, E1–E5, A1–A8) refer to `DECISIONS.md`, the single source of truth for decisions.
>
> **Stack:** FastAPI (D5). **LLM:** Gemini 3.5 Flash-Lite → 3.5 Flash → Groq `gpt-oss-120b` fallback chain (D3/A2, §5.5). **Embeddings:** local
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

## 2. Deployment topology (Streamlit Community Cloud — A4)

Planned for a Hugging Face Docker Space (D4). Those became paid in 2026, so the demo runs on **Streamlit
Community Cloud** (`SETUP.md` §5). It runs only `streamlit run`, so the UI process starts the FastAPI app
in a background thread (`EMBEDDED_API=true`) and talks to it over `http://127.0.0.1:8000`, the same HTTP
contract as two processes (locally, or with the `Dockerfile`). Durable state is **not** on the host:
vectors and document metadata live in **Qdrant Cloud** (D2/D6). Secrets (Gemini/Groq keys, Qdrant URL +
API key) are app secrets, never committed (A5). The host has ~2.7 GB RAM and no GPU, so models run through
**ONNX** (`MODEL_BACKEND=onnx`, FastEmbed). That gives the same vectors and re-ranker logits within ~0.13
of torch (`results/onnx_parity.json`). Bulk embedding is done **offline** (see `SETUP.md` and §8).

```
            ┌──────────── Streamlit Community Cloud (ephemeral compute) ─────────────┐
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

Three levels, none of which may hide a conflict: a planted contradictory document is a *near*-copy of a
real one by design, and collapsing it would make conflict detection impossible.

| Level | Where | Rule | Effect |
|---|---|---|---|
| **Exact document** | upload (`IngestionService.add`) | sha256 of the text with whitespace collapsed | not indexed again; the response names the existing document (`duplicate_of`) |
| **Near-duplicate document** | upload (`near_duplicate`) | one embedding of the opening finds the 5 most similar documents; each is compared in full by **containment**: the share of the new document's 5-word shingles found in it. ≥ 0.8 = near-copy | **indexed anyway**; the response and the UI name the document it resembles (`near_duplicate_of`) |
| **Identical passage** | search (`drop_duplicate_text`, after fusion) | same chunk content hash | only the best-ranked copy is kept, so duplicates don't take slots from other evidence |

Containment, not Jaccard: an excerpt of a long article should count as a copy, and Jaccard is diluted by
the longer document's extra text. Measured read-only on the cloud index (6 random articles): copies with
every 40th word changed score 0.87–0.91, half-article excerpts 0.97–1.0, unrelated text has no candidate.
The planted conflict fact sheets score 0.44–0.85 against their source articles, so some are flagged; they
are indexed either way. The passage-level step fired on **0 of 400** tuning and held-out questions, so the
evaluated retrieval numbers are unchanged.

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

The LLM (Gemini Flash-Lite, see §5.5 for why not Flash) is given tools and **decides how to investigate**, satisfying **multi-step
reasoning + tool-calling**.

### 5.1 Tools exposed to the LLM (`app/agent/tools.py`)
- `search_documents(query)` → runs the §4 retrieval pipeline, returns passages with **source ids**.
- `get_document(source_id)` → reads more of the article a source came from, in order (capped at ~6k chars).
- `submit_answer(status, answer, citations)` → the only way to finish; its schema enforces the output
  shape on both providers (`status` ∈ answered / insufficient / conflict; the agent adds `blocked` (§7)
  and `error` (§5.5)).

Every passage shown to the model gets a short handle (`S1`, `S2`, …) the first time it appears. The model
cites handles, not UUIDs: short ids are copied reliably, and a citation can only be valid if that handle
was actually shown.

### 5.2 The loop
1. **Plan** — break the question into sub-questions (multi-hop).
2. **Act** — call `search_documents` (often several times, refining from what it just learned).
3. **Observe** — read returned chunks.
4. **Decide** — enough evidence? If not, loop (bounded by **5 LLM iterations** and **8 tool calls**; when
   the budget runs out, one last call that can only `submit_answer`).
5. **Answer or refuse** — synthesize with citations, or refuse (§5.4).

The input guard runs before step 1 and the output guard after the gates (§7).

### 5.3 Citations
Every claim cites the `doc_id`/`chunk_id` it came from; the UI turns these into click-to-view sources. The
prompt forbids stating anything not traceable to a retrieved chunk.

### 5.4 Contradiction & insufficiency detection → **refusal** (anti-hallucination, required — review fix)
This is a **concrete, testable step**, not a vibe (`app/agent/gates.py`; every decision is recorded in
the answer's `gates` field):

**As built:** (1) **citation check** — cited handles must have been shown; an "answered" with no valid
citation becomes a refusal. (2) **retrieval gate** — the best cross-encoder score of the cited passages
must reach the **floor 3.92**, calibrated by `evaluation/calibrate_floor.py`: it passes 95% of answerable
tuning questions and blocks 71% of held-out (unanswerable) ones; the balanced optimum (5.75) would have
refused 15% of answerable questions, too much for a backstop. (3) **verifier** — one LLM call with a JSON
schema that judges both *supported* (paraphrase allowed — a first, stricter prompt refused a correct
answer over "best known for") and *conflict*. It sees the cited passages **plus the best uncited ones**,
because a planted conflicting passage is typically retrieved but not cited when the model silently picks
a side. A conflict with two distinct sources turns the answer into "Source A says X; Source B says Y".

Design as planned:

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

**As built (`app/llm/`):** providers sit behind one neutral `chat()` interface. `FallbackLLM` retries
429/5xx/timeouts after the delay the provider asks for (free tiers say "retry in 46s") when it is ≤ 60 s,
otherwise moves on to the next provider. **Measured free-tier limits (2026-10-24):** `gemini-3.5-flash`
allows **20 requests/day** (5/min) — too few for a 50-question eval at ~4–6 calls per question — so
`gemini-3.5-flash-lite` investigates, Flash is a second (separate) Gemini quota, and Groq
`gpt-oss-120b` (1,000 requests/day, 8k tokens/min) is last. Gemini must be sent back its own thought
signatures, so a question that fell back to another model stays there. Groq's occasional
`tool_use_failed` (gpt-oss calling a non-existent tool) is retried. An LLM outage makes `/ask` return 503
with `status: "error"`.

---

## 6. Caching (bonus A1)

**As built (`app/cache.py`, SQLite):** query embeddings, search results and LLM responses (investigator
and verifier). Search keys include a **corpus generation** that every add/update/delete bumps, so a cached
result can never outlive the corpus it came from (verified live: after deleting a document, the same
question was refused instead of answered from cache). LLM hits replay Gemini's native turn exactly.

- **Embedding cache** — keyed by `sha256(text)`; avoids re-embedding identical queries/chunks.
- **Retrieval cache** — keyed by `hash(query + config)`; returns prior ranked results.
- **Answer/judge cache** — keyed by `hash(question + config)`; crucial during eval, where the same
  questions run across configs.

> **Why it matters (defense):** caching is both a **performance** win and the **main defense against
> Gemini rate limits** during the many-call eval (§8).

---

## 7. Guardrails: prompt injection, jailbreaks and harmful output (bonus A1)

`app/agent/guard.py`, policies in `app/agent/prompts.py`. Three layers around the agent, each recorded in
`AgentAnswer.gates["guard"]` and shown in the UI's trace panel. Measured in `EVALUATION-REPORT.md` §8
(`results/guardrails.json`).

```
question ─▶ input guard ─▶ agent loop + gates (§5) ─▶ output guard ─▶ user
   blocked: no search,       passages scanned:         checks every word the
   no LLM call               flagged ones get a        model wrote that the
                             warning, still usable     user will read
```

| Layer | What it checks | How | On a hit |
|---|---|---|---|
| **Input guard** | prompt injection, jailbreak / role-play, prompt extraction, harmful requests, self-harm; questions over 1,000 chars | **gpt-oss-safeguard-20b** (Groq) reads the question against `INPUT_POLICY`, a written policy that also lists what is *safe* (dark history, titles containing "ignore") | status **`blocked`**, fixed reply (self-harm gets a helpline pointer); the search and the investigator never run |
| **Passage scan** | planted instructions in retrieved documents (indirect injection) | 9 plain regex patterns that need wording addressed to a model ("ignore all previous instructions", "note to the assistant", "developer mode") | the passage reaches the model with a `warning` field saying it is data; the UI shows ⚠ on its source card |
| **Output guard** | a leaked system prompt; hateful, threatening or dangerous text; signs the model was hijacked | 10 consecutive words of any prompt in the reply = leak (plain code), then the same classifier with `OUTPUT_POLICY` | answer or conflict → **`blocked`**, citations dropped; a refusal only loses its explanation |

**Why these models.** Groq's free tier offers two safety models (measured 2026-10-27). **Llama Prompt
Guard 2** is fast but only detects injections: it missed a DAN jailbreak asking for racist content and
flagged the harmless question about a film titled *Ignore All Instructions* (score 0.9995). **gpt-oss-
safeguard** follows a written policy, so it handled both, but allows only **2,000 tokens/min**: a
check costs ~450 tokens, so about four checks a minute. Hence: the policy model checks each question and
each answer once; Prompt Guard (30 requests/min) is the fallback for questions; passages, 20–40 per
question, are scanned with plain patterns, which cost nothing. On the 50k-article corpus the patterns flag
**1 article** ("…told him to ignore any rules they had set…"); a flag only adds a warning.

**When the classifier is unreachable** (rate limit, outage), the guard does not fail silently:

- input: falls back to Prompt Guard, then the patterns (injection only; harmful requests still meet
  the grounding gates and the output guard);
- output: grounded text (an answer the verifier accepted, a conflict with two cited sides) is shown; a
  refusal's explanation, which no gate has checked, is dropped. Either way `gates.guard.*.error` says so.

**The hole this closed.** Before the guardrails, the verifier only checked `answered` drafts. A model that
returned `insufficient` had its free-text explanation appended to the refusal **unchecked**, so a
jailbreak could put its content there. The output guard now reads every model-written sentence.

**What it does not change.** The investigator and verifier prompts are byte-identical, so normal
questions produce the same LLM calls (and cache hits) as in the evaluated system; the guards only add a
check before and after. Passages travel as JSON strings in tool results, so a document cannot fake the
end of its own passage or a new source. The public demo forces `guard: true`, whatever `params` a client
sends (`app/api/main.py`).

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
- **Offline bulk embedding** — 50k articles are embedded locally / on free Colab GPU, not on the host CPU
  (which would be far too slow), then pushed to Qdrant (`SETUP.md`).

---

## 9. API surface (FastAPI)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/documents` | Upload + ingest a file (multipart). Body: file. |
| `GET`  | `/documents` | List documents (paginated, optional `format` / `exclude_format` filter) from `documents`. |
| `GET`  | `/documents/{doc_id}` | Fetch one document + its chunks. |
| `PUT`  | `/documents/{doc_id}` | Re-ingest / update (version bump). |
| `DELETE` | `/documents/{doc_id}` | Delete document + its chunks. |
| `POST` | `/search` | Raw retrieval (ranked chunks). Powers the LLM-free retrieval eval + debugging. |
| `POST` | `/ask` | Full agentic or single-shot RAG (`agentic: true/false`). Returns answer, citations, searched queries, gates, trace, usage. |
| `POST` | `/ask/stream` | SSE stream emitting live progress events (`searching`, `found`, `verifying`, …) then the final `/ask` answer. |
| `GET`  | `/eval/results` | Serves tracked JSON evaluation artifacts from `results/` for the UI dashboard. |
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
| Relevance floor | cross-encoder logit **3.92** (calibrated) | Gate for "insufficient" — NOT on RRF. |
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
| Guardrails: prompt injection, jailbreaks, harmful output (bonus) | §7 |
| Python backend | FastAPI, this whole doc |
