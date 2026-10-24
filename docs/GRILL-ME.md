# GRILL-ME — The Decision Log & Defense Sheet

> **Purpose.** This is the single source of truth for *every* important decision in this project.
> If you (or any agent) ever need to know "what did we choose and why", it is here — you should
> never have to search the code to reconstruct a decision.
>
> **How to use it for interview prep.** Hand this file to an agent and say *"read GRILL-ME.md and
> grill me on it"* (this pairs with the `grilling` skill). The agent will interview you decision by
> decision until you can defend each one cold, with no AI in the room.
>
> **Format of each decision:**
> - **Choice** — what we picked.
> - **Why** — the reasoning in plain language.
> - **Alternatives rejected** — and the specific reason each lost.
> - **Defend it** — the one-paragraph answer you give when challenged.
> - **Likely attacks** — the follow-up questions an interviewer will push on, with answers.

---

## 0. What we are building (the one-paragraph version)

An **AI investigation engine** over a large document set (≥50,000 records). Not "chat with a PDF."
It answers complex questions that require **cross-referencing multiple sources, comparing conflicting
information, and drawing reasoned conclusions — always with citations**. When the evidence is missing
or self-contradictory, it **refuses and says so** instead of hallucinating. It ships with a document
management + Q&A UI, a custom automated evaluation suite, and is deployed live on a free host.

**The canonical example question** (from the spec): *"Which system changes caused the March
payment-processing outage, which services broke, and which documents confirm it?"* — one question that
forces multi-hop retrieval, conflict handling, and citation.

---

## 1. The five locked top-level decisions

These were decided first because everything else hangs off them.

### D1 — Dataset / corpus: **HotpotQA**

- **Choice.** Use **HotpotQA labels over full Wikipedia articles** as the ≥50k-document corpus and the
  ground-truth evaluation backbone. A *document* = a **full article** (not HotpotQA's opening paragraph
  only); the labeled 2017 opening paragraph is swapped back in so gold sentences stay valid (`DATASET.md`).
- **Why.** It ships **multi-hop questions with supporting-fact sentence labels**, so Recall@K, MRR, and
  Citation Accuracy are computed *automatically* against ground truth. Full articles give ≥50k genuinely
  **long** documents, which makes the chunking comparison real (short intros wouldn't). It covers the
  hardest required type — **multi-hop** — out of the box.
- **Alternatives rejected.**
  - *2WikiMultihopQA* — very similar, arguably cleaner evidence annotations. A fine fallback; HotpotQA
    is more widely known so it's easier to defend and compare against published numbers.
  - *danluu/post-mortems* — thematically perfect (it's literally outages, matching the example), but
    only a few hundred documents. We'd have to synthetically inflate it to 50k **and** hand-write all
    40+ eval questions. Too much manual eval work, and synthetic inflation is hard to defend.
  - *RAGBench / synthetic* — more setup, weaker provenance for the metrics.
- **Defend it.** *"I needed two things at once: ≥50k documents AND an eval set covering
  simple / multi-hop / unanswerable / contradictory with ground truth. HotpotQA gives me supporting-fact
  labels, so my retrieval metrics are computed against a real gold standard rather than my own opinion.
  I generate the unanswerable and contradictory cases myself on top of it."*
- **Likely attacks.**
  - *"HotpotQA is Wikipedia, not outage reports like your example."* → The example is illustrative; the
    pipeline is domain-agnostic. I can point the same ingestion + retrieval + reasoning stack at a
    post-mortems corpus and demo it. Nothing in the architecture assumes Wikipedia.
  - *"Where do unanswerable and contradictory questions come from if the dataset only has answerable
    multi-hop ones?"* → For **unanswerable** I remove a question's gold article(s) **and verify the answer
    string is absent** from the rest of the corpus (removal alone isn't enough). For **contradictory** I
    **inject a short standalone conflicting document** (not a modified copy), so the two sources stay
    genuinely distinct. Documented in `EVALUATION.md`.
  - *"Wikipedia intros are short, so your chunking comparison is trivial."* → Fixed by making a document a
    **full article** with a **256-token chunk cap**, so the strategies produce genuinely different chunks
    (measured: 9.8 vs 7.0 chunks/article), compared on Recall@K/MRR at 50k (`DATASET.md`, `EVALUATION.md`).

### D2 — Vector database: **Qdrant Cloud**

- **Choice.** **Qdrant Cloud** (free tier) as the vector store.
- **Why.** Free tier; **persistent** and external to the compute host (so it survives Space restarts);
  **native support for both dense and sparse vectors** (the foundation of hybrid search); rich
  **payload filtering** (needed for metadata-scoped search and precise delete/update); clean Python client.
- **Alternatives rejected.**
  - *pgvector (Supabase/Neon)* — great and would remove one service, but hybrid is bolted on via SQL
    full-text; Qdrant's native sparse vectors + HNSW tuning are purpose-built. Viable fallback.
  - *Weaviate Cloud* — fine, built-in hybrid, but heavier model; Qdrant is simpler to reason about.
  - *Chroma* — simplest locally, weakest story for hosted + persistent.
- **Defend it.** *"Vectors live in Qdrant Cloud so persistence is independent of my free compute host
  restarting. Qdrant stores dense and sparse vectors natively, which is exactly what my hybrid search
  needs, and its payload filters let me do metadata-scoped retrieval and surgical add/update/delete."*
- **Likely attacks.**
  - *"Why not pgvector to cut a service?"* → Acceptable tradeoff: one extra managed service in exchange
    for native sparse vectors, payload indexing, and HNSW controls I'd otherwise re-implement in SQL.
  - *"What's your free-tier size limit and what happens at 50k+ vectors?"* → Documented in `DATABASE.md`;
    bge embeddings are modest-dimensional, well within the free tier for this corpus.

### D3 — Models: **Gemini (LLM) + local BAAI `bge` embeddings + `bge-reranker`**

- **Choice.** **Google Gemini** free tier for generation and tool-calling; **local BAAI `bge`** model for
  embeddings; **local `bge-reranker`** as a second-stage re-ranker.
- **Why.** Gemini's free tier is generous, reasons well, and supports **tool/function calling** (a hard
  requirement). Local `bge` embeddings are free, strong, and run fine on the host we chose. The local
  reranker is both a quality boost and satisfies the **advanced re-ranking bonus**.
- **Alternatives rejected.**
  - *Groq (Llama/Qwen)* — very fast and free; a strong fallback LLM if Gemini limits bite.
  - *Hosted embeddings (Jina/Voyage/Gemini-embeddings)* — only needed if the host lacks RAM; ours doesn't.
  - *OpenAI* — best quality but **not free**; disqualified by the free-tier requirement.
- **Defend it.** *"Everything on the critical path is free. Gemini handles reasoning and tool-calling;
  embeddings and re-ranking run locally with `bge` models, which also gives me a real cross-encoder
  re-ranking stage to defend as a quality component."*
- **Likely attacks.**
  - *"Local models on a free host — cold starts and RAM?"* → Models are loaded once at startup and kept
    warm; the chosen host has enough RAM (see D4). Embedding model size is deliberately modest.
  - *"Why `bge` specifically, and what are its dimensions / similarity metric?"* → Documented in
    `BACKEND.md` (model variant, embedding dimensionality, normalization, cosine similarity).
  - *"What's your fallback if Gemini rate-limits mid-demo?"* → Provider is abstracted; Groq is the drop-in
    fallback.

### D4 — Hosting: **Hugging Face Spaces**

- **Choice.** Deploy on **Hugging Face Spaces** (free tier); vectors live in Qdrant Cloud.
- **Why.** Free; enough RAM to run the local embedding + reranker models; native support for the Streamlit
  UI; first-class **secret management** (repo secrets, never committed). The persistence that matters
  (the vector index) is offloaded to Qdrant Cloud, so Space restarts don't lose data.
- **Alternatives rejected.**
  - *Fly.io / Oracle Cloud Free VM* — always-on and powerful, but more manual infra setup.
  - *Render free* — 512MB RAM + spin-down forces hosted embeddings only; conflicts with D3.
  - *Vercel* — excellent for a JS frontend, but Python RAG becomes serverless functions with cold starts
    and no heavy local models.
  - *Streamlit Community Cloud* (`share.streamlit.io`) — the easiest possible Streamlit deploy (point it at
    a GitHub repo and it runs `streamlit run app.py`), but **rejected for this stack**: (1) ~1 GB RAM and
    limited CPU, which our **local `bge` embedding + `bge-reranker` models** (D3) would exhaust on their own;
    (2) it's designed to run **one Streamlit process**, not our two-process **FastAPI + Streamlit** topology
    (D5); (3) apps sleep on inactivity. HF Spaces gives the same "push-repo-and-it-runs" ease for Streamlit
    *plus* ~16 GB RAM, so it can carry the backend and local models too. Choosing Community Cloud would have
    forced switching D3 to a **hosted embedding API** and collapsing to a single process — a worse system.
- **Defend it.** *"Spaces gives me free compute with enough RAM for my local models and clean secret
  handling. I deliberately keep durable state (vectors, document registry) off the ephemeral Space so a
  restart never loses data. I didn't use Streamlit Community Cloud because its ~1 GB limit can't hold my
  local embedding + reranker models or my separate FastAPI process."*
- **Likely attacks.**
  - *"Spaces sleep and the disk is ephemeral — isn't your data gone?"* → Vectors are in Qdrant Cloud; the
    document registry's persistence boundary is documented in `DATABASE.md`. The Space is stateless compute.
  - *"Why not just host on Streamlit, it's a Streamlit app?"* → The UI is Streamlit, but the system also
    runs a FastAPI backend and loads local ML models; Streamlit Community Cloud's ~1 GB can't fit that, so
    the UI framework and the host are decided separately.

### D5 — Application stack: **FastAPI backend + Streamlit UI**

- **Choice.** A **FastAPI** backend exposing the RAG/ingestion API, with a **Streamlit** UI for document
  management and Q&A with source viewing.
- **Why.** Clean **API/UI separation** — the backend is independently testable and the eval suite calls
  the same API the UI does, so I'm evaluating the real system. Streamlit builds the required functional UI
  (upload/manage docs, ask questions, view sources) quickly.
- **Alternatives rejected.**
  - *Single Gradio app* — fastest to build, but blurs API/UI separation and is harder to test headlessly.
  - *FastAPI + Next.js/React* — most polished, most work; not worth it for an evaluation task where the
    engineering story matters more than pixel polish.
- **Defend it.** *"The eval harness and the UI both hit the same FastAPI endpoints, so my evaluation
  measures exactly what a user experiences. The separation also makes the pipeline unit-testable."*
- **Likely attacks.**
  - *"Two processes on one Space — how does that actually run?"* → Deployment topology (how FastAPI and
    Streamlit coexist on the Space) is documented in `BACKEND.md` / `FRONTEND.md`.

---

## 2. Round-2 decisions (locked)

These shape the module docs (`BACKEND.md`, `DATABASE.md`, `FRONTEND.md`).

### D6 — Document registry: **Qdrant-only, dedicated `documents` collection**

- **Choice.** No separate relational DB. Two Qdrant collections: `chunks` (vectors + payload) and
  `documents` (one point per document = the registry). Linked by a shared `doc_id`.
- **Why.** The free HF Space has an **ephemeral disk**; Qdrant Cloud is **durable**. Keeping everything in
  Qdrant means one durable, hosted store with no persistence mismatch and no local DB to lose on restart.
- **How the two stores connect.** They don't wire together — the **backend** coordinates them via the
  shared `doc_id`/`chunk_id`. Every chunk's payload repeats its `doc_id`, which is how a citation jumps to
  its source document.
- **Alternatives rejected.** SQLite-on-Space (ephemeral → would vanish on rebuild); SQLite-as-rebuildable-
  cache (extra sync code); hosted Postgres (extra service + secret to manage).
- **Defend it.** *"All durable state is in Qdrant Cloud; the compute host holds nothing it can't lose. A
  `documents` collection gives me document-level lifecycle, and `doc_id` on every chunk links the two —
  no second database to desync or lose."*
- **Likely attacks.** *"Qdrant isn't relational — how do you list/dedup?"* → scroll API + payload indexes
  on `doc_id`, `sha256`, `version`; dedup = `sha256` filter before insert.

### D7 — Hybrid keyword search: **Qdrant native sparse vectors**

- **Choice.** Each chunk stores a `dense` (bge) **and** a `sparse` (BM25-style) vector in Qdrant; both run,
  results fused by our RRF.
- **Why.** One persistent store, native hybrid, nothing to rebuild on restart.
- **Alternatives rejected.** In-memory BM25 (`rank_bm25`) — not persistent, rebuilt every boot; Postgres
  full-text — only if the registry were Postgres.
- **Defend it.** *"The keyword signal is a sparse vector living next to the dense vector in the same store,
  so hybrid search is one system with no separate inverted index to host or sync."*

### D8 — Custom ranking component: **BOTH — custom weighted RRF + custom feature re-ranker** (+ bge-reranker bonus)

- **Choice.** Two hand-built ranking stages: **(1)** custom weighted Reciprocal Rank Fusion merging the
  dense + sparse lists, then **(2)** a custom feature-based re-ranker whose signals are **title match,
  query-term/entity overlap, exact-phrase match, answer-type hint, length normalization, and a fusion
  prior** (weights tuned on a separate tuning set, never on the eval set); final top-k capped at 2 chunks
  per article. A **(3)**
  off-the-shelf `bge-reranker` cross-encoder refines the top-N as a *bonus*.
- **Why.** The spec bans a 100%-off-the-shelf search stack. Two custom stages overshoot the bar and give a
  strong interview story; the cross-encoder adds the advanced-reranking bonus.
- **Defend it.** *"RRF is scale-free rank fusion I wrote and can tune; the feature re-ranker adds signals a
  cross-encoder can't see (title match, entity/phrase overlap, answer-type) and is fully explainable; the
  bge cross-encoder is an optional precision pass on the short list."* (Detail: `BACKEND.md` §4.)
- **Likely attacks.** *"Why RRF over summing scores?"* → dense cosine and sparse BM25 are different scales;
  RRF uses ranks. *"Is the feature re-ranker redundant with bge?"* → no — cheaper, runs on all candidates,
  sees signals bge can't. *"Why not recency / source authority?"* → the corpus is single-source and
  undated, so they carry no signal — and recency would even **promote the injected contradictions**
  (ingested last). That's why I chose data-grounded signals instead.

### D9 — Chunking strategies to compare: **fixed-token-w/-overlap vs. recursive/structural**

- **Choice.** Compare (A) fixed 256-token/32-overlap windows vs. (B) recursive/structural splitting on
  headings→paragraphs→sentences to a 256-token cap. Semantic chunking = bonus/future.
- **Why.** Cleanest, most defensible contrast; B is expected to win on Recall@K/MRR. On **full articles**
  (revised D1) the 256-token cap makes the two strategies genuinely diverge. Compared at full 50k scale in
  local Qdrant; only the winner goes to Qdrant Cloud (disk limit).
- **Defend it.** *"Blind windows vs. structure-aware splitting is a crisp A/B; on full articles they
  produce clean vs. blended chunks, and I compare them on Recall@K/MRR at full scale."*
- **Attack.** *"Your chunk-level gold-sentence metric — does it separate them?"* → No, and I measured
  why: HotpotQA's gold sentences sit in the lead at offset 0, inside the first fixed window, so fixed cuts
  only 0.2% of them. I report that as a finding instead of leaning on a metric that can't discriminate.

### D10 — Multi-step reasoning: **agentic tool-calling loop**

- **Choice.** Gemini is given a `search_documents` tool (and `get_document`) and drives its own
  plan→act→observe→decide loop, bounded by a max-iteration cap and tool-call budget.
- **Why.** Satisfies tool-calling + multi-step reasoning + the agentic-RAG bonus, and matches the
  "investigation engine" framing.
- **Defend it.** *"The model decides what to look up next based on what it just learned — that's real
  multi-hop investigation, not a fixed pipeline. Iteration caps and a tool-call budget keep it bounded and
  traceable."* (Detail + the grounding/refusal check: `BACKEND.md` §5.)
- **Likely attacks.** *"Non-deterministic / infinite loops?"* → hard iteration cap, tool-call budget,
  timeouts. *"Trace it live?"* → every tool call + decision is logged and shown in the UI trace view.

### D11 — Frontend scope: **three Streamlit pages**

- **Choice.** (1) Documents (upload/list/update/delete), (2) Ask (streamed answer, inline citations +
  source viewer, refusal/contradiction display, trace view), (3) Evaluation dashboard (metrics + config
  comparison + failure modes).
- **Why.** Covers all required UI and makes the evaluation deliverable visual. (Detail: `FRONTEND.md`.)
- **Defend it.** *"The UI is a thin client over the same API the eval suite uses, so the demo is exactly
  what I measure."*

---

## 2c. Evaluation decisions (locked)

These drive `EVALUATION.md`, the basis of the Evaluation Report deliverable.

### E1 — Answer Correctness metric: **LLM-as-judge (primary) + token-F1/EM (secondary)**
- **Why.** HotpotQA gold answers are short spans, but our engine outputs cited prose. A judge model
  scores semantic correctness (handles reworded-but-correct answers); F1/EM is a cheap, reproducible
  cross-check that catches a drifting judge.
- **Defend it.** *"EM alone would fail a correct answer just for rephrasing, so I use a rubric-based judge
  as primary and keep F1/EM as a reproducible sanity check. Judge runs at temperature 0 with a fixed
  rubric to limit variance."*
- **Attack.** *"LLM judges are biased/unstable."* → temp-0 + rubric + F1/EM cross-check; disagreements are
  surfaced in failure analysis.

### E2 — Citation Accuracy: **precision / recall / F1 of cited docs vs. gold supporting docs**
- **Why.** HotpotQA ships gold supporting facts, so we can measure both **over-citing** (precision) and
  **missing evidence** (recall) against ground truth.
- **Defend it.** *"Binary 'cited something right' hides over- and under-citation; P/R/F1 against the gold
  supporting docs shows exactly how precise my citations are."*

### E3 — Test set: **~50 questions (15 simple / 15 multi-hop / 10 unanswerable / 10 contradictory)**
- **Why.** Comfortably clears the 40 minimum with balanced coverage of all four required types and a margin.

### E4 — Unanswerable & contradictory generation: **hold-out (+ absence check) + standalone injection**
- **Choice.** Unanswerable = take a **bridge question with a distinctive named-entity answer** (no yes/no,
  no common strings), remove its gold article(s) **and verify the gold answer string / entities are
  absent** from the rest of the corpus; questions that shared those articles are dropped from the other
  eval sets. Contradictory = **inject a short standalone conflicting
  document** (not a modified article copy).
- **Why.** Reuses real ground truth; the absence check closes the "answer might still be elsewhere" gap; a
  standalone doc stays distinct (survives exact-dedup) and avoids the near-dup-collapse trap.
- **Attack.** *"How do you know the unanswerable ones are truly unanswerable?"* → I remove the gold articles
  **and** confirm the answer string no longer appears anywhere in the index — which is only meaningful
  because I restrict to distinctive entity answers (a "yes" or "1990" appears everywhere).

### E5 — Configurations compared: **two axes — chunking A-vs-B AND hybrid-vs-dense-only**
- **Why.** Each comparison changes one variable: one isolates the chunking decision (D9), the other tests
  whether hybrid + RRF fusion (D7/D8) actually beats plain vector search. Meets the ≥2-config requirement.

---

## 2d. Build / implementation decisions (locked)

### A1 — Bonus scope: **caching + prompt-injection tests + lightweight monitoring** (multi-LLM = stretch)
- **In scope:** query/embedding **caching**, **prompt-injection tests** in the eval, **lightweight
  monitoring/logging**. (Advanced re-ranking is already in via `bge-reranker`; **streaming** already planned.)
- **Stretch:** multi-LLM/embedding comparison. **Dropped:** multi-agent, load testing.
- **Why.** Each chosen bonus earns spec credit *and* serves a core need: caching also shields us from Gemini
  rate limits; prompt-injection tests strengthen both eval and the security story; monitoring powers the UI
  trace view. The dropped items are high-effort / low-payoff for an evaluation task.
- **Defend it.** *"I picked bonuses that double as risk mitigations or reuse existing machinery, not
  features for their own sake."*

### A2 — Model pinning
- **LLM:** **Gemini 2.x Flash** primary, **Groq Llama** fallback (behind a provider abstraction).
  - *Why Flash not Pro:* the agentic eval makes many LLM calls; Pro's tighter free limits risk throttling.
    Flash is fast and sufficient; the abstraction lets me swap if a question needs more.
- **Embeddings:** **`bge-small-en-v1.5`** (384-dim).
  - *Why small:* RAM-safe on a free Space and fast; competitive quality for its size, and the cross-encoder
    reranker recovers precision. `bge-base` (768-dim) is a documented upgrade lever.
- **Defaults (accepted as a bundle):** reranker **`bge-reranker-base`**; sparse/keyword vectors via
  **BM25 (FastEmbed)**; chunk token counting via the **`bge` tokenizer** (not tiktoken — counting with
  the embedding model's own tokenizer means the 256-token cap is exactly what the model sees).
- **Attack.** *"small embeddings = weak retrieval?"* → hybrid + two rerank stages compensate; RAM safety on a
  free host is the deliberate trade-off, with `bge-base` ready if metrics demand it.

### A3 — Dataset ingestion: **full Wikipedia articles (HotpotQA-labeled) → ~50k, ingested offline**
- **Choice.** A document = a **full Wikipedia article** (`wikimedia/wikipedia 20231101.en`) with the
  HotpotQA 2017 paragraph replacing its **whole lead section**; fix the held-out (unanswerable) articles
  first, then take all gold+distractor articles of the remaining eval questions, then fill to **~50k**
  with a fixed seed; **embed offline** (local/Colab GPU), int8-quantized, then push to Qdrant. Full plan in
  `DATASET.md`.
- **Why.** Full articles give long documents so chunking is meaningful; offline embedding avoids the slow
  Space CPU; the seeded manifest makes the count reproducible.
- **Attack.** *"Labels were on 2017 Wikipedia — still valid?"* → yes; I swap the exact 2017 opening
  paragraph (which holds the gold sentences) back into the modern article; unresolved titles are dropped
  and logged in the manifest.
- **Attack.** *"Doesn't the modern article contradict your 2017 paragraph?"* → that's why I replace the
  **whole lead section**, not just its first paragraph — modern leads restate updated facts, which would
  plant unintended contradictions. Rarer conflicts deeper in the body are traceable via the manifest.

### A4 — Packaging: **Docker Space, one entrypoint running FastAPI + Streamlit**
- **Why.** Full control over the two-process topology, pinned dependencies, reproducible builds; preserves
  the API/UI separation (D5). The entrypoint starts uvicorn (FastAPI) then Streamlit; Streamlit calls the
  API on localhost.
- **Attack.** *"Two processes in one container — how?"* → documented entrypoint + ports; test the topology
  early rather than at deploy time.

### A5 — Secrets & config: **`pydantic-settings` + `.env` locally, Space secrets in deploy**
- **Why.** Satisfies "secrets managed properly": no keys in git; a committed `.env.example` documents the
  variables; the Space injects the same names as secrets. (Detail: `SETUP.md` §2.)

### A6 — AI-usage logging: **maintain `AI-USAGE.md` as a living log**
- **Why.** The AI Usage Report is a required deliverable; logging each session as we build keeps it accurate
  instead of reconstructed later. (Detail: `AI-USAGE.md`.)

### A7 — Testing: **small pytest suite** (ingestion, ranking, refusal)
- **Why.** Focused tests for the ingestion path, the custom ranking (RRF + feature reranker), and the
  refusal/contradiction logic — cheap insurance and directly supports the "fix a bug live" interview moment.

### A8 — CI/CD: **stretch** (manual deploys until then)
- **Why.** A GitHub Action to auto-deploy to the Space is nice-to-have; core system comes first.

---

## 3. Decision index (quick lookup)

| ID  | Decision                         | Choice (— = open)                              | Doc |
|-----|----------------------------------|------------------------------------------------|-----|
| D1  | Dataset / corpus                 | HotpotQA labels over full Wikipedia articles   | this / DATASET.md |
| D2  | Vector database                  | Qdrant Cloud                                   | DATABASE.md |
| D3  | LLM + embeddings + reranker      | Gemini + local bge + bge-reranker              | BACKEND.md |
| D4  | Hosting                          | Hugging Face Spaces                            | this |
| D5  | App stack                        | FastAPI + Streamlit                            | BACKEND / FRONTEND |
| D6  | Document registry store          | Qdrant-only + `documents` collection           | DATABASE.md |
| D7  | Hybrid keyword mechanism         | Qdrant native sparse vectors                   | DATABASE / BACKEND |
| D8  | Custom ranking component         | Custom RRF **+** custom feature re-ranker (+ bge bonus) | BACKEND.md |
| D9  | Chunking strategies (compare ≥2) | fixed vs. recursive (256-token cap)            | BACKEND.md |
| D10 | Multi-step reasoning pattern     | Agentic tool-calling loop                      | BACKEND.md |
| D11 | Frontend page scope              | Three Streamlit pages                          | FRONTEND.md |
| E1  | Answer Correctness metric        | LLM-as-judge + token-F1/EM                     | EVALUATION.md |
| E2  | Citation Accuracy                | Precision/Recall/F1 vs. gold supporting docs   | EVALUATION.md |
| E3  | Test set size & split            | ~50: 15 simple / 15 multi-hop / 10 unans / 10 contra | EVALUATION.md |
| E4  | Unanswerable/contradictory gen   | Hold-out + injection                           | EVALUATION.md |
| E5  | Configs compared (≥2)            | Chunking A/B + hybrid vs. dense-only           | EVALUATION.md |
| A1  | Bonus scope                      | Caching + prompt-injection + monitoring (multi-LLM stretch) | BACKEND / EVALUATION |
| A2  | Model pinning                    | Gemini Flash (+Groq), bge-small, bge-reranker-base, BM25/FastEmbed, bge tokenizer | BACKEND.md |
| A3  | Dataset ingestion                | Full Wiki articles (HotpotQA-labeled) → ~50k, offline | DATASET.md |
| A4  | Packaging on HF Spaces           | Docker Space, dual-process entrypoint          | SETUP.md |
| A5  | Secrets & config                 | pydantic-settings + .env / Space secrets       | SETUP.md |
| A6  | AI-usage logging                 | Living AI-USAGE.md log                          | AI-USAGE.md |
| A7  | Testing                          | Small pytest suite                             | repo |
| A8  | CI/CD                            | Stretch (manual deploy for now)                | repo |

_Last updated: round 5 — **design-review pass** applied: dataset → full Wikipedia articles (D1/A3),
feature-reranker signals de-recency'd (D8), exact-dedup only (D6/DATABASE), concrete grounding +
contradiction detection (BACKEND §5.4), offline ingestion + cost/perf (BACKEND §8, SETUP), `uuid5` point
IDs, `SETUP.md` added, A5/A6 filled. All **24** decisions locked (D1–D11, E1–E5, A1–A8) — now contiguous.
Round 6 — second review pass: whole-lead swap (A3), unanswerable eligibility + selection order (E4), bge
tokenizer (A2), Qdrant on-disk layout + free-cluster deletion (DATABASE §7–§8), `app_port` (SETUP)._
