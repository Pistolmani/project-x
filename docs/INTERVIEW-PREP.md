# INTERVIEW PREP — Drill Questions & Model Answers

> Practice doc for defending the project with **no AI in the room**. Each item: **Q** → **model answer**
> (key defensible phrases in **bold**) → **follow-up attack** the interviewer will push next.
>
> **How to drill:** cover the answer, say it out loud in your own words, then check. You don't need the
> exact wording — you need the **reasoning**. Decisions reference [`GRILL-ME.md`](./GRILL-ME.md).

---

## A. Architecture & pipeline

**Q1 — Trace the pipeline end to end (UI → answer).**
UI (**thin client**) → `POST /ask` on FastAPI → **agentic loop (Gemini Flash)** with tools
`search_documents`/`get_document` → the model **plans sub-questions (multi-hop)** → each tool call runs the
**retrieval pipeline**: embed query **dense (bge-small) + sparse (BM25)** → Qdrant returns **two ranked
lists** → **custom weighted RRF** → **custom feature re-ranker** → **bge-reranker (top-N)** → top-k chunks
→ agent loops if needed (**bounded by iteration cap + tool budget**) → **grounding check** (refuse /
surface conflict / synthesize) → **answer + inline citations** streamed back to Streamlit (clickable
sources + trace view).
→ *Follow-up:* "Where's the multi-hop?" → The agent issues a **second query informed by the first's
results**, not a fixed two-shot.

**Q2 — Why FastAPI + Streamlit as two pieces, not one app?**
**Clean API/UI separation**: the UI holds no logic, and **my eval suite hits the same `/ask` and `/search`
endpoints the UI does**, so I measure the real system. Streamlit builds the required functional UI fast.
→ *Follow-up:* "Two processes on one Space?" → A **Docker Space** with one entrypoint starts uvicorn then
Streamlit; Streamlit calls the API on **localhost**.

---

## B. Document processing

**Q3 — You compare two chunking strategies. Which, and what do you expect?**
**Fixed-token-with-overlap (256/32)** vs **recursive/structural** (split on headings→paragraphs→sentences
to a 256-token cap). This is a *real* comparison because **a document is a full Wikipedia article**, not a
short intro — recursive gives each section **clean chunks**, fixed windows **blend** sections (measured:
9.8 vs 7.0 chunks per article). I expect recursive to win on **Recall@K / MRR**.
→ *Follow-up:* "How do you measure which wins?" → the LLM-free retrieval eval at **full 50k scale** in
local Qdrant (a 2k sample was too easy — everything scored R@10 ≈ 0.9+).
→ *Follow-up:* "Does fixed cut your gold sentences?" → Almost never — **0.2%**, because HotpotQA's gold
sentences sit in the lead, inside the first window. I **measured that and report it** rather than claim a
metric that can't discriminate.

**Q4 — How do add / update / delete work without corrupting state?**
Everything keys on **`doc_id`**. Add = chunk/embed/upsert + write a `documents` row. Update = re-ingest,
**bump `version`**, **insert new chunks then delete old** (so a crash never leaves the doc unsearchable).
Delete = **payload-filtered delete by `doc_id`** in both collections.
→ *Follow-up:* "Dedup?" → **sha256** of normalized content in the payload; I filter on it before insert.

---

## C. Vector DB & search

**Q5 — Why Qdrant, and why two collections?**
Free tier, **persistent and external to my ephemeral host**, and **native dense + sparse vectors** for
hybrid. Two collections: **`chunks`** (searchable units) and **`documents`** (one point per doc = the
**registry**), linked by `doc_id`. *Why a registry collection:* list/paginate/version/dedup are
**document-level** ops I don't want to do by scanning every chunk.
→ *Follow-up:* "Qdrant isn't relational — how do you list/dedup?" → **scroll API + payload indexes** on
`doc_id`, `sha256`, `version`.

**Q6 — Defend RRF over summing scores.** (core custom component)
Dense **cosine** and **BM25** scores are on **different, unbounded scales** — summing lets one dominate
arbitrarily and needs fragile normalization. **RRF uses only rank position** — `Σ w/(k+rank)` — so it's
**scale-free, robust to outliers**, rewards docs ranked high by **either** retriever, and **k≈60 dampens**
low ranks. **Tunable weights** bias semantic vs lexical; I ablate them.
→ *Follow-up:* "Why also a feature re-ranker, and why those signals?" → It's **cheap, runs over all
candidates**, and uses signals the cross-encoder **can't see** — **title match, entity/phrase overlap,
answer-type**. I avoided **recency/source authority** on purpose: the corpus is single-source and undated,
so they're noise — and recency would even **promote the injected contradictions** (ingested last).

**Q7 — What makes your search "hybrid"?**
Each chunk stores **both** a dense (`bge`) and a sparse (BM25) vector **in the same store**; a query runs
both, and I **fuse with RRF**. Semantic catches meaning; sparse catches **exact keywords/entities**. No
separate inverted index to host or sync.

---

## D. LLM & RAG

**Q8 — Why an agentic loop instead of a fixed retrieve-then-generate pipeline?**
Because it's an **investigation** engine: the model **decides what to look up next based on what it just
learned**, which is real multi-hop. It also naturally satisfies **tool-calling + multi-step reasoning**.
I keep it safe with a **max-iteration cap and tool-call budget**.
→ *Follow-up:* "Non-deterministic / infinite loop risk?" → Hard **iteration cap, tool budget, timeouts**;
every step is **logged and shown in the trace view** so it's debuggable.

**Q9 — How does it avoid hallucinating?** (see also Q on eval)
A **grounding check before finalizing**, with concrete gates: (1) a **relevance floor on the cross-encoder
score** — *not* the RRF score, which is rank-only and always has a top value; (2) a **groundedness check**
that every claim maps to a cited chunk; else → **"not enough information."** Contradictions are caught by a
**dedicated consistency step** (an NLI cross-encoder, or a focused LLM consistency-judge call) that
surfaces the conflict instead of picking a side.
→ *Follow-up:* "Prove it works." → **Unanswerable set** (hold-out **+ answer-absence check**) measures
**refusal accuracy**; **contradictory set** (injected standalone doc) measures **conflict-detection
accuracy**.

**Q10 — How are citations produced and verified?**
Every claim cites the **`doc_id`/`chunk_id`** it came from; the UI turns them into click-to-view sources.
In eval, **Citation Accuracy = precision/recall/F1 of cited docs vs HotpotQA gold supporting docs**.
→ *Follow-up:* "What if it cites a doc it didn't use?" → Low **precision** flags over-citing; it's a
tracked failure bucket.

**Q11 — Gemini rate-limits mid-demo. What happens?**
Provider is **abstracted**; **Groq Llama is the drop-in fallback**, and **caching** absorbs repeat calls.
I chose **Flash over Pro** precisely because the many-call agentic eval would hit Pro's tighter limits.

---

## E. Evaluation

**Q12 — Walk me through your metrics.**
**Recall@K** = fraction of gold supporting docs in top-K; **MRR** = mean reciprocal rank of the first gold
chunk; **Answer Correctness** = **LLM-as-judge (primary) + token-F1/EM (secondary)**; **Citation
Accuracy** = **P/R/F1 vs gold supporting docs**; plus **refusal accuracy** and **conflict-detection
accuracy** for the behavioral sets.
→ *Follow-up:* "LLM judges are unreliable." → **Temperature 0 + fixed rubric + F1/EM cross-check**;
disagreements go into failure analysis.

**Q13 — You have no unanswerable/contradictory questions in HotpotQA. Where'd they come from?**
I **manipulated the corpus, not invented questions**: **unanswerable = take a bridge question with a
distinctive named-entity answer, remove its gold article(s) AND verify the answer string is absent** from
the rest of the corpus (removal alone isn't enough — long articles mention many facts; and I skip yes/no or
common answers like "1990", where an absence check proves nothing). Questions sharing a held-out article
are dropped from the other sets; **contradictory = inject a short standalone conflicting document** (not a
modified copy, so it stays distinct and survives dedup). Both tie back to **ground truth**.

**Q14 — What two configurations do you compare and why?**
**Chunking A vs B** and **hybrid vs dense-only** — each changes **one variable**: one tests my chunking
decision, the other tests whether **hybrid + fusion actually beats plain vector search**. Both are compared
on the **cheap LLM-free retrieval eval**; the expensive `/ask` eval runs on one config (cost control).

**Q14a — Your RRF and re-ranker weights are tuned. Didn't you just fit them to your test set?**
No — **weights are tuned on a separate ~100-question tuning set**, drawn from the same distribution but
**disjoint from the retrieval eval set** (enforced in `select.py` and tested). Reported Recall@K/MRR come
from questions the weights **never saw**.
→ *Follow-up:* "Why does the re-ranker include the RRF score?" → As a **fusion prior**, so it **refines**
the fused order instead of throwing it away; set the other weights to 0 and you get the RRF order back.

**Q14b — HotpotQA is a multi-hop dataset. Where do your "simple" questions come from?**
The dataset's **own** label: the authors split out a **`train-easy`** tier specifically for **mostly
single-hop** questions. So "simple" = `level == "easy"` (train split only — dev/test are all `hard`).
→ *Follow-up:* "Mostly?" → They still carry two supporting facts, but one paragraph usually suffices; I
report per-type metrics, so any leakage of harder questions is visible.

---

## F. Deployment

**Q15 — Your free host has an ephemeral disk. Isn't your data gone on restart?**
No — **all durable state is in Qdrant Cloud** (vectors **and** the `documents` registry). The Space is
**stateless compute**; a rebuild reconnects to the same collections. Secrets are **Space secrets**, never
committed.
→ *Follow-up:* "Why not host on Streamlit Cloud, it's a Streamlit app?" → Its **~1 GB RAM can't hold my
local embedding + reranker models or a separate FastAPI process**; UI framework and host are decided
separately. HF Spaces gives ~16 GB.

---

## G. Model choices

**Q16 — Small embeddings — isn't retrieval weak?**
`bge-small` (384-dim) is **competitive for its size**, **RAM-safe on a free host**, and **hybrid + two
rerank stages recover precision**. `bge-base` is a documented **upgrade lever** if metrics demand it.
→ *Follow-up:* "BM25 vs SPLADE?" → BM25 (FastEmbed) is **lighter and easier to defend**; SPLADE is a
learned-sparse upgrade path.

---

## H. Curveballs (they *will* ask you to do this live)

**"Fix this bug."** Narrate: reproduce → isolate the stage (ingest? retrieve? rank? reason?) → check the
**trace view / logs** → form a hypothesis → smallest fix → re-run the relevant **pytest**. The modular
pipeline + trace view is *why* you can localize fast.

**"Add this feature."** Map it to a stage: a new file format = new **loader**; a new ranking signal = add
to the **feature re-ranker**; a new tool = add to the **agent's tool list**. Say which seam it touches and
why the design makes it a localized change.

**"What would you improve with more time?"** Honest, pre-picked answers: semantic chunking as a third
strategy, the multi-LLM/embedding comparison (stretch), CI/CD auto-deploy, and calibrating the relevance
floor on a held-out set.

---

_Covers the 24 locked decisions (D1–D11, E1–E5, A1–A8). Next grilling round: you answer first, reveal
after._
