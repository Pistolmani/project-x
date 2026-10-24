# DATASET — Corpus Construction & Eval-Set Carving

> How we build the ≥50k-document corpus and the evaluation sets. Decision IDs refer to `GRILL-ME.md`.
> **A document = a full Wikipedia article** (D1, revised), labeled by HotpotQA.

---

## 1. What a "document" is (revised D1 — review fix)

HotpotQA's `context` paragraphs are just the **opening paragraph** of each Wikipedia article, so grouping
paragraphs by title gets nothing (one paragraph per title). To make chunking meaningful, **a document is
the full Wikipedia article**:

- **Base text:** the full article body from the Hugging Face dataset **`wikimedia/wikipedia`, config
  `20231101.en`** (plain text, keyed by title).
- **Label preservation:** HotpotQA was labeled on a **2017** Wikipedia dump and articles have since
  changed, so for each article we **replace its whole lead section (everything before the first section
  heading) with the exact HotpotQA 2017 paragraph**. That paragraph holds the labeled gold sentences, so
  **the gold supporting-fact labels stay valid**; the rest of the modern article is added **context**.
- **Why the whole lead, not just the first paragraph (review fix):** modern leads often run several
  paragraphs and restate facts that changed since 2017 (population, CEO, team, records). Keeping them next
  to the 2017 paragraph would plant **unintended contradictions** inside a single document, and the
  conflict detector would flag questions that are supposed to be plain. Replacing the entire lead removes
  the overlap. Facts that changed and also reappear deeper in the body are possible, but rarer; the
  manifest records every swap so any such case can be traced in failure analysis.
- **Unresolved titles:** titles not found in the `20231101.en` dump are **dropped and recorded in the
  manifest**.

> **Defense:** *"A real investigation corpus is long documents, not 100-token snippets. HotpotQA trimmed
> Wikipedia to opening paragraphs; I put the full articles back. The labeled opening paragraph is the exact
> original, so my gold sentences still match — and now chunking actually has something to work on."*

---

## 2. Why full articles improve the evaluation

With long documents the two strategies produce genuinely different chunks (measured on a 2k-article
sample: recursive ≈ 9.8 chunks/article averaging 150 tokens; fixed ≈ 7.0 chunks/article averaging 239):
- **Recursive** gives each section — including the lead — its own clean chunk(s).
- **Fixed** windows ignore structure, so a chunk often **blends** the end of one section with the start of
  the next, and can cut sentences anywhere in the body.

**What we measured, honestly (Phase 1):** HotpotQA's gold sentences all sit in the **lead**, at the very
start of the article — and the first fixed window also starts at character 0. At a 256-token cap the lead
fits in the first window in 96% of articles, so fixed windows cut only **0.2%** of gold sentences (20 of
8,022). So the **chunk-level gold-sentence metric cannot separate the strategies on this dataset**; the
comparison rests on **Recall@K / MRR at full 50k scale**, where clean-vs-blended chunks and chunk size
matter. The gold-sentence metric is still reported, with this explanation, as part of the analysis.

> *Dependency to verify in Phase 1:* whether `wikimedia/wikipedia` preserves **section headings**. If not,
> the recursive chunker still splits on paragraphs then sentences — the comparison holds either way.

---

## 3. Selecting the ~50k documents (A3, revised)

Order matters, because removing an article for one test case can break another (review fix):

1. **Pick the unanswerable set first** (§6) and fix its **held-out articles** — these are never ingested.
2. **Pick the other eval questions** (behavioral set + retrieval sample), **dropping any question whose
   gold articles include a held-out article** — otherwise it would silently become unanswerable too.
3. Take **every gold + distractor article** referenced by the remaining eval questions (so every eval case
   is fully supported).
4. **Fill up to ~50,000** with additional articles, chosen with a **fixed random seed**, never drawing a
   held-out article.
5. Write a **manifest** (`doc_id`, title, source, sha256, `lead_replaced` flag, held-out titles, injected
   docs, dropped questions) so the corpus is exactly reproducible and the count is defensible.

---

## 4. Ingestion pipeline (run OFFLINE — review fix)

Embedding 50k long articles on the Space's CPU would be far too slow, so bulk ingestion runs **locally or
on a free Colab GPU**, then pushes vectors to Qdrant Cloud; the Space only queries. See `SETUP.md`.

1. **Load** article bodies from `wikimedia/wikipedia 20231101.en` for the selected titles.
2. **Swap** the whole lead section for the HotpotQA 2017 paragraph (§1).
3. **Chunk** with the active strategy (D9), **256-token cap**, counted with the **`bge` tokenizer** (the
   same tokenizer the embedding model uses, so the cap matches what the model actually sees).
4. **Dedup** exact by `sha256` (`DATABASE.md` §4) — near-dups are **not** collapsed.
5. **Embed** dense (`bge-small`, int8-quantized) + sparse (BM25) and **upsert** to Qdrant.
6. **Idempotent:** re-runs skip anything already indexed.

---

## 5. Size & cost (revised — review fix)

Measured on a 2k-article sample (Phase 1), projected ×25 to 50k:
- Article length: mean **1,473** tokens, median 680, p95 6,015, max 20,586 → **~74M tokens** at 50k.
- Chunks: **~490k recursive**, **~350k fixed**.
- Indexing speed on an RTX 3060 Laptop GPU: ~40 s per 2k articles per strategy → **~17 min per
  strategy** at 50k.
- Both strategies together (~3.9 GB est.) don't reliably fit Qdrant Cloud's 4 GB disk, so: **the chunking
  comparison runs at full 50k scale in local Docker Qdrant** (no limits), and **only the winning strategy
  is pushed to Qdrant Cloud** for the live demo (`DATABASE.md` §7).

---

## 6. Carving the eval sets from this corpus (E3/E4)

- **Retrieval eval (hundreds, LLM-free):** seeded sample of HotpotQA questions whose gold articles are in
  the index.
- **Tuning (~100):** a second, disjoint sample from the same distribution. RRF and feature-re-ranker
  weights are tuned **only** here, so the retrieval eval set stays unseen (no tuning on the test set).
- **Simple (15):** HotpotQA **`level == "easy"`** questions — these exist **only in the train split**.
- **Multi-hop (15):** `level == "hard"` bridge questions (from the dev split, which is all `hard`).
- **Unanswerable (10):** eligible questions are **bridge questions whose gold answer is a distinctive named
  entity** — **no yes/no comparison questions and no short common answers** (e.g. "American", "1990"),
  because those strings appear somewhere in 50k articles and the absence check would prove nothing. Then
  remove the gold article(s) **and verify the gold answer string + key entities are absent from the
  remaining corpus** (long articles mention many facts, so removal alone isn't enough).
- **Contradictory (10):** inject a **short standalone conflicting document** (not a modified article copy),
  so the two sources are genuinely distinct and survive dedup.

All hold-outs and injected docs are recorded in the manifest.

---

## 7. Ground-truth mapping

Each question's `supporting_facts` = `(title, sentence_index)` pairs. We map `title → doc_id` for the
**gold supporting documents** (Recall@K, MRR, Citation Accuracy) and keep the **gold sentences** for the
chunk-level metric (`EVALUATION.md` §4).

---

## 8. Requirement → where it's satisfied (defense cheat-sheet)

| Spec requirement | Satisfied by |
|------------------|--------------|
| ≥50,000 documents | §3 (~50k full articles, seeded + manifested) |
| Suggested dataset (HotpotQA) | §1 (HotpotQA labels on full Wikipedia articles) |
| Meaningful chunking comparison | §1–§2 (long docs, 256 cap; Recall@K/MRR at 50k; gold-sentence limits stated) |
| Ground truth for retrieval/citation | §7 |
| Unanswerable + contradictory cases | §6 (hold-out+absence check / injected standalone doc) |
| Reproducible data | §3, §6 (seed + manifest) |
| Cost/perf of ingestion (brief) | §4–§5 (offline, quantization, sampling) |
