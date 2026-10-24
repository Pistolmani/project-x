# FRONTEND — Streamlit UI

> The user-facing app for document management and Q&A with source viewing. Decision IDs refer to
> `GRILL-ME.md`. **Framework:** Streamlit (D5). **Pages:** three (D11). The UI is a **thin client** — all
> logic lives in the FastAPI backend (`BACKEND.md`); the UI only calls its HTTP API.

---

## 1. Why Streamlit, and how it talks to the backend (D5)

Streamlit gives us a functional, data-app-style UI fast, which is what the spec asks for (document
management + Q&A + source viewing) — we're graded on the *system*, not pixel polish. It runs as a second
process on the same Hugging Face Space (D4) and calls FastAPI over `http://localhost:<port>`; the base URL
is read from an env var so the same UI works locally and deployed.

> **Defense:** *"The UI holds no business logic — it's a thin client over the same API my evaluation suite
> uses. So what I demo is exactly what I measure."*

---

## 2. Page 1 — Documents (management)

- **Upload:** drag-and-drop one or more files (PDF / HTML / Markdown / TXT — `BACKEND.md` §3.1), with a
  selector for **chunking strategy** (fixed vs. recursive, D9) so you can ingest the same corpus both ways
  for the comparison.
- **List:** table of documents from `GET /documents` — filename, format, version, #chunks, created/updated,
  chunking strategy.
- **Update:** re-ingest a document (triggers `PUT`, version bump — `DATABASE.md` §5).
- **Delete:** removes the document and all its chunks (`DELETE`).
- **Status/feedback:** shows ingest progress, dedup skips ("already indexed"), and errors.

---

## 3. Page 2 — Ask (the investigation UI)

- **Question box** + a config panel (which the interviewer will love): toggle **agentic vs. off**,
  **hybrid vs. dense-only**, chunking config, and top-k — all mapped to `/ask` parameters.
- **Streamed answer** (D-bonus: streaming) rendered as it generates.
- **Inline citations:** claims carry citation markers; clicking one opens the exact **source chunk** and a
  link to its **parent document** (uses `doc_id`/`chunk_id` from the payload — `DATABASE.md` §3).
- **Refusal / contradiction display:** when the engine refuses for *insufficient* evidence or surfaces a
  *contradiction* (`BACKEND.md` §5.4), the UI shows that clearly (e.g. a warning banner: "Not enough
  evidence" or "Sources conflict: A says X, B says Y") instead of a normal answer. This visibly
  demonstrates the anti-hallucination requirement.
- **Trace view (expandable):** shows the agent's steps — sub-questions, each `search_documents` call, and
  the retrieved chunks — so you can **trace the RAG pipeline live in the interview** straight from the UI.

---

## 4. Page 3 — Evaluation dashboard

- **Run / load results:** trigger the eval suite or load the latest run.
- **Metrics:** Recall@K, MRR, Answer Correctness, Citation Accuracy, broken down by question type
  (simple / multi-hop / unanswerable / contradictory).
- **Config comparison (≥2):** side-by-side charts — primarily **chunking A vs. B** (D9) and
  **hybrid vs. dense-only** — so the "comparison of configs" deliverable is visual, not buried in a notebook.
- **Failure-mode table:** worst cases with the retrieved chunks and the model's answer, to support the
  "analysis of failure modes" writeup in the Evaluation Report.

---

## 5. State, errors, and UX principles

- **Stateless UI:** page state is kept in `st.session_state`; the source of truth is always the backend.
- **Graceful errors:** backend errors (`BACKEND.md` §5.5) render as friendly messages; the UI never shows
  a fabricated answer — if the engine refuses, the UI respects that.
- **Honesty first:** refusals and contradictions are first-class UI states, not edge cases — this is the
  product's whole point.

---

## 6. Requirement → where it's satisfied (defense cheat-sheet)

| Spec requirement | Satisfied by |
|------------------|--------------|
| Functional UI | Streamlit app, three pages |
| Document management (add/update/delete) | Page 1 (§2) |
| Q&A with source viewing | Page 2 (§3), inline citations + source chunk viewer |
| Shows refusal / contradiction behavior | Page 2 (§3) |
| Config comparison + failure modes (visual) | Page 3 (§4) |
| Streaming (bonus) | Page 2 streamed answer (§3) |
