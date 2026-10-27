# FRONTEND — Streamlit UI

> The user-facing app for document management and Q&A with source viewing. Decision IDs refer to
> `DECISIONS.md`. **Framework:** Streamlit (`ui/streamlit_app.py`, D5). **Pages:** three (`st.navigation`, D11).
> The UI is a **thin client** (`ui/client.py`) — all business logic lives in the FastAPI backend
> (`BACKEND.md`); the UI only calls its HTTP API.

---

## 1. Why Streamlit, and how it talks to the backend (D5)

Streamlit gives us a functional, data-app-style UI fast, which is what the spec asks for (document
management + Q&A + source viewing) — we're graded on the *system*, not pixel polish. It runs as a second
process next to FastAPI locally, and on Streamlit Community Cloud (which runs only `streamlit run`) it
starts FastAPI in a background thread of its own process (`EMBEDDED_API=true`, `ui/embedded_api.py`). Either
way it calls FastAPI over HTTP at `API_URL` (default `http://localhost:8000`), so the same UI works locally
and deployed.

> **Defense:** *"The UI holds no business logic — it's a thin client over the same API my evaluation suite
> uses. So what I demo is exactly what I measure."*

---

## 2. Page 1 — Documents (`ui/pages/documents.py`)

- **Upload:** drag-and-drop one or more files (PDF / HTML / Markdown / TXT — `BACKEND.md` §3.1).
  Each file reports `created`, `already indexed` (exact `sha256` duplicate), or the loader's error. A
  created file that is a near-copy of an existing document also gets a warning naming that document and
  the share of its wording that matches; it is indexed anyway (`BACKEND.md` §3.5).
  The page states the deployed strategy (**`fixed`** — 256-token windows, 32-token overlap) and points to
  the Evaluation page for the 50k fixed-vs-recursive comparison (uploading into a non-deployed strategy
  would put chunks where nobody searches).
- **Two tabs (powered by `GET /documents` with `format` / `exclude_format` filters):**
  - **My uploads:** table of user-uploaded documents (`exclude_format=wikipedia`) showing title, filename,
    format, version, chunk count, and updated timestamp, with **View**, **Update** (re-upload → `PUT`,
    version bump — `DATABASE.md` §5), and **Delete** (confirmation step → `DELETE`).
  - **Corpus (50k Wikipedia):** paginated read-only browser over the 50,000 indexed Wikipedia articles
    (`format=wikipedia`).
- **Document view:** opens any document's chunks in reading order (`GET /documents/{doc_id}`).

---

## 3. Page 2 — Ask (`ui/pages/ask.py`)

- **Question box + example questions:** one-click simple, multi-hop, and unanswerable example prompts.
- **Sidebar config panel (mapped to `/ask` and `/ask/stream` parameters):**
  - **Agentic multi-step loop on/off** (`agentic: true` runs the multi-hop tool loop; `agentic: false`
    runs single-shot RAG — one search with the raw question, then one `submit_answer` call, through the
    same post-generation gates);
  - **Retrieval mode:** `hybrid` / `dense` / `sparse`;
  - **Custom feature re-ranker** on/off and **cross-encoder (`bge-reranker`)** on/off;
  - **Top-K passages**, **relevance floor** toggle + slider (calibrated default `3.92`), and **verifier**
    toggle.
- **Live progress streaming (`POST /ask/stream`, SSE):** streams real-time progress events as the agent
  works (`"planning…"`, `"searching: <query>"`, `"found 8 passages"`, `"verifying…"`) before rendering
  the verified final answer. We deliberately stream **progress rather than unverified draft tokens** so the
  UI never flashes an unverified claim that the verifier gate subsequently retracts.
- **Distinct displays for every outcome (`status`):**
  - **Answered (`answered`):** grounded answer with inline `[S#]` citations and citation chips.
  - **Not enough evidence (`insufficient`):** warning banner with the refusal reason and the list of
    queries searched.
  - **Sources conflict (`conflict`):** conflict banner stating both sides (`[S#]`) and the verifier's
    explanation.
  - **Blocked (`blocked`):** 🛡️ banner with the guardrail's category and reason (`BACKEND.md` §7). A
    cited passage that contains instruction-like text gets a ⚠ warning on its source card.
- **Sources & parent document viewer:** each cited passage shows the chunk text, its cross-encoder
  score, and an **Open parent document** button that loads the full article's chunks in order.
  Fixed-size chunks overlap by 32 tokens and ignore sentence boundaries, so the display
  (`ui/passages.py`, display only) **merges overlapping or touching chunks of one article** into one
  passage labelled with all its ids (`[S1, S2] 1975 in film · chunks #8–9`). The merge is exact, because
  chunk text is the article text between `start_char` and `end_char`. The display also **marks cut
  edges with "…"** and **shows the text literally**: line breaks kept and Markdown characters escaped, so
  list-style articles don't run together.
- **"How it got here" trace (expandable):** shows live stream events, every search query, each LLM/tool
  step with provider/model, latency and token counts, the input and output guardrails (which checks ran,
  and whether the classifier was unavailable), and the three anti-hallucination gates (citation check,
  relevance floor, verifier) with why each passed or failed.

---

## 4. Page 3 — Evaluation dashboard (`ui/pages/evaluation.py`)

- **Load-only from `GET /eval/results` (`results/`):** the dashboard reads tracked JSON artifacts rather
  than triggering ~250+ LLM calls live in the browser (which would exhaust the demo's free-tier quota).
- **1. Retrieval & chunking comparison (50k articles):** metrics cards (50,000 docs, 300,059 fixed chunks
  vs. 433,425 recursive chunks — 30.8% storage savings), full table and bar chart comparing `dense only`,
  `hybrid (RRF)`, `hybrid + features`, and `full pipeline` across both chunking strategies.
- **2. Custom RRF & feature re-ranker weight ablation:** 100-question tuning set vs. 300-question held-out
  set showing that default feature weights beat plain RRF on held-out questions while coordinate-ascent
  tuning on 100 questions overfit.
- **3. End-to-end `/ask` behavioral evaluation:** renders summary metrics, per-type breakdown (simple,
  multi-hop, unanswerable, contradictory), and the failure-mode table once Phase 4 writes
  `results/behavioral_eval.json`.
- **Live service metrics (`GET /metrics`):** uptime, questions answered, answer time p50/p95, guardrail
  blocks, outcomes and LLM calls per provider, request latency per route and cache hit rates, for the
  running backend since it started (`BACKEND.md` §9).

---

## 5. State, errors, and UX principles

- **Stateless UI:** page state is kept in `st.session_state`; the source of truth is always the backend.
- **Graceful errors (`ui/client.py`):** backend errors (`BACKEND.md` §5.5) and timeouts become `BackendError`
  messages rendered cleanly in the UI; the UI never fabricates an answer.
- **Honesty first:** refusals and contradictions are first-class UI states, not edge cases — this is the
  product's whole point.
- **Tested (`tests/test_ui.py`):** Streamlit `AppTest` suite covers all three pages, config toggles,
  document view/delete confirmation, and `answered` / `insufficient` / `conflict` / `blocked` states
  against a fake client.

---

## 6. Requirement → where it's satisfied (defense cheat-sheet)

| Spec requirement | Satisfied by |
|------------------|--------------|
| Functional UI | Streamlit app (`ui/streamlit_app.py`), three pages |
| Document management (add/update/delete) | Page 1 (`ui/pages/documents.py`, §2) |
| Q&A with source viewing | Page 2 (`ui/pages/ask.py`, §3), inline citations + chunk & parent-doc viewer |
| Shows refusal / contradiction behavior | Page 2 (§3) distinct banners + gate trace |
| Config comparison + failure modes (visual) | Page 3 (`ui/pages/evaluation.py`, §4) |
| Streaming (bonus) | Page 2 live SSE progress stream (`POST /ask/stream`, §3) |
