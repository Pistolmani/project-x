# AI USAGE REPORT

> Required deliverable. This log records **how AI was used to build this project**, kept as we go so it's
> accurate rather than reconstructed later.
>
> **Policy (from the task spec).** AI assistants are allowed during the **build phase** and are logged
> here. **No AI assistants are permitted during the technical interview** — every decision in
> `GRILL-ME.md` is one the human author made and must defend unaided.
>
> **How to use.** Append one entry per working session: what the AI did, what the human decided/verified,
> and which tools/models were used. Be specific and honest.

---

## Summary (fill in at the end)

- **Primary AI tools:** Cursor agent (Claude), Cursor CLI, Cursor subagents; Claude Code (Claude Opus 5.5)
  for design reviews and doc edits.
- **Where AI helped most:** _TBD_
- **Where AI was deliberately NOT used / where the human drove:** all architecture decisions (see `GRILL-ME.md`).
- **Verification practices:** _TBD_

---

## Session log

### 2026-10-24 — Session 1: Architecture & decisions (planning only, no code)

**What the AI did**
- Ran a structured decision process (rounds of questions, each with a recommended answer and the
  trade-offs) to settle the architecture — modeled on the `grilling` skill.
- Looked up the exact Cursor CLI resume commands via a `cursor-guide` subagent.
- Wrote the design/decision documents: `GRILL-ME.md`, `BACKEND.md`, `DATABASE.md`, `FRONTEND.md`,
  `EVALUATION.md`, `DATASET.md`, `README.md`, `VISUAL-PLAN.md`, `INTERVIEW-PREP.md`, and this file.

**What the human decided / owns**
- All locked decisions (D1–D11, E1–E5, A1–A8) — dataset, vector DB, models, hosting, app stack, registry,
  hybrid search, custom ranking, chunking, reasoning pattern, frontend scope, the five evaluation
  decisions, and the eight build decisions. The AI proposed options and recommendations; the human chose each.
- Explicitly steered decisions rather than accepting defaults (e.g. chose **both** custom ranking stages;
  requested the Streamlit-Community-Cloud-vs-HF-Spaces trade-off be documented).

**Tools / models used**
- Cursor agent (Claude) in the IDE.
- `cursor-guide` subagent (Cursor CLI documentation lookup).

**Notes**
- No code written yet — docs/spec only. The in-IDE terminal was unresponsive this session; file edits
  were unaffected.

---

### 2026-10-24 — Session 2: Design-review pass (still no code)

**What the AI did**
- Applied an external design review against the dataset. Fixed: dataset → **full Wikipedia articles**
  (so chunking is meaningful); replaced the **recency/source-authority** re-ranker signals with
  data-grounded ones; **exact-dedup only** (no near-dup collapse, which would have broken conflict
  detection); made **grounding + contradiction detection concrete** (cross-encoder floor + NLI/consistency
  judge); added **offline ingestion** + a **cost/perf** plan; switched Qdrant point IDs to **`uuid5`**.
- Added `SETUP.md` (free-tier limits, secrets, offline ingestion, deploy), filled decisions **A5/A6**, and
  corrected the decision count to a contiguous **24**.

**What the human decided / owns**
- The human commissioned the design review and chose which findings to adopt.
- The chunking fix (**document = full article**, 256-token cap, chunk-level gold-sentence metric) and the
  **standalone-injected** contradiction method were **recommended by Claude Code** in that review; the
  human decided to adopt them, and the Cursor agent applied them to the docs.

**Tools / models used**
- **Claude Code (Claude Opus 5.5)** — performed the design review and proposed the fixes listed above.
- Cursor agent (Claude) — applied the fixes to the docs. *(Build-phase model/API usage — e.g. Gemini,
  Groq, bge — will be logged here once code runs.)*

**Notes**
- Terminal still unresponsive; docs only.

---

### 2026-10-24 — Session 3: Second review pass (still no code)

**What the AI did**
- Claude Code re-checked the revised docs and found issues introduced by the full-article switch, then,
  after the human approved, edited the docs directly:
  - **Whole-lead swap** — replace the entire modern lead section, not just its first paragraph, to avoid
    planting unintended 2017-vs-today contradictions (`DATASET.md` §1).
  - **Unanswerable eligibility** — only bridge questions with distinctive entity answers (no yes/no /
    common strings), and a **selection order** that drops questions sharing held-out articles
    (`DATASET.md` §3/§6, `EVALUATION.md`).
  - **Qdrant RAM layout** — int8 in RAM; originals, payload and sparse index on disk (`DATABASE.md` §7).
  - **Free-cluster deletion** (suspend 1 wk / delete 4 wk) → rebuild-from-manifest defense (`DATABASE.md`
    §8, `SETUP.md` §6); **`app_port: 8501`** for the Space (`SETUP.md` §5).
  - **bge tokenizer** for chunk counting (A2); corrected the "simple = `easy`" note; removed stale
    recency/authority wording from `VISUAL-PLAN.md`; corrected Session 2 attribution in this file.
- Verified three external facts by web search before editing: HotpotQA's `train-easy` tier is mostly
  single-hop and dev/test are all `hard` (HotpotQA paper, arXiv 1809.09600); Qdrant free-cluster
  suspension/deletion timeline and 1 GB RAM / 4 GB disk (Qdrant Cloud docs); `app_port` in the Space
  README YAML.

**What the human decided / owns**
- Asked for the review, approved "make all edits", and owns all resulting decisions.

**Tools / models used**
- Claude Code (Claude Opus 5.5), with web search.

---

### 2026-10-24 — Session 4: Phase 1 (ingestion) + Phase 2a (retrieval core) — first code

**What the AI did (Claude Code, Claude Opus 5.5)**
- Moved the project out of OneDrive into a single repo folder (human's choice to keep everything in one
  place; AI recommended leaving OneDrive because of sync locks on caches/venv/.git).
- Verified dataset formats before coding (Wikipedia keeps section headings as standalone lines; HotpotQA
  fields), then wrote Phase 1: document model, fixed/recursive chunkers, embedders, Qdrant store,
  HotpotQA/Wikipedia pipeline, selection rules, two-stage `build_corpus` — with pytest coverage.
- Wrote Phase 2a: custom weighted RRF, feature re-ranker, cross-encoder wrapper, searcher, retrieval
  metrics and a quick comparison runner — with pytest coverage (65 tests total, all passing).
- Ran a 2k-article sample end to end and measured: article length (mean 1,473 tokens), chunk counts
  (recursive 9.8 / fixed 7.0 per article), GPU indexing speed (~40 s per 2k articles per strategy), and
  an absence-check failure rate (11 of 30 unanswerable candidates) that led to more spare candidates.
- **Found a flaw in its own earlier recommendation:** the chunk-level gold-sentence metric (proposed by
  the AI in Session 2's review) cannot separate fixed from recursive chunking here — fixed windows cut
  only 0.2% of gold sentences because HotpotQA's gold sentences sit in the article lead. Reported it and
  corrected the docs to state the limitation instead of the earlier overclaim.

**What the human decided / owns**
- Repo location; local Qdrant first, Qdrant Cloud later (human created the cluster and wrote `.env`
  themselves — the API key was never shared with the AI).
- Phase 2a scope: separate tuning set, per-article cap of 2, fusion-prior feature.
- After the flaw was reported: reframe the chunking comparison honestly (Recall@K/MRR at 50k) and run the
  full-scale comparison locally, pushing only the winning strategy to Qdrant Cloud.

**Tools / models used**
- Claude Code (Claude Opus 5.5), web search; models run locally: `bge-small-en-v1.5`, `bge-reranker-base`,
  FastEmbed BM25.

---

### 2026-10-24 — Session 5: Phase 2a results + Phase 2b (reasoning engine and API)

**What the AI did (Claude Code, Claude Opus 5.5)**
- Finished the 50k index for both strategies after a low-memory kill (streaming fix), ran the chunking /
  hybrid / re-ranker comparison and the weight tuning, and reported that tuning **overfit** (held-out
  results got worse), so the defaults were kept; wrote `copy_to_cloud.py` and deployed the fixed strategy.
- Planned Phase 2b (the human asked for the plan before any work and approved it), then wrote: the LLM
  layer (Gemini + Groq + fallback), tools and agent loop, gates (citation check, calibrated relevance
  floor, verifier), SQLite cache, PDF/HTML/Markdown/TXT loaders, add/update/delete service, FastAPI app —
  with pytest coverage (107 unit tests passing).
- Ran real questions and found problems the fakes could not: Gemini Flash's free tier is 20 requests/day
  (switched the investigator to Flash-Lite), provider retry delays were ignored (now honored), the first
  verifier prompt was too strict (refused a correct answer), and one "unanswerable" eval question is
  actually answerable from other articles because the absence check matches only the exact answer string
  ("Point coloration" vs. "colorpoint") — recorded for the failure analysis.

**What the human decided / owns**
- Deploying the fixed strategy with default weights; the Phase 2b plan and its three decisions (LLM-judge
  contradiction check, engine before API, the new dependencies). Created the Gemini and Groq keys and put
  them in `.env` themselves (never shown to the AI).

**Tools / models used**
- Claude Code (Claude Opus 5.5); Gemini 3.5 Flash-Lite / Flash and Groq gpt-oss-120b as the engine's LLMs.

---

<!-- Add the next session entry below this line -->
