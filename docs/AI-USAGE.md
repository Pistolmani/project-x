# AI Usage Report

AI assistants were used while building this project, as the task allows. None are used in the technical
interview. This report says which tools I used, what they did, what I did myself, and how I checked
their work. It was kept during the build, not reconstructed at the end.

## Tools

| Tool | Used for |
|---|---|
| **Claude Code** (Claude Opus 5.5) | Design reviews, phase plans, most of the code and tests around my core components, evaluation runs, deployment, docs |
| **Cursor agent** (Claude; Gemini 4 Argon) | The first architecture-decision sessions; implementing the Streamlit UI (Phase 3) |

## What I did myself

- **Every important decision.** The architecture, evaluation and build decisions (D1–D11, E1–E5, A1–A8 in
  [`DECISIONS.md`](./DECISIONS.md)). Also the calls during the build: which review findings to adopt,
  keeping the default fusion weights when tuning overfit, approving each phase plan, moving hosting to
  Streamlit Community Cloud when Hugging Face Docker Spaces became paid, and adding guardrails. The AI
  proposed options with trade-offs; I chose.
- **The core components.** I wrote these and gave them to the AI to integrate:

  | Component | Where |
  |---|---|
  | Custom weighted RRF fusion | `app/retrieval/fusion.py` |
  | Fixed chunking with overlap | `_fixed_spans` in `app/ingestion/chunking.py` |
  | Max 2 chunks per article | `cap_per_doc` in `app/retrieval/searcher.py` |
  | Relevance floor and citation check | `app/agent/gates.py` |
  | Recall@K and MRR | `evaluation/retrieval_metrics.py` |

- **Accounts and secrets.** I created the Qdrant Cloud cluster, the Gemini and Groq keys, and the
  Streamlit app and its secrets. Keys were never shown to the AI.

## What the AI did

- Ran structured decision rounds and wrote the first design docs.
- Wrote most of the code and tests around my components: ingestion, the Qdrant store, the agent loop,
  the LLM fallback chain, the API, the UI, the evaluation runners and deployment.
- Ran the system on real data and found what the test fakes could not:
  - Gemini Flash's free tier allows only 20 requests a day;
  - the relevance floor was scoring the wrong query;
  - long runs exceeded Groq's 8k-token limit;
  - the deployed app mishandled paths, secrets and memory.
- Fitted the free host: an ONNX model backend, with ONNX-vs-torch parity measured before it was deployed.
- Added the guardrails against prompt injection and harmful output, and their red-team evaluation.

## How I checked the AI's work

- A written plan before each phase; nothing was built until I approved it.
- Tests for every module, plus runs against the real Qdrant Cloud cluster and LLMs, not only mocks.
- Report numbers are read from `results/` files, not typed by hand.
- External facts (HotpotQA paper, Qdrant limits, hosting prices) were checked against sources.
- No tuning on the test set: the verifier was not changed after seeing the evaluation results.

## AI mistakes that were caught and fixed

- An evaluation metric the AI proposed (chunk-level gold sentences) could not tell the two chunking
  strategies apart. It was reported, and the comparison was redone with Recall@K and MRR.
- A first "is this question really unanswerable" check was too loose (it matched "citigroup" for
  "CIT Group").
- A test passed on a broken app layout. It was replaced with one that runs the app the way the host does.
- Uploads used too much memory and crashed the live app. Memory was measured, and the batch size and
  runtime settings were fixed.
- An earlier version of this report credited the AI with the components in the table above.

## Session log

| Date | Session | Main work | My decisions |
|---|---|---|---|
| 10-24 | 1. Architecture | Decision rounds; design docs | All 24 locked decisions |
| 10-24 | 2–3. Design reviews | Two review passes on the docs (full-article dataset, exact dedup, whole-lead swap, Qdrant RAM layout); facts checked by web search | Which findings to adopt |
| 10-24 | 4. Ingestion + retrieval | Ingestion pipeline, store, re-ranker, cross-encoder, searcher, around my components; 2k-article trial run | Repo layout; local Qdrant first; tuning set; 2-per-article cap |
| 10-24 | 5. Reasoning engine + API | 50k index for both strategies; LLM layer, agent loop, verifier, cache, loaders, FastAPI | Deploy fixed chunking with default weights; Phase 2b plan |
| 10-24 | 6. UI | Streamlit pages, `/ask/stream`, single-shot mode (Cursor agent) | Phase 3 plan |
| 10-24 | 7. Evaluation | Review fixes; 55-question set, judge, 3 configurations, report | Phase 4 plan and its five decisions |
| 10-25 | 8. Deployment | Demo protections, ONNX backend, Streamlit Cloud fixes, passage display | Hosting choice; app and secrets setup |
| 10-27 | 9. Hardening | Memory fix for uploads; guardrails and their red-team evaluation; this report shortened | Guardrails; polish scope; the attribution in this report |
