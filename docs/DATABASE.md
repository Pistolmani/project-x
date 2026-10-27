# DATABASE — Vector Store & Document Management

> How data is stored, linked, de-duplicated, and kept durable. Decision IDs (D1–D11) refer to
> `DECISIONS.md`. **Store:** Qdrant Cloud (D2). **No separate relational DB** — the document registry
> lives inside Qdrant too (D6).

---

## 1. Why Qdrant, and the persistence boundary (D2, D4, D6)

The host that runs our code (Streamlit Community Cloud; the original plan was a Hugging Face Space) has an
**ephemeral disk**: it resets on every restart or redeploy. So **all durable state lives in Qdrant Cloud**,
which is a hosted, persistent service. The app is treated as **stateless compute**. This is the single most important fact about our data layer, and the reason we did
**not** put the registry in a local SQLite file (that would be the one thing that disappears on restart).

> **Defense in one line:** *"Durable state is 100% in Qdrant Cloud; the compute host holds nothing it
> can't lose. That's why a Space restart never costs me data."*

---

## 2. The two collections

We use **two Qdrant collections**, linked by a shared `doc_id`.

### 2.1 `chunks` — one point per chunk (the searchable units)
- **Point ID:** must be a Qdrant-valid ID (uint64 or UUID), so we use **`uuid5` of
  `"{doc_id}:{chunking_strategy}:{chunk_index}"`** — deterministic (idempotent re-ingestion) and valid. The
  strategy is part of the key because both strategies' chunks of a document share this collection during
  the comparison. The human-readable `chunk_id` (same string) lives in the payload.
- **Vectors (named vectors on one point):**
  - `dense`: `bge-small-en-v1.5` embedding (384-d), cosine distance, stored **int8-quantized** (D3/A2).
  - `sparse`: BM25-style sparse vector — the keyword half of hybrid search (D7).
- **Payload:**
  `doc_id`, `chunk_id`, `text`, `title`, `source`, `format`, `chunk_index`, `chunking_strategy`,
  `start_char`, `end_char` (exact position in the document — powers the chunk-level gold-sentence
  metric), `n_tokens`, `version`, `sha256`, `created_at`.
- **Payload indexes** on `doc_id`, `sha256`, `chunking_strategy`, `version` for fast filtering, deletes,
  and dedup lookups.

### 2.2 `documents` — one point per document (the registry, D6)
Qdrant stores the document-level metadata here so we don't need a relational DB. Each point:
- **Vector:** none — Qdrant allows vectorless points, and we search chunks, not documents.
- **Payload:** `doc_id`, `filename`, `format`, `title`, `source`, `sha256`, `version`, `created_at`,
  `updated_at`, `lead_replaced`, and one **`num_chunks_<strategy>`** key per indexed strategy (e.g.
  `num_chunks_recursive`). Flat keys, not a nested dict, because `set_payload` merges top-level keys
  while `upsert` replaces the whole payload.
- **Written after its chunks:** during bulk indexing a document's registry point is upserted only once
  its chunks have landed, so "indexed" in the registry is always true and an interrupted run resumes
  exactly where it stopped.

> **Why a second collection instead of just chunk payloads (defense):** listing/paginating documents,
> version bumps, and dedup checks are document-level operations. A dedicated collection gives me a clean
> one-point-per-document registry to scroll and filter, without scanning every chunk.

---

## 3. ID scheme (how everything links — answers the "how are they connected" question)

- `doc_id` — UUID per document. Lives on the `documents` point **and** on every `chunks` point for that
  document.
- `doc_id` for corpus articles is **`uuid5("wiki:{title}")`** — deterministic, so a rebuild from the
  manifest reproduces the same IDs and the eval files stay valid. Uploaded files get a random UUID.
- `chunk_id` — human-readable `{doc_id}:{chunking_strategy}:{chunk_index}`, kept **in the payload**. The
  Qdrant **point ID** is `uuid5(chunk_id)` (Qdrant point IDs must be uint64 or UUID — a raw string is
  invalid).
- **The link is these shared IDs, coordinated by the backend** — there is no database-to-database wiring.
  On query, a chunk's payload carries its `doc_id`, which is how the UI jumps from a cited snippet to its
  source document.

---

## 4. Deduplication (D requirement)

- **Exact dedup only:** `sha256` of the normalized content is stored in payload. Before ingesting, we query
  by `sha256`; if present, we skip. O(1)-ish via the payload index.
- **We deliberately do NOT collapse near-duplicates (review fix).** The evaluation's injected contradictory
  documents are *near*-duplicates of real ones; collapsing them would make conflict detection impossible by
  design. Near-identical-but-distinct chunks are kept as separate points.

---

## 5. Document lifecycle — exact steps

### Add
1. Parse + chunk + embed (dense + sparse).
2. `sha256` check → skip if duplicate.
3. Upsert N points into `chunks` (with `doc_id`).
4. Upsert 1 point into `documents`.

### Update (re-ingest an existing doc)
1. Re-parse/chunk/embed. 2. **Bump `version`.** 3. Upsert new chunk points. 4. **Delete old chunks**
(`filter: doc_id == X AND version < new`). 5. Update the `documents` point.
> Insert-new-then-delete-old so a crash mid-update never leaves the doc unsearchable.

### Delete
1. `chunks`: delete by `filter: doc_id == X`. 2. `documents`: delete the point `doc_id == X`.

---

## 6. Hybrid search at the storage layer (D7)

Each `chunks` point holds **both** a `dense` and a `sparse` vector. A hybrid query issues both a dense
(cosine) and a sparse (lexical) search against the same collection; the two ranked lists are then fused by
our custom RRF in the backend (`BACKEND.md` §4[1]). Keeping both vectors in one store means **no separate
keyword index to build, host, or keep in sync** — and it's all persistent.

---

## 7. Free-tier sizing (D2, D4) — revised (review fix)

- Corpus: ~50k **full Wikipedia articles** (`DATASET.md`) ≈ ~74M tokens (measured mean 1,473/article)
  → **~490k recursive chunks** or **~350k fixed chunks** at a 256-token cap (measured on a 2k sample,
  ×25). This is **tight on Qdrant's free tier (1 GB RAM / 4 GB disk)**, not "comfortable."
- **Quantization alone does NOT save RAM (review fix).** By default Qdrant keeps the original float32
  vectors in RAM *alongside* the int8 copies. Rough math: 490k × 384 × 4 B ≈ **~750 MB of float32** for
  the recursive collection alone — most of the 1 GB. Chunk text (~74M tokens) would add hundreds of MB
  more. So the collection is configured explicitly:

  | What | Where | Qdrant setting |
  |------|-------|----------------|
  | int8 dense vectors (search) | **RAM** | `ScalarQuantization(type=INT8, always_ram=True)` |
  | original float32 dense vectors (rescoring) | **disk** | `VectorParams(..., on_disk=True)` |
  | chunk text + metadata payload | **disk** | `on_disk_payload=True` (payload *indexes* stay in RAM) |
  | sparse (BM25) index | **disk** | `SparseIndexParams(on_disk=True)` |

  Rough RAM after this: ~190 MB of int8 vectors (490k × 384 B) plus the HNSW graph for one strategy.
  Cost: rescoring reads originals from disk, adding a little latency — acceptable for a demo.
- **Disk decides where the comparison runs.** Estimated per point: float32 + int8 vectors ≈ 1.9 KB,
  sparse vector + index ≈ 1.4 KB, payload ≈ 1 KB, graph ≈ 0.2 KB → **~2.2 GB for recursive, ~1.7 GB for
  fixed, ~3.9 GB for both** — too close to the 4 GB disk. So the **chunking comparison runs at full 50k
  scale in local Docker Qdrant**, and **Qdrant Cloud receives only the winning strategy**.
  (A direct measurement of the on-disk size failed locally: Docker-on-Windows bind mounts break Qdrant's
  snapshot archiving, and raw `du` is dominated by pre-allocated segment files — 442 MB for 2k vectorless
  registry points — so the per-point estimate above is computed, not read off the disk. Qdrant Cloud's
  console shows the real figure after upload.)
- Further lever if still tight: cap very long articles (p95 is 6k tokens, max 20k).
- Free-tier operational limits (idle suspension, etc.)
  are in `SETUP.md` §6.

---

## 8. Backup / rebuild

Because Qdrant Cloud is the source of truth, "backup" = Qdrant snapshots. The app can be redeployed from
scratch at any time and reconnects to the same Qdrant collections — nothing to migrate.

**The free cluster itself is not forever (review fix):** Qdrant suspends an unused free cluster after
**1 week** and **deletes** it after **4 weeks** if not reactivated. So the real durability guarantee is
**reproducibility**: the seeded manifest (`DATASET.md` §3) plus the idempotent ingestion script rebuild the
exact same corpus into a fresh cluster (re-runs are safe thanks to `sha256` dedup, §4). Keep the cluster
active (periodic query) during the evaluation and interview window.

---

## 9. Requirement → where it's satisfied (defense cheat-sheet)

| Spec requirement | Satisfied by |
|------------------|--------------|
| Use a vector DB | Qdrant Cloud (§1) |
| Semantic search via embeddings | `dense` vectors, cosine (§2.1) |
| Hybrid search | `dense` + `sparse` in one store (§6) |
| Metadata extraction/storage | payload + `documents` collection (§2) |
| Dedup handling | `sha256` exact; near-dups kept distinct on purpose (§4) |
| Add / update / delete | §5 |
| Persistent data on free host | durable Qdrant Cloud, stateless Space (§1); limits in `SETUP.md` §6 |
