"""Copy one chunking strategy from the local Qdrant to another Qdrant (the free cloud cluster).

    # .env: QDRANT_URL / QDRANT_API_KEY = the cloud cluster (the destination)
    python -m ingest.copy_to_cloud --strategy fixed --source-url http://localhost:6333

Both strategies are compared locally (they do not fit the 1 GB free tier together); only the winner
is deployed. Vectors are copied as stored, so nothing is re-embedded. Point IDs are deterministic,
so the copy is idempotent: an interrupted run can be repeated, or continued with --start-offset.
"""

import argparse

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from app.config import get_settings
from app.store.qdrant_store import DENSE, NUM_CHUNKS_PREFIX, QdrantStore


def copy_chunks(src: QdrantStore, dst: QdrantStore, strategy: str, batch: int, start_offset=None) -> int:
    by_strategy = models.Filter(must=[models.FieldCondition(
        key="chunking_strategy", match=models.MatchValue(value=strategy))])
    total = src.count(src.chunks, strategy)
    copied, offset = 0, start_offset
    with tqdm(total=total, unit="chunk", desc=f"chunks[{strategy}]") as bar:
        while True:
            points, offset = src.client.scroll(src.chunks, scroll_filter=by_strategy, limit=batch,
                                               offset=offset, with_payload=True, with_vectors=True)
            if points:
                dst.client.upsert(dst.chunks, wait=True, points=[
                    models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
                copied += len(points)
                bar.update(len(points))
                bar.set_postfix(next_offset=str(offset))
            if offset is None:
                return copied


def copy_registry(src: QdrantStore, dst: QdrantStore, strategy: str, batch: int) -> int:
    """Registry points, keeping only the deployed strategy's chunk count."""
    copied, offset = 0, None
    while True:
        points, offset = src.client.scroll(src.documents, limit=batch, offset=offset,
                                           with_payload=True, with_vectors=False)
        records = [{k: v for k, v in p.payload.items()
                    if not k.startswith(NUM_CHUNKS_PREFIX) or k == NUM_CHUNKS_PREFIX + strategy}
                   for p in points]
        dst.upsert_documents(records)
        copied += len(records)
        if offset is None:
            return copied


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-url", default="http://localhost:6333", help="the local Qdrant holding both strategies")
    p.add_argument("--strategy", choices=["recursive", "fixed"], default="fixed")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--start-offset", default=None, help="resume the chunk copy from a printed next_offset")
    args = p.parse_args()

    settings = get_settings()
    src = QdrantStore(QdrantClient(url=args.source_url, timeout=120),
                      settings.chunks_collection, settings.documents_collection)
    dst = QdrantStore.from_settings(settings)
    if args.source_url.rstrip("/") == settings.qdrant_url.rstrip("/"):
        raise SystemExit("QDRANT_URL points at the source; set it to the destination cluster")

    # Same schema as the source, without loading the embedding model just to learn its size.
    dst.ensure_collections(src.client.get_collection(src.chunks).config.params.vectors[DENSE].size)
    n_chunks = copy_chunks(src, dst, args.strategy, args.batch, args.start_offset)
    n_docs = copy_registry(src, dst, args.strategy, 1000)
    print(f"copied {n_chunks} chunks and {n_docs} registry points; destination now holds "
          f"{dst.count(dst.chunks)} chunks, {dst.count(dst.documents)} documents")


if __name__ == "__main__":
    main()
