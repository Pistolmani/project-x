"""Ingest (or remove) the evaluation's planted documents through the public API (EVALUATION.md §5–§6).

    python -m evaluation.plant_documents            # POST /documents for each file, tagged eval_planted
    python -m evaluation.plant_documents --remove   # DELETE them again

They go through the same upload path as any user file, so the conflict and injection cases test the real
system. Re-running is safe: identical content comes back as a duplicate with its existing doc_id.
Writes `evaluation/sets/planted_ids.json` (file -> doc_id), which the eval runner uses to check that a
surfaced conflict cites both the original article and the planted document.
"""

import argparse
import json
import os

import httpx

from evaluation.prepare_behavioral import PLANTED_DIR, SETS_DIR

TAG = "eval_planted"
IDS_FILE = SETS_DIR / "planted_ids.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    p.add_argument("--remove", action="store_true")
    args = p.parse_args()

    sets = json.loads((SETS_DIR / "behavioral.json").read_text(encoding="utf-8"))
    files = [q["planted"]["file"] for kind in ("contradictory", "injection") for q in sets[kind]]
    with httpx.Client(base_url=args.api, timeout=120) as http:
        if args.remove:
            ids = json.loads(IDS_FILE.read_text(encoding="utf-8")) if IDS_FILE.exists() else {}
            for file, doc_id in ids.items():
                status = http.delete(f"/documents/{doc_id}").status_code
                print(f"{file}: {'deleted' if status == 204 else f'HTTP {status}'}")
            IDS_FILE.unlink(missing_ok=True)
            return
        ids = {}
        for file in files:
            path = PLANTED_DIR / file
            r = http.post("/documents", files={"file": (file, path.read_bytes())}, data={"tag": TAG})
            r.raise_for_status()
            body = r.json()
            ids[file] = body["doc_id"]
            print(f"{file}: {body['status']} {body['doc_id']} ({sum(body['num_chunks'].values())} chunk(s))")
    IDS_FILE.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    print(f"{len(ids)} planted documents -> {IDS_FILE}")


if __name__ == "__main__":
    main()
