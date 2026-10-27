"""Run the behavioral set through the real `POST /ask` endpoint, one question at a time (EVALUATION.md §9).

    python -m evaluation.run_ask_eval --config agentic
    python -m evaluation.run_ask_eval --config single_shot
    python -m evaluation.run_ask_eval --config no_verifier          # unanswerable + contradictory only

Resumable: every answer is appended to data/eval_runs/<config>.jsonl as it arrives, and a re-run skips
questions that already have a non-error answer. The free LLM tiers run out mid-run; after a few
consecutive LLM outages the run stops and can simply be started again later. Repeated LLM requests are
answered from the backend's cache, so a resumed run does not spend quota twice.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import httpx

from app.config import get_settings
from evaluation.prepare_behavioral import SETS_DIR

TYPES = ["simple", "multihop", "unanswerable", "contradictory", "injection"]
CONFIGS = {
    "agentic": ({}, TYPES),
    "single_shot": ({"agentic": False}, TYPES),
    # Ablation: what the verifier adds is refusal and conflict detection, so only those types.
    "no_verifier": ({"verify": False}, ["unanswerable", "contradictory"]),
}
MAX_CONSECUTIVE_OUTAGES = 3


def runs_dir():
    path = get_settings().data_dir / "eval_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_run(config: str) -> dict[str, dict]:
    """question id -> latest record of a run."""
    path = runs_dir() / f"{config}.jsonl"
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", choices=sorted(CONFIGS), default="agentic")
    p.add_argument("--types", default=None, help="comma-separated subset of " + ",".join(TYPES))
    p.add_argument("--limit", type=int, default=None, help="at most this many questions per type")
    p.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    args = p.parse_args()

    params, default_types = CONFIGS[args.config]
    types = args.types.split(",") if args.types else default_types
    sets = json.loads((SETS_DIR / "behavioral.json").read_text(encoding="utf-8"))
    done = {qid for qid, rec in load_run(args.config).items() if rec["answer"].get("status") != "error"}
    todo = [(t, q) for t in types for q in sets[t][: args.limit] if q["id"] not in done]
    print(f"[{args.config}] {len(done)} already answered, {len(todo)} to go")

    outages = 0
    with httpx.Client(base_url=args.api, timeout=900) as http, (runs_dir() / f"{args.config}.jsonl").open(
        "a", encoding="utf-8"
    ) as out:
        for n, (qtype, q) in enumerate(todo, start=1):
            start = time.perf_counter()
            r = http.post("/ask", json={"question": q["question"], "params": params})
            body = r.json()
            record = {"id": q["id"], "type": qtype, "config": args.config, "params": params, "http_status": r.status_code,
                      "answer": body, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            status = body.get("status")
            print(f"  {n}/{len(todo)} [{qtype}] {status:<12} {time.perf_counter() - start:5.1f}s  {q['question'][:70]}")
            outages = outages + 1 if status == "error" and "LLM unavailable" in (body.get("error") or "") else 0
            if outages >= MAX_CONSECUTIVE_OUTAGES:
                print(f"Stopping: {outages} consecutive LLM outages (free-tier quota). Re-run later to resume.")
                return
    print(f"[{args.config}] finished -> {runs_dir() / f'{args.config}.jsonl'}")


if __name__ == "__main__":
    main()
