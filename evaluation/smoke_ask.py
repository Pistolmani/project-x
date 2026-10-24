"""Run a few behavioral questions through the real agent and print what happened.

    python -m evaluation.smoke_ask --per-type 2 --types simple,multihop,unanswerable

A sanity check that the real LLM behaves like the scripted one in the tests, not the evaluation itself
(that is Phase 4: judge-scored correctness, citation P/R, refusal and conflict accuracy).
"""

import argparse
import json
import sys

from app.engine import build_engine


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-type", type=int, default=2)
    p.add_argument("--types", default="simple,multihop,unanswerable")
    p.add_argument("--out", default=None, help="optional JSON file for the full answers")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # passages contain characters the Windows console codepage lacks

    engine = build_engine()
    sets = json.loads((engine.settings.data_dir / "eval" / "questions.json").read_text(encoding="utf-8"))
    records = []
    for qtype in args.types.split(","):
        for q in sets["behavioral"][qtype][: args.per_type]:
            out = engine.agent.ask(q["question"])
            gold = {g["title"] for g in q.get("gold", [])}
            cited = [c.title for c in out.citations]
            print(f"\n[{qtype}] {q['question']}\n  gold: {q.get('answer')!r}  gold docs: {sorted(gold)}")
            print(f"  -> {out.status}: {out.answer}\n  cited: {cited}\n  searched: {out.searched}")
            print(f"  gates: { {k: v.get('passed', v.get('supported')) for k, v in out.gates.items()} }"
                  f"  llm calls: {out.usage['llm_calls']}  providers: "
                  f"{sorted({s['provider'] for s in out.trace if s['step'] == 'llm'})}  {out.ms / 1000:.1f}s")  # fmt: skip
            if out.error:
                print(f"  error: {out.error}")
            records.append({"type": qtype, "question": q, "status": out.status, "answer": out.answer,
                            "cited": cited, "searched": out.searched, "gates": out.gates, "usage": out.usage})  # fmt: skip
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)


if __name__ == "__main__":
    main()
