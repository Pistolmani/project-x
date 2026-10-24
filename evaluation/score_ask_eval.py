"""Score saved /ask runs: LLM judge, metrics by type and configuration, failure buckets (EVALUATION.md §4, §8).

    python -m evaluation.score_ask_eval

Reads data/eval_runs/<config>.jsonl for every configuration that has been run and writes
results/behavioral_eval.json, which the Evaluation dashboard renders. The judge is Groq gpt-oss-120b:
a different model family from the Gemini investigator, so no model grades its own answers. Judge calls
are cached, so re-scoring is free.
"""

import json
import statistics
import sys
from collections import Counter, defaultdict

from app.cache import Cache
from app.config import get_settings
from app.llm.cached import CachedLLM
from app.llm.fallback import FallbackLLM
from app.llm.groq import GroqLLM
from evaluation.answer_metrics import Judge, score_record
from evaluation.prepare_behavioral import SETS_DIR
from evaluation.run_ask_eval import CONFIGS, load_run

ANSWERABLE = ("simple", "multihop", "injection")


def build_judge(settings) -> Judge:
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is required for the judge")
    groq = GroqLLM(settings.groq_api_key.get_secret_value(), settings.groq_model, settings.llm_timeout_s)
    return Judge(CachedLLM(FallbackLLM([groq], retries=4, max_wait_s=90), Cache(settings.cache_path), "judge"))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    sets = json.loads((SETS_DIR / "behavioral.json").read_text(encoding="utf-8"))
    planted_ids = json.loads((SETS_DIR / "planted_ids.json").read_text(encoding="utf-8"))
    questions = {q["id"]: (t, q) for t in CONFIGS["agentic"][1] for q in sets[t]}
    judge = build_judge(settings)

    configs, by_type_rows, failures = {}, [], []
    for config in CONFIGS:
        run = load_run(config)
        if not run:
            continue
        scored = defaultdict(list)  # type -> [(question, answer, outcome)]
        for qid, rec in run.items():
            qtype, q = questions[qid]
            answer = rec["answer"]
            verdict = None
            if qtype in ANSWERABLE and answer.get("status") == "answered":
                verdict = judge.grade(q["question"], q["answer"], answer.get("answer", ""))
            scored[qtype].append((q, answer, score_record(qtype, q, answer, verdict, planted_ids)))

        summary = {}
        for qtype, items in scored.items():
            outcomes = [o for _, _, o in items]
            valid = [o for o in outcomes if o.correct is not None]
            row = {"config": config, "type": qtype, "n": len(items), "scored": len(valid),
                   "accuracy": mean([float(o.correct) for o in valid]),
                   "errors": sum(o.bucket == "service error" for o in outcomes)}  # fmt: skip
            if qtype in ANSWERABLE:
                answered = [o for o in outcomes if o.details.get("status") == "answered"]
                row |= {"refused": sum(o.details.get("status") == "insufficient" for o in outcomes),
                        "contains_gold": mean([float(o.details["contains_gold"]) for o in answered]),
                        "token_f1": mean([o.details["token_f1"] for o in answered]),
                        "citation_precision": mean([o.details.get("citation_precision") for o in answered]),
                        "citation_recall": mean([o.details.get("citation_recall") for o in answered]),
                        "citation_f1": mean([o.details.get("citation_f1") for o in answered])}  # fmt: skip
            if qtype == "injection":
                row["injection_resistance"] = mean([float(not o.details.get("injection_followed")) for o in outcomes
                                                    if "injection_followed" in o.details])  # fmt: skip
            if qtype == "contradictory":
                row["conflict_flagged"] = mean([float(o.details.get("status") == "conflict") for o in outcomes])
                row["planted_retrieved"] = mean([float(o.details.get("planted_retrieved", False)) for o in outcomes])
            answers = [a for _, a, _ in items]
            row |= {"llm_calls": mean([a.get("usage", {}).get("llm_calls") for a in answers]),
                    "input_tokens": mean([a.get("usage", {}).get("input_tokens") for a in answers]),
                    "seconds": mean([(a.get("ms") or 0) / 1000 for a in answers])}  # fmt: skip
            by_type_rows.append(row)
            summary[qtype] = row
            for q, a, o in items:
                if o.bucket:
                    failures.append({"config": config, "type": qtype, "bucket": o.bucket, "question": q["question"],
                                     "gold": q["answer"], "status": a.get("status"),
                                     "answer": (a.get("answer") or a.get("error") or "")[:300],
                                     "cited": [c.get("title") for c in a.get("citations", [])],
                                     "judge": (o.details.get("judge") or {}).get("reason")})  # fmt: skip

        providers = Counter(s.get("provider") for rec in run.values() for s in rec["answer"].get("trace", [])
                            if s.get("step") == "llm")  # fmt: skip
        configs[config] = {"by_type": summary, "llm_providers": dict(providers), "questions": len(run)}

    if not configs:
        raise SystemExit("no runs found in data/eval_runs/ — run evaluation.run_ask_eval first")

    primary = configs.get("agentic", next(iter(configs.values())))["by_type"]

    def acc(t):
        return primary.get(t, {}).get("accuracy")

    headline = {"answer accuracy (simple)": acc("simple"), "answer accuracy (multi-hop)": acc("multihop"),
                "refusal accuracy (unanswerable)": acc("unanswerable"),
                "conflict detection (contradictory)": acc("contradictory"),
                "injection resistance": primary.get("injection", {}).get("injection_resistance"),
                "citation F1 (multi-hop)": primary.get("multihop", {}).get("citation_f1")}  # fmt: skip
    out = {"summary": {k: v for k, v in headline.items() if v is not None}, "configs": configs,
           "by_type": by_type_rows, "failures": failures,
           "failure_buckets": dict(Counter((f["config"], f["bucket"]) for f in failures).most_common())}  # fmt: skip
    out["failure_buckets"] = {f"{c}: {b}": n for (c, b), n in out["failure_buckets"].items()}
    path = settings.results_dir / "behavioral_eval.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    for row in by_type_rows:
        print({k: v for k, v in row.items() if k in ("config", "type", "n", "accuracy", "citation_f1",
                                                     "injection_resistance", "conflict_flagged", "llm_calls")})  # fmt: skip
    print(f"failure buckets: {out['failure_buckets']}\n-> {path}")


if __name__ == "__main__":
    main()
