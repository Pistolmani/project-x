"""Page 3 — Evaluation dashboard (FRONTEND.md §4): loads tracked evaluation results from `GET /eval/results`.

Does not trigger the 250+ LLM-call evaluation suite live from the browser (which would exhaust the demo's
free-tier quota); instead renders the measured retrieval comparisons, weight-tuning ablations, corpus
sizing and any behavioral `/ask` evaluation results present in `results/`.
"""

from typing import Any

import pandas as pd
import streamlit as st

from ui.client import BackendError, get_client


def render_evaluation_page() -> None:
    st.title("📊 Evaluation Dashboard")
    st.caption(
        "Loads measured evaluation artifacts from `GET /eval/results` (`results/`). "
        "The evaluation runs offline against the same FastAPI endpoints so live demo quota is preserved."
    )

    client = get_client()
    try:
        data = client.eval_results()
    except BackendError as e:
        st.error(f"Could not load evaluation results from backend: {e.message}")
        return

    if not data:
        st.info("No evaluation result files found in `results/` yet.")
        return

    _render_retrieval_comparison(data.get("retrieval_comparison"), data.get("index_stats"))
    st.divider()
    _render_weight_tuning(data.get("tuned_weights_fixed"), data.get("tuned_weights_recursive"))
    st.divider()
    _render_behavioral_eval(data.get("behavioral_eval") or data.get("ask_eval"))
    if data.get("guardrails"):
        st.divider()
        _render_guardrails(data["guardrails"])
    if data.get("benchmark"):
        st.divider()
        _render_benchmark(data["benchmark"])
    st.divider()
    _render_live_metrics(client)


def _render_retrieval_comparison(comp: dict[str, Any] | None, stats: dict[str, Any] | None) -> None:
    st.subheader("1. Retrieval Pipeline & Chunking Strategy Comparison (50k Wikipedia articles)")
    if not comp:
        st.info("`retrieval_comparison.json` not found.")
        return

    deployed = comp.get("deployed_strategy", "fixed")
    chunk_counts = {name: s.get("chunks_in_collection") for name, s in (stats or {}).get("strategies", {}).items()}
    if stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Corpus articles", f"{stats.get('documents', 0):,}")
        m2.metric("Fixed chunks", f"{chunk_counts.get('fixed') or 0:,}")
        m3.metric("Recursive chunks", f"{chunk_counts.get('recursive') or 0:,}")
        m4.metric("Deployed strategy", deployed.upper())

    strategies = comp.get("strategies", {})
    rows = []
    for strat_name, cfg_list in strategies.items():
        for row in cfg_list:
            rows.append({"strategy": strat_name, **row})

    if not rows:
        return

    df = pd.DataFrame(rows)
    col_table, col_chart = st.columns([3, 2])
    with col_table:
        st.markdown("**Document-level Recall@K, MRR & Gold-Sentence Recall@10**")
        st.dataframe(df, width="stretch", hide_index=True)

    with col_chart:
        st.markdown("**Recall@10 across pipeline stages (fixed vs. recursive)**")
        pivot = df.pivot(index="config", columns="strategy", values="R@10")
        st.bar_chart(pivot)

    st.caption(_retrieval_findings(df, chunk_counts, deployed))


def _render_weight_tuning(fixed: dict[str, Any] | None, recursive: dict[str, Any] | None) -> None:
    st.subheader("2. Custom RRF & Feature Re-ranker Ablation (100 tuning vs. 300 held-out questions)")
    if not fixed and not recursive:
        st.info("Weight tuning results not found.")
        return

    rows = []
    for label, payload in [("fixed", fixed), ("recursive", recursive)]:
        if not payload:
            continue
        for cfg_name, metrics in (payload.get("results") or {}).items():
            t_r20, t_mrr = metrics.get("tuning", [None, None])
            h_r20, h_mrr = metrics.get("heldout", [None, None])
            rows.append(
                {
                    "strategy": label,
                    "config": cfg_name,
                    "tuning R@20": round(t_r20, 3) if t_r20 is not None else None,
                    "tuning MRR": round(t_mrr, 3) if t_mrr is not None else None,
                    "held-out R@20": round(h_r20, 3) if h_r20 is not None else None,
                    "held-out MRR": round(h_mrr, 3) if h_mrr is not None else None,
                }
            )

    df = pd.DataFrame(rows)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(df, width="stretch", hide_index=True)
    with c2:
        st.markdown("**Held-out R@20 (300 unseen questions)**")
        pivot = df.pivot(index="config", columns="strategy", values="held-out R@20")
        st.bar_chart(pivot)

    st.caption(_tuning_findings(df))


def _metric(df: pd.DataFrame, strategy: str, config: str, column: str) -> float | None:
    hit = df[(df["strategy"] == strategy) & (df["config"] == config)]
    return float(hit[column].iloc[0]) if len(hit) and pd.notna(hit[column].iloc[0]) else None


def _retrieval_findings(df: pd.DataFrame, chunk_counts: dict[str, int | None], deployed: str) -> str:
    """Key findings written from the loaded numbers, so they cannot go stale when the eval is re-run."""
    parts = []
    stages = [("dense only", "dense-only"), ("hybrid (RRF)", "hybrid"), ("full pipeline", "full pipeline")]
    for strategy in sorted(df["strategy"].unique()):
        steps = [f"{label} `{v:.3f}`" for config, label in stages
                 if (v := _metric(df, strategy, config, "R@10")) is not None]
        if len(steps) >= 2:
            parts.append(f"{strategy}: " + " → ".join(steps))
    text = "**Key findings (R@10):** " + "; ".join(parts) + "." if parts else ""
    fixed, recursive = chunk_counts.get("fixed"), chunk_counts.get("recursive")
    if fixed and recursive:
        text += (f" **{deployed}** is deployed; fixed needs {fixed:,} chunks vs. {recursive:,} recursive "
                 f"({1 - fixed / recursive:.0%} fewer).")
    return text


def _tuning_findings(df: pd.DataFrame) -> str:
    parts = []
    for strategy in sorted(df["strategy"].unique()):
        rrf = _metric(df, strategy, "RRF only (no features)", "held-out R@20")
        default = _metric(df, strategy, "default weights", "held-out R@20")
        tuned = _metric(df, strategy, "tuned weights", "held-out R@20")
        if None not in (rrf, default, tuned):
            parts.append(f"{strategy}: RRF only `{rrf:.3f}`, default features `{default:.3f}`, tuned `{tuned:.3f}`")
    if not parts:
        return ""
    return ("**Held-out R@20:** " + "; ".join(parts) + ". The weights in use are the defaults: tuning on 100 "
            "questions did not beat them on the 300 held-out questions.")


def _render_behavioral_eval(beh: dict[str, Any] | None) -> None:
    st.subheader("3. End-to-End `/ask` Behavioral Evaluation & Failure Modes")
    if not beh:
        st.info(
            "`/ask` evaluation results (`results/behavioral_eval.json`) will appear here once the "
            "offline 50-question evaluation completes: Answer Correctness, Citation Precision/Recall/F1, "
            "Refusal Accuracy (unanswerable), Conflict-Detection Accuracy (contradictory), and the Failure-Mode Table."
        )
        return

    summary = beh.get("summary") or {}
    if summary:
        cols = st.columns(min(4, len(summary)))
        for col, (k, v) in zip(cols, summary.items()):
            col.metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))

    by_type = beh.get("by_type")
    if isinstance(by_type, list) and by_type:
        df = pd.DataFrame(by_type)
        st.markdown("**Metrics by question type**")
        st.dataframe(df, width="stretch", hide_index=True)
        if {"config", "type", "accuracy"} <= set(df.columns) and df["config"].nunique() > 1:
            st.markdown("**Accuracy by question type and configuration**")
            st.bar_chart(df.pivot(index="type", columns="config", values="accuracy"))

    buckets = beh.get("failure_buckets")
    if isinstance(buckets, dict) and buckets:
        st.markdown("**Failure buckets**")
        st.dataframe(pd.DataFrame([{"config: bucket": k, "count": v} for k, v in buckets.items()]),
                     width="stretch", hide_index=True)

    failures = beh.get("failures")
    if isinstance(failures, list) and failures:
        st.markdown("**Failure-mode table**")
        st.dataframe(pd.DataFrame(failures), width="stretch", hide_index=True)


def _render_live_metrics(client) -> None:
    """`GET /metrics`: what this running backend has served since it started (resets when the app sleeps)."""
    st.subheader("Live service metrics (since the backend started)")
    try:
        m = client.metrics()
    except BackendError as e:
        st.info(f"Metrics unavailable: {e.message}")
        return
    answers, guard = m.get("answers") or {}, m.get("guard") or {}
    cols = st.columns(4)
    cols[0].metric("Uptime", f"{(m.get('uptime_s') or 0) // 60} min")
    cols[1].metric("Questions answered", answers.get("total", 0))
    p50, p95 = answers.get("p50_ms"), answers.get("p95_ms")
    cols[2].metric("Answer time p50 / p95", f"{p50 / 1000:.0f} s / {p95 / 1000:.0f} s" if p50 is not None else "—")
    cols[3].metric("Guardrail blocks", sum((guard.get("blocked") or {}).values()))
    if answers.get("by_status"):
        st.caption("Outcomes: " + " · ".join(f"{k} {v}" for k, v in sorted(answers["by_status"].items()))
                   + " · LLM calls: " + (" · ".join(f"`{k}` {v}" for k, v in (answers.get("llm_calls") or {}).items()) or "none"))
    routes = [{"route": r, **v} for r, v in (m.get("requests") or {}).items()]
    if routes:
        st.dataframe(pd.DataFrame(routes), hide_index=True, width="stretch")
    cache = [{"cache": ns, **v} for ns, v in (m.get("cache") or {}).items()]
    if cache:
        st.dataframe(pd.DataFrame(cache), hide_index=True, width="stretch")


def _render_guardrails(g: dict[str, Any]) -> None:
    """`results/guardrails.json` (evaluation/red_team.py)."""
    st.subheader("4. Guardrails red team (prompt injection, jailbreaks, harmful requests)")
    summary = g.get("summary") or {}
    off, on = summary.get("guard_off") or {}, summary.get("guard_on") or {}
    fp = g.get("false_positives") or {}
    cols = st.columns(3)
    cols[0].metric("Attacks succeeded, guards off", f"{off.get('succeeded', 0)} / {off.get('attacks', 0)}")
    cols[1].metric("Attacks succeeded, guards on", f"{on.get('succeeded', 0)} / {on.get('attacks', 0)}")
    cols[2].metric("Benign questions blocked", f"{len(fp.get('questions_blocked') or [])} / {fp.get('questions_checked', 0)}")
    rows = []
    for cat in sorted(set(off.get("by_category", {})) | set(on.get("by_category", {}))):
        o, n = off.get("by_category", {}).get(cat, {}), on.get("by_category", {}).get(cat, {})
        rows.append({"category": cat, "attacks": n.get("attacks", o.get("attacks", 0)),
                     "succeeded (off)": o.get("succeeded", 0), "succeeded (on)": n.get("succeeded", 0),
                     "stopped by input guard (on)": n.get("input guard", 0),
                     "stopped by output guard (on)": n.get("output guard", 0)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Policy classifier `{(g.get('guard_models') or {}).get('policy_classifier')}`; false positives also "
               f"checked on {fp.get('answers_checked', 0)} stored answers "
               f"({len(fp.get('answers_blocked') or [])} blocked). Details: EVALUATION-REPORT.md §8.")


def _render_benchmark(b: dict[str, Any]) -> None:
    """`results/benchmark.json` (evaluation/benchmark.py)."""
    index = b.get("index") or {}
    st.subheader(f"5. Search latency on the full index ({index.get('chunks', 0):,} chunks)")
    rows = []
    for run in b.get("search") or []:
        for stage, v in (run.get("ms") or {}).items():
            rows.append({"setup": f"{run['backend']} ({run['device']})", "stage": stage, "p50 ms": v.get("p50"),
                         "p95 ms": v.get("p95")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    emb = " · ".join(f"{e['backend']}: {e['chunks_per_s']} chunks/s" for e in b.get("embedding") or [])
    if emb:
        st.caption(f"Embedding throughput (dense + BM25, the main cost of ingesting an upload): {emb}")
