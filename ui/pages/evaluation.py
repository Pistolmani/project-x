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


def _render_retrieval_comparison(comp: dict[str, Any] | None, stats: dict[str, Any] | None) -> None:
    st.subheader("1. Retrieval Pipeline & Chunking Strategy Comparison (50k Wikipedia articles)")
    if not comp:
        st.info("`retrieval_comparison.json` not found.")
        return

    if stats:
        fixed_chunks = stats.get("strategies", {}).get("fixed", {}).get("chunks_in_collection", 300059)
        rec_chunks = stats.get("strategies", {}).get("recursive", {}).get("chunks_in_collection", 433425)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Corpus articles", f"{stats.get('documents', 50000):,}")
        m2.metric("Fixed chunks (deployed)", f"{fixed_chunks:,}")
        m3.metric("Recursive chunks", f"{rec_chunks:,}", delta="-30.8% storage with fixed")
        m4.metric("Deployed strategy", comp.get("deployed_strategy", "fixed").upper())

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
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown("**Recall@10 across pipeline stages (fixed vs. recursive)**")
        pivot = df.pivot(index="config", columns="strategy", values="R@10")
        st.bar_chart(pivot)

    st.caption(
        "**Key findings:** (1) Hybrid search (BM25 + dense fused by custom weighted RRF) beats dense-only on both "
        "strategies, with the largest jump on fixed windows (`0.765 → 0.885` R@10). "
        "(2) Fixed and recursive chunking tie at the top of the funnel (`0.915` vs `0.900` R@10; `0.953` vs `0.923` MRR), "
        "so **fixed is deployed** to Qdrant Cloud for ~30% lower vector storage."
    )


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
        st.dataframe(df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Held-out R@20 (300 unseen questions)**")
        pivot = df.pivot(index="config", columns="strategy", values="held-out R@20")
        st.bar_chart(pivot)

    st.caption(
        "**Key finding:** The custom feature re-ranker (default weights) improves held-out Recall@20 and MRR over plain "
        "RRF on both strategies (`0.908 → 0.922` fixed, `0.922 → 0.938` recursive). Coordinate-ascent tuning on 100 "
        "questions overfit (recursive set `w_sparse=0`), so the **default weights are kept**."
    )


def _render_behavioral_eval(beh: dict[str, Any] | None) -> None:
    st.subheader("3. End-to-End `/ask` Behavioral Evaluation & Failure Modes")
    if not beh:
        st.info(
            "Phase 4 `/ask` evaluation results (`results/behavioral_eval.json`) will appear here once the "
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
        st.markdown("**Metrics by question type**")
        st.dataframe(pd.DataFrame(by_type), use_container_width=True, hide_index=True)

    failures = beh.get("failures")
    if isinstance(failures, list) and failures:
        st.markdown("**Failure-mode table**")
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
