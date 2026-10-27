"""Page 2 — Ask (FRONTEND.md §3): question box, sidebar config, live progress stream, answer/refusal/conflict
banners, source chunk viewer with parent-document view, and "How it got here" trace."""

from typing import Any

import streamlit as st

from ui.client import BackendClient, BackendError, get_client
from ui.passages import as_plain_markdown, excerpt, group_citations, label

EXAMPLES = [
    (
        "Simple",
        "For which academic field is Lionel Trilling, one of the leading US critics of the 20th century best known, "
        "and for which Rudolf Christoph Eucken received the 1908 Nobel Prize?",
    ),
    (
        "Multi-hop",
        "Grigori 3's music is commonly compared to an American rock band founded in what year?",
    ),
    (
        "Unanswerable",
        "Nelson Chai is the former president of what financial holding company that was founded in 1908, "
        "and is headquartered in New York City?",
    ),
]


def render_ask_page() -> None:
    st.title("🔎 Ask & Investigate")
    st.caption(
        "Ask a question over the document collection. The engine searches, cross-references, checks citations "
        "and relevance floor, and verifies groundedness and contradictions before answering."
    )

    client = get_client()
    params = _render_config_sidebar()

    st.markdown("**Try an example question:**")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (label, q_text) in zip(ex_cols, EXAMPLES):
        if col.button(f"{label}: {q_text[:48]}…", key=f"ex_{label}", width="stretch"):
            st.session_state["ask_question_input"] = q_text

    question = st.text_area(
        "Question",
        placeholder="Ask a simple or multi-hop question…",
        height=90,
        key="ask_question_input",
    )

    if st.button("Investigate", type="primary", disabled=not (question and question.strip()), key="btn_ask"):
        _run_investigation(client, question.strip(), params)

    answer = st.session_state.get("last_answer")
    if answer is not None:
        st.divider()
        _render_answer(client, answer, st.session_state.get("last_progress", []))


def _render_config_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("⚙️ Investigation Config")
        agentic = st.toggle("Agentic multi-step loop", value=True, key="cfg_agentic")
        mode = st.selectbox("Retrieval mode", options=["hybrid", "dense", "sparse"], index=0, key="cfg_mode")
        feature_rerank = st.toggle("Custom feature re-ranker", value=True, key="cfg_features")
        cross_encoder = st.toggle("Cross-encoder (`bge-reranker`)", value=True, key="cfg_ce")
        top_k = st.slider("Top-K passages per search", min_value=1, max_value=20, value=8, key="cfg_topk")
        use_floor = st.toggle("Relevance floor gate", value=True, key="cfg_use_floor", disabled=not cross_encoder)
        floor_val = st.slider(
            "Relevance floor (CE logit)",
            min_value=-5.0,
            max_value=10.0,
            value=3.92,
            step=0.05,
            disabled=not (use_floor and cross_encoder),
            key="cfg_floor_val",
        )
        verify = st.toggle("Verifier (groundedness + conflict)", value=True, key="cfg_verify")

    return {
        "agentic": agentic,
        "verify": verify,
        "relevance_floor": floor_val if (use_floor and cross_encoder) else None,
        "search": {
            "strategy": "fixed",
            "mode": mode,
            "feature_rerank": feature_rerank,
            "cross_encoder": cross_encoder,
            "top_k": top_k,
        },
    }


def _run_investigation(client: BackendClient, question: str, params: dict[str, Any]) -> None:
    progress_log: list[str] = []
    answer_payload: dict[str, Any] | None = None
    st.session_state["opened_parent_doc_id"] = None

    try:
        with st.status("Investigating…", expanded=True) as status_box:
            for event in client.ask_stream(question, params):
                if event.get("type") == "progress":
                    msg = event.get("message", "")
                    progress_log.append(msg)
                    st.write(f"• {msg}")
                elif event.get("type") == "answer":
                    answer_payload = event.get("answer")
            status_box.update(label="Investigation complete", state="complete", expanded=False)
    except BackendError as e:
        st.error(f"Investigation failed: {e.message}")
        return

    if answer_payload is None:
        st.error("Investigation failed: the backend closed the stream without an answer.")
        return
    st.session_state["last_answer"] = answer_payload
    st.session_state["last_progress"] = progress_log


def _render_answer(client: BackendClient, ans: dict[str, Any], progress_log: list[str]) -> None:
    status = ans.get("status", "error")
    text = ans.get("answer", "")
    citations = ans.get("citations") or []
    searched = ans.get("searched") or []

    if status == "answered":
        st.success("✅ **Answered (grounded in cited sources)**")
        st.markdown(text)
        if citations:
            chips = "  ".join(f"`[{', '.join(p.source_ids)}] {p.title}`" for p in group_citations(citations))
            st.caption(f"Cited sources: {chips}")
    elif status == "insufficient":
        st.warning("⚠️ **Not enough evidence — refused to answer**")
        st.markdown(text or "I don't have enough information in the documents to answer this.")
        if searched:
            st.markdown("**Queries searched:** " + ", ".join(f"`{q}`" for q in searched))
    elif status == "conflict":
        st.error("⚡ **Sources conflict — disagreement surfaced**")
        st.markdown(text)
        verdict = (ans.get("gates") or {}).get("verifier") or {}
        if verdict.get("explanation"):
            st.caption(f"Verifier note: {verdict['explanation']}")
    elif status == "blocked":
        st.error("🛡️ **Blocked by a guardrail**")
        st.markdown(text)
        guard = (ans.get("gates") or {}).get("guard") or {}
        stopped = next((g for g in (guard.get("input"), guard.get("output")) if g and not g.get("passed")), None)
        if stopped:
            st.caption(f"Reason ({stopped.get('category')}): {stopped.get('reason')}")
    else:
        st.error(f"❌ **Service error:** {ans.get('error') or 'The engine could not complete the request.'}")

    if citations:
        passages = group_citations(citations)
        merged = f" from {len(citations)} chunks" if len(passages) < len(citations) else ""
        st.subheader(f"Sources ({len(passages)} passage{'s' if len(passages) != 1 else ''}{merged})")
        for passage in passages:
            sid, title, doc_id = passage.source_ids[0], passage.title, passage.doc_id
            with st.expander(label(passage), expanded=True):
                if passage.suspicious:
                    st.warning(f"⚠️ This passage contains instruction-like text (\"{as_plain_markdown(passage.suspicious)}\"); "
                               "the model was told to treat it as data.")
                st.markdown(as_plain_markdown(excerpt(passage.text, passage.start_char)))
                if doc_id and st.button(
                    f"Open parent document ({title})", key=f"open_parent_{sid}_{doc_id}"
                ):
                    st.session_state["opened_parent_doc_id"] = doc_id
                    st.session_state["opened_parent_title"] = title

    parent_id = st.session_state.get("opened_parent_doc_id")
    if parent_id:
        _render_parent_document(client, parent_id)

    _render_trace(ans, progress_log)


def _render_parent_document(client: BackendClient, doc_id: str) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([6, 1])
        if c2.button("Close ✕", key="close_parent_doc"):
            st.session_state["opened_parent_doc_id"] = None
            st.rerun()
        try:
            detail = client.get_document(doc_id, strategy="fixed")
        except BackendError as e:
            c1.subheader("Parent document")
            st.error(f"Could not load document `{doc_id}`: {e.message}")
            return
        doc = detail.get("document", {})
        chunks = detail.get("chunks", [])
        c1.subheader(f"📖 Parent document: {doc.get('title', doc_id)}")
        st.caption(f"`doc_id={doc_id}` · {len(chunks)} chunks in reading order · neighbouring chunks overlap "
                   "by 32 tokens, so each one repeats the end of the previous")
        for ch in chunks:
            idx = ch.get("chunk_index", 0)
            st.markdown(f"**Chunk #{idx}** ({ch.get('n_tokens', '—')} tokens)")
            st.markdown(as_plain_markdown(ch.get("text", "")))


def _render_trace(ans: dict[str, Any], progress_log: list[str]) -> None:
    usage = ans.get("usage") or {}
    ms = ans.get("ms", 0)
    with st.expander(
        f"🔍 How it got here ({usage.get('llm_calls', 0)} LLM calls · "
        f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out tokens · {ms} ms)",
        expanded=False,
    ):
        if progress_log:
            st.markdown("**Live stream events:** " + " → ".join(f"`{m}`" for m in progress_log))

        searched = ans.get("searched") or []
        if searched:
            st.markdown("**Searches executed:**")
            for i, q in enumerate(searched, 1):
                st.markdown(f"{i}. `{q}`")

        trace = ans.get("trace") or []
        if trace:
            st.markdown("**Step-by-step execution trace:**")
            for i, step in enumerate(trace, 1):
                if step.get("step") == "llm":
                    calls = ", ".join(step.get("tool_calls") or []) or "none"
                    st.markdown(
                        f"- **Step {i} (LLM):** `{step.get('provider')}/{step.get('model')}` → "
                        f"called `{calls}` ({step.get('ms')} ms)"
                    )
                elif step.get("step") == "tool":
                    err = f" · error: `{step['error']}`" if step.get("error") else ""
                    st.markdown(
                        f"- **Step {i} (Tool `{step.get('tool')}`):** args `{step.get('args')}` → "
                        f"`{step.get('results', 0)}` passages ({step.get('ms')} ms){err}"
                    )

        gates = ans.get("gates") or {}
        guard = gates.get("guard") or {}
        if guard:
            st.markdown("**Guardrails:**")
            for side in ("input", "output"):
                if g := guard.get(side):
                    badge = "✅ passed" if g.get("passed") else f"🛡️ blocked ({g.get('category')})"
                    note = f" — {g['reason']}" if g.get("reason") else ""
                    fallback = f" · classifier unavailable: `{g['error']}`" if g.get("error") else ""
                    st.markdown(f"- **{side.capitalize()} guard** (`{', '.join(g.get('checks', []))}`): "
                                f"{badge}{note}{fallback}")
        if {k: v for k, v in gates.items() if k != "guard"}:
            st.markdown("**Anti-hallucination gates:**")
            cit = gates.get("citations")
            if cit is not None:
                badge = "✅ passed" if cit.get("passed") else "❌ failed"
                st.markdown(
                    f"- **1. Citation gate:** {badge} — valid `{cit.get('valid', [])}`, "
                    f"dropped invalid `{cit.get('invalid', [])}`"
                )
            ret = gates.get("retrieval")
            if ret is not None:
                badge = "✅ passed" if ret.get("passed") else "❌ failed (below relevance floor)"
                st.markdown(
                    f"- **2. Retrieval relevance-floor gate:** {badge} — best CE score "
                    f"`{ret.get('best_ce_score')}`, floor `{ret.get('floor')}`"
                )
            ver = gates.get("verifier")
            if ver is not None:
                supported = ver.get("supported")
                conflict = ver.get("conflict")
                if conflict:
                    badge = "⚡ conflict detected"
                elif supported:
                    badge = "✅ supported"
                elif supported is False:
                    badge = "❌ unsupported claims"
                else:
                    badge = "⚠️ verifier error"
                st.markdown(
                    f"- **3. Verifier gate (`{ver.get('provider', 'llm')}`):** {badge} — "
                    f"checked passages `{ver.get('passages', [])}`"
                )
                if ver.get("explanation"):
                    st.caption(f"Verifier explanation: {ver['explanation']}")
                if ver.get("unsupported_claims"):
                    st.caption(f"Unsupported claims: {ver['unsupported_claims']}")
            loop = gates.get("loop")
            if loop is not None:
                st.markdown(f"- **Loop guard:** ❌ `{loop.get('reason')}`")
