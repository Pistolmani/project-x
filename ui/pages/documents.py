"""Page 1 — Documents (FRONTEND.md §2): upload, My uploads (update/delete), 50k Corpus browser, chunk viewer."""

import streamlit as st

from ui.client import BackendClient, BackendError, get_client


def render_documents_page() -> None:
    st.title("📄 Document Management")
    st.caption(
        "Chunking strategy: **fixed (deployed)** — 256-token windows with 32-token overlap (`bge` tokenizer). "
        "It tied recursive chunking on recall at 50k articles with fewer chunks; the **Evaluation** page has "
        "the measured comparison."
    )

    client = get_client()
    _render_upload_section(client)
    st.divider()

    tab_uploads, tab_corpus = st.tabs(["📁 My uploads", "📚 Corpus (50k Wikipedia)"])
    with tab_uploads:
        _render_uploads_tab(client)
    with tab_corpus:
        _render_corpus_tab(client)

    if st.session_state.get("selected_doc_id"):
        st.divider()
        _render_document_viewer(client, st.session_state["selected_doc_id"])


def _render_upload_section(client: BackendClient) -> None:
    st.subheader("Upload documents")
    files = st.file_uploader(
        "Upload PDF, HTML, Markdown or TXT files",
        type=["pdf", "html", "htm", "md", "markdown", "txt"],
        accept_multiple_files=True,
        key="doc_uploader",
    )
    if st.button("Ingest uploaded files", type="primary", disabled=not files, key="btn_ingest"):
        results = []
        for f in files or []:
            try:
                res = client.add_document(f.getvalue(), f.name)
                results.append({"name": f.name, "ok": True, **res})
            except BackendError as e:
                results.append({"name": f.name, "ok": False, "error": e.message})
        st.session_state["last_upload_results"] = results

    for item in st.session_state.get("last_upload_results", []):
        if not item.get("ok"):
            st.error(f"**{item['name']}** — error: {item['error']}")
        elif item["status"] == "duplicate":
            st.info(
                f"**{item['name']}** — already indexed as *{item['title']}* "
                f"(`{item['doc_id']}`, v{item['version']})."
            )
        else:
            n_chunks = sum((item.get("num_chunks") or {}).values())
            st.success(
                f"**{item['name']}** — created *{item['title']}* (`{item['doc_id']}`, v{item['version']}, "
                f"{n_chunks} chunks)."
            )


def _render_uploads_tab(client: BackendClient) -> None:
    try:
        resp = client.list_documents(limit=100, exclude_format="wikipedia")
    except BackendError as e:
        st.error(f"Could not load uploaded documents: {e.message}")
        return

    docs = resp.get("documents", [])
    if not docs:
        st.info("No uploaded documents yet. Use the uploader above to add PDF, HTML, Markdown or TXT files.")
        return

    for doc in docs:
        doc_id = doc["doc_id"]
        n_chunks = doc.get("num_chunks_fixed") or doc.get("num_chunks_recursive") or 0
        updated = doc.get("updated_at") or doc.get("created_at") or "—"
        with st.container(border=True):
            cols = st.columns([3, 1, 1, 1, 2, 2])
            cols[0].markdown(f"**{doc.get('title', doc_id)}**  \n`{doc.get('filename') or doc_id}`")
            cols[1].caption(f"Format  \n`{doc.get('format', '—')}`")
            cols[2].caption(f"Version  \n`v{doc.get('version', 1)}`")
            cols[3].caption(f"Chunks  \n`{n_chunks}`")
            cols[4].caption(f"Updated  \n`{updated}`")
            with cols[5]:
                b1, b2, b3 = st.columns(3)
                if b1.button("View", key=f"view_up_{doc_id}"):
                    st.session_state["selected_doc_id"] = doc_id
                if b2.button("Update", key=f"upd_toggle_{doc_id}"):
                    st.session_state["updating_doc_id"] = (
                        None if st.session_state.get("updating_doc_id") == doc_id else doc_id
                    )
                if b3.button("Delete", key=f"del_ask_{doc_id}"):
                    st.session_state["confirm_delete_doc_id"] = doc_id

            if st.session_state.get("updating_doc_id") == doc_id:
                new_file = st.file_uploader(
                    f"Re-upload replacement for {doc.get('title', doc_id)}",
                    type=["pdf", "html", "htm", "md", "markdown", "txt"],
                    key=f"upd_file_{doc_id}",
                )
                if st.button("Confirm update", key=f"upd_confirm_{doc_id}", disabled=new_file is None):
                    try:
                        res = client.update_document(doc_id, new_file.getvalue(), new_file.name)
                        st.session_state["updating_doc_id"] = None
                        if res["status"] == "duplicate":
                            st.info(f"Unchanged content — stayed at v{res['version']}.")
                        else:
                            st.success(f"Updated *{res['title']}* to v{res['version']}.")
                        st.rerun()
                    except BackendError as e:
                        st.error(f"Update failed: {e.message}")

            if st.session_state.get("confirm_delete_doc_id") == doc_id:
                st.warning(f"Delete **{doc.get('title', doc_id)}** and all of its chunks?")
                c1, c2, _ = st.columns([1, 1, 4])
                if c1.button("Yes, delete", type="primary", key=f"del_confirm_{doc_id}"):
                    try:
                        client.delete_document(doc_id)
                        st.session_state["confirm_delete_doc_id"] = None
                        if st.session_state.get("selected_doc_id") == doc_id:
                            st.session_state["selected_doc_id"] = None
                        st.success(f"Deleted {doc.get('title', doc_id)}.")
                        st.rerun()
                    except BackendError as e:
                        st.error(f"Delete failed: {e.message}")
                if c2.button("Cancel", key=f"del_cancel_{doc_id}"):
                    st.session_state["confirm_delete_doc_id"] = None
                    st.rerun()


def _render_corpus_tab(client: BackendClient) -> None:
    offsets: list[str | None] = st.session_state.setdefault("corpus_offsets", [None])
    page_idx: int = st.session_state.setdefault("corpus_page", 0)
    current_offset = offsets[page_idx] if page_idx < len(offsets) else None

    try:
        resp = client.list_documents(limit=25, offset=current_offset, fmt="wikipedia")
    except BackendError as e:
        st.error(f"Could not load corpus page: {e.message}")
        return

    docs = resp.get("documents", [])
    next_offset = resp.get("next_offset")

    nav1, nav2, nav3 = st.columns([1, 1, 4])
    if nav1.button("◀ Previous", disabled=page_idx == 0, key="corpus_prev"):
        st.session_state["corpus_page"] = max(0, page_idx - 1)
        st.rerun()
    if nav2.button("Next ▶", disabled=next_offset is None, key="corpus_next"):
        if page_idx + 1 >= len(offsets):
            offsets.append(next_offset)
        st.session_state["corpus_page"] = page_idx + 1
        st.rerun()
    nav3.caption(f"Page {page_idx + 1} · Read-only 50k Wikipedia corpus")

    if not docs:
        st.info("No Wikipedia corpus articles found in this collection.")
        return

    for doc in docs:
        doc_id = doc["doc_id"]
        n_chunks = doc.get("num_chunks_fixed") or doc.get("num_chunks_recursive") or 0
        cols = st.columns([4, 1, 2, 1])
        cols[0].markdown(f"**{doc.get('title', doc_id)}**")
        cols[1].caption(f"{n_chunks} chunks")
        cols[2].caption(f"`{doc_id}`")
        if cols[3].button("View chunks", key=f"view_corp_{doc_id}"):
            st.session_state["selected_doc_id"] = doc_id


def _render_document_viewer(client: BackendClient, doc_id: str) -> None:
    header_col, close_col = st.columns([6, 1])
    if close_col.button("Close ✕", key="close_doc_viewer"):
        st.session_state["selected_doc_id"] = None
        st.rerun()

    try:
        detail = client.get_document(doc_id, strategy="fixed")
    except BackendError as e:
        header_col.subheader("Document view")
        st.error(f"Could not fetch document `{doc_id}`: {e.message}")
        return

    doc = detail.get("document", {})
    chunks = detail.get("chunks", [])
    header_col.subheader(f"📖 {doc.get('title', doc_id)}")
    st.caption(
        f"`doc_id={doc_id}` · format=`{doc.get('format', '—')}` · version=`v{doc.get('version', 1)}` · "
        f"{len(chunks)} chunks (`fixed`)"
    )
    for ch in chunks:
        idx = ch.get("chunk_index", 0)
        tokens = ch.get("n_tokens", "—")
        span = f"chars {ch.get('start_char', 0)}–{ch.get('end_char', 0)}"
        with st.expander(f"Chunk #{idx} ({tokens} tokens, {span})", expanded=(idx == 0)):
            st.write(ch.get("text", ""))
