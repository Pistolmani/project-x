"""Streamlit UI entrypoint (FRONTEND.md §1): three pages via `st.navigation`.

    streamlit run ui/streamlit_app.py

Not named app.py on purpose: `streamlit run` puts ui/ first on sys.path, so a ui/app.py would shadow the
backend's `app` package.

Talks to the FastAPI backend at `API_URL` (default `http://localhost:8000`). Holds no business logic.
With EMBEDDED_API=true it first starts that backend in-process (ui/embedded_api.py; Streamlit Cloud).
"""

import os
import sys
from pathlib import Path

import streamlit as st

# `streamlit run` puts only ui/ on sys.path; the repo root is needed for the `ui` and `app` packages.
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.client import BackendError, get_client
from ui.pages.ask import render_ask_page
from ui.pages.documents import render_documents_page
from ui.pages.evaluation import render_evaluation_page

st.set_page_config(page_title="AI Investigation Engine", page_icon="🔎", layout="wide")

from ui.embedded_api import ensure_backend, export_secrets

export_secrets(st.secrets)
if os.environ.get("EMBEDDED_API", "").lower() == "true":
    ensure_backend()


def _sidebar_status() -> None:
    client = get_client()
    with st.sidebar:
        st.caption(f"Backend: `{client.base_url}`")
        try:
            info = client.health()
            status = info.get("status", "unknown")
            docs = info.get("documents")
            providers = ", ".join(info.get("llm_providers") or []) or "none"
            badge = "🟢 ok" if status == "ok" else "🟠 degraded"
            st.caption(f"Status: {badge} · Docs: `{docs}` · LLMs: `{providers}`")
        except BackendError as e:
            st.warning(f"Backend unreachable: {e.message}")


_sidebar_status()

pg = st.navigation(
    [
        st.Page(render_ask_page, title="Ask & Investigate", icon="🔎", url_path="ask", default=True),
        st.Page(render_documents_page, title="Documents", icon="📄", url_path="documents"),
        st.Page(render_evaluation_page, title="Evaluation", icon="📊", url_path="evaluation"),
    ]
)
pg.run()
