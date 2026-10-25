"""Runs the FastAPI backend inside the Streamlit process, for hosts that only run `streamlit run`
(Streamlit Community Cloud). Enabled with EMBEDDED_API=true.

The UI still reaches the backend over HTTP on 127.0.0.1, so it stays the same thin client it is when
the two run as separate processes (locally, or with the Dockerfile).
"""

import os
import threading
import time

import httpx
import streamlit as st

PORT = 8000
STARTUP_TIMEOUT_S = 300  # first start downloads ~1.2 GB of models, then loads them


def export_secrets(secrets) -> None:
    """Copy top-level Streamlit secrets into the environment, where the backend's settings read them.
    Streamlit does this itself only once something reads `st.secrets`; real environment variables win."""
    try:
        items = list(secrets.items())
    except Exception:  # no secrets configured (local runs): StreamlitSecretNotFoundError
        return
    for key, value in items:
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))


def start(app, port: int = PORT, timeout_s: float = STARTUP_TIMEOUT_S):
    """Serve `app` from a daemon thread and block until /health answers."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="embedded-api", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("the embedded backend exited during startup; see the app logs")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return server
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"the embedded backend did not answer within {timeout_s:.0f} s")


@st.cache_resource(show_spinner="Starting the backend: loading the search models (about a minute on a cold start)…")
def ensure_backend():
    """Once per Streamlit process, shared by all sessions."""
    from app.api.main import app

    os.environ["API_URL"] = f"http://127.0.0.1:{PORT}"  # not "localhost": it may resolve to ::1 first
    return start(app)
