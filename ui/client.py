"""HTTP client for the FastAPI backend (FRONTEND.md §1, §5).

The UI is a thin client over the same HTTP endpoints the evaluation suite drives. It turns network
errors, timeouts and structured backend error payloads into `BackendError`, and never fabricates an
answer the backend did not return.
"""

import json
import os
from typing import Any, Iterator

import httpx

from app.config import get_settings


class BackendError(RuntimeError):
    def __init__(self, message: str, code: str = "backend_error", status: int | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class BackendClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("API_URL") or get_settings().api_url).rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_documents(
        self,
        limit: int = 50,
        offset: str | None = None,
        fmt: str | None = None,
        exclude_format: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if offset is not None:
            params["offset"] = offset
        if fmt is not None:
            params["format"] = fmt
        if exclude_format is not None:
            params["exclude_format"] = exclude_format
        return self._request("GET", "/documents", params=params)

    def get_document(self, doc_id: str, strategy: str = "fixed") -> dict[str, Any]:
        return self._request("GET", f"/documents/{doc_id}", params={"strategy": strategy})

    def add_document(self, data: bytes, filename: str) -> dict[str, Any]:
        return self._request("POST", "/documents", files={"file": (filename, data)})

    def update_document(self, doc_id: str, data: bytes, filename: str) -> dict[str, Any]:
        return self._request("PUT", f"/documents/{doc_id}", files={"file": (filename, data)})

    def delete_document(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}")

    def search(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", "/search", json={"query": query, "params": params or {}})

    def ask(self, question: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", "/ask", json={"question": question, "params": params or {}}, allow_503_answer=True)

    def ask_stream(self, question: str, params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Yield SSE progress events (`type == "progress"`), then the final answer (`type == "answer"`)."""
        url = f"{self.base_url}/ask/stream"
        try:
            with httpx.Client(timeout=self.timeout) as http:
                with http.stream("POST", url, json={"question": question, "params": params or {}}) as r:
                    if r.status_code >= 400:
                        r.read()
                        raise _to_error(r)
                    for line in r.iter_lines():
                        if line.startswith("data: "):
                            yield json.loads(line.removeprefix("data: "))
        except httpx.TimeoutException as e:
            raise BackendError("The backend timed out while investigating; try again.", "timeout") from e
        except httpx.HTTPError as e:
            raise BackendError(f"Could not reach the backend at {self.base_url}: {e}", "unreachable") from e

    def eval_results(self) -> dict[str, Any]:
        return self._request("GET", "/eval/results")

    def _request(self, method: str, path: str, allow_503_answer: bool = False, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as http:
                r = http.request(method, url, **kwargs)
        except httpx.TimeoutException as e:
            raise BackendError("The backend timed out; please try again.", "timeout") from e
        except httpx.HTTPError as e:
            raise BackendError(f"Could not reach the backend at {self.base_url}: {e}", "unreachable") from e

        if r.status_code == 204:
            return None
        if allow_503_answer and r.status_code == 503:
            try:
                body = r.json()
                if isinstance(body, dict) and body.get("status") == "error":
                    return body
            except ValueError:
                pass
        if r.status_code >= 400:
            raise _to_error(r)
        return r.json()


def _to_error(r: httpx.Response) -> BackendError:
    try:
        body = r.json()
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
            return BackendError(err.get("message", r.text), err.get("code", "backend_error"), r.status_code)
        if isinstance(body, dict) and "detail" in body:
            return BackendError(str(body["detail"]), "validation_error", r.status_code)
    except ValueError:
        pass
    return BackendError(r.text or f"HTTP {r.status_code}", "http_error", r.status_code)


def get_client() -> BackendClient:
    """Return a fake/injected client from `st.session_state['_client']` if present (for AppTest),
    otherwise a default `BackendClient` pointing at `API_URL`."""
    import streamlit as st

    return st.session_state.get("_client") or BackendClient()
