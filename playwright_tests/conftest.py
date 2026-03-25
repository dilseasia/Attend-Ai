"""Auto-generated conftest — Playwright API testing fixtures."""
import pytest
from playwright.sync_api import sync_playwright


BASE_URL = 'http://10.8.21.51:8000/api'


class _WrappedContext:
    """Thin wrapper that ensures every path is resolved against BASE_URL."""

    def __init__(self, ctx):
        self._ctx = ctx

    def _url(self, path):
        path = str(path)
        if path.startswith(("http://", "https://")):
            return path
        return BASE_URL + "/" + path.lstrip("/")

    def get(self, url, **kw):
        return self._ctx.get(self._url(url), **kw)

    def post(self, url, **kw):
        if "json" in kw:
            kw["data"] = kw.pop("json")
        return self._ctx.post(self._url(url), **kw)

    def put(self, url, **kw):
        if "json" in kw:
            kw["data"] = kw.pop("json")
        return self._ctx.put(self._url(url), **kw)

    def patch(self, url, **kw):
        if "json" in kw:
            kw["data"] = kw.pop("json")
        return self._ctx.patch(self._url(url), **kw)

    def delete(self, url, **kw):
        return self._ctx.delete(self._url(url), **kw)

    def head(self, url, **kw):
        return self._ctx.head(self._url(url), **kw)

    def dispose(self):
        self._ctx.dispose()


@pytest.fixture(scope="session")
def qa_base_url():
    """Return the QA base URL string."""
    return BASE_URL


@pytest.fixture(scope="session")
def pw():
    """Session-scoped Playwright instance (sync)."""
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def api_request_context(pw):
    """Session-scoped API request context pre-configured with the base URL.

    Transparently handles:
      - Leading-slash paths (e.g. "/auth/me" correctly maps to BASE_URL + "/auth/me")
      - `json=` keyword (auto-converted to `data=` for Playwright compatibility)
    """
    raw_ctx = pw.request.new_context()
    wrapped = _WrappedContext(raw_ctx)
    yield wrapped
    raw_ctx.dispose()
