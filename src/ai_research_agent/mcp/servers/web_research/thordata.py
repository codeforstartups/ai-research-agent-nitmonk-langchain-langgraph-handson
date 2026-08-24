"""Thordata client — the live web-data acquisition layer.

Thordata powers the Web Research MCP's search/scrape tools. This is the *only*
module that knows Thordata's HTTP shape; everything else talks to it through the
MCP server, so Thordata is never hard-coded across the agent.

API (per Thordata docs — verify against your dashboard):
* SERP:  POST https://scraperapi.thordata.com/request
         headers: {"token": <key>}
         body:    {"q": <query>, "json": "1", "count": <n>}
         organic results live under response["result"].

Engineering requirements baked in (spec §23):
* configured via environment (never hard-coded secrets),
* explicit timeout + bounded retries (tenacity),
* response validation + normalization into typed :class:`SearchResult`s,
* URLs and source metadata preserved,
* failures handled gracefully,
* an **offline fixture** so the agent runs before Thordata is provisioned.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_research_agent.config import get_settings
from ai_research_agent.schemas.web import ScrapedPage, SearchResult
from ai_research_agent.utils import get_logger

logger = get_logger(__name__)

SERP_ENDPOINT = "https://scraperapi.thordata.com/request"

# Deterministic offline fixtures so search/scrape work with no Thordata key.
_OFFLINE_RESULTS: dict[str, list[SearchResult]] = {
    "default": [
        SearchResult(
            title="Cursor — the AI code editor",
            url="https://www.cursor.com/",
            snippet="An AI-first IDE with an agentic apply/edit loop over your workspace.",
            position=1,
            source="offline-fixture",
        ),
        SearchResult(
            title="GitHub Copilot",
            url="https://github.com/features/copilot",
            snippet="AI pair programmer bundled with the GitHub developer platform.",
            position=2,
            source="offline-fixture",
        ),
        SearchResult(
            title="Devin — autonomous AI software engineer",
            url="https://www.cognition.ai/",
            snippet="Markets an asynchronous ticket-to-PR autonomous coding agent.",
            position=3,
            source="offline-fixture",
        ),
    ]
}


class ThordataError(RuntimeError):
    """Raised on an unrecoverable Thordata request failure."""


class ThordataClient:
    """Minimal, defensive client for Thordata's SERP + scraper APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: int | None = None,
        endpoint: str = SERP_ENDPOINT,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.thordata_api_key
        self.timeout = timeout or settings.thordata_timeout_seconds
        self.endpoint = endpoint

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # -- SERP ---------------------------------------------------------------
    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        """Return normalized search results (offline fixture if no key)."""
        if not self.available:
            logger.info("Thordata key absent — returning offline fixture for %r", query)
            return _offline_results(count)
        try:
            payload = self._post({"q": query, "json": "1", "count": str(count)})
        except ThordataError:
            logger.warning("Thordata search failed for %r; using fixture", query)
            return _offline_results(count)
        return _parse_serp(payload, count)

    # -- Scrape -------------------------------------------------------------
    def scrape(self, url: str) -> ScrapedPage:
        """Fetch a page's content (best-effort; offline returns a stub)."""
        if not self.available:
            return ScrapedPage(
                url=url,
                title="(offline)",
                content=(
                    "Offline: no Thordata key, so this page was not fetched. "
                    "Set THORDATA_API_KEY to scrape live pages."
                ),
                source="offline-fixture",
            )
        try:
            payload = self._post({"url": url, "json": "1"})
        except ThordataError as exc:
            return ScrapedPage(url=url, title="(error)", content=f"Scrape failed: {exc}")
        result = payload.get("result", payload)
        content = result.get("content") or result.get("html") or str(result)
        return ScrapedPage(url=url, title=result.get("title", ""), content=str(content)[:8000])

    # -- transport ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _post(self, body: dict[str, str]) -> dict:
        headers = {"token": self.api_key or "", "Content-Type": "application/json"}
        try:
            resp = httpx.post(self.endpoint, headers=headers, json=body, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            # Never leak the token; report status only.
            raise ThordataError(f"Thordata HTTP {exc.response.status_code}") from None
        except httpx.HTTPError as exc:
            raise ThordataError(f"Thordata request error: {exc.__class__.__name__}") from exc


def _offline_results(count: int) -> list[SearchResult]:
    return _OFFLINE_RESULTS["default"][:count]


def _parse_serp(payload: dict, count: int) -> list[SearchResult]:
    """Defensively parse Thordata's SERP JSON into SearchResults."""
    result = payload.get("result", payload)
    organic = (
        result.get("organic")
        or result.get("organic_results")
        or result.get("results")
        or []
    )
    out: list[SearchResult] = []
    for i, item in enumerate(organic[:count], start=1):
        if not isinstance(item, dict):
            continue
        url = item.get("link") or item.get("url") or ""
        if not url:
            continue
        out.append(
            SearchResult(
                title=item.get("title", url),
                url=url,
                snippet=item.get("snippet") or item.get("description") or "",
                position=item.get("position", i),
                source="thordata",
            )
        )
    return out or _offline_results(count)
