"""HTTP client for the Palo Alto Cortex documentation API."""

from __future__ import annotations

from collections.abc import Iterable
import json
import logging
from typing import Any
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DocsClient",
    "DocsClientError",
    "DocHit",
    "DocSection",
]

logger = logging.getLogger(__name__)


class DocsClientError(Exception):
    """Raised when the Cortex documentation API call fails."""


class DocHit(BaseModel):
    """Structured representation of a clustered search TOPIC result."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    map_id: str = Field(alias="mapId")
    toc_id: str = Field(alias="tocId")
    content_id: str = Field(alias="contentId")
    title: str
    topic_url: str = Field(alias="topicUrl")
    content_url: str = Field(alias="contentUrl")
    reader_url: str = Field(alias="readerUrl")


class DocSection(BaseModel):
    """Documentation section content returned from the API."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    section_id: str
    title: str | None = None
    body: Any
    reader_url: str | None = Field(default=None, alias="readerUrl")


class DocsClient:
    """Reusable async client that wraps the Cortex documentation API."""

    _SEARCH_ENDPOINT = "api/khub/clustered-search"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    async def search(self, query: str, limit: int = 5) -> list[DocHit]:
        """Run the clustered search endpoint and return TOPIC hits only."""

        body = {
            "query": query,
            "metadataFilters": [],
            "paging": {"page": 1, "perPage": max(1, limit)},
            "contentLocale": "en-US",
        }
        payload = await self._request_json("POST", self._SEARCH_ENDPOINT, json=body)
        results_field = payload.get("results")
        if not isinstance(results_field, Iterable):
            raise DocsClientError("Docs API returned an unexpected search payload.")

        hits: list[DocHit] = []
        for cluster in results_field:
            entries = cluster.get("entries") if isinstance(cluster, dict) else None
            if not isinstance(entries, Iterable):
                continue
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("type") != "TOPIC":
                    continue
                topic = entry.get("topic")
                if not isinstance(topic, dict):
                    continue
                try:
                    hits.append(DocHit.model_validate(topic))
                except Exception as exc:  # pragma: no cover - defensive
                    raise DocsClientError(
                        "Docs API returned an invalid TOPIC payload."
                    ) from exc
                if len(hits) >= limit:
                    logger.debug("Docs search completed", extra={"query": query, "hits": len(hits)})
                    return hits
        logger.debug("Docs search completed", extra={"query": query, "hits": len(hits)})
        return hits

    async def fetch_section(self, section_id: str) -> DocSection:
        """Retrieve the body of a section given its content URL."""

        if not section_id:
            raise DocsClientError("Section identifier must be provided.")

        resolved_url = self._resolve_section_url(section_id)
        response = await self._send_request("GET", resolved_url)

        section_data: dict[str, Any] | None = None
        try:
            section_data = response.json()
        except ValueError:
            body_text = response.text.strip()
            if not body_text:
                raise DocsClientError("Docs API returned an empty section body.")
            section = DocSection(
                section_id=section_id,
                title=None,
                body=body_text,
                reader_url=None,
            )
            logger.debug(
                "Docs section fetched (HTML body)",
                extra={"section_id": section_id, "body_length": len(body_text)},
            )
            return section

        title = _first_not_none(
            section_data.get("title"),
            section_data.get("topic", {}).get("title")
            if isinstance(section_data.get("topic"), dict)
            else None,
        )
        body = section_data.get("body", section_data)
        reader_url = section_data.get("readerUrl")
        section = DocSection(
            section_id=section_id,
            title=title,
            body=body,
            reader_url=reader_url,
        )
        logger.debug("Docs section fetched", extra={"section_id": section_id})
        return section

    def _resolve_section_url(self, section_id: str) -> str:
        """Return an absolute URL for the content endpoint."""

        if section_id.startswith("http://") or section_id.startswith("https://"):
            return section_id
        section_path = section_id.lstrip("/")
        return urljoin(self._base_url, section_path)

    async def _request_json(
        self,
        method: str,
        url_or_path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request and return the decoded JSON payload."""

        url = (
            url_or_path
            if url_or_path.startswith("http://") or url_or_path.startswith("https://")
            else urljoin(self._base_url, url_or_path.lstrip("/"))
        )

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = await self._send_request(
            method,
            url,
            json=json,
            params=params,
            headers=headers,
        )

        try:
            payload = response.json()
        except ValueError:
            payload = self._extract_json_from_html(response.text)

        if not isinstance(payload, dict):
            raise DocsClientError("Docs API returned an unexpected payload.")
        return payload

    async def _send_request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Send an HTTP request and return the response with shared error handling."""

        headers = headers or {}
        if self._api_key:
            headers.setdefault("Authorization", f"Bearer {self._api_key}")

        client = self._client
        try:
            if client is not None:
                response = await client.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as session:
                    response = await session.request(
                        method,
                        url,
                        json=json,
                        params=params,
                        headers=headers,
                    )
        except httpx.RequestError as exc:
            raise DocsClientError(f"Failed to call docs API: {exc!s}") from exc

        if response.status_code >= 400:
            raise DocsClientError(
                f"Docs API error {response.status_code}: {response.text.strip()}"
            )
        return response

    @staticmethod
    def _extract_json_from_html(raw_text: str) -> dict[str, Any]:
        """Attempt to extract JSON from HTML-wrapped responses."""

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            snippet = raw_text[:200].strip()
            raise DocsClientError(
                f"Docs API returned invalid JSON. Payload starts with: {snippet}"
            )
        try:
            parsed = json.loads(raw_text[start : end + 1])
        except ValueError as exc:  # pragma: no cover - defensive
            snippet = raw_text[:200].strip()
            raise DocsClientError(
                f"Docs API returned invalid JSON. Payload starts with: {snippet}"
            ) from exc
        if not isinstance(parsed, dict):
            raise DocsClientError("Docs API returned an unexpected payload.")
        return parsed


def _first_not_none(*values: Any) -> Any:
    """Return the first non-None value from the provided sequence."""

    for value in values:
        if value is not None:
            return value
    return None
