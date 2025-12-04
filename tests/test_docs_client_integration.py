"""Live integration tests for DocsClient against the real Cortex docs API."""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from cortex_mcp.domain.docs_client import DocHit, DocSection, DocsClient, DocsClientError

load_dotenv()

BASE_URL = os.getenv("CORTEX_DOCS_BASE_URL", "https://docs-cortex.paloaltonetworks.com")

RUN_INTEGRATION = os.getenv("CORTEX_DOCS_INTEGRATION") == "1"
if not RUN_INTEGRATION:  # pragma: no cover - intentionally skipped unless opted in.
    pytest.skip(
        "Set CORTEX_DOCS_INTEGRATION=1 to run docs integration tests",
        allow_module_level=True,
    )


@pytest_asyncio.fixture
async def client() -> AsyncIterator[DocsClient]:
    """Provide a DocsClient bound to the live API and ensure cleanup."""

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=15.0) as http_client:
        docs_client = DocsClient(
            base_url=BASE_URL,
            client=http_client,
        )
        yield docs_client


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_real_docs_returns_hits(client: DocsClient) -> None:
    """Real search should return TOPIC hits with populated fields."""

    hits = await _search_or_fail(client, "xql", limit=3)

    assert hits, "Expected at least one search hit"
    for hit in hits:
        assert isinstance(hit, DocHit)
        assert hit.title
        assert hit.content_url
        assert hit.reader_url

@pytest.mark.asyncio
@pytest.mark.integration
async def test_fetch_section_real_content(client: DocsClient) -> None:
    """Fetch a real section from the docs API and ensure basic shape.

    This test uses a generic, stable query ('xql') to reduce brittleness:
    we only care that at least one hit exists and that we can fetch its content.
    """

    # Use a generic query that is very likely to return something.
    hits = await _search_or_fail(client, "xql", limit=5)

    if not hits:
        pytest.skip("No hits for 'xql' on live docs; search behaviour has changed")

    section_id = hits[0].content_url
    section = await _fetch_section_or_fail(client, section_id)

    assert isinstance(section, DocSection)
    assert section.body, "Expected section body to be present"
    assert section.title is None or isinstance(section.title, str)



async def _search_or_fail(client: DocsClient, query: str, *, limit: int) -> list[DocHit]:
    """Run a search and fail loudly with API response details on error."""

    try:
        return await client.search(query, limit=limit)
    except DocsClientError as exc:  # pragma: no cover - network failure
        raise AssertionError(
            f"Docs API section fetch failed (base_url={BASE_URL}, section_id={section_id}): {exc}"
        ) from exc


async def _fetch_section_or_fail(client: DocsClient, section_id: str) -> DocSection:
    """Fetch a docs section and fail with context if it errors."""

    try:
        return await client.fetch_section(section_id)
    except DocsClientError as exc:  # pragma: no cover - network failure
        raise AssertionError(
            f"Docs API section fetch failed (base_url={BASE_URL}, section_id={section_id}): {exc}"
        ) from exc
