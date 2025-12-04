"""Tests for DocsClient."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import httpx
import pytest

from cortex_mcp.domain.docs_client import DocHit, DocSection, DocsClient, DocsClientError


def create_client(response_handler: httpx.MockTransport) -> DocsClient:
    """Helper to build a DocsClient with a mock transport."""

    client = httpx.AsyncClient(transport=response_handler, base_url="https://example.com/")
    return DocsClient(base_url="https://example.com", client=client)


@pytest.mark.asyncio
async def test_search_returns_topic_hits():
    """Successful search returns DocHit instances."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "results": [
                {
                    "entries": [
                        {
                            "type": "TOPIC",
                            "topic": {
                                "mapId": "map1",
                                "tocId": "toc1",
                                "contentId": "content1",
                                "title": "Title",
                                "topicUrl": "https://topic",
                                "contentUrl": "/content",
                                "readerUrl": "https://reader",
                            }
                        }
                    ]
                }
            ]
        }
        return httpx.Response(200, json=body)

    client = create_client(httpx.MockTransport(handler))
    hits = await client.search("query", limit=1)

    assert len(hits) == 1
    assert isinstance(hits[0], DocHit)
    assert hits[0].title == "Title"
    assert hits[0].content_url == "/content"


@pytest.mark.asyncio
async def test_search_non_200_raises_error():
    """Search propagates DocsClientError when status code is non-200."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = create_client(httpx.MockTransport(handler))
    with pytest.raises(DocsClientError):
        await client.search("query")


@pytest.mark.asyncio
async def test_fetch_section_returns_doc_section():
    """fetch_section returns DocSection when payload is valid."""

    section_payload = {"title": "Section", "body": {"content": "data"}, "readerUrl": "https://reader"}

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=section_payload)

    client = create_client(httpx.MockTransport(handler))

    section = await client.fetch_section("/api/khub/maps/m/topics/t/content")

    assert isinstance(section, DocSection)
    assert section.title == "Section"
    assert section.body == {"content": "data"}
    assert section.reader_url == "https://reader"


@pytest.mark.asyncio
async def test_fetch_section_handles_html_body():
    """fetch_section falls back to raw HTML when JSON isn't returned."""

    html_body = "<div>hello</div>"

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_body)

    client = create_client(httpx.MockTransport(handler))

    section = await client.fetch_section("/content")

    assert section.body == html_body
    assert section.title is None
