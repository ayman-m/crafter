"""FastMCP resource definitions for Cortex Content MCP."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from cortex_mcp.domain.docs_client import DocHit, DocSection, DocsClient, DocsClientError
from cortex_mcp.domain.examples_repository import Example, ExamplesRepository
from cortex_mcp.domain.sanitization import sanitize_example_content
from cortex_mcp.domain.xql_reference import XqlReference, default_xql_reference_path
from cortex_mcp.mcp import mcp

logger = logging.getLogger(__name__)

DOCS_BASE_URL = os.getenv("CORTEX_DOCS_BASE_URL", "https://docs-cortex.paloaltonetworks.com")
DOCS_API_KEY = os.getenv("CORTEX_DOCS_API_KEY")
EXAMPLES_DATA_PATH = Path(os.getenv("CONTENT_EXAMPLES_DATA_PATH", "data/examples/examples.json"))
EXAMPLES_CHROMA_PATH = Path(os.getenv("CONTENT_EXAMPLES_CHROMA_PATH", "data/chroma"))
XQL_REFERENCE_PATH = Path(os.getenv("CORTEX_XQL_DOC_PATH", default_xql_reference_path()))
MCP_OVERVIEW_PATH = Path(
    os.getenv(
        "MCP_OVERVIEW_PATH",
        Path(__file__).resolve().parents[3] / "docs" / "mcp_overview.txt",
    )
)

_docs_client: DocsClient | None = None
_examples_repository: ExamplesRepository | None = None
_xql_reference: XqlReference | None = None


def get_docs_client() -> DocsClient:
    """Return a singleton DocsClient instance built from environment variables."""

    global _docs_client
    if _docs_client is None:
        _docs_client = DocsClient(
            base_url=DOCS_BASE_URL,
            api_key=DOCS_API_KEY,
        )
    return _docs_client


def get_examples_repository() -> ExamplesRepository:
    """Return the shared ExamplesRepository instance."""

    global _examples_repository
    if _examples_repository is None:
        _examples_repository = ExamplesRepository(
            data_path=EXAMPLES_DATA_PATH,
            chroma_path=EXAMPLES_CHROMA_PATH,
        )
    return _examples_repository


def get_xql_reference() -> XqlReference:
    """Return the cached XQL reference parser."""

    global _xql_reference
    if _xql_reference is None:
        _xql_reference = XqlReference(XQL_REFERENCE_PATH)
    return _xql_reference


def _error_payload(message: str, details: str | None = None) -> dict[str, str | None]:
    payload: dict[str, str | None] = {"error": message}
    if details:
        payload["details"] = details
    return payload


def _format_hit(hit: DocHit) -> dict[str, Any]:
    return {
        "title": hit.title,
        "snippet": None,
        "url": hit.reader_url,
        "path": hit.topic_url,
        "section_id": hit.content_url,
    }


def _format_section(section: DocSection) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "title": section.title,
        "body": section.body,
        "url": section.reader_url,
        "path": section.section_id,
    }


def _format_example(example: Example) -> dict[str, Any]:
    content = sanitize_example_content(example.content)
    return {
        "id": example.id,
        "type": example.type,
        "description": example.description,
        "content": content,
        "tags": example.tags,
    }


def _format_xql_section(section) -> dict[str, Any]:
    return {
        "name": section.name,
        "number": section.number,
        "heading": section.heading,
        "body": section.body,
    }


@mcp.resource("cortex-docs://search/{query}{?limit}")
async def cortex_docs_search(query: str, limit: int = 5) -> dict[str, Any]:
    """Search Cortex documentation topics."""

    client = get_docs_client()
    try:
        hits = await client.search(query, limit=limit)
    except DocsClientError as exc:
        return _error_payload("Docs search failed", str(exc))
    logger.info("Docs search query=%s hits=%s", query, len(hits))
    return {
        "query": query,
        "results": [_format_hit(hit) for hit in hits],
    }


@mcp.resource("cortex-docs://section/{id}")
async def cortex_docs_section(id: str) -> dict[str, Any]:
    """Fetch a specific Cortex documentation section."""

    client = get_docs_client()
    try:
        section = await client.fetch_section(id)
    except DocsClientError as exc:
        return _error_payload("Docs section lookup failed", str(exc))
    logger.info("Docs section lookup id=%s title=%s", id, section.title)
    return _format_section(section)


@mcp.resource("content-examples://search/{query}{?top_k}")
async def content_examples_search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search locally stored Cortex content examples."""

    repo = get_examples_repository()
    hits = repo.search_examples(query, top_k=top_k)
    logger.info("Examples search query=%s hits=%s", query, len(hits))
    return [_format_example(example) for example in hits]
@mcp.resource(
    "docs://cortex-mcp/overview",
    description="Static summary of the Cortex Content MCP project.",
    tags={"documentation"},
)
def cortex_mcp_overview() -> dict[str, Any]:
    """Expose a short overview describing what this MCP server provides."""

    try:
        body = MCP_OVERVIEW_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Overview file not found at %s", MCP_OVERVIEW_PATH)
        return _error_payload(
            "Overview document missing",
            f"Expected file at {MCP_OVERVIEW_PATH}",
        )

    return {"title": "Cortex Content MCP Overview", "body": body}


@mcp.resource(
    "cortex-xql://stage/{name}",
    description="Fetch reference documentation for an XQL stage.",
    tags={"xql", "reference"},
)
async def cortex_xql_stage(name: str) -> dict[str, Any]:
    reference = get_xql_reference()
    section = reference.get_stage(name)
    if not section:
        return _error_payload("Stage not found", f"No XQL stage named '{name}'.")
    return _format_xql_section(section)


@mcp.resource(
    "cortex-xql://function/{name}",
    description="Fetch reference documentation for an XQL function.",
    tags={"xql", "reference"},
)
async def cortex_xql_function(name: str) -> dict[str, Any]:
    reference = get_xql_reference()
    section = reference.get_function(name)
    if not section:
        return _error_payload("Function not found", f"No XQL function named '{name}'.")
    return _format_xql_section(section)
