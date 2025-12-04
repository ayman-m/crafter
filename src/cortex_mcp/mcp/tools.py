"""FastMCP tool registrations for Cortex Content MCP."""

from __future__ import annotations

from typing import Any

from cortex_mcp.domain.xql_design_service import XqlDesignError, XqlDesignService
from cortex_mcp.domain.content_testing_service import ContentTestingService
from cortex_mcp.domain.curation_service import CurationError, CurationService
from cortex_mcp.domain.fake_data_service import FakeDataService
from cortex_mcp.domain.sanitization import sanitize_example_content
from cortex_mcp.domain.xsiam_client import XsiamClient, XsiamClientError
from cortex_mcp.domain.xql_reference import default_xql_reference_index_path
from cortex_mcp.mcp import mcp
from cortex_mcp.mcp.resources import get_docs_client, get_examples_repository, get_xql_reference
import yaml

_xql_design_service: XqlDesignService | None = None
_testing_service: ContentTestingService | None = None
_curation_service: CurationService | None = None
_fake_data_service: FakeDataService | None = None
_xsiam_client: XsiamClient | None = None


def _get_xql_design_service() -> XqlDesignService:
    global _xql_design_service
    if _xql_design_service is None:
        examples_repo = get_examples_repository()
        _xql_design_service = XqlDesignService(
            docs_client=get_docs_client(),
            examples_repository=examples_repo,
            llm_client=None,
            xql_reference=get_xql_reference(),
            xql_reference_index_path=default_xql_reference_index_path(),
            include_llm_output=False,
        )
    return _xql_design_service


def _get_fake_data_service() -> FakeDataService:
    global _fake_data_service
    if _fake_data_service is None:
        _fake_data_service = FakeDataService()
    return _fake_data_service


def _get_xsiam_client() -> XsiamClient:
    global _xsiam_client
    if _xsiam_client is None:
        _xsiam_client = XsiamClient()
    return _xsiam_client


def _get_testing_service() -> ContentTestingService:
    global _testing_service
    if _testing_service is None:
        _testing_service = ContentTestingService(
            xsiam_client=_get_xsiam_client(),
            fake_data_service=_get_fake_data_service(),
        )
    return _testing_service


def _get_curation_service() -> CurationService:
    global _curation_service
    if _curation_service is None:
        _curation_service = CurationService(
            repository=get_examples_repository(),
            testing_service=_get_testing_service(),
        )
    return _curation_service


@mcp.tool(
    name="design_xql_content",
    description="Design Cortex XQL content using docs, examples, and LLM reasoning.",
)
async def design_xql_content(
    goal: str,
    environment_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate Cortex XQL content for the provided goal."""

    service = _get_xql_design_service()
    try:
        return await service.design(goal, environment_hints)
    except XqlDesignError as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool(
    name="test_cortex_content",
    description="Validate and optionally simulate Cortex content.",
)
async def test_cortex_content(
    content_type: str,
    content: str,
    run_simulation: bool = True,
    generate_fake_data: bool = False,
    fake_data_scenario: str | None = None,
    fake_data_count: int = 5,
    fetch_results: bool = True,
) -> dict[str, Any]:
    """Validate and optionally simulate a Cortex artifact."""

    service = _get_testing_service()
    try:
        return await service.test_content(
            content_type=content_type,
            content=content,
            run_simulation=run_simulation,
            generate_fake_data=generate_fake_data,
            fake_data_scenario=fake_data_scenario,
            fake_data_count=fake_data_count,
            fetch_results=fetch_results,
        )
    except XsiamClientError as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool(
    name="curate_cortex_example",
    description="Promote validated Cortex content into the examples repository.",
)
async def curate_cortex_example(
    content_type: str,
    content: str,
    description: str,
    tags: list[str] | None = None,
    must_be_valid: bool = True,
) -> dict[str, Any]:
    """Persist validated Cortex content into the examples vector store."""

    service = _get_curation_service()
    try:
        return await service.curate(
            content_type=content_type,
            content=content,
            description=description,
            tags=tags,
            must_be_valid=must_be_valid,
        )
    except CurationError as exc:
        return {"status": "error", "message": str(exc)}


_EXAMPLE_LIMITS = {
    "xql_query": 5,
    "correlation_rule": 1,
    "widget": 2,
    "xql_parsing_rule": 1,
    "xql_modeling_rule": 2,
    "playbook": 1,
    "automation_script": 1,
    "integration": 1,
}


@mcp.tool(
    name="retrieve_cortex_examples",
    description=(
        "Return representative stored Cortex examples for a query. "
        "Supported content_type values: xql_query, correlation_rule, widget, "
        "xql_parsing_rule, xql_modeling_rule, playbook, automation_script, integration."
    ),
)
async def retrieve_cortex_examples(
    query: str,
    content_type: str,
) -> dict[str, Any]:
    """Return sanitized stored examples for the requested type."""

    content_type = content_type.lower().strip()
    limit = _EXAMPLE_LIMITS.get(content_type)
    if not limit:
        return {
            "status": "error",
            "message": f"Content type '{content_type}' is not supported.",
        }

    repo = get_examples_repository()
    hits = repo.search_examples(query, top_k=max(limit * 2, limit))
    examples: list[dict[str, Any]] = []
    for example in hits:
        if example.type != content_type:
            continue
        sanitized = sanitize_example_content(example.content)
        content_payload = _extract_primary_content(content_type, sanitized)
        examples.append(
            {
                "id": example.id,
                "type": example.type,
                "description": example.description,
                "content": content_payload,
                "tags": example.tags,
            }
        )
        if len(examples) >= limit:
            break

    return {
        "status": "retrieved",
        "query": query,
        "content_type": content_type,
        "examples": examples,
    }


def _extract_primary_content(content_type: str, raw_text: str) -> str:
    """Return the key content payload for the example."""

    if content_type not in {
        "xql_query",
        "correlation_rule",
        "xql_parsing_rule",
        "xql_modeling_rule",
    }:
        return raw_text

    try:
        parsed = yaml.safe_load(raw_text)
    except Exception:  # pragma: no cover - best effort
        return raw_text
    if isinstance(parsed, dict):
        xql = parsed.get("xql")
        if isinstance(xql, str):
            return xql
    return raw_text
