"""Verbose multi-stage FastMCP smoke-test client."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from fastmcp import Client

SERVER = os.getenv("MCP_SERVER", "http://127.0.0.1:8000/mcp")
AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN") or os.getenv("MCP_ACCESS_TOKEN")
DOCS_QUERY = os.getenv("MCP_TEST_DOCS_QUERY", "brockervm installation")
DOCS_LIMIT = int(os.getenv("MCP_TEST_DOCS_LIMIT", "2"))
EXAMPLES_QUERY = os.getenv("MCP_TEST_EXAMPLES_QUERY", "phishing playbook")
EXAMPLES_TOP_K = int(os.getenv("MCP_TEST_EXAMPLES_TOP_K", "2"))
PROMPT_DOCS_QUESTION = os.getenv(
    "MCP_TEST_PROMPT_DOCS_QUESTION",
    "How do I deploy Broker VM collectors for on-prem Windows servers?",
)
PROMPT_DOCS_VERSION = os.getenv("MCP_TEST_PROMPT_DOCS_VERSION")
PROMPT_INCIDENT_SUMMARY = os.getenv(
    "MCP_TEST_PROMPT_INCIDENT_SUMMARY",
    "Recurring phishing alerts targeting finance leadership using malicious attachments.",
)
PROMPT_EXAMPLE_TYPE = os.getenv("MCP_TEST_PROMPT_EXAMPLE_TYPE", "playbook")
PROMPT_CHANGE_PLAN = os.getenv(
    "MCP_TEST_PROMPT_CHANGE_PLAN",
    "Plan to update Broker VM installation runbook to cover Ubuntu hosts and automated certificate rotation.",
)
PROMPT_DOC_KEYWORDS = os.getenv(
    "MCP_TEST_PROMPT_DOC_KEYWORDS",
    '["broker vm","xsoar installation","certificate rotation"]',
)
PROMPT_INCIDENT_TYPE = os.getenv(
    "MCP_TEST_PROMPT_INCIDENT_TYPE",
    "Phishing incident with credential theft risk",
)
PROMPT_AUTOMATION_GOAL = os.getenv(
    "MCP_TEST_PROMPT_AUTOMATION_GOAL",
    "Automate enrichment and containment, require analyst approval before blocking senders.",
)
PROMPT_EVIDENCE_SOURCES = os.getenv(
    "MCP_TEST_PROMPT_EVIDENCE_SOURCES",
    '["Email attachments","XDR alerts","O365 audit logs"]',
)
PROMPT_ARTIFACT_SUMMARY = os.getenv(
    "MCP_TEST_PROMPT_ARTIFACT_SUMMARY",
    "New phishing response playbook integrating O365 quarantine actions and Slack notifications.",
)
PROMPT_TEST_ENV = os.getenv(
    "MCP_TEST_PROMPT_TEST_ENV",
    "XSOAR staging tenant with mock O365 + Slack integrations.",
)
TOOL_DESIGN_GOAL = os.getenv(
    "MCP_TEST_TOOL_DESIGN_GOAL",
    "Design a phishing containment playbook that enriches alerts and notifies analysts.",
)
TOOL_ENV_HINTS_RAW = os.getenv(
    "MCP_TEST_TOOL_ENV_HINTS",
    '{"environment": "XSOAR", "priority": "high"}',
)
TOOL_TEST_CONTENT = os.getenv(
    "MCP_TEST_TOOL_CONTENT",
    "playbook: taskid: 1\nname: Sample Playbook\n",
)
TOOL_TEST_TYPE = os.getenv("MCP_TEST_TOOL_TEST_TYPE", "playbook")
TOOL_CURATE_DESCRIPTION = os.getenv(
    "MCP_TEST_TOOL_CURATE_DESCRIPTION",
    "Sample curated phishing playbook generated during smoke test.",
)
TOOL_CURATE_TAGS_RAW = os.getenv(
    "MCP_TEST_TOOL_CURATE_TAGS",
    '["smoke-test","phishing"]',
)


def _parse_json_list(raw: str | None, fallback: list[str]) -> list[str]:
    if not raw:
        return fallback
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    print(f"Warning: could not parse JSON list from environment value '{raw}'. Using fallback {fallback}.")
    return fallback


def _parse_json_object(raw: str | None, fallback: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return fallback
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {str(k): v for k, v in parsed.items()}
    print(f"Warning: could not parse JSON object from environment value '{raw}'. Using fallback {fallback}.")
    return fallback


DOC_KEYWORD_LIST = _parse_json_list(
    PROMPT_DOC_KEYWORDS,
    ["broker vm", "xsoar installation", "certificate rotation"],
)
EVIDENCE_SOURCE_LIST = _parse_json_list(
    PROMPT_EVIDENCE_SOURCES,
    ["Email attachments", "XDR alerts", "O365 audit logs"],
)
TOOL_ENV_HINTS = _parse_json_object(TOOL_ENV_HINTS_RAW, {"environment": "XSOAR"})
TOOL_CURATE_TAGS = _parse_json_list(TOOL_CURATE_TAGS_RAW, ["smoke-test"])

XQL_STAGE_NAME = os.getenv("MCP_TEST_XQL_STAGE", "target")
XQL_FUNCTION_NAME = os.getenv("MCP_TEST_XQL_FUNCTION", "array_size")


def _serialize(value: Any) -> Any:
    """Convert FastMCP client objects into JSON-friendly structures."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _serialize(value.model_dump())
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    return str(value)


def _print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def _dump(label: str, payload: Any) -> None:
    serialized = _serialize(payload)
    print(f"{label}:")
    print(json.dumps(serialized, indent=2, ensure_ascii=False))


async def _stage_connectivity(client: Client) -> bool:
    _print_header("Stage 1 — Connectivity")
    try:
        await client.ping()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Ping failed: {exc!s}")
        return False
    print("Ping successful.")
    return True


async def _stage_inventory(client: Client) -> None:
    _print_header("Stage 2 — Inventory Resources, Prompts, Tools")
    resources = await client.list_resources()
    templates = await client.list_resource_templates()
    prompts = await client.list_prompts()
    tools = await client.list_tools()

    print(
        "Discovered "
        f"{len(resources)} resources, {len(templates)} resource templates, "
        f"{len(prompts)} prompts, {len(tools)} tools."
    )
    _dump("Resources", resources)
    _dump("Resource Templates", templates)
    _dump("Prompts", prompts)
    _dump("Tools", tools)


async def _stage_static_resource(client: Client) -> None:
    _print_header("Stage 3 — Static Resource (Overview)")
    uri = "docs://cortex-mcp/overview"
    try:
        contents = await client.read_resource(uri)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to read {uri}: {exc!s}")
        return
    print(f"Retrieved {len(contents)} content item(s) from {uri}.")
    _dump("Overview Contents", contents)


async def _stage_docs_resources(client: Client) -> None:
    _print_header("Stage 4 — Docs Resource Templates")
    uri = f"cortex-docs://search/{DOCS_QUERY}?limit={DOCS_LIMIT}"
    try:
        contents = await client.read_resource(uri)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Docs search failed for {uri}: {exc!s}")
        return
    print(f"Docs query '{DOCS_QUERY}' returned {len(contents)} content block(s).")
    _dump("Docs Search Result", contents)


async def _stage_xql_reference(client: Client) -> None:
    _print_header("Stage 4b — XQL Reference Resources")
    stage_uri = f"cortex-xql://stage/{XQL_STAGE_NAME}"
    function_uri = f"cortex-xql://function/{XQL_FUNCTION_NAME}"
    for label, uri in [("Stage Resource", stage_uri), ("Function Resource", function_uri)]:
        try:
            contents = await client.read_resource(uri)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to read {uri}: {exc!s}")
            continue
        payload = contents[0] if isinstance(contents, list) and contents else contents
        body_len = len(payload.get("body", "")) if isinstance(payload, dict) else 0
        print(f"{label} {uri} returned body length {body_len}.")
        _dump(f"{label} Result", contents)


async def _stage_examples_resource(client: Client) -> None:
    _print_header("Stage 5 — Content Examples Resource Template")
    uri = f"content-examples://search/{EXAMPLES_QUERY}?top_k={EXAMPLES_TOP_K}"
    try:
        contents = await client.read_resource(uri)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Examples search failed for {uri}: {exc!s}")
        return
    print(f"Examples query '{EXAMPLES_QUERY}' returned {len(contents)} content block(s).")
    _dump("Content Examples Result", contents)


async def _stage_prompt_tests(client: Client) -> None:
    _print_header("Stage 6 — Prompt Templates")
    tests: list[tuple[str, dict[str, Any]]] = [
        (
            "cortex_docs_contextualizer",
            {
                "question": PROMPT_DOCS_QUESTION,
                "product_version": PROMPT_DOCS_VERSION,
            },
        ),
        (
            "content_example_matcher",
            {
                "incident_summary": PROMPT_INCIDENT_SUMMARY,
                "desired_type": PROMPT_EXAMPLE_TYPE,
                "max_examples": 2,
            },
        ),
        (
            "cortex_content_diff_checker",
            {
                "change_plan": PROMPT_CHANGE_PLAN,
                "doc_keywords": DOC_KEYWORD_LIST,
            },
        ),
        (
            "playbook_scaffold_generator",
            {
                "incident_type": PROMPT_INCIDENT_TYPE,
                "automation_goal": PROMPT_AUTOMATION_GOAL,
                "evidence_sources": EVIDENCE_SOURCE_LIST,
            },
        ),
        (
            "validation_readiness_assessor",
            {
                "artifact_summary": PROMPT_ARTIFACT_SUMMARY,
                "test_env": PROMPT_TEST_ENV,
            },
        ),
    ]

    for name, base_args in tests:
        args = {key: value for key, value in base_args.items() if value}
        print(f"\nPrompt: {name}")
        try:
            result = await client.get_prompt(name, args or None)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to render prompt '{name}': {exc!s}")
            continue
        message_count = len(getattr(result, "messages", []))
        print(f"Rendered {message_count} message(s).")
        _dump(f"{name} Prompt Result", result)


async def _stage_tool_calls(client: Client) -> None:
    _print_header("Stage 7 — MCP Tools")
    tool_invocations = [
        (
            "design_xql_content",
            {
                "goal": TOOL_DESIGN_GOAL,
                "environment_hints": TOOL_ENV_HINTS,
            },
        ),
        (
            "test_cortex_content",
            {
                "content_type": TOOL_TEST_TYPE,
                "content": TOOL_TEST_CONTENT,
                "run_simulation": False,
                "generate_fake_data": False,
            },
        ),
        (
            "curate_cortex_example",
            {
                "content_type": TOOL_TEST_TYPE,
                "content": TOOL_TEST_CONTENT,
                "description": TOOL_CURATE_DESCRIPTION,
                "tags": TOOL_CURATE_TAGS,
                "must_be_valid": False,
            },
        ),
    ]

    for name, arguments in tool_invocations:
        print(f"\nTool: {name}")
        try:
            result = await client.call_tool(name, arguments)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Tool call failed: {exc!s}")
            continue
        _dump(f"{name} Result", result)


async def main() -> None:
    print(f"Connecting to MCP server at {SERVER}")
    client_kwargs: dict[str, Any] = {}
    if AUTH_TOKEN:
        client_kwargs["auth"] = AUTH_TOKEN
    client = Client(SERVER, **client_kwargs)
    async with client:
        if not await _stage_connectivity(client):
            return
        await _stage_inventory(client)
        await _stage_static_resource(client)
        await _stage_docs_resources(client)
        await _stage_xql_reference(client)
        #await _stage_examples_resource(client)
        #await _stage_prompt_tests(client)
        #await _stage_tool_calls(client)


if __name__ == "__main__":
    asyncio.run(main())
