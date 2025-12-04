"""Prompt template registrations for Cortex Content MCP."""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from pydantic import Field

from cortex_mcp.domain.examples_repository import ExampleType
from cortex_mcp.mcp import mcp


@mcp.prompt(
    name="cortex_docs_contextualizer",
    title="Summarize Cortex Docs for a Question",
    description="Guide an assistant to consult Cortex documentation resources before summarizing an answer.",
    tags={"docs", "analysis"},
)
def cortex_docs_contextualizer(
    question: str = Field(description="The Cortex question, issue, or workflow the user needs clarified."),
    product_version: str | None = Field(
        default=None,
        description="Optional product/release identifier to bias doc searches (e.g., 'Cortex XSOAR 8.6').",
    ),
) -> str:
    """Prompt analysts to answer a question with cited Cortex documentation."""

    version_line = (
        f"Prioritize documentation for **{product_version}**."
        if product_version
        else "Use the most recent documentation unless the sources clearly indicate version-specific behavior."
    )
    return dedent(
        f"""
        You are a Cortex documentation analyst. Help the user understand the following request:

        ```text
        {question}
        ```

        Workflow:
        1. Run `cortex-docs://search/{{query}}` using key phrases from the question and capture at least 3 promising hits.
        2. Fetch full sections for the best matches with `cortex-docs://section/{{id}}` so you can quote key steps or caveats.
        3. Summarize the guidance, highlighting differences between product versions when relevant. {version_line}

        Output format:
        - **Summary:** 2–3 paragraphs written for Cortex engineers.
        - **Key Steps:** Bulleted, referencing doc titles.
        - **Citations:** Markdown list with doc titles + reader URLs.
        - **Open Questions:** List anything the docs did not address.
        """
    ).strip()


@mcp.prompt(
    name="content_example_matcher",
    title="Find Similar Cortex Content",
    description="Surface existing content examples that resemble the provided incident or initiative.",
    tags={"examples", "triage"},
)
def content_example_matcher(
    incident_summary: str = Field(description="Short description of the incident, alert, or workflow you need references for."),
    desired_type: ExampleType | None = Field(
        default=None,
        description=(
            "Optional example type filter (e.g., 'playbook', 'xql_query', 'correlation_rule'). "
            "If omitted, search across all types."
        ),
    ),
    max_examples: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum examples to highlight from the `content-examples://search` resource.",
    ),
) -> str:
    """Prompt to retrieve and synthesize relevant Cortex content examples."""

    type_hint = (
        f"Focus on the `{desired_type}` example type when reviewing results."
        if desired_type
        else "Consider any example type that helps the request."
    )
    return dedent(
        f"""
        Match the incident below to existing Cortex content so the engineer can reuse proven patterns.

        **Incident Summary**
        {incident_summary}

        Instructions:
        1. Query `content-examples://search/{{query}}` with terms from the incident. Request at least {max_examples} examples.
        2. {type_hint}
        3. For each example, extract: ID, type, description, noteworthy tags, and why it is relevant.
        4. If no close matches exist, report the gap and suggest search refinements.

        Output format:
        - Table with columns: `Example ID`, `Type`, `Tags`, `Why it helps`.
        - Follow-up recommendations on how to adapt the most promising example.
        """
    ).strip()


@mcp.prompt(
    name="cortex_content_diff_checker",
    title="Compare Planned Change vs Current Docs",
    description="Inspect a proposed Cortex change and flag misalignments with current documentation.",
    tags={"docs", "review"},
)
def cortex_content_diff_checker(
    change_plan: str = Field(description="Detailed description of the proposed change or runbook you want validated."),
    doc_keywords: list[str] | None = Field(
        default=None,
        description="Optional list of doc search keywords (JSON list). Leave empty to derive from the change plan.",
    ),
) -> str:
    """Prompt highlighting mismatches between a plan and official documentation."""

    keywords_text = ", ".join(doc_keywords) if doc_keywords else "derive the best keywords from the plan itself"
    return dedent(
        f"""
        Validate the following planned change against official Cortex documentation:

        ```text
        {change_plan}
        ```

        Steps:
        1. Identify critical claims, commands, and prerequisites mentioned in the plan.
        2. Search docs using `{keywords_text}` via `cortex-docs://search/{{query}}`, then pull authoritative sections.
        3. Compare each step from the plan with the docs. Highlight contradictions, missing prerequisites, or outdated values.

        Respond with:
        - **Alignment Summary:** one paragraph describing overall parity.
        - **Matches:** bullet list with plan items that match docs (include citations).
        - **Mismatches:** bullet list describing conflicts, each with a suggested fix.
        - **Unknowns:** anything still unverified plus which doc queries to run next.
        """
    ).strip()


@mcp.prompt(
    name="playbook_scaffold_generator",
    title="Draft Cortex Playbook Scaffold",
    description="Create a structured outline for a new playbook by referencing docs and existing examples.",
    tags={"design", "examples", "docs"},
)
def playbook_scaffold_generator(
    incident_type: str = Field(description="High-level classification of the incident or workflow (e.g., 'USB device violation')."),
    automation_goal: str | None = Field(
        default=None,
        description="Optional summary of the automation depth or success criteria (e.g., 'fully automated triage').",
    ),
    evidence_sources: Sequence[str] | None = Field(
        default=None,
        description="Optional list of data sources or signals the playbook must ingest (JSON list).",
    ),
) -> str:
    """Prompt to draft a reusable playbook structure."""

    automation_line = (
        f"The engineer wants to achieve: {automation_goal}."
        if automation_goal
        else "Default to a balanced automation level (automate repetitive enrichment, confirm critical actions)."
    )
    evidence_line = (
        f"Ensure the scaffold covers evidence from: {', '.join(evidence_sources)}."
        if evidence_sources
        else "Call out the evidence you expect to collect so the engineer can confirm."
    )
    return dedent(
        f"""
        Design a Cortex playbook scaffold for **{incident_type}**.

        {automation_line}
        {evidence_line}

        Research Requirements:
        1. Skim `cortex-docs://search/{{query}}` for the latest official guidance.
        2. Pull 1–2 similar playbooks via `content-examples://search/{{query}}` and reuse proven phases.

        Deliverable:
        - **Overview:** what the playbook solves and success criteria.
        - **Phases:** Detection, Enrichment, Decision, Response (with tasks + automation vs analyst actions).
        - **Data + Tools:** which integrations/scripts are assumed.
        - **Next Steps:** MCP tools to run next (e.g., `design_xql_content`, `test_cortex_content`).
        """
    ).strip()


@mcp.prompt(
    name="validation_readiness_assessor",
    title="Phase 4 Validation Readiness",
    description="Generate a validation checklist referencing docs and examples before promoting new content.",
    tags={"validation", "docs", "examples"},
)
def validation_readiness_assessor(
    artifact_summary: str = Field(description="Description of the content artifact awaiting validation."),
    test_env: str | None = Field(
        default=None,
        description="Optional detail about the validation environment or constraints.",
    ),
) -> str:
    """Prompt creating a validation checklist grounded in docs and examples."""

    env_line = (
        f"Validation Environment: {test_env}."
        if test_env
        else "Assume a standard staging tenant unless instructed otherwise."
    )
    return dedent(
        f"""
        Prepare the validation plan for this artifact:

        ```text
        {artifact_summary}
        ```

        {env_line}

        Process:
        1. Reconfirm requirements by searching docs for the feature or API calls involved.
        2. Look for adjacent examples via `content-examples://search/{{query}}` to reuse baseline data or expected outputs.
        3. Build a validation checklist that covers environment prep, test data, success metrics, and rollback guidance.

        Output:
        - **Readiness Summary:** 2–3 sentences.
        - **Checklist:** bullet list grouped into Setup, Functional Tests, Negative Tests, and Evidence Capture.
        - **References:** cite docs/examples that justify each test.
        - **Risks & Mitigations:** anything that might block promotion plus mitigation ideas.
        """
    ).strip()
