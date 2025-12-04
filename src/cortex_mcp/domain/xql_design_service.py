"""Service that orchestrates Cortex XQL design using docs + examples + LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from typing import Any, Protocol

from cortex_mcp.domain.docs_client import DocsClient
from cortex_mcp.domain.examples_repository import Example, ExamplesRepository
from cortex_mcp.domain.sanitization import sanitize_example_content
from cortex_mcp.domain.xql_reference import (
    XqlReference,
    default_xql_reference_index_path,
)


class LLMClient(Protocol):
    """Minimal protocol describing the LLM abstraction used for design."""

    async def complete(self, prompt: str) -> str:
        """Return an LLM completion as a raw string."""


class XqlDesignError(RuntimeError):
    """Raised when the XQL design workflow cannot be completed."""


class XqlDesignService:
    """Combine docs, examples, and LLM reasoning to craft Cortex XQL content."""

    def __init__(
        self,
        docs_client: DocsClient,
        examples_repository: ExamplesRepository,
        llm_client: LLMClient | None,
        *,
        doc_hits: int = 3,
        example_hits: int = 3,
        xql_reference: XqlReference | None = None,
        xql_reference_index_path: Path | None = None,
        include_llm_output: bool = True,
    ) -> None:
        self._docs_client = docs_client
        self._examples_repository = examples_repository
        self._llm_client = llm_client
        self._doc_hits = doc_hits
        self._example_hits = example_hits
        self._xql_reference = xql_reference
        self._include_llm_output = include_llm_output
        self._stage_patterns: list[tuple[str, re.Pattern[str]]] = []
        self._function_patterns: list[tuple[str, re.Pattern[str]]] = []
        if self._xql_reference:
            self._load_reference_index(xql_reference_index_path)

    async def design(
        self,
        goal: str,
        environment_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Design Cortex content for the provided goal."""

        if not goal.strip():
            raise XqlDesignError("Goal must be provided.")

        examples = self._examples_repository.search_examples(goal, top_k=self._example_hits)
        xql_refs = self._collect_xql_reference(examples)
        prompt = self._build_prompt(
            goal,
            examples,
            environment_hints,
            xql_refs,
        )
        content_types = ["xql"]
        parsed: dict[str, Any] | None = None
        if self._include_llm_output and self._llm_client:
            response_text = await self._llm_client.complete(prompt)
            parsed = self._parse_llm_response(response_text)
        result: dict[str, Any] = {
            "status": "designed",
            "goal": goal,
            "content_types": content_types,
            "environment_hints": environment_hints or {},
            "examples_used": [self._summarize_example(example) for example in examples],
            "xql_reference": xql_refs,
        }
        if parsed:
            result["generated_content"] = parsed.get("generated_content")
            result["explanation"] = parsed.get("explanation")
        return result

    def _build_prompt(
        self,
        goal: str,
        examples: list[Example],
        environment_hints: dict[str, Any] | None,
        xql_refs: list[dict[str, Any]],
    ) -> str:
        hints_section = json.dumps(environment_hints or {}, indent=2)
        docs_block_lines = [
            f"- {ref['type']} {ref['name']} ({ref['number']}): {ref['heading']}"
            for ref in xql_refs
        ] or ["- XQL reference pulled from examples:"]
        examples_block_lines = [
            f"- {example.id} [{example.type}]: {example.description}"
            for example in examples
        ] or ["- No similar examples located."]
        xql_detail_blocks = [
            f"{ref['heading']}\n{ref['body']}"
            for ref in xql_refs
        ] or ["No XQL reference snippets available."]

        return "\n".join(
            [
                "You are the Cortex Helper prompt.",
                "Design a complete Cortex XQL query using the references provided.",
                "generated_content must be a runnable XQL query (with config/dataset, stages, and comments).",
                "explanation should summarize why the query works.",
                f"Goal: {goal}",
                "Environment hints:",
                hints_section,
                "Relevant XQL references:",
                *docs_block_lines,
                "Related examples:",
                *examples_block_lines,
                "Detailed XQL guidance:",
                *xql_detail_blocks,
                "Respond ONLY as JSON with fields 'generated_content' and 'explanation'.",
            ]
        )

    def _parse_llm_response(self, response_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise XqlDesignError("LLM response was not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise XqlDesignError("LLM response must be a JSON object.")
        if "generated_content" not in payload:
            raise XqlDesignError("LLM response missing 'generated_content'.")
        return payload

    @staticmethod
    def _summarize_example(example: Example) -> dict[str, Any]:
        sanitized = sanitize_example_content(example.content)
        xql_snippet = XqlDesignService._extract_xql_text(sanitized)
        return {
            "id": example.id,
            "type": example.type,
            "xql": xql_snippet if isinstance(xql_snippet, str) else None,
        }

    def _collect_xql_reference(self, examples: list[Example]) -> list[dict[str, Any]]:
        if not self._xql_reference or not examples:
            return []
        matches: dict[tuple[str, str], dict[str, Any]] = {}
        for example in examples:
            sanitized = sanitize_example_content(example.content)
            xql_text = self._extract_xql_text(sanitized)
            lowered = xql_text.lower() if isinstance(xql_text, str) else ""
            for section in self._match_stages(lowered):
                key = ("stage", section.number)
                entry = matches.setdefault(
                    key,
                    {
                        "type": "stage",
                        "name": section.name,
                        "number": section.number,
                        "heading": section.heading,
                        "body": section.body,
                        "examples": set(),
                    },
                )
                entry["examples"].add(example.id)
            for section in self._match_functions(lowered):
                key = ("function", section.number)
                entry = matches.setdefault(
                    key,
                    {
                        "type": "function",
                        "name": section.name,
                        "number": section.number,
                        "heading": section.heading,
                        "body": section.body,
                        "examples": set(),
                    },
                )
                entry["examples"].add(example.id)
        result: list[dict[str, Any]] = []
        for entry in matches.values():
            result.append(
                {
                    "type": entry["type"],
                    "name": entry["name"],
                    "number": entry["number"],
                    "heading": entry["heading"],
                    "body": entry["body"],
                    "examples": sorted(entry["examples"]),
                }
            )
        return result

    def _match_stages(self, lowered_text: str) -> list[XqlSection]:
        sections: list[XqlSection] = []
        seen: set[str] = set()
        for name, pattern in self._stage_patterns:
            if pattern.search(lowered_text):
                section = self._xql_reference.get_stage(name)
                if section and section.number not in seen:
                    sections.append(section)
                    seen.add(section.number)
        return sections

    def _match_functions(self, lowered_text: str) -> list[XqlSection]:
        sections: list[XqlSection] = []
        seen: set[str] = set()
        for name, call_pattern, word_pattern in self._function_patterns:
            if call_pattern.search(lowered_text) or word_pattern.search(lowered_text):
                section = self._xql_reference.get_function(name)
                if section and section.number not in seen:
                    sections.append(section)
                    seen.add(section.number)
        return sections

    @staticmethod
    def _extract_xql_text(sanitized: Any) -> Any:
        if isinstance(sanitized, dict):
            xql_value = sanitized.get("xql")
            if isinstance(xql_value, str):
                return xql_value
            return sanitized
        if isinstance(sanitized, str):
            try:
                loaded = yaml.safe_load(sanitized)
            except yaml.YAMLError:
                extracted = XqlDesignService._extract_xql_from_string(sanitized)
                return extracted if extracted is not None else sanitized
            if isinstance(loaded, dict) and isinstance(loaded.get("xql"), str):
                return loaded["xql"]
        return sanitized

    @staticmethod
    def _extract_xql_from_string(raw: str) -> str | None:
        quoted_match = re.search(r"xql:\s*'(?P<body>.*?)'", raw, re.DOTALL)
        if not quoted_match:
            quoted_match = re.search(r'xql:\s*"(?P<body>.*?)"', raw, re.DOTALL)
        if quoted_match:
            return quoted_match.group("body").strip()
        block_match = re.search(r"xql:\s*\|\s*\n(?P<body>(?:[ \t].*\n?)*)", raw)
        if block_match:
            block = block_match.group("body")
            lines = block.splitlines()
            stripped_lines = []
            min_indent = None
            for line in lines:
                if not line.strip():
                    stripped_lines.append("")
                    continue
                indent = len(line) - len(line.lstrip(" \t"))
                if min_indent is None or indent < min_indent:
                    min_indent = indent
            min_indent = min_indent or 0
            for line in lines:
                stripped_lines.append(line[min_indent:])
            return "\n".join(stripped_lines).strip()
        inline_match = re.search(r"xql:\s*(.*)", raw)
        if inline_match:
            first_line = inline_match.group(1).splitlines()[0]
            return first_line.strip()
        return None

    def _load_reference_index(self, index_path: Path | None) -> None:
        path = index_path or default_xql_reference_index_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        stages = data.get("stages") or []
        functions = data.get("functions") or []
        self._stage_patterns = []
        for name in stages:
            token = re.escape(name.lower())
            pattern = re.compile(rf"(?:^\s*|\|\s*){token}\b", re.MULTILINE)
            self._stage_patterns.append((name, pattern))
        self._function_patterns = []
        for name in functions:
            base = re.escape(name.lower())
            call_pattern = re.compile(rf"\b{base}\s*\(")
            word_pattern = re.compile(rf"\b{base}\b")
            self._function_patterns.append((name, call_pattern, word_pattern))


class LocalXqlLLMClient:
    """Fallback LLM client that assembles runnable XQL using local examples."""

    def __init__(self, examples_repository: ExamplesRepository | None = None) -> None:
        self._examples_repository = examples_repository

    async def complete(self, prompt: str) -> str:
        """Return a deterministic JSON block for offline environments."""

        goal = self._extract_goal(prompt)
        reference_names = self._extract_reference_names(prompt)
        example_ids = self._extract_example_ids(prompt)
        contexts = self._build_example_contexts(example_ids)
        generated = self._compose_query(goal, contexts, reference_names)
        explanation = self._compose_explanation(goal, contexts, reference_names)
        return json.dumps(
            {
                "generated_content": generated,
                "explanation": explanation,
            }
        )

    @staticmethod
    def _extract_goal(prompt: str) -> str:
        match = re.search(r"Goal:\s*(.+)", prompt)
        if match:
            return match.group(1).strip()
        return "Cortex content design"

    @staticmethod
    def _extract_reference_names(prompt: str) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"-\s*(?:stage|function)\s+([A-Za-z0-9_]+)", prompt):
            name = match.group(1)
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(name)
        return names

    @staticmethod
    def _extract_example_ids(prompt: str) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"-\s*([a-z0-9_-]+)\s*\[[^\]]+\]:", prompt):
            example_id = match.group(1)
            if example_id in seen:
                continue
            seen.add(example_id)
            ids.append(example_id)
        return ids

    def _build_example_contexts(self, example_ids: list[str]) -> list[dict[str, str]]:
        contexts: list[dict[str, str]] = []
        if not self._examples_repository:
            return contexts
        for example_id in example_ids:
            example = self._examples_repository.get_example(example_id)
            if not example:
                continue
            sanitized = sanitize_example_content(example.content)
            xql_text = XqlDesignService._extract_xql_text(sanitized)
            if not isinstance(xql_text, str):
                continue
            contexts.append(
                {
                    "id": example_id,
                    "description": example.description.strip(),
                    "xql": xql_text.strip(),
                }
            )
        return contexts

    def _compose_query(
        self,
        goal: str,
        contexts: list[dict[str, str]],
        reference_names: list[str],
    ) -> str:
        if contexts:
            header_lines = [f"// Goal: {goal}"]
            if reference_names:
                header_lines.append(f"// Key XQL references: {', '.join(reference_names)}")
            base_query = contexts[0]["xql"]
            for ctx in contexts[1:]:
                summary = self._shorten_description(ctx["description"])
                header_lines.append(f"// Additional inspiration: {ctx['id']} - {summary}")
            header_lines.append(base_query)
            return "\n".join(header_lines).strip()
        dataset = self._default_dataset(goal)
        stage_chain = self._default_stage_chain(reference_names)
        query_lines = [
            f"// Goal: {goal}",
            "config timeframe = 7d",
            f"| dataset = {dataset}",
            *stage_chain,
        ]
        return "\n".join(query_lines).strip()

    def _compose_explanation(
        self,
        goal: str,
        contexts: list[dict[str, str]],
        reference_names: list[str],
    ) -> str:
        if contexts:
            base = contexts[0]
            dataset = self._detect_dataset(base["xql"])
            explanation_parts = [
                f"Reused example {base['id']} to satisfy '{goal}'.",
            ]
            if dataset:
                explanation_parts.append(f"The query runs against {dataset}.")
            if reference_names:
                explanation_parts.append(f"References: {', '.join(reference_names)}.")
            if len(contexts) > 1:
                others = ", ".join(ctx["id"] for ctx in contexts[1:])
                explanation_parts.append(f"Also reviewed: {others}.")
            return " ".join(explanation_parts)
        explanation = f"Built heuristic XQL query for '{goal}'"
        if reference_names:
            explanation += f" using {', '.join(reference_names)}"
        explanation += "."
        return explanation

    @staticmethod
    def _shorten_description(text: str, limit: int = 80) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3].rstrip() + "..."

    @staticmethod
    def _detect_dataset(xql: str) -> str | None:
        match = re.search(r"(?:dataset|preset)\s*=\s*([A-Za-z0-9_]+)", xql)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _default_dataset(goal: str) -> str:
        lowered = goal.lower()
        if "process" in lowered or "execution" in lowered or "lateral" in lowered:
            return "xdr_process"
        if "network" in lowered or "command and control" in lowered or "c2" in lowered:
            return "xdr_network_connection"
        if "event" in lowered or "logon" in lowered or "privileged" in lowered:
            return "xdr_event_log"
        return "xdr_data"

    @staticmethod
    def _default_stage_chain(reference_names: list[str]) -> list[str]:
        lowered = {name.lower() for name in reference_names}
        lines: list[str] = []
        if "alter" in lowered:
            lines.append('| alter note = "heuristic_match"')
        if "regextract" in lowered:
            lines.append(
                '| alter parsed_value = arrayindex(regextract(action_process_image_command_line, '
                '"\\"([^\\\\"]+)\\""), 0)'
            )
        if "filter" in lowered:
            lines.append("| filter note != null")
        else:
            lines.append("| filter true")
        return lines
