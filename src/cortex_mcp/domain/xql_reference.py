"""Utilities for retrieving Cortex XQL reference snippets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class XqlSection:
    """Represents a single stage or function section extracted from the docs."""

    name: str
    number: str
    heading: str
    body: str


class XqlReference:
    """Parses the XQL markdown reference and exposes stage/function snippets."""

    SECTION_PATTERN = re.compile(r"^###\s+(14\.(?P<section>\d)\.\d+)\s+(?P<title>.+)$")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._stages: dict[str, XqlSection] = {}
        self._functions: dict[str, XqlSection] = {}
        self._load()

    def get_stage(self, name: str) -> XqlSection | None:
        return self._stages.get(self._normalize(name))

    def get_function(self, name: str) -> XqlSection | None:
        return self._functions.get(self._normalize(name))

    def find_stages(self, text: str) -> list[XqlSection]:
        return self._find_sections(text, self._stages, is_function=False)

    def find_functions(self, text: str) -> list[XqlSection]:
        return self._find_sections(text, self._functions, is_function=True)

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"XQL documentation file not found at {self._path}")
        text = self._path.read_text(encoding="utf-8")
        self._parse(text)

    def _parse(self, text: str) -> None:
        current: dict[str, Any] | None = None
        for line in text.splitlines():
            match = self.SECTION_PATTERN.match(line)
            if match:
                self._store_section(current)
                current = {
                    "number": match.group(1),
                    "section": match.group("section"),
                    "title": match.group("title").strip(),
                    "heading": line.strip(),
                    "lines": [],
                }
                continue
            if current is not None:
                current["lines"].append(line)
        self._store_section(current)

    def _store_section(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        section_type = data.get("section")
        if section_type not in {"3", "4"}:
            return
        body = "\n".join(data.get("lines", [])).strip()
        entry = XqlSection(
            name=data["title"],
            number=data["number"],
            heading=data["heading"],
            body=body,
        )
        key = self._normalize(entry.name)
        target = self._stages if section_type == "3" else self._functions
        target.setdefault(key, entry)

    def _find_sections(
        self,
        text: str,
        target: dict[str, XqlSection],
        *,
        is_function: bool,
    ) -> list[XqlSection]:
        normalized = text.lower()
        matches: list[XqlSection] = []
        seen_numbers: set[str] = set()
        for section in target.values():
            base_name = section.name.lower().strip()
            token = base_name.rstrip("()")
            token_alpha = re.sub(r"[\W_]+", "", token)
            if is_function:
                indicators = [
                    f"{token}(",
                    f"{token_alpha}(",
                ]
            else:
                indicators = [
                    f"| {token}",
                    f"|{token}",
                    f"| {token_alpha}",
                    f"|{token_alpha}",
                    f"{token} ",
                    f"{token}=",
                    f"{token_alpha} ",
                    f"{token_alpha}=",
                    f" {token} ",
                    f" {token_alpha} ",
                ]
            if any(indicator in normalized for indicator in indicators) and section.number not in seen_numbers:
                matches.append(section)
                seen_numbers.add(section.number)
        return matches

    @staticmethod
    def _normalize(name: str) -> str:
        slug = re.sub(r"[^\w]", "", name).lower()
        return slug


def default_xql_reference_path() -> Path:
    """Return the default location for the crawled XQL markdown chapter."""

    root = Path(__file__).resolve().parents[3]
    return root / "data" / "cortex_docs" / "markdown" / "Cortex XSIAM Premium" / "14_Cortex XSIAM XQL" / "chapter.md"


def default_xql_reference_index_path() -> Path:
    """Return the default location for the precomputed XQL index JSON."""

    root = Path(__file__).resolve().parents[3]
    return root / "data" / "cortex_docs" / "xql_reference_index.json"
