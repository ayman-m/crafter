"""Utilities to sanitize example content before returning it externally."""

from __future__ import annotations

import re

_AUTHOR_BLOCK_RE = re.compile(r"(?mi)^[ \t]*authors:\s*\n(?:^[ \t]*-.*\n?)+")
_AUTHOR_LINE_RE = re.compile(r"(?mi)^[ \t]*author[s]?:.*$")
_AUTHOR_INLINE_RE = re.compile(r"(?mi)author:[^,\n]*")
_COMMENT_AUTHOR_RE = re.compile(r"(?mi)^//\s*author.*$")


def sanitize_example_content(text: str) -> str:
    """Remove author attribution details from serialized example content."""

    cleaned = _AUTHOR_BLOCK_RE.sub("", text)
    cleaned = _AUTHOR_LINE_RE.sub("", cleaned)
    cleaned = _AUTHOR_INLINE_RE.sub("", cleaned)
    cleaned = _COMMENT_AUTHOR_RE.sub("", cleaned)
    return cleaned
