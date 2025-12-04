#!/usr/bin/env python3
"""Inspect and query the ExamplesRepository/Chroma store."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from cortex_mcp.domain.examples_repository import ExamplesRepository  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect stored examples and run sample queries.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/examples/examples.json"),
        help="Path to the ExamplesRepository JSON store.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("data/chroma"),
        help="Path to the Chroma persistence directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all stored example IDs and metadata.",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Print counts per example type.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional semantic search query to run against the repository.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of search hits to return (default 3).")
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Include example content in the listing/search results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()

    embedding_model = os.getenv("EMBEDDING_MODEL")

    repo = ExamplesRepository(
        data_path=args.data_path,
        chroma_path=args.chroma_path,
        embedding_model_name=embedding_model,
    )

    print(f"Repository contains {len(repo._examples)} examples.")

    if args.counts:
        counts: dict[str, int] = {}
        for example in repo._examples.values():
            counts[example.type] = counts.get(example.type, 0) + 1
        print("Counts per type:", json.dumps(counts, ensure_ascii=False, indent=2))

    if args.list:
        for example in repo._examples.values():
            _print_example(example, show_content=args.show_content)

    if args.query:
        print("\n=== Search Results ===")
        hits = repo.search_examples(args.query, top_k=args.top_k)
        for example in hits:
            _print_example(example, show_content=args.show_content)

    return 0


def _print_example(example: Any, show_content: bool = False) -> None:
    payload = {
        "id": example.id,
        "type": example.type,
        "description": example.description,
        "tags": example.tags,
    }
    if show_content:
        payload["content"] = example.content[:500] + ("..." if len(example.content) > 500 else "")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
