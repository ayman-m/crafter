#!/usr/bin/env python3
"""Seed the ExamplesRepository with standardized YAML examples and persistent embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import yaml

from cortex_mcp.domain.examples_repository import Example, ExamplesRepository

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {
    "xql_query",
    "correlation_rule",
    "widget",
    "xql_parsing_rule",
    "xql_modeling_rule",
    "playbook",
    "automation_script",
    "integration",
}
REQUIRED_PRODUCT = "xsiam"


def seed_examples_repository(
    *,
    source_dir: Path,
    data_path: Path,
    chroma_path: Path,
    reset_storage: bool = False,
) -> dict[str, int]:
    """Load normalized examples and persist them into the ExamplesRepository."""

    if reset_storage:
        _reset_storage(data_path=data_path, chroma_path=chroma_path)

    repo = ExamplesRepository(
        data_path=data_path,
        chroma_path=chroma_path,
        embedding_model_name=os.getenv("EMBEDDING_MODEL") or None,
    )

    existing_before = dict(repo._examples)
    chroma_ids = _load_chroma_ids(repo)
    combined_seen = set(existing_before.keys()) | chroma_ids
    logger.info(
        "Repository loaded with %s JSON examples and %s Chroma entries. Beginning seed from %s",
        len(existing_before),
        len(chroma_ids),
        source_dir,
    )

    appended = 0
    skipped = 0
    duplicate_ids: list[str] = []
    total = 0
    source_type_counts: dict[str, int] = {}
    for example in _discover_examples(source_dir):
        total += 1
        source_type_counts[example.type] = source_type_counts.get(example.type, 0) + 1
        if example.id in combined_seen:
            skipped += 1
            duplicate_ids.append(example.id)
            logger.debug("Example %s already present (pre-check); skipping.", example.id)
            continue
        try:
            repo.append_example(example)
            appended += 1
            combined_seen.add(example.id)
            logger.debug("Appended example=%s type=%s", example.id, example.type)
        except ValueError as exc:
            skipped += 1
            duplicate_ids.append(example.id)
            logger.debug("Example %s already present (append guard); skipping. reason=%s", example.id, exc)

    logger.info(
        "Seeding complete. candidates=%s appended=%s skipped=%s new_total=%s",
        total,
        appended,
        skipped,
        len(repo._examples),
    )
    repo_type_counts = _count_repo_types(repo)
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        logger.info("Duplicate IDs skipped (%s total): %s", len(duplicate_ids), preview)
    return {
        "appended": appended,
        "skipped": skipped,
        "source_counts": source_type_counts,
        "repo_counts": repo_type_counts,
    }


def _reset_storage(*, data_path: Path, chroma_path: Path) -> None:
    if data_path.exists():
        try:
            data_path.unlink()
            logger.info("Deleted existing data file %s", data_path)
        except OSError as exc:
            logger.warning("Unable to delete %s: %s", data_path, exc)
    if chroma_path.exists():
        try:
            shutil.rmtree(chroma_path)
            logger.info("Deleted existing chroma directory %s", chroma_path)
        except OSError as exc:  # pragma: no cover
            logger.warning("Unable to delete %s: %s", chroma_path, exc)


def _discover_examples(source_dir: Path) -> Iterable[Example]:
    for path in sorted(source_dir.rglob("*.yaml")):
        if not path.is_file():
            continue
        try:
            record = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - invalid YAML
            logger.warning("Skipping %s due to parse error: %s", path, exc)
            continue
        if not isinstance(record, dict):
            logger.warning("Skipping %s because root is not a mapping.", path)
            continue
        example = _record_to_example(record, path)
        if example is not None:
            yield example


def _record_to_example(record: dict, path: Path) -> Example | None:
    example_id = record.get("id") or path.stem
    example_type = _normalize_type(record.get("type") or "generic_text")
    description = record.get("description") or record.get("title") or f"Example from {path.name}"

    content_node = record.get("content")
    if content_node is None:
        logger.warning("Record %s missing 'content'; skipping.", path)
        return None
    if isinstance(content_node, (dict, list)):
        content_str = yaml.safe_dump(content_node, allow_unicode=True, sort_keys=False)
    else:
        content_str = str(content_node)

    tags = list(record.get("tags") or [])
    product = (record.get("product") or REQUIRED_PRODUCT).lower()
    if product and product != REQUIRED_PRODUCT:
        logger.info("Skipping %s because product '%s' is not %s.", path, product, REQUIRED_PRODUCT)
        return None
    if example_type not in ALLOWED_TYPES:
        logger.info("Skipping %s because type '%s' is not allowed.", path, example_type)
        return None
    if product:
        tags.append(str(product))
    metadata = record.get("metadata") or {}
    for key, value in metadata.items():
        tags.append(f"{key}:{_format_metadata_value(value)}")

    return Example(
        id=example_id,
        type=example_type,
        description=description,
        content=content_str,
        tags=tags,
    )


def _normalize_type(raw_type: str) -> str:
    mapping = {
        "xql_correlation_rule": "correlation_rule",
        "xql_widget": "widget",
        "xql_reporting_widget": "widget",
        "xql_parsing_rule": "xql_parsing_rule",
        "xql_modeling_rule": "xql_modeling_rule",
        "automation": "automation_script",
        "automation_script": "automation_script",
        "script": "automation_script",
        "integration": "integration",
    }
    normalized = raw_type.lower()
    return mapping.get(normalized, normalized)


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _count_repo_types(repo: ExamplesRepository) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in repo._examples.values():
        counts[example.type] = counts.get(example.type, 0) + 1
    return counts


def _load_chroma_ids(repo: ExamplesRepository) -> set[str]:
    collection = getattr(repo, "_chroma_collection", None)
    if collection is None:
        return set()
    try:
        response = collection.get(include=[], limit=None)
    except Exception:  # pragma: no cover - chroma failure
        logger.exception("Unable to fetch IDs from Chroma collection.")
        return set()
    ids = response.get("ids") or []
    return set(ids)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the ExamplesRepository with normalized examples.")
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("processed_examples"),
        help="Directory containing normalized YAML examples.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/examples/examples.json"),
        help="Path to JSON file used by ExamplesRepository.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("data/chroma"),
        help="Directory where Chroma stores persistent data.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing stored data/chroma indices before seeding.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    results = seed_examples_repository(
        source_dir=args.examples_dir,
        data_path=args.data_path,
        chroma_path=args.chroma_path,
        reset_storage=args.reset,
    )
    logger.info(
        "Seeded repository using %s. appended=%s skipped=%s",
        args.examples_dir,
        results["appended"],
        results["skipped"],
    )
    logger.info("Source examples per type: %s", results["source_counts"])
    logger.info("Repository totals per type: %s", results["repo_counts"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
import os
