"""Prepare Cortex content examples for embedding-friendly seeding."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv
from openai import OpenAI

SUPPORTED_EXTENSIONS = {".json", ".yaml", ".yml", ".txt"}
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_INPUT_DIR = Path("examples")
DEFAULT_OUTPUT_DIR = Path("processed_examples")
CLASSIFICATION_TYPES = [
    "xql_query",
    "correlation_rule",
    "widget",
    "xql_parsing_rule",
    "xql_modeling_rule",
    "playbook",
    "automation_script",
    "integration",
    "unknown",
]
CLASSIFICATION_PRODUCTS = ["xsiam", "unknown"]
PRIMARY_PRODUCT = "xsiam"
ALLOWED_TYPES = set(CLASSIFICATION_TYPES)
ALLOWED_PRODUCTS = {"xsiam"}

logger = logging.getLogger(__name__)


@dataclass
class ContentObject:
    parsed_content: Any | None
    raw_text: str | None
    original_format: str
    source_file_rel: str
    source_index: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Cortex content examples for embeddings.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with raw content files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for normalized YAML outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use for classification (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and classify but do not write files.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for number of files to process.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args(argv)


def load_api_key(env_var: str) -> str:
    load_dotenv()
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Environment variable {env_var} is not set.")
    return key


def collect_input_files(input_dir: Path, max_files: int | None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
            if max_files is not None and len(files) >= max_files:
                break
    return files


def load_content_objects(path: Path, base_dir: Path) -> list[ContentObject]:
    relative_path = str(path.relative_to(base_dir))
    suffix = path.suffix.lower()
    objects: list[ContentObject] = []
    if suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8")
        objects.append(
            ContentObject(
                parsed_content=None,
                raw_text=raw_text,
                original_format="txt",
                source_file_rel=relative_path,
                source_index=0,
            )
        )
        return objects

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            iterable = enumerate(data)
        else:
            iterable = [(0, data)]
        for idx, item in iterable:
            objects.append(
                ContentObject(
                    parsed_content=item,
                    raw_text=None,
                    original_format="json",
                    source_file_rel=relative_path,
                    source_index=idx,
                )
            )
        return objects

    if suffix in {".yaml", ".yml"}:
        docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        idx_counter = 0
        for doc in docs:
            if doc is None:
                continue
            if isinstance(doc, list):
                for item in doc:
                    objects.append(
                        ContentObject(
                            parsed_content=item,
                            raw_text=None,
                            original_format="yaml",
                            source_file_rel=relative_path,
                            source_index=idx_counter,
                        )
                    )
                    idx_counter += 1
            else:
                objects.append(
                    ContentObject(
                        parsed_content=doc,
                        raw_text=None,
                        original_format="yaml",
                        source_file_rel=relative_path,
                        source_index=idx_counter,
                    )
                )
                idx_counter += 1
        return objects

    raise ValueError(f"Unsupported file type for {path}")


def classify_content(
    *,
    client: OpenAI,
    model: str,
    original_format: str,
    parsed_content: Any | None,
    raw_text: str | None,
    source_file_path: str,
    max_retries: int = 1,
) -> dict[str, Any]:
    prompt = build_classification_prompt(
        original_format=original_format,
        parsed_content=parsed_content,
        raw_text=raw_text,
        source_file_path=source_file_path,
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Palo Alto Cortex content classifier. "
                            "Return ONLY valid JSON with the exact schema specified."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            message = response.choices[0].message.content or ""
            return parse_classification_response(message)
        except Exception as exc:  # pragma: no cover - network errors
            last_error = exc
            logger.warning("Classification attempt %s failed for %s: %s", attempt + 1, source_file_path, exc)
    logger.error("Classification failed for %s. Falling back to unknown.", source_file_path)
    return fallback_classification(last_error)


def build_classification_prompt(
    *,
    original_format: str,
    parsed_content: Any | None,
    raw_text: str | None,
    source_file_path: str,
) -> str:
    snippet = ""
    if parsed_content is not None:
        snippet = json.dumps(parsed_content, ensure_ascii=False)[:4000]
    elif raw_text is not None:
        snippet = raw_text[:4000]

    schema = {
        "title": "string | null",
        "description": "string | null",
        "type": f"one of {CLASSIFICATION_TYPES}",
        "tags": "list[str]",
        "product": f"one of {CLASSIFICATION_PRODUCTS}",
        "language": "ISO language code or null",
        "severity": "string or null",
        "confidence": "float 0-1 or null",
        "metadata": "object with any extra info",
    }
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "You receive snippets of Palo Alto Cortex XSIAM content ONLY. "
        "Infer metadata and respond with VALID JSON using the schema below. "
        "Do not include markdown, comments, or explanations.\n"
        f"Source file: {source_file_path}\n"
        f"Original format: {original_format}\n"
        f"Schema:\n{schema_text}\n"
        f"Snippet:\n{snippet}"
    )


def parse_classification_response(response_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse classification response as JSON: %s", exc)
        return fallback_classification(exc)

    ctype = parsed.get("type") or "unknown"
    if ctype not in ALLOWED_TYPES:
        ctype = "unknown"
    parsed["type"] = ctype

    parsed.setdefault("tags", [])
    product = parsed.get("product") or PRIMARY_PRODUCT
    if product not in ALLOWED_PRODUCTS:
        product = PRIMARY_PRODUCT
    parsed["product"] = product
    parsed.setdefault("metadata", {})
    return parsed


def fallback_classification(error: Exception | None) -> dict[str, Any]:
    if error:
        logger.debug("Fallback classification triggered due to error: %s", error)
    payload = {
        "title": None,
        "description": None,
        "type": "unknown",
        "tags": [],
        "product": PRIMARY_PRODUCT,
        "language": None,
        "severity": None,
        "confidence": None,
        "metadata": {},
    }
    return payload


def slugify(value: str, fallback_seed: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        slug = hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:12]
    return slug


def build_output_record(
    obj: ContentObject,
    classification: dict[str, Any],
    *,
    input_dir: Path,
) -> tuple[dict[str, Any], Path]:
    product = classification.get("product") or PRIMARY_PRODUCT
    ctype = classification.get("type") or "unknown"
    title = classification.get("title")
    slug_seed = title or f"{obj.source_file_rel}-{obj.source_index}"
    slug = slugify(slug_seed, slug_seed)

    output_id = f"{product}:{ctype}:{slug}"
    metadata = {
        key: value
        for key, value in classification.items()
        if key
        not in {
            "title",
            "description",
            "type",
            "tags",
            "product",
            "language",
            "severity",
            "confidence",
            "metadata",
        }
    }
    metadata.update(classification.get("metadata") or {})

    content_field: Any
    if obj.original_format in {"json", "yaml"}:
        content_field = obj.parsed_content
    else:
        content_field = {"raw_text": obj.raw_text or ""}

    record = {
        "id": output_id,
        "source_file": obj.source_file_rel,
        "source_index": obj.source_index,
        "original_format": obj.original_format,
        "product": classification.get("product"),
        "type": classification.get("type"),
        "title": title,
        "description": classification.get("description"),
        "tags": classification.get("tags") or [],
        "language": classification.get("language"),
        "severity": classification.get("severity"),
        "confidence": classification.get("confidence"),
        "metadata": metadata,
        "content": content_field,
    }

    product_dir = product or "generic"
    type_dir = ctype or "unknown"
    filename = f"{product_dir}_{type_dir}_{slug}.yaml"
    output_path = Path(product_dir) / type_dir / filename
    return record, output_path


def write_yaml(record: dict[str, Any], base_output_dir: Path, relative_output: Path) -> None:
    target_path = base_output_dir / relative_output
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False, allow_unicode=True)
    logger.debug("Wrote %s", target_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key(args.openai_api_key_env)
    client = OpenAI(api_key=api_key)

    files = collect_input_files(input_dir, args.max_files)
    logger.info("Discovered %s files (limit=%s).", len(files), args.max_files or "none")

    processed_files = 0
    processed_objects = 0
    errors = 0

    for file_path in files:
        processed_files += 1
        try:
            objects = load_content_objects(file_path, input_dir)
        except Exception as exc:
            errors += 1
            logger.error("Failed to parse %s: %s", file_path, exc)
            continue

        for obj in objects:
            processed_objects += 1
            classification = classify_content(
                client=client,
                model=args.model,
                original_format=obj.original_format,
                parsed_content=obj.parsed_content,
                raw_text=obj.raw_text,
                source_file_path=obj.source_file_rel,
            )
            record, rel_output_path = build_output_record(obj, classification, input_dir=input_dir)
            if args.dry_run:
                logger.info("Dry-run: would write %s", rel_output_path)
                continue
            try:
                write_yaml(record, output_dir, rel_output_path)
            except Exception as exc:  # pragma: no cover - filesystem error
                errors += 1
                logger.error("Failed to write %s: %s", rel_output_path, exc)

    logger.info(
        "Processing complete. files=%s objects=%s errors=%s dry_run=%s",
        processed_files,
        processed_objects,
        errors,
        args.dry_run,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
