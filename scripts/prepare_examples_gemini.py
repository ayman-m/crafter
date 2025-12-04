"""Prepare Cortex XSIAM examples using Gemini for metadata + descriptions."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable

from google import genai
from google.genai import types
import yaml
from dotenv import load_dotenv

SUPPORTED_EXTENSIONS = {".json", ".yaml", ".yml", ".txt"}
DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_INPUT_DIR = Path("examples")
DEFAULT_OUTPUT_DIR = Path("processed_examples")
DEFAULT_OUTPUT_PRODUCT = "xsiam"

# Gemini 3 + late 2.5 preview models are only available in the "global" location
# for the Vertex AI Gemini API, not regional locations like "us-central1".
DEFAULT_GCP_LOCATION = "global"

FOLDER_TYPE_MAP = {
    "xql_correlation": "correlation_rule",
    "xql_queries": "xql_query",
    "xql_widgets": "widget",
    "xql_models": "model_rule",
    "xql_parsing": "parsing_rule",
    "xql_correlation": "xql_correlation",
    "playbooks": "playbook",
    "integrations": "integration",
    "automations": "automation_script",
}

logger = logging.getLogger(__name__)


@dataclass
class ContentObject:
    parsed_content: Any | None
    raw_text: str | None
    original_format: str
    source_file_rel: str
    source_index: int
    content_type: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--gcp-credentials",
        type=Path,
        default=Path("cortex-gcp-labs-6639860ab0b6.json"),
    )
    parser.add_argument("--gcp-project", default="cortex-gcp-labs")
    # changed default from "us-central1" -> "global"
    parser.add_argument("--gcp-location", default=DEFAULT_GCP_LOCATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def configure_gcp_credentials(credentials_path: Path) -> None:
    load_dotenv()
    if not credentials_path.exists():
        raise RuntimeError(f"GCP credentials file not found at {credentials_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path.resolve())


def collect_input_files(input_dir: Path, max_files: int | None) -> list[Path]:
    files: list[Path] = []
    for folder in FOLDER_TYPE_MAP:
        folder_path = input_dir / folder
        if not folder_path.exists():
            logger.debug("Input subfolder %s missing; skipping.", folder_path)
            continue
        for path in sorted(folder_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
                if max_files is not None and len(files) >= max_files:
                    return files
    return files


def load_content_objects(path: Path, base_dir: Path) -> list[ContentObject]:
    relative_path = path.relative_to(base_dir)
    folder_name = relative_path.parts[0]
    content_type = FOLDER_TYPE_MAP.get(folder_name)
    if content_type is None:
        logger.warning("Skipping %s because folder %s is not mapped.", path, folder_name)
        return []

    suffix = path.suffix.lower()
    objects: list[ContentObject] = []
    if suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8")
        objects.append(
            ContentObject(
                parsed_content=None,
                raw_text=raw_text,
                original_format="txt",
                source_file_rel=str(relative_path),
                source_index=0,
                content_type=content_type,
            )
        )
        return objects

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        iterable: Iterable[tuple[int, Any]]
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
                    source_file_rel=str(relative_path),
                    source_index=idx,
                    content_type=content_type,
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
                            source_file_rel=str(relative_path),
                            source_index=idx_counter,
                            content_type=content_type,
                        )
                    )
                    idx_counter += 1
            else:
                objects.append(
                    ContentObject(
                        parsed_content=doc,
                        raw_text=None,
                        original_format="yaml",
                        source_file_rel=str(relative_path),
                        source_index=idx_counter,
                        content_type=content_type,
                    )
                )
                idx_counter += 1
        return objects

    logger.warning("Unsupported file type for %s; skipping.", path)
    return []


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return str(value)


def build_prompt(content: ContentObject) -> str:
    if content.parsed_content is not None:
        snippet = json.dumps(content.parsed_content, ensure_ascii=False, default=_json_default)[:6000]
    else:
        snippet = (content.raw_text or "")[:6000]

    schema = {
        "title": "Short human-readable title (string)",
        "detailed_description": (
            "Multi-paragraph description explaining purpose, trigger conditions, "
            "key fields, and downstream actions."
        ),
        "use_case": "One paragraph describing a realistic use-case or scenario for this artifact.",
        "tools": "list[str] naming technologies/services/products referenced (e.g., Palo Alto firewall, Active Directory).",
        "actions": "list[str] describing notable actions/steps performed (relevant for playbooks/integrations/automations).",
        "tags": "list[str] describing technologies, tactics, products, etc.",
        "sensitivity": "string (low|medium|high) or null",
        "confidence": "float 0-1 representing metadata confidence",
    }
    return (
        "You are classifying Palo Alto Cortex XSIAM content. "
        "Provide a richly detailed description for semantic search. "
        "Respond ONLY with valid JSON using the schema provided.\n"
        f"Content type: {content.content_type}\n"
        f"Source file: {content.source_file_rel}\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Snippet:\n{snippet}"
    )


def generate_gemini_text(client: genai.Client, model_name: str, prompt: str) -> str:
    text_part = types.Part.from_text(text=prompt)
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=[types.Content(role="user", parts=[text_part])],
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=4096,
            response_modalities=["TEXT"],
        ),
    ):
        if isinstance(chunk, tuple):
            chunk = chunk[0]
        if getattr(chunk, "text", None):
            response_text += chunk.text
    return response_text


def classify_with_gemini(
    client: genai.Client, model_name: str, content: ContentObject
) -> dict[str, Any]:
    prompt = build_prompt(content)
    try:
        text = generate_gemini_text(client, model_name, prompt)
    except Exception as exc:  # pragma: no cover - API errors
        logger.error("Gemini classification failed for %s: %s", content.source_file_rel, exc)
        return fallback_metadata()

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse Gemini response for %s: %s. Raw text (truncated): %r",
            content.source_file_rel,
            exc,
            text[:300],
        )
        return fallback_metadata()

    parsed.setdefault("title", None)
    parsed.setdefault("detailed_description", None)
    parsed.setdefault("use_case", None)
    parsed.setdefault("tools", [])
    parsed.setdefault("actions", [])
    parsed.setdefault("tags", [])
    parsed.setdefault("sensitivity", None)
    parsed.setdefault("confidence", None)
    return parsed


def fallback_metadata() -> dict[str, Any]:
    return {
        "title": None,
        "detailed_description": None,
        "use_case": None,
        "tools": [],
        "actions": [],
        "tags": [],
        "sensitivity": None,
        "confidence": None,
    }


def build_record(content: ContentObject, metadata: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    if content.original_format in {"json", "yaml"}:
        normalized_content = content.parsed_content
    else:
        normalized_content = {"raw_text": content.raw_text or ""}

    record = {
        "id": slug_id(content),
        "product": DEFAULT_OUTPUT_PRODUCT,
        "type": content.content_type,
        "source_file": content.source_file_rel,
        "source_index": content.source_index,
        "title": metadata.get("title"),
        "description": metadata.get("detailed_description"),
        "tags": metadata.get("tags") or [],
        "sensitivity": metadata.get("sensitivity"),
        "confidence": metadata.get("confidence"),
        "content": normalized_content,
    }

    rel_path = Path(DEFAULT_OUTPUT_PRODUCT) / content.content_type / f"{record['id']}.yaml"
    return record, rel_path


def slug_id(content: ContentObject) -> str:
    safe_name = Path(content.source_file_rel).stem.lower().replace(" ", "-")
    return f"{content.content_type}-{safe_name}-{content.source_index}"


def write_yaml(record: dict[str, Any], base_output_dir: Path, relative_output: Path) -> None:
    target_path = base_output_dir / relative_output
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, allow_unicode=True, sort_keys=False)
    logger.debug("Wrote %s", target_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_gcp_credentials(args.gcp_credentials)
    client = genai.Client(
        vertexai=True,
        project=args.gcp_project,
        location=args.gcp_location,
    )

    files = collect_input_files(input_dir, args.max_files)
    logger.info("Discovered %s files across typed folders.", len(files))

    total_objects = 0
    errors = 0
    for file_path in files:
        objects = load_content_objects(file_path, input_dir)
        for obj in objects:
            total_objects += 1
            _, rel_path = build_record(obj, {})
            output_file = output_dir / rel_path
            if output_file.exists():
                logger.debug("Skipping %s because %s exists.", obj.source_file_rel, rel_path)
                continue

            metadata = classify_with_gemini(client, args.model, obj)
            record, rel_path = build_record(obj, metadata)

            if args.dry_run:
                logger.info("Dry-run: would write %s", rel_path)
                continue

            try:
                write_yaml(record, output_dir, rel_path)
            except Exception as exc:  # pragma: no cover - filesystem
                errors += 1
                logger.error("Failed to write %s: %s", rel_path, exc)

    logger.info("Completed. objects=%s errors=%s dry_run=%s", total_objects, errors, args.dry_run)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
