"""Inventory XSIAM modeling rules from the Cortex content repo."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import requests
import yaml
from dotenv import load_dotenv
from github import Github

DEFAULT_REPO = "demisto/content"
DEFAULT_OUTPUT_DIR = Path("examples/xql_models")
RULE_EXT = ".xif"
SCHEMA_EXT = ".json"
YAML_EXT = ".yml"

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository in owner/name form.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for downloaded modeling rules (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args(argv)


def load_github_client() -> Github:
    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    token = os.environ.get("GITHUB_TOKEN")
    if token:
        logger.info("Authenticating to GitHub using provided token.")
        return Github(token)

    logger.warning("GITHUB_TOKEN not set; using anonymous GitHub client (rate limited).")
    return Github()


def iter_modeling_folders(repo, pack_path: str) -> Iterable[str]:
    """Yield ModelingRules directory entries inside a pack."""

    modeling_path = f"{pack_path}/ModelingRules"
    try:
        entries = repo.get_contents(modeling_path)
    except Exception as exc:
        logger.debug("Pack %s missing ModelingRules (%s)", pack_path, exc)
        return []

    return [entry for entry in entries if entry.type == "dir"]


def download_text(session: requests.Session, url: str) -> str:
    response = session.get(url)
    response.raise_for_status()
    return response.text


def write_yaml(output_dir: Path, filename: str, content: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    client = load_github_client()
    repo = client.get_repo(args.repo)
    session = requests.Session()

    downloaded = 0
    for pack in repo.get_contents("Packs"):
        if pack.type != "dir":
            continue

        folders = list(iter_modeling_folders(repo, pack.path))
        if not folders:
            continue

        logger.info("Found %s modeling folders in %s.", len(folders), pack.path)
        for folder in folders:
            files = repo.get_contents(folder.path)
            rules_file = next((f for f in files if f.path.endswith(RULE_EXT)), None)
            schema_file = next((f for f in files if f.path.endswith(SCHEMA_EXT)), None)
            yaml_file = next((f for f in files if f.path.endswith(YAML_EXT)), None)
            if not (rules_file and schema_file and yaml_file):
                logger.debug("Skipping %s due to missing files.", folder.path)
                continue
            try:
                rules_text = download_text(session, rules_file.download_url)
                schema_text = download_text(session, schema_file.download_url)
                yaml_content = yaml.safe_load(download_text(session, yaml_file.download_url)) or {}
            except Exception as exc:
                logger.warning("Failed to fetch modeling components for %s: %s", folder.path, exc)
                continue

            yaml_content["rules"] = rules_text
            yaml_content["schema"] = schema_text
            output_name = Path(yaml_file.name).with_suffix(".yaml").name
            output_path = write_yaml(args.output_dir, output_name, yaml_content)
            downloaded += 1
            logger.debug("Saved %s", output_path)

    logger.info("Completed. Stored %s merged modeling rules in %s.", downloaded, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
