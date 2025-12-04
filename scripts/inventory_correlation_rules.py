"""Inventory XSIAM correlation rules from the Cortex content repo."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv
from github import Github

DEFAULT_REPO = "demisto/content"
DEFAULT_OUTPUT_DIR = Path("examples/xql_correlation")

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository in owner/name form.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where YAML rules will be written (default: {DEFAULT_OUTPUT_DIR}).",
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

    logger.warning("GITHUB_TOKEN not set; falling back to anonymous GitHub client (rate limited).")
    return Github()


def iter_correlation_files(repo, pack_path: str) -> Iterable[str]:
    """Yield YAML file paths underneath a pack's CorrelationRules directory."""

    correlation_dir = f"{pack_path}/CorrelationRules"
    try:
        entries = repo.get_contents(correlation_dir)
    except Exception as exc:
        logger.debug("Pack %s does not include CorrelationRules (%s)", pack_path, exc)
        return []

    return [entry for entry in entries if entry.path.endswith(".yml")]


def download_rule(session: requests.Session, file_obj, output_dir: Path) -> Path:
    response = session.get(file_obj.download_url)
    response.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_obj.name
    output_path.write_text(response.text, encoding="utf-8")
    return output_path


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

        correlation_files = list(iter_correlation_files(repo, pack.path))
        if not correlation_files:
            continue

        logger.info("Found %s correlation rules in %s.", len(correlation_files), pack.path)
        for file_obj in correlation_files:
            try:
                output_path = download_rule(session, file_obj, args.output_dir)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", file_obj.path, exc)
                continue
            downloaded += 1
            logger.debug("Saved %s -> %s", file_obj.path, output_path)

    logger.info("Completed. Downloaded %s correlation rules into %s.", downloaded, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
