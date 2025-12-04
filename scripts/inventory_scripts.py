"""Inventory Cortex pack scripts and store YAML automations."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from github import Github

DEFAULT_REPO = "demisto/content"
DEFAULT_OUTPUT_DIR = Path("examples/automations")
SUPPORTED_SUFFIXES = (".yml", ".yaml")

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository in owner/name form.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for downloaded automation YAMLs (default: {DEFAULT_OUTPUT_DIR}).",
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


def iter_script_files(repo, scripts_path: str) -> list:
    """Recursively collect YAML script files within a pack."""

    entries = repo.get_contents(scripts_path)
    files = []
    for entry in entries:
        if entry.type == "file" and entry.path.endswith(SUPPORTED_SUFFIXES):
            files.append(entry)
        elif entry.type == "dir":
            files.extend(iter_script_files(repo, entry.path))
    return files


def download_yaml(session: requests.Session, url: str, output_dir: Path, name: str) -> Path:
    response = session.get(url)
    response.raise_for_status()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(response.text, encoding="utf-8")
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
        scripts_dir = f"{pack.path}/Scripts"
        try:
            repo.get_contents(scripts_dir, ref=repo.default_branch)
        except Exception:
            continue

        files = iter_script_files(repo, scripts_dir)
        if not files:
            continue

        logger.info("Found %s script YAML files in %s.", len(files), scripts_dir)
        for file_obj in files:
            try:
                path = download_yaml(session, file_obj.download_url, args.output_dir, file_obj.name)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", file_obj.path, exc)
                continue
            downloaded += 1
            logger.debug("Saved %s -> %s", file_obj.path, path)

    logger.info("Completed. Downloaded %s automation scripts into %s.", downloaded, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
