"""Inventory XSIAM dashboards and extract XQL widgets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

import requests
import yaml
from dotenv import load_dotenv
from github import Github

DEFAULT_REPO = "demisto/content"
DEFAULT_DASHBOARD_DIR = Path("data/source/dashboard_files")
DEFAULT_WIDGET_DIR = Path("examples/xql_widgets")

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository in owner/name form.")
    parser.add_argument(
        "--dashboards-dir",
        type=Path,
        default=DEFAULT_DASHBOARD_DIR,
        help=f"Directory for downloaded dashboards (default: {DEFAULT_DASHBOARD_DIR}).",
    )
    parser.add_argument(
        "--widgets-dir",
        type=Path,
        default=DEFAULT_WIDGET_DIR,
        help=f"Directory for extracted XQL widgets (default: {DEFAULT_WIDGET_DIR}).",
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


def iter_dashboard_files(repo, pack_path: str) -> Iterable[str]:
    """Yield dashboard file objects underneath XSIAMDashboards."""

    dashboards_path = f"{pack_path}/XSIAMDashboards"
    try:
        entries = repo.get_contents(dashboards_path)
    except Exception as exc:
        logger.debug("Pack %s missing XSIAMDashboards (%s)", pack_path, exc)
        return []

    return [entry for entry in entries if entry.path.endswith(".json")]


def download_dashboard(session: requests.Session, file_obj, output_dir: Path) -> tuple[Path, dict]:
    response = session.get(file_obj.download_url)
    response.raise_for_status()

    payload = response.json()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_obj.name
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, payload


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "xql-widget"


def extract_widget_name(widget_data: dict, widget_meta: dict | None) -> str:
    view_options = widget_data.get("viewOptions") or {}
    for entry in view_options.get("commands") or []:
        command = entry.get("command")
        if command and command.get("name") == "header":
            return command.get("value", "").strip("\"'")
    if widget_meta:
        title = widget_meta.get("title")
        if title:
            return title
    return widget_data.get("key") or (widget_meta or {}).get("widget_key") or "xql_widget"


def extract_widgets(payload: dict, widgets_dir: Path) -> int:
    dashboards = payload.get("dashboards_data") or []
    widget_lookup = {w.get("widget_key"): w for w in payload.get("widgets_data") or []}

    widgets_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for dashboard in dashboards:
        d_name = dashboard.get("name") or "xsiam_dashboard"
        layout_rows = dashboard.get("layout") or []
        for row in layout_rows:
            for widget in row.get("data") or []:
                data = widget.get("data") or {}
                if data.get("type") != "Custom XQL":
                    continue
                widget_meta = widget_lookup.get(widget.get("key"))
                widget_name = extract_widget_name(data, widget_meta)
                filename = f"{slugify(d_name)}_{slugify(widget_name)}.yaml"
                output_path = widgets_dir / filename

                record = {
                    "dashboard": {
                        "name": d_name,
                        "description": dashboard.get("description"),
                    },
                    "widget": {
                        "key": widget.get("key"),
                        "title": widget_meta.get("title") if widget_meta else None,
                        "description": widget_meta.get("description") if widget_meta else None,
                    },
                    "phrase": data.get("phrase"),
                    "time_frame": data.get("time_frame"),
                    "view_options": data.get("viewOptions"),
                }
                output_path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8")
                saved += 1

    return saved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    client = load_github_client()
    repo = client.get_repo(args.repo)
    session = requests.Session()

    downloaded = 0
    total_widgets = 0
    for pack in repo.get_contents("Packs"):
        if pack.type != "dir":
            continue

        dashboard_files = list(iter_dashboard_files(repo, pack.path))
        if not dashboard_files:
            continue

        logger.info("Found %s dashboards in %s.", len(dashboard_files), pack.path)
        for file_obj in dashboard_files:
            try:
                output_path, payload = download_dashboard(session, file_obj, args.dashboards_dir)
                widgets_saved = extract_widgets(payload, args.widgets_dir)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", file_obj.path, exc)
                continue
            downloaded += 1
            total_widgets += widgets_saved
            logger.debug("Saved %s -> %s and extracted %s widgets", file_obj.path, output_path, widgets_saved)

    logger.info(
        "Completed. Dashboards=%s Widgets=%s (dashboards in %s, widgets in %s).",
        downloaded,
        total_widgets,
        args.dashboards_dir,
        args.widgets_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
