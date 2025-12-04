#!/usr/bin/env python3
"""Utility for mirroring Cortex documentation trees locally.

The crawler accepts a doctree JSON file describing the API endpoints and
documents to mirror.  Each document's full HTML + Markdown is written as a
single file, and every top-level chapter is also emitted into its own
subdirectory (HTML + Markdown) for downstream indexing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup  # type: ignore[import-not-found]
from markdownify import markdownify as md  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-not-found]

LOG = logging.getLogger("crawl_cortex_docs")
HTML_TEMPLATE = """<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <title>{title}</title>\n</head>\n<body>\n{content}\n</body>\n</html>\n"""


@dataclass
class DocConfig:
    name: str
    pretty_url: str
    update: bool = False

    def normalized_pretty_url(self) -> str:
        value = self.pretty_url.strip().lstrip("/")
        if value.startswith("r/"):
            value = value[2:]
        return value


@dataclass
class CrawlConfig:
    base_url: str
    pretty_url_endpoint: str
    document_map_endpoint: str
    pages_endpoint: str
    content_endpoint: str
    documents: list[DocConfig]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CrawlConfig":
        documents = [
            DocConfig(
                name=entry["name"],
                pretty_url=entry["pretty_url"],
                update=entry.get("update", False),
            )
            for entry in payload.get("documents", [])
        ]
        return cls(
            base_url=payload["base_url"].rstrip("/"),
            pretty_url_endpoint=payload["pretty_url_endpoint"],
            document_map_endpoint=payload["document_map_endpoint"],
            pages_endpoint=payload["pages_endpoint"],
            content_endpoint=payload["content_endpoint"],
            documents=documents,
        )

    def resolve(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return f"{self.base_url}/{path_or_url.lstrip('/') }"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Cortex documentation locally")
    parser.add_argument(
        "--doctree",
        type=Path,
        default=Path("data/cortex_docs/doctree.json"),
        help="Path to doctree JSON describing the crawl.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("data/cortex_docs/html"),
        help="Folder for HTML output.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/cortex_docs/markdown"),
        help="Folder for Markdown output.",
    )
    parser.add_argument(
        "--document",
        action="append",
        dest="documents",
        help="Optional document name filter (can be repeated).",
    )
    parser.add_argument("--force", action="store_true", help="Re-crawl even if files exist.")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay between requests (seconds).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def make_request(
    method: str,
    url: str,
    *,
    json_body: Any | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 5,
    delay: float = 0.0,
) -> requests.Response:
    headers = headers or {}
    if json_body is not None:
        headers.setdefault("Content-Type", "application/json")

    payload = {
        "method": method,
        "url": url,
        "params": params,
        "headers": headers,
        "json": json_body,
    }

    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(
                method,
                url,
                json=json_body,
                params=params,
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as exc:
            LOG.warning(
                "Request error (%s) attempt %s/%s: %s -- request=%s",
                url,
                attempt,
                attempts,
                exc,
                json.dumps(payload, default=str),
            )
            if attempt == attempts:
                raise
            time.sleep(max(delay, 5))
    raise RuntimeError("unreachable")


def sanitize_filename(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", title).strip()


def count_toc_items(toc: Iterable[dict[str, Any]]) -> int:
    total = 0
    for item in toc:
        total += 1
        children = item.get("children") or []
        total += count_toc_items(children)
    return total


class CortexDocCrawler:
    def __init__(
        self,
        config: CrawlConfig,
        html_output: Path,
        markdown_output: Path,
        *,
        delay: float = 0.0,
    ) -> None:
        self._config = config
        self._html_output = html_output
        self._markdown_output = markdown_output
        self._delay = delay
        self._html_output.mkdir(parents=True, exist_ok=True)
        self._markdown_output.mkdir(parents=True, exist_ok=True)

    def crawl_all(self, selected: list[str] | None, *, force: bool) -> None:
        targets = {name.lower() for name in selected} if selected else None
        for doc in self._config.documents:
            if targets and doc.name.lower() not in targets:
                continue
            self._process_document(doc, force=force or doc.update)

    def _process_document(self, doc: DocConfig, *, force: bool) -> None:
        LOG.info("Processing %s", doc.name)
        doc_slug = sanitize_filename(doc.name)
        html_doc_dir = self._html_output / doc_slug
        html_doc_dir.mkdir(parents=True, exist_ok=True)
        chapters_html_dir = html_doc_dir / "chapters"
        chapters_html_dir.mkdir(parents=True, exist_ok=True)

        md_doc_dir = self._markdown_output / doc_slug
        md_doc_dir.mkdir(parents=True, exist_ok=True)

        full_html_path = html_doc_dir / "full_documentation.html"
        if full_html_path.exists() and not force:
            LOG.info("Skipping %s (already crawled). Use --force to refresh.", doc.name)
            return

        document_id, toc_id = self._fetch_pretty_url(doc.normalized_pretty_url())
        fingerprint = self._fetch_document_map(document_id)
        toc = self._fetch_pages(document_id, fingerprint)
        sections: list[str] = []
        total = count_toc_items(toc)
        LOG.info("%s has %s sections (tocId=%s)", doc.name, total, toc_id)

        with tqdm(total=total, desc=f"Crawling {doc.name}") as progress:
            self._build_html_tree(
                toc,
                document_id,
                fingerprint,
                sections,
                chapter_html_root=chapters_html_dir,
                chapter_md_root=md_doc_dir,
                parent_prefix="",
                progress=progress,
            )

        full_html = HTML_TEMPLATE.format(title=doc.name, content="\n".join(sections))
        full_html_path.write_text(full_html, encoding="utf-8")
        LOG.info("Wrote %s", full_html_path)

        markdown_payload = md(full_html)
        markdown_path = md_doc_dir / "full_documentation.md"
        markdown_path.write_text(markdown_payload, encoding="utf-8")
        legacy_md = self._markdown_output / f"{doc_slug}.md"
        legacy_md.write_text(markdown_payload, encoding="utf-8")
        LOG.info("Wrote %s and %s", markdown_path, legacy_md)

    # --- HTTP helpers -------------------------------------------------

    def _fetch_pretty_url(self, pretty_url: str) -> tuple[str, str]:
        endpoints = [self._config.pretty_url_endpoint]
        fallback = "internal/api/webapp/pretty-url/reader"
        if fallback not in self._config.pretty_url_endpoint:
            endpoints.append(fallback)

        payload = {"prettyUrl": pretty_url, "forcedTocId": None}
        last_error: Exception | None = None
        for endpoint in endpoints:
            url = self._config.resolve(endpoint)
            try:
                response = make_request("POST", url, json_body=payload, delay=self._delay)
                data = response.json()
                return data["documentId"], data["tocId"]
            except Exception as exc:  # pragma: no cover - network/parse failures
                last_error = exc
                LOG.warning("Pretty URL lookup failed via %s: %s", url, exc)
        if last_error:
            raise last_error
        raise RuntimeError("Unable to resolve pretty URL")

    def _fetch_document_map(self, document_id: str) -> str:
        endpoint = self._config.document_map_endpoint.format(document_id=document_id)
        url = self._config.resolve(endpoint)
        response = make_request("GET", url, delay=self._delay)
        data = response.json()
        return data.get("fingerprint") or data.get("map", {}).get("fingerprint")

    def _fetch_pages(self, document_id: str, fingerprint: str) -> list[dict[str, Any]]:
        endpoint = self._config.pages_endpoint.format(document_id=document_id)
        url = self._config.resolve(endpoint)
        response = make_request("GET", url, params={"v": fingerprint}, delay=self._delay)
        payload = response.json()
        toc = payload.get("paginatedToc") or []
        if not toc:
            raise RuntimeError(f"No paginated TOC for document {document_id}")
        return toc[0].get("pageToc", [])

    def _fetch_content(self, document_id: str, topic_id: str, fingerprint: str) -> str:
        endpoint = self._config.content_endpoint.format(document_id=document_id, topic_id=topic_id)
        url = self._config.resolve(endpoint)
        params = {"target": "DESIGNED_READER", "v": fingerprint}
        response = make_request("GET", url, params=params, delay=self._delay)
        return response.text

    # --- HTML helpers -------------------------------------------------

    def _build_html_tree(
        self,
        toc: list[dict[str, Any]],
        document_id: str,
        fingerprint: str,
        sections: list[str],
        *,
        chapter_html_root: Path,
        chapter_md_root: Path,
        parent_prefix: str,
        progress: tqdm,
    ) -> None:
        for idx, item in enumerate(toc, start=1):
            title = item["title"]
            content_id = item["contentId"]
            prefix = f"{parent_prefix}{idx}" if parent_prefix else str(idx)
            html_content = self._fetch_content(document_id, content_id, fingerprint)
            section_html = self._wrap_section(item, html_content, prefix)
            sections.append(section_html)
            if not parent_prefix:
                self._write_chapter_files(
                    chapter_html_root,
                    chapter_md_root,
                    prefix,
                    title,
                    section_html,
                )
            progress.update(1)

            children = item.get("children") or []
            if children:
                self._build_html_tree(
                    children,
                    document_id,
                    fingerprint,
                    sections,
                    chapter_html_root=chapter_html_root,
                    chapter_md_root=chapter_md_root,
                    parent_prefix=f"{prefix}.",
                    progress=progress,
                )

    def _wrap_section(self, toc_entry: dict[str, Any], raw_html: str, prefix: str) -> str:
        soup = BeautifulSoup(raw_html, "html.parser")
        content_div = soup.find("div", class_="content-locale-en-US") or soup
        level = toc_entry.get("topic-level") or prefix.count(".") + 1
        title = toc_entry["title"]
        content_id = toc_entry["contentId"]
        return (
            f"<section id='{content_id}'>"
            f"<h{level}>{prefix} {title}</h{level}>"
            f"{content_div}"
            "</section>"
        )

    def _write_chapter_files(
        self,
        chapter_html_root: Path,
        chapter_md_root: Path,
        prefix: str,
        title: str,
        section_html: str,
    ) -> None:
        folder = f"{prefix}_{sanitize_filename(title)}"
        html_dir = chapter_html_root / folder
        md_dir = chapter_md_root / folder
        html_dir.mkdir(parents=True, exist_ok=True)
        md_dir.mkdir(parents=True, exist_ok=True)

        chapter_title = f"{prefix} {title}"
        html_payload = HTML_TEMPLATE.format(title=chapter_title, content=section_html)
        (html_dir / "chapter.html").write_text(html_payload, encoding="utf-8")

        markdown_text = md(section_html)
        md_path = md_dir / "chapter.md"
        md_path.write_text(f"# {chapter_title}\n\n{markdown_text}", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if not args.doctree.exists():
        raise SystemExit(f"Doctree file not found: {args.doctree}")

    payload = json.loads(args.doctree.read_text(encoding="utf-8"))
    config = CrawlConfig.from_json(payload)
    crawler = CortexDocCrawler(
        config,
        html_output=args.output_html,
        markdown_output=args.output_md,
        delay=args.delay,
    )
    crawler.crawl_all(selected=args.documents, force=args.force)


if __name__ == "__main__":
    main()
