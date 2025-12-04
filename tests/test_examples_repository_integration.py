"""Integration tests for ExamplesRepository with real Chroma + embeddings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cortex_mcp.domain.examples_repository import ExamplesRepository
from scripts.seed_examples_repository import seed_examples_repository

RUN_INTEGRATION = os.getenv("EXAMPLES_REPO_INTEGRATION") == "1"
if not RUN_INTEGRATION:  # pragma: no cover - opt-in
    pytest.skip(
        "Set EXAMPLES_REPO_INTEGRATION=1 to run ExamplesRepository integration tests",
        allow_module_level=True,
    )

EMBED_MODEL = os.getenv("EXAMPLES_REPO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@pytest.mark.integration
def test_repository_seed_and_search(tmp_path: Path) -> None:
    """Seed real Chroma DB and ensure search works end-to-end."""

    source_dir = Path(__file__).resolve().parents[1] / "processed_examples"
    data_path = tmp_path / "data" / "examples.json"
    chroma_path = tmp_path / "chroma"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    chroma_path.mkdir(parents=True, exist_ok=True)

    results = seed_examples_repository(
        source_dir=source_dir,
        data_path=data_path,
        chroma_path=chroma_path,
        embedding_model_name=EMBED_MODEL,
    )
    assert results["appended"] > 0 or results["skipped"] > 0

    repo = ExamplesRepository(
        data_path=data_path,
        chroma_path=chroma_path,
        embedding_model_name=EMBED_MODEL,
    )
    hits = repo.search_examples("playbook", top_k=5)

    assert hits, "Expected seeded repository to return results"
    assert any("playbook" in example.id for example in hits)
