"""Tests for ExamplesRepository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from cortex_mcp.domain.examples_repository import Example, ExamplesRepository


class DummySentenceTransformer:
    """Deterministic embedding model for tests."""

    def __init__(self, model_name: str, **_: Any) -> None:  # pragma: no cover - trivial init
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts: Iterable[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                np.array(
                    [
                        1.0 if "xql" in lower else 0.0,
                        1.0 if "playbook" in lower else 0.0,
                        float(len(lower)),
                    ],
                    dtype="float32",
                )
            )
        matrix = np.stack(vectors)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


@pytest.fixture(autouse=True)
def stub_vector_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force repository to use dummy embeddings + numpy fallback."""

    import cortex_mcp.domain.examples_repository as repo_module

    monkeypatch.setattr(repo_module, "SentenceTransformer", DummySentenceTransformer)
    monkeypatch.setattr(repo_module, "chromadb", None)
    monkeypatch.setattr(repo_module, "faiss", None)


@pytest.fixture
def repo_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Create isolated paths for data and chroma persistence."""

    data_dir = tmp_path / "examples"
    data_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    return data_dir / "examples.json", chroma_dir


def seed_examples_file(path: Path) -> None:
    """Write a deterministic set of examples to disk."""

    examples = [
        {
            "id": "ex-xql",
            "type": "xql_query",
            "description": "XQL hunting query for suspicious endpoint behavior",
            "content": "Run XQL query to detect anomalies",
            "tags": ["xql", "query"],
        },
        {
            "id": "ex-playbook",
            "type": "playbook",
            "description": "Playbook for phishing investigation",
            "content": "Playbook steps include manual review",
            "tags": ["playbook", "automation"],
        },
        {
            "id": "ex-layout",
            "type": "incident_layout",
            "description": "Incident layout for SOC dashboard",
            "content": "Layout includes severity widgets",
            "tags": ["layout"],
        },
    ]
    path.write_text(json.dumps(examples), encoding="utf-8")


def create_repo(data_path: Path, chroma_path: Path) -> ExamplesRepository:
    """Helper to instantiate repository with deterministic paths."""

    return ExamplesRepository(data_path=data_path, chroma_path=chroma_path)


def test_search_examples_returns_relevant_results(repo_paths: tuple[Path, Path]) -> None:
    """search_examples should surface the most relevant document."""

    data_path, chroma_path = repo_paths
    seed_examples_file(data_path)
    repo = create_repo(data_path, chroma_path)

    hits = repo.search_examples("Need an XQL query", top_k=2)

    assert hits
    assert hits[0].id == "ex-xql"
    assert all(isinstance(hit, Example) for hit in hits)


def test_append_example_persists_and_updates_index(repo_paths: tuple[Path, Path]) -> None:
    """Appending should persist to disk and make the example searchable."""

    data_path, chroma_path = repo_paths
    seed_examples_file(data_path)
    repo = create_repo(data_path, chroma_path)

    new_example = Example(
        id="ex-widget",
        type="widget",
        description="Widget for executive dashboard",
        content="Playbook widget renders KPI charts",
        tags=["widget", "playbook"],
    )
    repo.append_example(new_example)

    persisted = json.loads(data_path.read_text(encoding="utf-8"))
    assert any(item["id"] == "ex-widget" for item in persisted)

    hits = repo.search_examples("Need a playbook widget", top_k=3)
    assert hits and hits[0].id == "ex-widget"


def test_repository_reloads_examples_from_disk(repo_paths: tuple[Path, Path]) -> None:
    """Creating a new repository instance should reload saved examples."""

    data_path, chroma_path = repo_paths
    seed_examples_file(data_path)
    repo = create_repo(data_path, chroma_path)

    repo.append_example(
        Example(
            id="ex-case",
            type="case_type",
            description="Case template for VIP investigations",
            content="Case definition outlines response steps",
            tags=["case", "vip"],
        )
    )

    # Recreate repository pointing at the same files to simulate restart.
    repo_reloaded = create_repo(data_path, chroma_path)
    results = repo_reloaded.search_examples("VIP case template", top_k=2)

    assert any(example.id == "ex-case" for example in results)
