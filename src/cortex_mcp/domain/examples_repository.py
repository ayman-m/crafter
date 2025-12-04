"""Repository for Cortex content examples backed by JSON + vector search."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

try:  # pragma: no cover - optional dependency
    import chromadb
except ImportError:  # pragma: no cover - environment without Chroma
    chromadb = None

try:  # pragma: no cover - optional dependency
    import faiss
except ImportError:  # pragma: no cover - faiss not installed
    faiss = None

logger = logging.getLogger(__name__)

ExampleType = Literal[
    "xql_query",
    "correlation_rule",
    "widget",
    "xql_parsing_rule",
    "xql_modeling_rule",
    "playbook",
    "automation_script",
    "integration",
    "incident_layout",
    "custom_field",
    "indicator_type",
    "case_type",
]


class Example(BaseModel):
    """Domain representation of a Cortex content example."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: ExampleType
    description: str
    content: str
    tags: list[str] = Field(default_factory=list)

    def document_text(self) -> str:
        """Combine description/content/tags into a single searchable document."""

        tags_block = " ".join(self.tags)
        return "\n\n".join(part for part in (self.description, self.content, tags_block) if part)


class ExamplesRepository:
    """Loads, stores, and semantically searches Cortex content examples."""

    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v2-moe"

    def __init__(
        self,
        *,
        data_path: str | Path = Path("data/examples/examples.json"),
        chroma_path: str | Path = Path("data/chroma"),
        embedding_model_name: str | None = None,
        embedding_model_trust_remote_code: bool | None = None,
        collection_name: str = "content_examples",
    ) -> None:
        self._lock = threading.RLock()
        self._data_path = Path(data_path)
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._chroma_path = Path(chroma_path)
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name or self.DEFAULT_MODEL
        if embedding_model_trust_remote_code is None:
            self._embedding_model_trust_remote_code = self._embedding_model_name.startswith("nomic-ai/")
        else:
            self._embedding_model_trust_remote_code = embedding_model_trust_remote_code

        self._examples: dict[str, Example] = {}
        self._embedding_model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None

        self._chroma_client: chromadb.PersistentClient | None = None
        self._chroma_collection = None

        self._faiss_index = None
        self._faiss_ids: list[str] = []
        self._numpy_matrix: np.ndarray | None = None
        self._numpy_ids: list[str] = []

        self._load_examples()
        self._init_vector_store()

    # ------------------------------------------------------------------ Public API

    def search_examples(self, query: str, top_k: int = 3) -> list[Example]:
        """Return the most relevant examples for the given query."""

        if not query.strip():
            return []

        with self._lock:
            if not self._examples:
                return []
            top_k = max(1, min(top_k, len(self._examples)))
            embeddings = self._embed_texts([query])
            vector = embeddings[0]
            results: list[str]
            if self._chroma_collection is not None:
                chroma_response = self._chroma_collection.query(
                    query_embeddings=[vector.tolist()],
                    n_results=top_k,
                    include=["metadatas"],
                )
                metadata_list = chroma_response.get("metadatas", [[]])[0] or []
                ids_from_metadata = [
                    metadata.get("id")
                    for metadata in metadata_list
                    if metadata and metadata.get("id")
                ]
                if ids_from_metadata:
                    results = ids_from_metadata
                else:
                    ids_raw = chroma_response.get("ids", [[]])[0] or []
                    results = ids_raw
            elif self._faiss_index is not None:
                scores, indices = self._faiss_index.search(
                    np.asarray([vector], dtype="float32"),
                    k=top_k,
                )
                idx_row = indices[0]
                results = [
                    self._faiss_ids[i]
                    for i in idx_row
                    if 0 <= i < len(self._faiss_ids)
                ]
            else:
                matrix = self._numpy_matrix
                if matrix is None or not len(self._numpy_ids):
                    return []
                similarities = matrix @ vector
                order = np.argsort(similarities)[::-1][:top_k]
                results = [self._numpy_ids[i] for i in order]

            hits = [self._examples[example_id] for example_id in results if example_id in self._examples]
            logger.debug(
                "Examples search completed",
                extra={"query": query, "top_k": top_k, "hits": [hit.id for hit in hits]},
            )
            return hits

    def get_example(self, example_id: str) -> Example | None:
        """Return a single example by id without hitting the vector index."""

        with self._lock:
            return self._examples.get(example_id)

    def append_example(self, example: Example) -> Example:
        """Persist a new example and update the vector index."""

        with self._lock:
            if example.id in self._examples:
                raise ValueError(f"Example with id '{example.id}' already exists.")
            self._examples[example.id] = example
            self._save_examples()
            embedding = self._embed_texts([example.document_text()])[0]
            self._upsert_embedding(example, embedding)
            logger.info("Example appended", extra={"example_id": example.id, "type": example.type})
            return example

    # ------------------------------------------------------------------ Internal helpers

    def _load_examples(self) -> None:
        if not self._data_path.exists():
            logger.info("Examples file %s not found; starting empty.", self._data_path)
            self._examples = {}
            return

        try:
            raw_text = self._data_path.read_text(encoding="utf-8")
            parsed = json.loads(raw_text)
            if not isinstance(parsed, list):
                raise ValueError("Root JSON must be a list.")
        except (OSError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load examples from {self._data_path}: {exc}") from exc

        loaded: dict[str, Example] = {}
        for item in parsed:
            example = Example.model_validate(item)
            loaded[example.id] = example
        self._examples = loaded
        logger.info("Loaded %s examples from %s", len(self._examples), self._data_path)

    def _save_examples(self) -> None:
        payload = [example.model_dump() for example in self._examples.values()]
        self._data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Persisted %s examples to %s", len(payload), self._data_path)

    def _init_vector_store(self) -> None:
        self._embedding_model = SentenceTransformer(
            self._embedding_model_name,
            trust_remote_code=self._embedding_model_trust_remote_code,
        )
        self._embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
        logger.info(
            "SentenceTransformer initialized",
            extra={"model": self._embedding_model_name, "dim": self._embedding_dim},
        )

        if chromadb is not None:
            try:
                client = chromadb.PersistentClient(path=str(self._chroma_path))
                collection = client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._chroma_client = client
                self._chroma_collection = collection
                self._sync_chroma_collection()
                logger.info(
                    "Chroma vector store ready",
                    extra={"collection": self._collection_name, "path": str(self._chroma_path)},
                )
                return
            except Exception:  # pragma: no cover - fallback path
                logger.exception("Failed to initialize Chroma. Falling back to local similarity.")

        self._chroma_collection = None
        self._rebuild_fallback_index()

    def _sync_chroma_collection(self) -> None:
        if self._chroma_collection is None:
            return

        existing = self._chroma_collection.get(include=[], limit=None)
        existing_ids = set(existing.get("ids") or [])
        current_ids = set(self._examples.keys())

        stale_ids = list(existing_ids - current_ids)
        if stale_ids:
            self._chroma_collection.delete(ids=stale_ids)

        missing_ids = list(current_ids - existing_ids)
        if missing_ids:
            examples = [self._examples[i] for i in missing_ids]
            documents = [ex.document_text() for ex in examples]
            embeddings = self._embed_texts(documents)
            self._chroma_collection.add(
                ids=[ex.id for ex in examples],
                documents=documents,
                embeddings=[vec.tolist() for vec in embeddings],
                metadatas=[self._build_metadata(ex) for ex in examples],
            )

    def _rebuild_fallback_index(self) -> None:
        documents = [example.document_text() for example in self._examples.values()]
        if not documents:
            self._faiss_index = None
            self._faiss_ids = []
            self._numpy_matrix = None
            self._numpy_ids = []
            return

        embeddings = self._embed_texts(documents)
        ids = list(self._examples.keys())

        if faiss is not None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.asarray(embeddings, dtype="float32"))
            self._faiss_index = index
            self._faiss_ids = ids
            self._numpy_matrix = None
            self._numpy_ids = []
            logger.info("FAISS fallback index built with %s vectors.", len(ids))
        else:
            self._faiss_index = None
            self._faiss_ids = []
            self._numpy_matrix = np.asarray(embeddings, dtype="float32")
            self._numpy_ids = ids
            logger.info("NumPy similarity matrix built with %s vectors.", len(ids))

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not initialized.")
        embeddings = self._embedding_model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype("float32")

    def _upsert_embedding(self, example: Example, embedding: np.ndarray) -> None:
        if self._chroma_collection is not None:
            self._chroma_collection.add(
                ids=[example.id],
                documents=[example.document_text()],
                embeddings=[embedding.tolist()],
                metadatas=[self._build_metadata(example)],
            )
            return

        if self._faiss_index is not None:
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            self._faiss_index.add(np.asarray(embedding, dtype="float32"))
            self._faiss_ids.append(example.id)
        elif self._numpy_matrix is not None:
            self._numpy_matrix = np.vstack([self._numpy_matrix, embedding])
            self._numpy_ids.append(example.id)
        else:
            # First vector in fallback mode.
            self._numpy_matrix = embedding.reshape(1, -1)
            self._numpy_ids = [example.id]

    def _build_metadata(self, example: Example) -> dict[str, str]:
        tags_value = ", ".join(example.tags)
        return {"type": example.type, "tags": tags_value, "id": example.id}
