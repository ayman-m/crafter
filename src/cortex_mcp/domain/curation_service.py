"""Service that promotes validated Cortex content into the examples store."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from cortex_mcp.domain.content_testing_service import ContentTestingService
from cortex_mcp.domain.examples_repository import Example, ExamplesRepository


class CurationError(RuntimeError):
    """Raised when curation cannot proceed."""


class CurationService:
    """Persist validated Cortex content into the examples repository."""

    def __init__(
        self,
        repository: ExamplesRepository,
        testing_service: ContentTestingService | None = None,
    ) -> None:
        self._repository = repository
        self._testing_service = testing_service

    async def curate(
        self,
        content_type: str,
        content: str,
        description: str,
        tags: list[str] | None = None,
        *,
        must_be_valid: bool = True,
    ) -> dict[str, Any]:
        """Store a new example after optional validation."""

        normalized_tags = [tag for tag in (tags or []) if tag]
        validation_result = None

        if must_be_valid and self._testing_service is not None:
            test_result = await self._testing_service.test_content(
                content_type,
                content,
                run_simulation=False,
                generate_fake_data=False,
            )
            validation_result = test_result.get("validation")
            if not validation_result or not validation_result.get("success", False):
                return {
                    "status": "validation_failed",
                    "validation": validation_result,
                }

        example = Example(
            id=str(uuid4()),
            type=content_type,
            description=description,
            content=content,
            tags=normalized_tags,
        )
        saved = self._repository.append_example(example)
        return {
            "status": "curated",
            "example_id": saved.id,
            "type": saved.type,
            "description": saved.description,
            "tags": saved.tags,
            "validation": validation_result,
        }
