"""Pydantic models describing validation, simulation, and fake data results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Outcome of validating Cortex content."""

    success: bool
    content_type: str
    summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Outcome of simulating Cortex content or playbooks."""

    success: bool
    content_type: str
    summary: str | None = None
    records_processed: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FakeDataResult(BaseModel):
    """Description of a generated fake data batch."""

    scenario: str
    content_type: str
    count: int
    records: list[dict[str, Any]]
