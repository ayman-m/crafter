"""Service coordinating validation, fake data generation, and simulation."""

from __future__ import annotations

from typing import Any

from cortex_mcp.domain.fake_data_service import FakeDataService
from cortex_mcp.domain.models.validation import SimulationResult, ValidationResult
from cortex_mcp.domain.xsiam_client import XsiamClient, XsiamClientError

XQL_TYPES = {"xql_query", "correlation_rule", "widget"}


class ContentTestingError(RuntimeError):
    """Raised when content testing cannot proceed."""


class ContentTestingService:
    """High-level orchestrator for validation + fake data + simulation."""

    def __init__(self, xsiam_client: XsiamClient, fake_data_service: FakeDataService) -> None:
        self._xsiam_client = xsiam_client
        self._fake_data_service = fake_data_service

    async def test_content(
        self,
        content_type: str,
        content: str,
        *,
        run_simulation: bool = True,
        generate_fake_data: bool = False,
        fake_data_scenario: str | None = None,
        fake_data_count: int = 5,
        fetch_results: bool = True,
    ) -> dict[str, Any]:
        """Validate and optionally simulate Cortex content."""

        validation = await self._validate(
            content_type,
            content,
            fetch_results=fetch_results,
        )
        if not validation.success:
            return {
                "status": "validation_failed",
                "validation": validation.model_dump(),
                "simulation": None,
                "fake_data": None,
            }

        fake_data = None
        if generate_fake_data:
            fake_data = self._fake_data_service.generate(fake_data_scenario, content_type, fake_data_count)

        simulation_result = None
        status = "validated_only"
        if run_simulation:
            try:
                simulation_result = await self._simulate(content_type, content, fake_data)
                status = "simulated" if simulation_result.success else "simulation_failed"
            except XsiamClientError as exc:
                return {
                    "status": "simulation_failed",
                    "validation": validation.model_dump(),
                    "simulation": None,
                    "fake_data": fake_data,
                    "error": str(exc),
                }

        return {
            "status": status,
            "validation": validation.model_dump(),
            "simulation": simulation_result.model_dump() if simulation_result else None,
            "fake_data": fake_data,
        }

    async def _validate(
        self,
        content_type: str,
        content: str,
        *,
        fetch_results: bool,
    ) -> ValidationResult:
        if content_type in XQL_TYPES:
            return await self._xsiam_client.validate_xql(
                content_type,
                content,
                fetch_results=fetch_results,
            )
        if content_type == "playbook":
            return await self._xsiam_client.validate_playbook(content)
        return ValidationResult(
            success=True,
            content_type=content_type,
            summary="No server-side validation available; marked as warning only.",
            warnings=["Validation skipped because content type is not supported."],
        )

    async def _simulate(
        self,
        content_type: str,
        content: str,
        fake_data: list[dict[str, Any]] | None,
    ) -> SimulationResult:
        if content_type in XQL_TYPES:
            return await self._xsiam_client.simulate_xql(content_type, content, fake_data=fake_data)
        if content_type == "playbook":
            return await self._xsiam_client.simulate_playbook(content, fake_data=fake_data)
        return SimulationResult(
            success=False,
            content_type=content_type,
            summary="Simulation is not supported for this content type.",
            records_processed=0,
        )
