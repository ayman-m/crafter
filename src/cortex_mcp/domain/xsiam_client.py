"""Simplified XsiamClient abstraction for validation and simulation."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from cortex_mcp.domain.models.validation import SimulationResult, ValidationResult
from dotenv import load_dotenv


class XsiamClientError(RuntimeError):
    """Raised when the simulated XSIAM backend encounters a failure."""


class XsiamClient:
    """Lightweight stand-in for the real Cortex XSIAM/XDR API client."""

    def __init__(self, *, latency_seconds: float = 0.0) -> None:
        self._latency_seconds = latency_seconds
        load_dotenv()
        self._base_url = os.getenv("XSIAM_STANDARD_URL")
        self._api_key = os.getenv("XSIAM_STANDARD_KEY")
        self._auth_id = os.getenv("XSIAM_STANDARD_ID")

    async def validate_xql(
        self,
        content_type: str,
        content: str,
        *,
        fetch_results: bool = True,
    ) -> ValidationResult:
        """Execute a lightweight XQL query run against XSIAM."""

        await self._maybe_wait()
        response = await self.run_xql_query(content)
        status_code = response["status_code"]
        body = response["body"]
        reply = body.get("reply", body)
        reply = body.get("reply", body)
        if status_code == 200:
            query_id = reply if isinstance(reply, str) else reply.get("query_id") or reply.get("reply")
            warnings: list[str] = []
            results_payload: dict[str, Any] | None = None
            if fetch_results:
                if not query_id:
                    warnings.append("Query succeeded but result identifier was missing; unable to fetch results.")
                else:
                    results_payload, fetch_warnings = await self._fetch_results_with_poll(query_id)
                    warnings.extend(fetch_warnings)
            return ValidationResult(
                success=True,
                content_type=content_type,
                summary="XQL query executed successfully.",
                warnings=warnings,
                metadata={
                    "xsiam": reply,
                    "results": results_payload,
                    "results_fetch_requested": fetch_results,
                },
            )
        errors = []
        err_extra = reply.get("err_extra", {})
        parse_err = err_extra.get("parse_err")
        if parse_err:
            errors.append(
                parse_err.get("message")
                or f"Parse error at line {parse_err.get('line')} column {parse_err.get('column')}."
            )
        err_msg = reply.get("err_msg")
        if err_msg:
            errors.append(err_msg)
        if not errors:
            errors.append("Query rejected.")
        return ValidationResult(
            success=False,
            content_type=content_type,
            summary=f"XSIAM returned HTTP {status_code}.",
            errors=errors,
            metadata={"xsiam": reply},
        )

    async def validate_playbook(self, content: str) -> ValidationResult:
        """Basic structure validation for playbooks."""

        await self._maybe_wait()
        lowered = content.lower()
        errors: list[str] = []
        if "taskid" not in lowered:
            errors.append("At least one task must be defined (missing 'taskid').")
        if "playbook" not in lowered:
            errors.append("Payload does not look like a Cortex playbook.")

        if errors:
            return ValidationResult(
                success=False,
                content_type="playbook",
                summary="Playbook validation failed.",
                errors=errors,
            )
        return ValidationResult(
            success=True,
            content_type="playbook",
            summary="Playbook structure validated.",
        )

    async def simulate_xql(
        self,
        content_type: str,
        content: str,
        *,
        fake_data: list[dict[str, Any]] | None = None,
    ) -> SimulationResult:
        """Pretend to execute a simulation for XQL or correlation content."""

        await self._maybe_wait()
        records = fake_data or []
        success = "fail_sim" not in content.lower()
        summary = (
            f"Simulated {len(records) or 'live'} records for {content_type}."
        )
        warnings: list[str] = []
        if not records:
            warnings.append("Simulation used live-like evaluation; provide fake data if possible.")
        return SimulationResult(
            success=success,
            content_type=content_type,
            summary=summary,
            records_processed=len(records),
            warnings=warnings,
        )

    async def simulate_playbook(
        self,
        content: str,
        *,
        fake_data: list[dict[str, Any]] | None = None,
    ) -> SimulationResult:
        """Pretend to run a playbook simulation."""

        await self._maybe_wait()
        if "simulate_error" in content.lower():
            raise XsiamClientError("Playbook automation failed during simulation.")
        records = fake_data or []
        summary = f"Playbook executed against {len(records) or 'synthetic'} alerts."
        return SimulationResult(
            success=True,
            content_type="playbook",
            summary=summary,
            records_processed=len(records),
        )

    async def run_xql_query(self, query: str) -> dict[str, Any]:
        """Execute a single XQL query via the public API."""

        if not (self._base_url and self._api_key and self._auth_id):
            raise XsiamClientError("XSIAM credentials not configured.")

        payload = {
            "request_data": {
                "query": query,
                "timeframe": {"relativeTime": 10},
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-xdr-auth-id": str(self._auth_id),
            "Authorization": self._api_key,
        }
        url = f"{self._base_url.rstrip('/')}/public_api/v1/xql/start_xql_query"

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json=payload, headers=headers)

        body = {}
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        return {"status_code": response.status_code, "body": body}

    async def get_query_results(self, query_id: str | None) -> dict[str, Any]:
        if not query_id:
            raise XsiamClientError("Missing query_id for results retrieval.")
        await asyncio.sleep(5)

        payload = {"request_data": {"query_id": query_id, "format": "csv"}}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-xdr-auth-id": str(self._auth_id),
            "Authorization": self._api_key,
        }
        url = f"{self._base_url.rstrip('/')}/public_api/v1/xql/get_query_results"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json=payload, headers=headers)
        try:
            return response.json().get("reply", {})
        except ValueError:
            return {"status": "error", "raw": response.text}

    async def _fetch_results_with_poll(
        self,
        query_id: str,
        *,
        attempts: int = 3,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Poll get_query_results until success or attempts exhausted."""

        payload: dict[str, Any] | None = None
        warnings: list[str] = []
        try:
            for attempt in range(1, attempts + 1):
                payload = await self.get_query_results(query_id)
                status = (payload or {}).get("status")
                normalized = status.upper() if isinstance(status, str) else None
                if not normalized or normalized == "SUCCESS":
                    return payload, warnings
                if normalized in {"PENDING", "IN_PROGRESS", "RUNNING"} and attempt < attempts:
                    continue
                warnings.append(
                    f"Query succeeded but results endpoint for query_id {query_id} returned status '{status}' "
                    f"after {attempts} attempts. Try rerunning or check XSIAM for query details."
                )
                return payload, warnings
        except Exception as exc:  # pragma: no cover - network failure
            warnings.append(f"Query succeeded but fetching results for query_id {query_id} failed: {exc}")
            return {"status": "ERROR", "message": str(exc)}, warnings

        warnings.append(
            f"Query succeeded but results endpoint for query_id {query_id} never returned a status after {attempts} attempts."
        )
        return payload, warnings

    async def _maybe_wait(self) -> None:
        if self._latency_seconds > 0:
            await asyncio.sleep(self._latency_seconds)
