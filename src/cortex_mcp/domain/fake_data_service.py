"""Utility for generating synthetic Cortex content test data."""

from __future__ import annotations

import random
import time
from typing import Any, Iterable

SCENARIO_HINTS = {
    "phishing": [
        "suspicious attachment",
        "credential harvest",
        "spoofed sender domain",
    ],
    "lateral_movement": [
        "admin share access",
        "RDP brute-force",
        "new service creation",
    ],
    "generic": [
        "anomalous behavior",
        "high-risk alert",
        "automatic containment required",
    ],
}


class FakeDataService:
    """Generate lightweight fake records for validation/simulation."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def generate(self, scenario: str | None, content_type: str, count: int = 5) -> list[dict[str, Any]]:
        """Return a list of synthetic records describing a scenario."""

        scenario_key = (scenario or self._default_scenario(content_type)).lower()
        hints = SCENARIO_HINTS.get(scenario_key, SCENARIO_HINTS["generic"])
        now = int(time.time())

        records: list[dict[str, Any]] = []
        for index in range(max(1, count)):
            hint = hints[index % len(hints)]
            records.append(
                {
                    "scenario": scenario_key,
                    "content_type": content_type,
                    "timestamp": now - (index * 60),
                    "description": f"Simulated {hint}",
                    "severity": self._random.choice(["low", "medium", "high"]),
                }
            )
        return records

    @staticmethod
    def _default_scenario(content_type: str) -> str:
        mapping: dict[str, Iterable[str]] = {
            "playbook": ("phishing",),
            "xql_query": ("lateral_movement",),
            "correlation_rule": ("lateral_movement",),
            "widget": ("generic",),
        }
        for key, scenarios in mapping.items():
            if content_type.startswith(key):
                return next(iter(scenarios))
        return "generic"
