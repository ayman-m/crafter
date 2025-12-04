"""MCP layer initialization for Cortex Content MCP server."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from fastmcp import FastMCP

from cortex_mcp.domain.authentication import EnvTokenAuthProvider

__all__ = ["mcp"]

load_dotenv()


def _parse_scopes(raw: str | None) -> list[str]:
    if not raw:
        return []
    scopes: list[str] = []
    for part in raw.replace(";", ",").split(","):
        scope = part.strip()
        if scope:
            scopes.append(scope)
    return scopes


def _build_auth_provider() -> EnvTokenAuthProvider | None:
    token = os.getenv("MCP_AUTH_TOKEN") or os.getenv("MCP_ACCESS_TOKEN")
    if not token:
        return None
    base_url = os.getenv("MCP_PUBLIC_URL")
    scopes = _parse_scopes(os.getenv("MCP_AUTH_SCOPES"))
    return EnvTokenAuthProvider(token=token, base_url=base_url, required_scopes=scopes or ["cortex:xql"])


_auth_provider = _build_auth_provider()

mcp_kwargs: dict[str, object] = {
    "name": "Cortex Content MCP",
    "instructions": "Use docs and examples resources to answer Cortex content questions.",
}
if _auth_provider:
    mcp_kwargs["auth"] = _auth_provider

mcp = FastMCP(**mcp_kwargs)

# Ensure resource/tool definitions are registered on import.
from cortex_mcp.mcp import prompts as _mcp_prompts  # noqa: F401,E402
from cortex_mcp.mcp import resources as _mcp_resources  # noqa: F401,E402
from cortex_mcp.mcp import tools as _mcp_tools  # noqa: F401,E402
