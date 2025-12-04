"""Main entrypoint for the Cortex Content MCP server."""

from __future__ import annotations

import logging
import os

from cortex_mcp.mcp import mcp  # Registers resources/tools on import

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def run() -> None:
    """Start the FastMCP server using environment-driven transport settings."""

    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8000"))
        path = os.getenv("MCP_HTTP_PATH")
        kwargs = {"host": host, "port": port}
        if path:
            kwargs["path"] = path
        mcp.run(transport="http", **kwargs)
    elif transport == "sse":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    run()
