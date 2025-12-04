# Cortex Content MCP (Skeleton)

This repository is a **skeleton** for a Content Engineering MCP server for Palo Alto Networks Cortex (XSIAM/XDR/XSOAR).

- No business logic is implemented yet.
- All Python modules contain only placeholders and TODOs.
- All implementation work is meant to be driven by Codex/ChatGPT using the prompt playbooks in `docs/`.

## How to use this skeleton

1. Open this folder in VS Code.
2. Start with the prompt files in `docs/` (e.g. `prompts_phase1_docs_examples.md`).
3. For each file under `src/` and `rft/`, follow the relevant prompts to generate code and tests.

## Docker Usage

Build the container image (copies the repo, installs all Python dependencies, and exposes the MCP server):

```bash
docker build -t cortex-mcp .
```

Run the server over HTTP (defaults to port `8000`). All configuration can be supplied either through a mounted `.env` file or via standard environment variables:

```bash
# using an env file (e.g., secrets + MCP_AUTH_TOKEN, CORTEX_* values, etc.)
docker run --env-file .env -p 8000:8000 cortex-mcp

# or pass only the variables you need
docker run -p 8000:8000 \
  -e MCP_AUTH_TOKEN=changeme \
  -e CORTEX_DOCS_BASE_URL=https://docs-cortex.paloaltonetworks.com \
  cortex-mcp
```

By default the container runs `python -m cortex_mcp.main`, which honours both `.env` values (loaded from the working directory) and any environment variables set by Docker/Kubernetes. Override `MCP_TRANSPORT`, `MCP_HOST`, or `MCP_PORT` if you need to change the exposed protocol/port.
