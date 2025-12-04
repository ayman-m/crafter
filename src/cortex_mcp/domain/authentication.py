"""Authentication helpers for protecting the MCP server."""

from __future__ import annotations

import secrets
from typing import Sequence

from fastmcp.server.auth import AccessToken, AuthProvider


class EnvTokenAuthProvider(AuthProvider):
    """Simple AuthProvider that validates a static bearer token from the environment."""

    def __init__(
        self,
        *,
        token: str,
        base_url: str | None = None,
        required_scopes: Sequence[str] | None = None,
    ) -> None:
        if not token:
            raise ValueError("A non-empty bearer token must be provided.")
        self._token = token
        super().__init__(
            base_url=base_url,
            required_scopes=list(required_scopes or []),
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        """Validate the presented bearer token."""

        if not token or not secrets.compare_digest(token, self._token):
            return None
        return AccessToken(
            token=token,
            client_id="env-token-client",
            scopes=list(self.required_scopes),
            expires_at=None,
            resource=None,
        )
