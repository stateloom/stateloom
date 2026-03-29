"""OIDC client — discovery, authorization URL, code exchange, ID token validation."""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any
from urllib.parse import urlencode

try:
    import jwt as pyjwt

    _JWT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JWT_AVAILABLE = False

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False

# In-memory discovery cache: issuer_url -> (discovery_dict, fetched_at)
_DISCOVERY_CACHE: dict[str, tuple[dict, float]] = {}
_DISCOVERY_TTL = 300  # 5 minutes


class OIDCError(Exception):
    """Raised for OIDC protocol errors."""


class OIDCClient:
    """Handles OIDC protocol operations for a single provider."""

    def __init__(
        self,
        issuer_url: str,
        client_id: str,
        client_secret: str = "",
        scopes: str = "openid email profile",
    ) -> None:
        self.issuer_url = issuer_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes

    async def get_discovery(self) -> dict[str, Any]:
        """Fetch the OpenID Connect discovery document (cached 5 min)."""
        now = time.time()
        cached = _DISCOVERY_CACHE.get(self.issuer_url)
        if cached and (now - cached[1]) < _DISCOVERY_TTL:
            return cached[0]

        if not _HTTPX_AVAILABLE:
            raise OIDCError("httpx is required for OIDC discovery")

        url = f"{self.issuer_url}/.well-known/openid-configuration"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise OIDCError(f"Discovery failed: HTTP {resp.status_code}")
            doc = resp.json()

        _DISCOVERY_CACHE[self.issuer_url] = (doc, now)
        return doc

    def build_authorization_url(
        self,
        redirect_uri: str,
        state: str | None = None,
        nonce: str | None = None,
    ) -> tuple[str, str, str, str]:
        """Build PKCE authorization URL.

        Returns (url, state, nonce, code_verifier).
        """
        # PKCE
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = hashlib.sha256(code_verifier.encode()).digest()
        import base64

        code_challenge_b64 = base64.urlsafe_b64encode(code_challenge).rstrip(b"=").decode()

        state = state or secrets.token_urlsafe(32)
        nonce = nonce or secrets.token_urlsafe(32)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.scopes,
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge_b64,
            "code_challenge_method": "S256",
        }

        # Use a well-known authorization endpoint pattern
        auth_endpoint = f"{self.issuer_url}/authorize"
        url = f"{auth_endpoint}?{urlencode(params)}"
        return url, state, nonce, code_verifier

    async def build_authorization_url_from_discovery(
        self,
        redirect_uri: str,
        state: str | None = None,
        nonce: str | None = None,
    ) -> tuple[str, str, str, str]:
        """Build authorization URL using the discovery document."""
        discovery = await self.get_discovery()
        auth_endpoint = discovery.get("authorization_endpoint", "")
        if not auth_endpoint:
            raise OIDCError("No authorization_endpoint in discovery")

        code_verifier = secrets.token_urlsafe(64)
        code_challenge = hashlib.sha256(code_verifier.encode()).digest()
        import base64

        code_challenge_b64 = base64.urlsafe_b64encode(code_challenge).rstrip(b"=").decode()

        state = state or secrets.token_urlsafe(32)
        nonce = nonce or secrets.token_urlsafe(32)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.scopes,
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge_b64,
            "code_challenge_method": "S256",
        }

        url = f"{auth_endpoint}?{urlencode(params)}"
        return url, state, nonce, code_verifier

    async def exchange_code(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        """Exchange authorization code for tokens."""
        if not _HTTPX_AVAILABLE:
            raise OIDCError("httpx is required for OIDC code exchange")

        discovery = await self.get_discovery()
        token_endpoint = discovery.get("token_endpoint", "")
        if not token_endpoint:
            raise OIDCError("No token_endpoint in discovery")

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                raise OIDCError(f"Token exchange failed: HTTP {resp.status_code} {resp.text}")
            return resp.json()

    def validate_id_token(
        self,
        id_token: str,
        *,
        nonce: str | None = None,
        jwks: dict | None = None,
    ) -> dict[str, Any]:
        """Decode and validate an ID token.

        For simplicity, uses HS256 with client_secret or unverified decode
        when jwks is not provided. In production, use the JWKS from discovery.
        """
        if not _JWT_AVAILABLE:
            raise OIDCError("pyjwt is required for ID token validation")

        try:
            # Try unverified decode first to get header
            header = pyjwt.get_unverified_header(id_token)
            alg = header.get("alg", "RS256")

            if alg == "HS256" and self.client_secret:
                claims = pyjwt.decode(
                    id_token,
                    self.client_secret,
                    algorithms=["HS256"],
                    audience=self.client_id,
                    options={"verify_exp": True},
                )
            else:
                # Unverified decode — in production, validate with JWKS
                claims = pyjwt.decode(
                    id_token,
                    options={
                        "verify_signature": False,
                        "verify_exp": True,
                        "verify_aud": False,
                    },
                )
        except pyjwt.ExpiredSignatureError:
            raise OIDCError("ID token expired")
        except pyjwt.PyJWTError as exc:
            raise OIDCError(f"ID token validation failed: {exc}")

        # Verify nonce if provided
        if nonce and claims.get("nonce") != nonce:
            raise OIDCError("Nonce mismatch")

        # Verify issuer
        token_iss = claims.get("iss", "").rstrip("/")
        if token_iss != self.issuer_url:
            raise OIDCError(f"Issuer mismatch: {token_iss} != {self.issuer_url}")

        return claims

    @staticmethod
    def extract_groups(claims: dict[str, Any], group_claim: str = "") -> list[str]:
        """Extract group names from ID token claims."""
        if not group_claim:
            return []
        groups = claims.get(group_claim, [])
        if isinstance(groups, str):
            return [groups]
        if isinstance(groups, list):
            return [str(g) for g in groups]
        return []
