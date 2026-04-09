"""GAIA Inter-Service Authentication — HMAC-based request validation.

Every inter-service HTTP request must include an X-GAIA-Auth header with
an HMAC signature. Requests without valid auth are rejected.

The shared secret is mounted via Docker secrets at /run/secrets/gaia_service_key.
All GAIA services share the same key.

Usage (client — making requests):
    from gaia_common.utils.service_auth import sign_request, auth_headers

    # Option 1: Get headers dict to pass to httpx/requests
    headers = auth_headers()
    httpx.get("http://gaia-core:6415/health", headers=headers)

    # Option 2: Sign an existing request
    sign_request(request)

Usage (server — validating requests):
    from gaia_common.utils.service_auth import validate_request, AuthMiddleware

    # FastAPI middleware (recommended)
    app.add_middleware(AuthMiddleware)

    # Manual validation
    if not validate_request(request):
        raise HTTPException(401, "Invalid service auth")
"""

import hashlib
import hmac
import logging
import os
import time
from functools import lru_cache
from typing import Dict, Optional

logger = logging.getLogger("GAIA.ServiceAuth")

# Where the shared secret lives
_SECRET_PATHS = [
    "/run/secrets/gaia_service_key",
    os.environ.get("GAIA_SERVICE_KEY_PATH", ""),
    "/shared/secrets/gaia_service_key",
]

# Header name
AUTH_HEADER = "X-GAIA-Auth"

# Timestamp tolerance (seconds) — reject requests with timestamps too far off
_TIMESTAMP_TOLERANCE = 30

# Paths that don't require auth (health checks, metrics)
_PUBLIC_PATHS = {
    "/health", "/status", "/metrics", "/openapi.json", "/docs",
    "/favicon.ico", "/static",
    # Monitoring endpoints used by orchestrator/doctor
    "/model/status", "/gpu/status", "/queue/status",
    "/immune", "/config",
    # API endpoints — TODO: wire auth headers into clients, then remove these
    # For now, auth is enforced on the middleware level for unknown paths,
    # and these known GAIA paths are allowed to maintain functionality while
    # we wire proper HMAC headers into each client.
    "/process_packet", "/api/", "/session",
    "/sleep", "/wake", "/audio",
    "/jsonrpc",
    "/model/", "/cache/", "/adapter/",
    "/consciousness",
    "/engine",
    "/refresh_pool",
}


@lru_cache(maxsize=1)
def _load_secret() -> Optional[bytes]:
    """Load the shared service key from Docker secrets or env."""
    for path in _SECRET_PATHS:
        if path and os.path.exists(path):
            try:
                secret = open(path, "rb").read().strip()
                if secret:
                    logger.info("Service auth key loaded from %s", path)
                    return secret
            except Exception:
                continue

    # Fallback: env var (less secure, but works for development)
    env_key = os.environ.get("GAIA_SERVICE_KEY")
    if env_key:
        logger.warning("Service auth key from env var (less secure than Docker secret)")
        return env_key.encode()

    # No key found — auth is disabled
    logger.warning("No service auth key found — inter-service auth DISABLED")
    return None


def _compute_hmac(secret: bytes, timestamp: str, method: str = "", path: str = "") -> str:
    """Compute HMAC signature for a request."""
    message = f"{timestamp}:{method}:{path}".encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def auth_headers(method: str = "GET", path: str = "") -> Dict[str, str]:
    """Generate auth headers for an outbound inter-service request.

    Returns dict with X-GAIA-Auth header, or empty dict if auth is disabled.
    """
    secret = _load_secret()
    if not secret:
        return {}

    timestamp = str(int(time.time()))
    signature = _compute_hmac(secret, timestamp, method, path)

    return {
        AUTH_HEADER: f"{timestamp}:{signature}",
    }


def sign_request(headers: dict, method: str = "GET", path: str = "") -> dict:
    """Add auth headers to an existing headers dict. Returns the dict."""
    headers.update(auth_headers(method, path))
    return headers


def validate_auth_header(header_value: str, method: str = "", path: str = "") -> bool:
    """Validate an X-GAIA-Auth header value.

    Header format: "timestamp:hmac_hex"
    """
    secret = _load_secret()
    if not secret:
        # No key configured — allow all (dev mode)
        return True

    if not header_value:
        return False

    parts = header_value.split(":", 1)
    if len(parts) != 2:
        return False

    timestamp_str, signature = parts

    # Check timestamp is recent
    try:
        timestamp = int(timestamp_str)
        now = int(time.time())
        if abs(now - timestamp) > _TIMESTAMP_TOLERANCE:
            logger.warning("Service auth: timestamp too old (%ds drift)", abs(now - timestamp))
            return False
    except ValueError:
        return False

    # Verify HMAC
    expected = _compute_hmac(secret, timestamp_str, method, path)
    return hmac.compare_digest(signature, expected)


def validate_request(request) -> bool:
    """Validate a FastAPI/Starlette request object."""
    # Skip auth for public paths
    path = getattr(request, "url", None)
    if path:
        path_str = str(path.path) if hasattr(path, "path") else str(path)
        for public in _PUBLIC_PATHS:
            if path_str.startswith(public):
                return True

    header = None
    if hasattr(request, "headers"):
        header = request.headers.get(AUTH_HEADER.lower()) or request.headers.get(AUTH_HEADER)

    method = getattr(request, "method", "GET")
    path_str = ""
    if path:
        path_str = str(path.path) if hasattr(path, "path") else str(path)

    return validate_auth_header(header or "", method, path_str)


# ── FastAPI/Starlette Middleware ──────────────────────────────────────

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class AuthMiddleware(BaseHTTPMiddleware):
        """Middleware that validates X-GAIA-Auth on all non-public requests."""

        async def dispatch(self, request, call_next):
            # Skip if no key configured (dev mode)
            if _load_secret() is None:
                return await call_next(request)

            # Skip public paths
            path = str(request.url.path)
            if any(path.startswith(p) for p in _PUBLIC_PATHS):
                return await call_next(request)

            # Validate
            header = request.headers.get(AUTH_HEADER.lower()) or request.headers.get(AUTH_HEADER)
            if not validate_auth_header(header or "", request.method, path):
                logger.warning("Service auth REJECTED: %s %s (from %s)",
                               request.method, path, request.client.host if request.client else "?")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing service authentication"},
                )

            return await call_next(request)

except ImportError:
    # Starlette not available — middleware won't be usable but functions still work
    AuthMiddleware = None


# ── Key Generation Utility ────────────────────────────────────────────

def generate_key(length: int = 32) -> str:
    """Generate a random service key. Call once at initial setup."""
    import secrets
    return secrets.token_hex(length)
