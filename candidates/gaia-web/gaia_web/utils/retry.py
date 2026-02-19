"""
Async retry helper for gaia-web HTTP calls.

Provides retry-with-backoff for outbound requests to gaia-core and other
GAIA services. Retries on transient network errors; does NOT retry on
client errors (4xx) or read timeouts (which indicate core is genuinely stuck).

Supports optional HA failover: after all primary retries are exhausted on
retryable errors, a single attempt is made to a fallback URL (unless
maintenance mode is active).
"""

import asyncio
import logging
from pathlib import Path
from typing import TypeVar

import httpx

logger = logging.getLogger("GAIA.Web.Retry")

T = TypeVar("T")

# Exceptions that indicate a transient failure worth retrying
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)

# HTTP status codes that warrant a retry (service temporarily unavailable)
RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})

# File-based maintenance mode flag (shared Docker volume)
_MAINTENANCE_FLAG = Path("/shared/ha_maintenance")


def _is_maintenance_mode() -> bool:
    """Check if HA maintenance mode is active."""
    return _MAINTENANCE_FLAG.exists()


async def post_with_retry(
    url: str,
    *,
    json: dict,
    headers: dict | None = None,
    timeout: float = 300.0,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    fallback_url: str | None = None,
) -> httpx.Response:
    """POST to a URL with retry-on-transient-failure and optional HA failover.

    Args:
        url: Target URL.
        json: JSON payload.
        headers: Optional HTTP headers.
        timeout: Per-request timeout in seconds.
        max_attempts: Maximum number of attempts (default 3).
        base_delay: Base delay in seconds; doubles each attempt (2s, 4s).
        fallback_url: Optional HA fallback URL. After all primary retries
            are exhausted on retryable errors, a single POST is attempted
            here (unless maintenance mode is active). NOT used for timeouts.

    Returns:
        The successful httpx.Response.

    Raises:
        httpx.TimeoutException: If the request times out (not retried — a
            300s timeout means core is genuinely stuck).
        httpx.HTTPStatusError: If a non-retryable HTTP error occurs.
        The last retryable exception if all attempts are exhausted.
    """
    last_exc: Exception | None = None
    retryable_failure = False  # Track whether failure was retryable (for failover)

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url, json=json, headers=headers or {"Content-Type": "application/json"},
                )

                # Retry on 502/503/504 — service restarting
                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "POST %s returned %d on attempt %d/%d, retrying in %.1fs...",
                            url, response.status_code, attempt, max_attempts, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Last attempt with retryable status — mark for failover
                    retryable_failure = True
                    response.raise_for_status()

                response.raise_for_status()
                return response

        except httpx.TimeoutException:
            # Do NOT retry timeouts — 300s means core is genuinely stuck
            # Do NOT failover on timeouts — slow is not down
            raise

        except RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            retryable_failure = True
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "POST %s failed on attempt %d/%d (%s), retrying in %.1fs...",
                    url, attempt, max_attempts, type(exc).__name__, delay,
                )
                await asyncio.sleep(delay)
                continue
            # Fall through to failover check below

    # All primary attempts exhausted with retryable errors — try failover
    if retryable_failure and fallback_url and not _is_maintenance_mode():
        logger.warning(
            "Primary POST %s exhausted %d attempts, attempting HA fallback to %s",
            url, max_attempts, fallback_url,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    fallback_url, json=json,
                    headers=headers or {"Content-Type": "application/json"},
                )
                response.raise_for_status()
                logger.info("HA fallback POST %s succeeded", fallback_url)
                return response
        except Exception as fallback_exc:
            logger.error(
                "HA fallback POST %s also failed (%s). Raising original error.",
                fallback_url, type(fallback_exc).__name__,
            )
            # Fall through to raise the original error

    if last_exc:
        raise last_exc
    raise RuntimeError(f"POST {url} failed after {max_attempts} attempts")
