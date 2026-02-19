"""
Async retry helper for gaia-web HTTP calls.

Provides retry-with-backoff for outbound requests to gaia-core and other
GAIA services. Retries on transient network errors; does NOT retry on
client errors (4xx) or read timeouts (which indicate core is genuinely stuck).
"""

import asyncio
import logging
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


async def post_with_retry(
    url: str,
    *,
    json: dict,
    headers: dict | None = None,
    timeout: float = 300.0,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> httpx.Response:
    """POST to a URL with retry-on-transient-failure.

    Args:
        url: Target URL.
        json: JSON payload.
        headers: Optional HTTP headers.
        timeout: Per-request timeout in seconds.
        max_attempts: Maximum number of attempts (default 3).
        base_delay: Base delay in seconds; doubles each attempt (2s, 4s).

    Returns:
        The successful httpx.Response.

    Raises:
        httpx.TimeoutException: If the request times out (not retried — a
            300s timeout means core is genuinely stuck).
        httpx.HTTPStatusError: If a non-retryable HTTP error occurs.
        The last retryable exception if all attempts are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url, json=json, headers=headers or {"Content-Type": "application/json"},
                )

                # Retry on 502/503/504 — service restarting
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "POST %s returned %d on attempt %d/%d, retrying in %.1fs...",
                        url, response.status_code, attempt, max_attempts, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return response

        except httpx.TimeoutException:
            # Do NOT retry timeouts — 300s means core is genuinely stuck
            raise

        except RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "POST %s failed on attempt %d/%d (%s), retrying in %.1fs...",
                    url, attempt, max_attempts, type(exc).__name__, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise

    # Safety net — should not be reached
    if last_exc:
        raise last_exc
    raise RuntimeError(f"POST {url} failed after {max_attempts} attempts")
