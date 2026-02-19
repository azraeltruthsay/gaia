"""
Resilience utilities for GAIA inter-service communication.

Provides retry-with-backoff for both async (httpx) and sync (requests)
HTTP clients. Used by ServiceClient, gaia-web retry helpers, and
VLLMRemoteModel to handle transient network failures gracefully.
"""

import asyncio
import logging
import time
from typing import (
    Callable,
    Collection,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

logger = logging.getLogger("GAIA.Resilience")

T = TypeVar("T")


async def async_retry(
    fn: Callable[..., T],
    *args,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    retryable_exceptions: Sequence[Type[BaseException]] = (),
    retryable_status_codes: Collection[int] = frozenset({502, 503, 504}),
    on_retry: Optional[Callable[[int, int, Exception], None]] = None,
    **kwargs,
) -> T:
    """Retry an async callable with exponential backoff.

    Args:
        fn: Async callable to invoke.
        *args: Positional arguments forwarded to fn.
        max_attempts: Maximum number of attempts.
        base_delay: Base delay in seconds (doubles each retry).
        retryable_exceptions: Exception types that trigger a retry.
        retryable_status_codes: HTTP status codes to retry on (checked
            if the exception has a ``response`` attribute).
        on_retry: Optional callback(attempt, max_attempts, exc) invoked
            before each retry sleep.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        The return value of fn on success.

    Raises:
        The last exception if all attempts are exhausted.
    """
    last_exc: Optional[Exception] = None
    retryable = tuple(retryable_exceptions)

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except retryable as exc:
            last_exc = exc
            # Check if it's an HTTP error with a non-retryable status
            resp = getattr(exc, "response", None)
            if resp is not None and hasattr(resp, "status_code"):
                if resp.status_code not in retryable_status_codes:
                    raise

            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                if on_retry:
                    on_retry(attempt, max_attempts, exc)
                else:
                    logger.warning(
                        "Attempt %d/%d failed (%s), retrying in %.1fs...",
                        attempt, max_attempts, type(exc).__name__, delay,
                    )
                await asyncio.sleep(delay)
            else:
                raise

    # Safety net
    if last_exc:
        raise last_exc
    raise RuntimeError(f"async_retry: all {max_attempts} attempts failed")


def sync_retry(
    fn: Callable[..., T],
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.5,
    retryable_exceptions: Sequence[Type[BaseException]] = (),
    retryable_status_codes: Collection[int] = frozenset({503}),
    on_retry: Optional[Callable[[int, int, Exception], None]] = None,
    **kwargs,
) -> T:
    """Retry a synchronous callable with exponential backoff.

    Same semantics as async_retry but uses time.sleep instead of
    asyncio.sleep. Intended for requests-based clients (VLLMRemoteModel).

    Args:
        fn: Callable to invoke.
        *args: Positional arguments forwarded to fn.
        max_attempts: Maximum number of attempts.
        base_delay: Base delay in seconds (multiplied by attempt number).
        retryable_exceptions: Exception types that trigger a retry.
        retryable_status_codes: HTTP status codes to retry on.
        on_retry: Optional callback(attempt, max_attempts, exc).
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        The return value of fn on success.

    Raises:
        The last exception if all attempts are exhausted.
    """
    last_exc: Optional[Exception] = None
    retryable = tuple(retryable_exceptions)

    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except retryable as exc:
            last_exc = exc
            resp = getattr(exc, "response", None)
            if resp is not None and hasattr(resp, "status_code"):
                if resp.status_code not in retryable_status_codes:
                    raise

            if attempt < max_attempts:
                delay = base_delay * attempt
                if on_retry:
                    on_retry(attempt, max_attempts, exc)
                else:
                    logger.warning(
                        "Attempt %d/%d failed (%s), retrying in %.1fs...",
                        attempt, max_attempts, type(exc).__name__, delay,
                    )
                time.sleep(delay)
            else:
                raise

    if last_exc:
        raise last_exc
    raise RuntimeError(f"sync_retry: all {max_attempts} attempts failed")
