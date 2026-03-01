"""Tests for gaia_common.utils.resilience retry utilities."""

import asyncio
import pytest
from unittest.mock import MagicMock

from gaia_common.utils.resilience import async_retry, sync_retry


# ── async_retry tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_retry_succeeds_first_attempt():
    call_count = 0

    async def ok():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await async_retry(ok, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_async_retry_succeeds_after_transient_failure():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "recovered"

    result = await async_retry(
        flaky,
        max_attempts=3,
        base_delay=0.01,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_async_retry_exhausts_attempts():
    call_count = 0

    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("permanent")

    with pytest.raises(ConnectionError, match="permanent"):
        await async_retry(
            always_fail,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
    assert call_count == 3


@pytest.mark.asyncio
async def test_async_retry_does_not_retry_non_retryable():
    call_count = 0

    async def wrong_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        await async_retry(
            wrong_error,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
    assert call_count == 1


@pytest.mark.asyncio
async def test_async_retry_calls_on_retry_callback():
    attempts_seen = []

    async def flaky():
        if len(attempts_seen) < 2:
            raise ConnectionError("transient")
        return "ok"

    def on_retry(attempt, max_attempts, exc):
        attempts_seen.append(attempt)

    await async_retry(
        flaky,
        max_attempts=3,
        base_delay=0.01,
        retryable_exceptions=(ConnectionError,),
        on_retry=on_retry,
    )
    assert attempts_seen == [1, 2]


# ── sync_retry tests ─────────────────────────────────────────────────────────


def test_sync_retry_succeeds_first_attempt():
    call_count = 0

    def ok():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = sync_retry(ok, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert call_count == 1


def test_sync_retry_succeeds_after_transient_failure():
    call_count = 0

    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("transient")
        return "recovered"

    result = sync_retry(
        flaky,
        max_attempts=3,
        base_delay=0.01,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "recovered"
    assert call_count == 2


def test_sync_retry_exhausts_attempts():
    call_count = 0

    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("permanent")

    with pytest.raises(ConnectionError, match="permanent"):
        sync_retry(
            always_fail,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
    assert call_count == 3


def test_sync_retry_does_not_retry_non_retryable():
    call_count = 0

    def wrong_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        sync_retry(
            wrong_error,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
    assert call_count == 1
