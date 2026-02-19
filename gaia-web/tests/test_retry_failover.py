"""Tests for post_with_retry HA failover behavior."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from gaia_web.utils.retry import post_with_retry


@pytest.mark.asyncio
async def test_primary_succeeds_no_fallback():
    """When primary succeeds, fallback is never attempted."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {"ok": True}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("gaia_web.utils.retry.httpx.AsyncClient", return_value=mock_client):
        result = await post_with_retry(
            "http://primary:6415/process_packet",
            json={"msg": "hello"},
            fallback_url="http://fallback:6415/process_packet",
        )

    assert result.status_code == 200
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_connect_error_triggers_fallback():
    """ConnectError after retries should trigger fallback."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None

    async def _side_effect(url, **kwargs):
        if "fallback" in url:
            return mock_response
        raise httpx.ConnectError("Connection refused")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.side_effect = _side_effect

    with (
        patch("gaia_web.utils.retry.httpx.AsyncClient", return_value=mock_client),
        patch("gaia_web.utils.retry._MAINTENANCE_FLAG", Path("/nonexistent/path")),
    ):
        result = await post_with_retry(
            "http://primary:6415/process_packet",
            json={"msg": "hello"},
            max_attempts=2,
            base_delay=0.01,
            fallback_url="http://fallback:6415/process_packet",
        )

    assert result.status_code == 200


@pytest.mark.asyncio
async def test_timeout_does_not_trigger_fallback():
    """TimeoutException should NOT trigger fallback."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.side_effect = httpx.ReadTimeout("Timed out")

    with patch("gaia_web.utils.retry.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.ReadTimeout):
            await post_with_retry(
                "http://primary:6415/process_packet",
                json={"msg": "hello"},
                fallback_url="http://fallback:6415/process_packet",
                max_attempts=1,
            )

    # Should only call once (no retry on timeout, no fallback)
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_maintenance_mode_suppresses_fallback():
    """Maintenance mode ON should suppress fallback."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    with (
        patch("gaia_web.utils.retry.httpx.AsyncClient", return_value=mock_client),
        patch.object(Path, "exists", return_value=True),
    ):
        with pytest.raises(httpx.ConnectError):
            await post_with_retry(
                "http://primary:6415/process_packet",
                json={"msg": "hello"},
                max_attempts=1,
                base_delay=0.01,
                fallback_url="http://fallback:6415/process_packet",
            )


@pytest.mark.asyncio
async def test_no_fallback_url_raises_normally():
    """Without fallback_url, retryable errors propagate normally."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    with patch("gaia_web.utils.retry.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.ConnectError):
            await post_with_retry(
                "http://primary:6415/process_packet",
                json={"msg": "hello"},
                max_attempts=1,
                base_delay=0.01,
            )


@pytest.mark.asyncio
async def test_fallback_also_fails_raises_primary():
    """If fallback also fails, the original error should be raised."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    with (
        patch("gaia_web.utils.retry.httpx.AsyncClient", return_value=mock_client),
        patch("gaia_web.utils.retry._MAINTENANCE_FLAG", Path("/nonexistent/path")),
    ):
        with pytest.raises(httpx.ConnectError):
            await post_with_retry(
                "http://primary:6415/process_packet",
                json={"msg": "hello"},
                max_attempts=1,
                base_delay=0.01,
                fallback_url="http://fallback:6415/process_packet",
            )
