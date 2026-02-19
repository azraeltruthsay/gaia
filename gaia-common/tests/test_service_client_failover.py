"""Tests for ServiceClient HA failover behavior."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from gaia_common.utils.service_client import ServiceClient


def _make_client(fallback: str = "http://fallback:6415") -> ServiceClient:
    """Create a ServiceClient with fallback configured, fast retries."""
    return ServiceClient(
        "test-service",
        default_port=6415,
        max_retries=2,
        retry_base_delay=0.01,
        fallback_base_url=fallback,
    )


def _make_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response (sync .json(), not async)."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _mock_async_client(side_effect):
    """Build an AsyncMock that acts as an httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    for method in ("get", "post", "delete"):
        getattr(mock_client, method).side_effect = side_effect
    return mock_client


@pytest.mark.asyncio
async def test_primary_succeeds_no_fallback_called():
    """When primary succeeds, fallback is never attempted."""
    client = _make_client()
    mock_response = _make_response({"status": "ok"})

    mock_http = _mock_async_client(None)
    mock_http.get.side_effect = None
    mock_http.get.return_value = mock_response

    with patch("gaia_common.utils.service_client.httpx.AsyncClient", return_value=mock_http):
        result = await client.get("/health", retry=False)

    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_connect_error_triggers_fallback():
    """ConnectError after retries should trigger fallback."""
    client = _make_client()
    mock_response = _make_response({"status": "from-fallback"})

    async def _side_effect(url, **kwargs):
        if "fallback" in str(url):
            return mock_response
        raise httpx.ConnectError("Connection refused")

    mock_http = _mock_async_client(None)
    mock_http.get.side_effect = _side_effect

    with (
        patch("gaia_common.utils.service_client.httpx.AsyncClient", return_value=mock_http),
        patch("gaia_common.utils.service_client._MAINTENANCE_FLAG", Path("/nonexistent/path")),
    ):
        result = await client.get("/health")

    assert result == {"status": "from-fallback"}


@pytest.mark.asyncio
async def test_timeout_does_not_trigger_fallback():
    """TimeoutException should NOT trigger fallback — timeout means alive but slow."""
    client = _make_client()
    mock_http = _mock_async_client(httpx.ReadTimeout("Timed out"))

    with patch("gaia_common.utils.service_client.httpx.AsyncClient", return_value=mock_http):
        with pytest.raises(httpx.ReadTimeout):
            await client.get("/health", retry=False)


@pytest.mark.asyncio
async def test_maintenance_mode_suppresses_fallback():
    """When maintenance mode is ON, fallback is NOT attempted even on retryable errors."""
    client = _make_client()
    mock_http = _mock_async_client(httpx.ConnectError("Connection refused"))

    with (
        patch("gaia_common.utils.service_client.httpx.AsyncClient", return_value=mock_http),
        patch.object(Path, "exists", return_value=True),
    ):
        with pytest.raises(httpx.ConnectError):
            await client.get("/health", retry=False)


@pytest.mark.asyncio
async def test_no_fallback_configured():
    """Without fallback_base_url, retryable errors propagate normally."""
    client = ServiceClient(
        "test-service", default_port=6415,
        max_retries=1, retry_base_delay=0.01,
    )
    mock_http = _mock_async_client(httpx.ConnectError("Connection refused"))

    with patch("gaia_common.utils.service_client.httpx.AsyncClient", return_value=mock_http):
        with pytest.raises(httpx.ConnectError):
            await client.get("/health")


@pytest.mark.asyncio
async def test_fallback_also_fails_raises_primary_error():
    """If fallback also fails, the ORIGINAL primary error is raised."""
    client = _make_client()

    async def _always_fail(url, **kwargs):
        raise httpx.ConnectError(f"Connection refused: {url}")

    mock_http = _mock_async_client(None)
    mock_http.get.side_effect = _always_fail

    with (
        patch("gaia_common.utils.service_client.httpx.AsyncClient", return_value=mock_http),
        patch("gaia_common.utils.service_client._MAINTENANCE_FLAG", Path("/nonexistent/path")),
    ):
        with pytest.raises(httpx.ConnectError, match="test-service"):
            await client.get("/health")


@pytest.mark.asyncio
async def test_post_failover():
    """POST requests should also failover on retryable errors."""
    client = _make_client()
    mock_response = _make_response({"result": "from-fallback"})

    async def _side_effect(url, **kwargs):
        if "fallback" in str(url):
            return mock_response
        raise httpx.ConnectError("Connection refused")

    mock_http = _mock_async_client(None)
    mock_http.post.side_effect = _side_effect

    with (
        patch("gaia_common.utils.service_client.httpx.AsyncClient", return_value=mock_http),
        patch("gaia_common.utils.service_client._MAINTENANCE_FLAG", Path("/nonexistent/path")),
    ):
        result = await client.post("/process", data={"msg": "hello"})

    assert result == {"result": "from-fallback"}


@pytest.mark.asyncio
async def test_is_maintenance_mode_reads_file():
    """_is_maintenance_mode should check the file-based flag."""
    with patch("gaia_common.utils.service_client._MAINTENANCE_FLAG", Path("/nonexistent/path")):
        assert not ServiceClient._is_maintenance_mode()

    import tempfile
    with tempfile.NamedTemporaryFile() as f:
        with patch("gaia_common.utils.service_client._MAINTENANCE_FLAG", Path(f.name)):
            assert ServiceClient._is_maintenance_mode()


@pytest.mark.asyncio
async def test_get_core_client_with_fallback_env():
    """get_core_client should pick up CORE_FALLBACK_ENDPOINT from env."""
    from gaia_common.utils.service_client import get_core_client

    with patch.dict("os.environ", {"CORE_FALLBACK_ENDPOINT": "http://candidate:6415"}):
        client = get_core_client()
        assert client.fallback_base_url == "http://candidate:6415"


@pytest.mark.asyncio
async def test_get_core_client_no_fallback():
    """get_core_client without env var should have no fallback."""
    from gaia_common.utils.service_client import get_core_client

    with patch.dict("os.environ", {}, clear=True):
        client = get_core_client()
        assert client.fallback_base_url is None
