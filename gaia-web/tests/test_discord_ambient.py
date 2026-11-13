"""GAIA's Grotto (2026-11-12): an ambient-channel turn (addressed_to_gaia=False)
must stay silent on its own -- no presence change, no error reply, no
wake/queue -- unless she actually decides to speak. Follows the mocking
convention in test_discord_wait_for_core.py (bare DiscordInterface, fake
httpx client)."""
import asyncio

import httpx
import pytest

from gaia_web import discord_interface as di


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _sleep(_):
        return None
    monkeypatch.setattr(di.asyncio, "sleep", _sleep)


class FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class FakeClient:
    """Fakes both the distracted-check GET (made via a raw
    httpx.AsyncClient(...) context manager, same as the real code) and, if
    reached, the process_packet stream (via get_core_client()) -- tests
    only need to go as far as they assert."""
    def __init__(self, distracted_data=None, get_raises=None, stream_raises=None):
        self.distracted_data = distracted_data or {"state": "active"}
        self.get_raises = get_raises
        self.stream_raises = stream_raises
        self.get_calls = []
        self.stream_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, timeout=None):
        self.get_calls.append(url)
        if self.get_raises:
            raise self.get_raises
        if "distracted-check" in url:
            return FakeResponse(200, self.distracted_data)
        return FakeResponse(200, {})  # /health

    def stream(self, method, url, json=None, timeout=None):
        self.stream_calls.append(url)
        if self.stream_raises:
            raise self.stream_raises
        raise AssertionError("test did not expect the stream to be reached")

    async def aclose(self):
        pass


class FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeChannel:
    def __init__(self, channel_id="123"):
        self.id = channel_id

    def typing(self):
        return FakeTyping()


class FakeMessage:
    def __init__(self):
        self.channel = FakeChannel()


def _iface(monkeypatch, client, ambient_channel_ids=None):
    iface = di.DiscordInterface.__new__(di.DiscordInterface)
    iface.core_endpoint = "http://gaia-core:6415"
    iface.message_queue = None
    iface.ambient_channel_ids = ambient_channel_ids or set()
    presence_calls = []

    async def fake_set_presence(status, activity):
        presence_calls.append((status, activity))
    iface._set_presence = fake_set_presence

    sent = []
    async def fake_send_response(message_obj, content, is_dm):
        sent.append(content)
    iface._send_response = fake_send_response

    async def fake_get_core_client():
        return client
    monkeypatch.setattr(di, "get_core_client", fake_get_core_client)
    # The distracted-check (both ambient and addressed paths) opens its own
    # raw httpx.AsyncClient(...) context manager, not get_core_client().
    monkeypatch.setattr(di.httpx, "AsyncClient", lambda *a, **kw: client)

    return iface, presence_calls, sent


def _run(coro):
    return asyncio.run(coro)


def test_ambient_turn_while_asleep_is_never_forwarded(monkeypatch):
    client = FakeClient(distracted_data={"state": "asleep"})
    iface, presence_calls, sent = _iface(monkeypatch, client)
    _run(iface._handle_message(
        content="anyone around?", channel_id="123", user_id="1", guild_id="g",
        author_name="Nick", message_id="m1", is_dm=False, message_obj=FakeMessage(),
        addressed_to_gaia=False,
    ))
    assert not client.stream_calls
    assert not presence_calls
    assert not sent


def test_ambient_turn_while_drowsy_is_never_forwarded(monkeypatch):
    client = FakeClient(distracted_data={"state": "drowsy"})
    iface, presence_calls, sent = _iface(monkeypatch, client)
    _run(iface._handle_message(
        content="hey", channel_id="123", user_id="1", guild_id="g",
        author_name="Nick", message_id="m1", is_dm=False, message_obj=FakeMessage(),
        addressed_to_gaia=False,
    ))
    assert not client.stream_calls and not sent


def test_ambient_turn_while_active_skips_presence_change(monkeypatch):
    client = FakeClient(distracted_data={"state": "active"}, stream_raises=httpx.TimeoutException("x"))
    iface, presence_calls, sent = _iface(monkeypatch, client)
    _run(iface._handle_message(
        content="hey everyone", channel_id="123", user_id="1", guild_id="g",
        author_name="Nick", message_id="m1", is_dm=False, message_obj=FakeMessage(),
        addressed_to_gaia=False,
    ))
    assert ("online", "Thinking...") not in presence_calls
    assert client.stream_calls  # it did reach gaia-core, just errored


def test_ambient_turn_error_never_sends_a_visible_reply(monkeypatch):
    for exc in (httpx.TimeoutException("x"), RuntimeError("boom")):
        client = FakeClient(distracted_data={"state": "active"}, stream_raises=exc)
        iface, presence_calls, sent = _iface(monkeypatch, client)
        _run(iface._handle_message(
            content="hey everyone", channel_id="123", user_id="1", guild_id="g",
            author_name="Nick", message_id="m1", is_dm=False, message_obj=FakeMessage(),
            addressed_to_gaia=False,
        ))
        assert not sent, f"ambient turn sent a reply on {exc!r}"


def test_ambient_turn_unreachable_core_sends_nothing(monkeypatch):
    client = FakeClient(get_raises=httpx.ConnectError("down"))
    iface, presence_calls, sent = _iface(monkeypatch, client)
    _run(iface._handle_message(
        content="hey everyone", channel_id="123", user_id="1", guild_id="g",
        author_name="Nick", message_id="m1", is_dm=False, message_obj=FakeMessage(),
        addressed_to_gaia=False,
    ))
    assert not sent


def test_addressed_turn_error_still_sends_the_usual_reply(monkeypatch):
    # Regression: the normal (non-ambient) path is unchanged.
    client = FakeClient(distracted_data={"state": "active"}, stream_raises=httpx.TimeoutException("x"))
    iface, presence_calls, sent = _iface(monkeypatch, client)
    _run(iface._handle_message(
        content="GAIA, you there?", channel_id="123", user_id="1", guild_id="g",
        author_name="Nick", message_id="m1", is_dm=False, message_obj=FakeMessage(),
        addressed_to_gaia=True,
    ))
    assert presence_calls and presence_calls[0] == ("online", "Thinking...")
    assert sent and "too long" in sent[0]
