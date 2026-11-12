"""2026-11-12: asked to tell Nick something in the Grotto, she had no way to
post and stored a palace note instead. discord_post posts to her allowlisted
channels only, and keeps the post in that channel's history."""
import pytest

from gaia_mcp import discord_tools as dt


class Resp:
    def __init__(self, code=200, body=None):
        self.status_code, self._body, self.text = code, body or {}, str(body)

    def json(self):
        return self._body


class Client:
    calls = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        Client.calls.append((url, json))
        return Resp(200, {"delivered": True} if "/push" in url else {"ok": True})


@pytest.fixture
def http(monkeypatch):
    Client.calls = []
    monkeypatch.setattr(dt.httpx, "AsyncClient", Client)
    monkeypatch.delenv("GAIA_DISCORD_CHANNELS", raising=False)
    return Client


@pytest.mark.asyncio
async def test_posts_to_the_grotto_and_keeps_it(http):
    out = await dt.discord_post({"message": "I hear you're a silly fella named Nick."})
    assert out["ok"] and out["channel_id"] == "1484003096337191067"
    (push, pj), (hist, hj) = http.calls
    assert push.endswith("/api/conversations/discord_channel_1484003096337191067/push")
    assert pj["text"] == "I hear you're a silly fella named Nick."
    assert hist.endswith("/api/sessions/discord_channel_1484003096337191067/posted")
    assert hj["text"] == pj["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["The Grotto", "the-grotto", "#grotto", "GAIA's Grotto",
                                  "grotto channel", "1484003096337191067"])
async def test_the_grotto_by_name(http, name):
    assert (await dt.discord_post({"message": "hi", "channel": name}))["ok"]


@pytest.mark.asyncio
async def test_only_allowlisted_channels(http):
    with pytest.raises(ValueError):
        await dt.discord_post({"message": "hi", "channel": "general"})
    with pytest.raises(ValueError):
        await dt.discord_post({"message": " "})
    assert not http.calls


def test_not_approval_gated_but_listed():
    from gaia_mcp.tools import SENSITIVE_TOOLS
    assert "discord_post" not in SENSITIVE_TOOLS
    assert dt.channels() == {"grotto": "1484003096337191067"}
