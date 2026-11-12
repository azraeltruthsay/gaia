"""w1hf: a Discord message that arrived while gaia-core restarted was lost."""
import asyncio

import httpx
import pytest

from gaia_web import discord_interface as di


class Client:
    def __init__(self, fails):
        self.fails, self.calls = fails, 0

    async def get(self, url, timeout=None):
        self.calls += 1
        if self.calls <= self.fails:
            raise httpx.ConnectError("All connection attempts failed")
        return type("R", (), {"status_code": 200})()


def _iface():
    iface = di.DiscordInterface.__new__(di.DiscordInterface)
    iface.core_endpoint = "http://gaia-core:6415"
    return iface


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _sleep(_):
        return None
    monkeypatch.setattr(di.asyncio, "sleep", _sleep)


def test_up_at_once():
    c = Client(0)
    assert asyncio.run(_iface()._wait_for_core(c)) is True and c.calls == 1


def test_back_after_a_restart():
    c = Client(4)                       # down for 1+2+4+8 s
    assert asyncio.run(_iface()._wait_for_core(c)) is True and c.calls == 5


def test_gives_up_after_the_deadline():
    c = Client(10 ** 6)
    assert asyncio.run(_iface()._wait_for_core(c, deadline_s=30)) is False
