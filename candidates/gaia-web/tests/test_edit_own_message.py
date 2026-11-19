"""She edits only her own message, the latest one containing the text."""
import asyncio
from types import SimpleNamespace

from gaia_web import discord_interface as di


class Msg:
    def __init__(self, author_id, content):
        self.author = SimpleNamespace(id=author_id)
        self.content, self.edited = content, None

    async def edit(self, content):
        self.edited = content


class Channel:
    def __init__(self, msgs):
        self.msgs = msgs

    def history(self, limit=40):
        async def gen():
            for m in self.msgs[:limit]:
                yield m
        return gen()


def test_edits_her_latest_matching_message_only(monkeypatch):
    mine_new, theirs, mine_old = Msg(1, "Morning Andy. Yes"), Msg(2, "Morning Andy. Yes"), Msg(1, "Morning Andy. Older")
    ch = Channel([theirs, mine_new, mine_old])          # newest first, as Discord returns
    bot = SimpleNamespace(is_ready=lambda: True, user=SimpleNamespace(id=1),
                          get_channel=lambda cid: ch)
    monkeypatch.setattr(di, "_bot", bot)

    async def on_loop(coro, timeout=30.0):
        return await coro
    monkeypatch.setattr(di, "_await_on_bot_loop", on_loop)
    old = asyncio.run(di.edit_own_message("5", "Morning Andy", "Morning Azrael. Yes"))
    assert old == "Morning Andy. Yes" and mine_new.edited == "Morning Azrael. Yes"
    assert theirs.edited is None and mine_old.edited is None
    assert asyncio.run(di.edit_own_message("5", "", "x")) is None
