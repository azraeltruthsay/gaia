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


def test_edits_in_a_dm_by_user_id(monkeypatch):
    """aidw.10: a DM session names the person, not the channel."""
    mine = Msg(1, "Twelve rooms, including a kennel.")
    dm = Channel([Msg(7, "rooms?"), mine])
    user = SimpleNamespace(dm_channel=None)

    async def create_dm():
        return dm
    user.create_dm = create_dm
    bot = SimpleNamespace(is_ready=lambda: True, user=SimpleNamespace(id=1),
                          get_user=lambda uid: user if uid == 7 else None,
                          get_channel=lambda cid: (_ for _ in ()).throw(AssertionError("not a channel")))
    monkeypatch.setattr(di, "_bot", bot)

    async def on_loop(coro, timeout=30.0):
        return await coro
    monkeypatch.setattr(di, "_await_on_bot_loop", on_loop)
    old = asyncio.run(di.edit_own_message("", "Twelve rooms", "Three rooms.", user_id="7"))
    assert old.startswith("Twelve rooms") and mine.edited == "Three rooms."


def test_web_edit_goes_out_as_an_event(tmp_path, monkeypatch):
    import json

    from gaia_web.routes import autonomous as au
    log = tmp_path / "auto.jsonl"
    monkeypatch.setattr(au, "_LOG_PATH", str(log))
    assert au.append_edit_event("web_ab12cd34", "Twelve rooms", "Three rooms.")
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["type"] == "edit" and rec["session_id"] == "web_ab12cd34"
    assert rec["new_text"] == "Three rooms." and "text" not in rec   # old pages don't show it as new
