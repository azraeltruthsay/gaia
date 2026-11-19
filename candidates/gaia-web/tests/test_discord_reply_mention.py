"""In a channel her answer replies to the message it answers and pings its
author, once per message; DMs and later parts are sent plainly."""
import asyncio

from gaia_web.discord_interface import DiscordInterface


class _Chan:
    def __init__(self, log):
        self.log = log

    async def send(self, msg):
        self.log.append(("send", msg))


class _Msg:
    def __init__(self, mid=1):
        self.id, self.log = mid, []
        self.channel = _Chan(self.log)

    async def reply(self, msg, mention_author=False):
        self.log.append(("reply", msg, mention_author))


def _iface():
    return DiscordInterface.__new__(DiscordInterface)


def test_channel_first_part_replies_with_ping_then_plain():
    d, m = _iface(), _Msg()
    asyncio.run(d._send_response(m, "The poem.", is_dm=False))
    asyncio.run(d._send_response(m, "My line.", is_dm=False))
    assert m.log == [("reply", "The poem.", True), ("send", "My line.")]


def test_dm_never_pings():
    d, m = _iface(), _Msg()
    asyncio.run(d._send_response(m, "Hi.", is_dm=True))
    assert m.log == [("send", "Hi.")]


def test_failed_reply_still_sends():
    d, m = _iface(), _Msg()

    async def boom(msg, mention_author=False):
        raise RuntimeError("deleted")
    m.reply = boom
    asyncio.run(d._send_response(m, "Hi.", is_dm=False))
    assert m.log == [("send", "Hi.")]
