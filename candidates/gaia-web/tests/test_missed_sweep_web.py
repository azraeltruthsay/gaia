"""Finding the Discord message her history holds, only after her own last one."""
import asyncio
from types import SimpleNamespace as NS

from gaia_web.discord_interface import find_missed_message
from gaia_web.link_notes import GIF_NOTE

ME = NS(id=1)


class _Chan:
    def __init__(self, msgs):
        self.msgs = msgs   # newest first

    def history(self, limit=40):
        async def gen():
            for m in self.msgs[:limit]:
                yield m
        return gen()


def _m(author, content):
    return NS(author=NS(id=author), content=content)


def test_found_after_her_last():
    link = _m(2, "Try this www.LinkedIn.com/in/andrewcprince")
    ch = _Chan([_m(3, "lol"), link, _m(1, "Andy, the link?")])
    assert asyncio.run(find_missed_message(ch, ME, "Try this www.LinkedIn.com/in/andrewcprince")) is link


def test_gif_note_and_mentions_ignored():
    gif = _m(2, "<@1> https://klipy.com/gifs/x")
    ch = _Chan([gif])
    assert asyncio.run(find_missed_message(ch, ME, f"https://klipy.com/gifs/x\n{GIF_NOTE}")) is gif


def test_not_past_her_own_message():
    ch = _Chan([_m(1, "Got it."), _m(2, "Try this")])
    assert asyncio.run(find_missed_message(ch, ME, "Try this")) is None
