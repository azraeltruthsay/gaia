"""A THOUGHT_SEED line anywhere in a streamed reply never reaches the person."""
import asyncio
import json

import pytest

from gaia_core.cognition.reply_scrub import SeedLineFilter, scrub_stream


def _run(chunks):
    f = SeedLineFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


HAIKU = "Autumn rain falls soft,\nNature's gentle sigh.\n\n*THOUGHT_SEED: Poem written in Japanese style.*"


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
def test_trailing_italic_seed_dropped(size):
    chunks = [HAIKU[i:i + size] for i in range(0, len(HAIKU), size)]
    assert _run(chunks) == "Autumn rain falls soft,\nNature's gentle sigh.\n\n"


def test_mid_reply_seed_line_dropped():
    assert _run(["One.\nTHOUGHT_SEED: x\nTwo."]) == "One.\nTwo."


@pytest.mark.parametrize("text", [
    "*[What I did]* answered\n\nHello", "* a bullet\n* another", "**Bold** start",
    "THOUGHT about seeds", "Thoughts: none", "*", "",
])
def test_ordinary_text_untouched(text):
    assert _run([text[i:i + 2] for i in range(0, len(text), 2)] or [""]) == text


def test_stream_drops_seed():
    async def lines():
        for piece in ["Hi there.\n", "*THOUGHT_", "SEED: note*", "\nBye."]:
            yield json.dumps({"type": "token", "value": piece}) + "\n"
        yield json.dumps({"type": "reply_done"}) + "\n"

    async def collect():
        return [json.loads(x) async for x in scrub_stream(lines())]

    events = asyncio.run(collect())
    text = "".join(e["value"] for e in events if e["type"] == "token")
    assert text == "Hi there.\nBye." and events[-1]["type"] == "reply_done"


def test_action_none_line_dropped():
    assert _run(["Haiku line.\n\n*Action: none (direct answer)*"]) == "Haiku line.\n\n"


def test_action_word_kept():
    assert _run(["Action: take the left road.\nAction items follow."]) == \
        "Action: take the left road.\nAction items follow."
