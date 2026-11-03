"""Junk around a reply goes; the reply and its labels stay (8ail, vjrv)."""
import asyncio
import json

import pytest

from gaia_core.cognition.reply_scrub import HeadScrubber, answered_already, scrub_reply, scrub_stream

CUE = 'They asked: "Tell me about it." You already did that, and it returned the above. Your reply to them:'

CASES = [
    ("[Core]\n\n<nooutput>\nThe Raven is a poem.", "[Core]\n\nThe Raven is a poem."),
    ('[Core]\n\n<nooutput>{"content": "The Raven is a poem.</nooutput>', "[Core]\n\nThe Raven is a poem."),
    ('{"content": "The Raven is a poem."}', "The Raven is a poem."),
    ("[Core]\n\n<blockquote>The Raven is a poem.", "[Core]\n\nThe Raven is a poem."),
    ("[Core]\n\n[Clock: 08:36 AM PDT (Local), Tuesday | 15:36 UTC]\nThe Raven is a poem.", "[Core]\n\nThe Raven is a poem."),
    (f"[Core]\n\n{CUE} It is a poem.", "[Core]\n\nIt is a poem."),
    (f"*[Searched the web for: x]*\n\n[Core]\n\n{CUE} It is a poem.", "*[Searched the web for: x]*\n\n[Core]\n\nIt is a poem."),
    ('The user asked: "x"\n\nYour reply to them, from what it returned: Paris.', "Paris."),
]

KEPT = [
    "Paris.",
    "[nano]\n\nParis.",
    "[1] is the first citation.",
    "They asked me to wait, and I did.",
    "You already did that yesterday, I think.",
    '{"contents": 1} is the JSON you wanted.',
    "A tag like <b>this</b> stays, and so does 2 < 3.",
    "[Clock: 08:36 AM]",
]


def _stream(text, size):
    s = HeadScrubber()
    pieces = [s.feed(text[i:i + size]) for i in range(0, len(text), size)]
    return "".join(pieces) + s.flush()


@pytest.mark.parametrize("raw,want", CASES)
def test_scrub_reply_removes_junk(raw, want):
    assert scrub_reply(raw) == want


@pytest.mark.parametrize("raw", KEPT)
def test_scrub_reply_keeps_replies(raw):
    assert scrub_reply(raw) == raw


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
@pytest.mark.parametrize("raw,want", CASES)
def test_stream_matches_finished_reply_however_it_is_split(raw, want, size):
    assert _stream(raw, size).strip() == want.strip()


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
@pytest.mark.parametrize("raw", KEPT[:-1])
def test_stream_keeps_replies(raw, size):
    assert _stream(raw, size) == raw


def test_a_long_reply_is_not_held_until_the_end():
    s = HeadScrubber()
    assert s.feed("Paris is the capital of France, and it") == "Paris is the capital of France, and it"


def test_an_unfinished_cue_lookalike_is_released_at_the_end():
    assert _stream("They asked me nothing at all.", 4) == "They asked me nothing at all."


def _run(events):
    async def source():
        for e in events:
            yield json.dumps(e) + "\n"

    async def collect():
        return [json.loads(line) async for line in scrub_stream(source())]

    return asyncio.run(collect())


def test_each_flush_starts_a_fresh_head():
    out = _run([
        {"type": "token", "value": "I'll fetch it."},
        {"type": "flush"},
        {"type": "token", "value": "<nooutput>"},
        {"type": "token", "value": "It is a page."},
    ])
    assert out[0] == {"type": "token", "value": "I'll fetch it."}
    assert out[1] == {"type": "flush"}
    assert "".join(e["value"] for e in out[2:]) == "It is a page."


def test_other_events_pass_and_release_what_was_held():
    out = _run([
        {"type": "token", "value": "<"},
        {"type": "tool_status", "tool": "web", "state": "calling"},
        {"type": "packet", "value": {"a": 1}},
    ])
    assert out == [
        {"type": "token", "value": "<"},
        {"type": "tool_status", "tool": "web", "state": "calling"},
        {"type": "packet", "value": {"a": 1}},
    ]


@pytest.mark.parametrize("prose", ["Paris.", "[nano]\n\nParis.", "Hi! Paris.", "It is Paris, of course."])
def test_an_answer_before_a_tool_call(prose):
    assert answered_already(prose)


@pytest.mark.parametrize("prose", ["", "[nano]\n\n", "I'll fetch the page.", "Let me check.", "Hi there!",
                                   "Sure!", "Here is what I found:", "Paris is the", "Is it Paris?",
                                   "Good question.", "[Core]\n\nI'll fetch the URL and read it. "])
def test_not_an_answer_before_a_tool_call(prose):
    assert not answered_already(prose)
