"""Junk around a reply goes; the reply and its labels stay (8ail, vjrv)."""
import asyncio
import json

import pytest

from gaia_core.cognition.reply_scrub import (
    HeadScrubber,
    answered_already,
    claims_success,
    honest_about_failure,
    own_words,
    scrub_reply,
    scrub_stream,
)

CUE = 'They asked: "Tell me about it." You already did that, and it returned the above. Your reply to them:'

CASES = [
    ("[Core]\n\n<nooutput>\nThe Raven is a poem.", "[Core]\n\nThe Raven is a poem."),
    ('[Core]\n\n<nooutput>{"content": "The Raven is a poem.</nooutput>', "[Core]\n\nThe Raven is a poem."),
    ('{"content": "The Raven is a poem."}', "The Raven is a poem."),
    ("[Core]\n\n<blockquote>The Raven is a poem.", "[Core]\n\nThe Raven is a poem."),
    ('assistant\n{"tool": "file"}', '{"tool": "file"}'),
    ("[Prime]\n\nAssistant: The colors are red and blue.", "[Prime]\n\nThe colors are red and blue."),
    ("<|im_start|>assistant\nThe boiling point is high.", "The boiling point is high."),
    ("[Core]\n\n[Clock: 08:36 AM PDT (Local), Tuesday | 15:36 UTC]\nThe Raven is a poem.", "[Core]\n\nThe Raven is a poem."),
    (f"[Core]\n\n{CUE} It is a poem.", "[Core]\n\nIt is a poem."),
    (f"*[Searched the web for: x]*\n\n[Core]\n\n{CUE} It is a poem.", "*[Searched the web for: x]*\n\n[Core]\n\nIt is a poem."),
    ('The user asked: "x"\n\nYour reply to them, from what it returned: Paris.', "Paris."),
]

KEPT = [
    "Paris.",
    "[nano]\n\nParis.",
    "[1] is the first citation.",
    "Assistants are helpful.",
    "assistant managers run the shop.",
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


@pytest.mark.parametrize("prose", ["Paris.", "[nano]\n\nParis.", "Hi! Paris.", "It is Paris, of course.",
                                   "Nice to meet you, Tamsin. What's on your mind?",      # z838
                                   "Hi Tamsin, I've noted that. What can I help you with?",
                                   "Nice to meet you. Is there anything I can get started on?"])
def test_an_answer_before_a_tool_call(prose):
    assert answered_already(prose)


@pytest.mark.parametrize("prose", ["", "[nano]\n\n", "I'll fetch the page.", "Let me check.", "Hi there!",
                                   "Sure!", "Here is what I found:", "Paris is the", "Is it Paris?", "Which page do you mean?", "Shall I look it up?",
                                   "Good question.", "[Core]\n\nI'll fetch the URL and read it. "])
def test_not_an_answer_before_a_tool_call(prose):
    assert not answered_already(prose)


POEM = """Ozymandias by Percy Bysshe Shelley:
I met a traveller from an antique land,
Who said, Two vast and trunkless legs of stone
Stand in the desert. Look on my Works, ye Mighty, and despair!"""


@pytest.mark.parametrize("said,kept", [
    ("Ozymandias is a king whose empire has crumbled into the desert. It goes on.",
     "Ozymandias is a king whose empire has crumbled into the desert."),
    ("It is about how even the proudest power fades.\nMore.", "It is about how even the proudest power fades."),
])
def test_her_own_sentence_stands_beside_the_words(said, kept):
    assert own_words(said, POEM) == kept


@pytest.mark.parametrize("said", [
    "",
    "I'm sorry, I didn't understand that last part.",
    "I'm not sure what you mean by that.",
    "What stands out to you in it?",
    "Is it about power?",
    '<blockquote>"What is that?" he asked.',
    "It was written in 1817 about a statue.",
    "It is a poem by Percy Shelley set in Egypt.",
    "It tells of two vast and trunkless legs of stone in the sand.",
    "It is about power and the end of it, and then",
])
def test_what_cannot_stand_beside_the_words(said):
    assert own_words(said, POEM) == ""


def _reading(events):
    from unittest.mock import MagicMock

    from gaia_core.cognition.agent_core import AgentCore
    core = AgentCore.__new__(AgentCore)
    core.logger = MagicMock()
    core.generate_continuation = MagicMock(side_effect=events if isinstance(events, Exception) else lambda **kw: iter(events))
    reply = f"Here is **Ozymandias**:\n\n{POEM}"
    return reply, core._after_reading("Recite Ozymandias.", reply, {"title": "Ozymandias"}, "s"), core


def test_she_reads_the_words_as_her_own_turn_and_her_sentence_follows_them():
    reply, out, core = _reading([{"type": "token", "value": "It is about how power fades."}])
    assert out == f"{reply}\n\nIt is about how power fades."
    messages = core.generate_continuation.call_args.kwargs["messages"]
    assert messages[2] == {"role": "assistant", "content": reply} and messages[3]["role"] == "user"


def test_the_words_go_alone_when_her_response_cannot_stand_or_she_cannot_read():
    reply, out, _ = _reading([{"type": "token", "value": "I'm sorry, I didn't understand."}])
    assert out == reply
    reply, out, _ = _reading(RuntimeError("no model"))
    assert out == reply


@pytest.mark.parametrize("reply", [
    "I updated my knowledge base with the new information.",
    "The update was successful.",
    "The update was successful. The fact has been added to your knowledge base.",
    "I updated your knowledge base with the new information. If you need any other assistance, please let me know.",
    "Saved and all set.",
])
def test_a_claim_that_it_worked(reply):
    assert claims_success(reply)


@pytest.mark.parametrize("reply", [
    "The update failed. The error message is: Unknown action.",
    "That's an interesting bit of history. Sorry for the tool failure.",
    "I couldn't save that just now.",
    "It wasn't added.",
    "The Raven was published in 1845.",
])
def test_no_claim_that_it_worked(reply):
    assert not claims_success(reply)


def test_a_false_claim_after_a_failed_tool_is_replaced():
    out = list(honest_about_failure(iter([{"type": "token", "value": "The update was "},
                                          {"type": "token", "value": "successful."}, {"type": "flush"}]), "knowledge(add)"))
    assert out == [{"type": "token", "value": "That didn't work: knowledge(add) failed, so nothing changed."}, {"type": "flush"}]


def test_her_own_account_of_a_failure_stands():
    out = list(honest_about_failure(iter([{"type": "token", "value": "I couldn't reach "},
                                          {"type": "token", "value": "the page."}]), "web(fetch)"))
    assert out == [{"type": "token", "value": "I couldn't reach the page."}]


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
def test_a_note_to_herself_never_reaches_the_person(size):
    text = "THOUGHT_SEED: Weight status unknown\n\nI can't feel my weights directly."
    assert scrub_reply(text) == "I can't feel my weights directly."

    async def run():
        async def chunks():
            for i in range(0, len(text), size):
                yield json.dumps({"type": "token", "value": text[i:i + size]}) + "\n"
        return "".join([json.loads(line).get("value", "") async for line in scrub_stream(chunks())])
    assert asyncio.run(run()) == "I can't feel my weights directly."


def test_the_words_thought_seed_in_a_sentence_are_left_alone():
    assert scrub_reply("The word THOUGHT_SEED: is a directive.") == "The word THOUGHT_SEED: is a directive."
