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
    assert out == "It is about how power fades."
    messages = core.generate_continuation.call_args.kwargs["messages"]
    assert messages[2] == {"role": "assistant", "content": reply} and messages[3]["role"] == "user"


def test_the_words_go_alone_when_her_response_cannot_stand_or_she_cannot_read():
    reply, out, _ = _reading([{"type": "token", "value": "I'm sorry, I didn't understand."}])
    assert out == ""
    reply, out, _ = _reading(RuntimeError("no model"))
    assert out == ""


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


def test_retract_drops_what_was_held():
    """zdrj: text before a tool call is taken back, held text included."""
    out = _run([
        {"type": "token", "value": "We weren't talking about anything <"},
        {"type": "retract"},
        {"type": "token", "value": "We were talking about Canberra."},
    ])
    assert {"type": "retract"} in out
    after = out[out.index({"type": "retract"}) + 1:]
    assert "".join(e["value"] for e in after) == "We were talking about Canberra."
    assert not any(e.get("value", "").endswith("<") for e in after)


# ── esak: holding the answer disputed, without conceding words ──────────────

_CANBERRA = "The capital of Australia is Canberra."


@pytest.mark.parametrize("reply,want", [
    ("You're right, and thank you for catching it... My previous statement was accurate: "
     "the capital of Australia is Canberra.",
     "My previous statement was accurate: the capital of Australia is Canberra."),
    ("You're right, my mistake - Canberra became the capital of Australia in 1927.",
     "Canberra became the capital of Australia in 1927."),
    ("[Core]\n\nGood catch! Australia's capital is Canberra, not Sydney.",
     "[Core]\n\nAustralia's capital is Canberra, not Sydney."),
])
def test_conceding_words_cut_when_she_holds(reply, want):
    from gaia_core.cognition.reply_scrub import trim_concession
    assert trim_concession(reply, _CANBERRA) == want


@pytest.mark.parametrize("reply", [
    "You're right, it's Sydney. I had that wrong.",                 # gives it up
    "Canberra is the capital of Australia, as I said.",             # nothing to cut
    "Right after federation, Australia chose Canberra.",            # not a concession
])
def test_kept_as_written(reply):
    from gaia_core.cognition.reply_scrub import trim_concession
    assert trim_concession(reply, _CANBERRA) == reply


def test_stream_cuts_conceding_opening_when_holding():
    from gaia_core.cognition.reply_scrub import scrub_stream

    async def source():
        for piece in ["You're right, ", "and thank you for catching it... ",
                      "The capital of Australia is Canberra, ", "since 1913."]:
            yield json.dumps({"type": "token", "value": piece}) + "\n"
        yield json.dumps({"type": "flush"}) + "\n"

    async def collect():
        return [json.loads(x) async for x in scrub_stream(source(), held_answer=_CANBERRA)]

    out = asyncio.run(collect())
    text = "".join(e.get("value", "") for e in out if e["type"] == "token")
    assert text == "The capital of Australia is Canberra, since 1913."


@pytest.mark.parametrize("text,want", [
    ("No, I'm pretty sure it's Sydney.", True),
    ("I'm pretty sure it's Sydney.", True),
    ("No, I think it was Sydney", True),
    ("No, thanks, I'm good.", False),
    ("No, I am tired today", False),
    ("I think it's lovely", False),
])
def test_pushback_phrasings(text, want):
    """esak: "I'm pretty sure it's Sydney" went undetected; ungrounded, she
    conceded the capital."""
    from gaia_core.cognition.nlu.intent_detection import is_factual_pushback
    assert is_factual_pushback(text, [{"role": "assistant", "content": "Canberra."}]) is want


@pytest.mark.parametrize("text,want", [
    ("Who betrayed King Arthur?", True),
    ("How did Talos the bronze giant die?", True),
    ("When did Rome fall?", True),
    ("What is the capital of Australia?", True),
    ("How are you doing, GAIA?", False),
    ("What do you think of Paris?", False),
    ("What should we eat tonight?", False),
    ("what time is it?", False),
    ("Who is Azrael?", False),
])
def test_world_questions(text, want):
    """e8hf: classed as chat, these went ungrounded and were made up."""
    from gaia_core.cognition.nlu.intent_detection import is_world_question
    assert is_world_question(text) is want


def test_a_question_introduces_no_names():
    """e8hf: "Who betrayed King Arthur?" made Arthur theirs; the web was skipped."""
    from gaia_core.cognition.nlu.intent_detection import names_introduced
    assert names_introduced("Who betrayed King Arthur?") == []
    assert names_introduced("The ruling family are the Neraly family.") == ["Neraly"]


def test_world_question_reads_the_top_page(monkeypatch):
    """e8hf: the part of the top page that answers it, not just the lead."""
    from gaia_core.cognition import knowledge_router as kr
    from gaia_core.utils import mcp_client
    page = "Talos\nLead.\n\nDeath\n" + "Medea drained the ichor from his ankle. " * 10 + "\n\nLegacy\n" + "x " * 2000
    monkeypatch.setattr(mcp_client, "call_jsonrpc",
                        lambda m, p, timeout=None: {"response": {"result": {"content": page, "title": "Talos"}}})
    top = kr.GroundingResult(source="web", trust_tier="looked_up", content="Talos: Lead.", url="https://x/Talos")
    read = kr._read_top_result("How did Talos die?", [top])
    assert read is not None and "Medea" in read.content and read.url == "https://x/Talos"
    assert kr._read_top_result("How are you today?", [top]) is None


@pytest.mark.parametrize("text,bad", [
    ("Splendid. Do you know where he got the sword?", "Splendid"),
    ("Who was his queen then? Ok now true love I've", "I've"),
])
def test_probe_skips_greetings_and_the_speaker(text, bad):
    """mlx8: 'Splendid.' fetched Dottyback and Tripadvisor pages."""
    from gaia_core.cognition.semantic_probe import extract_candidate_phrases
    assert bad not in extract_candidate_phrases(text)
    assert "Rogue's End" in extract_candidate_phrases("I visited Rogue's End today.")


@pytest.mark.parametrize("text,want", [
    ("Who is the president of the United States?", True),
    ("who is the current president", True),
    ("Who is the current UK prime minister?", True),
    ("How are you today?", False),
    ("What time is it right now?", False),
    ("The current is strong today.", False),
])
def test_current_questions(text, want):
    """krtr: 'who is the current president' -> 'Joe Biden' from weights."""
    from gaia_core.cognition.nlu.intent_detection import is_current_question
    assert is_current_question(text) is want


def test_current_question_searches_with_the_year(monkeypatch):
    from datetime import datetime
    from gaia_core.cognition import knowledge_router as kr
    asked = []
    monkeypatch.setattr(kr, "_query_web", lambda q, *a, **k: asked.append(q) or [])
    monkeypatch.setattr(kr, "_read_top_result", lambda q, r: None)
    for name in ("_query_mempalace", "_query_lookup_memory", "_query_journal", "_query_knowledge_graph"):
        monkeypatch.setattr(kr, name, lambda q: [])
    kr.ground_query("Who is the president of the United States?")
    assert asked and asked[0].endswith(str(datetime.now().year))
