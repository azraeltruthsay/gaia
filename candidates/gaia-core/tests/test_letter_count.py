"""Letter questions get the right word spelled out (2026-11-26: "the word
\\"hillbilly\\"" was read as the word "word", and she answered "None")."""
import pytest

from gaia_core.cognition.letter_count import (PAIR_ON_CPU, PAIR_ON_GPU, letter_question, total_of,
                                              work_it_out, working_prompt)


@pytest.mark.parametrize("q,expected", [
    ('how many r\'s are in the word "strawberry"?', ("strawberry", "r")),
    ('how many l\'s are in the word "hillbilly"', ("hillbilly", "l")),
    ("How many times does the letter r appear in 'strawberry'? Give the final answer on the last line.",
     ("strawberry", "r")),
    ("how many rs in strawberry", ("strawberry", "r")),
    ("how many r's in the word strawberry", ("strawberry", "r")),
    ("How many 'e's are in “excellence”?", ("excellence", "e")),
    ("how many times does the letter s appear in Mississippi", ("mississippi", "s")),
    ("how many s's in the name Jessica?", ("jessica", "s")),
    ("how do you spell necessary?", ("necessary", None)),
    ('what letters are in the word "rhythm"', ("rhythm", None)),
])
def test_finds_the_word_and_letter(q, expected):
    assert letter_question(q) == expected


@pytest.mark.parametrize("q", [
    "how many I can eat in a day",
    "how many people live in Seattle?",
    "how many letters should I write to my senator?",
    "What is the first line of The Raven?",
    "",
])
def test_not_a_letter_question(q):
    assert letter_question(q) is None


def test_she_walks_the_right_word_twice_and_agreement_passes_one_working():
    asked = []

    def mind(messages):
        asked.append(messages[-1]["content"])
        return "h\ni\nl 1\nl 2\nb\ni\nl 3\nl 4\ny\nTotal: 4"
    note = work_it_out('how many l\'s are in the word "hillbilly"', mind)
    assert asked == [working_prompt("hillbilly", "l", w) for w in PAIR_ON_GPU]
    assert "Total: 4" in note and '"hillbilly"' in note and "different" not in note


def test_on_cpu_uses_the_cpu_pair():
    asked = []
    work_it_out('how many r\'s in "strawberry"', lambda m: asked.append(m[-1]["content"]) or "Total: 3", on_gpu=False)
    assert asked == [working_prompt("strawberry", "r", w) for w in PAIR_ON_CPU]


def test_a_disagreement_is_broken_by_the_third_walk():
    asked = []
    answers = iter(["Total: 4", "Total: 3", "m i s s i s s i p p i\nTotal: 4"])

    def mind(m):
        asked.append(m[-1]["content"])
        return next(answers)
    note = work_it_out('how many s\'s are in "mississippi"?', mind)
    assert len(asked) == 3 and "Total: 4" in note and "different" not in note


def test_agreeing_walks_need_no_third():
    asked = []
    work_it_out('how many s\'s are in "mississippi"?', lambda m: asked.append(1) or "Total: 4")
    assert len(asked) == 2


def test_no_two_agree_she_is_told_only_that():
    answers = iter(["Working A\nTotal: 5", "Working B\nTotal: 2", "Working C\nTotal: 3"])
    note = work_it_out('how many s\'s are in "mississippi"?', lambda m: next(answers))
    assert "different totals (5 and 2 and 3)" in note and "not sure" in note
    assert "Working" not in note        # the workings she'd talk about stay out


def test_no_answer_is_supplied_by_code():
    note = work_it_out('how many r\'s are in the word "strawberry"?', lambda m: "Total: 2")
    assert "3" not in note          # her (wrong) working stands; code adds no count


def test_total_of():
    assert total_of("1. p\n2. a\nTotal: 3") == 3
    assert total_of("so the answer is three") == 3
    assert total_of("no idea") is None


def test_nothing_for_other_questions_or_failed_walks():
    assert work_it_out("how do you spell necessary?", lambda m: "x") == ""
    assert work_it_out("What is the capital of France?", lambda m: "x") == ""

    def broken(messages):
        raise RuntimeError("mind unavailable")
    assert work_it_out('how many r\'s in "strawberry"', broken) == ""
    assert work_it_out('how many r\'s in "strawberry"', lambda m: "  ") == ""


def _packet(prompt):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Context, SessionHistoryRef
    return CognitionPacket(
        context=Context(session_history_ref=SessionHistoryRef(type="hash", value=""),
                        cheatsheets=[], constraints=None, relevant_history_snippet=[]),
        content=Content(original_prompt=prompt, data_fields=[]))


@pytest.mark.parametrize("slim", [True, False])
def test_working_reaches_the_prompt_not_the_question(slim):
    """The working goes into the model's prompt after the question, as a data
    field; the packet's own prompt text (what history and routing see) stays
    the person's words (2026-11-26: appended to user_input, it was saved as
    theirs and set off a web search and a shift to Prime)."""
    from gaia_common.protocols.cognition_packet import DataField
    from gaia_core.utils.prompt_builder import build_from_packet
    q = 'how many s\'s are in the word "mississippi"?'
    packet = _packet(q)
    packet.content.data_fields.append(DataField(key="letter_working", value="WORKING-NOTE Total: 4",
                                                type="string", source="letter_count"))
    messages = build_from_packet(packet, slim_mode=slim)
    users = [m for m in messages if m["role"] == "user"]
    assert users[-1]["content"].startswith(q)
    assert users[-1]["content"].rstrip().endswith("WORKING-NOTE Total: 4")
    assert sum("WORKING-NOTE" in str(m["content"]) for m in messages) == 1
    assert packet.content.original_prompt == q
