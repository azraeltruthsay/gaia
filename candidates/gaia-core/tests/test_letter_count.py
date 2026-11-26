"""Letter questions get the right word spelled out (2026-11-26: "the word
\\"hillbilly\\"" was read as the word "word", and she answered "None")."""
import pytest

from gaia_core.cognition.letter_count import letter_question, work_it_out, working_prompt


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


def test_she_works_it_out_for_the_right_word():
    asked = []

    def mind(messages):
        asked.append(messages[-1]["content"])
        return "h\ni\nl 1\nl 2\nb\ni\nl 3\nl 4\ny\nTotal: 4"
    note = work_it_out('how many l\'s are in the word "hillbilly"', mind)
    assert asked == [working_prompt("hillbilly", "l")]
    assert "Total: 4" in note and '"hillbilly"' in note


def test_no_answer_is_supplied_by_code():
    note = work_it_out('how many r\'s are in the word "strawberry"?', lambda m: "Total: 2")
    assert "3" not in note          # her (wrong) working stands; code adds no count


def test_nothing_for_other_questions_or_a_failed_step():
    assert work_it_out("how do you spell necessary?", lambda m: "x") == ""
    assert work_it_out("What is the capital of France?", lambda m: "x") == ""

    def broken(messages):
        raise RuntimeError("mind unavailable")
    assert work_it_out('how many r\'s in "strawberry"', broken) == ""
    assert work_it_out('how many r\'s in "strawberry"', lambda m: "  ") == ""
