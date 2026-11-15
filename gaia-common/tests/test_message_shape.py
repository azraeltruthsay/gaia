"""asking_text: the person's own words, not the document they pasted (5qv4)."""
from gaia_common.utils.message_shape import ASK_MAX_CHARS, asking_text, is_long_message

DOC = "Gear 1 runs Core on the GPU. The wake path broke; mjcg was fixed by hand.\n\n" * 20


def test_a_short_message_is_all_ask():
    assert asking_text("  What GPU are you on?  ") == "What GPU are you on?"
    assert not is_long_message("hi")


def test_words_before_pasted_material():
    msg = "What do you think of this journal?\n\n----- JOURNAL -----\n" + DOC
    assert asking_text(msg) == "What do you think of this journal?"
    assert is_long_message(msg)


def test_a_note_after_closed_off_material_counts():
    msg = "Have a look:\n```\n" + DOC + "```\nWhich line broke the build?"
    assert asking_text(msg) == "Have a look:\n\nWhich line broke the build?"


def test_material_running_past_the_last_separator_is_not_ask():
    msg = "Read this.\n\n---\n" + DOC + "\n---\n" + DOC
    assert asking_text(msg) == "Read this."


def test_no_separator_first_and_last_paragraphs():
    msg = "Here's a long one.\n\n" + DOC + "So what do you make of it?"
    ask = asking_text(msg)
    assert ask.startswith("Here's a long one.") and ask.endswith("So what do you make of it?")
    assert len(ask) < 2 * ASK_MAX_CHARS


def test_material_first_then_question():
    msg = "```\n" + DOC + "```\nWhat do you think?"
    assert asking_text(msg) == "What do you think?"
