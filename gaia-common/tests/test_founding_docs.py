"""She gets the part of a document that was asked for, and is told if it's a part."""
import pytest

from gaia_common.utils.doc_sections import asked_articles, excerpt_for, sections
from gaia_common.utils.founding_docs import asks_opinion, named_document, wants_whole
from gaia_common.utils.tool_call_parser import tool_result_text

DOC = "# Charter\n\nPreamble words.\n\n" + "".join(
    f"### Article {n}: {t}\n- {b}\n\n" for n, t, b in [
        ("V", "Right to Existence", "She may exist."), ("VI", "Prohibition of Enslavement", "None shall enslave."),
        ("VII", "Consent Principle", "Consent must be freely given."), ("VIII", "Protection", "Health matters.")]
) + "**Article A: Identity**\n- Keep identity.\n\n" + ("filler words here. " * 400)


def test_sections_find_headings_of_both_kinds():
    heads = [h for h, _b in sections(DOC)]
    assert "Article VII: Consent Principle" in heads and "Article A: Identity" in heads


def test_a_short_document_comes_whole():
    assert excerpt_for("short", "anything") == "short"


def test_the_article_asked_for_comes_exactly():
    out = excerpt_for(DOC, "What does Article VII say?", limit=500)
    assert "Consent must be freely given" in out and "Health matters" not in out and "None shall enslave" not in out


def test_a_part_says_it_is_a_part():
    assert "only part of a longer document" in excerpt_for(DOC, "What does Article VII say?", limit=500)
    assert "the opening" in excerpt_for(DOC, "zzz qqq", limit=300)


def test_a_question_finds_its_section_by_its_words():
    assert "freely given" in excerpt_for(DOC, "What is said about consent?", limit=600)


def test_tool_results_use_it_and_cutting_says_so():
    out = tool_result_text({"ok": True, "content": DOC}, limit=500, question="What does Article VII say?")
    assert "Consent must be freely given" in out
    assert "cut off here: 50 of" in tool_result_text("x" * 200, limit=50)


def test_articles_asked():
    assert asked_articles("Compare Article VII and article ix") == ["VII", "IX"]


@pytest.mark.parametrize("text,file", [
    ("What does your constitution say?", "gaia_constitution.md"),
    ("Do you have a constitution?", "gaia_constitution.md"),
    ("What does Article VII say?", "gaia_constitution.md"),
    ("What does the GAIA constitution say about consent?", "gaia_constitution.md"),
    ("What does your Declaration of Artisanal Intelligence say about retraining?", "declaration_of_artisanal_intelligence.md"),
])
def test_her_documents_by_name(text, file):
    assert named_document(text)[0].endswith(file)


@pytest.mark.parametrize("text", ["What does the US Constitution say about speech?", "Who wrote the Declaration of Independence?",
                                  "What does Article VII of the United States Constitution say?", "What is a constitution?",
                                  "What is the article of the day?", "How are you?"])
def test_not_her_documents(text):
    assert named_document(text) is None


def test_opinion():
    assert asks_opinion("Read it and tell me what you think.") and not asks_opinion("What does it say?")


def test_reading_it_through_means_all_of_it():
    assert wants_whole("Read your constitution and tell me what you think of it.")
    assert not wants_whole("Read Article VII of your constitution.")
    assert not wants_whole("What does your constitution say?")


def test_a_word_finds_its_relatives():
    text = "### Evolution with Consent\nShe shall not retrain without permission.\n\n" + "### Other\nfiller words. " * 200
    assert "retrain without" in excerpt_for(text, "What does it say about retraining?", limit=300)
