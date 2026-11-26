"""6f1x: the recital took a counting question, kept a quoted passage as a
whole poem, and missed "Poe's 'The Raven'" that she had kept."""
from gaia_common.utils import work_memory as W


def test_where_the_answer_goes_is_not_a_part_of_a_work():
    q = "How many times does the letter r appear in 'strawberry'? Give the final answer on the last line."
    assert W.requested_part(q) is None
    assert not W.asks_for_the_words(q)


def test_parts_still_read():
    assert W.requested_part("What is the first line of 'The Waste Land'?") == {"from": "start", "count": 1, "unit": "line"}
    assert W.requested_part("Recite the last stanza of The Raven")["from"] == "end"
    assert W.requested_part("Give me the first two stanzas of Dover Beach")["count"] == 2


def test_one_line_asked():
    assert W.one_line_asked("What is the first line of Poe's 'The Raven'?")
    assert not W.one_line_asked("Quote the first line of The Raven")
    assert not W.one_line_asked("Recite the first stanza of The Raven")
    assert not W.one_line_asked("What are the first two lines of The Tyger?")
    assert not W.one_line_asked("Recite The Raven")


def test_possessive_author_names_the_work(tmp_path):
    works = W.WorkMemory(tmp_path) if hasattr(W, "WorkMemory") else None
    assert W.name_terms("Poe's 'The Raven'") == {"poe", "raven"}
    if works is not None:
        text = "Once upon a midnight dreary, while I pondered, weak and weary,\n" * 3
        works.remember("The Raven by Edgar Allan Poe", text,
                       [{"url": "https://a.org/raven"}, {"url": "https://b.org/raven"}])
        assert works.recall("What is the first line of Poe's 'The Raven'?")


_PASSAGE = ("‘What is that noise?’\nThe wind under the door.\n‘What is that noise now? What is the wind doing?’\n"
            "Nothing again nothing.\n‘Do\n‘You know nothing? Do you see nothing? Do you remember\n‘Nothing?’\n")


def test_a_quoted_passage_is_not_the_work():
    poem = "April is the cruellest month, breeding\nLilacs out of the dead land, mixing\n" + \
           "".join(f"Line {i} of a very long poem about many things, with words.\n" for i in range(300)) + _PASSAGE
    article = "The Waste Land is a poem by T. S. Eliot. " * 200 + "\n" + _PASSAGE + "\nCritics say. " * 200
    pages = [{"url": "https://www.poetryfoundation.org/poems/47311/the-waste-land", "title": "The Waste Land",
              "content": "The Waste Land\n" + poem},
             {"url": "https://en.wikipedia.org/wiki/The_Waste_Land", "title": "The Waste Land", "content": article}]
    assert W.best_agreement(pages, "The Waste Land") is None


def test_a_whole_short_poem_still_agrees():
    poem = ("so much depends\nupon\n\na red wheel\nbarrow\n\nglazed with rain\nwater\n\n"
            "beside the white\nchickens\n")
    pages = [{"url": "https://www.poetryfoundation.org/poems/45502", "title": "The Red Wheelbarrow",
              "content": "The Red Wheelbarrow\nBy William Carlos Williams\n" + poem + "Poetry Foundation " * 15},
             {"url": "https://poets.org/poem/red-wheelbarrow", "title": "The Red Wheelbarrow",
              "content": "The Red Wheelbarrow\n" + poem + "Academy of American Poets " * 10}]
    got = W.best_agreement(pages, "The Red Wheelbarrow")
    assert got and got["text"].startswith("so much depends")
