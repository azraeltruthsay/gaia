import pytest
"""yx93: work memory — what two sites agree on, kept word for word."""

from gaia_common.utils.work_memory import (
    WorkMemory, agreed_text, best_agreement, compact_work, excerpt,
    asks_for_the_words, requested_part, strip_heading, strip_trailer, work_name,
)

STANZA_1 = """Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.
“’Tis some visitor,” I muttered, “tapping at my chamber door—
Only this and nothing more.”"""
STANZA_2 = """Ah, distinctly I remember it was in the bleak December;
And each separate dying ember wrought its ghost upon the floor.
Eagerly I wished the morrow;—vainly I had sought to borrow
From my books surcease of sorrow—sorrow for the lost Lenore—
For the rare and radiant maiden whom the angels name Lenore—
Nameless here for evermore."""
POEM = f"{STANZA_1}\n\n{STANZA_2}"
NAME = "The Raven by Edgar Allan Poe"
PAGE_A = {"url": "https://poets.org/poem/raven", "domain": "poets.org", "title": "The Raven by Edgar Allan Poe",
          "content": f"Menu\nDonate\nThe Raven\nEdgar Allan Poe\n{POEM}\nThis poem is in the public domain."}
# Another edition: other punctuation, one changed word, other furniture.
PAGE_B = {"url": "https://www.poetryfoundation.org/poems/48860", "domain": "www.poetryfoundation.org",
          "title": "The Raven | The Poetry Foundation",
          "content": "Subscribe\nThe Raven\nBy Edgar Allan Poe\n" + POEM.replace("visitor", "visiter").replace(";", ",")
                     + "\nMore poems"}
CATALOG = {"url": "https://www.gutenberg.org/ebooks/17192", "domain": "www.gutenberg.org",
           "title": "The Raven by Edgar Allan Poe | Project Gutenberg",
           "content": '"The Raven" by Edgar Allan Poe is a narrative poem first published in January 1845. '
                      "A raven speaks only one word. Download for free.\n"}


def test_agreed_text_keeps_only_the_shared_work():
    agreed = agreed_text(PAGE_A["content"], PAGE_B["content"])
    assert STANZA_1 in agreed and STANZA_2 in agreed
    assert "Donate" not in agreed and "public domain" not in agreed


def test_lines_shared_only_in_part_are_left_out():
    a = f"Our site has its own menu ending with once upon\n{STANZA_1}\nThe end of site one here"
    b = f"Their pages have one menu ending with once upon\n{STANZA_1}\nThe end of site two here"
    assert agreed_text(a, b) == STANZA_1


def test_unrelated_pages_do_not_agree():
    assert agreed_text(PAGE_A["content"], CATALOG["content"]) is None


def test_best_agreement_learns_from_two_sites():
    got = best_agreement([CATALOG, PAGE_A, PAGE_B], NAME)
    assert got["text"].startswith("Once upon a midnight dreary")
    assert got["text"].endswith("Nameless here for evermore.")
    assert {s["domain"] for s in got["sources"]} == {"poets.org", "www.poetryfoundation.org"}


def test_two_pages_of_one_site_are_not_two_sources():
    same = dict(PAGE_B, domain="poets.org", url="https://poets.org/other")
    assert best_agreement([PAGE_A, same], NAME) is None


def test_shared_blurb_is_not_the_work():
    mirror = dict(CATALOG, url="https://en.wikisource.org/x", domain="en.wikisource.org")
    assert best_agreement([CATALOG, mirror], NAME) is None


def test_shared_caption_is_not_the_work():
    caption = ("United States Declaration of Independence, broadside printed by John Dunlap in the year 1776.\n"
               "United States Declaration of Independence, broadside printed by Mary Goddard in the year 1776.")
    pages = [{"url": f"https://{d}/x", "domain": d, "title": "Declaration of Independence", "content": f"{d}\n{caption}"}
             for d in ("en.wikipedia.org", "en.wikisource.org")]
    assert best_agreement(pages, "the Declaration of Independence") is None


def test_pages_about_another_work_do_not_count():
    assert best_agreement([PAGE_A, PAGE_B], "Annabel Lee by Edgar Allan Poe") is None
    assert best_agreement([PAGE_A, PAGE_B], "The Raven by Ted Hughes") is None


def test_stanza_layout_beats_a_flat_copy():
    flat = dict(PAGE_B, content=PAGE_B["content"].replace("\n\n", "\n"))
    got = best_agreement([flat, PAGE_A], NAME)
    assert "\n\n" in got["text"]


def test_work_name_finds_the_work():
    assert work_name("Please recite the first three stanzas of The Raven by Edgar Allan Poe.") == NAME
    assert work_name("Recite The Raven for me") == "The Raven"
    assert work_name('Can you recite "Ozymandias"?') == "Ozymandias"
    assert work_name("Read me the full text of the Gettysburg Address") == "the Gettysburg Address"
    assert work_name("give me the last 2 lines of Sonnet 18") == "Sonnet 18"
    assert work_name("What is the first line of The Raven?") == "The Raven"


def test_requested_part():
    assert requested_part("the first three stanzas of X") == {"from": "start", "count": 3, "unit": "stanza"}
    assert requested_part("the last stanza of X") == {"from": "end", "count": 1, "unit": "stanza"}
    assert requested_part("the opening 2 lines of X") == {"from": "start", "count": 2, "unit": "line"}
    assert requested_part("the third verse of X") == {"from": "nth", "count": 3, "unit": "verse"}
    assert requested_part("recite The Raven") is None
    assert requested_part("Recite The First Noel") is None


def test_excerpt():
    assert excerpt(POEM, requested_part("first stanza")) == STANZA_1
    assert excerpt(POEM, requested_part("last stanza")) == STANZA_2
    assert excerpt(POEM, requested_part("second stanza")) == STANZA_2
    assert excerpt(POEM, requested_part("first 2 lines")).splitlines()[1].startswith("Over many")
    assert excerpt(POEM, None) == POEM
    # Can't be cut honestly: more than there is, or no stanza breaks stored.
    assert excerpt(POEM, requested_part("first nine stanzas")) is None
    assert excerpt(POEM.replace("\n\n", "\n"), requested_part("first stanza")) is None


def test_strip_heading():
    assert strip_heading(f"The Raven\nby Edgar Allan Poe\n\n{POEM}", NAME) == POEM
    assert strip_heading(POEM, NAME) == POEM
    opening = "Do not go gentle into that good night,\nOld age should burn and rave at close of day;"
    assert strip_heading(opening, "Do Not Go Gentle Into That Good Night by Dylan Thomas") == opening


def _kept(tmp_path):
    mem = WorkMemory(tmp_path)
    rid = mem.remember(NAME, POEM, [{"url": "a"}, {"url": "b"}])
    return mem, rid


def test_recall_by_name_or_request(tmp_path):
    mem, rid = _kept(tmp_path)
    for query in (NAME, "The Raven", "Raven", "Recite the first stanza of The Raven by Poe"):
        assert mem.recall(query)["id"] == rid, query
    assert mem.recall(query)["text"] == POEM


def test_recall_does_not_guess_a_different_work(tmp_path):
    mem, _ = _kept(tmp_path)
    for query in ("Annabel Lee by Edgar Allan Poe", "The Raven by Ted Hughes", "a poem by Poe",
                  "Edgar Allan Poe", "The Raven Part 2", ""):
        assert mem.recall(query) is None, query


def test_learning_again_keeps_the_fuller_text_and_all_sources(tmp_path):
    mem, rid = _kept(tmp_path)
    assert mem.remember("The Raven", STANZA_1, [{"url": "a"}, {"url": "c"}]) == rid
    rec = mem.recall("The Raven")
    assert rec["text"] == POEM
    assert [s["url"] for s in rec["sources"]] == ["a", "b", "c"]


def test_note_recited_and_stats(tmp_path):
    mem, rid = _kept(tmp_path)
    mem.note_recited(rid)
    assert mem.recall(NAME)["times_recited"] == 1
    assert mem.stats() == {"works": 1}
    assert compact_work(mem.recall(NAME), max_chars=10)["truncated_from_chars"] == len(POEM)


def test_strip_trailer_drops_shared_credits():
    credit = 'Edgar Allan Poe, "The Raven" from Collected Poems. Copyright 1945.'
    assert strip_trailer(f"{POEM}\n{credit}", NAME) == POEM
    assert strip_trailer(POEM, NAME) == POEM


def test_mentioned_finds_a_work_named_in_a_question(tmp_path):
    mem, rid = _kept(tmp_path)
    assert mem.mentioned("What is the first line of The Raven?")["id"] == rid
    assert mem.mentioned("Is a raven smarter than a crow?") is None
    assert mem.mentioned("Tell me about Lenore") is None


def test_asks_for_the_words():
    for wanted in ("Recite The Raven", "Please recite the first three stanzas of The Raven",
                   "What is the first line of The Raven?", "give me the last 2 lines of Sonnet 18",
                   "the full text of the Gettysburg Address"):
        assert asks_for_the_words(wanted), wanted
    for about in ("What does the bird say in The Raven?", "Who is Lenore in The Raven?",
                  "What does the first stanza of The Raven mean?",
                  "Use your memory palace to recall The Raven and tell me what it's about."):
        assert not asks_for_the_words(about), about


@pytest.mark.parametrize("asked,name", [
    ('Splendid! Can you please recite to me the poem "Jabberwocky"?', "Jabberwocky"),
    ("Thanks. Now read me Invictus.", "Invictus"),
    ("Mr. Darcy said hello. Recite Ozymandias.", "Ozymandias"),
    ('Recite "Do Not Go Gentle Into That Good Night" by Dylan Thomas.', "Do Not Go Gentle Into That Good Night by Dylan Thomas"),
    ("Say, can you recite The Road Not Taken?", "The Road Not Taken"),
])
def test_what_comes_before_the_request_is_not_the_title(asked, name):
    assert work_name(asked) == name
