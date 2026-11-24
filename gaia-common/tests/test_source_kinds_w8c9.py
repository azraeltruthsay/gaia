"""w8c9: she's shown what kind of site a source is, so an AI-written answer
page isn't taken for research."""
from gaia_common.utils.source_kinds import AI_ANSWER, site_kind
from gaia_common.utils.tool_call_parser import tool_result_text

# The search she answered from on 2026-11-23 ("civilizations earlier than the Sumerians?").
SUMER = {"ok": True, "query": "civilizations before Sumerians", "results": [
    {"title": "Cradle of civilization - Wikipedia", "url": "https://en.wikipedia.org/wiki/Cradle_of_civilization",
     "snippet": "A cradle of civilization is a location...", "domain": "en.wikipedia.org", "trust_tier": "trusted"},
    {"title": "10 Ancient Civilizations That Existed Before Recorded History",
     "url": "https://www.thearchaeologist.org/blog/10-ancient-civilizations-that-existed-before-recorded-history",
     "snippet": "The Ubaid Culture (c. 6500 BCE)...", "domain": "www.thearchaeologist.org", "trust_tier": "unknown"},
    {"title": "6 Early Human Civilizations - HISTORY", "url": "https://www.history.com/articles/first-earliest-human-civilizations",
     "snippet": "Here's a look at six of the earliest civilizations", "domain": "www.history.com", "trust_tier": "unknown"},
    {"title": "Civilizations Before Sumer | Ithy", "url": "https://ithy.com/article/civilizations-before-sumer-63fi3vq1",
     "snippet": "archaeological and anthropological research shows...", "domain": "ithy.com", "trust_tier": "unknown"},
]}


def test_kinds():
    assert site_kind("https://ithy.com/article/x") == AI_ANSWER
    assert site_kind("https://www.perplexity.ai/page/x") == AI_ANSWER
    assert site_kind("https://en.wikipedia.org/wiki/Sumer") == "encyclopedia"
    assert site_kind("plato.stanford.edu") == "encyclopedia"
    assert site_kind("https://www.stanford.edu/x") == "university site"
    assert site_kind("https://www.nasa.gov/x") == "government site"
    assert site_kind("https://www.ox.ac.uk/x") == "university site"
    assert site_kind("https://www.reuters.com/world/x") == "news site"
    assert site_kind("https://someone.substack.com/p/x") == "blog"
    assert site_kind("https://www.thearchaeologist.org/blog/x") == "blog post"
    assert site_kind("https://www.history.com/articles/x") is None   # not known: no guess
    assert site_kind("") is None
    # Seen live: Cambridge's dictionary isn't a journal.
    assert site_kind("https://dictionary.cambridge.org/dictionary/english/earliest") == "dictionary"
    assert site_kind("https://en.wiktionary.org/wiki/x") == "dictionary"


def test_search_lines_say_the_kind():
    text = tool_result_text(SUMER)
    ithy = next(line for line in text.splitlines() if "Ithy" in line)
    assert "AI-written answer page, not research" in ithy
    assert "encyclopedia" in next(line for line in text.splitlines() if "Wikipedia" in line)
    assert "blog post" in next(line for line in text.splitlines() if "thearchaeologist" in line)
    history = next(line for line in text.splitlines() if "history.com" in line)
    assert "(6 Early Human Civilizations - HISTORY, https://" in history


def test_a_fetched_page_says_the_kind():
    text = tool_result_text({"ok": True, "title": "Civilizations Before Sumer",
                             "url": "https://ithy.com/article/x", "content": "Some text."})
    assert text.splitlines()[:3] == ["Civilizations Before Sumer", "https://ithy.com/article/x",
                                     f"({AI_ANSWER})"]


def test_rows_without_a_web_address_are_unchanged():
    text = tool_result_text({"results": [{"fact": "Sumer was in Mesopotamia", "source": "my notes", "url": "palace"}]})
    assert text == "- Sumer was in Mesopotamia (my notes, palace)"
