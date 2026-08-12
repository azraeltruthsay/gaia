"""Regression tests for GAIA_Project-ssgd — a live conversation asked "Have
you heard of the Ship of Theseus Paradox?" (a philosophy question) and GAIA
replied with a raw dump of today's news headlines.

Root cause: _deterministic_tool_match's news_re had a bare
"have you heard (about|of)" trigger — an ordinary conversational opener for
ANY topic, not a news signal. Topic extraction only looks for
"about|on|regarding" (not "of"), so it found nothing and silently fell back
to querying "today's news headlines" — completely unrelated to what was
asked. Fix: drop that bare trigger. Genuinely news-shaped phrasing (the
bc6e96d fix this sits on top of) must keep matching.
"""
from gaia_core.cognition.tool_selector import _deterministic_tool_match


def _match(user_input: str):
    return _deterministic_tool_match(user_input.lower(), user_input)


# ── The exact repro + variants: must NOT be treated as a news request ──────

def test_have_you_heard_of_philosophy_question_not_routed_to_news_search():
    text = "Have you heard of the Ship of Theseus Paradox?"
    result = _match(text)
    assert result is None or not (
        result.tool_name == "web" and result.params.get("query") == "today's news headlines"
    )


def test_have_you_heard_about_topic_not_routed_to_news_search():
    text = "Have you heard about the halting problem?"
    result = _match(text)
    assert result is None or not (
        result.tool_name == "web" and result.params.get("query") == "today's news headlines"
    )


def test_full_conversational_repro_not_routed_to_news_search():
    text = (
        "Have you heard of the Ship of Theseus Paradox? From what I "
        "understand, it's a thought experiment where we consider whether a "
        "ship is the same ship if all of its parts are replaced one by one. "
        "Or is that a new ship? That's the paradox. have you heard it?"
    )
    result = _match(text)
    assert result is None or not (
        result.tool_name == "web" and result.params.get("query") == "today's news headlines"
    )


# ── Regression guard: genuine news phrasing (bc6e96d) must still fire ──────

def test_do_you_know_any_recent_news_still_routes_to_web_search():
    result = _match("Do you know any recent news?")
    assert result is not None
    assert result.tool_name == "web"
    assert result.params["action"] == "search"


def test_whats_happening_today_still_routes_to_web_search():
    result = _match("What's happening today?")
    assert result is not None
    assert result.tool_name == "web"


def test_any_breaking_news_still_routes_to_web_search():
    result = _match("any breaking news?")
    assert result is not None
    assert result.tool_name == "web"


def test_news_about_topic_still_extracts_topic():
    result = _match("tell me the news about SpaceX")
    assert result is not None
    assert result.tool_name == "web"
    assert "spacex" in result.params["query"]
