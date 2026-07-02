"""Regression tests for GAIA_Project-h34: the deterministic web-search
query extractor fell back to the ENTIRE raw user utterance (including the
speaker's own "you should search" meta-commentary) whenever the trigger
phrase was embedded in ordinary prose instead of an explicit "search for X"
pattern. Pins the sentence-dropping fallback that replaced it.
"""

from gaia_core.cognition.tool_selector import _deterministic_tool_match


def _query_for(user_input: str) -> str:
    result = _deterministic_tool_match(user_input.lower(), user_input)
    assert result is not None
    assert result.tool_name == "web"
    assert result.params["action"] == "search"
    return result.params["query"]


def test_trigger_sentence_dropped_from_conversational_prose():
    # The exact GAIA_Project-h34 repro: a multi-sentence explanation ending
    # with a meta-instruction to search. The query must not be the whole
    # utterance, and must not contain the speaker's own instruction to search.
    text = (
        "So, it's called a Spare Auto Encoder. Anthropic designed it to "
        "allow more fine detailed reading and influence of model inference "
        "in process. It would be probably best to do a web search to learn "
        "more."
    )
    query = _query_for(text)
    assert "web search" not in query
    assert "it would be probably best" not in query
    assert "spare auto encoder" in query


def test_explicit_search_for_pattern_still_works():
    query = _query_for("Please search the web for Jabberwocky poem")
    assert "jabberwocky" in query
    assert "search" not in query


def test_subject_before_tool_phrase_still_works():
    query = _query_for(
        "Recite Jabberwocky for me? If you're unsure, feel free to use the web search."
    )
    assert "jabberwocky" in query


def test_bare_search_request_falls_back_to_full_text():
    # Nothing but the trigger phrase itself — there's genuinely no other
    # subject matter, so the full (short) text is the only reasonable query.
    query = _query_for("please do a web search")
    assert query
