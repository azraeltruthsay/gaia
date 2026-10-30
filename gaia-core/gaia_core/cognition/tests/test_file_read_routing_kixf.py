"""kixf: a request naming a file to read is routed deterministically. Only
"read /path" matched; "Read the file /path and tell me …" went to the LLM
selector and review, which declined it twice, and the turn fell back to a raw
preview that never answered the question."""

import pytest

from gaia_core.cognition.tool_selector import _deterministic_tool_match

P = "/knowledge/blueprints/COGNITIVE_ARCHITECTURE.md"


@pytest.mark.parametrize("text", [
    f"Read the file {P} and tell me its first heading",
    f"Can you read {P} and tell me how many regions it lists?",
    f"Please open {P}.",
    f"Look at the contents of {P} — what's it about?",
    f"read {P}",
])
def test_file_read_requests_match_deterministically(text):
    m = _deterministic_tool_match(text.lower(), text)
    assert m is not None and m.tool_name == "file"
    assert m.params == {"action": "read", "path": P}  # case kept, no trailing period
    assert m.selection_confidence >= 0.9  # skips the LLM review


@pytest.mark.parametrize("text", [
    "I read about /r/python on reddit and liked it",
    "What do you think about reading?",
    "Tell me about /knowledge in general",
])
def test_non_requests_do_not_read(text):
    m = _deterministic_tool_match(text.lower(), text)
    assert m is None or m.tool_name != "file"
