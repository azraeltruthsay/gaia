"""i4vn: palace(store, topic=..., content=...) failed with "text parameter required"."""
from gaia_mcp.tools import _store_text


def test_store_takes_the_words_she_reaches_for():
    assert _store_text({"text": "a fact"}) == "a fact"
    assert _store_text({"topic": "Tamsin's pronouns", "content": "xe/xem"}) == "Tamsin's pronouns: xe/xem"
    assert _store_text({"topic": "pronouns", "content": "Tamsin uses xe/xem pronouns."}) == "Tamsin uses xe/xem pronouns."
    assert _store_text({"memory": "  kept  "}) == "kept"
    assert _store_text({"topic": "only a topic"}) == ""
    assert _store_text({}) == ""
