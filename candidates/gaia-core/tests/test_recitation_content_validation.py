"""
Unit tests for AgentCore._validate_recitation_content / _is_instructional_url
(GAIA_Project-9n8z).

Before this fix, validation only checked length + "does >=1 request keyword
appear in the page" — a lesson-plan page mentioning "haiku" repeatedly passed
that check just as easily as an actual haiku would have. These tests pin the
new behavior: reject pages that are ABOUT a work (lesson plans, study guides)
even when domain trust and keyword overlap would otherwise pass them.
"""
from unittest.mock import MagicMock, patch

from gaia_core.cognition.agent_core import AgentCore


def _make_agent_core():
    ai_manager = MagicMock()
    ai_manager.config = MagicMock()
    ai_manager.config.constants = {}
    ai_manager.config.SHARED_DIR = "/tmp/test_shared"
    ai_manager.config.config.SHARED_DIR = "/tmp/test_shared"
    ai_manager.model_pool = MagicMock()
    ai_manager.session_manager = MagicMock()
    return AgentCore(ai_manager)


# The actual poets.org page that triggered GAIA_Project-9n8z: a lesson plan
# about teaching a Sonia Sanchez haiku, not the haiku itself.
_LESSON_PLAN_CONTENT = """
Teach This Poem: "Haiku [for you]" by Sonia Sanchez | Academy of American Poets

Teach This Poem, though developed with a classroom in mind, can be easily
adapted for remote learning, hybrid learning models, or in-person classes.
Warm-up: Write out your full name. Count the syllables in your name.
Small Group Discussion: Share what you noticed about the poem with a small
group. Whole Class Discussion: Read the definition of haiku.
Extension for Grades 9-12: Over the next week, write one haiku a day.
""" * 3  # pad past the 200-char floor with realistic repetition

_ACTUAL_POEM_CONTENT = """
The Raven
by Edgar Allan Poe

Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.
"""


class TestValidateRecitationContent:
    def test_rejects_lesson_plan_content(self):
        ac = _make_agent_core()
        assert ac._validate_recitation_content(_LESSON_PLAN_CONTENT, "recite a haiku") is False

    def test_accepts_actual_poem_content(self):
        ac = _make_agent_core()
        assert ac._validate_recitation_content(_ACTUAL_POEM_CONTENT, "recite The Raven") is True

    def test_rejects_too_short_content(self):
        ac = _make_agent_core()
        assert ac._validate_recitation_content("too short", "recite a poem") is False

    def test_rejects_empty_content(self):
        ac = _make_agent_core()
        assert ac._validate_recitation_content("", "recite a poem") is False


class TestIsInstructionalUrl:
    def test_flags_lesson_plan_path(self):
        assert AgentCore._is_instructional_url(
            "https://poets.org/lesson-plan/teach-poem-haiku-you-sonia-sanchez"
        ) is True

    def test_flags_study_guide_path(self):
        assert AgentCore._is_instructional_url(
            "https://example.com/study-guide/the-raven"
        ) is True

    def test_flags_pages_that_are_never_the_work(self):
        for url in ("https://poets.org/anthology/funerals-and-death", "https://www.youtube.com/watch?v=sOSCuf0gli0"):
            assert AgentCore._is_instructional_url(url) is True

    def test_allows_wikisource(self):
        assert AgentCore._is_instructional_url("https://en.wikisource.org/wiki/The_Raven_(Poe)") is False

    def test_allows_plain_poem_path(self):
        assert AgentCore._is_instructional_url(
            "https://www.poetryfoundation.org/poems/48860/the-raven"
        ) is False


class TestRecitationReply:
    """yx93: the part asked for, or she says she doesn't have it."""
    POEM = "One two three\nfour five six\n\nseven eight nine\nten eleven twelve"
    DOC = {"title": "Numbers", "content": POEM}

    def test_part_asked_for(self):
        reply = AgentCore._recitation_reply("Recite the first stanza of Numbers", self.DOC)
        assert "One two three" in reply and "seven" not in reply

    def test_whole_work(self):
        assert self.POEM in AgentCore._recitation_reply("Recite Numbers", self.DOC)

    def test_part_that_cannot_be_cut_gets_the_whole_work(self):
        assert self.POEM in AgentCore._recitation_reply("Recite the first nine stanzas of Numbers", self.DOC)

    def test_no_source_says_so(self):
        reply = AgentCore._recitation_reply("Please recite the first stanza of The Raven by Edgar Allan Poe.", None)
        assert "The Raven by Edgar Allan Poe" in reply and "memory palace" in reply
        assert "first stanza" not in reply


def _web_mock(pages):
    """mcp_client stand-in: palace empty, search finds `pages`, palace saves."""
    stored = {}

    def call(method, params=None):
        if method == "palace_recall_work":
            return {"ok": True, "response": {"result": {"ok": True, "found": False}}}
        if method == "web_search":
            return {"ok": True, "response": {"result": {"results": [
                {"url": u, "title": p["title"], "trust_tier": "trusted"} for u, p in pages.items()]}}}
        if method == "web_fetch":
            page = pages.get(params["url"])
            if not page:
                return {"ok": False, "error": "gone"}
            return {"ok": True, "response": {"result": {"ok": True, **page}}}
        if method == "palace_store_work":
            stored.update(params)
            return {"ok": True}
        raise AssertionError(method)
    return call, stored


class TestWebRetrieve:
    REQUEST = "Recite the first stanza of The Raven by Edgar Allan Poe"
    BODY = ("Once upon a midnight dreary, while I pondered, weak and weary,\n"
            "Over many a quaint and curious volume of forgotten lore")

    def _page(self, domain, extra):
        return {"title": "The Raven by Edgar Allan Poe", "domain": domain,
                "content": f"{extra}\n{self.BODY}\nThe end of {domain}"}

    def test_two_sites_agree_and_it_is_remembered(self):
        call, stored = _web_mock({"https://a.org/x": self._page("a.org", "Menu"),
                                  "https://b.org/y": self._page("b.org", "Home")})
        with patch("gaia_core.cognition.agent_core.mcp_client.call_jsonrpc", side_effect=call):
            doc = _make_agent_core()._web_retrieve_for_recitation(self.REQUEST, "s")
        assert doc["content"] == self.BODY
        assert stored["name"] == "The Raven by Edgar Allan Poe" and stored["text"] == self.BODY
        assert len(stored["sources"]) == 2

    def test_one_site_is_not_enough(self):
        call, stored = _web_mock({"https://a.org/x": self._page("a.org", "Menu")})
        with patch("gaia_core.cognition.agent_core.mcp_client.call_jsonrpc", side_effect=call):
            assert _make_agent_core()._web_retrieve_for_recitation(self.REQUEST, "s") is None
        assert not stored

    def test_memory_comes_before_the_web(self):
        calls = []

        def call(method, params=None):
            calls.append(method)
            return {"ok": True, "response": {"result": {"found": True, "name": "The Raven", "text": self.BODY}}}
        with patch("gaia_core.cognition.agent_core.mcp_client.call_jsonrpc", side_effect=call):
            doc = _make_agent_core()._web_retrieve_for_recitation(self.REQUEST, "s")
        assert doc == {"title": "The Raven", "content": self.BODY}
        assert calls == ["palace_recall_work"]
