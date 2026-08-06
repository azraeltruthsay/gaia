"""
Unit tests for AgentCore._validate_recitation_content / _is_instructional_url
(GAIA_Project-9n8z).

Before this fix, validation only checked length + "does >=1 request keyword
appear in the page" — a lesson-plan page mentioning "haiku" repeatedly passed
that check just as easily as an actual haiku would have. These tests pin the
new behavior: reject pages that are ABOUT a work (lesson plans, study guides)
even when domain trust and keyword overlap would otherwise pass them.
"""
from unittest.mock import MagicMock

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

    def test_allows_plain_poem_path(self):
        assert AgentCore._is_instructional_url(
            "https://www.poetryfoundation.org/poems/48860/the-raven"
        ) is False
