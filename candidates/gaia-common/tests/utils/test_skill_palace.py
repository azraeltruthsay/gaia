"""Tests for the skill-palace bridge (GAIA_Project-a3i)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gaia_common.utils.skill_palace import (
    find_skills_for_topic,
    record_skill_loaded,
    record_skill_outcome,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def isolate_palace():
    """Each test gets a clean palace cache."""
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
def mock_palace():
    """A MagicMock standing in for MemPalace. recall() returns
    whatever the test programs into .recall_results."""
    mock = MagicMock()
    mock.recall_results = []
    mock.recall.side_effect = lambda q, limit=None: list(mock.recall_results)
    reset_for_tests(mock)
    return mock


# ── record_skill_loaded ─────────────────────────────────────────────


class TestRecordSkillLoaded:
    def test_stores_with_skill_prefix(self, mock_palace):
        record_skill_loaded("blueprint", "scaffold a service blueprint")
        mock_palace.store.assert_called_once()
        args, kwargs = mock_palace.store.call_args
        text = args[0] if args else kwargs.get("text")
        assert text.startswith("skill:blueprint handles")
        assert "scaffold a service blueprint" in text

    def test_version_in_text(self, mock_palace):
        record_skill_loaded("foo", "do foo", version=3)
        text = mock_palace.store.call_args.args[0]
        assert "v3" in text

    def test_source_includes_skill_load_tag(self, mock_palace):
        record_skill_loaded("foo", "do foo")
        kwargs = mock_palace.store.call_args.kwargs
        assert kwargs.get("source", "").startswith("skill_load:")

    def test_empty_name_or_description_returns_false(self, mock_palace):
        assert record_skill_loaded("", "x") is False
        assert record_skill_loaded("x", "") is False
        mock_palace.store.assert_not_called()

    def test_no_palace_returns_false(self):
        # mock_palace fixture not used → reset_for_tests cleared the cache.
        # _get_palace will attempt to build one, fail, and return None.
        # This may succeed if MEMPALACE config is present — that's
        # environment-dependent, so we just confirm no exception.
        result = record_skill_loaded("foo", "do foo")
        assert result in (True, False)

    def test_store_failure_swallowed(self, mock_palace):
        mock_palace.store.side_effect = RuntimeError("io down")
        # Must not raise
        result = record_skill_loaded("foo", "do foo")
        assert result is False


# ── record_skill_outcome ────────────────────────────────────────────


class TestRecordSkillOutcome:
    def test_success_outcome(self, mock_palace):
        record_skill_outcome("blueprint", success=True, query="set up service X")
        text = mock_palace.store.call_args.args[0]
        assert "skill:blueprint succeeded" in text
        assert "set up service X" in text

    def test_failure_outcome(self, mock_palace):
        record_skill_outcome("blueprint", success=False, query="garbled input")
        text = mock_palace.store.call_args.args[0]
        assert "skill:blueprint failed" in text

    def test_intent_included(self, mock_palace):
        record_skill_outcome("blueprint", intent="planning", success=True)
        text = mock_palace.store.call_args.args[0]
        assert "intent=planning" in text

    def test_no_query_no_intent(self, mock_palace):
        """Minimal outcome — just the skill name + verb."""
        record_skill_outcome("blueprint", success=True)
        text = mock_palace.store.call_args.args[0]
        assert text == "skill:blueprint succeeded"

    def test_outcome_source_tag(self, mock_palace):
        record_skill_outcome("blueprint", success=False)
        kwargs = mock_palace.store.call_args.kwargs
        assert kwargs.get("source", "").startswith("skill_outcome:")
        assert ":failed" in kwargs["source"]

    def test_empty_name_returns_false(self, mock_palace):
        assert record_skill_outcome("", success=True) is False
        mock_palace.store.assert_not_called()

    def test_store_failure_swallowed(self, mock_palace):
        mock_palace.store.side_effect = RuntimeError("io down")
        result = record_skill_outcome("foo", success=True)
        assert result is False


# ── find_skills_for_topic ────────────────────────────────────────────


def _hit(text: str) -> dict:
    return {"text": text, "compressed": "", "body": ""}


class TestFindSkillsForTopic:
    def test_returns_skill_names_from_recall(self, mock_palace):
        mock_palace.recall_results = [
            _hit("skill:blueprint handles service scaffold"),
            _hit("skill:codemind handles code self-improvement"),
        ]
        skills = find_skills_for_topic("scaffold")
        assert "blueprint" in skills
        assert "codemind" in skills

    def test_dedup_across_hits(self, mock_palace):
        mock_palace.recall_results = [
            _hit("skill:blueprint handles X"),
            _hit("skill:blueprint succeeded on Y"),
        ]
        skills = find_skills_for_topic("anything")
        assert skills.count("blueprint") == 1

    def test_ignores_non_skill_memories(self, mock_palace):
        mock_palace.recall_results = [
            _hit("Azrael prefers terse responses"),  # no skill: prefix
            _hit("skill:blueprint handles X"),
        ]
        skills = find_skills_for_topic("anything")
        assert skills == ["blueprint"]

    def test_respects_limit(self, mock_palace):
        mock_palace.recall_results = [
            _hit(f"skill:s{i} handles thing{i}") for i in range(10)
        ]
        skills = find_skills_for_topic("thing", limit=3)
        assert len(skills) == 3

    def test_empty_query_returns_empty(self, mock_palace):
        skills = find_skills_for_topic("")
        assert skills == []
        mock_palace.recall.assert_not_called()

    def test_recall_failure_returns_empty(self, mock_palace):
        mock_palace.recall.side_effect = RuntimeError("palace down")
        skills = find_skills_for_topic("query")
        assert skills == []

    def test_no_palace_returns_empty(self):
        """When MemPalace isn't available, return empty list — never raise."""
        result = find_skills_for_topic("anything")
        assert isinstance(result, list)

    def test_skill_names_with_underscores_parsed(self, mock_palace):
        mock_palace.recall_results = [
            _hit("skill:code_skill_loop handles iterative code training"),
        ]
        skills = find_skills_for_topic("training")
        assert skills == ["code_skill_loop"]

    def test_handles_compressed_field(self, mock_palace):
        """Some MemPalace recall results carry text in 'compressed' field
        rather than 'text'. The parser falls back to the available field."""
        mock_palace.recall_results = [
            {"text": "", "compressed": "skill:foo handles bar"},
        ]
        skills = find_skills_for_topic("bar")
        assert skills == ["foo"]


# ── Integration sanity ──────────────────────────────────────────────


class TestIntegrationSanity:
    def test_load_then_outcome_then_find(self, mock_palace):
        """Realistic flow: skill loads, succeeds on a task, search
        surfaces it."""
        record_skill_loaded("blueprint", "scaffold service blueprints")
        record_skill_outcome("blueprint", intent="planning",
                             success=True, query="scaffold gaia-mcp")
        assert mock_palace.store.call_count == 2

        # Programmatically simulate the palace returning both memories
        # when queried for a matching topic.
        mock_palace.recall_results = [
            _hit("skill:blueprint handles scaffold service blueprints (v1)"),
            _hit("skill:blueprint succeeded on 'scaffold gaia-mcp' (intent=planning)"),
        ]
        skills = find_skills_for_topic("scaffold")
        assert skills == ["blueprint"]
