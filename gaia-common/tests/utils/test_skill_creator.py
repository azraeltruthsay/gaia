"""Tests for the skill draft generator (GAIA_Project-5qy Phase 1)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gaia_common.utils.skill_creator import (
    DEFAULT_AUTO_SKILLS_DIR,
    draft_skill_from_failures,
    maybe_draft_skills,
    slugify_pattern,
)
from gaia_common.utils.skill_failures import record_failure


@pytest.fixture
def auto_dir(tmp_path: Path) -> Path:
    return tmp_path / "skills" / "auto"


@pytest.fixture
def failure_log(tmp_path: Path) -> Path:
    return tmp_path / "failure_log.jsonl"


def _failures(count: int, *, intent="research", query_prefix="x") -> list[dict]:
    """Build fake failure dicts for the generator."""
    return [
        {
            "t": f"2026-05-27T12:0{i}:00+00:00",
            "intent": intent,
            "query": f"{query_prefix} {i}",
            "pattern": f"{intent}:{query_prefix}",
        }
        for i in range(count)
    ]


# ── slugify_pattern ─────────────────────────────────────────────────


class TestSlugifyPattern:
    def test_basic(self):
        assert slugify_pattern("research:latest python") == "research_latest_python"

    def test_punctuation_collapsed(self):
        assert slugify_pattern("x:foo bar! baz?") == "x_foo_bar_baz"

    def test_empty_pattern_returns_unnamed(self):
        assert slugify_pattern("") == "unnamed"

    def test_capped_at_64(self):
        long = "a" * 200
        assert len(slugify_pattern(long)) == 64


# ── draft_skill_from_failures ───────────────────────────────────────


class TestDraftSkillFromFailures:
    def test_writes_skill_md(self, auto_dir):
        result = draft_skill_from_failures(
            "research:latest python",
            _failures(3),
            output_dir=auto_dir,
        )
        assert result["ok"] is True
        path = Path(result["path"])
        assert path.exists()
        assert path.name == "SKILL.md"
        # Skill dir has the auto_ prefix
        assert path.parent.name == "auto_research_latest_python"

    def test_frontmatter_complete(self, auto_dir):
        result = draft_skill_from_failures(
            "research:latest python",
            _failures(3),
            output_dir=auto_dir,
        )
        content = Path(result["path"]).read_text()
        assert content.startswith("---\n")
        # Required frontmatter fields
        for field in ("name:", "description:", "version:", "execution_mode:",
                      "domain:", "status:", "generated_at:",
                      "trigger_pattern:", "failure_count:"):
            assert field in content, f"missing frontmatter field {field!r}"

    def test_body_contains_failure_samples(self, auto_dir):
        failures = _failures(3, query_prefix="python release info")
        result = draft_skill_from_failures(
            "research:python release",
            failures, output_dir=auto_dir,
        )
        body = Path(result["path"]).read_text()
        # Each query should appear
        for f in failures:
            assert f["query"] in body

    def test_body_has_todo_marker(self, auto_dir):
        result = draft_skill_from_failures(
            "x:y", _failures(3), output_dir=auto_dir,
        )
        body = Path(result["path"]).read_text()
        assert "TODO" in body
        # Stage 1 deterministic body explicitly says playbook not written
        assert "playbook body not yet written" in body.lower()

    def test_status_draft_in_frontmatter(self, auto_dir):
        result = draft_skill_from_failures(
            "x:y", _failures(3), output_dir=auto_dir,
        )
        content = Path(result["path"]).read_text()
        assert "status: draft" in content

    def test_does_not_overwrite_existing(self, auto_dir):
        result1 = draft_skill_from_failures(
            "x:y", _failures(3), output_dir=auto_dir,
        )
        result2 = draft_skill_from_failures(
            "x:y", _failures(3), output_dir=auto_dir,
        )
        assert result1["ok"] is True
        assert result2["ok"] is False
        assert result2["skipped"] is True
        assert result2["skip_reason"] == "exists"

    def test_overwrite_flag(self, auto_dir):
        draft_skill_from_failures(
            "x:y", _failures(3), output_dir=auto_dir,
        )
        result2 = draft_skill_from_failures(
            "x:y", _failures(5), output_dir=auto_dir,
            overwrite=True,
        )
        assert result2["ok"] is True
        # Failure count should now reflect the second batch
        content = Path(result2["path"]).read_text()
        assert "failure_count: 5" in content

    def test_custom_body_generator_used(self, auto_dir):
        captured: list = []

        def gen(pattern, failures):
            captured.append((pattern, len(failures)))
            return "CUSTOM-BODY-MARKER"

        result = draft_skill_from_failures(
            "x:y", _failures(3),
            output_dir=auto_dir,
            body_generator=gen,
        )
        assert captured == [("x:y", 3)]
        content = Path(result["path"]).read_text()
        assert "CUSTOM-BODY-MARKER" in content

    def test_body_generator_failure_falls_back_to_default(self, auto_dir):
        def bad_gen(pattern, failures):
            raise RuntimeError("LLM down")

        result = draft_skill_from_failures(
            "x:y", _failures(3),
            output_dir=auto_dir,
            body_generator=bad_gen,
        )
        assert result["ok"] is True
        # Default body marker
        body = Path(result["path"]).read_text()
        assert "TODO" in body

    def test_write_error_returns_failure(self, tmp_path):
        # Use a path under a file (not directory) so mkdir fails
        blocker = tmp_path / "block"
        blocker.write_text("not a dir")
        # output_dir under the file → mkdir will fail
        result = draft_skill_from_failures(
            "x:y", _failures(3),
            output_dir=blocker / "nested",
        )
        assert result["ok"] is False
        assert result["skip_reason"] == "write_error"


# ── maybe_draft_skills (end-to-end) ─────────────────────────────────


class TestMaybeDraftSkills:
    def test_no_failures_no_drafts(self, auto_dir, failure_log):
        results = maybe_draft_skills(
            log_path=str(failure_log),
            output_dir=auto_dir,
        )
        assert results == []

    def test_below_threshold_no_drafts(self, auto_dir, failure_log):
        # Only 2 failures of the same pattern, threshold is 3
        for _ in range(2):
            record_failure(
                intent="x", query="some query", log_path=str(failure_log),
            )
        results = maybe_draft_skills(
            log_path=str(failure_log), output_dir=auto_dir,
        )
        assert results == []

    def test_drafts_when_threshold_hit(self, auto_dir, failure_log):
        for _ in range(3):
            record_failure(
                intent="research", query="latest python release",
                log_path=str(failure_log),
            )
        results = maybe_draft_skills(
            log_path=str(failure_log),
            output_dir=auto_dir,
        )
        assert len(results) == 1
        assert results[0]["ok"] is True
        # Skill file exists
        assert Path(results[0]["path"]).exists()

    def test_clears_pattern_after_draft(self, auto_dir, failure_log):
        for _ in range(3):
            record_failure(
                intent="x", query="alpha test", log_path=str(failure_log),
            )
        results = maybe_draft_skills(
            log_path=str(failure_log),
            output_dir=auto_dir,
            clear_after_draft=True,
        )
        assert results[0].get("pattern_entries_cleared") == 3
        # Log should now have no entries for that pattern
        from gaia_common.utils.skill_failures import (
            count_failures_for_pattern, derive_pattern,
        )
        pat = derive_pattern("x", "alpha test")
        assert count_failures_for_pattern(pat, log_path=str(failure_log)) == 0

    def test_keeps_pattern_if_clear_disabled(self, auto_dir, failure_log):
        for _ in range(3):
            record_failure(
                intent="x", query="alpha test", log_path=str(failure_log),
            )
        results = maybe_draft_skills(
            log_path=str(failure_log),
            output_dir=auto_dir,
            clear_after_draft=False,
        )
        assert "pattern_entries_cleared" not in results[0]

    def test_idempotent_second_run(self, auto_dir, failure_log):
        """After the first run drafts + clears, a second run does nothing
        — the pattern's failures are gone."""
        for _ in range(3):
            record_failure(
                intent="x", query="alpha test", log_path=str(failure_log),
            )
        r1 = maybe_draft_skills(log_path=str(failure_log), output_dir=auto_dir)
        r2 = maybe_draft_skills(log_path=str(failure_log), output_dir=auto_dir)
        assert r1
        assert r2 == []
