"""Tests for the skill-failure accumulator (GAIA_Project-5qy Phase 1)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gaia_common.utils.skill_failures import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_HOURS,
    clear_pattern,
    count_failures_for_pattern,
    derive_pattern,
    find_patterns_above_threshold,
    recent_failures_for_pattern,
    record_failure,
)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "failure_log.jsonl"


# ── derive_pattern ──────────────────────────────────────────────────


class TestDerivePattern:
    def test_strips_stopwords(self):
        p = derive_pattern("research", "what is the latest python release")
        # "what", "is", "the" are stopwords — should be removed
        assert "what" not in p
        assert "the" not in p
        assert "latest" in p
        assert "python" in p
        assert p.startswith("research:")

    def test_lowercase_normalized(self):
        p1 = derive_pattern("Research", "Latest Python Release")
        p2 = derive_pattern("research", "latest python release")
        assert p1 == p2

    def test_punctuation_stripped(self):
        p = derive_pattern("chat", "Hey! What's that?")
        assert "?" not in p
        assert "!" not in p

    def test_empty_query_yields_intent_only(self):
        p = derive_pattern("planning", "")
        assert p == "planning:_"

    def test_empty_intent_falls_back_to_other(self):
        p = derive_pattern("", "some query")
        assert p.startswith("other:")

    def test_max_token_cap(self):
        # Default cap is 5 tokens
        p = derive_pattern("x", "one two three four five six seven eight")
        token_part = p.split(":", 1)[1]
        assert len(token_part.split()) <= 5


# ── record_failure ──────────────────────────────────────────────────


class TestRecordFailure:
    def test_appends_to_log(self, log_path):
        entry = record_failure(
            intent="research", query="latest python",
            skill_name="web", source="duckduckgo",
            log_path=str(log_path),
        )
        assert entry is not None
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["intent"] == "research"
        assert rec["query"] == "latest python"
        assert rec["skill_name"] == "web"
        assert rec["pattern"]

    def test_multiple_appends(self, log_path):
        for i in range(3):
            record_failure(
                intent="x", query=f"q{i}", log_path=str(log_path),
            )
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_empty_intent_and_query_skipped(self, log_path):
        entry = record_failure(intent="", query="", log_path=str(log_path))
        assert entry is None
        assert not log_path.exists()

    def test_creates_parent_dir(self, tmp_path):
        deep = tmp_path / "nested" / "deeper" / "log.jsonl"
        entry = record_failure(
            intent="x", query="y", log_path=str(deep),
        )
        assert entry is not None
        assert deep.exists()

    def test_bad_path_returns_none_no_raise(self):
        # Writing to /proc which is read-only
        result = record_failure(
            intent="x", query="y", log_path="/proc/should-fail/log.jsonl",
        )
        assert result is None


# ── count_failures_for_pattern ──────────────────────────────────────


class TestCountFailures:
    def test_counts_matches(self, log_path):
        for _ in range(3):
            record_failure(
                intent="research", query="latest python release",
                log_path=str(log_path),
            )
        pattern = derive_pattern("research", "latest python release")
        n = count_failures_for_pattern(pattern, log_path=str(log_path))
        assert n == 3

    def test_no_log_returns_zero(self, tmp_path):
        n = count_failures_for_pattern("foo", log_path=str(tmp_path / "missing.jsonl"))
        assert n == 0

    def test_empty_pattern_returns_zero(self, log_path):
        record_failure(intent="x", query="y", log_path=str(log_path))
        assert count_failures_for_pattern("", log_path=str(log_path)) == 0

    def test_filters_by_window(self, log_path):
        # Manually inject an old entry
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=200)).isoformat()
        with open(log_path, "w") as f:
            f.write(json.dumps({
                "t": old, "intent": "x", "query": "y",
                "pattern": derive_pattern("x", "y"),
            }) + "\n")
        record_failure(
            intent="x", query="y", log_path=str(log_path),
        )
        # 7-day window excludes the old entry
        n = count_failures_for_pattern(
            derive_pattern("x", "y"),
            window_hours=168, log_path=str(log_path),
        )
        assert n == 1


# ── find_patterns_above_threshold ───────────────────────────────────


class TestFindPatternsAboveThreshold:
    def test_returns_empty_when_no_log(self, tmp_path):
        result = find_patterns_above_threshold(
            log_path=str(tmp_path / "missing.jsonl"),
        )
        assert result == []

    def test_filters_by_threshold(self, log_path):
        # 2 instances of pattern A, 4 of pattern B
        for _ in range(2):
            record_failure(intent="a", query="alpha test", log_path=str(log_path))
        for _ in range(4):
            record_failure(intent="b", query="beta test", log_path=str(log_path))
        # Threshold 3 → only B
        result = find_patterns_above_threshold(threshold=3, log_path=str(log_path))
        patterns = {r["pattern"] for r in result}
        assert any("beta" in p for p in patterns)
        assert not any("alpha" in p for p in patterns)

    def test_includes_metadata(self, log_path):
        for _ in range(3):
            record_failure(
                intent="research", query="latest python release",
                log_path=str(log_path),
            )
        result = find_patterns_above_threshold(threshold=3, log_path=str(log_path))
        assert len(result) == 1
        entry = result[0]
        assert entry["count"] == 3
        assert entry["intent"] == "research"
        assert len(entry["recent_queries"]) == 3
        assert entry["oldest_t"]
        assert entry["newest_t"]

    def test_sorted_by_count_then_recency(self, log_path):
        # Pattern A: 3 failures
        for i in range(3):
            record_failure(intent="x", query="alpha", log_path=str(log_path))
        # Pattern B: 5 failures
        for i in range(5):
            record_failure(intent="y", query="beta", log_path=str(log_path))
        result = find_patterns_above_threshold(threshold=3, log_path=str(log_path))
        # Higher count first
        assert result[0]["count"] >= result[-1]["count"]


# ── recent_failures_for_pattern ─────────────────────────────────────


class TestRecentFailuresForPattern:
    def test_newest_first(self, log_path):
        for i in range(5):
            record_failure(
                intent="x", query="something", log_path=str(log_path),
            )
        pat = derive_pattern("x", "something")
        recent = recent_failures_for_pattern(pat, log_path=str(log_path))
        # Newest_t should be later than oldest_t in the returned list
        assert recent[0]["t"] >= recent[-1]["t"]

    def test_respects_limit(self, log_path):
        for _ in range(10):
            record_failure(intent="x", query="q", log_path=str(log_path))
        pat = derive_pattern("x", "q")
        recent = recent_failures_for_pattern(pat, limit=3, log_path=str(log_path))
        assert len(recent) == 3


# ── clear_pattern ───────────────────────────────────────────────────


class TestClearPattern:
    def test_removes_only_matching_entries(self, log_path):
        record_failure(intent="x", query="alpha", log_path=str(log_path))
        record_failure(intent="x", query="alpha", log_path=str(log_path))
        record_failure(intent="x", query="beta", log_path=str(log_path))
        pat = derive_pattern("x", "alpha")
        removed = clear_pattern(pat, log_path=str(log_path))
        assert removed == 2
        # Beta still there
        beta = derive_pattern("x", "beta")
        assert count_failures_for_pattern(beta, log_path=str(log_path)) == 1
        # Alpha gone
        assert count_failures_for_pattern(pat, log_path=str(log_path)) == 0

    def test_missing_log_returns_zero(self, tmp_path):
        assert clear_pattern("anything", log_path=str(tmp_path / "no.jsonl")) == 0

    def test_empty_pattern_returns_zero(self, log_path):
        record_failure(intent="x", query="y", log_path=str(log_path))
        assert clear_pattern("", log_path=str(log_path)) == 0
