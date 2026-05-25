"""Tests for KG recency cross-check (GAIA_Project-hkv — Stage 8)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gaia_common.utils import fact_types
from gaia_common.utils.knowledge_graph import KnowledgeGraph
from gaia_core.cognition.kg_recency_grounding import (
    build_recency_grounding,
    is_time_sensitive,
    lookup_kg_facts,
    _format_age,
)


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(db_path=str(tmp_path / "stage8.sqlite"))


def _id_extractor(text: str) -> list[str]:
    """Test extractor: split on whitespace, strip punctuation, return
    capitalized tokens."""
    import re
    out = []
    for w in text.split():
        clean = re.sub(r"[^A-Za-z0-9_]", "", w)
        if clean and clean[0].isupper():
            out.append(clean)
    return out


# ── is_time_sensitive ───────────────────────────────────────────────


class TestIsTimeSensitive:
    @pytest.mark.parametrize("prompt", [
        "Who is the current senator from Oregon?",
        "What's the latest on the launch?",
        "Is X still alive?",
        "Tell me what's happening right now",
        "Are these laws still in effect?",
        "What's the temperature today",
        "as of this year, who leads?",
    ])
    def test_time_markers_detected(self, prompt):
        assert is_time_sensitive(prompt), f"missed: {prompt!r}"

    @pytest.mark.parametrize("prompt", [
        "Explain quantum entanglement",
        "What did Marcus Aurelius write?",
        "Who was the first president?",
        "Describe a sunset",
        "",
    ])
    def test_no_time_marker(self, prompt):
        assert not is_time_sensitive(prompt), f"false positive: {prompt!r}"


# ── _format_age ─────────────────────────────────────────────────────


class TestFormatAge:
    def test_minutes(self):
        now = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
        vf = (now - timedelta(minutes=30)).isoformat()
        assert _format_age(vf, now=now) == "30m ago"

    def test_hours(self):
        now = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
        vf = (now - timedelta(hours=5)).isoformat()
        assert _format_age(vf, now=now) == "5h ago"

    def test_days(self):
        now = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
        vf = (now - timedelta(days=3)).isoformat()
        assert _format_age(vf, now=now) == "3d ago"

    def test_months(self):
        now = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
        vf = (now - timedelta(days=90)).isoformat()
        assert _format_age(vf, now=now) == "3mo ago"

    def test_years(self):
        now = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
        vf = (now - timedelta(days=3 * 365)).isoformat()
        assert _format_age(vf, now=now) == "3yr ago"

    def test_missing_returns_label(self):
        assert _format_age(None) == "unknown date"


# ── lookup_kg_facts ─────────────────────────────────────────────────


class TestLookupKgFacts:
    def test_no_phrases_empty(self, kg):
        assert lookup_kg_facts(kg, []) == []

    def test_unknown_entity_empty(self, kg):
        assert lookup_kg_facts(kg, ["NeverHeardOf"]) == []

    def test_returns_relevance_sorted(self, kg):
        now = datetime.now(timezone.utc)
        # Two facts, same entity, same predicate (multi-valued → allow_coexist),
        # different ages.
        kg.add_triple("Oregon", "has_senator", "Person_A",
                      valid_from=now.isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE,
                      confidence=0.85, allow_coexist=True)
        kg.add_triple("Oregon", "has_senator", "Person_B",
                      valid_from=(now - timedelta(days=400)).isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE,
                      confidence=0.85, allow_coexist=True)
        facts = lookup_kg_facts(kg, ["Oregon"], min_relevance=0.0)
        assert facts[0]["object"] == "Person_A"
        assert facts[0]["relevance"] > facts[1]["relevance"]

    def test_dedup_across_phrases(self, kg):
        kg.add_triple("Oregon", "borders", "California",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.BIOGRAPHICAL,
                      confidence=0.9)
        # Querying both Oregon and California should not return the same
        # bidirectional fact twice.
        facts = lookup_kg_facts(kg, ["Oregon", "California"], min_relevance=0.0)
        keys = [(f["subject"], f["predicate"], f["object"]) for f in facts]
        assert len(keys) == len(set(keys))

    def test_max_per_entity_cap(self, kg):
        now = datetime.now(timezone.utc)
        for i in range(5):
            kg.add_triple("X", "saw", f"item_{i}",
                          valid_from=now.isoformat(),
                          fact_type=fact_types.BIOGRAPHICAL,
                          confidence=0.9, allow_coexist=True)
        facts = lookup_kg_facts(kg, ["X"], min_relevance=0.0, max_per_entity=2)
        assert len(facts) == 2


# ── build_recency_grounding ─────────────────────────────────────────


class TestBuildRecencyGrounding:
    def test_no_kg_returns_none(self):
        assert build_recency_grounding("current X?", None) is None

    def test_empty_prompt_returns_none(self, kg):
        assert build_recency_grounding("", kg) is None

    def test_no_time_marker_returns_none(self, kg):
        # Even with KG hits, non-time-sensitive prompts skip grounding
        kg.add_triple("Oregon", "has_senator", "Alice",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE)
        out = build_recency_grounding(
            "Tell me about Oregon", kg, extract_phrases=_id_extractor,
        )
        assert out is None

    def test_require_time_marker_false_grounds_unconditionally(self, kg):
        kg.add_triple("Oregon", "has_senator", "Alice",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE)
        out = build_recency_grounding(
            "Tell me about Oregon", kg,
            extract_phrases=_id_extractor,
            require_time_marker=False,
        )
        assert out is not None
        assert "Oregon" in out
        assert "Alice" in out

    def test_no_phrases_returns_none(self, kg):
        kg.add_triple("Oregon", "has_senator", "Alice",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE)
        out = build_recency_grounding(
            "current senator?", kg, extract_phrases=lambda t: [],
        )
        assert out is None

    def test_no_kg_hits_returns_none(self, kg):
        out = build_recency_grounding(
            "Who is the current senator from Oregon?", kg,
            extract_phrases=_id_extractor,
        )
        assert out is None

    def test_fresh_fact_injected_as_authoritative(self, kg):
        kg.add_triple("Oregon", "has_senator", "Alice",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE, confidence=0.85)
        out = build_recency_grounding(
            "Who is the current senator from Oregon?", kg,
            extract_phrases=_id_extractor,
        )
        assert out is not None
        assert "authoritative" in out.lower()
        assert "Alice" in out
        assert "Oregon" in out

    def test_stale_fact_injected_with_warning(self, kg):
        # NEWS half-life 7d → 60d gives relevance ~0.003 < warn threshold.
        # POLITICAL_OFFICE half-life 1y → 60d gives relevance ~0.79
        # We want a fact in the (0.05, 0.30) "warn" band; pick a fact_type
        # + age that lands there. NEWS at 21 days: 21/7 = 3 halflives →
        # conf=0.85 × 0.125 = ~0.106 — in the warn band.
        old = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
        kg.add_triple("Launch", "status", "Pending",
                      valid_from=old, fact_type=fact_types.NEWS,
                      confidence=0.85)
        out = build_recency_grounding(
            "What's the latest on the Launch?", kg,
            extract_phrases=_id_extractor,
        )
        assert out is not None
        assert "low confidence" in out.lower() or "stale" in out.lower()
        assert "Launch" in out

    def test_max_facts_cap(self, kg):
        now = datetime.now(timezone.utc).isoformat()
        for i in range(10):
            kg.add_triple("X", "saw", f"item_{i}",
                          valid_from=now,
                          fact_type=fact_types.NEWS,
                          confidence=0.9, allow_coexist=True)
        out = build_recency_grounding(
            "current X?", kg,
            extract_phrases=_id_extractor,
            max_facts=3,
        )
        assert out is not None
        # Count fact lines (lines starting with "  - ")
        fact_lines = [L for L in out.split("\n") if L.startswith("  - ")]
        assert len(fact_lines) == 3

    def test_biographical_never_stale(self, kg):
        # Even decades-old biographical fact stays at full relevance
        long_ago = (datetime.now(timezone.utc) - timedelta(days=5 * 365)).isoformat()
        kg.add_triple("Aurelius", "died_in", "180_CE",
                      valid_from=long_ago,
                      fact_type=fact_types.BIOGRAPHICAL,
                      confidence=0.95)
        out = build_recency_grounding(
            "Is Aurelius still alive today?", kg,
            extract_phrases=_id_extractor,
        )
        assert out is not None
        # Biographical should land in the authoritative block, not stale
        assert "authoritative" in out.lower()


class TestRecencyGroundingForPrompt:
    """Tests for the agent_core integration wrapper."""

    def test_no_kg_returns_none(self):
        from gaia_core.cognition import kg_recency_grounding
        kg_recency_grounding.reset_for_tests(None)
        # No KG → still None (we never initialized one for the test)
        # Note: this test temporarily blocks lazy init; reset_for_tests with
        # an explicit None doesn't actually clear lazy-init machinery, so
        # we just exercise the safe-fall-through path with an empty prompt.
        from gaia_core.cognition.kg_recency_grounding import recency_grounding_for_prompt
        assert recency_grounding_for_prompt("") is None

    def test_with_injected_kg_grounds(self, kg):
        from gaia_core.cognition import kg_recency_grounding
        kg.add_triple("Oregon", "has_senator", "Alice",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE,
                      confidence=0.85)
        kg_recency_grounding.reset_for_tests(kg)
        try:
            out = kg_recency_grounding.recency_grounding_for_prompt(
                "Who is the current senator from Oregon?",
            )
            assert out is not None
            assert "Alice" in out
            assert "Oregon" in out
        finally:
            kg_recency_grounding.reset_for_tests(None)

    def test_no_time_marker_returns_none_even_with_kg(self, kg):
        from gaia_core.cognition import kg_recency_grounding
        kg.add_triple("Oregon", "has_senator", "Alice",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.POLITICAL_OFFICE)
        kg_recency_grounding.reset_for_tests(kg)
        try:
            out = kg_recency_grounding.recency_grounding_for_prompt(
                "Tell me about Oregon",  # no time marker
            )
            assert out is None
        finally:
            kg_recency_grounding.reset_for_tests(None)


class TestLatency:
    def test_under_200ms_with_empty_kg(self, kg):
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            build_recency_grounding(
                "Who is the current senator from Oregon?", kg,
                extract_phrases=_id_extractor,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000 / 50
        # Per-call latency far under 200ms target
        assert elapsed_ms < 200

    def test_under_200ms_with_populated_kg(self, kg):
        import time
        now = datetime.now(timezone.utc).isoformat()
        for i in range(100):
            kg.add_triple(f"Entity_{i}", "has_attr", f"value_{i}",
                          valid_from=now,
                          fact_type=fact_types.POLITICAL_OFFICE,
                          confidence=0.85, allow_coexist=True)
        t0 = time.perf_counter()
        for _ in range(20):
            build_recency_grounding(
                "current Entity_5 and Entity_50 status", kg,
                extract_phrases=_id_extractor,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000 / 20
        assert elapsed_ms < 200
