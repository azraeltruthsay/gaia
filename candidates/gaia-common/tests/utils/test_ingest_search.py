"""Tests for the web-search-to-triple ingester + KG relevance query
(GAIA_Project-lw4 Stage 7)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gaia_common.utils import fact_types
from gaia_common.utils.knowledge_graph import KnowledgeGraph
from gaia_common.utils.ingest_search import (
    confidence_for_tier,
    ingest_search_response,
    lookup_cached_results,
)


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(db_path=str(tmp_path / "ingest_test.sqlite"))


def _search_response(query: str, rows: list[dict]) -> dict:
    return {"ok": True, "query": query, "results": rows}


class TestConfidenceForTier:
    def test_trusted_is_highest(self):
        assert confidence_for_tier("trusted") > confidence_for_tier("reliable")
        assert confidence_for_tier("reliable") > confidence_for_tier("unknown")

    def test_unknown_tier_defaults_to_unknown(self):
        assert confidence_for_tier("garbage") == confidence_for_tier("unknown")
        assert confidence_for_tier(None) == confidence_for_tier("unknown")

    def test_blocked_is_zero(self):
        assert confidence_for_tier("blocked") == 0.0

    def test_case_insensitive(self):
        assert confidence_for_tier("TRUSTED") == confidence_for_tier("trusted")


class TestIngestSearchResponse:
    def test_ok_false_returns_empty(self, kg):
        assert ingest_search_response(kg, {"ok": False, "results": []}) == []

    def test_no_results_returns_empty(self, kg):
        resp = _search_response("test", [])
        assert ingest_search_response(kg, resp) == []

    def test_missing_query_returns_empty(self, kg):
        resp = _search_response("", [{"url": "https://x.com", "trust_tier": "trusted"}])
        assert ingest_search_response(kg, resp) == []

    def test_ingest_single_result(self, kg):
        resp = _search_response("Super Bowl 2026 winner", [
            {"url": "https://espn.com/article", "title": "Patriots Win", "trust_tier": "trusted"},
        ])
        ids = ingest_search_response(kg, resp)
        # 2 triples per row: has_web_result + has_title
        assert len(ids) == 2

    def test_skipped_when_no_url(self, kg):
        resp = _search_response("q", [
            {"url": "", "title": "no url", "trust_tier": "trusted"},
        ])
        assert ingest_search_response(kg, resp) == []

    def test_blocked_tier_skipped(self, kg):
        resp = _search_response("q", [
            {"url": "https://blocked.com", "trust_tier": "blocked"},
        ])
        assert ingest_search_response(kg, resp) == []

    def test_confidence_reflects_tier(self, kg):
        resp = _search_response("q", [
            {"url": "https://a.com", "title": "trusted source", "trust_tier": "trusted"},
            {"url": "https://b.com", "title": "reliable source", "trust_tier": "reliable"},
            {"url": "https://c.com", "title": "unknown source", "trust_tier": "unknown"},
        ])
        ingest_search_response(kg, resp)
        results = kg.query_entity("q", direction="outgoing")
        urls_to_conf = {r["object"]: r["confidence"] for r in results
                        if r["predicate"] == "has_web_result"}
        assert urls_to_conf["https://a.com"] > urls_to_conf["https://b.com"]
        assert urls_to_conf["https://b.com"] > urls_to_conf["https://c.com"]

    def test_title_persisted_as_biographical(self, kg):
        resp = _search_response("q", [
            {"url": "https://x.com", "title": "Some Headline", "trust_tier": "trusted"},
        ])
        ingest_search_response(kg, resp)
        title_rows = kg.query_entity("https://x.com", direction="outgoing")
        title_facts = [r for r in title_rows if r["predicate"] == "has_title"]
        assert len(title_facts) == 1
        assert title_facts[0]["object"] == "Some Headline"
        assert title_facts[0]["fact_type"] == fact_types.BIOGRAPHICAL

    def test_default_fact_type_is_news(self, kg):
        resp = _search_response("q", [
            {"url": "https://x.com", "title": "T", "trust_tier": "trusted"},
        ])
        ingest_search_response(kg, resp)
        rows = kg.query_entity("q", direction="outgoing")
        results = [r for r in rows if r["predicate"] == "has_web_result"]
        assert results[0]["fact_type"] == fact_types.NEWS

    def test_override_fact_type(self, kg):
        resp = _search_response("portland weather", [
            {"url": "https://noaa.gov", "title": "Forecast", "trust_tier": "trusted"},
        ])
        ingest_search_response(kg, resp, fact_type=fact_types.WEATHER)
        rows = kg.query_entity("portland weather", direction="outgoing")
        results = [r for r in rows if r["predicate"] == "has_web_result"]
        assert results[0]["fact_type"] == fact_types.WEATHER


class TestLookupCachedResults:
    def test_returns_recent_results_by_relevance(self, kg):
        resp = _search_response("test query", [
            {"url": "https://a.com", "title": "A", "trust_tier": "trusted"},
            {"url": "https://b.com", "title": "B", "trust_tier": "reliable"},
        ])
        ingest_search_response(kg, resp)
        cached = lookup_cached_results(kg, "test query")
        assert len(cached) == 2
        # Trusted's higher confidence → higher relevance at the same age
        assert cached[0]["url"] == "https://a.com"
        assert cached[0]["relevance"] > cached[1]["relevance"]

    def test_empty_for_unknown_query(self, kg):
        assert lookup_cached_results(kg, "never asked") == []

    def test_stale_filtered_by_min_relevance(self, kg):
        # Ingest with timestamp from 2 months ago (well past news halflife)
        old = datetime.now(timezone.utc) - timedelta(days=60)
        resp = _search_response("old query", [
            {"url": "https://stale.com", "title": "Stale", "trust_tier": "trusted"},
        ])
        ingest_search_response(kg, resp, now=old)
        # 60d / 7d ≈ 8.5 halflives → tiny relevance
        cached = lookup_cached_results(kg, "old query", min_relevance=0.05)
        assert cached == []
        # But it IS in the cache without the threshold
        no_threshold = lookup_cached_results(kg, "old query", min_relevance=0.0)
        assert len(no_threshold) == 1


class TestQueryEntityWithRelevance:
    def test_sorts_by_relevance_desc(self, kg):
        # Two coexisting facts (multi-valued predicate), same confidence,
        # different ages — newer should rank higher in relevance.
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        old = datetime.now(timezone.utc) - timedelta(days=14)
        kg.add_triple("e", "saw_url", "x", valid_from=recent.isoformat(),
                      fact_type=fact_types.NEWS, confidence=0.8, allow_coexist=True)
        kg.add_triple("e", "saw_url", "y", valid_from=old.isoformat(),
                      fact_type=fact_types.NEWS, confidence=0.8, allow_coexist=True)
        rows = kg.query_entity_with_relevance("e")
        assert rows[0]["object"] == "x"
        assert rows[1]["object"] == "y"
        assert rows[0]["relevance"] > rows[1]["relevance"]

    def test_excludes_closed_triples_by_default(self, kg):
        kg.add_triple("e", "had", "old_state",
                      valid_from="2026-01-01", valid_to="2026-02-01",
                      fact_type=fact_types.TEMPORARY_STATE)
        rows = kg.query_entity_with_relevance("e", current_only=True)
        assert rows == []
        # But includes if asked
        rows_all = kg.query_entity_with_relevance("e", current_only=False)
        assert len(rows_all) == 1

    def test_min_relevance_filter(self, kg):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        kg.add_triple("e", "p", "stale", valid_from=old,
                      fact_type=fact_types.NEWS, confidence=0.9,
                      allow_coexist=True)
        kg.add_triple("e", "p", "fresh",
                      valid_from=datetime.now(timezone.utc).isoformat(),
                      fact_type=fact_types.NEWS, confidence=0.9,
                      allow_coexist=True)
        rows = kg.query_entity_with_relevance("e", min_relevance=0.1)
        # The stale one should be filtered out
        objs = {r["object"] for r in rows}
        assert "fresh" in objs
        assert "stale" not in objs

    def test_biographical_stays_high_at_any_age(self, kg):
        long_ago = (datetime.now(timezone.utc) - timedelta(days=5 * 365)).isoformat()
        kg.add_triple("Aurelius", "died_in", "180_CE",
                      valid_from=long_ago, fact_type=fact_types.BIOGRAPHICAL,
                      confidence=0.95)
        rows = kg.query_entity_with_relevance("Aurelius")
        assert rows[0]["relevance"] == pytest.approx(0.95, abs=1e-6)
