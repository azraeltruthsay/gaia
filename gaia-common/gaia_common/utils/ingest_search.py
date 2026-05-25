"""Web-search-result → KG triple ingester (GAIA_Project-lw4 Stage 7).

Persists web_search responses into the KnowledgeGraph as recency-tracked
facts so that the same query can later be answered from the KG (with
proper decay) rather than re-searching. Each result row becomes a triple:

    (query, has_web_result, url)
        valid_from = retrieved_at
        confidence = trust-tier-based
        source     = "web_search|<title>|<retrieved_at>"
        fact_type  = "news" (web content) by default

A second triple captures the title for lookup-by-URL:

    (url, has_title, title)
        fact_type = "biographical"  (titles don't decay)

This is intentionally a simple heuristic ingester — LLM-based claim
extraction from snippet text is Stage 7.5 (out of scope for lw4).

Idempotency: re-ingesting the same (query, url) row updates the existing
triple via add_triple's dedup path; titles are dedup'd similarly.

Usage:
    from gaia_mcp.web_tools import web_search
    from gaia_common.utils.ingest_search import ingest_search_response

    resp = web_search({"query": "Patriots Super Bowl 2026", "max_results": 5})
    triples = ingest_search_response(kg, resp)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from gaia_common.utils import fact_types
from gaia_common.utils.knowledge_graph import KnowledgeGraph


# ── Trust tier → confidence mapping ─────────────────────────────────
# Pulled from the web_tools.py allow-list intent: trusted domains are
# wire services / governments / encyclopedias; reliable are major news
# outlets; unknown is anything else that wasn't blocked outright.

_TIER_CONFIDENCE: dict[str, float] = {
    "trusted":  0.85,
    "reliable": 0.70,
    "unknown":  0.40,
    "blocked":  0.0,    # shouldn't reach the ingester, but defensive
}


def confidence_for_tier(tier: Optional[str]) -> float:
    """Map a web_tools.trust_tier label to a [0, 1] confidence."""
    if not tier:
        return _TIER_CONFIDENCE["unknown"]
    return _TIER_CONFIDENCE.get(tier.lower(), _TIER_CONFIDENCE["unknown"])


def _canon_query(query: str) -> str:
    """Canonicalize a query string for use as a KG subject."""
    return (query or "").strip()[:200]  # KG entity names are bounded


def ingest_search_response(
    kg: KnowledgeGraph,
    response: dict,
    *,
    world: str = "actuality",
    now: Optional[datetime] = None,
    fact_type: Optional[str] = None,
    query_override: Optional[str] = None,
) -> list[str]:
    """Persist a web_search response into the KG.

    Returns the list of triple IDs created (or de-duped to). Empty list
    if the response wasn't `ok` or had no results.

    `fact_type` defaults to NEWS — most web content is news-shaped and
    decays with a 7-day half-life. Override for known-different content
    (e.g. WEATHER for weather-API results).
    """
    if not response or not response.get("ok"):
        return []
    rows = response.get("results") or []
    if not rows:
        return []

    query = _canon_query(query_override or response.get("query") or "")
    if not query:
        return []

    if now is None:
        now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    ft = fact_type or fact_types.NEWS

    created: list[str] = []
    for row in rows:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        title = (row.get("title") or "").strip()
        tier = row.get("trust_tier")
        conf = confidence_for_tier(tier)
        if conf <= 0.0:
            continue  # blocked / no trust

        source_str = f"web_search|{title[:80]}|{now_iso}" if title else f"web_search|{now_iso}"

        # (query) --has_web_result--> (url)
        # allow_coexist: multiple URLs for the same query are NOT
        # contradictions — they're a result list.
        triple_id = kg.add_triple(
            subject=query,
            predicate="has_web_result",
            obj=url,
            valid_from=now_iso,
            confidence=conf,
            source=source_str,
            world=world,
            fact_type=ft,
            allow_coexist=True,
        )
        created.append(triple_id)

        # (url) --has_title--> (title): no-decay biographical, so the
        # title persists even as the search result freshness decays.
        if title:
            tid = kg.add_triple(
                subject=url,
                predicate="has_title",
                obj=title,
                valid_from=now_iso,
                confidence=conf,
                source=f"web_search|{now_iso}",
                world=world,
                fact_type=fact_types.BIOGRAPHICAL,
            )
            created.append(tid)

    return created


def lookup_cached_results(
    kg: KnowledgeGraph,
    query: str,
    *,
    world: str = "actuality",
    min_relevance: float = 0.05,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Look up cached search results for a query, ranked by recency.

    Returns a list of {"url", "confidence", "valid_from", "relevance"}
    dicts sorted by relevance DESC. Filters out triples below
    min_relevance — useful for "is the cache still fresh, or should we
    refetch?". Empty list means re-search.
    """
    q = _canon_query(query)
    if not q:
        return []
    rows = kg.query_entity_with_relevance(
        q,
        direction="outgoing",
        world=world,
        now=now,
        min_relevance=min_relevance,
    )
    return [
        {
            "url": r["object"],
            "confidence": r["confidence"],
            "valid_from": r["valid_from"],
            "relevance": r["relevance"],
            "fact_type": r.get("fact_type"),
        }
        for r in rows
        if r.get("predicate") == "has_web_result"
    ]
