"""
Knowledge Router — unified retrieval with trust-tiered grounding.

Implements a Memento-style Read-Write-Reflect loop across all knowledge sources:
  1. MemPalace (local structured knowledge) — highest trust
  2. Knowledge Graph (entity relationships) — high trust
  3. Vector Store (semantic search) — medium trust
  4. Web Search (external) — lowest trust, requires citation

After response, optionally writes new knowledge back (skill creation).

Replaces the disconnected pre-inference grounding in agent_core.py with
a unified pipeline that queries ALL knowledge sources in priority order.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.KnowledgeRouter")


@dataclass
class GroundingResult:
    """Result from a knowledge retrieval attempt."""
    source: str           # "mempalace", "kg", "vector", "web"
    trust_tier: str       # "verified_local", "verified_external", "unverified"
    content: str          # The actual knowledge text
    url: str = ""         # Source URL (for web results)
    utility_score: float = 1.0  # Memento-style trust score (0-1)
    query: str = ""


@dataclass
class GroundingContext:
    """Aggregated grounding from all sources."""
    results: List[GroundingResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    sources_queried: List[str] = field(default_factory=list)

    @property
    def has_grounding(self) -> bool:
        return len(self.results) > 0

    @property
    def best_result(self) -> Optional[GroundingResult]:
        if not self.results:
            return None
        # Highest trust tier, then highest utility score
        tier_order = {"verified_local": 3, "verified_external": 2, "unverified": 1}
        return max(self.results, key=lambda r: (tier_order.get(r.trust_tier, 0), r.utility_score))

    def format_for_prompt(self, max_chars: int = 600) -> str:
        """Format grounding results for injection into the model prompt."""
        if not self.results:
            return ""
        parts = []
        remaining = max_chars
        for r in sorted(self.results, key=lambda x: x.utility_score, reverse=True):
            entry = f"[{r.trust_tier.upper()}] {r.content[:remaining]}"
            if r.url:
                entry += f" (Source: {r.url})"
            parts.append(entry)
            remaining -= len(entry)
            if remaining <= 50:
                break
        return "\n".join(parts)


# ── Skill Utility Scores ─────────────────────────────────────────────
# Track success/failure rates per query domain for Memento-style learning.
# Stored in a simple JSON file for persistence across restarts.

_UTILITY_SCORES_PATH = "/shared/knowledge_router/utility_scores.json"
_utility_scores: Dict[str, float] = {}


def _load_utility_scores():
    """Load utility scores from disk."""
    global _utility_scores
    try:
        import os
        if os.path.exists(_UTILITY_SCORES_PATH):
            with open(_UTILITY_SCORES_PATH) as f:
                _utility_scores = json.load(f)
    except Exception:
        _utility_scores = {}


def _save_utility_scores():
    """Persist utility scores to disk."""
    try:
        import os
        os.makedirs(os.path.dirname(_UTILITY_SCORES_PATH), exist_ok=True)
        with open(_UTILITY_SCORES_PATH, "w") as f:
            json.dump(_utility_scores, f, indent=2)
    except Exception:
        logger.debug("Failed to save utility scores", exc_info=True)


def record_outcome(query_domain: str, source: str, success: bool):
    """Record whether a knowledge source gave a good answer for a domain.

    Memento-style utility scoring: success bumps score up, failure bumps down.
    Over time, the system learns which sources are reliable for which domains.
    """
    key = f"{query_domain}:{source}"
    current = _utility_scores.get(key, 0.5)
    if success:
        _utility_scores[key] = min(1.0, current + 0.1)
    else:
        _utility_scores[key] = max(0.0, current - 0.15)  # Failures penalized more
    _save_utility_scores()
    logger.debug("Utility score %s: %.2f → %.2f (%s)",
                 key, current, _utility_scores[key], "success" if success else "failure")


def get_utility_score(query_domain: str, source: str) -> float:
    """Get the utility score for a source in a domain."""
    return _utility_scores.get(f"{query_domain}:{source}", 0.5)


# ── Knowledge Source Queries ──────────────────────────────────────────

def _query_mempalace(query: str, timeout: float = 2.0) -> List[GroundingResult]:
    """Search MemPalace for local structured knowledge."""
    results = []
    try:
        from gaia_core.utils import mcp_client
        raw = mcp_client.call_jsonrpc("palace_recall", {
            "query": query, "max_results": 2
        })
        # Parse MCP response
        inner = raw
        if isinstance(raw, dict):
            inner = raw.get("response", raw)
            if isinstance(inner, dict):
                inner = inner.get("result", inner)

        memories = []
        if isinstance(inner, dict):
            memories = inner.get("memories", inner.get("results", []))
        elif isinstance(inner, list):
            memories = inner

        for mem in memories[:2]:
            content = mem.get("content", mem.get("body", "")) if isinstance(mem, dict) else str(mem)
            if content and len(content) > 10:
                results.append(GroundingResult(
                    source="mempalace",
                    trust_tier="verified_local",
                    content=content[:300],
                    utility_score=get_utility_score("general", "mempalace"),
                    query=query,
                ))
    except Exception as e:
        logger.debug("MemPalace query failed: %s", e)
    return results


def _query_knowledge_graph(query: str) -> List[GroundingResult]:
    """Search the knowledge graph for entity relationships."""
    results = []
    try:
        from gaia_core.utils import mcp_client
        raw = mcp_client.call_jsonrpc("kg_query", {
            "query": query, "max_results": 3
        })
        inner = raw
        if isinstance(raw, dict):
            inner = raw.get("response", raw)
            if isinstance(inner, dict):
                inner = inner.get("result", inner)

        triples = []
        if isinstance(inner, dict):
            triples = inner.get("triples", inner.get("results", []))
        elif isinstance(inner, list):
            triples = inner

        for triple in triples[:3]:
            if isinstance(triple, dict):
                subj = triple.get("subject", "")
                pred = triple.get("predicate", "")
                obj = triple.get("object", "")
                content = f"{subj} {pred} {obj}"
            else:
                content = str(triple)
            if content and len(content) > 5:
                results.append(GroundingResult(
                    source="kg",
                    trust_tier="verified_local",
                    content=content[:200],
                    utility_score=get_utility_score("general", "kg"),
                    query=query,
                ))
    except Exception as e:
        logger.debug("KG query failed: %s", e)
    return results


def _query_web(query: str) -> List[GroundingResult]:
    """Search the web for external knowledge."""
    results = []
    try:
        from gaia_core.utils import mcp_client
        raw = mcp_client.call_jsonrpc("web_search", {
            "query": query[:120], "max_results": 2
        })
        inner = raw
        if isinstance(raw, dict):
            inner = raw.get("response", raw)
            if isinstance(inner, dict):
                inner = inner.get("result", inner)
            if isinstance(inner, dict):
                inner = inner.get("results", [])

        if isinstance(inner, list):
            for r in inner[:2]:
                snippet = r.get("snippet", "")
                title = r.get("title", "")
                url = r.get("url", "")
                if snippet:
                    results.append(GroundingResult(
                        source="web",
                        trust_tier="verified_external",
                        content=f"{title}: {snippet[:250]}",
                        url=url,
                        utility_score=get_utility_score("general", "web"),
                        query=query,
                    ))
    except Exception as e:
        logger.debug("Web search failed: %s", e)
    return results


# ── Main Router ───────────────────────────────────────────────────────

# Intents that should skip grounding (identity, tool operations, etc.)
_SKIP_GROUNDING_INTENTS = {
    "identity", "identity_query", "self_reference",
    "list_tools", "list_tree", "find_file", "read_file",
    "list_files", "write_file", "shell_command",
}


def ground_query(
    query: str,
    intent: str = "other",
    skip_web: bool = False,
    max_total_ms: float = 3000,
) -> GroundingContext:
    """Query all knowledge sources in priority order and return grounding context.

    Priority: MemPalace → KG → Web (if local sources insufficient)

    Args:
        query: The user's question
        intent: Detected intent (identity questions skip grounding)
        skip_web: If True, skip web search (for offline/fast mode)
        max_total_ms: Total time budget for all queries

    Returns:
        GroundingContext with results from all sources
    """
    if intent in _SKIP_GROUNDING_INTENTS:
        return GroundingContext()

    if not query or len(query.strip()) < 10:
        return GroundingContext()

    _load_utility_scores()
    t0 = time.time()
    ctx = GroundingContext()

    # 1. MemPalace — local structured knowledge (highest trust)
    try:
        palace_results = _query_mempalace(query)
        ctx.results.extend(palace_results)
        ctx.sources_queried.append("mempalace")
        if palace_results:
            logger.info("KnowledgeRouter: MemPalace returned %d results", len(palace_results))
    except Exception:
        pass

    # 2. Knowledge Graph — entity relationships
    elapsed = (time.time() - t0) * 1000
    if elapsed < max_total_ms * 0.6:
        try:
            kg_results = _query_knowledge_graph(query)
            ctx.results.extend(kg_results)
            ctx.sources_queried.append("kg")
            if kg_results:
                logger.info("KnowledgeRouter: KG returned %d results", len(kg_results))
        except Exception:
            pass

    # 3. Web Search — external knowledge (lowest trust, always cite)
    # Only search if local sources didn't provide enough
    elapsed = (time.time() - t0) * 1000
    local_has_answer = any(r.trust_tier == "verified_local" and len(r.content) > 50 for r in ctx.results)
    if not skip_web and not local_has_answer and elapsed < max_total_ms * 0.8:
        try:
            web_results = _query_web(query)
            ctx.results.extend(web_results)
            ctx.sources_queried.append("web")
            if web_results:
                logger.info("KnowledgeRouter: Web returned %d results", len(web_results))
        except Exception:
            pass

    ctx.elapsed_ms = (time.time() - t0) * 1000
    logger.info("KnowledgeRouter: %d results from %s in %.0fms",
                len(ctx.results), ctx.sources_queried, ctx.elapsed_ms)
    return ctx


# ── Post-Response Learning (Skill Write) ──────────────────────────────

def save_learned_knowledge(
    query: str,
    answer: str,
    source: str,
    success: bool,
    domain: str = "general",
):
    """After a successful grounded response, save the knowledge for future retrieval.

    Memento-style skill creation: complex successful interactions become
    reusable knowledge entries in MemPalace.
    """
    if not success or not answer or len(answer) < 20:
        return

    # Record utility outcome
    record_outcome(domain, source, success)

    # Save to MemPalace for future local retrieval
    try:
        from gaia_core.utils import mcp_client
        mcp_client.call_jsonrpc("palace_store", {
            "content": f"Q: {query}\nA: {answer}",
            "metadata": {
                "source": source,
                "domain": domain,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "utility_score": get_utility_score(domain, source),
            }
        })
        logger.info("KnowledgeRouter: Saved learned knowledge to MemPalace (domain=%s)", domain)
    except Exception as e:
        logger.debug("KnowledgeRouter: Failed to save to MemPalace: %s", e)
