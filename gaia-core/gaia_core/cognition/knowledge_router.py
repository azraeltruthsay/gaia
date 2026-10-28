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
import re
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.KnowledgeRouter")


@dataclass
class GroundingResult:
    """Result from a knowledge retrieval attempt."""
    source: str           # "mempalace", "journal", "kg", "vector", "web", "system", "memory"
    trust_tier: str       # "verified_local", "verified_external", "unverified", "remembered"
    content: str          # The actual knowledge text
    url: str = ""         # Source URL (for web results)
    utility_score: float = 1.0  # Memento-style trust score (0-1)
    query: str = ""
    # Lookup memory (0vuz): provenance for citing and revalidating.
    title: str = ""
    retrieved_at: str = ""
    stale: bool = False
    score: float = 0.0
    question_match: float = 0.0
    memory_id: str = ""
    # 6afh: how close the question is to one this record answered well /
    # failed to answer before (lookup memory feedback).
    confirmed_match: float = 0.0
    unhelpful_match: float = 0.0


@dataclass
class GroundingContext:
    """Aggregated grounding from all sources."""
    results: List[GroundingResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    sources_queried: List[str] = field(default_factory=list)
    # 6afh: the question lookup memory was asked — feedback on the answer
    # goes back under this same question, so a later recall can match it.
    query: str = ""

    @property
    def has_grounding(self) -> bool:
        return len(self.results) > 0

    @property
    def best_result(self) -> Optional[GroundingResult]:
        if not self.results:
            return None
        # Highest trust tier, then highest utility score
        tier_order = {"verified_local": 3, "verified_external": 2, "remembered": 2, "unverified": 1}
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

def _query_journal(query: str, k: int = 3, body_chars: int = 320) -> List[GroundingResult]:
    """Search GAIA's self-narrative journal for past entries that match.

    Phase D of vam — surfaces consolidation/reflection entries when a
    user references past events ("remember when...", "that day we...").
    Trust tier is verified_local because these entries are GAIA's own
    grounded narrative, written from LiteJournal ticks + samvega
    artifacts (not free-form recollection).
    """
    results: List[GroundingResult] = []
    try:
        from gaia_core.memory.journal import search_entries
    except Exception as e:
        logger.debug("Journal search unavailable: %s", e)
        return results
    try:
        # 8xr4.19: skip the 'cognition' stream. Those are per-turn deliberation
        # transcripts — a record of what she SAID, not of what is true — and
        # they outnumber the narrative streams ~150:1, so they won nearly every
        # search. Fed back as grounding, a confabulated answer from the
        # previous turn came back as "evidence" for the next one.
        hits = [
            (entry, score)
            for entry, score in search_entries(query, k=k * 4)
            if getattr(entry, "context", "self") != "cognition"
        ][:k]
        for entry, score in hits:
            body = (entry.body or "").strip()
            if len(body) > body_chars:
                body = body[:body_chars].rstrip() + "…"
            tag_blurb = ", ".join(entry.tags or []) or "—"
            content = (
                f"Journal entry {entry.id} (sig {entry.significance}, "
                f"tags: {tag_blurb}): {body}"
            )
            # Map relevance score (typically 1.0–~6.0) to utility 0–1
            utility = min(1.0, score / 6.0 + 0.2)
            results.append(GroundingResult(
                source="journal",
                trust_tier="verified_local",
                content=content,
                utility_score=utility,
                query=query,
            ))
    except Exception:
        logger.debug("Journal search failed", exc_info=True)
    return results


def _query_mempalace(query: str, timeout: float = 2.0) -> List[GroundingResult]:
    """Search MemPalace for local structured knowledge."""
    results = []
    try:
        from gaia_core.utils import mcp_client
        raw = mcp_client.call_jsonrpc("palace_recall", {
            "query": query, "top_k": 2,  # palace_recall reads top_k (was max_results)
            # Lookup memory is queried on its own step (1a); don't count it twice.
            "include_lookups": False,
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
            if _AAAK_RECORD_RE.match(content or ""):
                # 0vuz: AAAK-only memories ("0:ACT+KIN+ART|actually_king_arthur|…")
                # hold codes, not facts; as verified_local they also satisfied
                # ground_query's local_has_answer and switched off the web.
                logger.debug("MemPalace: skipping AAAK-only memory %s", mem.get("palace_id", "?") if isinstance(mem, dict) else "?")
                continue
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


_AAAK_RECORD_RE = re.compile(r"^\w+\|\w+\|\d{4}-\d{2}-\d{2}\|")

# Lookup memory (0vuz). A fresh hit that is relevant (score) AND was looked
# up for much the same question (question_match) answers in place of a fresh
# search; other hits ride along with the web results. Relevance alone isn't
# enough: in the live test the sword's-*name* results scored 0.75 for "Where
# did he get the sword?" (same subject words), the search was skipped, and
# Core answered "Morgan Le Fay".
#
# 6afh: being retrieved FOR a question isn't proof a record ANSWERS it (the
# Excalibur pages stored for "where did he get it" never say where). So
# memory alone answers only when a record was confirmed — an earlier answer
# from it held up (lookup_memory.feedback) — for much the same question.
# Unconfirmed memory rides along with a fresh search; records that failed
# this question before are dropped.
LOOKUP_MIN_SCORE = 0.34
STRONG_MEMORY_SCORE = 0.5
STRONG_QUESTION_MATCH = 0.6


def _mcp_result(raw: Any) -> Any:
    """Unwrap mcp_client.call_jsonrpc's {"response": {"result": ...}}."""
    inner = raw
    if isinstance(raw, dict):
        inner = raw.get("response", raw)
        if isinstance(inner, dict):
            inner = inner.get("result", inner)
    return inner


def _query_lookup_memory(query: str, top_k: int = 3,
                         min_score: float = LOOKUP_MIN_SCORE) -> List[GroundingResult]:
    """What GAIA looked up before that bears on ``query`` (gaia-mcp captures
    every web search/fetch into the palace's lookup memory)."""
    results: List[GroundingResult] = []
    try:
        from gaia_core.utils import mcp_client
        inner = _mcp_result(mcp_client.call_jsonrpc("palace_recall_lookup", {
            "query": query, "top_k": top_k, "min_score": min_score,
        }))
        hits = inner.get("results", []) if isinstance(inner, dict) else []
        for hit in hits:
            if not isinstance(hit, dict) or not hit.get("snippet"):
                continue
            results.append(GroundingResult(
                source="memory",
                trust_tier="remembered",
                content=hit["snippet"],
                url=hit.get("url", ""),
                query=query,
                title=hit.get("title") or hit.get("domain") or "",
                retrieved_at=hit.get("retrieved_at", ""),
                stale=bool(hit.get("stale")),
                score=float(hit.get("score") or 0.0),
                question_match=float(hit.get("question_match") or 0.0),
                memory_id=hit.get("id", ""),
                confirmed_match=float(hit.get("confirmed_match") or 0.0),
                unhelpful_match=float(hit.get("unhelpful_match") or 0.0),
            ))
    except Exception as e:
        logger.debug("Lookup memory recall failed: %s", e)
    failed = [r for r in results if r.unhelpful_match >= STRONG_QUESTION_MATCH]
    if failed:
        logger.info("KnowledgeRouter: dropping %d remembered lookup(s) that failed this question before",
                    len(failed))
    return [r for r in results if r.unhelpful_match < STRONG_QUESTION_MATCH]


def _strong_memory(results: List[GroundingResult]) -> bool:
    """A fresh, relevant record already confirmed for (nearly) this question."""
    return any(r.source == "memory" and not r.stale and r.score >= STRONG_MEMORY_SCORE
               and r.question_match >= STRONG_QUESTION_MATCH
               and r.confirmed_match >= STRONG_QUESTION_MATCH for r in results)


def _asked_before(results: List[GroundingResult]) -> bool:
    """She looked up (nearly) this question before, whatever came of it."""
    return any(r.source == "memory" and r.question_match >= STRONG_QUESTION_MATCH for r in results)


def _dispute_memory(results: List[GroundingResult]) -> None:
    """Mark the remembered lookups behind a disputed answer for revalidation.
    Called BEFORE the fresh search: any the search returns again are
    refreshed by gaia-mcp's capture (clearing the mark); the rest stay
    flagged and can't stand in for a lookup until re-confirmed."""
    ids = [r.memory_id for r in results if r.source == "memory" and r.memory_id]
    if not ids:
        return
    try:
        from gaia_core.utils import mcp_client
        mcp_client.call_jsonrpc("palace_dispute_lookup", {"ids": ids})
        for r in results:
            r.stale = True
        logger.info("KnowledgeRouter: %d remembered lookup(s) marked for revalidation", len(ids))
    except Exception as e:
        logger.warning("Could not mark remembered lookups for revalidation: %s", e)


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


def _query_web(query: str, max_results: int = 2, snippet_chars: int = 250) -> List[GroundingResult]:
    """Search the web for external knowledge."""
    results = []
    try:
        from gaia_core.utils import mcp_client
        raw = mcp_client.call_jsonrpc("web_search", {
            "query": query[:120], "max_results": max_results
        })
        inner: Any = raw
        if isinstance(raw, dict):
            inner = raw.get("response", raw)
            if isinstance(inner, dict):
                inner = inner.get("result", inner)
            if isinstance(inner, dict):
                inner = inner.get("results", [])

        if isinstance(inner, list):
            for r in inner[:max_results]:
                snippet = r.get("snippet", "")
                title = r.get("title", "")
                url = r.get("url", "")
                if snippet:
                    results.append(GroundingResult(
                        source="web",
                        trust_tier="verified_external",
                        content=f"{title}: {snippet[:snippet_chars]}",
                        url=url,
                        utility_score=get_utility_score("general", "web"),
                        query=query,
                        # gaia-mcp remembered it; feedback goes to this record.
                        memory_id=r.get("memory_id", ""),
                    ))
    except Exception as e:
        logger.debug("Web search failed: %s", e)
    return results


def _query_system_context(query: str) -> List[GroundingResult]:
    """Provide live system data for time/status/uptime questions.

    This is NOT the model answering — it's the SYSTEM providing verified
    sensor data. The model just needs to relay it.
    """
    results = []
    q = query.lower()
    time_signals = ["time", "clock", "date", "day", "what time", "current time"]
    uptime_signals = ["uptime", "how long", "running"]

    if any(s in q for s in time_signals):
        from datetime import datetime, timezone, timedelta
        try:
            _tz_off = int(os.environ.get("LOCAL_TZ_OFFSET", "-7"))
            _tz = timezone(timedelta(hours=_tz_off))
            _now = datetime.now(_tz)
            _display = _now.strftime("%-I:%M %p %Z, %A %B %d, %Y")
            _utc = _now.astimezone(timezone.utc).strftime("%H:%M UTC")
            results.append(GroundingResult(
                source="system_clock",
                trust_tier="verified_local",
                content=f"The current time is {_display} ({_utc}).",
                utility_score=1.0,
                query=query,
            ))
            logger.info("SystemContext: clock → %s", _display)
        except Exception as e:
            logger.warning("SystemContext: clock query failed: %s", e)

    if any(s in q for s in uptime_signals):
        try:
            with open("/proc/uptime") as f:
                uptime_secs = float(f.read().split()[0])
            hours = int(uptime_secs // 3600)
            mins = int((uptime_secs % 3600) // 60)
            results.append(GroundingResult(
                source="system_uptime",
                trust_tier="verified_local",
                content=f"System uptime: {hours} hours, {mins} minutes ({int(uptime_secs)} seconds total).",
                utility_score=1.0,
                query=query,
            ))
        except Exception:
            pass

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
    ctx = GroundingContext(query=query)

    # 0. System context — live sensor data (time, uptime, GPU)
    # Highest trust — these are direct system readings, not model knowledge.
    try:
        sys_results = _query_system_context(query)
        ctx.results.extend(sys_results)
        if sys_results:
            ctx.sources_queried.append("system")
            logger.info("KnowledgeRouter: System context returned %d results", len(sys_results))
    except Exception:
        pass

    # 1. MemPalace — local structured knowledge (highest trust)
    try:
        palace_results = _query_mempalace(query)
        ctx.results.extend(palace_results)
        ctx.sources_queried.append("mempalace")
        if palace_results:
            logger.info("KnowledgeRouter: MemPalace returned %d results", len(palace_results))
    except Exception:
        pass

    # 1a. Lookup memory (0vuz) — what she has looked up before, with its
    # source. A strong fresh hit makes the web search below unnecessary.
    try:
        memory_results = _query_lookup_memory(query)
        ctx.results.extend(memory_results)
        ctx.sources_queried.append("memory")
        if memory_results:
            logger.info("KnowledgeRouter: Lookup memory returned %d results (best %.2f)",
                        len(memory_results), memory_results[0].score)
    except Exception:
        logger.debug("Lookup memory step failed", exc_info=True)

    # 1b. Self-narrative journal — GAIA's own consolidation/reflection
    # entries. Same trust tier as MemPalace (local, structured, grounded
    # in LiteJournal + samvega), but distinct in voice — first-person
    # narrative that surfaces past events when the user references them.
    try:
        journal_results = _query_journal(query)
        ctx.results.extend(journal_results)
        ctx.sources_queried.append("journal")
        if journal_results:
            logger.info("KnowledgeRouter: Journal returned %d results", len(journal_results))
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
    local_has_answer = _strong_memory(ctx.results) or any(
        r.trust_tier == "verified_local" and len(r.content) > 50 for r in ctx.results)
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


_CONVERSATION_QUERY_STOPWORDS = frozenset("""
a an the and or but so of to in on at by for from with about into over as is
are was were be been being do does did didn't doesn't wasn't isn't weren't
aren't don't has have had it its it's he she they them his her their him this
that these those there here what which who whom whose when where why how can
could would should will shall may might must you your yours i me my we our us
tell know think thought sure please just really actually yes no not did get got
""".split())


def conversation_query(user_turns: List[str], max_chars: int = 120) -> str:
    """Keyword query built from the last few user turns, oldest first.

    A follow-up like "Didn't he get it from the lady in the lake?" names no
    one; searching it alone (or Stage 0's 'he' + 'sword') finds Onimusha and
    Roblox codes. The earlier turns carry the subject ("King Arthur's sword").
    """
    seen: List[str] = []
    for turn in user_turns:
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", turn or ""):
            w = word.strip("'-")
            if not w or w.lower() in _CONVERSATION_QUERY_STOPWORDS:
                continue
            if w.lower() not in (s.lower() for s in seen):
                seen.append(w)
    query = ""
    for w in seen:
        if len(query) + len(w) + 1 > max_chars:
            break
        query = f"{query} {w}".strip()
    return query


_SENTENCE_OPENERS = frozenset("""
according well sure okay ok however also then yes no great good oh hmm in on
from after before as at by for if once since when while
""".split())


def proper_nouns(text: str) -> str:
    """The capitalised, non-common words of an answer: its names ("Excalibur",
    "Lady") without its prose ("According to the legend, ...")."""
    out: List[str] = []
    for w in re.findall(r"\b[A-Z][\w'-]+", text or ""):
        low = w.lower()
        if low in _CONVERSATION_QUERY_STOPWORDS or low in _SENTENCE_OPENERS:
            continue
        if low not in (o.lower() for o in out):
            out.append(w)
    return " ".join(out)


def _name_key(word: str) -> str:
    return re.sub(r"'s$", "", word.lower())


def followup_context(turns: List[Dict[str, Any]], strip=lambda t: t) -> List[str]:
    """Context for a follow-up's search query, from the recent turns: the
    user's turns as they were, and from GAIA's answers only the names the
    conversation returns to — said in at least two of these turns, by
    either side. Those name the subject ("Excalibur", "Arthur"); a name
    said once is a particular of one answer, possibly a wrong one (2vei:
    a confabulated "…Merlin … Malory's Le Morte … Pellinore" answer sent
    "Who threw it back into the lake?" to Pellinore pages; with only the
    subject names the top result was Bedivere). When the user's turns name
    nothing and no name recurs, the latest answer is the only subject
    available, so its names are used.

    `strip` cleans an answer's text (think tags, mind tag) first.
    """
    texts = [(t.get("role"), (t.get("content") or "") if t.get("role") == "user"
              else strip(t.get("content") or "")) for t in turns
             if isinstance(t, dict) and t.get("role") in ("user", "assistant")]
    turn_names = [{_name_key(w) for w in proper_nouns(text).split()} for _, text in texts]
    seen_in = {}
    for names in turn_names:
        for n in names:
            seen_in[n] = seen_in.get(n, 0) + 1
    user_named = any(names for (role, _), names in zip(texts, turn_names) if role == "user")
    context: List[str] = []
    recurring_any = False
    for (role, text), names in zip(texts, turn_names):
        if role == "user":
            context.append(text)
            continue
        keep = [w for w in proper_nouns(text).split() if seen_in.get(_name_key(w), 0) >= 2]
        recurring_any = recurring_any or bool(keep)
        context.append(" ".join(keep))
    if not user_named and not recurring_any:
        for i in range(len(texts) - 1, -1, -1):
            if texts[i][0] == "assistant":
                context[i] = proper_nouns(texts[i][1])
                break
    return context


def followup_query(question: str, earlier_turns: List[str], max_chars: int = 120) -> str:
    """The follow-up question as asked, plus the subject words from the turns
    before it: "Where did he get the sword? King Arthur's Excalibur". A keyword
    bag alone drops "where … get" and finds pages that never say where.
    gaia-mcp's search takes 120 chars, so the subject words fill what's left."""
    question = " ".join((question or "").split())
    asked = {w.lower() for w in re.findall(r"[\w'-]+", question)}
    extra = [w for w in conversation_query(earlier_turns, max_chars=max_chars).split()
             if w.lower() not in asked]
    query = question
    for w in extra:
        if len(query) + len(w) + 1 > max_chars:
            break
        query = f"{query} {w}".strip()
    return query


# Questions about the user ("What is my favorite color?") are answered by the
# conversation, not by anything she looked up. ecnc smoke test 16: an earlier
# hedge-triggered web search for that question was kept in lookup memory, so
# the next ask counted as "looked up before" (6afh), searched the web again,
# and the results crowded out "my favorite color is cerulean" said a turn ago.
# First person as owner or subject only: "tell me" / "show us" are how
# knowledge questions are asked ("Can you tell me the name of King Arthur's
# sword?"), not questions about the user.
_ABOUT_THE_USER_RE = re.compile(r"\b(my|mine|i|i'm|i've|i'd|myself|our|ours|we)\b", re.IGNORECASE)


def _about_the_user(question: str) -> bool:
    return bool(_ABOUT_THE_USER_RE.search(question or ""))


def recall_answer(question: str) -> Optional[GroundingContext]:
    """Grounding for turns the router otherwise skips (intent 'chat' covers
    plenty of factual questions), if she looked much the same question up
    before. Returns None otherwise, so chitchat stays ungrounded and doesn't
    register as a knowledge gap.
      - memory confirmed for this question (_strong_memory): memory only.
      - asked before but not confirmed (6afh): it's a knowledge question
        her memory hasn't answered yet — search again, memory after.
    """
    if _about_the_user(question):
        logger.debug("KnowledgeRouter: %r is about the user; lookup memory skipped", question[:80])
        return None
    t0 = time.time()
    memory = _query_lookup_memory(question)
    if _strong_memory(memory):
        ctx = GroundingContext(results=[r for r in memory if not r.stale],
                               sources_queried=["memory"], query=question)
        ctx.elapsed_ms = (time.time() - t0) * 1000
        logger.info("KnowledgeRouter: %r answered from %d remembered lookup(s) (question match %.2f)",
                    question[:80], len(ctx.results), max(r.question_match for r in memory))
        return ctx
    if not _asked_before(memory):
        return None
    ctx = GroundingContext(query=question)
    try:
        ctx.results.extend(_query_web(question, max_results=3, snippet_chars=500))
        ctx.sources_queried.append("web")
    except Exception:
        logger.debug("Re-lookup of an unconfirmed memory failed", exc_info=True)
    ctx.results.extend(memory)
    ctx.sources_queried.append("memory")
    ctx.elapsed_ms = (time.time() - t0) * 1000
    logger.info("KnowledgeRouter: %r was looked up before but never confirmed; searched again "
                "(%d web + %d remembered)", question[:80], len(ctx.results) - len(memory), len(memory))
    return ctx


def ground_conversation(user_turns: List[str], reason: str = "follow-up",
                        context_turns: Optional[List[str]] = None) -> GroundingContext:
    """Web-ground a turn whose subject lives in the earlier turns.

    With ``context_turns`` the last of ``user_turns`` is searched as asked,
    followed by the subject words of ``context_turns`` (follow-ups). Without,
    the query is a keyword bag over ``user_turns`` (pushback — the earlier
    answer is the thing in dispute, so it isn't searched). The journal and
    palace wings are skipped: they hold her own past answers to this same
    conversation. Lookup memory (0vuz) is not her answers but what sources
    said, so it is used:
      - follow-up: a strong fresh memory answers without a search; weaker
        hits follow the web results.
      - pushback: the remembered lookups are marked for revalidation and the
        web is asked again; memory is only the fallback if the web is empty.
    """
    ctx = GroundingContext()
    if context_turns is not None and user_turns:
        query = followup_query(user_turns[-1], context_turns)
    else:
        query = conversation_query(user_turns)
    if not query:
        return ctx
    ctx.query = query
    pushback = reason.startswith("pushback")
    t0 = time.time()
    memory = _query_lookup_memory(query)
    if pushback and memory:
        _dispute_memory(memory)
    elif _strong_memory(memory):
        ctx.results.extend(memory)
        ctx.sources_queried.append("memory")
        ctx.elapsed_ms = (time.time() - t0) * 1000
        logger.info("KnowledgeRouter: %s grounding %r → answered from %d remembered lookup(s), no search",
                    reason, query, len(memory))
        return ctx
    web: List[GroundingResult] = []
    try:
        # The question was asked of these results specifically; give the
        # model enough of each to find the answer, not just the page lead.
        web = _query_web(query, max_results=3, snippet_chars=500)
        ctx.results.extend(web)
        ctx.sources_queried.append("web")
    except Exception:
        logger.debug("Conversation grounding failed", exc_info=True)
    if memory and (not pushback or not web):
        # Follow-up: remembered context after the fresh results. Pushback
        # with an empty web: the (now stale-marked) memory beats nothing.
        ctx.results.extend(memory)
        ctx.sources_queried.append("memory")
    ctx.elapsed_ms = (time.time() - t0) * 1000
    logger.info("KnowledgeRouter: %s grounding %r → %d web + %d remembered in %.0fms",
                reason, query, len(web), len(ctx.results) - len(web), ctx.elapsed_ms)
    return ctx


def disputed_exchange(history: List[Dict[str, Any]], text: str, lookback: int = 3) -> Tuple[str, str]:
    """The (question, answer) pair a pushback is about, from the last few.

    The most recent answer sharing a content word with the pushback wins,
    else the most recent answer. "Wasn't it Lancelot who threw it back?" after
    a "how are you?" detour is about "…Bedivere threw it back", not the
    health reply; taking the last turn disputed the wrong answer and searched
    "feeling today" (8xr4.25).
    """
    pairs: List[Tuple[str, str]] = []
    question = ""
    for h in history or []:
        if not isinstance(h, dict):
            continue
        content = (h.get("content") or "").strip()
        if h.get("role") == "user":
            question = content
        elif h.get("role") == "assistant" and content:
            pairs.append((question, content))
    if not pairs:
        return "", ""
    words = {w for w in re.findall(r"[\w'-]+", (text or "").lower())
             if w not in _CONVERSATION_QUERY_STOPWORDS and len(w) > 2}
    for q, a in reversed(pairs[-lookback:]):
        if words & set(re.findall(r"[\w'-]+", f"{q} {a}".lower())):
            return q, a
    return pairs[-1]


def ground_pushback(user_turns: List[str]) -> GroundingContext:
    """8xr4.19: web-ground a turn where the user disputes GAIA's last answer.

    Neither tier could judge "Didn't he get it from the lady in the lake?" from
    weights alone — both defended the wizard they'd invented. With one search
    result in front of them they answered correctly every time.
    """
    return ground_conversation(user_turns, reason="pushback")


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

    # bza: skip conversational domains. MemPalace was designed for curated
    # long-term knowledge, not chat transcripts. Saving every philosophical
    # reply as "learned knowledge" pollutes future retrievals — confabulated
    # answers (e.g. "Amin Howeidi" being woven into an AGI discussion via
    # bad grounding) get persisted and then re-surfaced by knowledge_router
    # on adjacent queries, forming a confabulation feedback loop. Real
    # knowledge saves should go through explicit knowledge.add tool calls,
    # not implicit transcript persistence.
    _conversational_domains = {
        "chat", "greeting", "identity", "recitation", "time",
        "status", "casual_conversation", "small_talk",
    }
    if domain.lower() in _conversational_domains:
        logger.debug(
            "KnowledgeRouter: skipping MemPalace save — domain '%s' is "
            "conversational (transcripts != knowledge)", domain,
        )
        return

    # Record utility outcome
    record_outcome(domain, source, success)

    # Save to MemPalace for future local retrieval.
    # bza: param schema fixed — palace_store expects {text, source, date_str},
    # not {content, metadata}. The previous mismatch made every save fail
    # silently with "text parameter required", while the caller still logged
    # "Saved learned knowledge" — three months of false-success logs.
    try:
        from gaia_core.utils import mcp_client
        _rpc_resp = mcp_client.call_jsonrpc("palace_store", {
            "text": f"Q: {query}\nA: {answer}",
            "source": f"{source}|{domain}|util={get_utility_score(domain, source):.2f}",
            "date_str": time.strftime("%Y-%m-%d"),
        })
        # call_jsonrpc returns {"ok": bool, "response": {...}, "error": "..."}
        # Only log success when the call actually succeeded — don't lie.
        _ok = bool(_rpc_resp.get("ok"))
        if _ok:
            logger.info("KnowledgeRouter: saved to MemPalace (domain=%s)", domain)
        else:
            logger.warning(
                "KnowledgeRouter: MemPalace save FAILED (domain=%s): %s",
                domain, _rpc_resp.get("error") or _rpc_resp,
            )
    except Exception as e:
        logger.debug("KnowledgeRouter: exception during MemPalace save: %s", e)
