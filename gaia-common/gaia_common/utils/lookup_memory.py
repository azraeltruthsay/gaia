"""Lookup memory — what GAIA has looked up before, kept with where it came from.

GAIA_Project-0vuz. Every successful web search/fetch is remembered verbatim
(not AAAK-compressed: the compression kept only codes like
'ACT+KIN+ART|actually_king_arthur', so the facts themselves were lost), with
its provenance — query, title, url, domain, trust tier, when it was
retrieved — so she can answer from memory, cite it, and know when to look
again instead of searching every time.

Sibling of the skill fabric: capture happens automatically at the tool layer
(gaia-mcp web_tools), recall happens at grounding time (gaia-core
knowledge_router), and freshness/dispute marks drive revalidation.

Layout: one JSON record per (url, snippet) under
<palace root>/knowledge/lookups/<id>.json. Recall is a keyword-overlap scan
over the records — fine for thousands of records; an index or embeddings
can replace it later without changing the record format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("GAIA.LookupMemory")

# Snippets shorter than this carry no answer worth remembering.
MIN_SNIPPET_CHARS = 40
# Page fetches are long; keep the opening, which is what grounding shows.
MAX_SNIPPET_CHARS = 1500
MAX_QUERIES_PER_RECORD = 10

# How long a lookup stays trusted before it should be checked again. Stable
# facts (legends, history, definitions) barely change; anything about "now"
# goes stale within days.
STABLE_DAYS = 365
VOLATILE_DAYS = 2
_VOLATILE_RE = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|this (week|month|year)|latest|current(ly)?|"
    r"right now|recent(ly)?|news|headlines?|price|prices|cost|stock|shares|weather|"
    r"forecast|score|scores|standings|election|polls?|release date|newest|version|"
    r"who is the (ceo|president|prime minister|head|leader))\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does doing for from get got
had has have he her hers him his how i if in into is it its just me my of on
or our she so than that the their them then there these they this those to
too up us was we were what when where which who whom why will with would you
your yours about tell know please name named called
""".split())

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

# Only function words. Question words and verbs stay: "where … get" and
# "what … name" are different questions about the same sword.
_FUNCTION_WORDS = frozenset("""
a an the and or but of to in on at by for with as is are was were be been
did do does has have had he she it they them his her its their him you your
i me my we our us this that these those can could would should will please
tell know
""".split())


def _words(text: str) -> List[str]:
    # "Arthur's" and "Arthur" are the same name for matching.
    return [re.sub(r"'s$", "", w).strip("'-") for w in _WORD_RE.findall((text or "").lower())]


def content_terms(text: str) -> set:
    """Lowercased content words (len > 2, not stopwords) for matching."""
    return {w for w in _words(text) if len(w) > 2 and w not in _STOPWORDS}


def question_terms(text: str) -> set:
    """Words that identify a question, including what it asks ("where",
    "get"), for matching a new question against ones asked before."""
    return {w for w in _words(text) if len(w) > 1 and w not in _FUNCTION_WORDS}


def question_match(query: str, past_queries: Iterable[str]) -> float:
    """Best Jaccard overlap between `query` and any past query."""
    q = question_terms(query)
    best = 0.0
    for past in past_queries:
        p = question_terms(past)
        if q and p:
            best = max(best, len(q & p) / len(q | p))
    return best


def is_volatile(query: str) -> bool:
    return bool(_VOLATILE_RE.search(query or ""))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


class LookupMemory:
    """File-backed store of remembered lookups."""

    def __init__(self, root: Path):
        self.root = Path(root)

    # ── write ─────────────────────────────────────────────────────────

    def _record_id(self, url: str, snippet: str) -> str:
        norm = " ".join(snippet.split())[:300].lower()
        return hashlib.sha1(f"{url}|{norm}".encode("utf-8")).hexdigest()[:16]

    def _path(self, rid: str) -> Path:
        return self.root / f"{rid}.json"

    def _load(self, path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("LookupMemory: unreadable record %s: %s", path.name, e)
            return None

    def _save(self, rec: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(rec["id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)  # atomic: a concurrent recall never reads half a record

    def remember(self, query: str, results: Iterable[dict], kind: str = "search",
                 now: Optional[datetime] = None) -> List[str]:
        """Remember each usable result of a lookup. Returns the record ids.

        A result seen again (same url + snippet) is refreshed — new
        retrieved_at, query added, any dispute cleared, since the web just
        corroborated it — rather than duplicated.
        """
        now = now or _now()
        days = VOLATILE_DAYS if is_volatile(query) else STABLE_DAYS
        ids: List[str] = []
        for r in results or []:
            if not isinstance(r, dict):
                continue
            snippet = (r.get("snippet") or r.get("content") or "").strip()[:MAX_SNIPPET_CHARS]
            if len(snippet) < MIN_SNIPPET_CHARS:
                continue
            url = (r.get("url") or "").strip()
            rid = self._record_id(url, snippet)
            path = self._path(rid)
            rec = self._load(path) if path.exists() else None
            if rec is None:
                rec = {
                    "id": rid,
                    "kind": kind,
                    "queries": [],
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "domain": (r.get("domain") or "").strip(),
                    "trust_tier": r.get("trust_tier") or "unverified",
                    "snippet": snippet,
                    "first_retrieved_at": _iso(now),
                    "times_recalled": 0,
                    "last_recalled_at": None,
                }
            if query and query not in rec["queries"]:
                rec["queries"] = (rec["queries"] + [query])[-MAX_QUERIES_PER_RECORD:]
            rec["retrieved_at"] = _iso(now)
            rec["revalidate_after"] = _iso(now + timedelta(days=days))
            rec.pop("disputed_at", None)
            try:
                self._save(rec)
                ids.append(rid)
            except OSError as e:
                logger.warning("LookupMemory: could not save %s: %s", rid, e)
        if ids:
            logger.info("LookupMemory: remembered %d result(s) for %r (%s, fresh for %dd)",
                        len(ids), (query or "")[:80], kind, days)
        return ids

    def mark_disputed(self, ids: Iterable[str], now: Optional[datetime] = None) -> int:
        """Flag records as needing revalidation (e.g. the user pushed back on
        an answer they supported). A disputed record is still recalled, but
        marked stale so it never stands in for a fresh lookup; the next
        lookup that returns it again clears the flag."""
        now = now or _now()
        n = 0
        for rid in ids:
            path = self._path(rid)
            rec = self._load(path) if path.exists() else None
            if rec is None:
                continue
            rec["disputed_at"] = _iso(now)
            try:
                self._save(rec)
                n += 1
            except OSError as e:
                logger.warning("LookupMemory: could not mark %s disputed: %s", rid, e)
        if n:
            logger.info("LookupMemory: %d record(s) marked for revalidation", n)
        return n

    # ── read ──────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.34,
               now: Optional[datetime] = None, touch: bool = True) -> List[dict]:
        """Remembered lookups relevant to `query`, best first.

        The record's own title + snippet must cover at least `min_score` of
        the query's content words. Its past queries only add a half-weight
        bonus for words the text lacks: every result of a search inherits
        that search's query, so counting queries fully let an off-topic
        result ("Onimusha: Way of the Sword" for "King Arthur sword") look
        relevant. Each hit carries `score`, `question_match` (how close
        `query` is to a question this record was actually retrieved for —
        the bar for answering from memory without a fresh lookup) and
        `stale` (past revalidate_after, or disputed).
        """
        now = now or _now()
        terms = content_terms(query)
        if not terms or not self.root.exists():
            return []
        scored = []
        for path in self.root.glob("*.json"):
            rec = self._load(path)
            if not rec:
                continue
            text_terms = content_terms(f"{rec.get('title', '')} {rec.get('snippet', '')}")
            text_hit = terms & text_terms
            text_score = len(text_hit) / len(terms)
            if text_score < min_score:
                continue
            query_only = (terms & content_terms(" ".join(rec.get("queries", [])))) - text_hit
            score = min(1.0, text_score + 0.5 * len(query_only) / len(terms))
            qmatch = question_match(query, rec.get("queries", []))
            scored.append((score, qmatch, rec))
        # Rank by relevance + question match: many records mention the
        # subject equally ("Excalibur is the sword of King Arthur" scores 1.0
        # for any Excalibur question); the one looked up for THIS question
        # (the Lady of the Lake, for "where did he get it") should lead.
        scored.sort(key=lambda sqr: (sqr[0] + sqr[1], sqr[2].get("retrieved_at", "")), reverse=True)
        hits = []
        for score, qmatch, rec in scored[:top_k]:
            revalidate = _parse(rec.get("revalidate_after", ""))
            stale = bool(rec.get("disputed_at")) or (revalidate is not None and now > revalidate)
            hits.append({**rec, "score": round(score, 3), "stale": stale,
                         "question_match": round(qmatch, 3)})
            if touch:
                rec["times_recalled"] = rec.get("times_recalled", 0) + 1
                rec["last_recalled_at"] = _iso(now)
                try:
                    self._save(rec)
                except OSError:
                    # Usage stats are best-effort; recall still succeeds.
                    logger.debug("LookupMemory: could not update recall stats for %s", rec.get("id"))
        if hits:
            logger.info("LookupMemory: recalled %d for %r (best %.2f%s)", len(hits), query[:80],
                        hits[0]["score"], ", stale" if hits[0]["stale"] else "")
        return hits

    def stats(self) -> Dict[str, int]:
        n = disputed = 0
        for path in self.root.glob("*.json") if self.root.exists() else []:
            rec = self._load(path)
            if rec:
                n += 1
                disputed += bool(rec.get("disputed_at"))
        return {"lookups": n, "disputed": disputed}
