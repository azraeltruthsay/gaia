"""Skill-failure pattern accumulator (GAIA_Project-5qy Phase 1).

The existing record_outcome in knowledge_router tracks scalar utility
scores (domain:source → float) — useful for ranking but it forgets the
individual queries that failed. We need that context: to draft a new
skill from N consistent failures, we need to know WHAT was being
asked, not just that something failed.

This module is the append-only failure log + threshold detector.
It does NOT generate skills (that's skill_creator.py). It does NOT
call LLMs. It just remembers what failed, lets callers ask "have we
seen this pattern 3+ times in the last week?", and supports the
trigger condition the self-improving skill creator needs.

Log format: JSONL at /shared/knowledge_router/failure_log.jsonl

  {"t": <iso>, "intent": "...", "query": "...", "skill_name": "...",
   "source": "...", "pattern": "<derived>"}

Pattern derivation: stopword-trimmed first 3-5 content words of the
query, joined with the intent. e.g. (intent="research",
query="what is the latest python release") → "research:latest python release"

The pattern is what we count against the threshold. Patterns are
text-similarity-friendly without LLM normalization — we don't need
perfect grouping, just "consistent failure of the same KIND of ask."
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GAIA.SkillFailures")


DEFAULT_LOG_PATH = os.environ.get(
    "GAIA_SKILL_FAILURE_LOG",
    "/shared/knowledge_router/failure_log.jsonl",
)

# Threshold defaults: 3 failures in a 7-day window triggers consideration.
DEFAULT_THRESHOLD = 3
DEFAULT_WINDOW_HOURS = 7 * 24

# Lightweight stopword list for query-pattern derivation. Goal is
# stable grouping, not linguistic correctness — keep tight.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "if", "then", "else",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "we", "us", "our", "this",
    "that", "these", "those", "there", "here",
    "tell", "show", "give", "get", "ask",
})

_log_lock = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_query_tokens(query: str, *, max_tokens: int = 5) -> list[str]:
    """Strip stopwords + punctuation, return the first N content tokens."""
    if not query:
        return []
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", " ", query.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS]
    return tokens[:max_tokens]


def derive_pattern(intent: Optional[str], query: str) -> str:
    """Build the grouping key for failure-pattern threshold counting.

    Empty intent gets normalized to "other"; empty query yields a
    pattern with just the intent (so we still group same-intent
    blank-query failures together).
    """
    intent_key = (intent or "other").strip().lower() or "other"
    tokens = _normalize_query_tokens(query)
    if not tokens:
        return f"{intent_key}:_"
    return f"{intent_key}:{' '.join(tokens)}"


def record_failure(
    *,
    intent: Optional[str],
    query: str,
    skill_name: Optional[str] = None,
    source: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Optional[dict]:
    """Append a failure entry to the JSONL log.

    Returns the recorded entry dict or None on write failure. Best-
    effort — log path errors are logged at DEBUG and the function
    returns None so the calling skill-gateway code never breaks on
    log IO.
    """
    if not query and not intent:
        # Nothing to record — both fields blank means a malformed call;
        # don't pollute the log with empty rows.
        return None
    path = log_path or DEFAULT_LOG_PATH
    entry = {
        "t": _utcnow_iso(),
        "intent": intent or "",
        "query": query or "",
        "skill_name": skill_name or "",
        "source": source or "",
        "pattern": derive_pattern(intent, query),
    }
    try:
        with _log_lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry
    except OSError as e:
        logger.debug("record_failure write failed: %s", e)
        return None


def _read_entries_within(
    log_path: str, window_hours: float, *, now: Optional[datetime] = None,
) -> list[dict]:
    """Read all entries within the time window, oldest first."""
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)
    out: list[dict] = []
    if not os.path.exists(log_path):
        return out
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("t")
                if not ts:
                    continue
                try:
                    rec_t = datetime.fromisoformat(ts)
                    if rec_t.tzinfo is None:
                        rec_t = rec_t.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if rec_t < cutoff:
                    continue
                out.append(rec)
    except OSError as e:
        logger.debug("read_entries_within failed: %s", e)
    return out


def count_failures_for_pattern(
    pattern: str,
    *,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    log_path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """Count entries matching a pattern within the window."""
    if not pattern:
        return 0
    path = log_path or DEFAULT_LOG_PATH
    entries = _read_entries_within(path, window_hours, now=now)
    return sum(1 for e in entries if e.get("pattern") == pattern)


def find_patterns_above_threshold(
    *,
    threshold: int = DEFAULT_THRESHOLD,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    log_path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Group failure entries by pattern; return patterns at or above
    the threshold within the time window.

    Each returned dict has:
      pattern, count, intent, recent_queries (last 10), oldest_t, newest_t

    Sorted by count DESC, then newest_t DESC so highest-impact
    patterns surface first.
    """
    path = log_path or DEFAULT_LOG_PATH
    entries = _read_entries_within(path, window_hours, now=now)
    by_pattern: dict[str, list[dict]] = {}
    for e in entries:
        pat = e.get("pattern")
        if not pat:
            continue
        by_pattern.setdefault(pat, []).append(e)
    out: list[dict] = []
    for pat, group in by_pattern.items():
        if len(group) < threshold:
            continue
        group_sorted = sorted(group, key=lambda r: r.get("t", ""))
        recent_queries = [r.get("query", "") for r in group_sorted[-10:]]
        intent = group_sorted[0].get("intent", "")
        out.append({
            "pattern": pat,
            "count": len(group),
            "intent": intent,
            "recent_queries": recent_queries,
            "oldest_t": group_sorted[0].get("t", ""),
            "newest_t": group_sorted[-1].get("t", ""),
        })
    out.sort(key=lambda d: (-d["count"], d["newest_t"]), reverse=False)
    out.sort(key=lambda d: (d["count"], d["newest_t"]), reverse=True)
    return out


def recent_failures_for_pattern(
    pattern: str,
    *,
    limit: int = 10,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    log_path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Return the most recent failure entries for a pattern (newest first)."""
    if not pattern:
        return []
    path = log_path or DEFAULT_LOG_PATH
    entries = _read_entries_within(path, window_hours, now=now)
    matched = [e for e in entries if e.get("pattern") == pattern]
    matched.sort(key=lambda r: r.get("t", ""), reverse=True)
    return matched[:limit]


def clear_pattern(
    pattern: str,
    *,
    log_path: Optional[str] = None,
) -> int:
    """Remove all entries for a pattern from the log.

    Used after a skill has been created for the pattern — clearing
    prevents re-triggering the creator on the same accumulation.
    Returns the number of entries removed.
    """
    if not pattern:
        return 0
    path = log_path or DEFAULT_LOG_PATH
    if not os.path.exists(path):
        return 0
    removed = 0
    kept: list[str] = []
    try:
        with _log_lock:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        kept.append(line)  # preserve malformed lines unchanged
                        continue
                    if rec.get("pattern") == pattern:
                        removed += 1
                        continue
                    kept.append(line)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                for line in kept:
                    f.write(line + "\n")
            os.replace(tmp, path)
    except OSError as e:
        logger.debug("clear_pattern failed: %s", e)
        return 0
    return removed
