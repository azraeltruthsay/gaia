"""MemPalace integration for skills (GAIA_Project-a3i).

Today the skills system has two stores with no atomic link:
  - SKILL.md files on disk at /knowledge/skills/<name>/SKILL.md
  - utility_scores.json at /shared/knowledge_router/utility_scores.json

Neither remembers WHICH skill was used to solve WHAT in past sessions.
"How did I solve X last week?" can't surface the relevant skill.

This module bridges that gap: when a skill loads or learn() records an
outcome, a corresponding memory lands in MemPalace alongside other
persistent facts. MemPalace's recall surface then includes skill
events, so cross-session lookup works.

Three entry points:

  record_skill_loaded(name, description, version) — on skill load,
      store "skill:<name> handles <description> (v<version>)" so
      future searches for the description match.

  record_skill_outcome(name, intent, success, query=None) — on
      learn() outcomes, store "skill:<name> succeeded on '<query>'
      (intent=<intent>)" or "...failed on...". Outcome events get
      tagged so analytics can separate skill-event memories from
      regular ones.

  find_skills_for_topic(query, limit=5) — MemPalace recall filtered
      to skill-tagged entries, returning skill names sorted by
      recall relevance.

Each function is defensive — MemPalace failure (uninitialized config,
missing wing, IO error) is logged at DEBUG and returns gracefully.
The skills system itself is unaffected if the palace is offline.

The MemPalace instance is a lazy singleton — first call instantiates,
subsequent calls reuse. Tests can inject a mock via reset_for_tests.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger("GAIA.SkillPalace")


_palace_instance = None
_lock = threading.Lock()


def _get_palace():
    """Lazy-init MemPalace. Returns the instance or None on failure.

    Subsequent failures (after a successful init) propagate as None
    only when the cached instance was never built — once cached, we
    trust it. Tests can clear the cache via reset_for_tests.
    """
    global _palace_instance
    if _palace_instance is not None:
        return _palace_instance
    with _lock:
        if _palace_instance is not None:
            return _palace_instance
        try:
            from gaia_common.config import Config
            from gaia_common.utils.mempalace import MemPalace
            conf = Config()
            palace_config = conf.constants.get("MEMPALACE", {})
            if not palace_config:
                logger.debug("MEMPALACE section absent from constants; skill palace disabled")
                return None
            _palace_instance = MemPalace(palace_config)
            logger.info("Skill palace bridge initialized")
        except Exception as e:
            logger.debug("MemPalace init failed; skill palace disabled: %s", e)
            _palace_instance = None
    return _palace_instance


def reset_for_tests(palace=None) -> None:
    """Replace or clear the cached MemPalace. Tests only."""
    global _palace_instance
    with _lock:
        _palace_instance = palace


# ── Write-side: skill events → memories ──────────────────────────────


def record_skill_loaded(
    name: str,
    description: str,
    *,
    version: int = 1,
    source_path: Optional[str] = None,
) -> bool:
    """Record that a skill was loaded into the registry.

    Stores "skill:<name> handles <description> (v<version>)" as a
    memory. Returns True if the memory landed, False on any error
    (logged at DEBUG; never raises). Idempotent at the MemPalace
    layer — duplicate-prevention is the palace's job.
    """
    if not name or not description:
        return False
    palace = _get_palace()
    if palace is None:
        return False
    try:
        text = f"skill:{name} handles {description} (v{version})"
        source = f"skill_load:{source_path or name}"
        palace.store(text, source=source)
        logger.debug("Skill loaded → palace: %s", name)
        return True
    except Exception as e:
        logger.debug("record_skill_loaded failed for %s: %s", name, e)
        return False


def record_skill_outcome(
    name: str,
    *,
    intent: Optional[str] = None,
    success: bool = True,
    query: Optional[str] = None,
) -> bool:
    """Record a skill execution outcome from learn().

    Stores a tagged memory like:
      "skill:<name> succeeded on '<query>' (intent=<intent>)"

    The "skill:<name>" prefix is what find_skills_for_topic matches
    against. Returns True on success, False on any error (logged at
    DEBUG; never raises).
    """
    if not name:
        return False
    palace = _get_palace()
    if palace is None:
        return False
    try:
        verb = "succeeded" if success else "failed"
        text_parts = [f"skill:{name} {verb}"]
        if query:
            text_parts.append(f"on '{query}'")
        if intent:
            text_parts.append(f"(intent={intent})")
        text = " ".join(text_parts)
        source = f"skill_outcome:{name}:{verb}"
        palace.store(text, source=source)
        logger.debug("Skill outcome → palace: %s %s", name, verb)
        return True
    except Exception as e:
        logger.debug("record_skill_outcome failed for %s: %s", name, e)
        return False


# ── Read-side: query past skill usage ────────────────────────────────


def find_skills_for_topic(query: str, *, limit: int = 5) -> list[str]:
    """Search MemPalace for skill events matching a topic.

    Returns a list of skill names (dedup'd, recall-order) whose
    associated memories matched the query. Empty list on any failure
    or when the palace has no relevant entries.

    Uses MemPalace.recall — which performs full-text or AAAK-based
    matching depending on palace configuration. The "skill:" prefix
    in the stored text gives us a clean filter.
    """
    if not query:
        return []
    palace = _get_palace()
    if palace is None:
        return []
    try:
        hits = palace.recall(query, limit=limit * 3) if hasattr(palace, "recall") else []
    except Exception as e:
        logger.debug("find_skills_for_topic recall failed: %s", e)
        return []

    seen: set[str] = set()
    out: list[str] = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        text = (hit.get("text") or hit.get("body") or hit.get("compressed") or "")
        if not isinstance(text, str):
            continue
        # Look for the "skill:<name>" prefix anywhere in the text
        idx = text.find("skill:")
        if idx < 0:
            continue
        # Take the token between "skill:" and the next space / colon
        # (skill names are snake-case alphanumeric).
        tail = text[idx + len("skill:"):]
        end = len(tail)
        for i, ch in enumerate(tail):
            if not (ch.isalnum() or ch in "_-"):
                end = i
                break
        skill_name = tail[:end].strip()
        if skill_name and skill_name not in seen:
            seen.add(skill_name)
            out.append(skill_name)
            if len(out) >= limit:
                break
    return out
