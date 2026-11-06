"""What she looked up a moment ago, for the turns that ask about it (bxzc).

"Look up some news" ran a search; "summarize the top 3 stories" and "which
outlet put out these articles?" came after it, read as chat, and were
answered from nothing: she made up three stories, then gave them the
outlets and dates of an unrelated second search (2026-11-06). A turn that
asks about what she found is answered from what that search returned, as a
tool that already ran.
"""
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.EarlierLookup")

META_KEY = "last_lookup"
MAX_AGE_S = 30 * 60
_MAX_ROWS = 8

# Asks about what was found. Not "the story of King Arthur": only words that
# point back at results ("these articles", "top 3", "which outlet").
_REFERS_RE = re.compile(r"""
    \b(?:these|those|them|that|your)\s+(?:\w+\s+){0,2}?
        (?:stories|story|articles?|headlines?|results?|links?|pieces?|sources?|items?|pages?|news|findings)\b
  | \btop\s+(?:\d+|one|two|three|four|five|ten)\b
  | \b(?:which|what)\s+(?:news\s+)?(?:outlets?|sites?|websites?|publications?|papers?|sources?)\b
  | \bwhat\s+did\s+you\s+find\b
  | \bdid\s+you\s+find\s+anything\b
  | \bsummari[sz]e\s+(?:them|those|these|it|that)\b
  | \bwhen\s+were\s+(?:they|these|those)\s+(?:published|written|posted)\b
""", re.IGNORECASE | re.VERBOSE)


def refers_to_results(text: str) -> bool:
    return bool(_REFERS_RE.search(text or ""))


def remember(session_manager, session_id: str, query: str, results: List[Dict[str, Any]], note: str = "") -> None:
    """Keep a search's results for the turns after it."""
    rows = []
    for r in results or []:
        if isinstance(r, dict):
            rows.append({"title": str(r.get("title") or "")[:200],
                         "url": str(r.get("url") or r.get("href") or "")[:300],
                         "snippet": str(r.get("snippet") or r.get("body") or "")[:600]})
    try:
        session_manager.set_session_meta(session_id, META_KEY, {
            "query": (query or "")[:200], "results": rows[:_MAX_ROWS], "ts": time.time(), "note": note[:300]})
    except Exception:
        logger.debug("Earlier lookup: could not keep results", exc_info=True)


def recall(session_manager, session_id: str, text: str, now: Optional[float] = None) -> Optional[dict]:
    """The recent search this turn asks about, or None."""
    if not refers_to_results(text):
        return None
    try:
        lookup = session_manager.get_session_meta(session_id, META_KEY)
    except Exception:
        return None
    if not isinstance(lookup, dict) or not lookup.get("results"):
        return None
    if (now or time.time()) - float(lookup.get("ts") or 0) > MAX_AGE_S:
        return None
    return lookup


def read_at(ts: float) -> str:
    """When the results were read, in the operator's time: with none given,
    she dated BBC headlines "November 11, 2020" (2026-11-06)."""
    import os
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        when = datetime.fromtimestamp(ts, ZoneInfo(os.environ.get("GAIA_USER_TZ", "America/Los_Angeles")))
    except Exception:
        when = datetime.fromtimestamp(ts)
    return when.strftime("%A, %B %-d, %Y at %-I:%M %p")


def as_tool_result(lookup: dict) -> dict:
    """The packet's tool_result for a turn answered from the earlier search."""
    query = lookup.get("query") or ""
    label = f'your web search for "{query}" a moment ago' if query else "your web search a moment ago"
    note = " ".join(x for x in (lookup.get("note") or "",
                                f"Read {read_at(float(lookup.get('ts') or time.time()))}.") if x)
    return {"tool": label, "params": {}, "success": True,
            "output": {"query": query, "note": note, "results": lookup.get("results") or []}}


# Front pages she can read for headlines. The ones a search returns are
# mostly off her fetch allowlist (NBC, CBS, Yahoo); AP and Reuters refuse
# plain fetches; BBC's front page gives ten (2026-11-06).
NEWS_FRONT_PAGES = ("https://www.bbc.com/news",)


def _mcp(method: str, params: dict) -> dict:
    from gaia_core.utils.mcp_client import call_jsonrpc
    r = call_jsonrpc(method, params) or {}
    return r.get("response", {}).get("result", r) if isinstance(r, dict) else {}


def headlines_for(results: List[Dict[str, Any]], call=_mcp) -> Optional[dict]:
    """When a search found only news sites' front pages, the stories on one:
    {"source", "url", "results"}. None if the results are stories already or
    no front page could be read."""
    from gaia_common.utils.tool_call_parser import front_pages_only
    if not front_pages_only(results):
        return None
    urls = [str(r.get("url") or r.get("href") or "") for r in results if isinstance(r, dict)]
    for url in [u for u in urls if u] + [u for u in NEWS_FRONT_PAGES if u not in urls]:
        try:
            page = call("web_fetch", {"url": url, "headlines": True}) or {}
        except Exception:
            continue
        found = page.get("headlines") or []
        if page.get("ok") and len(found) >= 3:
            # "BBC News - Breaking news, video and ..." -> "BBC News"
            source = re.split(r"\s+[-|–]\s+", page.get("title") or "")[0].strip() or url
            logger.info("Earlier lookup: front pages only; read %d headlines from %s", len(found), url)
            # Headline first, its summary after; the outlet is the source.
            return {"source": source, "url": url, "results": [
                {"title": source, "url": h.get("url", ""),
                 "snippet": " — ".join(x for x in (h.get("title", ""), h.get("snippet", "")) if x)}
                for h in found]}
    return None
