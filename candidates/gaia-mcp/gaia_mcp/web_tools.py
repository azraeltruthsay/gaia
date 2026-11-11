"""
GAIA MCP Web Tools — web_search and web_fetch

Read-only tools that give GAIA access to real, verifiable web sources.
Safety via domain allowlist + rate limits (not SENSITIVE_TOOLS).
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from gaia_common.config import Config

logger = logging.getLogger("GAIA.WebTools")

# ---------------------------------------------------------------------------
# Source Trust Configuration
# ---------------------------------------------------------------------------

class SourceTrustConfig:
    """Tiered domain classification: trusted / reliable / blocked / unknown.

    Loads overrides from gaia_constants.json["WEB_RESEARCH"] if available,
    otherwise falls back to hardcoded defaults.
    """

    _DEFAULT_TRUSTED = [
        "gutenberg.org",
        "poetryfoundation.org",
        "poets.org",
        "britannica.com",
        "wikipedia.org",
        "en.wikisource.org",
        "arxiv.org",
        "docs.python.org",
        "developer.mozilla.org",
        "rust-lang.org",
        "cppreference.com",
        "ocw.mit.edu",
        "mitocw.mit.edu",
    ]

    _DEFAULT_RELIABLE = [
        "github.com",
        "stackoverflow.com",
        "stackexchange.com",
        "bbc.com",
        "reuters.com",
        "apnews.com",
        "nature.com",
        "science.org",
        "ncbi.nlm.nih.gov",
        # Test endpoints (used by smoke tests and during development)
        "httpbin.org",
    ]

    _DEFAULT_BLOCKED = [
        "reddit.com",
        "4chan.org",
        "twitter.com",
        "x.com",
        "facebook.com",
        "tiktok.com",
        "instagram.com",
    ]

    _DEFAULT_CONTENT_TYPE_SOURCES = {
        "poem": ["gutenberg.org", "poetryfoundation.org", "poets.org", "en.wikisource.org"],
        "facts": ["britannica.com", "wikipedia.org"],
        "code": ["github.com", "docs.python.org", "developer.mozilla.org", "stackoverflow.com"],
        "science": ["arxiv.org", "nature.com", "science.org", "ncbi.nlm.nih.gov"],
        "news": ["bbc.com", "reuters.com", "apnews.com"],
    }

    def __init__(self, config_override: Optional[Dict] = None):
        cfg = config_override or {}
        self.trusted: List[str] = cfg.get("trusted_domains", self._DEFAULT_TRUSTED)
        self.reliable: List[str] = cfg.get("reliable_domains", self._DEFAULT_RELIABLE)
        self.blocked: List[str] = cfg.get("blocked_domains", self._DEFAULT_BLOCKED)
        self.content_type_sources: Dict[str, List[str]] = cfg.get(
            "content_type_sources", self._DEFAULT_CONTENT_TYPE_SOURCES
        )
        # Build a flat allowlist (trusted + reliable) for fetch gating
        self._allowlist = set(self.trusted) | set(self.reliable)

    # ---- public helpers ----

    def tier_for_domain(self, domain: str) -> str:
        """Return 'trusted', 'reliable', 'blocked', or 'unknown'."""
        base = self._base_domain(domain)
        if base in self.trusted or domain in self.trusted:
            return "trusted"
        if base in self.reliable or domain in self.reliable:
            return "reliable"
        if base in self.blocked or domain in self.blocked:
            return "blocked"
        return "unknown"

    def is_allowed(self, domain: str) -> bool:
        """True if domain is trusted or reliable (fetchable)."""
        base = self._base_domain(domain)
        return base in self._allowlist or domain in self._allowlist

    def is_blocked(self, domain: str) -> bool:
        base = self._base_domain(domain)
        return base in self.blocked or domain in self.blocked

    def domains_for_content_type(self, content_type: str) -> List[str]:
        return self.content_type_sources.get(content_type, [])

    # ---- internals ----

    @staticmethod
    def _base_domain(domain: str) -> str:
        """Strip subdomains: 'en.wikipedia.org' → 'wikipedia.org'."""
        parts = domain.lower().split(".")
        if len(parts) > 2:
            return ".".join(parts[-2:])
        return domain.lower()


# ---------------------------------------------------------------------------
# Rate Limiter (in-memory sliding window)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple per-tool sliding-window rate limiter.

    Resets on container restart (in-memory only — no Redis needed).
    """

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self.max_calls:
            return False
        self._timestamps.append(now)
        return True

    @property
    def remaining(self) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        active = [t for t in self._timestamps if t > cutoff]
        return max(0, self.max_calls - len(active))


# ---------------------------------------------------------------------------
# Module-level singletons (lazy init)
# ---------------------------------------------------------------------------

_trust_config: Optional[SourceTrustConfig] = None
_search_limiter: Optional[_RateLimiter] = None
_fetch_limiter: Optional[_RateLimiter] = None


def _get_config() -> Tuple[SourceTrustConfig, _RateLimiter, _RateLimiter]:
    global _trust_config, _search_limiter, _fetch_limiter
    if _trust_config is None:
        try:
            cfg = Config()
            web_cfg = cfg.constants.get("WEB_RESEARCH", {})
        except Exception:
            web_cfg = {}

        _trust_config = SourceTrustConfig(web_cfg)

        rate = web_cfg.get("rate_limits", {})
        _search_limiter = _RateLimiter(
            max_calls=rate.get("search_per_hour", 100),
            window_seconds=3600,
        )
        _fetch_limiter = _RateLimiter(
            max_calls=rate.get("fetch_per_hour", 50),
            window_seconds=3600,
        )
    return _trust_config, _search_limiter, _fetch_limiter


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def _remember_lookup(params: dict, query: str, results: List[dict], kind: str) -> None:
    """0vuz: keep what was just looked up in the palace's lookup memory,
    verbatim and with provenance, so GAIA can recall and cite it later
    without searching again. Best-effort — a memory failure is logged and
    never fails the lookup itself. Callers can opt out with remember=False
    (e.g. health probes)."""
    if params.get("remember") is False or not results:
        return
    try:
        from gaia_mcp.tools import _get_palace  # lazy: tools imports this module
        _get_palace().lookups.remember(query, results, kind=kind)
    except Exception as e:
        logger.warning("web %s: could not remember lookup for %r: %s", kind, query[:80], e)


def web_search(params: dict) -> dict:
    """Search the web via DuckDuckGo and return annotated results.

    Params:
        query (str, required): Search query.
        content_type (str, optional): Hint like 'poem', 'facts', 'code', 'science', 'news'.
            Automatically adds site: clauses for relevant domains.
        domain_filter (str, optional): Explicit domain to restrict results to.
        max_results (int, optional): Number of results (default 5, max 10).
        recent (str, optional): 'd', 'w', 'm' or 'y': news from the last day,
            week, month or year, dated. "Latest news on fusion energy" searched
            plain "fusion energy" and she answered from Wikipedia's "As of 2025".

    Returns dict with keys: ok, results (list of dicts), query, filtered_count.
    """
    query = (params.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    # Security: reject queries that leak internal system state to external search
    _internal_markers = [
        "gaia-core", "gaia-mcp", "gaia-prime", "gaia-web", "gaia-study",
        "gaia-orchestrator", "gaia-doctor", "gaia-monkey", "gaia-audio",
        "restart loop", "recursive restart", "system repair", "root cause",
        "traceback", "stack trace", "stderr", "engine-manager",
        "/shared/", "/knowledge/", "/models/", "docker exec",
        "HTTP/1.1\" 200", "HTTP/1.1\" 500",
    ]
    _query_lower = query.lower()
    if any(marker.lower() in _query_lower for marker in _internal_markers):
        logger.warning("web_search: BLOCKED — query contains internal system state: %s", query[:100])
        return {
            "ok": False,
            "error": "Query contains internal system information that should not be sent to external search engines.",
            "blocked_reason": "internal_state_leak",
        }

    # Sanity: reject excessively long queries (likely dumped logs)
    if len(query) > 500:
        logger.warning("web_search: query too long (%d chars), truncating", len(query))
        query = query[:200]

    content_type = params.get("content_type")
    domain_filter = params.get("domain_filter")
    max_results = min(int(params.get("max_results", 5)), 10)

    trust, search_lim, _ = _get_config()

    if not search_lim.allow():
        return {
            "ok": False,
            "error": "Rate limit exceeded for web_search. Try again later.",
            "remaining": search_lim.remaining,
        }

    # Build the effective query with site: clauses
    effective_query = query
    if domain_filter:
        effective_query = f"site:{domain_filter} {query}"
    elif content_type:
        domains = trust.domains_for_content_type(content_type)
        if domains:
            site_clause = " OR ".join(f"site:{d}" for d in domains[:4])
            effective_query = f"({site_clause}) {query}"

    recent = str(params.get("recent") or "").strip().lower()[:1]
    raw_results = _ddg_news(effective_query, max_results, recent) if recent in _TIMELIMITS else []
    if not raw_results:
        # Try duckduckgo_search library first, then fallback to Instant Answer API
        raw_results = _ddg_search(effective_query, max_results)

    # Filter out blocked domains and annotate with trust tier
    results = []
    filtered_count = 0
    for r in raw_results:
        url = r.get("href") or r.get("url", "")
        domain = _domain_from_url(url)
        if trust.is_blocked(domain):
            filtered_count += 1
            continue
        snippet = mark_upcoming(r.get("body") or r.get("snippet", ""))
        if r.get("date"):   # news: when, and who said it, read with the words
            snippet = f"{str(r['date'])[:10]}, {r.get('source') or domain}: {snippet}"
        results.append({
            "title": r.get("title", ""),
            "url": url,
            "snippet": snippet,
            "domain": domain,
            "trust_tier": trust.tier_for_domain(domain),
        })

    # Remembered under the question as asked, not the site:-decorated query.
    _remember_lookup(params, query, results[:max_results], "search")

    return {
        "ok": True,
        "query": effective_query,
        "results": results[:max_results],
        "result_count": len(results[:max_results]),
        "filtered_count": filtered_count,
    }


_ddgs_patched = False


def _patch_ddgs_text_joining() -> None:
    """Fix ddgs gluing words together in every result snippet.

    ddgs (9.x) BaseSearchEngine.extract_results strips each HTML text node
    and joins them with "", so the engine's <b>highlighted</b> query terms
    lose the spaces around them: "<b>Excalibur</b> is the sword of <b>King
    Arthur</b>" came back "Excaliburis the sword ofKingArthur". Every web
    result GAIA read looked like that, and lookup memory (0vuz) couldn't
    match "Arthur" or "Excalibur" in it. The text nodes already carry their
    own spaces, so join them raw and collapse whitespace instead.
    Best-effort: if ddgs changes shape, searches still run unpatched.
    """
    global _ddgs_patched
    if _ddgs_patched:
        return
    _ddgs_patched = True
    try:
        from ddgs.base import BaseSearchEngine

        def extract_results(self, html_text):
            html_text = self.pre_process_html(html_text)
            tree = self.extract_tree(html_text)
            results = []
            for item in tree.xpath(self.items_xpath):
                result = self.result_type()
                for key, value in self.elements_xpath.items():
                    result.__setattr__(key, " ".join("".join(item.xpath(value)).split()))
                results.append(result)
            return results

        BaseSearchEngine.extract_results = extract_results
        logger.info("web_search: patched ddgs snippet text joining (words were glued at highlights)")
    except Exception as e:
        logger.warning("web_search: could not patch ddgs text joining; snippets may have glued words: %s", e)


def _absolute_results(results) -> List[dict]:
    """Only results whose link can actually be fetched."""
    return [r for r in results if (r.get("href") or r.get("url") or "").startswith(("http://", "https://"))]


_TIMELIMITS = ("d", "w", "m", "y")

_MONTHS = {m: i for i, names in enumerate((
    ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
    ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
    ("nov", "november"), ("dec", "december")), 1) for m in names}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "November 13", "Nov. 13, 2026", "13 November 2026", "2026-11-13"
_DATE_RE = re.compile(
    rf"\b(?:(?P<m1>{_MON})\.?\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y1>\d{{4}}))?"
    rf"|(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m2>{_MON})\.?(?:,?\s+(?P<y2>\d{{4}}))?"
    rf"|(?P<y3>\d{{4}})-(?P<m3>\d{{2}})-(?P<d3>\d{{2}}))\b", re.IGNORECASE)


def mark_upcoming(text: str, today=None) -> str:
    """Dates after today marked "(upcoming)". Read "SpaceX will attempt a
    double catch on November 13", she said on the 11th that it "attempted"
    one (sn26). A date with no year is taken as this year."""
    import datetime as _dt
    today = today or _dt.date.today()

    def _mark(m):
        try:
            if m.group("y3"):
                when = _dt.date(int(m.group("y3")), int(m.group("m3")), int(m.group("d3")))
            else:
                month = _MONTHS[(m.group("m1") or m.group("m2")).lower()]
                day = int(m.group("d1") or m.group("d2"))
                when = _dt.date(int(m.group("y1") or m.group("y2") or today.year), month, day)
        except (ValueError, KeyError):
            return m.group(0)
        return f"{m.group(0)} (upcoming)" if when > today else m.group(0)

    return _DATE_RE.sub(_mark, text or "")


def _ddg_news(query: str, max_results: int, timelimit: str) -> List[dict]:
    """Dated news articles, newest first; empty if the news search fails."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            found = _absolute_results(ddgs.news(query, max_results=max_results, timelimit=timelimit))
        return sorted(found, key=lambda r: str(r.get("date") or ""), reverse=True)
    except Exception as e:
        logger.warning("ddgs news search failed (%s); plain search instead", e)
        return []


def _ddg_search(query: str, max_results: int) -> List[dict]:
    """Search DuckDuckGo using the library, falling back to Instant Answer API."""
    # Try ddgs library first (renamed from duckduckgo_search)
    _patch_ddgs_text_joining()
    try:
        from ddgs import DDGS
        # ddgs picks engines at random; its Google results can be relative
        # redirects ("/url?opi=...") that nothing can fetch. Ask once more
        # rather than hand those on (yx93).
        for _ in range(2):
            with DDGS() as ddgs:
                results = _absolute_results(ddgs.text(query, max_results=max_results))
            if results:
                return results
    except ImportError:
        logger.info("ddgs library not available, trying duckduckgo_search fallback")
    except Exception as e:
        logger.warning("ddgs search failed: %s", e)

    # Try legacy duckduckgo_search library
    try:
        from duckduckgo_search import DDGS  # type: ignore[assignment]
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if results:
            return results
    except ImportError:
        logger.info("duckduckgo_search library not available, trying API fallback")
    except Exception as e:
        logger.warning("duckduckgo_search failed: %s", e)

    # Fallback: DDG Instant Answer API (limited but zero-dependency)
    try:
        import requests
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for topic in data.get("RelatedTopics", []):
            if "FirstURL" in topic:
                results.append({
                    "title": topic.get("Text", "")[:120],
                    "href": topic.get("FirstURL", ""),
                    "body": topic.get("Text", ""),
                })
        if data.get("AbstractURL"):
            results.insert(0, {
                "title": data.get("Heading", ""),
                "href": data.get("AbstractURL", ""),
                "body": data.get("AbstractText", ""),
            })
        return results[:max_results]
    except Exception as e:
        logger.warning("DDG API fallback failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

_MAX_CONTENT_BYTES = 500 * 1024  # 500KB
_FETCH_TIMEOUT = 15  # seconds


def web_fetch(params: dict) -> dict:
    """Fetch and extract text from a URL (allowlisted domains only).

    Params:
        url (str, required): The URL to fetch.
        keep_breaks (bool, optional): Return the whole page's text with its
            line and stanza breaks, for reading a poem or speech word for word.

    Returns dict with keys: ok, title, content, domain, trust_tier, bytes.
    """
    url = (params.get("url") or "").strip()
    if not url:
        raise ValueError("url is required")

    # Validate URL structure
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    domain = parsed.netloc.lower()
    if not domain:
        raise ValueError("Could not parse domain from URL")

    trust, _, fetch_lim = _get_config()

    # Domain gating: only fetch from allowlisted domains
    if trust.is_blocked(domain):
        return {
            "ok": False,
            "error": f"Domain '{domain}' is blocked. Cannot fetch content from this source.",
            "domain": domain,
            "trust_tier": "blocked",
        }
    if not trust.is_allowed(domain):
        return {
            "ok": False,
            "error": (
                f"Domain '{domain}' is not in the allowlist. "
                "Use web_search first to find content from trusted sources, "
                "then fetch from those results."
            ),
            "domain": domain,
            "trust_tier": "unknown",
        }

    if not fetch_lim.allow():
        return {
            "ok": False,
            "error": "Rate limit exceeded for web_fetch. Try again later.",
            "remaining": fetch_lim.remaining,
        }

    # Fetch the page
    try:
        import requests
        resp = requests.get(
            url,
            timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": "GAIA-Research/1.0 (educational AI assistant)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"Failed to fetch URL: {e}", "domain": domain}

    raw_bytes = resp.content
    if len(raw_bytes) > _MAX_CONTENT_BYTES:
        raw_bytes = raw_bytes[:_MAX_CONTENT_BYTES]

    content_type = resp.headers.get("Content-Type", "")
    if "text" not in content_type and "html" not in content_type and "json" not in content_type:
        return {
            "ok": False,
            "error": f"Non-text content type: {content_type}",
            "domain": domain,
        }

    html = raw_bytes.decode("utf-8", errors="replace")

    if params.get("headlines"):
        # bxzc: a news site's front page, read for its stories. Its text is
        # menus and blurbs; its links to articles are the headlines.
        found = _headlines(html, url)
        return {"ok": True, "title": _extract_content(html, url)[0], "url": url,
                "domain": domain, "headlines": found, "count": len(found)}

    # Extract clean text via trafilatura → BeautifulSoup → regex fallback
    title, text = _extract_content(html, url)

    tier = trust.tier_for_domain(domain)
    _remember_lookup(params, f"{url} {title}".strip(), [{
        "title": title, "url": url, "snippet": text,
        "domain": domain, "trust_tier": tier,
    }], "fetch")
    # After remembering: lookups keep the article, not the page's menus.
    if params.get("keep_breaks"):
        text = _text_with_breaks(html) or text
    return {
        "ok": True,
        "title": title,
        "content": text,
        "domain": domain,
        "trust_tier": tier,
        "source_type": "trusted" if tier == "trusted" else "verified",
        "bytes": len(text),
        "word_count": len(text.split()),
    }


_BLOCK_TAGS = frozenset("""
address article aside blockquote center dd details div dl dt figcaption figure
form h1 h2 h3 h4 h5 h6 hr li main ol p pre section table tbody td th thead tr ul
""".split())


def _text_with_breaks(html: str) -> str:
    """A page's text with its breaks: one line per line, a blank line between
    stanzas. trafilatura flattens these, and sites mark a stanza in different
    ways: a <p> of <br> lines, a <p> or <div> holding only <br>, or a blank
    line inside pre-formatted text."""
    try:
        from bs4 import BeautifulSoup, NavigableString, Tag
    except ImportError:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    lines = []
    current = []

    def end_line():
        text = " ".join("".join(current).replace("\xa0", " ").split())
        current.clear()
        if text:
            lines.append(text)

    def blank():
        end_line()
        if lines and lines[-1] != "":
            lines.append("")

    def walk(node):
        for child in node.children:
            if isinstance(child, NavigableString):
                if type(child) is NavigableString:
                    parts = re.split(r"\n[ \t\xa0]*\n", str(child))
                    for i, part in enumerate(parts):
                        if i:
                            blank()
                        pieces = part.split("\n") if _is_preformatted(node) else [part]
                        for j, piece in enumerate(pieces):
                            if j:
                                end_line()
                            current.append(piece)
            elif isinstance(child, Tag):
                if child.name == "br":
                    end_line()
                elif child.name in _BLOCK_TAGS:
                    end_line()
                    before = len(lines)
                    walk(child)
                    had_br = child.find("br") is not None
                    end_line()
                    if not child.get_text(strip=True).replace("\xa0", ""):
                        if had_br:
                            blank()
                    elif had_br and len(lines) - before > 1 and child.name in ("p", "blockquote", "div"):
                        blank()
                else:
                    walk(child)

    walk(soup.body or soup)
    end_line()
    return "\n".join(lines).strip()


def _is_preformatted(node) -> bool:
    style = (node.get("style") or "") if hasattr(node, "get") else ""
    return getattr(node, "name", "") == "pre" or "pre-" in style or "pre;" in style


def _headlines(html: str, url: str, limit: int = 10) -> list:
    """Links on a page that read like stories: a few words of text, pointing
    at an article on the same site (an id, a date or a long slug in the path),
    not a section, a login or a video."""
    from urllib.parse import urljoin, urlparse
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    site = ".".join((urlparse(url).hostname or "").split(".")[-2:])
    out, seen = [], set()
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        whole = " ".join(a.get_text(" ").split())
        # A card's link holds its heading and its summary: the heading is the headline.
        heading = a.find(["h1", "h2", "h3", "h4"])
        text = " ".join(heading.get_text(" ").split()) if heading else whole
        text = re.sub(r"^\d{1,2}\s+", "", text)  # "4 US midterm..." from a most-read list
        if not (25 <= len(text) <= 180) or len(text.split()) < 5:
            continue
        summary = whole[len(text):].strip() if heading and whole.startswith(text) else ""
        link = urljoin(url, a["href"]).split("#")[0]
        parts = urlparse(link)
        if not (parts.hostname or "").endswith(site) or link in seen or text in seen:
            continue
        path = parts.path.lower()
        if any(w in path for w in ("/video", "/live", "/login", "/subscribe", "/account", "/newsletter",
                                   "cookie", "privacy", "/usingthe")):
            continue
        last = [seg for seg in path.split("/") if seg][-1:] or [""]
        # An article: an id ("1954155", "c0lkz4e3rmxo"), a dated path, or a long slug.
        if not (re.search(r"\d{4,}|/20\d\d/", path) or last[0].count("-") >= 3
                or re.fullmatch(r"(?=[a-z0-9]*\d)[a-z0-9]{8,}", last[0])):
            continue
        seen.update((link, text))
        out.append({"title": text, "url": link, **({"snippet": summary[:300]} if summary else {})})
        if len(out) >= limit:
            break
    return out


def _extract_content(html: str, url: str) -> Tuple[str, str]:
    """Extract title and main text from HTML.

    Priority: trafilatura → BeautifulSoup → regex fallback.
    """
    title = ""
    text = ""

    # Try trafilatura (best for article extraction)
    try:
        import trafilatura
        result = trafilatura.extract(html, url=url, include_comments=False)
        if result and len(result) > 50:
            text = result
            # trafilatura doesn't always extract title; try BS4 for just the title
            title = _extract_title_bs4(html)
            return title, text
    except ImportError:
        pass
    except Exception as e:
        logger.debug("trafilatura extraction failed: %s", e)

    # Try BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        title = _extract_title_bs4(html)
        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if text and len(text) > 50:
            return title, text
    except ImportError:
        pass
    except Exception as e:
        logger.debug("BeautifulSoup extraction failed: %s", e)

    # Regex fallback (last resort)
    title = _title_regex(html)
    # Strip all tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    return title, text


def _extract_title_bs4(html: str) -> str:
    """The page's title: <title>, else og:title, else the first <h1>.

    ecnc: httpbin's /html has no <title>; its name is the <h1> "Herman
    Melville - Moby-Dick", which trafilatura drops from the text. Without
    it Core was asked to name the work from the prose alone."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string and soup.title.string.strip():
            return soup.title.string.strip()
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and (og.get("content") or "").strip():
            return og["content"].strip()
        h1 = soup.find("h1")
        return " ".join(h1.get_text(" ", strip=True).split()) if h1 else ""
    except Exception as e:
        logger.debug("BeautifulSoup title extraction failed (%s); using regex", e)
        return _title_regex(html)


def _title_regex(html: str) -> str:
    """Regex fallback for _extract_title_bs4: <title>, else the first <h1>."""
    for pattern in (r"<title[^>]*>(.*?)</title>", r"<h1[^>]*>(.*?)</h1>"):
        m = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if m:
            text = re.sub(r"<[^>]+>", " ", m.group(1))
            text = " ".join(text.split())
            if text:
                return text
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain_from_url(url: str) -> str:
    """Extract domain from a URL string."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""
