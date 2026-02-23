"""
GAIA MCP Web Tools — web_search and web_fetch

Read-only tools that give GAIA access to real, verifiable web sources.
Safety via domain allowlist + rate limits (not SENSITIVE_TOOLS).
"""

import json as json_mod
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
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
        # Authenticated domains: config-driven API key + header injection
        self.authenticated_domains: Dict[str, Dict] = cfg.get("authenticated_domains", {})
        # Build a flat allowlist (trusted + reliable + authenticated) for fetch gating
        self._allowlist = set(self.trusted) | set(self.reliable)
        for domain in self.authenticated_domains:
            self._allowlist.add(domain)

    # ---- public helpers ----

    def tier_for_domain(self, domain: str) -> str:
        """Return 'trusted', 'reliable', 'authenticated', 'blocked', or 'unknown'."""
        base = self._base_domain(domain)
        if base in self.trusted or domain in self.trusted:
            return "trusted"
        if base in self.reliable or domain in self.reliable:
            return "reliable"
        if self.auth_config_for_domain(domain) is not None:
            return "authenticated"
        if base in self.blocked or domain in self.blocked:
            return "blocked"
        return "unknown"

    def auth_config_for_domain(self, domain: str) -> Optional[Dict]:
        """Return auth config if domain matches an authenticated domain entry."""
        base = self._base_domain(domain)
        if domain in self.authenticated_domains:
            return self.authenticated_domains[domain]
        if base in self.authenticated_domains:
            return self.authenticated_domains[base]
        return None

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
            max_calls=rate.get("search_per_hour", 20),
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

def web_search(params: dict) -> dict:
    """Search the web via DuckDuckGo and return annotated results.

    Params:
        query (str, required): Search query.
        content_type (str, optional): Hint like 'poem', 'facts', 'code', 'science', 'news'.
            Automatically adds site: clauses for relevant domains.
        domain_filter (str, optional): Explicit domain to restrict results to.
        max_results (int, optional): Number of results (default 5, max 10).

    Returns dict with keys: ok, results (list of dicts), query, filtered_count.
    """
    query = (params.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

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
        results.append({
            "title": r.get("title", ""),
            "url": url,
            "snippet": r.get("body") or r.get("snippet", ""),
            "domain": domain,
            "trust_tier": trust.tier_for_domain(domain),
        })

    return {
        "ok": True,
        "query": effective_query,
        "results": results[:max_results],
        "result_count": len(results[:max_results]),
        "filtered_count": filtered_count,
    }


def _ddg_search(query: str, max_results: int) -> List[dict]:
    """Search DuckDuckGo using the library, falling back to Instant Answer API."""
    # Try ddgs library first (renamed from duckduckgo_search)
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if results:
            return results
    except ImportError:
        logger.info("ddgs library not available, trying duckduckgo_search fallback")
    except Exception as e:
        logger.warning("ddgs search failed: %s", e)

    # Try legacy duckduckgo_search library
    try:
        from duckduckgo_search import DDGS
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
    """Fetch and extract text from a URL.

    Supports allowlisted domains (trusted/reliable) and authenticated API
    domains configured in gaia_constants.json. Authenticated domains have
    their URLs rewritten to API endpoints and auth headers injected.

    Params:
        url (str, required): The URL to fetch.

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

    # Domain gating
    if trust.is_blocked(domain):
        return {
            "ok": False,
            "error": f"Domain '{domain}' is blocked. Cannot fetch content from this source.",
            "domain": domain,
            "trust_tier": "blocked",
        }

    # Check for authenticated domain (API key + header injection)
    auth_cfg = trust.auth_config_for_domain(domain)
    auth_headers: Dict[str, str] = {}

    if auth_cfg:
        key_env = auth_cfg.get("api_key_env", "")
        api_key = os.getenv(key_env, "").strip() if key_env else ""
        if not api_key:
            return {
                "ok": False,
                "error": (
                    f"Authenticated domain '{domain}' requires {key_env} "
                    "but it is not set or empty. Add it to .env and restart."
                ),
                "domain": domain,
                "trust_tier": "authenticated",
            }
        # Apply URL rewrites (e.g., app.kanka.io → api.kanka.io)
        url = _apply_url_rewrites(url, auth_cfg)
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Build auth headers
        header_name = auth_cfg.get("auth_header", "Authorization")
        header_value = auth_cfg.get("auth_format", "Bearer {key}").replace("{key}", api_key)
        auth_headers[header_name] = header_value
        for k, v in auth_cfg.get("extra_headers", {}).items():
            auth_headers[k] = v
    elif not trust.is_allowed(domain):
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
        headers = {"User-Agent": "GAIA-Research/1.0 (educational AI assistant)"}
        headers.update(auth_headers)
        resp = requests.get(url, timeout=_FETCH_TIMEOUT, headers=headers)
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

    body_text = raw_bytes.decode("utf-8", errors="replace")

    # JSON responses (typical for authenticated API domains) get formatted directly
    if "json" in content_type:
        try:
            data = json_mod.loads(body_text)
            title, text = _format_json_content(data)
        except Exception:
            title, text = "", body_text
    else:
        # HTML: extract clean text via trafilatura → BeautifulSoup → regex
        title, text = _extract_content(body_text, url)

    tier = "authenticated" if auth_cfg else trust.tier_for_domain(domain)
    return {
        "ok": True,
        "title": title,
        "content": text,
        "domain": domain,
        "trust_tier": tier,
        "bytes": len(text),
        "url": url,
    }


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
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
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
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    # Strip all tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    return title, text


def _extract_title_bs4(html: str) -> str:
    """Extract just the title via BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        return soup.title.string.strip() if soup.title and soup.title.string else ""
    except Exception:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return title_match.group(1).strip() if title_match else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_url_rewrites(url: str, auth_cfg: Dict) -> str:
    """Apply URL rewrite rules from an authenticated domain config.

    Returns the rewritten URL if a rule matches, or the original URL.
    Capture groups are referenced as {1}, {2}, etc. in the replacement.
    """
    for rule in auth_cfg.get("url_rewrites", []):
        pattern = rule.get("match", "")
        replacement = rule.get("replace", "")
        if not pattern:
            continue
        m = re.match(pattern, url)
        if m:
            result = replacement
            for i, group in enumerate(m.groups(), 1):
                result = result.replace(f"{{{i}}}", group)
            logger.info("URL rewrite: %s -> %s", url, result)
            return result
    return url


def _format_json_content(data: Any) -> Tuple[str, str]:
    """Format a JSON API response as readable text.

    Returns (title, formatted_text). Extracts title from common JSON
    patterns (data.name, data.title) and pretty-prints the payload.
    """
    title = ""
    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, dict):
            title = (
                inner.get("name")
                or inner.get("title")
                or inner.get("label")
                or ""
            )
    formatted = json_mod.dumps(data, indent=2, ensure_ascii=False, default=str)
    return title, formatted


def _domain_from_url(url: str) -> str:
    """Extract domain from a URL string."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""
