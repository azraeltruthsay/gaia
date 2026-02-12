"""Tests for GAIA MCP web_search and web_fetch tools."""

import time
from unittest.mock import patch, MagicMock

import pytest

from gaia_mcp.web_tools import (
    SourceTrustConfig,
    _RateLimiter,
    web_search,
    web_fetch,
    _domain_from_url,
)


# ---------------------------------------------------------------------------
# SourceTrustConfig
# ---------------------------------------------------------------------------

class TestSourceTrustConfig:

    def test_default_tiers(self):
        cfg = SourceTrustConfig()
        assert cfg.tier_for_domain("wikipedia.org") == "trusted"
        assert cfg.tier_for_domain("github.com") == "reliable"
        assert cfg.tier_for_domain("reddit.com") == "blocked"
        assert cfg.tier_for_domain("example.com") == "unknown"

    def test_subdomain_matching(self):
        cfg = SourceTrustConfig()
        assert cfg.tier_for_domain("en.wikipedia.org") == "trusted"
        assert cfg.tier_for_domain("m.wikipedia.org") == "trusted"
        assert cfg.tier_for_domain("old.reddit.com") == "blocked"
        assert cfg.tier_for_domain("docs.github.com") == "reliable"

    def test_is_allowed(self):
        cfg = SourceTrustConfig()
        assert cfg.is_allowed("wikipedia.org") is True
        assert cfg.is_allowed("github.com") is True
        assert cfg.is_allowed("reddit.com") is False
        assert cfg.is_allowed("example.com") is False

    def test_is_blocked(self):
        cfg = SourceTrustConfig()
        assert cfg.is_blocked("reddit.com") is True
        assert cfg.is_blocked("x.com") is True
        assert cfg.is_blocked("wikipedia.org") is False

    def test_custom_config_overrides(self):
        custom = {
            "trusted_domains": ["mysite.org"],
            "reliable_domains": ["helper.com"],
            "blocked_domains": ["evil.net"],
        }
        cfg = SourceTrustConfig(custom)
        assert cfg.tier_for_domain("mysite.org") == "trusted"
        assert cfg.tier_for_domain("helper.com") == "reliable"
        assert cfg.tier_for_domain("evil.net") == "blocked"
        # Defaults no longer present
        assert cfg.tier_for_domain("wikipedia.org") == "unknown"

    def test_domains_for_content_type(self):
        cfg = SourceTrustConfig()
        poem_domains = cfg.domains_for_content_type("poem")
        assert "gutenberg.org" in poem_domains
        assert "poetryfoundation.org" in poem_domains
        assert cfg.domains_for_content_type("nonexistent") == []


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:

    def test_allows_within_limit(self):
        limiter = _RateLimiter(max_calls=3, window_seconds=60)
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is False

    def test_remaining_count(self):
        limiter = _RateLimiter(max_calls=5, window_seconds=60)
        assert limiter.remaining == 5
        limiter.allow()
        assert limiter.remaining == 4

    def test_window_expiry(self):
        limiter = _RateLimiter(max_calls=1, window_seconds=1)
        assert limiter.allow() is True
        assert limiter.allow() is False
        time.sleep(1.1)
        assert limiter.allow() is True


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class TestWebSearch:

    def setup_method(self):
        """Reset module singletons before each test."""
        import gaia_mcp.web_tools as wt
        wt._trust_config = None
        wt._search_limiter = None
        wt._fetch_limiter = None

    def test_missing_query_raises(self):
        with pytest.raises(ValueError, match="query is required"):
            web_search({})

    @patch("gaia_mcp.web_tools._ddg_search")
    @patch("gaia_mcp.web_tools._get_config")
    def test_basic_search_returns_results(self, mock_config, mock_ddg):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        fetch_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)

        mock_ddg.return_value = [
            {"title": "Ozymandias", "href": "https://www.poetryfoundation.org/poems/46565/ozymandias", "body": "By Shelley"},
            {"title": "Wiki", "href": "https://en.wikipedia.org/wiki/Ozymandias", "body": "A poem"},
        ]

        result = web_search({"query": "Ozymandias Shelley"})
        assert result["ok"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["trust_tier"] == "trusted"
        assert result["results"][0]["domain"] == "www.poetryfoundation.org"

    @patch("gaia_mcp.web_tools._ddg_search")
    @patch("gaia_mcp.web_tools._get_config")
    def test_blocked_domains_filtered(self, mock_config, mock_ddg):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        fetch_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)

        mock_ddg.return_value = [
            {"title": "Good", "href": "https://wikipedia.org/wiki/test", "body": "Good"},
            {"title": "Bad", "href": "https://reddit.com/r/test", "body": "Bad"},
        ]

        result = web_search({"query": "test"})
        assert result["ok"] is True
        assert len(result["results"]) == 1
        assert result["filtered_count"] == 1

    @patch("gaia_mcp.web_tools._ddg_search")
    @patch("gaia_mcp.web_tools._get_config")
    def test_rate_limit_blocks(self, mock_config, mock_ddg):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=0, window_seconds=3600)  # already exhausted
        fetch_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)

        result = web_search({"query": "test"})
        assert result["ok"] is False
        assert "Rate limit" in result["error"]

    @patch("gaia_mcp.web_tools._ddg_search")
    @patch("gaia_mcp.web_tools._get_config")
    def test_content_type_adds_site_clause(self, mock_config, mock_ddg):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        fetch_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)
        mock_ddg.return_value = []

        web_search({"query": "Ozymandias", "content_type": "poem"})
        # Verify the effective query includes site: clauses
        call_args = mock_ddg.call_args[0][0]
        assert "site:gutenberg.org" in call_args
        assert "site:poetryfoundation.org" in call_args


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

class TestWebFetch:

    def setup_method(self):
        import gaia_mcp.web_tools as wt
        wt._trust_config = None
        wt._search_limiter = None
        wt._fetch_limiter = None

    def test_missing_url_raises(self):
        with pytest.raises(ValueError, match="url is required"):
            web_fetch({})

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            web_fetch({"url": "ftp://example.com/file"})

    @patch("gaia_mcp.web_tools._get_config")
    def test_blocked_domain_rejected(self, mock_config):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        fetch_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)

        result = web_fetch({"url": "https://reddit.com/r/poetry"})
        assert result["ok"] is False
        assert result["trust_tier"] == "blocked"

    @patch("gaia_mcp.web_tools._get_config")
    def test_unknown_domain_rejected(self, mock_config):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        fetch_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)

        result = web_fetch({"url": "https://randomsite.xyz/page"})
        assert result["ok"] is False
        assert result["trust_tier"] == "unknown"
        assert "allowlist" in result["error"]

    @patch("gaia_mcp.web_tools.requests")
    @patch("gaia_mcp.web_tools._extract_content")
    @patch("gaia_mcp.web_tools._get_config")
    def test_successful_fetch(self, mock_config, mock_extract, mock_requests):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        fetch_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)

        mock_resp = MagicMock()
        mock_resp.content = b"<html><title>Test</title><body>Hello world</body></html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_requests.get.return_value = mock_resp

        mock_extract.return_value = ("Test", "Hello world")

        result = web_fetch({"url": "https://en.wikipedia.org/wiki/Test"})
        assert result["ok"] is True
        assert result["title"] == "Test"
        assert result["content"] == "Hello world"
        assert result["trust_tier"] == "trusted"

    @patch("gaia_mcp.web_tools._get_config")
    def test_rate_limit_blocks_fetch(self, mock_config):
        trust = SourceTrustConfig()
        search_lim = _RateLimiter(max_calls=100, window_seconds=3600)
        fetch_lim = _RateLimiter(max_calls=0, window_seconds=3600)
        mock_config.return_value = (trust, search_lim, fetch_lim)

        result = web_fetch({"url": "https://wikipedia.org/wiki/Test"})
        assert result["ok"] is False
        assert "Rate limit" in result["error"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_domain_from_url(self):
        assert _domain_from_url("https://en.wikipedia.org/wiki/Test") == "en.wikipedia.org"
        assert _domain_from_url("https://example.com:8080/path") == "example.com:8080"
        assert _domain_from_url("invalid") == ""
