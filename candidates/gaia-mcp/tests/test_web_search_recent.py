"""Asked for the latest news on fusion energy, she answered from Wikipedia's
"As of 2025": the search was a plain one for "fusion energy"."""
from gaia_mcp import web_tools


def _patch(monkeypatch, news, text):
    calls = []
    monkeypatch.setattr(web_tools, "_ddg_news", lambda q, n, t: calls.append(("news", t)) or news)
    monkeypatch.setattr(web_tools, "_ddg_search", lambda q, n: calls.append(("text", None)) or text)
    monkeypatch.setattr(web_tools, "_remember_lookup", lambda *a, **k: None)
    return calls


def test_recent_asks_for_dated_news(monkeypatch):
    calls = _patch(monkeypatch, [{"title": "Shortcut to fusion", "url": "https://www.sciencedaily.com/x",
                                  "body": "Scientists may have...", "date": "2026-11-08T20:57:52+00:00",
                                  "source": "Science Daily"}], [])
    out = web_tools.web_search({"query": "fusion energy", "recent": "m"})
    assert calls == [("news", "m")]
    assert out["results"][0]["snippet"] == "2026-11-08, Science Daily: Scientists may have..."


def test_no_news_found_falls_back_to_a_plain_search(monkeypatch):
    calls = _patch(monkeypatch, [], [{"title": "Fusion", "href": "https://en.wikipedia.org/wiki/F", "body": "Fusion is"}])
    out = web_tools.web_search({"query": "fusion energy", "recent": "week"})
    assert calls == [("news", "w"), ("text", None)]
    assert out["results"][0]["snippet"] == "Fusion is"


def test_without_recent_it_is_a_plain_search(monkeypatch):
    calls = _patch(monkeypatch, [], [])
    web_tools.web_search({"query": "fusion energy", "recent": "soon"})
    assert calls == [("text", None)]
