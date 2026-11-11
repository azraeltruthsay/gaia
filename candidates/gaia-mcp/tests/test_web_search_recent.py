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


def test_dates_after_today_are_marked_upcoming():
    """sn26: on 2026-11-11 she said SpaceX "attempted" a catch set for November 13."""
    import datetime as dt
    today = dt.date(2026, 11, 11)
    mark = web_tools.mark_upcoming
    assert mark("SpaceX will attempt it on Friday, November 13, the company says", today) == \
        "SpaceX will attempt it on Friday, November 13 (upcoming), the company says"
    assert mark("targeting Nov. 13, 2026, ...", today) == "targeting Nov. 13, 2026 (upcoming), ..."
    assert mark("launched on 13 November 2025", today) == "launched on 13 November 2025"
    assert mark("set for 2026-12-01; flew 2026-11-10", today) == "set for 2026-12-01 (upcoming); flew 2026-11-10"
    assert mark("It happened on November 11.", today) == "It happened on November 11."
    assert mark("February 30 isn't a day", today) == "February 30 isn't a day"
    assert mark("", today) == ""
