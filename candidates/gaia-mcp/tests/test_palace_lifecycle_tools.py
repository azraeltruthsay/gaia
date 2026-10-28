"""6afh: gaia-mcp's palace tools for answer feedback, pinning, restoring
and the lifecycle sweep, over a real palace in a temp dir."""

from unittest.mock import patch

import pytest

from gaia_common.utils.mempalace import MemPalace
from gaia_mcp import tools

LADY = {"title": "Lady of the Lake", "url": "https://en.wikipedia.org/wiki/Lady_of_the_Lake",
        "snippet": "The Lady of the Lake gives King Arthur the sword Excalibur."}
Q = "Where did King Arthur get Excalibur?"


@pytest.fixture
def palace(tmp_path):
    p = MemPalace({"root_dir": str(tmp_path / "palace"), "wings": {},
                   "kg_path": str(tmp_path / "kg.db")})
    with patch.object(tools, "_get_palace", return_value=p):
        yield p


def test_feedback_confirms_and_recall_reports_it(palace):
    rid = palace.lookups.remember(Q, [dict(LADY)])[0]
    out = tools._palace_lookup_feedback_impl(
        {"question": Q, "answer": "The Lady of the Lake gave it to him.", "sources": [{"memory_id": rid}]})
    assert out["ok"] and out["verdict"] == "supported" and out["records"] == 1
    hit = tools._palace_recall_lookup_impl({"query": Q})["results"][0]
    assert hit["confirmed_match"] >= 0.6


def test_feedback_validates_params(palace):
    assert not tools._palace_lookup_feedback_impl({"question": Q})["ok"]
    assert not tools._palace_pin_impl({})["ok"]
    assert not tools._palace_pin_impl({"ids": "abc"})["ok"]


def test_pin_sweep_restore(palace):
    rid = palace.lookups.remember(Q, [dict(LADY)])[0]
    assert tools._palace_pin_impl({"ids": [rid]})["lookups"] == 1
    out = tools._palace_lifecycle_sweep_impl({"dry_run": True})
    assert out["dry_run"] and out["lookups"]["pinned"] == 1 and out["lookups"]["archived"] == []
    assert tools._palace_pin_impl({"ids": [rid], "pinned": False})["lookups"] == 1
    assert tools._palace_restore_impl({"ids": [rid]})["lookups"] == 0  # not archived


def test_recall_lookup_passes_include_archive(palace):
    with patch.object(palace.lookups, "recall", return_value=[]) as rec:
        tools._palace_recall_lookup_impl({"query": Q, "include_archive": True})
    assert rec.call_args.kwargs["include_archive"] is True


# ── prip ─────────────────────────────────────────────────────────────

def test_lookups_tool_recalls_compactly_and_browses(palace):
    palace.lookups.remember(Q, [dict(LADY)])
    hit = tools._palace_lookups_impl({"query": "King Arthur Excalibur"})["results"][0]
    assert hit["fact"] == LADY["snippet"] and hit["source"] == "Lady of the Lake"
    assert tools._palace_lookups_impl({})["results"][0]["fact"] == LADY["snippet"]
    empty = tools._palace_lookups_impl({"query": "capital of France"})
    assert empty["results"] == [] and "Nothing remembered" in empty["note"]


def test_palace_recall_includes_lookups_unless_told_not_to(palace):
    palace.lookups.remember(Q, [dict(LADY)])
    assert tools._palace_recall_impl({"query": "King Arthur Excalibur"})["lookups"][0]["fact"] == LADY["snippet"]
    assert tools._palace_recall_impl({"query": "King Arthur Excalibur", "include_lookups": False})["lookups"] == []


def test_domain_routes_for_lookups():
    from gaia_common.utils.domain_tools import validate_domain_call
    for domain, action in (("palace", "lookups"), ("knowledge", "recall"), ("memory", "recall")):
        assert validate_domain_call(domain, action) == "palace_lookups"


class _Limiter:
    def __init__(self, remaining):
        self.remaining = remaining


def _stale_used(palace):
    from datetime import datetime, timedelta, timezone
    old = datetime.now(timezone.utc) - timedelta(days=10)
    news = dict(LADY, snippet="The latest Excalibur exhibit news was announced today at the museum.")
    rid = palace.lookups.remember("latest Excalibur exhibit news", [news], now=old)[0]
    palace.lookups.recall("Excalibur exhibit news", now=old)
    return rid, news


def test_revalidate_searches_and_reports_refreshed(palace):
    from gaia_mcp import web_tools
    rid, news = _stale_used(palace)

    def fake_search(params):
        results = [dict(news)]
        palace.lookups.remember(params["query"], results)  # what web_search does
        return {"ok": True, "results": results}
    with patch.object(web_tools, "_get_config", return_value=(None, _Limiter(80), None)), \
            patch.object(web_tools, "web_search", side_effect=fake_search) as ws:
        out = tools._palace_lookup_revalidate_impl({})
    ws.assert_called_once()
    assert out["checked"] == [rid] and out["refreshed"] == [rid]
    assert not palace.lookups.recall("Excalibur exhibit news")[0]["stale"]


def test_revalidate_leaves_the_search_budget_for_conversation(palace):
    from gaia_mcp import web_tools
    rid, _ = _stale_used(palace)
    with patch.object(web_tools, "_get_config", return_value=(None, _Limiter(tools.REVALIDATE_SEARCH_RESERVE), None)), \
            patch.object(web_tools, "web_search") as ws:
        out = tools._palace_lookup_revalidate_impl({})
    ws.assert_not_called()
    assert out["deferred"] == [rid] and out["checked"] == []


def test_recall_accepts_the_words_she_uses_and_browses_without_one(palace):
    # prip live: palace(recall, topic="Excalibur") failed with "query parameter required".
    palace.lookups.remember(Q, [dict(LADY)])
    assert tools._palace_recall_impl({"topic": "King Arthur Excalibur"})["lookups"][0]["fact"] == LADY["snippet"]
    assert tools._palace_lookups_impl({"about": "King Arthur Excalibur"})["results"]
    browse = tools._palace_recall_impl({})
    assert browse["ok"] and browse["results"][0]["fact"] == LADY["snippet"]
    assert "No query given" in browse["note"]
