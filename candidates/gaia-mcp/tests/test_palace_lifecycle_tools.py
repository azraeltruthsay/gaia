"""6afh: gaia-mcp's palace tools for answer feedback, pinning, restoring
and the lifecycle sweep, over a real palace in a temp dir."""

from unittest.mock import MagicMock, patch

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


# ── yx93: work memory ────────────────────────────────────────────────

POEM = "One two three four five six seven eight.\n\nNine ten eleven twelve thirteen fourteen."


def test_store_work_requires_two_sources(palace):
    assert not tools._palace_store_work_impl({"name": "Numbers", "text": POEM, "sources": [{"url": "a"}]})["ok"]
    assert tools._palace_store_work_impl({"name": "Numbers", "text": POEM, "sources": [{"url": "a"}, {"url": "b"}]})["ok"]


def test_recall_work_whole_part_and_unknown(palace):
    tools._palace_store_work_impl({"name": "Numbers", "text": POEM, "sources": [{"url": "a"}, {"url": "b"}]})
    assert tools._palace_recall_work_impl({"query": "Numbers"})["text"] == POEM
    assert tools._palace_recall_work_impl({"query": "the last stanza of Numbers"})["text"].startswith("Nine")
    assert tools._palace_recall_work_impl({"query": "Letters"})["found"] is False
    assert not tools._palace_recall_work_impl({})["ok"]


def test_palace_recall_brings_the_work_along(palace):
    tools._palace_store_work_impl({"name": "Numbers", "text": POEM, "sources": [{"url": "a"}, {"url": "b"}]})
    out = palace.recall("Numbers")
    assert out["works"][0]["text"] == POEM and out["works"][0]["sources"] == ["a", "b"]
    assert palace.status()["works"] == 1


def test_palace_recall_finds_a_work_a_request_names_in_passing(palace):
    tools._palace_store_work_impl({"name": "Numbers", "text": POEM, "sources": [{"url": "a"}, {"url": "b"}]})
    out = tools._palace_recall_impl({"query": "Use your memory palace to recall Numbers and tell me what it is about"})
    assert out["works"] and out["works"][0]["text"] == POEM


def test_knowledge_add_keeps_a_told_fact_where_recall_finds_it(palace):
    fact = "The Raven was first published in the New York Evening Mirror on January 29, 1845."
    out = tools._knowledge_add_impl({"fact": fact}, None)
    assert out["ok"] and out["trust"] == "told"
    hit = palace.recall("Where was The Raven first published?")["lookups"][0]
    assert hit["fact"] == fact and hit["trust"] == "told" and hit["pinned"]
    assert "not independently checked" in hit["source"]
    assert not palace.lookups.revalidation_candidates()


def test_a_short_told_fact_is_kept(palace):
    assert tools._knowledge_add_impl({"fact": "My favorite color is cerulean."}, None)["ok"]


def test_a_fact_arrives_under_whichever_param_she_chose(palace):
    assert tools._knowledge_add_impl({"query": "The attic lantern is kept in the north attic."}, None)["ok"]
    assert palace.recall("Where is the attic lantern kept?")["lookups"]


def test_knowledge_add_still_takes_a_file_and_says_when_it_has_nothing():
    indexer = MagicMock()
    tools._knowledge_add_impl({"file_path": "/knowledge/a.md"}, indexer)
    indexer.instance.return_value.add_document.assert_called_once_with("/knowledge/a.md")
    assert tools._knowledge_add_impl({}, None)["ok"] is False


@pytest.mark.parametrize("domain,action", [("knowledge", "add"), ("knowledge", "update"), ("knowledge", "add_fact"),
                                           ("knowledge", "learn"), ("memory", "remember")])
def test_the_verbs_she_tries_for_a_fact_all_reach_it(domain, action):
    from gaia_common.utils.domain_tools import validate_domain_call
    assert validate_domain_call(domain, action) == "add_document"


AZ = {"who": "Azrael", "id": "discord:1", "where": "in a Discord DM", "when": "2026-11-03T10:00:00+00:00",
      "why": "asked me to keep it"}
BO = {**AZ, "who": "Bo", "id": "discord:2"}
LANTERN = "The north lantern is kept in the attic."


def test_a_told_fact_records_who_when_where_why(palace):
    tools._knowledge_add_impl({"fact": LANTERN, "_told_by": AZ}, None)
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["told_by"] == [AZ]
    assert hit["source"].startswith("told by Azrael on 2026-11-03, in a Discord DM")


def test_a_second_person_raises_the_level_but_validates_nothing(palace):
    tools._knowledge_add_impl({"fact": LANTERN, "_told_by": AZ}, None)
    tools._knowledge_add_impl({"fact": LANTERN, "_told_by": AZ}, None)
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "told" and len(hit["told_by"]) == 1
    tools._knowledge_add_impl({"fact": LANTERN, "_told_by": BO}, None)
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "told_by_several" and "and by 1 other" in hit["source"]
    assert "not independently checked" in hit["source"]


def test_validation_needs_evidence_and_answers_a_dispute(palace):
    rid = tools._knowledge_add_impl({"fact": LANTERN, "_told_by": AZ}, None)["id"]
    assert not tools._palace_validate_lookup_impl({"id": rid})["ok"]
    palace.lookups.mark_disputed([rid])
    assert palace.recall("Where is the north lantern kept?")["lookups"][0]["trust"] == "disputed"
    tools._knowledge_add_impl({"fact": LANTERN, "_told_by": BO}, None)
    assert palace.recall("Where is the north lantern kept?")["lookups"][0]["trust"] == "disputed"
    assert tools._palace_validate_lookup_impl({"id": rid, "evidence": "seen in the attic", "_by": "Azrael"})["ok"]
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "validated" and "checked by Azrael" in hit["source"]


def test_what_is_about_the_teller_answers_only_the_teller(palace):
    tools._knowledge_add_impl({"fact": "My favorite color is cerulean.", "_told_by": AZ}, None)
    q = "What is my favorite color?"
    assert palace.lookups.recall(q, asker="discord:1")
    assert not palace.lookups.recall(q, asker="discord:2")
    assert not palace.lookups.recall(q)
    assert not palace.lookups.recent(asker="discord:2")
    assert palace.recall(q, asker="discord:1")["lookups"]
    assert not palace.recall(q)["lookups"]


def test_web_lookups_get_a_level_from_their_evidence(palace):
    from gaia_common.utils.lookup_memory import trust_level
    assert trust_level({"url": "https://a.org/x"}) == "looked_up"
    assert trust_level({"url": "https://a.org/x", "also_at": ["https://www.a.org/y"]}) == "looked_up"
    assert trust_level({"url": "https://a.org/x", "also_at": ["https://b.org/y"]}) == "corroborated"


def test_conversations_without_an_identity_count_as_one_teller(palace):
    for sid in ("session:a", "session:b"):
        tools._knowledge_add_impl({"fact": LANTERN, "_told_by": {"who": "someone", "id": sid}}, None)
    assert palace.recall("Where is the north lantern kept?")["lookups"][0]["trust"] == "told"


ARCHITECT = {**AZ, "operator": True}


def test_the_architect_has_a_level_of_their_own_and_still_needs_checking(palace):
    rid = tools._knowledge_add_impl({"fact": LANTERN, "_told_by": ARCHITECT}, None)["id"]
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "told_by_architect"
    assert "the architect" in hit["source"] and "not independently checked" in hit["source"]
    palace.lookups.mark_disputed([rid])
    assert palace.recall("Where is the north lantern kept?")["lookups"][0]["trust"] == "disputed"
    tools._palace_validate_lookup_impl({"id": rid, "evidence": "seen in the attic"})
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "validated" and "checked by GAIA" in hit["source"]


def test_she_cannot_validate_by_calling_a_tool():
    from gaia_common.utils.domain_tools import DOMAIN_ACTIONS
    assert "validate" not in DOMAIN_ACTIONS["palace"]


def test_which_told_facts_are_due_a_check(palace):
    rid = tools._knowledge_add_impl({"fact": LANTERN, "_told_by": AZ}, None)["id"]
    tools._knowledge_add_impl({"fact": "My favorite color is cerulean.", "_told_by": AZ}, None)
    assert [r["id"] for r in palace.lookups.told_to_check()] == [rid]
    palace.lookups.record_check(rid, "no_evidence", [])
    assert palace.lookups.told_to_check() == []
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "told" and "couldn't confirm it" in hit["source"]


def test_a_check_validates_or_disputes_whoever_told_her(palace):
    blanks = [{"term": "attic", "supports": ["https://a.org/x", "https://b.org/y"], "contrary": []}]
    rid = tools._knowledge_add_impl({"fact": LANTERN, "_told_by": ARCHITECT}, None)["id"]
    palace.lookups.record_check(rid, "contradicted", [])
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "disputed" and "web sources say otherwise" in hit["source"]
    palace.lookups.record_check(rid, "supported", blanks)
    hit = palace.recall("Where is the north lantern kept?")["lookups"][0]
    assert hit["trust"] == "validated" and hit["validated"]["evidence"] == ["https://a.org/x", "https://b.org/y"]
    assert palace.lookups.told_to_check() == []


def test_the_ladder_orders_that_word_between_one_site_and_two():
    from gaia_common.utils.lookup_memory import TRUST_LEVELS
    assert TRUST_LEVELS.index("looked_up") < TRUST_LEVELS.index("told_by_architect") < TRUST_LEVELS.index("corroborated")
