"""Tests for GAIA MCP Kanka.io tools."""

import time
from unittest.mock import patch, MagicMock
import pytest

from gaia_mcp.kanka_tools import (
    KankaClient,
    kanka_search,
    kanka_get_entity,
    kanka_list_entities,
    kanka_list_campaigns,
    kanka_create_entity,
    kanka_update_entity,
    _VALID_ENTITY_TYPES,
)
import gaia_mcp.kanka_tools as kt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level client singleton between tests."""
    kt._client = None
    yield
    kt._client = None


# ---------------------------------------------------------------------------
# KankaClient unit tests
# ---------------------------------------------------------------------------

class TestKankaClient:

    def test_missing_api_key_returns_error(self, monkeypatch):
        # api_key="" falls back to the env var, so clear it or a real key leaks in.
        monkeypatch.delenv("KANKA_API_KEY", raising=False)
        client = KankaClient(api_key="")
        result = client.get("campaigns")
        assert result["ok"] is False
        assert "KANKA_API_KEY" in result["error"]

    def test_rate_limiter_enforces_cap(self):
        client = KankaClient(api_key="test", max_requests_per_minute=2)
        assert client._check_rate_limit() is True
        assert client._check_rate_limit() is True
        assert client._check_rate_limit() is False

    def test_remaining_requests(self):
        client = KankaClient(api_key="test", max_requests_per_minute=5)
        assert client.remaining_requests == 5
        client._check_rate_limit()
        assert client.remaining_requests == 4

    def test_cache_hit_and_miss(self):
        client = KankaClient(api_key="test", cache_ttl_seconds=60)
        client._cache_set("key1", {"data": "cached"}, 60)
        assert client._cache_get("key1") == {"data": "cached"}
        assert client._cache_get("nonexistent") is None

    def test_cache_expiry(self):
        client = KankaClient(api_key="test", cache_ttl_seconds=0)
        client._cache_set("key1", {"data": "cached"}, 0)
        # TTL of 0 means it expires immediately on next check
        time.sleep(0.05)
        assert client._cache_get("key1") is None

    def test_cache_size_pruning(self):
        client = KankaClient(api_key="test")
        for i in range(210):
            client._cache_set(f"key{i}", {"i": i}, 3600)
        assert len(client._cache) <= 200

    def test_invalidate_cache(self):
        client = KankaClient(api_key="test")
        client._cache_set("k1", {"a": 1}, 600)
        client._cache_set("k2", {"b": 2}, 600)
        assert len(client._cache) == 2
        client.invalidate_cache()
        assert len(client._cache) == 0

    @patch("gaia_mcp.kanka_tools.req_lib.get")
    def test_429_returns_rate_limit_error(self, mock_get):
        client = KankaClient(api_key="test_token")
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp
        result = client.get("campaigns")
        assert result["ok"] is False
        assert "rate limit" in result["error"].lower()

    @patch("gaia_mcp.kanka_tools.req_lib.get")
    def test_403_returns_access_denied(self, mock_get):
        client = KankaClient(api_key="test_token")
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp
        result = client.get("campaigns/999/characters")
        assert result["ok"] is False
        assert "Access denied" in result["error"]

    @patch("gaia_mcp.kanka_tools.req_lib.get")
    def test_404_returns_not_found(self, mock_get):
        client = KankaClient(api_key="test_token")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp
        result = client.get("campaigns/999/characters/1")
        assert result["ok"] is False
        assert "Not found" in result["error"]

    @patch("gaia_mcp.kanka_tools.req_lib.get")
    def test_successful_get_caches_result(self, mock_get):
        client = KankaClient(api_key="test_token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": 1, "name": "Test"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # First call hits the API
        result1 = client.get("campaigns")
        assert result1["data"][0]["name"] == "Test"
        assert mock_get.call_count == 1

        # Second call should use cache
        result2 = client.get("campaigns")
        assert result2["data"][0]["name"] == "Test"
        assert mock_get.call_count == 1  # No additional API call

    @patch("gaia_mcp.kanka_tools.req_lib.post")
    def test_post_does_not_cache(self, mock_post):
        client = KankaClient(api_key="test_token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"id": 42, "name": "New"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = client.post("campaigns/1/notes", {"name": "Test"})
        assert result["data"]["id"] == 42
        assert len(client._cache) == 0


# ---------------------------------------------------------------------------
# Tool function tests
# ---------------------------------------------------------------------------

class TestKankaListCampaigns:

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_returns_campaigns(self, mock_gc):
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "data": [
                {"id": 36323, "name": "Dawn of An Age", "locale": "en", "entry": "A campaign", "visibility": "public", "updated_at": "2025-01-01"},
                {"id": 36156, "name": "Twilight of the Gods", "locale": "en", "entry": "", "visibility": "public", "updated_at": "2025-06-01"},
            ]
        }
        mock_gc.return_value = mock_client

        result = kanka_list_campaigns({})
        assert result["ok"] is True
        assert result["count"] == 2
        assert result["campaigns"][0]["name"] == "Dawn of An Age"

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_propagates_error(self, mock_gc):
        mock_client = MagicMock()
        mock_client.get.return_value = {"ok": False, "error": "no key"}
        mock_gc.return_value = mock_client

        result = kanka_list_campaigns({})
        assert result["ok"] is False


# Shapes as Kanka API 1.0 returns them (2026-11-17): search and /entities give
# ENTITY ids; the module id is only in urls.api.
_LESHEN = {"id": 10154654, "name": "Leshen", "type": "location", "type_field": "Capital city",
           "module": {"code": "location", "plural": "Locations"}, "is_private": True,
           "parent_id": 10154652, "child_id": 2288193,
           "urls": {"view": "https://app.kanka.io/w/162151/entities/10154654",
                    "api": "https://api.kanka.io/1.0/campaigns/162151/locations/2288193"},
           "child": {"id": 2288193, "name": "Leshen",
                     "entry": "<p>Capital of Llasak.</p><ul><li>Palaces &amp; slums</li></ul>"}}
_PRIMER = {"id": 10155000, "name": "Session Zero: Player Primer", "type": "note", "type_field": "",
           "module": {"code": "note", "plural": "Notes"}, "is_private": False, "parent_id": None,
           "child_id": 777, "urls": {"view": "v", "api": "https://api.kanka.io/1.0/campaigns/162151/notes/777"},
           "child": {"entry": "<p>Welcome, players.</p>"}}
_HITS = [{"id": 10154654, "name": "Leshen", "type": "Capital city", "entity_type": "location",
          "is_private": True, "url": "u1"},
         {"id": 10155000, "name": "Session Zero: Player Primer", "type": "", "entity_type": "note",
          "is_private": False, "url": "u2"}]


class FakeKanka:
    def __init__(self):
        self.posts, self.patches = [], []

    def get(self, path, params=None, cache_ttl=None):
        if "/search/" in path:
            q = path.rsplit("/", 1)[1].lower()
            return {"data": [h for h in _HITS if q in h["name"].lower()]}
        if path.endswith("/entities/10154654"):
            return {"data": _LESHEN}
        if path.endswith("/entities/10155000"):
            return {"data": _PRIMER}
        if path.endswith("/entities/10154652"):
            return {"data": {"id": 10154652, "name": "Llasak", "type": "location", "child_id": 2288190}}
        if path.endswith("/locations/2288193"):
            return {"data": {"id": 2288193, "entity_id": 10154654}}
        if path.endswith("/locations"):
            return {"data": [{"entity_id": 10154654, "name": "Leshen", "is_private": True},
                             {"entity_id": 1, "name": "Market", "is_private": False}],
                    "meta": {"total": 2, "last_page": 1}, "links": {"next": None}}
        return {"ok": False, "error": f"Not found: {path}"}

    def post(self, path, body):
        self.posts.append((path, body))
        return {"data": {"entity_id": 1, "name": body["name"], "is_private": body.get("is_private")}}

    def patch(self, path, body):
        self.patches.append((path, body))
        return {"data": {"name": "Leshen"}}

    def invalidate_cache(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    f = FakeKanka()
    monkeypatch.setattr(kt, "_get_client", lambda: f)
    return f


class TestPrivacy:
    """Most of the campaign is DM-only and players are in the Grotto: private
    entries only when gaia-core says the conversation is the DM's alone."""

    def test_search_hides_private_without_dm_view(self, fake):
        r = kanka_search({"query": "e"})
        assert [x["name"] for x in r["results"]] == ["Session Zero: Player Primer"]
        assert "public entries" in r["note"]

    def test_nothing_shareable_is_said_and_reveals_nothing(self, fake):
        hidden = kanka_search({"query": "Leshen"})          # a DM-only entry matches
        absent = kanka_search({"query": "Zzyzx"})           # nothing matches at all
        assert hidden["count"] == absent["count"] == 0
        assert "Nothing in the campaign matches" in hidden["note"]
        assert hidden["note"].replace("Leshen", "Q") == absent["note"].replace("Zzyzx", "Q")

    def test_search_shows_all_in_dm_view(self, fake):
        r = kanka_search({"query": "e", "_dm_view": True})
        assert r["count"] == 2 and "note" not in r
        assert r["results"][0]["entity_id"] == 10154654 and r["results"][0]["kind"] == "location"

    def test_get_private_refused_without_dm_view(self, fake):
        r = kanka_get_entity({"entity_id": 10154654})
        assert r["ok"] is False and "DM-only" in r["error"]
        assert "Llasak" not in str(r)

    def test_get_public_entry_anywhere(self, fake):
        r = kanka_get_entity({"entity_id": 10155000})
        assert r["ok"] and r["entry"] == "Welcome, players."

    def test_list_hides_private_and_total(self, fake):
        r = kanka_list_entities({"entity_type": "locations"})
        assert [e["name"] for e in r["entities"]] == ["Market"] and "total" not in r

    def test_writes_refused_outside_dm_view(self, fake):
        assert not kanka_create_entity({"entity_type": "notes", "name": "x"})["ok"]
        assert not kanka_update_entity({"entity_id": 10154654, "append_entry": "x"})["ok"]
        assert not fake.posts and not fake.patches


class TestKankaSearch:

    def test_exact_name_comes_with_its_text(self, fake):
        r = kanka_search({"query": "leshen", "_dm_view": True})
        assert r["title"] == "Leshen (Capital city)" and r["content"].startswith("Capital of Llasak")
        from gaia_common.utils.tool_call_parser import tool_result_text
        assert "Capital of Llasak" in tool_result_text(r, question="what is Leshen?")

    def test_no_text_without_dm_view_for_private(self, fake):
        r = kanka_search({"query": "Leshen"})
        assert "content" not in r and "Llasak" not in str(r)

    def test_missing_query_raises(self):
        with pytest.raises(ValueError, match="query is required"):
            kanka_search({})


class TestKankaGetEntity:

    def test_missing_entity_id_raises(self):
        with pytest.raises(ValueError, match="entity_id is required"):
            kanka_get_entity({})

    def test_reads_entry_as_text(self, fake):
        r = kanka_get_entity({"entity_id": 10154654, "_dm_view": True})
        assert r["name"] == "Leshen" and r["kind"] == "locations" and r["type"] == "Capital city"
        assert r["entry"] == "Capital of Llasak.\n- Palaces & slums"
        assert r["parent_entity_id"] == 10154652

    def test_module_id_with_type_still_works(self, fake):
        r = kanka_get_entity({"entity_id": 2288193, "entity_type": "locations", "_dm_view": True})
        assert r["ok"] and r["name"] == "Leshen"


class TestKankaListEntities:

    def test_missing_entity_type_raises(self):
        with pytest.raises(ValueError):
            kanka_list_entities({})


class TestKankaCreateEntity:

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name is required"):
            kanka_create_entity({"entity_type": "journals", "_dm_view": True})

    def test_private_by_default(self, fake):
        r = kanka_create_entity({"entity_type": "notes", "name": "Secret", "_dm_view": True})
        assert r["ok"] and fake.posts[0][1]["is_private"] is True

    def test_public_when_asked(self, fake):
        kanka_create_entity({"entity_type": "notes", "name": "Handout", "public": True, "_dm_view": True})
        assert fake.posts[0][1]["is_private"] is False

    def test_location_parent_is_entity_id(self, fake):
        # 1iq0: a location's parent is parent_id = the parent's ENTITY id.
        kanka_create_entity({"entity_type": "locations", "name": "Old Quarter",
                             "parent": "Leshen", "_dm_view": True})
        assert fake.posts[0][1]["parent_id"] == 10154654

    def test_character_location_is_module_id_list(self, fake):
        # 1iq0: characters take locations: [location MODULE ids].
        kanka_create_entity({"entity_type": "characters", "name": "Mira",
                             "location": 10154654, "_dm_view": True})
        body = fake.posts[0][1]
        assert body["locations"] == [2288193] and "location_id" not in body

    def test_unknown_parent_raises(self, fake):
        with pytest.raises(ValueError, match="parent"):
            kanka_create_entity({"entity_type": "locations", "name": "X", "parent": "Nowhere",
                                 "_dm_view": True})


class TestKankaUpdateEntity:

    def test_nothing_to_change_raises(self, fake):
        with pytest.raises(ValueError, match="nothing to change"):
            kanka_update_entity({"entity_id": 10154654, "_dm_view": True})

    def test_append_keeps_existing_entry(self, fake):
        r = kanka_update_entity({"entity_id": 10154654, "append_entry": "The lords meet at dusk.",
                                 "_dm_view": True})
        path, body = fake.patches[0]
        assert r["ok"] and path.endswith("/locations/2288193")
        assert body["entry"].startswith("<p>Capital of Llasak.</p>")
        assert body["entry"].endswith("<p>The lords meet at dusk.</p>")

    def test_fields_by_entity_id(self, fake):
        kanka_update_entity({"entity_id": 10154654, "fields": {"type": "City"}, "_dm_view": True})
        assert fake.patches[0][1] == {"type": "City"}


class TestWriteProblem:
    """A malformed change is refused before anyone is asked to approve it."""

    def test_her_bad_proposal_is_refused(self, fake):
        p = {"entity_id": None, "campaign": "Kanka", "key": "Lords_of_Llasak_meeting_time",
             "value": "dusk in the Obsidian Spire", "_dm_view": True, "_allow_pending": True}
        msg = kt.write_problem("kanka_update_entity", p)
        assert "doesn't take key, value" in msg

    def test_good_update_by_name_passes(self, fake):
        assert kt.write_problem("kanka_update_entity", {"entity": "Leshen", "append_entry": "x",
                                                         "_dm_view": True}) is None

    def test_unknown_entry_and_empty_change(self, fake):
        assert "no single" in kt.write_problem("kanka_update_entity",
                                               {"entity": "Nowhere", "append_entry": "x", "_dm_view": True})
        assert "nothing to change" in kt.write_problem("kanka_update_entity",
                                                       {"entity": "Leshen", "_dm_view": True})

    def test_not_in_a_shared_conversation(self, fake):
        assert "direct message" in kt.write_problem("kanka_create_entity", {"entity_type": "notes", "name": "x"})

    def test_preflight_uses_it(self, fake):
        from gaia_mcp.tools import preflight
        assert "doesn't take" in preflight("worldbuild", {"action": "update", "entity": "Leshen",
                                                          "value": "x", "_dm_view": True})

    def test_update_by_name(self, fake):
        r = kanka_update_entity({"entity": "Leshen", "append_entry": "Dusk.", "_dm_view": True})
        assert r["ok"] and fake.patches[0][0].endswith("/locations/2288193")


class TestValidEntityTypes:
    """Ensure the valid entity types set is complete."""

    def test_expected_types_present(self):
        expected = {"characters", "locations", "journals", "items", "notes", "quests", "races", "organisations"}
        assert expected.issubset(_VALID_ENTITY_TYPES)

    def test_count(self):
        assert len(_VALID_ENTITY_TYPES) == 17
