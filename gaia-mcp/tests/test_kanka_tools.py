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

    def test_missing_api_key_returns_error(self):
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


class TestKankaSearch:

    def test_missing_query_raises(self):
        with pytest.raises(ValueError, match="query is required"):
            kanka_search({})

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_search_returns_results(self, mock_gc):
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "data": [
                {"id": 1, "entity_id": 100, "name": "Aldric", "type": "character", "is_private": False, "url": ""},
            ]
        }
        mock_gc.return_value = mock_client

        result = kanka_search({"query": "Aldric", "campaign_id": 36156})
        assert result["ok"] is True
        assert result["count"] == 1
        assert result["results"][0]["name"] == "Aldric"
        assert result["campaign_id"] == 36156


class TestKankaGetEntity:

    def test_missing_entity_type_raises(self):
        with pytest.raises(ValueError, match="entity_type"):
            kanka_get_entity({"entity_id": 1})

    def test_invalid_entity_type_raises(self):
        with pytest.raises(ValueError, match="Invalid entity_type"):
            kanka_get_entity({"entity_type": "dragons", "entity_id": 1})

    def test_missing_entity_id_raises(self):
        with pytest.raises(ValueError, match="entity_id is required"):
            kanka_get_entity({"entity_type": "characters"})

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_returns_entity(self, mock_gc):
        mock_client = MagicMock()
        mock_client.get.return_value = {"data": {"id": 42, "name": "Aldric", "entry": "<p>Brave</p>"}}
        mock_gc.return_value = mock_client

        result = kanka_get_entity({"entity_type": "characters", "entity_id": 42})
        assert result["ok"] is True
        assert result["entity"]["name"] == "Aldric"

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_related_flag(self, mock_gc):
        mock_client = MagicMock()
        mock_client.get.return_value = {"data": {"id": 1}}
        mock_gc.return_value = mock_client

        kanka_get_entity({"entity_type": "characters", "entity_id": 1, "related": True})
        _, kwargs = mock_client.get.call_args
        assert kwargs.get("params") == {"related": "1"}


class TestKankaListEntities:

    def test_missing_entity_type_raises(self):
        with pytest.raises(ValueError):
            kanka_list_entities({})

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_returns_page(self, mock_gc):
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "data": [{"id": 1, "name": "Aldric", "type": "NPC", "is_private": False, "updated_at": "2025-01-01"}],
            "meta": {"total": 50, "last_page": 2},
            "links": {"next": "page=2", "prev": None},
        }
        mock_gc.return_value = mock_client

        result = kanka_list_entities({"entity_type": "characters"})
        assert result["ok"] is True
        assert result["count"] == 1
        assert result["total"] == 50
        assert result["has_next"] is True


class TestKankaCreateEntity:

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name is required"):
            kanka_create_entity({"entity_type": "journals"})

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_create_succeeds(self, mock_gc):
        mock_client = MagicMock()
        mock_client.post.return_value = {"data": {"id": 99, "name": "Session 5"}}
        mock_gc.return_value = mock_client

        result = kanka_create_entity({
            "entity_type": "journals",
            "name": "Session 5",
            "entry": "<p>The party fought.</p>",
            "fields": {"date": "2024-01-15"},
        })
        assert result["ok"] is True
        assert result["created"]["name"] == "Session 5"
        mock_client.invalidate_cache.assert_called_once()


class TestKankaUpdateEntity:

    def test_missing_fields_raises(self):
        with pytest.raises(ValueError, match="fields"):
            kanka_update_entity({"entity_type": "characters", "entity_id": 1})

    @patch("gaia_mcp.kanka_tools._get_client")
    def test_update_succeeds(self, mock_gc):
        mock_client = MagicMock()
        mock_client.patch.return_value = {"data": {"id": 42, "name": "Updated Name"}}
        mock_gc.return_value = mock_client

        result = kanka_update_entity({
            "entity_type": "characters",
            "entity_id": 42,
            "fields": {"name": "Updated Name"},
        })
        assert result["ok"] is True
        assert result["updated"]["name"] == "Updated Name"
        mock_client.invalidate_cache.assert_called_once()


class TestValidEntityTypes:
    """Ensure the valid entity types set is complete."""

    def test_expected_types_present(self):
        expected = {"characters", "locations", "journals", "items", "notes", "quests", "races", "organisations"}
        assert expected.issubset(_VALID_ENTITY_TYPES)

    def test_count(self):
        assert len(_VALID_ENTITY_TYPES) == 17
