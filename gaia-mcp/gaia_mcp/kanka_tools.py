"""
Kanka.io MCP Tools — structured access to world-building campaign data.

Provides 6 tools:
  - kanka_list_campaigns   (read)
  - kanka_search           (read)
  - kanka_list_entities    (read)
  - kanka_get_entity       (read)
  - kanka_create_entity    (write, sensitive)
  - kanka_update_entity    (write, sensitive)

All HTTP calls go through KankaClient which enforces a 25 req/min
client-side rate limit (below the 30/min hard cap) and a TTL cache
for GET requests.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests as req_lib

logger = logging.getLogger("GAIA.KankaTools")

_KANKA_API_BASE = "https://api.kanka.io/1.0"
_DEFAULT_CAMPAIGN_ID = 36323  # "Dawn of An Age" (owner)

_VALID_ENTITY_TYPES = frozenset({
    "characters", "locations", "journals", "items", "events",
    "organisations", "races", "quests", "families", "maps",
    "calendars", "notes", "abilities", "tags", "timelines",
    "creatures", "conversations",
})


# ---------------------------------------------------------------------------
# KankaClient
# ---------------------------------------------------------------------------

class KankaClient:
    """HTTP client for the Kanka.io API with rate limiting and TTL cache."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _KANKA_API_BASE,
        max_requests_per_minute: int = 25,
        cache_ttl_seconds: int = 300,
    ):
        self.api_key = api_key or os.getenv("KANKA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._max_rpm = max_requests_per_minute
        self._request_timestamps: List[float] = []
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._cache_ttl = cache_ttl_seconds

    # ---- public HTTP helpers ------------------------------------------------

    def get(self, path: str, params: Optional[Dict] = None, cache_ttl: Optional[int] = None) -> dict:
        return self._request("GET", path, params=params, cache_ttl=cache_ttl)

    def post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body=body)

    def patch(self, path: str, body: dict) -> dict:
        return self._request("PATCH", path, body=body)

    # ---- core request -------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[dict] = None,
        cache_ttl: Optional[int] = None,
    ) -> dict:
        if not self.api_key:
            return {
                "ok": False,
                "error": "KANKA_API_KEY is not set. Add it to .env and restart gaia-mcp.",
            }

        # Cache check (GET only)
        cache_key = None
        if method == "GET":
            ttl = cache_ttl if cache_ttl is not None else self._cache_ttl
            cache_key = self._make_cache_key(path, params)
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: %s", path)
                return cached

        # Rate limit
        if not self._check_rate_limit():
            return {
                "ok": False,
                "error": "Kanka rate limit reached (25 req/min client cap). Wait a moment and retry.",
                "retry_after_seconds": 60,
            }

        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        try:
            if method == "GET":
                resp = req_lib.get(url, headers=headers, params=params, timeout=15)
            elif method == "POST":
                resp = req_lib.post(url, headers=headers, json=body, timeout=15)
            elif method == "PATCH":
                resp = req_lib.patch(url, headers=headers, json=body, timeout=15)
            else:
                return {"ok": False, "error": f"Unsupported HTTP method: {method}"}

            if resp.status_code == 429:
                return {"ok": False, "error": "Kanka API rate limit exceeded (429). Wait 60s.", "retry_after_seconds": 60}
            if resp.status_code == 403:
                return {"ok": False, "error": "Access denied. Check API key permissions for this campaign."}
            if resp.status_code == 404:
                return {"ok": False, "error": f"Not found: {path}. Verify campaign ID and entity ID."}

            resp.raise_for_status()
            data = resp.json()

            if method == "GET" and cache_key:
                self._cache_set(cache_key, data, ttl)

            return data

        except req_lib.RequestException as e:
            logger.error("Kanka API %s %s failed: %s", method, url, e)
            return {"ok": False, "error": f"Kanka API request failed: {e}"}

    # ---- rate limiting (sliding window) ------------------------------------

    def _check_rate_limit(self) -> bool:
        now = time.time()
        cutoff = now - 60
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        if len(self._request_timestamps) >= self._max_rpm:
            return False
        self._request_timestamps.append(now)
        return True

    @property
    def remaining_requests(self) -> int:
        now = time.time()
        active = [t for t in self._request_timestamps if t > now - 60]
        return max(0, self._max_rpm - len(active))

    # ---- TTL cache ---------------------------------------------------------

    def _make_cache_key(self, path: str, params: Optional[Dict]) -> str:
        raw = f"{path}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[dict]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expiry, data = entry
        if time.time() > expiry:
            del self._cache[key]
            return None
        return data

    def _cache_set(self, key: str, data: dict, ttl: int):
        if len(self._cache) > 200:
            now = time.time()
            expired = [k for k, (exp, _) in self._cache.items() if now > exp]
            for k in expired:
                del self._cache[k]
            if len(self._cache) > 200:
                sorted_keys = sorted(self._cache, key=lambda k: self._cache[k][0])
                for k in sorted_keys[:100]:
                    del self._cache[k]
        self._cache[key] = (time.time() + ttl, data)

    def invalidate_cache(self):
        """Clear all cached entries (called after writes)."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[KankaClient] = None


def _get_client() -> KankaClient:
    global _client
    if _client is None:
        _client = KankaClient()
    return _client


def _validate_entity_type(entity_type: str) -> str:
    t = entity_type.strip().lower()
    if t not in _VALID_ENTITY_TYPES:
        raise ValueError(
            f"Invalid entity_type '{t}'. Valid types: {', '.join(sorted(_VALID_ENTITY_TYPES))}"
        )
    return t


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def kanka_list_campaigns(params: dict) -> dict:
    """List campaigns accessible to the authenticated user."""
    client = _get_client()
    result = client.get("campaigns", cache_ttl=600)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    campaigns = result.get("data", [])
    return {
        "ok": True,
        "campaigns": [
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "locale": c.get("locale"),
                "entry": (c.get("entry") or "")[:200],
                "visibility": c.get("visibility"),
                "updated_at": c.get("updated_at"),
            }
            for c in campaigns
        ],
        "count": len(campaigns),
    }


def _resolve_campaign_id(params: dict) -> int:
    """Resolve campaign_id from params, supporting both ID and name lookup.

    Accepts:
        campaign_id (int)  — used directly
        campaign (str)     — fuzzy-matched against accessible campaign names
    Falls back to _DEFAULT_CAMPAIGN_ID if neither is provided.
    """
    # Explicit ID takes priority
    cid = params.get("campaign_id")
    if cid is not None:
        return int(cid)

    # Name-based lookup
    name = (params.get("campaign") or "").strip()
    if not name:
        return _DEFAULT_CAMPAIGN_ID

    client = _get_client()
    result = client.get("campaigns", cache_ttl=600)
    if isinstance(result, dict) and result.get("ok") is False:
        logger.warning("Campaign lookup failed, using default: %s", result)
        return _DEFAULT_CAMPAIGN_ID

    campaigns = result.get("data", [])
    name_lower = name.lower()
    # Exact match first
    for c in campaigns:
        if (c.get("name") or "").lower() == name_lower:
            logger.info("Resolved campaign '%s' → ID %s", name, c["id"])
            return c["id"]
    # Substring match
    for c in campaigns:
        if name_lower in (c.get("name") or "").lower():
            logger.info("Resolved campaign '%s' (substring) → ID %s (%s)", name, c["id"], c["name"])
            return c["id"]

    logger.warning("Campaign '%s' not found, using default %s", name, _DEFAULT_CAMPAIGN_ID)
    return _DEFAULT_CAMPAIGN_ID


def kanka_search(params: dict) -> dict:
    """Search across all entity types within a campaign."""
    query = (params.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    campaign_id = _resolve_campaign_id(params)
    client = _get_client()

    result = client.get(f"campaigns/{campaign_id}/search/{query}", cache_ttl=120)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    entities = result.get("data", [])
    return {
        "ok": True,
        "campaign_id": campaign_id,
        "query": query,
        "results": [
            {
                "id": e.get("id"),
                "entity_id": e.get("entity_id"),
                "name": e.get("name"),
                "type": e.get("type"),
                "is_private": e.get("is_private", False),
                "url": e.get("url", ""),
            }
            for e in entities
        ],
        "count": len(entities),
    }


def kanka_list_entities(params: dict) -> dict:
    """List entities of a given type, with optional name filter and pagination."""
    entity_type = _validate_entity_type(params.get("entity_type") or "")
    campaign_id = _resolve_campaign_id(params)
    name_filter = params.get("name")
    page = int(params.get("page", 1))

    client = _get_client()
    query_params: Dict[str, Any] = {"page": page}
    if name_filter:
        query_params["name"] = name_filter

    result = client.get(f"campaigns/{campaign_id}/{entity_type}", params=query_params, cache_ttl=180)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    entities = result.get("data", [])
    meta = result.get("meta", {})
    links = result.get("links", {})

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_type": entity_type,
        "entities": [
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "type": e.get("type", ""),
                "is_private": e.get("is_private", False),
                "updated_at": e.get("updated_at", ""),
            }
            for e in entities
        ],
        "count": len(entities),
        "total": meta.get("total"),
        "page": page,
        "last_page": meta.get("last_page"),
        "has_next": links.get("next") is not None,
    }


def kanka_get_entity(params: dict) -> dict:
    """Get a specific entity by type and ID, optionally with related data."""
    entity_type = _validate_entity_type(params.get("entity_type") or "")
    entity_id = params.get("entity_id")
    campaign_id = _resolve_campaign_id(params)
    include_related = params.get("related", False)

    if not entity_id:
        raise ValueError("entity_id is required")

    client = _get_client()
    query_params = {"related": "1"} if include_related else None

    result = client.get(
        f"campaigns/{campaign_id}/{entity_type}/{entity_id}",
        params=query_params,
        cache_ttl=300,
    )

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_type": entity_type,
        "entity": result.get("data", result),
    }


def kanka_create_entity(params: dict) -> dict:
    """Create a new entity in a Kanka campaign. Requires approval."""
    entity_type = _validate_entity_type(params.get("entity_type") or "")
    campaign_id = _resolve_campaign_id(params)
    name = (params.get("name") or "").strip()
    entry = params.get("entry", "")
    extra_fields = params.get("fields", {})

    if not name:
        raise ValueError("name is required")

    client = _get_client()
    body: Dict[str, Any] = {"name": name}
    if entry:
        body["entry"] = entry
    if isinstance(extra_fields, dict):
        body.update(extra_fields)

    result = client.post(f"campaigns/{campaign_id}/{entity_type}", body)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    client.invalidate_cache()

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_type": entity_type,
        "created": result.get("data", result),
    }


def kanka_update_entity(params: dict) -> dict:
    """Update an existing entity. Requires approval."""
    entity_type = _validate_entity_type(params.get("entity_type") or "")
    entity_id = params.get("entity_id")
    campaign_id = _resolve_campaign_id(params)
    fields = params.get("fields", {})

    if not entity_id:
        raise ValueError("entity_id is required")
    if not fields or not isinstance(fields, dict):
        raise ValueError("fields dict is required with at least one field to update")

    client = _get_client()
    result = client.patch(f"campaigns/{campaign_id}/{entity_type}/{entity_id}", fields)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    client.invalidate_cache()

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "updated": result.get("data", result),
    }
