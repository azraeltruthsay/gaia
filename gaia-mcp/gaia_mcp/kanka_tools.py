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

Who may see what (Azrael, 2026-11-17): the campaign is mostly DM-only (87 of
90 entries private), and players are in the Grotto. Private entries are read
and written only when the caller says the conversation is the DM's alone
(`_dm_view`, set by gaia-core from the session, never by the model). Without
it she sees only public entries and can't propose changes. Every write still
waits for Azrael's authenticator code (SENSITIVE_TOOLS / the worldbuild
domain's sensitive actions).

Kanka API 1.0 notes (1iq0, found writing the Dusk in Leshen batch): an
entity id (search, /entities) differs from its module id (/locations/<id>);
a location's parent is parent_id = the parent's ENTITY id; characters,
families, organisations and items take locations: [location MODULE ids]; a
scalar location_id is silently ignored.
"""

import hashlib
import html as html_lib
import re
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests as req_lib

logger = logging.getLogger("GAIA.KankaTools")

_KANKA_API_BASE = "https://api.kanka.io/1.0"
# Default campaign when a call names none. 162151 = "Sunrise in the Valley" (active campaign
# since 2026-11-08); 36323 = "Dawn of An Age" (the original Strauthauk world).
_DEFAULT_CAMPAIGN_ID = int(os.getenv("KANKA_DEFAULT_CAMPAIGN_ID", "162151"))

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

    def put(self, path: str, body: dict) -> dict:
        return self._request("PUT", path, body=body)

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
            elif method == "PUT":
                resp = req_lib.put(url, headers=headers, json=body, timeout=15)
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


_DM_ONLY = ("That entry is DM-only, and this conversation isn't the DM's alone, "
            "so it can't be read or changed here.")
_NOT_HERE = ("Changes to the campaign can only be proposed in a conversation with the DM alone "
             "(a direct message), not here.")
_ENTRY_MAX = 4000


def _dm_view(params: dict) -> bool:
    return bool((params or {}).pop("_dm_view", False))


def entry_text(entry_html: str) -> str:
    """Kanka's HTML entry as plain text: list items on their own lines."""
    t = re.sub(r"(?i)<\s*(br|/p|/li|/h\d|/div|/tr)\s*/?>", "\n", entry_html or "")
    t = re.sub(r"(?i)<\s*li[^>]*>", "- ", t)
    t = html_lib.unescape(re.sub(r"<[^>]+>", "", t))
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def _module_ref(api_url: str) -> Tuple[str, Optional[int]]:
    """'.../campaigns/1/locations/2288193' -> ('locations', 2288193)."""
    m = re.search(r"/campaigns/\d+/([a-z_]+)/(\d+)", api_url or "")
    return (m.group(1), int(m.group(2))) if m else ("", None)


def _entity(client: "KankaClient", campaign_id: int, entity_id: Any) -> dict:
    return client.get(f"campaigns/{campaign_id}/entities/{int(entity_id)}", cache_ttl=300)


def _resolve_entity(client: "KankaClient", campaign_id: int, ref: Any) -> Optional[dict]:
    """An entity by entity id, or by exact (else only) name match."""
    if ref is None or ref == "":
        return None
    if isinstance(ref, int) or str(ref).isdigit():
        r = _entity(client, campaign_id, ref)
        return r.get("data") if isinstance(r, dict) and r.get("data") else None
    r = client.get(f"campaigns/{campaign_id}/search/{str(ref).strip()}", cache_ttl=120)
    hits = (r or {}).get("data", []) if isinstance(r, dict) else []
    exact = [h for h in hits if (h.get("name") or "").lower() == str(ref).strip().lower()]
    pick = exact[0] if len(exact) == 1 else (hits[0] if len(hits) == 1 else None)
    if not pick:
        return None
    r = _entity(client, campaign_id, pick["id"])
    return r.get("data") if isinstance(r, dict) else None


def _link_fields(client: "KankaClient", campaign_id: int, entity_type: str, params: dict,
                 fields: Dict[str, Any]) -> Dict[str, Any]:
    """Map parent/location names or entity ids to what Kanka API 1.0 accepts (1iq0)."""
    out = dict(fields)
    parent = out.pop("parent", params.get("parent"))
    location = out.pop("location", params.get("location"))
    for stale in ("location_id", "parent_location_id"):
        if stale in out and entity_type == "locations" and parent is None:
            parent = out.pop(stale)
    if parent not in (None, ""):
        ent = _resolve_entity(client, campaign_id, parent)
        if not ent:
            raise ValueError(f"No single entry found for parent {parent!r}")
        out["parent_id"] = ent["id"]
    if location not in (None, ""):
        ent = _resolve_entity(client, campaign_id, location)
        if not ent or (ent.get("type") or "") != "location":
            raise ValueError(f"No single location found for {location!r}")
        if entity_type == "locations":
            out["parent_id"] = ent["id"]
        else:
            out["locations"] = [ent.get("child_id")]
    if "location_id" in out and entity_type != "locations":
        out["locations"] = [out.pop("location_id")]
    return out


def kanka_search(params: dict) -> dict:
    """Search across all entity types within a campaign."""
    dm = _dm_view(params)
    query = (params.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    campaign_id = _resolve_campaign_id(params)
    client = _get_client()

    result = client.get(f"campaigns/{campaign_id}/search/{query}", cache_ttl=120)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    entities = result.get("data", [])
    shown = [e for e in entities if dm or not e.get("is_private")]
    out = {
        "ok": True,
        "campaign_id": campaign_id,
        "query": query,
        "results": [
            {
                "entity_id": e.get("id"),
                "name": e.get("name"),
                "kind": e.get("entity_type") or e.get("child_type"),
                "type": e.get("type"),
                "is_private": e.get("is_private", False),
                "url": e.get("url", ""),
            }
            for e in shown
        ],
        "count": len(shown),
    }
    # Asked about one entry by name, she searched, said "I'll check the entity
    # page" and stopped: one tool per turn (2026-11-17). An exact name match
    # comes with its text.
    exact = [e for e in shown if (e.get("name") or "").strip().lower() == query.lower()]
    if len(exact) == 1:
        got = kanka_get_entity({"entity_id": exact[0]["id"], "campaign_id": campaign_id, "_dm_view": dm})
        if got.get("ok"):
            # title + content: what tool_result_text shows as a page.
            others = [e.get("name") for e in shown if e is not exact[0]]
            out["title"] = got["name"] + (f" ({got['type']})" if got.get("type") else "")
            out["content"] = (got.get("entry") or "(no text yet)") + (
                "\n\nOther entries matching: " + ", ".join(others) if others else "")
    # Said as a fact either way (an empty result with only "public entries
    # shown" read as room to fill: she described the Witcher's leshen,
    # 2026-11-17), and worded the same whether DM-only matches exist or not.
    if not shown:
        out["note"] = (f"Nothing in the campaign matches {query!r}" +
                       ("." if dm else " among the entries that can be shared in this conversation "
                                       "(only the campaign's public entries can be)."))
    elif not dm:
        out["note"] = "Only the campaign's public entries can be shared in this conversation."
    return out


def kanka_list_entities(params: dict) -> dict:
    """List entities of a given type, with optional name filter and pagination."""
    dm = _dm_view(params)
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
    shown = [e for e in entities if dm or not e.get("is_private")]
    meta = result.get("meta", {})
    links = result.get("links", {})

    out = {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_type": entity_type,
        "entities": [
            {
                "entity_id": e.get("entity_id"),
                "name": e.get("name"),
                "type": e.get("type", ""),
                "is_private": e.get("is_private", False),
                "updated_at": e.get("updated_at", ""),
            }
            for e in shown
        ],
        "count": len(shown),
        "page": page,
        "last_page": meta.get("last_page"),
        "has_next": links.get("next") is not None,
    }
    if dm:
        out["total"] = meta.get("total")
    else:
        out["note"] = "Only the campaign's public entries can be shared in this conversation."
    if not shown:
        out["note"] = f"No {entity_type} on this page" + ("." if dm else " that can be shared in this conversation.")
    return out


def kanka_get_entity(params: dict) -> dict:
    """One entry by its entity_id (from search or list): what it is and its text."""
    dm = _dm_view(params)
    entity_id = params.get("entity_id")
    campaign_id = _resolve_campaign_id(params)
    if not entity_id:
        raise ValueError("entity_id is required")

    client = _get_client()
    result = _entity(client, campaign_id, entity_id)
    if isinstance(result, dict) and result.get("ok") is False:
        # An older caller may pass a module id with its entity_type.
        et = (params.get("entity_type") or "").strip().lower()
        if et in _VALID_ENTITY_TYPES:
            mod = client.get(f"campaigns/{campaign_id}/{et}/{int(entity_id)}", cache_ttl=300)
            if isinstance(mod, dict) and mod.get("data", {}).get("entity_id"):
                result = _entity(client, campaign_id, mod["data"]["entity_id"])
        if isinstance(result, dict) and result.get("ok") is False:
            return result

    e = result.get("data", {}) or {}
    if e.get("is_private") and not dm:
        return {"ok": False, "error": _DM_ONLY}
    child = e.get("child") or {}
    text = entry_text(child.get("entry") or "")
    out = {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_id": e.get("id"),
        "name": e.get("name"),
        "kind": (e.get("module") or {}).get("plural", "").lower() or e.get("type"),
        "type": e.get("type_field") or child.get("type") or "",
        "is_private": e.get("is_private", False),
        "parent_entity_id": e.get("parent_id"),
        "entry": text[:_ENTRY_MAX] + (" ... [cut]" if len(text) > _ENTRY_MAX else ""),
        "url": (e.get("urls") or {}).get("view", ""),
    }
    return out


def kanka_create_entity(params: dict) -> dict:
    """Create a new entry. Private (DM-only) unless public is true. Requires approval."""
    if not _dm_view(params):
        return {"ok": False, "error": _NOT_HERE}
    entity_type = _validate_entity_type(params.get("entity_type") or "")
    campaign_id = _resolve_campaign_id(params)
    name = (params.get("name") or "").strip()
    entry = params.get("entry", "")
    extra_fields = params.get("fields") or {}

    if not name:
        raise ValueError("name is required")
    if not isinstance(extra_fields, dict):
        raise ValueError("fields must be an object")

    client = _get_client()
    body: Dict[str, Any] = {"name": name, "is_private": not bool(params.get("public", False))}
    if entry:
        body["entry"] = entry
    if params.get("type"):
        body["type"] = params["type"]
    body.update(_link_fields(client, campaign_id, entity_type, params, extra_fields))

    result = client.post(f"campaigns/{campaign_id}/{entity_type}", body)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    client.invalidate_cache()
    d = result.get("data", result) or {}
    return {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_type": entity_type,
        "created": {"entity_id": d.get("entity_id"), "name": d.get("name"),
                    "is_private": d.get("is_private"), "url": (d.get("urls") or {}).get("view", "")},
    }


def kanka_update_entity(params: dict) -> dict:
    """Change an entry by its entity_id. append_entry adds to the text instead of
    replacing it. Requires approval."""
    if not _dm_view(params):
        return {"ok": False, "error": _NOT_HERE}
    campaign_id = _resolve_campaign_id(params)
    fields = params.get("fields") or {}
    append = (params.get("append_entry") or "").strip()

    client = _get_client()
    ref = params.get("entity_id") or params.get("entity")
    if not ref:
        raise ValueError("entity is required: the entry's name or entity_id")
    found = _resolve_entity(client, campaign_id, ref)
    if not found:
        raise ValueError(f"No single campaign entry is named {ref!r}")
    entity_id = found["id"]
    if not isinstance(fields, dict):
        raise ValueError("fields must be an object")
    if not fields and not append and params.get("parent") is None and params.get("location") is None:
        raise ValueError("nothing to change: give fields, append_entry, parent or location")

    ent = _entity(client, campaign_id, entity_id)
    if isinstance(ent, dict) and ent.get("ok") is False:
        return ent
    e = ent.get("data", {}) or {}
    entity_type, module_id = _module_ref((e.get("urls") or {}).get("api", ""))
    if not module_id:
        return {"ok": False, "error": f"Couldn't tell what kind of entry {entity_id} is."}

    body = _link_fields(client, campaign_id, entity_type, params, fields)
    if append:
        current = (e.get("child") or {}).get("entry") or ""
        para = "".join(f"<p>{html_lib.escape(line)}</p>" for line in append.splitlines() if line.strip())
        body["entry"] = current + para
    result = client.patch(f"campaigns/{campaign_id}/{entity_type}/{module_id}", body)

    if isinstance(result, dict) and result.get("ok") is False:
        return result

    client.invalidate_cache()
    d = result.get("data", result) or {}
    return {
        "ok": True,
        "campaign_id": campaign_id,
        "entity_id": e.get("id"),
        "updated": {"name": d.get("name"), "changed": sorted(body),
                    "url": (e.get("urls") or {}).get("view", "")},
    }


def name_hash(name: str) -> str:
    return hashlib.sha256(" ".join((name or "").lower().split()).encode()).hexdigest()


def kanka_name_hashes(params: dict) -> dict:
    """Hashes of every entry's name, for gaia-core to tell the campaign's own
    names from words worth a web search ("Leshen" is a city here, not the
    Witcher monster, 2026-11-17). Hashes, not names: DM-only names are
    spoilers too, and a match needs the name already in the message."""
    campaign_id = _resolve_campaign_id(params or {})
    client = _get_client()
    hashes, words, page = set(), 1, 1
    while page <= 20:
        r = client.get(f"campaigns/{campaign_id}/entities", params={"page": page}, cache_ttl=1800)
        if isinstance(r, dict) and r.get("ok") is False:
            if not hashes:
                return r
            break
        for e in r.get("data", []):
            n = e.get("name") or ""
            if n.strip():
                hashes.add(name_hash(n))
                words = max(words, len(n.split()))
        if not (r.get("links") or {}).get("next"):
            break
        page += 1
    return {"ok": True, "campaign_id": campaign_id, "hashes": sorted(hashes), "max_words": min(words, 8)}


_WRITE_KEYS = {
    "kanka_create_entity": {"entity_type", "name", "entry", "type", "parent", "location", "public",
                            "fields", "campaign", "campaign_id"},
    "kanka_update_entity": {"entity", "entity_id", "append_entry", "fields", "parent", "location",
                            "campaign", "campaign_id"},
}


def write_problem(method: str, params: dict) -> Optional[str]:
    """Why a proposed campaign change couldn't run, said before anyone is asked
    to approve it: she once proposed worldbuild(update) with entity_id null and
    made-up "key"/"value" fields (2026-11-17)."""
    if method not in _WRITE_KEYS:
        return None
    p = dict(params or {})
    if not p.pop("_dm_view", False):
        return _NOT_HERE
    p.pop("_allow_pending", None)
    unknown = sorted(k for k in p if k not in _WRITE_KEYS[method] and not k.startswith("_"))
    if unknown:
        return (f"{method} doesn't take {', '.join(unknown)}; it takes "
                f"{', '.join(sorted(_WRITE_KEYS[method] - {'campaign_id', 'entity_id'}))}")
    try:
        if method == "kanka_create_entity":
            _validate_entity_type(p.get("entity_type") or "")
            if not (p.get("name") or "").strip():
                return "a new entry needs a name"
            return None
        if not (p.get("append_entry") or p.get("fields") or p.get("parent") or p.get("location")):
            return "nothing to change: give append_entry (text to add), fields, parent or location"
        ref = p.get("entity_id") or p.get("entity")
        if not ref:
            return "say which entry: entity is its name"
        if not _resolve_entity(_get_client(), _resolve_campaign_id(p), ref):
            return f"no single campaign entry is named {ref!r}"
    except ValueError as e:
        return str(e)
    return None
