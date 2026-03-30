"""Persistent DM blocklist for Discord users.

Mirrors the VoiceWhitelist pattern: thread-safe JSON persistence,
seen-user tracking for the dashboard, and block/unblock operations.
Blocked users' DMs are silently ignored (no wake, no response).

Also tracks when GAIA last replied to each DM user, so the typing-wake
feature can gate on "GAIA replied within the past N hours".
"""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("GAIA.Web.DMBlocklist")


class DMBlocklist:
    """Persistent blocklist of Discord users whose DMs GAIA ignores.

    Also tracks all DM users so the dashboard can offer a selectable
    list to toggle blocking.
    """

    def __init__(self, data_dir: str = "/app/data") -> None:
        self._path = Path(data_dir) / "dm_blocklist.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"blocked": [], "dm_users": {}}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = self._path.read_text()
                self._data = json.loads(raw)
            except Exception:
                logger.warning("Failed to load DM blocklist; starting fresh")
                self._data = {"blocked": [], "dm_users": {}}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))
        except Exception:
            logger.error("Failed to save DM blocklist", exc_info=True)

    # -- Blocklist operations --

    def block(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._data["blocked"]:
                self._data["blocked"].append(user_id)
                self._save()

    def unblock(self, user_id: str) -> None:
        with self._lock:
            if user_id in self._data["blocked"]:
                self._data["blocked"].remove(user_id)
                self._save()

    def is_blocked(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._data["blocked"]

    def get_blocked(self) -> list[str]:
        with self._lock:
            return list(self._data["blocked"])

    # -- DM user tracking --

    def record_dm(self, user_id: str, name: str) -> None:
        """Record that a user sent a DM. Updates last_dm and message_count."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            entry = self._data["dm_users"].get(user_id, {
                "name": name,
                "first_dm": now,
                "last_dm": now,
                "last_gaia_reply": None,
                "message_count": 0,
            })
            entry["name"] = name
            entry["last_dm"] = now
            entry["message_count"] = entry.get("message_count", 0) + 1
            self._data["dm_users"][user_id] = entry
            self._save()

    def record_gaia_reply(self, user_id: str) -> None:
        """Record that GAIA sent a reply to this DM user."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            entry = self._data["dm_users"].get(user_id)
            if entry:
                entry["last_gaia_reply"] = now
                self._save()

    def has_recent_gaia_reply(self, user_id: str, hours: int = 48) -> bool:
        """Check if GAIA replied to this user within the past `hours` hours."""
        with self._lock:
            entry = self._data["dm_users"].get(user_id)
            if not entry:
                return False
            last_reply = entry.get("last_gaia_reply")
            if not last_reply:
                return False
            try:
                ts = datetime.fromisoformat(last_reply)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
                return ts > cutoff
            except (ValueError, TypeError):
                return False

    def get_dm_users(self) -> list[dict]:
        """Return all DM users with blocked status, for dashboard display."""
        with self._lock:
            result = []
            for uid, info in self._data["dm_users"].items():
                result.append({
                    "user_id": uid,
                    "name": info.get("name", "Unknown"),
                    "first_dm": info.get("first_dm", ""),
                    "last_dm": info.get("last_dm", ""),
                    "last_gaia_reply": info.get("last_gaia_reply"),
                    "message_count": info.get("message_count", 0),
                    "blocked": uid in self._data["blocked"],
                })
            return sorted(result, key=lambda u: u["name"].lower())
