"""Posting to her Discord channel (2026-11-12).

Asked by Azrael to tell Nick in the Grotto something, she had no way to post:
she stored a palace note and said "It's in the Grotto now". The Grotto is
essentially her channel (Azrael), so she can post there.

Only channels on the allowlist can be posted to: GAIA_DISCORD_CHANNELS,
"name:id" pairs separated by commas (default: the Grotto). gaia-web's bot sends
it; gaia-core keeps it in that channel's history, so when someone answers she
knows what she said.
"""

import logging
import os
import re
from typing import Dict

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

logger = logging.getLogger("GAIA.MCP.Discord")

WEB_URL = os.environ.get("GAIA_WEB_URL", "http://gaia-web:6414")
CORE_URL = os.environ.get("GAIA_CORE_URL", "http://gaia-core:6415")
DEFAULT_CHANNELS = "grotto:1484003096337191067"
MAX_CHARS = 1900   # Discord's limit is 2000


def channels() -> Dict[str, str]:
    """{name: channel id} she may post to."""
    out = {}
    for pair in os.environ.get("GAIA_DISCORD_CHANNELS", DEFAULT_CHANNELS).split(","):
        name, _, cid = pair.strip().partition(":")
        if name and cid.strip().isdigit():
            out[name.strip().lower()] = cid.strip()
    return out


def _channel_id(name: str) -> str:
    allowed = channels()
    key = re.sub(r"^the\s+", "", (name or "").strip().lower()) or next(iter(allowed), "")
    if key in allowed:
        return allowed[key]
    if key in allowed.values():
        return key
    raise ValueError(f"not a channel she can post to: {name!r} (allowed: {', '.join(allowed) or 'none'})")


async def discord_post(params: dict) -> dict:
    """Post a message in one of her Discord channels (the Grotto by default)."""
    message = (params.get("message") or params.get("text") or params.get("content") or "").strip()
    if not message:
        raise ValueError("message is required")
    if len(message) > MAX_CHARS:
        raise ValueError(f"message is {len(message)} characters; Discord takes {MAX_CHARS}")
    channel_id = _channel_id(params.get("channel") or "")
    session_id = f"discord_channel_{channel_id}"
    if httpx is None:
        return {"ok": False, "error": "httpx not available"}

    async with httpx.AsyncClient(timeout=20.0) as http:
        r = await http.post(f"{WEB_URL}/api/conversations/{session_id}/push",
                            json={"text": message, "source": "discord_post"})
        delivered = r.status_code == 200 and bool(r.json().get("delivered"))
        if not delivered:
            return {"ok": False, "error": f"the Discord bot didn't deliver it ({r.status_code}: {r.text[:200]})"}
        try:
            await http.post(f"{CORE_URL}/api/sessions/{session_id}/posted",
                            json={"text": message, "via": "discord_post"})
        except Exception as e:  # posted; only her record of it failed
            logger.warning("discord_post: posted, but not kept in %s history: %s", session_id, e)
    logger.info("discord_post: posted %d chars to %s", len(message), session_id)
    return {"ok": True, "posted_to": params.get("channel") or "grotto", "channel_id": channel_id,
            "message": message}
