"""A person asks her to add a site to the sites she can read (2026-11-20).

"Can you add scribd.com to your web search allow list?" She explained the
tool, or said she couldn't. A request this plain is recognized by code: the
domain goes to web(action=trust_domain), which parks behind Azrael's
authenticator code, and she shows the approval card like any gated action.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger("GAIA.SiteRequest")

_DOMAIN = r"(?:https?://)?(?:www\.)?([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]{1,63})*\.[a-z]{2,63})"
_LIST = r"(?:(?:web\s+)?(?:search\s+)?(?:allow\s*-?\s*list|white\s*-?\s*list|safe\s*list|trusted\s+(?:list|sites?|websites?|domains?)|approved\s+sites?|allowed\s+(?:list|sites?|websites?)))"
_ASK_RE = re.compile(
    rf"\b(?:add|put|include|allow|trust|whitelist|allowlist)\s+{_DOMAIN}(?:\s+(?:to|on|in)\s+(?:your|the|her)?\s*{_LIST})?"
    rf"|\b(?:add|put)\s+{_DOMAIN}\b"
    rf"|\b{_LIST}\b[^.?!\n]{{0,40}}?\b{_DOMAIN}",
    re.IGNORECASE)
_ABOUT_LISTS = re.compile(_LIST + r"|\btrust(?:ed)?\b", re.IGNORECASE)


def requested_domain(text: str) -> Optional[str]:
    body = re.sub(r"^[^:\n]{1,64}:\s", "", text or "", count=1)
    if not _ABOUT_LISTS.search(body):
        return None
    m = _ASK_RE.search(body)
    if not m:
        return None
    domain = next((g for g in m.groups() if g), "")
    return domain.lower() or None


def handle(user_input: str, session_id: str, session_manager, call=None) -> Optional[str]:
    """Her reply if this is a request to add a site, else None."""
    from gaia_core.cognition.ambient_interject import not_for_her
    domain = requested_domain(user_input)
    if not domain or not_for_her(user_input):
        return None
    speaker = re.match(r"^\s*([^:\n]{1,40}):", user_input or "")
    reason = f"asked for by {speaker.group(1)} in chat" if speaker else "asked for in chat"
    params = {"action": "trust_domain", "domain": domain, "reason": reason}
    if call is None:
        from gaia_core.utils.mcp_client import call_jsonrpc as call
    r = call("web", params) or {}
    if r.get("pending_approval"):
        label = "web(trust_domain)"
        shown = {"domain": domain, "reason": reason}
        try:
            from gaia_core.cognition import approval_reply
            approval_reply.remember(session_manager, session_id, r.get("action_id"), label, shown)
        except Exception:
            logger.warning("site request: could not remember the pending approval", exc_info=True)
        return (f"I've asked to add {domain} to the sites I can read. "
                f"That needs your approval first: `{label}` {json.dumps(shown)}. "
                f"Reply with your current 6-digit authenticator code to approve, or say cancel.")
    err = str(r.get("error") or "")
    m = re.search(r"can't run: (.*)$", err)
    if m:
        return f"I can't add that one: {m.group(1).rstrip('.')}."
    if r.get("ok"):
        return f"{domain} is now a site I can read."
    logger.info("site request for %s: %s", domain, err[:200])
    return f"I tried to ask for {domain}, but the request didn't go through ({err[:120] or 'no answer'})."
