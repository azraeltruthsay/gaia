"""9nxd: a reply that says she did something must match a tool that did it.

Asked to add a note to an issue, she made no tool call and answered "Noted:
gi-000000 — Checked Again". Asked what happened, she looked in her memory and
said "The note was added successfully." Nothing had run. Such a reply enters
her history as a thing she did (gb7y was the heartbeat form of it), and the
next turn builds on it.

This code reads the reply for a claim of a finished action ("I've added",
"the note was saved", "Noted: ...") and checks it against the tools this
session actually ran (outcome_ledger.session_tools: every path records there).
A claim with no successful write behind it gets a correction that says what
really happened, shown to the person and kept in her history. The check is
plain pattern matching, no model: a claim is a fact about her own tool calls,
and those are known exactly.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# Finished actions she might claim. Future ("I'll add") and negated forms
# ("I haven't saved", "wasn't added") don't match.
_VERBS = (r"added|filed|created|saved|recorded|logged|stored|written|wrote|updated|committed|"
          r"deleted|removed|appended|noted|remembered|kept|pinned|scheduled|sent|posted|messaged")
# Saying it reached someone needs a call that sends: asked to post in the
# Grotto, she stored a palace note and said "Sent to the Grotto" (2026-11-12);
# any write backed it.
_SEND_RE = re.compile(
    r"\bI(?:'ve| have| just|'ve just)?\s+(?:sent|posted|messaged)\b"
    r"|(?:^|(?<=[.!]\s))\s*\**(?:sent|posted)\b"
    r"|\bit'?s (?:in|up in|posted in) the \w+ now\b"
    r"|\b(?:message|it|that|note)\s+(?:was|has been|is)\s+(?:now\s+)?(?:sent|posted)\b", re.I | re.M)
_SENDS = {("discord", "post"), ("discord_post", "")}
_FIRST_PERSON = re.compile(rf"\bI(?:'ve| have| just|'ve just| have just)?\s+(?:successfully\s+)?(?:{_VERBS})\b", re.I)
# Passive claims need a subject she could have acted on, or "successfully":
# "the note was added" is a claim, "Python was created in 1991" is not.
_SUBJECTS = (r"note|notes|issue|issues|entry|entries|file|memory|memories|record|reminder|task|ticket|"
             r"fact|facts|change|changes|it|that|this|comment|update|item|bead")
_PASSIVE = re.compile(
    rf"\b(?:(?:the|your|my|a|this|that)\s+)?(?:{_SUBJECTS})\s+(?:(?:has|have|was|were)\s+(?:now\s+)?(?:been\s+)?|is\s+now\s+)"
    rf"(?:successfully\s+)?(?:{_VERBS})\b"
    rf"|\b(?:{_VERBS})\s+successfully\b", re.I)
# "Noted: gi-000000", "Noted in the issue ledger.", "Saved to memory." (a bare
# "Noted." is an acknowledgement, not a claim)
_NOTED = re.compile(r"(?:^|(?<=[.!]\s))\s*\**(?:noted|added|saved|filed|recorded|logged|stored)\**"
                    r"\s*(?:[:—–-]|(?:in|to|into|on|under|as)\b)", re.I | re.M)
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_QUOTE = re.compile(r"^\s*>.*$", re.M)

# Actions that only read. Anything else that ran and succeeded is a write
# that could back a claim (kept generous: a missed correction costs less
# than a wrong one).
_READS = {
    "find", "list", "read", "tree", "count_chars", "describe", "events", "logs", "recall", "tools",
    "turns", "world", "kg_query", "kg_stats", "kg_timeline", "memory", "query", "search", "status",
    "lookup", "lookups", "navigate", "remembered", "get", "campaigns", "notes", "sources", "artifacts",
    "fetch", "browse", "links", "snapshot", "screenshot", "forms", "info", "adapter_info", "adapter_list",
    "assess", "blueprint", "promote_list", "promote_status", "inbox_list", "inbox_status",
    "listen_status", "fragment_list", "fragment_read", "chat",
}
_READ_DOMAINS = {"web", "introspect", "recall", "worldstate", "expand_context"}
_READ_WORDS = {"list", "read", "find", "search", "get", "query", "fetch", "describe", "status", "lookup", "recall"}


def is_write(tool: str, action: str = "") -> bool:
    tool, action = (tool or "").lower(), (action or "").lower()
    if tool == "issue":
        return action not in ("list", "")
    if tool in _READ_DOMAINS:
        return False
    if action:
        return action not in _READS
    # a legacy name with no action: read_file, list_files, web_search ...
    return not (set(re.split(r"[_\W]+", tool)) & _READ_WORDS)


def claims(reply: str) -> List[str]:
    """The phrases in a reply that claim a finished action."""
    text = _QUOTE.sub("", _CODE.sub("", reply or ""))
    found = [m.group(0).strip() for rx in (_NOTED, _FIRST_PERSON, _PASSIVE) for m in rx.finditer(text)]
    return list(dict.fromkeys(found))


Call = Tuple[float, str, str, bool]


def check(reply: str, calls: Sequence[Call]) -> Optional[dict]:
    """None if the reply claims nothing, or a write that succeeded backs it.
    Otherwise {"claims", "failed": [labels of writes that failed]}."""
    said = claims(reply)
    if _SEND_RE.search(_QUOTE.sub("", _CODE.sub("", reply or ""))):
        said = list(dict.fromkeys(said + [_SEND_RE.search(reply).group(0)]))
        sends = [c for c in calls if ((c[1] or "").lower(), (c[2] or "").lower()) in _SENDS]
        if not any(c[3] for c in sends):
            failed = [f"{c[1]}({c[2]})" if c[2] else c[1] for c in sends]
            return {"claims": said, "failed": list(dict.fromkeys(failed)), "sent": True}
    if not said:
        return None
    writes = [c for c in calls if is_write(c[1], c[2])]
    if any(c[3] for c in writes):
        return None
    failed = [f"{c[1]}({c[2]})" if c[2] else c[1] for c in writes]
    return {"claims": said, "failed": list(dict.fromkeys(failed))}


def correction(result: dict) -> str:
    if result.get("sent") and not result["failed"]:
        return ("Correction: I didn't actually send that. Nothing that posts a message ran, "
                "so no one has received it.")
    if result["failed"]:
        return (f"Correction: that didn't go through. {', '.join(result['failed'][:3])} failed, "
                "so nothing was changed.")
    return ("Correction: I didn't actually do that. No tool that saves or changes anything ran, "
            "so nothing was added or changed.")
