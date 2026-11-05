"""People — the palace's record of the humans she knows (GAIA_Project-751u).

Asked about Azrael's friend Andy, she called Azrael "Andy": nothing told her
who she was speaking with. One JSON file per person under <palace>/people/:
a permanent id, names, the accounts and sessions that are them, and facts,
each with who said it, when and where.

Written only by code, never by a model-facing tool. What it holds is what was
said or what happened, not her guesses: a guess kept here would be a permanent
fact about someone. A corrected fact stays, marked wrong, so the failure can
still be seen.

Trust: a Discord account is that person. A web or voice session is whoever it
says it is. Azrael's call (2026-11-05): he is present for anyone on his portal.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.People")

DIRNAME = "people"
_DISCORD_DM_RE = re.compile(r"^discord_dm_(\d{17,20})$")
_CLAIM_RE = re.compile(r"\b(?:[Mm]y name is|I am|I'm|[Tt]his is|[Cc]all me)\s+([A-Z][a-z]{1,20})\b")
# Capitalized, but how someone is rather than who ("I'm Happy with that").
_NOT_A_NAME = frozenset("""
Good Great Fine Okay Sure Sorry Happy Glad Back Here Done Ready Not Just Still Also So Very Really
Thanks Tired Busy Well Home Going Trying Looking Working Curious Confused Excited
GAIA Gaia
""".split())

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_root() -> Path:
    from gaia_common.utils.conversation_memory import default_root as conversations
    return conversations().parent / DIRNAME


def claimed_name(text: str) -> Optional[str]:
    """The name in "Hello GAIA. I am Andy.", if there is one."""
    for match in _CLAIM_RE.finditer(text or ""):
        if match.group(1) not in _NOT_A_NAME:
            return match.group(1)
    return None


class PeopleRegistry:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_root()

    def _path(self, person_id: str) -> Path:
        return self.root / f"{re.sub(r'[^A-Za-z0-9_-]', '_', person_id)}.json"

    def _write(self, person: Dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        person["updated_at"] = _now()
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(person, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path(person["id"]))

    def everyone(self) -> List[Dict]:
        people = []
        for path in sorted(self.root.glob("*.json")) if self.root.is_dir() else []:
            try:
                people.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                logger.warning("People: unreadable %s", path.name)
        return people

    def get(self, person_id: str) -> Optional[Dict]:
        return next((p for p in self.everyone() if p.get("id") == person_id), None)

    def find_by_name(self, name: str) -> List[Dict]:
        """Everyone answering to a name: there can be two Andys."""
        wanted = (name or "").strip().lower()
        return [p for p in self.everyone() if wanted in {n.lower() for n in p.get("names", [])}]

    def for_session(self, session_id: str) -> Optional[Dict]:
        """Who a session is: its Discord account, or a session they claimed."""
        dm = _DISCORD_DM_RE.match(session_id or "")
        for person in self.everyone():
            for account in person.get("accounts", []):
                if account.get("kind") == "session" and account.get("id") == session_id:
                    return person
                if dm and account.get("kind") == "discord" and account.get("id") == dm.group(1):
                    return person
        return None

    def create(self, name: str, accounts: Optional[List[Dict]] = None, relationship: str = "",
               source: str = "") -> Dict:
        person = {"id": f"person_{uuid.uuid4().hex[:12]}", "names": [name], "relationship": relationship,
                  "accounts": [{**a, "linked_at": _now(), "source": source} for a in accounts or []],
                  "facts": [], "created_at": _now(), "source": source}
        with _lock:
            self._write(person)
        return person

    def link_session(self, person_id: str, session_id: str, source: str = "") -> Optional[Dict]:
        with _lock:
            person = self.get(person_id)
            if not person:
                return None
            if not any(a.get("id") == session_id for a in person.get("accounts", [])):
                person.setdefault("accounts", []).append(
                    {"kind": "session", "id": session_id, "linked_at": _now(), "source": source})
                self._write(person)
        return person

    def add_fact(self, person_id: str, text: str, told_by: str, where: str = "") -> Optional[Dict]:
        """Keep what someone said about a person, and who said it."""
        with _lock:
            person = self.get(person_id)
            if not person or not (text or "").strip():
                return None
            fact = {"id": uuid.uuid4().hex[:12], "text": text.strip(), "told_by": told_by, "where": where,
                    "when": _now(), "status": "told"}
            person.setdefault("facts", []).append(fact)
            self._write(person)
        return fact

    def correct_fact(self, person_id: str, fact_id: str, why: str = "", by: str = "") -> bool:
        """A wrong fact stays on file, marked, and stops being said."""
        with _lock:
            person = self.get(person_id)
            fact = next((f for f in (person or {}).get("facts", []) if f.get("id") == fact_id), None)
            if not fact:
                return False
            fact.update(status="wrong", corrected_at=_now(), corrected_by=by, why=why)
            self._write(person)
        return True

    def note_claim(self, session_id: str, text: str) -> Optional[Dict]:
        """A web or voice session that says who it is, taken at its word. A
        name already on file is that person; a new name is a new person."""
        if not re.match(r"^(web_|voice_)", session_id or "") or self.for_session(session_id):
            return None
        name = claimed_name(text)
        if not name:
            return None
        known = self.find_by_name(name)
        if len(known) == 1:
            logger.info("People: %s says they are %s", session_id, name)
            return self.link_session(known[0]["id"], session_id, source="said so")
        if known:
            return None    # two people by that name: it can't say which
        logger.info("People: %s introduced themselves as %s", session_id, name)
        return self.create(name, [{"kind": "session", "id": session_id}], source="said so")


def speaker_line(session_id: str, registry: Optional[PeopleRegistry] = None) -> str:
    """What she is told about who she's speaking with. Empty for her own loops
    and test batteries, which aren't anyone."""
    from gaia_common.utils.conversation_memory import recorded
    if not recorded(session_id):
        return ""
    try:
        person = (registry or PeopleRegistry()).for_session(session_id)
    except OSError:
        return ""
    if not person:
        return ("You don't know who you're speaking with yet. If they tell you, take their word for it.")
    line = f"You are speaking with {person['names'][0]}"
    if person.get("relationship"):
        line += f", {person['relationship']}"
    line += ". Speak to them as \"you\", never in the third person."
    facts = [f"{f['text']} (per {f.get('told_by') or 'someone'})"
             for f in person.get("facts", []) if f.get("status") != "wrong"][:5]
    if facts:
        line += " What you've been told about them: " + "; ".join(facts) + "."
    return line
