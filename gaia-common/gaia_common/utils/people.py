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

# "My pronouns are she/her", "I use they/them pronouns", "I go by he/him".
# 2026-11-11: with nothing on file she guessed ("the architect herself").
_PRONOUNS_RE = re.compile(
    r"\b(?:my\s+pronouns\s+(?:are|is)|pronouns\s*:|I\s+(?:use|go\s+by|prefer))\s+"
    r"([a-z]{1,5}/[a-z]{1,6}(?:/[a-z]{1,6})?)\b", re.IGNORECASE)
# What she's told when none are on file: no guess at anyone's gender.
NO_PRONOUNS = "their pronouns aren't on record, so for them use their name or \"they\""

CREATOR = "Azrael"     # named as her creator in every system prompt

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


def stated_pronouns(text: str) -> Optional[str]:
    """"she/her" from "btw my pronouns are she/her", if someone says theirs."""
    m = _PRONOUNS_RE.search(text or "")
    return m.group(1).lower() if m else None


def pronoun_words(person: Dict) -> str:
    """How she is told to speak of someone: their stated pronouns, or no guess.
    By name: "use their name or they" left Azrael "he" 2 in 3 (i4vn). Phrased
    as a writing instruction, not a fact about them ("X's pronouns are Y") —
    the fact-shaped version got echoed back unprompted (2026-11-12, GAIA's
    Grotto: "his name's Nick and his pronouns are he/him", when nobody had
    asked). A direct "when you write, do Y" instruction, like the one right
    next to this in speaker_line ("Speak to them as \"you\", never in the
    third person"), doesn't invite being relayed as content the same way."""
    name = _main_name(person) if (person or {}).get("names") else "them"
    pronouns = (person or {}).get("pronouns")
    if pronouns:
        return f"When you write about {name}, use \"{pronouns}\" — just use it, don't mention it"
    return (f"When you write about {name}, say \"{name}\" or \"they\" — no pronouns are "
            f"on record, and that's for your own phrasing, not something to bring up")


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

    def set_pronouns(self, person_id: str, pronouns: str, source: str = "") -> Optional[Dict]:
        """Pronouns as the person gave them. Never inferred from a name."""
        pronouns = (pronouns or "").strip().lower()
        with _lock:
            person = self.get(person_id)
            if not person or not pronouns:
                return None
            if person.get("pronouns") != pronouns:
                person.update(pronouns=pronouns, pronouns_source=source, pronouns_at=_now())
                self._write(person)
        return person

    def note_pronouns(self, session_id: str, text: str, speaker: Optional[Dict] = None) -> Optional[Dict]:
        """The speaker saying their own pronouns, kept for whoever the session
        is, or for `speaker`: in a channel the session isn't anyone, the
        message's author is."""
        pronouns = stated_pronouns(text)
        if not pronouns:
            return None
        person = speaker or self.for_session(session_id)
        if not person:
            return None
        logger.info("People: %s gave their pronouns", person["id"])
        return self.set_pronouns(person["id"], pronouns, source=f"said so in {session_id}")

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
        return ("You don't know who you're speaking with yet. If they tell you, take their word for it. "
                "Until they say theirs, " + NO_PRONOUNS + ".")
    line = f"You are speaking with {person['names'][0]}"
    if person.get("relationship"):
        line += f", {person['relationship']}"
    line += ". Speak to them as \"you\", never in the third person. " + pronoun_words(person) + "."
    facts = [f"{f['text']} (per {f.get('told_by') or 'someone'})"
             for f in person.get("facts", []) if f.get("status") != "wrong"][:5]
    if facts:
        line += " What you've been told about them: " + "; ".join(facts) + "."
    return line


_NAME_WORD_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")


def mentioned_line(text: str, session_id: str = "", registry: Optional[PeopleRegistry] = None) -> str:
    """What she knows of the people a message names, other than the speaker.

    2026-11-11: "tomorrow I'm probably going to see Andy again" — she knew
    Andy as Azrael's friend, but only the speaker was ever described to her,
    so she guessed, 3 in 3, that Andy was her architect."""
    try:
        registry = registry or PeopleRegistry()
        speaker = registry.for_session(session_id) if session_id else None
    except OSError:
        return ""
    from gaia_common.utils.conversation_memory import channel_of, local_day
    lines, seen = [], {(speaker or {}).get("id")}
    for m in _NAME_WORD_RE.finditer(text or ""):
        for name in (m.group(0), m.group(0).split()[0]):
            found = registry.find_by_name(name)
            if len(found) != 1 or found[0].get("id") in seen:
                continue
            person = found[0]
            seen.add(person.get("id"))
            name = person['names'][0]
            line = f"{name}, whom they mention"
            if person.get("relationship"):
                line += f", is {person['relationship']}"
            met = [a for a in person.get("accounts", []) if a.get("kind") == "session"]
            if met:
                # First, before anything else: with the pronoun clause between
                # "is Azrael's friend" and "you have spoken with them", she
                # said "I haven't met them yet" (2026-11-11).
                line += (f". You have met {name}: you spoke with them in a "
                         f"{channel_of(met[0].get('id', ''))} on {local_day(met[0].get('linked_at') or '')}")
            facts = [f["text"] for f in person.get("facts", []) if f.get("status") != "wrong"][:3]
            if facts:
                line += ". What you've been told about them: " + "; ".join(facts)
            line += f". {pronoun_words(person)}"
            lines.append(line + ".")
            break
    return " ".join(lines)


def person_for_discord(author_id: str, registry: Optional["PeopleRegistry"] = None) -> Optional[Dict]:
    """The person whose Discord account this is, if anyone's."""
    if not author_id:
        return None
    return next((p for p in (registry or PeopleRegistry()).everyone() for a in p.get("accounts", [])
                 if a.get("kind") == "discord" and a.get("id") == str(author_id)), None)


def creator_line(session_id: str = "", text: str = "", registry: Optional[PeopleRegistry] = None,
                 author_id: str = "") -> str:
    """Azrael's pronouns, or that there are none on file, when Azrael is neither
    the speaker nor named in the message: the system prompt names Azrael as
    her creator every turn, and she guessed "He's also friends with Andy" (i4vn)."""
    try:
        registry = registry or PeopleRegistry()
        found = registry.find_by_name(CREATOR)
        # In a channel the session isn't anyone: the message's author is.
        speaker = (person_for_discord(author_id, registry) if author_id
                   else registry.for_session(session_id) if session_id else None)
    except OSError:
        return ""
    if len(found) != 1 or (speaker or {}).get("id") == found[0].get("id") or re.search(rf"\b{CREATOR}\b", text or ""):
        return ""
    # Someone else speaking took Azrael's name 1/3 ("you're Azrael, my architect").
    return f"{CREATOR}, who created you, is not the one speaking now. " + pronoun_words(found[0]) + "."


# ── Discord channels (kkij, 2026-11-12) ──────────────────────────────────
# In the Grotto, Nick ("Schwiftyboi") joked he'd eat her for dinner; she had no
# idea who was speaking: only DMs and claimed web sessions map to a person.
# Each channel message carries its author's account and display name.
_CHANNEL_RE = re.compile(r"^discord_channel_\d+$")
_speaking: Dict[str, str] = {}   # channel session -> who said the message in hand

# "Schwiftyboi is a friend of mine named Nick", "X is my brother Sam".
_INTRO_RE = re.compile(
    r"\b(?:named |called )?(?P<a>[A-Z][\w-]{2,30}) is (?:actually )?(?:a |my |our )?"
    r"(?P<rel>friend|brother|sister|partner|wife|husband|son|daughter|cousin|coworker|colleague|roommate|neighbou?r)"
    r"(?: of mine| of ours)?,? (?:named |called )?(?P<b>[A-Z][a-z]{1,20})\b")


def is_channel(session_id: str) -> bool:
    return bool(_CHANNEL_RE.match(session_id or ""))


def channel_speaker(session_id: str) -> str:
    """Who said the message being handled in a channel session, for its history."""
    return _speaking.get(session_id, "")


def _alias(registry: "PeopleRegistry", person: Dict, name: str, source: str) -> Dict:
    if name.lower() not in {n.lower() for n in person.get("names", [])}:
        with _lock:
            fresh = registry.get(person["id"]) or person
            fresh.setdefault("names", []).append(name)
            fresh.setdefault("name_sources", {})[name] = source
            registry._write(fresh)
            person = fresh
    return person


def note_channel_author(session_id: str, author_id: str, display_name: str,
                        registry: Optional["PeopleRegistry"] = None) -> Optional[Dict]:
    """The person who wrote a channel message: by their Discord account, else
    by a name someone told her (then the account is linked), else a new record
    of the account under its display name. An account speaking is a fact."""
    if not (is_channel(session_id) and author_id and str(author_id).isdigit()):
        return None
    registry = registry or PeopleRegistry()
    person = next((p for p in registry.everyone() for a in p.get("accounts", [])
                   if a.get("kind") == "discord" and a.get("id") == str(author_id)), None)
    if person is None and display_name:
        named = registry.find_by_name(display_name)
        if len(named) == 1 and not any(a.get("kind") == "discord" for a in named[0].get("accounts", [])):
            person = named[0]
            with _lock:
                person.setdefault("accounts", []).append(
                    {"kind": "discord", "id": str(author_id), "linked_at": _now(), "source": f"spoke in {session_id}"})
                registry._write(person)
            logger.info("People: %s's Discord account linked by display name", person["id"])
    if person is None:
        person = registry.create(display_name or f"Discord user {author_id}",
                                 [{"kind": "discord", "id": str(author_id)}], source=f"spoke in {session_id}")
        logger.info("People: new person %s from %s", person["id"], session_id)
    if display_name:
        person = _alias(registry, person, display_name, f"Discord display name in {session_id}")
        if "display name" not in (person.get("name_sources") or {}).get(display_name, ""):
            with _lock:
                fresh = registry.get(person["id"]) or person
                fresh.setdefault("name_sources", {})[display_name] = f"Discord display name in {session_id}"
                registry._write(fresh)
                person = fresh
    _speaking[session_id] = _main_name(person)
    return person


def note_introduction(text: str, told_by: str, where: str = "",
                      registry: Optional["PeopleRegistry"] = None) -> List[Dict]:
    """"Schwiftyboi is a friend of mine named Nick": Nick is another name for
    Schwiftyboi, and Schwiftyboi is the teller's friend. Kept as told, with who
    told it. A name not on file yet becomes a person with both names, linked to
    an account when that name speaks."""
    registry = registry or PeopleRegistry()
    out = []
    for m in _INTRO_RE.finditer(text or ""):
        a, rel, b = m.group("a"), m.group("rel").lower(), m.group("b")
        if a in _NOT_A_NAME or b in _NOT_A_NAME or a.lower() == b.lower():
            continue
        relationship = f"{told_by}'s {rel}" if told_by else rel
        known = registry.find_by_name(a) or registry.find_by_name(b)
        if len(known) > 1:
            continue
        source = f"told by {told_by or 'someone'}" + (f" in {where}" if where else "")
        if known:
            person = _alias(registry, _alias(registry, known[0], a, source), b, source)
            if not person.get("relationship"):
                with _lock:
                    person["relationship"] = relationship
                    registry._write(person)
        else:
            person = registry.create(a, [], relationship=relationship, source=source)
            person = _alias(registry, person, b, source)
        registry.add_fact(person["id"], f"{a} is {relationship}, named {b}", told_by or "someone", where)
        logger.info("People: %s introduced %s as %s", told_by or "someone", a, b)
        out.append(person)
    return out


def channel_speaker_line(session_id: str, author_id: str, registry: Optional["PeopleRegistry"] = None) -> str:
    """Who is speaking in a channel, for this message."""
    if not is_channel(session_id):
        return ""
    try:
        registry = registry or PeopleRegistry()
        person = next((p for p in registry.everyone() for a in p.get("accounts", [])
                       if a.get("kind") == "discord" and a.get("id") == str(author_id)), None)
    except OSError:
        return ""
    if not person:
        return "You are in a group Discord channel; you don't know who wrote this message."
    # The DM form ("You are speaking with Azrael"): told "This message is from
    # Testbot" she called a made-up introspect(whois) to find out (2026-11-12).
    line = f"You are speaking with {_display(person)}"
    if person.get("relationship"):
        line += f", {person['relationship']}"
    line += (", in a group Discord channel where others speak too; earlier lines start with who "
             "said them. Speak to them as \"you\".")
    facts = [f["text"] for f in person.get("facts", []) if f.get("status") != "wrong"][:3]
    if facts:
        line += " What you've been told about them: " + "; ".join(facts) + "."
    return line + " " + pronoun_words(person) + "."


def _display(person: Dict) -> str:
    """"Nick (Schwiftyboi on Discord)": the name they were introduced by, with
    the Discord display name when it differs."""
    return f"{_main_name(person)} ({_screen_name(person)} on Discord)" \
        if _screen_name(person) and _screen_name(person) != _main_name(person) else _main_name(person)


def _main_name(person: Dict) -> str:
    sources = person.get("name_sources") or {}
    told = [n for n in person.get("names", []) if sources.get(n, "").startswith("told")
            and n != _screen_name(person)]
    return (told or person.get("names") or ["someone"])[-1] if told else (person.get("names") or ["someone"])[0]


def _screen_name(person: Dict) -> str:
    sources = person.get("name_sources") or {}
    shown = [n for n in person.get("names", []) if "display name" in sources.get(n, "")]
    if shown:
        return shown[-1]
    return person["names"][0] if person.get("source", "").startswith("spoke in") and person.get("names") else ""
