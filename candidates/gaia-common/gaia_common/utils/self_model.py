"""What each of her minds is running, observed from the machine (kki4).

The names were hardcoded in the prompt, so they said what someone last typed. Now
each engine is read when asked, and what it says goes into the world model, as an
observed fact in actuality with the time it became true. If an engine can't be read
(mid-shift, parked) the last observation stands, and she's told how old it is. Swap
a model and this follows with no prompt edit. /models/LINEAGE.json says where a set
of weights came from, kept next to them by whatever builds them.
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, Optional

from gaia_common.utils.host_facts import host_facts

logger = logging.getLogger("GAIA.SelfModel")

LINEAGE_PATH = os.environ.get("GAIA_LINEAGE_PATH", "/models/LINEAGE.json")
_observed: Dict[str, dict] = {}
_loaded = False
_kg = None


def _graph():
    global _kg
    if _kg is None:
        from gaia_common.utils.knowledge_graph import KnowledgeGraph
        _kg = KnowledgeGraph()
    return _kg


def _subject(mind: str) -> str:
    return f"gaia_{mind.lower()}"


def _lineage() -> dict:
    try:
        with open(LINEAGE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def lineage_of(weights: str) -> dict:
    """The entry whose key starts this file name, longest first: an NF4 copy or a
    GGUF shares its source's entry unless it has its own."""
    entries = _lineage()
    for key in sorted(entries, key=len, reverse=True):
        if (weights or "").startswith(key):
            return entries[key]
    return {}


def display_name(weights: str) -> str:
    """What the weights are called: the lineage's name, else the file name up to
    our own suffix, so a model nobody has described yet still gets its true name."""
    return lineage_of(weights).get("name") or re.split(r"-GAIA\b|\.gguf$", weights or "")[0]


def _record(mind: str, new: dict, old: Optional[dict]) -> None:
    """A change becomes a fact; the fact it replaces is closed, not deleted."""
    try:
        kg, subject = _graph(), _subject(mind)
        when = new["since"]
        for predicate, key in (("runs_weights", "weights"), ("runs_on_device", "device")):
            if old and old.get(key) == new[key]:
                continue
            if old and old.get(key):
                kg.invalidate(subject, predicate, old[key], ended=when)
            kg.add_triple(subject, predicate, new[key], valid_from=when, source="observed:engine_status",
                          fact_type="state", allow_coexist=True)
    except Exception as e:
        logger.debug("Self model: could not record %s: %s", mind, e)


def _recall() -> None:
    """Pick up what was last observed, once per process."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        for mind in ("Core", "Prime"):
            entry = {}
            for fact in _graph().query_entity(_subject(mind), direction="outgoing"):
                if not fact.get("current"):
                    continue
                if fact.get("predicate") == "runs_weights":
                    entry.update(weights=fact["object"], since=fact.get("valid_from"))
                elif fact.get("predicate") == "runs_on_device":
                    entry["device"] = fact["object"]
            if entry.get("weights"):
                _observed.setdefault(mind, {**entry, "live": False})
    except Exception as e:
        logger.debug("Self model: could not recall: %s", e)


def minds(facts: Optional[dict] = None) -> Dict[str, dict]:
    """{mind: {weights, name, device, live, since}}, from the engines now, else from
    the last time they could be read."""
    _recall()
    engines = (host_facts() if facts is None else facts).get("engines") or {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for mind, engine in engines.items():
        device = "GPU" if str(engine.get("device")).startswith(("cuda", "gpu")) else "CPU"
        old = _observed.get(mind)
        changed = not old or old.get("weights") != engine["model"] or old.get("device") != device
        new = {"weights": engine["model"], "device": device, "since": now if changed else old.get("since", now)}
        if changed:
            _record(mind, new, old)
        _observed[mind] = new
    out = {}
    for mind, seen in _observed.items():
        out[mind] = {**seen, "name": display_name(seen["weights"]), "live": mind in engines}
    return out


def self_model_text(facts: Optional[dict] = None) -> str:
    """Her minds in a plain sentence or two, for the world state. "" if never read."""
    lines = []
    for mind, m in sorted(minds(facts).items(), key=lambda kv: kv[1].get("device") != "GPU"):
        role = "generating" if m.get("device") == "GPU" else "watching"
        state = f"on the {m.get('device')}, {role}" if m["live"] else f"not reachable just now, last on the {m.get('device')}"
        lines.append(f"{mind} is {m['name']} ({state})")
    return "; ".join(lines) + "." if lines else ""
