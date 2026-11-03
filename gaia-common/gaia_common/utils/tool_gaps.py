"""What she keeps reaching for and can't do (yi22).

Her failed attempts say what a tool should be called. If she keeps trying
knowledge(add, fact=...), that is the tool to make work, not a differently
named one she would have to learn. This mines the outcome ledger for repeats
and proposes the cheapest fix: her verb as an alias of an action that exists,
a param mapping, or a new action under her name.

Nothing changes until the architect approves it. Attempts show what she
expected to exist, not whether it's a good idea. Core isn't asked to match
verbs to actions: it got 5/15 on a known-answer test (file ls -> nothing,
worldbuild sing -> everything). Actions taking the same params are listed as
candidates instead.
"""
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger("GAIA.ToolGaps")

STORE_DIR = Path(os.environ.get("GAIA_TOOL_GAPS_DIR", "/shared/tool_gaps"))
MIN_ATTEMPTS = 3
MIN_OCCASIONS = 2  # one looping conversation or one repeated test isn't demand

_UNKNOWN_ACTION_RE = re.compile(r"Unknown action '([^']*)' for domain '([^']*)'")
_NO_TOOL_RE = re.compile(r"Tool '([^']+)' is not a valid|Unknown domain: (\w+)")
_BAD_PARAMS_RE = re.compile(r"required|Internal error|Internal Server Error|expected str|Nothing to add|missing", re.IGNORECASE)
# Not verbs she chose: the selector's own step, and calls with no action at all.
_NOISE_VERBS = {"", "review"}
# The prompts are artificial but the verb she picks is hers, so test traffic
# counts, as one occasion a day: one test repeated 50 times is one signal.
_TEST_SESSION_RE = re.compile(r"^(?:smoke|probe|fact|fl|f18|live|ledger|diagnostics|test|sovereign_review|discord_dm_900\d)[-_\d]")


def _rows(paths: Iterable[str]) -> Iterable[dict]:
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
        except OSError:
            continue


def _occasion(row: dict, is_test: bool) -> tuple:
    """When and where an attempt happened, for telling repeats from demand. A
    day apart counts: the heartbeat is one session running for weeks."""
    return ("test" if is_test else str(row.get("session_id") or ""), int((row.get("ts") or 0) // 86400))


def _classify(row: dict) -> Optional[tuple]:
    """(kind, tool, verb) for a record that shows a gap, else None. Failures
    like a missing file or a review rejection aren't gaps: the tool exists and
    took her call."""
    if row.get("kind") not in ("tool_execution", "tool_selection") or row.get("ok") is True:
        return None
    error = str(row.get("error") or "")
    m = _UNKNOWN_ACTION_RE.search(error)
    if m:
        return None if m.group(1) in _NOISE_VERBS else ("unknown_action", m.group(2), m.group(1))
    m = _NO_TOOL_RE.search(error)
    if m:
        return ("new_tool", m.group(1) or m.group(2), str(row.get("action") or ""))
    if row.get("kind") == "tool_execution" and row.get("action") and _BAD_PARAMS_RE.search(error):
        return ("params", row.get("tool"), row["action"])
    return None


def mine(rows: Iterable[dict], is_known: Callable[[str, str], bool]) -> List[dict]:
    """Repeated gaps, most attempted first. Gone if the call works now: a verb
    that has become known, or params that have since succeeded."""
    groups: Dict[tuple, dict] = {}
    worked = set()
    for row in rows:
        is_test = bool(_TEST_SESSION_RE.match(str(row.get("session_id") or "")))
        if row.get("ok") is True and row.get("kind") == "tool_execution":
            worked.add((row.get("tool"), row.get("action")))
            continue
        found = _classify(row)
        if not found:
            continue
        g = groups.setdefault(found, {"packets": set(), "occasions": set(), "real": set(), "params": set(),
                                      "first": row.get("ts"), "last": row.get("ts")})
        packet = row.get("packet_id") or id(row)  # selection and execution are one attempt
        g["packets"].add(packet)
        g["occasions"].add(_occasion(row, is_test))
        if not is_test:
            g["real"].add(packet)
        g["params"].update(set(row.get("param_keys") or []) - {"action"})
        g["last"] = row.get("ts") or g["last"]
    out = []
    for (kind, tool, verb), g in groups.items():
        if len(g["packets"]) < MIN_ATTEMPTS or len(g["occasions"]) < MIN_OCCASIONS:
            continue
        if kind == "unknown_action" and is_known(tool, verb):
            continue
        if kind == "params" and (tool, verb) in worked:
            continue
        out.append({"id": proposal_id(kind, tool, verb), "kind": kind, "tool": tool, "verb": verb,
                    "attempts": len(g["packets"]), "occasions": len(g["occasions"]),
                    "outside_tests": len(g["real"]),
                    "params": sorted(g["params"]), "first": g["first"], "last": g["last"]})
    return sorted(out, key=lambda p: -p["attempts"])


def proposal_id(kind: str, tool: str, verb: str) -> str:
    return hashlib.sha1(f"{kind}|{tool}|{verb}".encode()).hexdigest()[:8]


def candidates(tool: str, params: Iterable[str], domain_tools: dict) -> List[str]:
    """Existing actions that take any of the params she sent: this tool's, or
    failing that another's as "tool.action", since she can pick the right action
    under the wrong tool (knowledge(study, curriculum=...))."""
    sent = set(params)

    def matching(name: str) -> List[str]:
        actions = (domain_tools.get(name) or {}).get("actions") or {}
        return [a for a, spec in actions.items() if sent & set(spec.get("params") or {})]

    return matching(tool) or [f"{other}.{a}" for other in domain_tools if other != tool for a in matching(other)]


def _store_path() -> Path:
    return STORE_DIR / "proposals.json"


def load() -> dict:
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(store: dict) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _store_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(_store_path())


def refresh(mined: List[dict], domain_tools: dict) -> int:
    """Merge a mining pass into the store. A decision stands; new evidence only
    updates the counts. Returns how many new proposals await the architect."""
    store = load()
    new = 0
    for p in mined:
        have = store.get(p["id"])
        if have:
            have.update({k: p[k] for k in ("attempts", "occasions", "outside_tests", "params", "last")})
            continue
        store[p["id"]] = {**p, "status": "pending", "candidates": candidates(p["tool"], p["params"], domain_tools)}
        new += 1
    if mined or new:
        _save(store)
    return new


def decide(pid: str, status: str, target: str = "") -> dict:
    """Record the architect's decision on a proposal."""
    store = load()
    rec = store.get(pid)
    if not rec:
        raise KeyError(pid)
    rec.update({"status": status, "decided": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if target:
        rec["target"] = target
    _save(store)
    return rec


def approved_aliases() -> List[tuple]:
    """(tool, her verb, target tool, target action) for each approved alias."""
    out = []
    for r in load().values():
        if r.get("status") == "approved" and r.get("kind") == "unknown_action" and r.get("target"):
            tool, _dot, action = r["target"].rpartition(".")
            out.append((r["tool"], r["verb"], tool or r["tool"], action))
    return out


def capability_gap_text(rec: dict) -> str:
    """A proposal as CodeMind's capability-gap seed: her name, and the evidence."""
    payload = {"attempted_tool_name": rec["tool"], "attempted_tool_action": rec.get("verb", ""),
               "attempted_tool_params": {k: "" for k in rec.get("params", [])},
               "user_request": f"Attempted {rec['attempts']} times on {rec['occasions']} occasions; approved by the architect."}
    return f"Capability gap: {rec['tool']}({rec.get('verb', '')}). Payload: {json.dumps(payload)}"
