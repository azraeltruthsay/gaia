"""The correction transmission: shift up to Prime when Core keeps being corrected.

Azrael, 2026-11-17: instead of shifting to gear 2 on keywords or intent, watch
the bicameral mind. When two of a conversation's last three answers had to be
corrected — the person disputed the answer, or the second mind posted a
correction — shift up so Prime speaks. Shift back after Prime has gone five
turns without a correction, but not before it has had ten minutes. The intent
trigger stays for the obvious cases (deep focus).

Measured 2026-11-17: gear 1 -> 2 took ~52 s until Prime answered from the GPU;
with the GPU handed over as soon as it's free (consciousness_matrix
_apply_targets) it's ~24 s, and 2 -> 1 puts Core on the GPU in ~15 s. The shift starts right after the reply that drew the second
correction, so it happens while the person reads and writes back.

Only acts when the resting gear is gear 1 (awake), and only shifts down what
it shifted up: a gear someone chose by hand is theirs. Strikes are counted
per conversation; the gear is one GPU, shared.

Engineering (Azrael, 2026-11-19): her engineering adapter lives on Prime's GPU
weights, so an engineering turn in gear 1 shifts up first and waits for Prime
(shift_for_engineering); the shift goes back down once no engineering turn has
come for ENGINEERING_IDLE_S. Parked, she wakes for a message as before, unless
the park is held: a park chosen on purpose (hold) stays parked.

State: /shared/transmission/state.json. Flag: TRANSMISSION_ENABLED (default on).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.Transmission")

STATE_PATH = Path(os.environ.get("TRANSMISSION_STATE", "/shared/transmission/state.json"))
ORCHESTRATOR = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
RESTING_GEAR_FILE = Path(os.environ.get("GAIA_RESTING_GEAR_FILE", "/shared/orchestrator/resting_gear.json"))

STRIKES_TO_SHIFT = 2          # corrected answers ...
WINDOW = 3                    # ... among a conversation's last this many
CLEAN_TURNS_TO_RETURN = 5     # Prime answers in a row without a correction
MIN_DWELL_S = 600             # at least this long in gear 2 once shifted
MAX_TURNS = 50
ENGINEERING_IDLE_S = 900      # back to gear 1 after this long without an engineering turn
ENGINEERING_WAIT_S = 150      # how long an engineering turn waits for Prime on the GPU
HOLD_PATH = Path(os.environ.get("GAIA_PARK_HOLD_FILE", "/shared/orchestrator/park_hold.json"))

_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("TRANSMISSION_ENABLED", "1").lower() in ("1", "true", "yes", "on")


def _load() -> Dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {"turns": [], "shift": None}


def _save(state: Dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_PATH.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE_PATH)


def _resting_gear() -> str:
    try:
        return json.loads(RESTING_GEAR_FILE.read_text()).get("state", "")
    except (OSError, ValueError):
        return os.environ.get("GAIA_RESTING_GEAR", "focusing")


def _current_gear() -> str:
    try:
        import httpx
        return httpx.get(f"{ORCHESTRATOR}/lifecycle/state", timeout=5).json().get("state", "")
    except Exception:
        return ""


def _shift(target: str, reason: str) -> bool:
    try:
        import httpx
        r = httpx.post(f"{ORCHESTRATOR}/consciousness/{target}", json={"reason": reason}, timeout=10)
        ok = r.status_code == 200 and (r.json() or {}).get("ok", True)
        logger.info("Transmission: shift to %s (%s) -> %s", target, reason, "started" if ok else r.text[:200])
        return ok
    except Exception as e:
        logger.warning("Transmission: shift to %s failed: %s", target, e)
        return False


# ── Pure decisions (tested without services) ─────────────────────────────

def should_shift_up(turns: List[Dict], session_id: str, since: float = 0.0) -> bool:
    """Strikes count only since the last shift either way: answers given
    before it belong to the gear they were given in."""
    mine = [t for t in turns if t.get("session") == session_id and t.get("at", 0) >= since][-WINDOW:]
    return sum(1 for t in mine if t.get("strikes", 0) > 0) >= STRIKES_TO_SHIFT


def should_shift_down(turns: List[Dict], shift: Optional[Dict], now: float) -> bool:
    if not shift or shift.get("by") != "transmission":
        return False
    if shift.get("why") == "engineering":
        return now - shift.get("last_engineering", shift.get("at", now)) >= ENGINEERING_IDLE_S
    if now - shift.get("at", now) < MIN_DWELL_S:
        return False
    since = [t for t in turns if t.get("at", 0) >= shift.get("at", 0)]
    tail = since[-CLEAN_TURNS_TO_RETURN:]
    return len(tail) >= CLEAN_TURNS_TO_RETURN and all(t.get("strikes", 0) == 0 for t in tail)


_CONCEDE_RE = None


def _key_terms(text: str) -> List[str]:
    """What an answer committed to: its names and numbers (else its long words)."""
    import re
    body = re.sub(r"^\s*\[[A-Za-z ]+\]\s*", "", text or "")
    body = re.sub(r"\*\[[^\]]*\]\*", " ", body)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+|\d[\d,.]*", body)
    keys = [w for i, w in enumerate(words) if w[0].isdigit() or (w[0].isupper() and i > 0
                                                                   and w not in ("I", "GAIA"))]
    if not keys:
        keys = [w for w in words if len(w) > 5]
    return list(dict.fromkeys(k.lower().strip(",.") for k in keys))[:8]


def conceded(disputed: str, reply: str) -> bool:
    """Did her reply to a dispute give up the answer disputed? Held when most
    of what that answer committed to (names, numbers) is still in the reply:
    "You're right... the capital is indeed Canberra" held; "it's Paris" didn't."""
    keys = _key_terms(disputed)
    if not keys:
        return False
    low = (reply or "").lower()
    kept = sum(1 for k in keys if k in low)
    return kept / len(keys) < 0.5


# ── Hooks ────────────────────────────────────────────────────────────────

def note_pushback(session_id: str, disputed_answer: str) -> None:
    """The person disputed her last answer. Whether that answer was wrong is
    decided by her reply: conceding and changing it is a strike on the
    disputed answer; holding it (the person was mistaken) is not. Disputing a
    right answer mustn't send her to Prime."""
    if not enabled() or not session_id:
        return
    with _lock:
        state = _load()
        state.setdefault("pending", {})[session_id] = (disputed_answer or "")[:1000]
        _save(state)


def note_turn(session_id: str, mind: str = "", reply: str = "") -> Optional[str]:
    """A reply went out in a conversation. Returns 'up' if it conceded a
    dispute that made the second strike, 'down' if it ends a transmission
    shift (Prime's fifth clean answer after the dwell)."""
    if not enabled() or not session_id:
        return None
    disputed = None
    with _lock:
        state = _load()
        disputed = (state.get("pending") or {}).pop(session_id, None)
        _save(state)
    if disputed is not None and conceded(disputed, reply):
        # A strike on the answer disputed: still the newest turn on record.
        result = note_correction(session_id, "pushback", "conceded: " + (reply or "")[:100])
        _record_turn(session_id, mind)
        return result
    return _record_turn(session_id, mind)


def _record_turn(session_id: str, mind: str = "") -> Optional[str]:
    with _lock:
        state = _load()
        now = time.time()
        state["turns"] = (state["turns"] + [{"session": session_id, "at": now, "mind": mind,
                                             "strikes": 0, "why": []}])[-MAX_TURNS:]
        decision = None
        if should_shift_down(state["turns"][:-1], state.get("shift"), now):
            decision = "down"
        shifted = bool(state.get("shift"))
        _save(state)
    # Someone else moved the gear (by hand, or an idle park): the shift is over.
    if shifted and decision is None and _current_gear() not in ("focusing", ""):
        with _lock:
            state = _load()
            state["shift"] = None
            state["since"] = time.time()
            _save(state)
        logger.info("Transmission: gear moved elsewhere; shift record cleared")
        return None
    if decision == "down" and _current_gear() == "focusing":
        if _shift("awake", f"transmission: {CLEAN_TURNS_TO_RETURN} answers without a correction"):
            with _lock:
                state = _load()
                state["shift"] = None
                state["since"] = time.time()
                _save(state)
            return "down"
    return None


def note_correction(session_id: str, kind: str, detail: str = "") -> Optional[str]:
    """The latest answer in this conversation was corrected ('pushback' for the
    person's own dispute — that's about the previous answer; 'second_mind' for
    the other mind's follow-up on this one). Returns 'up' when it shifts."""
    if not enabled() or not session_id:
        return None
    with _lock:
        state = _load()
        mine = [t for t in state["turns"] if t.get("session") == session_id]
        if not mine:
            return None
        # A dispute arrives with the next message, about the answer before it:
        # the newest turn on record is still that answer (note_turn runs at
        # the end of the turn).
        target = mine[-1]
        target["strikes"] = target.get("strikes", 0) + 1
        target["why"] = (target.get("why") or []) + [f"{kind}: {detail[:120]}"]
        up = (should_shift_up(state["turns"], session_id, state.get("since", 0.0)) and not state.get("shift")
              and _resting_gear() == "awake")
        _save(state)
    if up and _current_gear() == "awake":
        reason = f"transmission: {STRIKES_TO_SHIFT} of the last {WINDOW} answers corrected ({kind})"
        if _shift("focusing", reason):
            with _lock:
                state = _load()
                state["shift"] = {"by": "transmission", "at": time.time(), "session": session_id,
                                  "reason": reason}
                state["since"] = state["shift"]["at"]
                _save(state)
            return "up"
    return None


# ── Held park ────────────────────────────────────────────────────────────

def held() -> Optional[Dict]:
    """The hold on a park chosen on purpose, or None."""
    try:
        h = json.loads(HOLD_PATH.read_text())
        return h if h.get("held") else None
    except (OSError, ValueError):
        return None


def set_hold(on: bool, reason: str = "", by: str = "") -> Dict:
    HOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    h = {"held": bool(on), "reason": reason, "by": by, "at": time.time()}
    fd, tmp = tempfile.mkstemp(dir=HOLD_PATH.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(h, f)
    os.replace(tmp, HOLD_PATH)
    logger.info("Transmission: park hold %s (%s)", "on" if on else "off", reason)
    return h


# ── Engineering ──────────────────────────────────────────────────────────

def shift_for_engineering(session_id: str, wait_s: float = ENGINEERING_WAIT_S, sleep=time.sleep) -> bool:
    """An engineering turn: True when Prime is (now) on the GPU to answer with
    the adapter. Shifts up from gear 1 and waits; never moves a held park or a
    gear in transition."""
    if not enabled() or held():
        return False
    gear = _current_gear()
    now = time.time()
    # mid-shift (a wake for this turn already under way): wait for it to land
    settle = time.time() + wait_s
    while gear not in ("awake", "parked", "focusing", "") and time.time() < settle:
        sleep(3)
        gear = _current_gear()
    if gear == "focusing":
        with _lock:
            state = _load()
            if (state.get("shift") or {}).get("why") == "engineering":
                state["shift"]["last_engineering"] = now
                _save(state)
        return True
    # parked (not held): wake straight into gear 2 -- what the adapter needs,
    # and from parked gear 1 may not fit (2026-11-20: "insufficient VRAM for
    # awake", GAIA's own parked buffers counted as an outside tenant)
    if gear not in ("awake", "parked"):
        return False
    with _lock:
        state = _load()
        if state.get("shift") and gear == "awake":
            return False
    if not _shift("focusing", "engineering question: the adapter is on Prime"):
        return False
    with _lock:
        state = _load()
        state["shift"] = {"by": "transmission", "why": "engineering", "at": now, "session": session_id,
                          "last_engineering": now, "reason": "engineering question"}
        state["since"] = now
        _save(state)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        sleep(3)
        if _current_gear() == "focusing":
            logger.info("Transmission: Prime on the GPU for an engineering turn after %.0fs", time.time() - now)
            return True
    logger.warning("Transmission: Prime not on the GPU after %.0fs; answering without the adapter", wait_s)
    return False


def status() -> Dict:
    state = _load()
    return {"enabled": enabled(), "shift": state.get("shift"), "park_hold": held(),
            "recent": [{k: t.get(k) for k in ("session", "mind", "strikes")} for t in state["turns"][-6:]]}
