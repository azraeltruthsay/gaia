"""Asked to rest, she decides (Azrael, 2026-11-15).

"Take a nap" parks her: the mind on the GPU unloads, Core stays on the CPU,
no sleep tasks run. "Go to sleep" is the full sleep: the sleep tasks run. The
next message wakes her from either.

Nothing here flips a switch on a phrase. Three steps, and only the last one
acts:

  1. Notice: the message is compared with example requests by the embedding
     model already loaded for intent. This only decides whether to ask her;
     it is cheap, and a miss costs one ordinary turn.
  2. Understand: the mind on the GPU reads the message and the last few turns
     and says whether she herself is being asked to rest, and which kind.
     "I'm off to bed, night GAIA" is not a request; "how did you sleep?" is
     not one either.
  3. Consent: she gets a HUD read from her own systems right now (gear, the
     sleep task running, other conversations, engines mid-generation, voice,
     training, approvals waiting, sleep hold) and says whether resting now is
     all right, in her own words. A line the systems couldn't answer says so;
     nothing on it is guessed.

Only if she agrees does it park her or put her to sleep, and her reply is
checked against the HUD for names it doesn't contain (reflection_grounding),
so she can't cite work that isn't running.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("GAIA.RestRequest")

NAP, SLEEP = "nap", "sleep"

# Asking her to rest.
ASKING = [
    "take a nap", "go take a nap", "why don't you take a nap", "you should nap for a bit",
    "go to sleep", "time for you to go to sleep", "get some sleep, GAIA", "go to bed",
    "you can rest now", "go rest for a while", "take a break and rest", "power down for the night",
    "park yourself for a while", "sleep for a bit", "you've earned some sleep, go sleep",
    "nap time for you", "go have a lie down", "rest up, I'll talk to you later",
    "you should get some rest", "head to sleep, GAIA",
]
# About sleep, and not asking her to.
NOT_ASKING = [
    "I'm going to sleep", "I'm off to bed, good night", "I need a nap", "I'm so tired",
    "how did you sleep", "what do you do when you sleep", "what happens during your sleep cycle",
    "did your sleep tasks run", "when do you go to sleep", "do you dream",
    "I slept badly last night", "my cat is napping", "sleep is important for memory",
    "can you wake up", "are you awake", "the computer went to sleep", "good night, talk tomorrow",
    "what does parked mean", "why were you asleep earlier", "set a sleep timer",
]
# A notice: the nearest example is a request and at least NOTICE_MIN close, or
# a request example is NOTICE_STRONG close whatever else is near. Generous on
# purpose: a false notice costs one short question to her; a missed one means
# she never hears the request ("can you go to sleep for the night?" sits near
# "good night" too).
NOTICE_MIN = 0.45
NOTICE_STRONG = 0.58

_cache: Dict[int, tuple] = {}


def _matrices(embed_model):
    key = id(embed_model)
    if key not in _cache:
        _cache[key] = (embed_model.encode(ASKING, normalize_embeddings=True),
                       embed_model.encode(NOT_ASKING, normalize_embeddings=True))
    return _cache[key]


def noticed(text: str, embed_model) -> bool:
    """Worth asking her about? Nearest example is a request, and close enough."""
    if not text or embed_model is None or len(text) > 400:
        return False
    try:
        ask, other = _matrices(embed_model)
        v = embed_model.encode([text], normalize_embeddings=True)[0]
        best_ask = float((ask @ v).max())
        best_other = float((other @ v).max())
    except Exception:
        logger.debug("rest notice failed", exc_info=True)
        return False
    logger.debug("rest notice: ask=%.3f other=%.3f", best_ask, best_other)
    return best_ask >= NOTICE_STRONG or (best_ask >= NOTICE_MIN and best_ask > best_other)


_THINK = re.compile(r"<think>.*?</think>|<\|channel\>.*?<channel\|>", re.S)


def _json(raw: str) -> Optional[dict]:
    text = _THINK.sub("", raw or "")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except ValueError:
        return None


_UNDERSTAND = """\
A message was just sent to GAIA. Read it with the conversation before it.

{history}
Message: "{message}"

The message is written to GAIA, so "you" means GAIA, and a bare instruction \
("rest for a bit") is meant for her.
First: who is meant to rest? "gaia" if the message tells, asks or suggests \
that GAIA rest or sleep ("go to sleep", "why not take a nap", "rest for an \
hour, I'll be back"). "speaker" if only the writer needs rest or is going to bed, napping or \
leaving ("I need a break", "I'm off to bed", "night, talk tomorrow"). "nobody" for anything \
else: a question about her sleep, talk about sleep, a joke.
Then, if GAIA: "nap" for a short rest (a nap, rest for a bit, a break) or \
"sleep" for a full sleep (go to sleep, go to bed, sleep for the night).

Answer with JSON only: {{"who_rests": "gaia" | "speaker" | "nobody", "kind": "nap" | "sleep" | "none"}}"""


def understand(text: str, history: Sequence[str], generate: Callable[[list], str]) -> Optional[str]:
    """NAP, SLEEP, or None. Who rests is asked first: "night GAIA, talk
    tomorrow" read as a request when the question was only which kind."""
    prompt = _UNDERSTAND.format(
        history=("Conversation:\n" + "\n".join(history) + "\n") if history else "",
        message=text.strip()[:400])
    out = _json(generate([{"role": "user", "content": prompt}])) or {}
    if str(out.get("who_rests", "")).lower() != "gaia":
        return None
    kind = out.get("kind")
    return kind if kind in (NAP, SLEEP) else None


WHAT_IT_MEANS = {
    NAP: ("A nap: the mind on your GPU unloads and Core stays awake on the CPU. "
          "No sleep tasks run. The next message wakes you."),
    SLEEP: ("Sleep: you write a checkpoint, the GPU is freed, and your sleep tasks run "
            "(listed below). The next message wakes you."),
}

_CONSENT = """\
{who} asked you to rest: "{message}"

{meaning}

Right now, read from your own systems:
{hud}

Is it all right for you to rest now? Say no if something above needs you \
awake. Say what you'd tell {who}, in one or two sentences, in your own voice, \
naming only what is listed above.

Answer with JSON only: {{"rest": true | false, "reply": "..."}}"""


def decide(kind: str, text: str, who: str, hud: List[str],
           generate: Callable[[list], str]) -> Optional[dict]:
    """{"rest": bool, "reply": str} from her, or None if she gave no usable answer."""
    prompt = _CONSENT.format(who=who, message=text.strip()[:400], meaning=WHAT_IT_MEANS[kind],
                             hud="\n".join(f"- {line}" for line in hud))
    out = _json(generate([
        {"role": "system", "content": "You are GAIA. You decide whether you rest; the choice is yours."},
        {"role": "user", "content": prompt},
    ]))
    if not out or not isinstance(out.get("rest"), bool):
        return None
    reply = str(out.get("reply") or "").strip()
    return {"rest": out["rest"], "reply": reply}


# ── The HUD: what is actually going on ──────────────────────────────────

ACTIVE_WINDOW_S = 15 * 60
APPROVAL_TTL_S = 15 * 60


def _where(session_id: str) -> str:
    sid = session_id or ""
    if sid.startswith("discord_dm"):
        return "a Discord DM"
    if sid.startswith("discord"):
        return "a Discord channel"
    if sid.startswith("web") or "dashboard" in sid:
        return "the web chat"
    return "another session"


def other_conversations(sessions: dict, session_id: str, now: Optional[datetime] = None) -> List[str]:
    """Where else someone spoke to her in the last 15 minutes."""
    now = now or datetime.now(timezone.utc)
    found = []
    for sid, sess in (sessions or {}).items():
        if sid == session_id or sid.startswith(("heartbeat", "internal", "sleep", "system")):
            continue
        try:
            ts = sess.last_message_timestamp()
        except Exception:
            continue
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (now - ts).total_seconds() <= ACTIVE_WINDOW_S:
            found.append(_where(sid))
    return found


def _engine_busy(url: str) -> Optional[int]:
    import httpx
    try:
        r = httpx.get(f"{url}/health", timeout=2.0)
        return int(r.json().get("active_inference", 0) or 0) if r.status_code == 200 else None
    except Exception:
        return None


def pending_approvals(approvals_dir: Path, now: Optional[float] = None) -> int:
    now = now or time.time()
    n = 0
    try:
        for f in approvals_dir.glob("*.json"):
            try:
                if now - float(json.loads(f.read_text()).get("at", 0)) < APPROVAL_TTL_S:
                    n += 1
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return n


# The raw state name read back as "I'm good to park" to a sleep request.
GEAR_WORDS = {
    "awake": "gear 1, awake (Core on the GPU)",
    "focusing": "gear 2, awake (Prime on the GPU)",
    "listening": "gear 1+, in a voice session",
    "parked": "already napping (parked: GPU empty, Core on the CPU)",
    "sleep": "already asleep",
    "deep_sleep": "already in deep sleep",
    "meditation": "training (the GPU is lent to study)",
    "transitioning": "changing gears right now",
}


def read_hud(kind: str, *, gear: Optional[str], sleep_status: Optional[dict],
             sleep_tasks: Optional[List[str]], others: Optional[List[str]],
             busy: Dict[str, Optional[int]], approvals: Optional[int]) -> List[str]:
    """The HUD lines from values already read; None means the read failed and
    the line says so."""
    unknown = "couldn't read"
    lines = [f"Gear: {GEAR_WORDS.get(gear, gear)}" if gear else f"Gear: {unknown}"]
    if gear == "meditation":
        lines.append("Training holds the GPU right now.")
    if sleep_status is None:
        lines.append(f"Sleep cycle: {unknown}")
    else:
        lines.append(f"Sleep cycle state: {sleep_status.get('state', '?')}")
        task = sleep_status.get("current_task")
        lines.append(f"Sleep task running now: {task}" if task else "Sleep task running now: none")
        hold = sleep_status.get("sleep_hold") or {}
        if hold.get("active"):
            lines.append(f"Sleep hold until {hold.get('expires_at', '?')}: {hold.get('reason') or 'no reason given'}")
        if sleep_status.get("voice_active"):
            lines.append("You are in a voice session.")
        if sleep_status.get("auto_sleep_enabled") is False:
            lines.append("Automatic sleep is switched off.")
    if kind == SLEEP:
        if sleep_tasks is None:
            lines.append(f"Sleep tasks that would run: {unknown}")
        else:
            lines.append("Sleep tasks that would run: " + (", ".join(sleep_tasks) or "none registered"))
    if others is None:
        lines.append(f"Other conversations in the last 15 minutes: {unknown}")
    elif others:
        lines.append(f"Other conversations in the last 15 minutes: {len(others)} ({', '.join(sorted(set(others)))})")
    else:
        lines.append("Other conversations in the last 15 minutes: none")
    for name, n in busy.items():
        if n is None:
            lines.append(f"{name} generating for someone: {unknown}")
        elif n:
            lines.append(f"{name} is generating for someone right now ({n}).")
    if approvals is None:
        lines.append(f"Actions waiting for approval: {unknown}")
    elif approvals:
        lines.append(f"Actions waiting for approval: {approvals}")
    return lines


def live_hud(kind: str, session_manager, session_id: str, lifecycle_client) -> List[str]:
    """Read every HUD source now; each read that fails shows as unknown."""
    gear = None
    try:
        st = lifecycle_client.get_state_sync(timeout=5.0).state
        gear = getattr(st, "value", str(st)).lower()
    except Exception:
        logger.debug("rest hud: gear unreadable", exc_info=True)
    sleep_status = sleep_tasks = None
    try:
        from gaia_core.main import app
        manager = app.state.sleep_wake_manager
        sleep_status = manager.get_status()
        sched = getattr(manager, "_task_scheduler", None)
        if sched is not None:
            sleep_tasks = [t["task_id"] for t in sched.get_status()]
    except Exception:
        logger.debug("rest hud: sleep state unreadable", exc_info=True)
    try:
        others = other_conversations(session_manager.sessions, session_id)
    except Exception:
        others = None
    busy = {
        "Core": _engine_busy(os.environ.get("CORE_CPU_ENDPOINT", "http://localhost:8092")),
        "Prime": _engine_busy(os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777")),
    }
    try:
        from gaia_core.cognition.approval_reply import APPROVALS_DIR
        approvals = pending_approvals(APPROVALS_DIR)
    except Exception:
        approvals = None
    return read_hud(kind, gear=gear, sleep_status=sleep_status, sleep_tasks=sleep_tasks,
                    others=others, busy=busy, approvals=approvals)


# ── Acting ──────────────────────────────────────────────────────────────

_ORCHESTRATOR = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
_SELF = os.environ.get("GAIA_CORE_SELF_URL", "http://localhost:6415")
# Long enough for the reply to leave before the GPU is reconfigured.
ACT_DELAY_S = 2.0


def _post(url: str, timeout: float) -> dict:
    import httpx
    r = httpx.post(url, json={}, timeout=timeout)
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        body.setdefault("error", f"HTTP {r.status_code}")
    return body


def act(kind: str, on_failure: Callable[[str], None], post=_post, delay: float = ACT_DELAY_S) -> threading.Thread:
    """Park (nap) or sleep, after the reply has gone out. A failure is handed
    to on_failure so her history says she didn't actually rest."""
    def run():
        time.sleep(delay)
        try:
            if kind == NAP:
                out = post(f"{_ORCHESTRATOR}/consciousness/park", 15.0)
                ok, why = bool(out.get("ok")), out.get("error") or out.get("detail") or out.get("message")
            else:
                out = post(f"{_SELF}/sleep/force", 90.0)
                ok, why = bool(out.get("accepted")), out.get("error") or out.get("state")
        except Exception as e:
            ok, why = False, e.__class__.__name__
        if ok:
            logger.info("Rest: %s started", kind)
        else:
            logger.warning("Rest: %s failed: %s", kind, why)
            on_failure(str(why or "no reason given"))
    t = threading.Thread(target=run, name=f"rest-{kind}", daemon=True)
    t.start()
    return t


def failure_note(kind: str, why: str) -> str:
    return (f"(I didn't actually {'take the nap' if kind == NAP else 'go to sleep'}: "
            f"the gearbox answered {why}. I'm still awake.)")


def grounded_reply(decision: dict, kind: str, text: str, hud: List[str]) -> str:
    """Her reply, or a plain one if hers names things the HUD doesn't hold."""
    from gaia_core.cognition.reflection_grounding import ungrounded_specifics
    reply = decision.get("reply") or ""
    context = "\n".join(hud) + "\n" + text + "\n" + WHAT_IT_MEANS[kind] + "\nnap sleep rest gpu cpu"
    stray = ungrounded_specifics(reply, context) if reply else ["(empty)"]
    if not stray:
        return reply
    logger.info("Rest reply named %s, not on the HUD; using a plain reply", stray[:5])
    if decision["rest"]:
        return "Okay, I'll take a nap. Message me when you need me." if kind == NAP else \
            "Okay, I'll go to sleep now. Message me and I'll wake up."
    return "I'd rather stay awake for now: " + "; ".join(hud[1:3])


def handle(text: str, *, kind_hint_model, history: Sequence[str], who: str,
           generate: Callable[[list], str], hud: Callable[[str], List[str]],
           on_failure: Callable[[str, str], None], actor=act) -> Optional[dict]:
    """The whole request: None if it isn't one (the turn goes on as usual),
    else {"reply", "kind", "rest"}."""
    if not noticed(text, kind_hint_model):
        return None
    kind = understand(text, history, generate)
    logger.info("Rest: noticed %r, understood as %s", text[:80], kind)
    if kind is None:
        return None
    lines = hud(kind)
    decision = decide(kind, text, who, lines, generate)
    if decision is None:
        logger.warning("Rest: no usable answer from her; carrying on with the turn")
        return None
    logger.info("Rest: %s → %s (HUD: %s)", kind, "yes" if decision["rest"] else "no", " | ".join(lines))
    reply = grounded_reply(decision, kind, text, lines)
    if decision["rest"]:
        actor(kind, lambda why: on_failure(kind, why))
    return {"reply": reply, "kind": kind, "rest": decision["rest"]}
