"""She improves herself: gap -> study -> training -> evaluation -> approval (dcrs/p3ta, 2026-11-20).

Azrael: "she can identify, prepare test questions, run a training session,
evaluate and decide, switch. All autonomously ... until the final application
of the adapter using my code." A sleep task ticks this state machine while she
sleeps (so only when nobody is talking to her):

  idle       enough new gaps (learning_gaps: corrections, unkept lookups,
             minds disagreeing) and no training for a while -> she names the
             study topics behind them in her inner voice; code finds a trusted
             page for each (fetched to be sure it reads) and adds them to the
             syllabus, then her study run starts
  studying   her study run (study_syllabus) reads the pages and writes checked
             notes and pairs; with enough new pairs -> a training run
  training   gaia-study runs scripts/self_improve_gpu.py: blind review, build,
             augment, train the next version, web eval against the active one,
             decide. A pass waits for Azrael; anything else is logged
  awaiting   the activation is a sensitive action (adapter_activate): she asks
             for his authenticator code in his DMs with the verdict; his code
             switches it (live on the next turn); unanswered, it lapses after a day
             and she asks again next cycle

State: /shared/self_improve/state.json. Flag: SELF_IMPROVE_ENABLED (default on).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.SelfImprove")

SUBJECT = "professional_engineering"
CURRICULUM = "knowledge/curricula/professional_engineering"
STUDY_DIR = f"/knowledge/studies/{SUBJECT}"
REPO = Path(os.environ.get("GAIA_REPO", "/gaia/GAIA_Project"))
STUDY_URL = os.environ.get("STUDY_ENDPOINT", "http://gaia-study:8766")
MIN_GAPS = 3                       # new gaps before she plans a study round
MIN_NEW_PAIRS = 20                 # new pairs before a training run
COOLDOWN_S = 3 * 86400             # between training runs
APPROVAL_TTL_S = 86400             # how long the approval request waits
MAX_TOPICS = 3

_study_thread: Optional[threading.Thread] = None


def enabled() -> bool:
    return os.environ.get("SELF_IMPROVE_ENABLED", "1").lower() in ("1", "true", "yes", "on")


def _state_path() -> Path:
    return Path(os.environ.get("SHARED_DIR", "/shared")) / "self_improve" / "state.json"


def load() -> Dict:
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {"stage": "idle", "gaps_since": 0, "last_train": 0}


def save(state: Dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, p)


def _http(method: str, url: str, body: Optional[dict] = None, timeout: int = 30) -> Dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── planning: gaps -> topics -> pages ────────────────────────────────────

PLAN_THOUGHT = ("Questions I couldn't answer well lately (I was corrected, couldn't find a source, or my "
                "two minds disagreed):\n{gaps}\n\nWhich engineering subjects would I need to study to answer "
                "questions like these? Name up to {n}, each a short encyclopedia-style article title. "
                "Questions that aren't about engineering, I'll leave.")


def plan_topics(gaps: List[Dict], generate: Callable[[List[dict]], str]) -> List[str]:
    from gaia_core.cognition.inner_monologue import ask_myself
    listed = "\n".join(f"- {g['question'][:200]} ({g['kind']})" for g in gaps[:12])
    said = ask_myself(PLAN_THOUGHT.format(gaps=listed, n=MAX_TOPICS), generate, about="what to study next",
                      label="TOPICS", form="the article titles, separated by semicolons",
                      my_form="the article titles, separated by semicolons")["say"] or ""
    topics = [t.strip(" -*.\"'") for t in re.split(r"[;\n]", said)]
    return [t for t in topics if 3 <= len(t) <= 80][:MAX_TOPICS]


def find_page(topic: str, call=None) -> Optional[str]:
    """A trusted page on the topic that actually reads (Wikipedia first)."""
    if call is None:
        from gaia_core.utils import mcp_client
        call = mcp_client.call_jsonrpc

    def result(r):
        r = r.get("response", r) if isinstance(r, dict) else {}
        return r.get("result", r) if isinstance(r, dict) else {}
    hits = result(call("web_search", {"query": f"{topic} wikipedia", "max_results": 5}, timeout=60)).get("results") or []
    urls = [h.get("url") for h in hits if h.get("url")]
    urls.sort(key=lambda u: "wikipedia.org/wiki/" not in u)
    for url in urls[:3]:
        page = result(call("web_fetch", {"url": url}, timeout=60))
        if page.get("ok", True) and len(page.get("content") or "") > 1500:
            return url
    return None


def add_to_syllabus(entries: List[Dict], syllabus: Path) -> int:
    """Append her topics as a self-directed unit (her own suggestion, with the gap behind it)."""
    if not entries:
        return 0
    stamp = time.strftime("%Y-%m-%d")
    lines = [f"\n# Self-directed ({stamp}): topics GAIA chose from questions she couldn't answer.",
             f"  Self-directed study ({stamp}):"]
    for e in entries:
        why = e["why"].replace('"', "'")[:120]
        lines.append(f'    - {{topic: "{e["topic"]}", url: "{e["url"]}", suggested_by: GAIA, '
                     f'why: "{why}"}}')
    with syllabus.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(entries)


def _pairs_count() -> int:
    try:
        return sum(1 for _ in open(f"{STUDY_DIR}/pairs.jsonl", encoding="utf-8"))
    except OSError:
        return 0


def _start_study(state: Dict) -> None:
    global _study_thread
    syllabus_path = REPO / CURRICULUM / "syllabus.yaml"

    def run():
        try:
            import yaml
            from gaia_core.cognition.study_syllabus import http_generator, study
            syl = yaml.safe_load(syllabus_path.read_text())
            study(syl, Path(STUDY_DIR), http_generator(), log=lambda m: logger.info("self_improve study: %s", m),
                  max_pages=20)
        except Exception:
            logger.warning("self_improve: study run failed", exc_info=True)
    _study_thread = threading.Thread(target=run, name="self-improve-study", daemon=True)
    _study_thread.start()


# ── approval ─────────────────────────────────────────────────────────────

def approver_session() -> str:
    """Azrael's DM session (his people record: the one who created her)."""
    override = os.environ.get("SELF_IMPROVE_APPROVER_SESSION", "")
    if override:
        return override
    try:
        from gaia_common.utils.people import PeopleRegistry
        for p in PeopleRegistry().everyone():
            if "created you" in (p.get("relationship") or ""):
                for a in p.get("accounts", []):
                    if a.get("kind") == "discord":
                        return f"discord_dm_{a['id']}"
    except Exception:
        logger.debug("approver lookup failed", exc_info=True)
    return ""


def request_approval(slot: str, name: str, verdict: List[str], session: str) -> Optional[str]:
    from gaia_core.cognition import approval_reply, long_read
    from gaia_core.utils.mcp_client import request_approval_via_mcp
    r = request_approval_via_mcp("adapter_activate", {"slot": slot, "name": name, "_allow_pending": True,
                                                      "_ttl_seconds": APPROVAL_TTL_S})
    action_id = r.get("action_id")
    if not action_id:
        logger.warning("self_improve: approval request failed: %s", r.get("error"))
        return None
    label = f"adapter_activate {slot} -> {name}"
    approval_reply.park_for(session, action_id, label, ttl=APPROVAL_TTL_S)
    lines = "\n".join(f"• {v}" for v in verdict)
    text = (f"I trained a new {slot} adapter, **{name}**, from what I studied, and it passed every check "
            f"against the one I use now:\n{lines}\n\n"
            f"That needs your approval first: `{label}`. Reply with your authenticator code to approve, "
            f"or say cancel.")
    long_read.push(session, text)
    return action_id


# ── the state machine ────────────────────────────────────────────────────

def tick(generate: Optional[Callable[[List[dict]], str]] = None, now: Optional[float] = None) -> Dict:
    """One step; returns the state. Safe to call often."""
    if not enabled():
        return {"stage": "disabled"}
    from gaia_core.cognition import learning_gaps
    from gaia_common.utils import adapter_registry as ar
    now = now or time.time()
    state = load()
    stage = state.get("stage", "idle")
    current = ar.active("engineering", os.environ.get("ENGINEERING_ADAPTER", ""))

    if stage == "idle":
        gaps = learning_gaps.since(state.get("gaps_since", 0))
        if len(gaps) < MIN_GAPS or now - state.get("last_train", 0) < COOLDOWN_S or generate is None:
            return state
        topics = plan_topics(gaps, generate)
        entries = []
        for t in topics:
            url = find_page(t)
            if url:
                entries.append({"topic": t, "url": url, "why": gaps[0]["question"]})
        added = add_to_syllabus(entries, REPO / CURRICULUM / "syllabus.yaml")
        state.update({"gaps_since": now, "planned": [e["topic"] for e in entries], "planned_at": now,
                      "pairs_before": _pairs_count()})
        if added:
            state["stage"] = "studying"
            _start_study(state)
        logger.info("self_improve: %d gaps -> topics %s -> %d added", len(gaps), topics, added)
        save(state)
        return state

    if stage == "studying":
        if _study_thread is not None and _study_thread.is_alive():
            return state
        if _pairs_count() - state.get("pairs_before", 0) < MIN_NEW_PAIRS:
            state["stage"] = "idle"            # not enough to train on; it waits for more
            save(state)
            return state
        r = _http("POST", f"{STUDY_URL}/pipeline/self-improve",
                  {"curriculum": CURRICULUM, "study": STUDY_DIR, "slot": "engineering", "current": current})
        if r.get("ok"):
            state.update({"stage": "training", "training_started": now})
        logger.info("self_improve: training run requested: %s", r)
        save(state)
        return state

    if stage == "training":
        st = _http("GET", f"{STUDY_URL}/pipeline/self-improve")
        last = st.get("last_run") or {}
        if st.get("running") or last.get("started", 0) < state.get("training_started", 0) - 60:
            return state
        state.update({"stage": "idle", "last_train": now, "last_outcome": last.get("outcome"),
                      "candidate": last.get("candidate")})
        decision = last.get("decision") or {}
        if decision.get("action") == "proposed":
            state["stage"] = "awaiting"
            state["verdict"] = decision.get("verdict", [])
        save(state)
        return tick(generate, now) if state["stage"] == "awaiting" else state

    if stage == "awaiting":
        cand = state.get("candidate", "")
        if current == cand or not ar.pending("engineering"):
            state["stage"] = "idle"
            save(state)
            return state
        if now - state.get("asked_at", 0) >= APPROVAL_TTL_S:
            session = approver_session()
            if session and request_approval("engineering", cand, state.get("verdict", []), session):
                state["asked_at"] = now
                save(state)
        return state
    return state
