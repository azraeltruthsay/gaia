"""Outcome ledger (gjuh.2): an autonomic record of what GAIA's tools and
memory lookups actually did.

Written by code at the decision points, never by the model. The failure log
skill_failures.py keeps (5qy) waits for the model to call learn(success=false)
on itself, and it has never been written. These records are the evidence the
sleep-cycle pattern miner (gjuh.3) groups into patterns, and they give tool
selection a real accuracy number to measure changes against.

Record kinds (one JSON object per line, all with ts/kind/session_id/packet_id):
  tool_selection  path=deterministic|llm, parsed, tool, param_keys, confidence,
                  reasoning; raw (first 400 chars) when the output didn't parse
  tool_execution  path=routing|inline, tool, action, param_keys, ok, error,
                  latency_ms, plus confidence/gate on the routing path and
                  hallucinated on the inline path
  memory_lookup   mode (ground|recall|follow-up|pushback), query, sources,
                  hits per source, best_score, grounded, elapsed_ms
  pushback        the user disputed the previous answer; refers_to names that
                  turn's packet_id so the miner can join it to its records

Every entry point swallows its own errors: a ledger failure must never reach
the cognitive loop. Off with OUTCOME_LEDGER_ENABLED=0.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger("GAIA.OutcomeLedger")

_DEFAULT_PATH = "/shared/outcomes/ledger.jsonl"
_MAX_BYTES = 20 * 1024 * 1024  # rotate to ledger.jsonl.1 past this (one generation kept)
RAW_CHARS = 400
_TEXT_CHARS = 300

_lock = threading.Lock()
# Per session, the [previous, current] turn's packet_id, so a pushback can
# name the turn it disputes even after this turn has written records of its
# own. Bounded: oldest sessions drop once it passes _MAX_SESSIONS.
_turns: Dict[str, list] = {}
_MAX_SESSIONS = 500


def enabled() -> bool:
    return os.environ.get("OUTCOME_LEDGER_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def ledger_path() -> str:
    return os.environ.get("OUTCOME_LEDGER_PATH", _DEFAULT_PATH)


def clip(text: Any, n: int = _TEXT_CHARS) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= n else s[:n] + "…"


def param_keys(params: Any) -> list:
    """The shape of a call's params (sorted keys), never their values: values
    can hold file contents or personal text, and the shape is what patterns
    are made of."""
    return sorted(params.keys()) if isinstance(params, dict) else []


def _rotate_if_needed(path: str) -> None:
    try:
        if os.path.getsize(path) > _MAX_BYTES:
            os.replace(path, path + ".1")
            logger.info("OutcomeLedger: rotated %s (> %d bytes)", path, _MAX_BYTES)
    except FileNotFoundError:
        pass


def record(kind: str, session_id: str = "", packet_id: str = "", **fields: Any) -> None:
    """Append one record. Never raises."""
    if not enabled():
        return
    try:
        rec = {"ts": round(time.time(), 3), "kind": kind,
               "session_id": session_id or "", "packet_id": packet_id or ""}
        rec.update(fields)
        line = json.dumps(rec, default=str, ensure_ascii=False)
        path = ledger_path()
        with _lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _rotate_if_needed(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            _remember(session_id, packet_id)
    except Exception:
        logger.debug("OutcomeLedger: record(%s) failed (non-fatal)", kind, exc_info=True)


def _remember(session_id: str, packet_id: str) -> None:
    """Advance the session's [previous, current] pair. Caller holds _lock."""
    if not (session_id and packet_id):
        return
    pair = _turns.pop(session_id, ["", ""])
    if pair[1] != packet_id:
        pair = [pair[1], packet_id]
    _turns[session_id] = pair  # re-insert: most recent sessions last
    while len(_turns) > _MAX_SESSIONS:
        _turns.pop(next(iter(_turns)))


def note_turn(session_id: str, packet_id: str) -> None:
    """Mark a turn even when it recorded nothing (a chat turn), so a pushback
    on it isn't attached to an older turn's tool call."""
    with _lock:
        _remember(session_id, packet_id)


def current_packet(session_id: str) -> str:
    """The session's newest turn (agent_core marks it at packet creation)."""
    with _lock:
        return _turns.get(session_id, ["", ""])[1]


def previous_packet(session_id: str, packet_id: str) -> str:
    """The turn before packet_id in this session ("" if none seen)."""
    with _lock:
        prev, cur = _turns.get(session_id, ["", ""])
        return prev if cur == packet_id else cur


def record_pushback(session_id: str, packet_id: str, text: str) -> None:
    """The user disputed the answer of the turn before this one."""
    record("pushback", session_id, packet_id,
           refers_to=previous_packet(session_id, packet_id), text=clip(text))


def record_lookup(session_id: str, packet_id: str, mode: str, query: str, ctx: Optional[Any]) -> None:
    """A memory lookup and what came back (ctx: knowledge_router.GroundingContext)."""
    try:
        results = list(getattr(ctx, "results", None) or []) if ctx is not None else []
        if ctx is not None and not results and not getattr(ctx, "sources_queried", None):
            return  # gated off before any source was asked (short query, skip intent)
        hits: Dict[str, int] = {}
        for r in results:
            src = getattr(r, "source", "") or "?"
            hits[src] = hits.get(src, 0) + 1
        best = max((float(getattr(r, "score", 0.0) or 0.0) for r in results), default=0.0)
        record("memory_lookup", session_id, packet_id,
               mode=mode, query=clip(query),
               sources=list(getattr(ctx, "sources_queried", None) or []) if ctx is not None else [],
               hits=hits, best_score=round(best, 3),
               grounded=bool(ctx is not None and getattr(ctx, "has_grounding", False)),
               elapsed_ms=round(float(getattr(ctx, "elapsed_ms", 0.0) or 0.0), 1))
    except Exception:
        logger.debug("OutcomeLedger: record_lookup failed (non-fatal)", exc_info=True)


def read(path: Optional[str] = None, kinds: Optional[Iterable[str]] = None) -> Iterable[Dict[str, Any]]:
    """Yield records, oldest first, from the rotated file then the live one.
    Unparseable lines are skipped (a crash mid-write leaves at most one)."""
    path = path or ledger_path()
    want = set(kinds) if kinds else None
    for p in (path + ".1", path):
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if want is None or rec.get("kind") in want:
                        yield rec
        except FileNotFoundError:
            continue
