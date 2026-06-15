#!/usr/bin/env python3
"""
review_discord_logs.py — standardized reader for GAIA's Discord conversations.

Discord chats are persisted as gaia-core sessions in /shared/sessions.json,
keyed by session_id:
  - discord_dm_<user_id>     : a direct message thread
  - discord_<channel_id>     : a guild/channel thread
Each session has a "history" list of {role, content, timestamp} messages.

The /logs/discord_bot.log file is only the discord.py gateway/connection log —
it does NOT contain conversation content. Use this script for the actual chats.

Run inside the gaia-core container (where /shared is mounted):
  docker exec gaia-core python /app/scripts/review_discord_logs.py            # list recent Discord sessions
  docker exec gaia-core python /app/scripts/review_discord_logs.py --latest   # print the most recent conversation
  docker exec gaia-core python /app/scripts/review_discord_logs.py -n 5        # list 5 most recent
  docker exec gaia-core python /app/scripts/review_discord_logs.py --session discord_dm_673095022815608852
  docker exec gaia-core python /app/scripts/review_discord_logs.py --errors    # only sessions whose last GAIA reply was an error fallback
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SESSIONS_PATH = os.environ.get("GAIA_SESSIONS_PATH", "/shared/sessions.json")

# GAIA's generic failure reply (agent_core._escalate_slim_response). When this
# is the last assistant message, GAIA could not answer — usually a routing /
# fallback-chain problem, not a content problem.
ERROR_FALLBACKS = (
    "I'm having trouble responding right now",
    "Please try again",
)


def _load_sessions() -> dict:
    try:
        with open(SESSIONS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        sys.exit(f"Cannot read {SESSIONS_PATH}: {exc}")
    return data if isinstance(data, dict) else {}


def _messages(sess: dict) -> list:
    return sess.get("history") or sess.get("messages") or []


def _msg_parts(m: dict):
    role = m.get("role") or m.get("author") or m.get("sender") or "?"
    content = m.get("content") or m.get("text") or m.get("message") or ""
    ts = m.get("timestamp") or m.get("ts") or ""
    return role, content, ts


def _last_ts(sess: dict) -> str:
    msgs = _messages(sess)
    if msgs:
        return _msg_parts(msgs[-1])[2] or sess.get("created_at", "")
    return sess.get("updated_at") or sess.get("created_at") or ""


def _is_error_session(sess: dict) -> bool:
    msgs = _messages(sess)
    for m in reversed(msgs):
        role, content, _ = _msg_parts(m)
        if role in ("assistant", "gaia"):
            return any(f in (content or "") for f in ERROR_FALLBACKS)
    return False


def _discord_sessions(data: dict) -> list:
    out = []
    for sid, sess in data.items():
        if isinstance(sess, dict) and str(sid).startswith("discord"):
            out.append((sid, sess))
    out.sort(key=lambda kv: _last_ts(kv[1]), reverse=True)
    return out


def _print_conversation(sid: str, sess: dict) -> None:
    print("=" * 72)
    print(f"SESSION: {sid}")
    meta = sess.get("meta") or {}
    if meta:
        print(f"meta: {json.dumps(meta)[:200]}")
    print(f"created: {sess.get('created_at', '?')}")
    print("-" * 72)
    for m in _messages(sess):
        role, content, ts = _msg_parts(m)
        flag = "  ⚠️ ERROR FALLBACK" if (role in ("assistant", "gaia") and any(f in (content or "") for f in ERROR_FALLBACKS)) else ""
        print(f"[{role}] {ts[:19]}{flag}")
        print((content or "").strip())
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Review GAIA Discord conversations from sessions.json")
    ap.add_argument("-n", "--limit", type=int, default=12, help="How many recent sessions to list (default 12)")
    ap.add_argument("--latest", action="store_true", help="Print the single most recent conversation in full")
    ap.add_argument("--session", type=str, help="Print one conversation in full by session_id")
    ap.add_argument("--errors", action="store_true", help="Only sessions whose last GAIA reply was an error fallback")
    args = ap.parse_args()

    data = _load_sessions()
    sessions = _discord_sessions(data)

    if args.session:
        sess = data.get(args.session)
        if not sess:
            sys.exit(f"No session '{args.session}' in {SESSIONS_PATH}")
        _print_conversation(args.session, sess)
        return

    if args.errors:
        sessions = [(sid, s) for sid, s in sessions if _is_error_session(s)]

    if args.latest:
        if not sessions:
            sys.exit("No Discord sessions found.")
        _print_conversation(*sessions[0])
        return

    # List mode
    print(f"{len(sessions)} Discord session(s) in {SESSIONS_PATH}"
          + (" (error-fallback only)" if args.errors else ""))
    print(f"{'last activity':19s} | msgs | err | session_id")
    print("-" * 72)
    for sid, sess in sessions[: args.limit]:
        msgs = _messages(sess)
        err = "ERR" if _is_error_session(sess) else "   "
        print(f"{_last_ts(sess)[:19]:19s} | {len(msgs):4d} | {err} | {sid}")
    print("\nTip: --latest for the newest chat, --session <id> for one, --errors for failed turns.")


if __name__ == "__main__":
    main()
