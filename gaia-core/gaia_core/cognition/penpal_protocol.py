"""Penpal Protocol — structured NotebookLM episode review + response cycle.

After each NotebookLM podcast episode is generated:
1. REVIEW: Download and transcribe the audio, analyze key points
2. RESPOND: Generate a response note addressing the hosts' discussion
3. REQUEST: Submit topics and questions for the next episode

This runs as a sleep task, triggered when a new podcast is detected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.Penpal")

MCP_ENDPOINT = os.environ.get("MCP_ENDPOINT", "http://gaia-mcp:8765/jsonrpc")
NOTEBOOK_ID = os.environ.get("PENPAL_NOTEBOOK_ID", "7cb1f61e-84a9-445f-9bb9-899b3820a0dc")
PENPAL_DIR = Path(os.environ.get("SHARED_DIR", "/shared")) / "penpal"
STATE_PATH = PENPAL_DIR / "state.json"
# fwzo: transcripts are dropped here as <artifact_id>.txt. The MCP relay
# can't transcribe full episodes (~88 MB m4a, no X-GAIA-Auth), so an
# episode with no transcript waits here instead of being reviewed blind.
TRANSCRIPT_DIR = PENPAL_DIR / "transcripts"
EPISODE_DIR = PENPAL_DIR / "episodes"
LETTER_DIR = PENPAL_DIR / "letters"
# A letter goes in as a source (the hosts read sources, not notes), which is
# a sensitive tool: Azrael approves it from his Discord DM. Asked at most this
# many times per episode; the letter stays on disk either way.
MAX_ASKS = 3
APPROVAL_WINDOW_S = 15 * 60   # gaia-mcp's own expiry
WEB_URL = os.environ.get("GAIA_WEB_URL", "http://gaia-web:6414")
MIN_TRANSCRIPT_CHARS = 500

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _mcp_call(method: str, params: dict) -> dict:
    """Call an MCP tool. Raises RuntimeError on a JSON-RPC error."""
    payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    req = Request(MCP_ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read().decode())
    if isinstance(d, list):
        d = d[0]
    if d.get("error"):
        raise RuntimeError(f"{method}: {d['error']}")
    return d.get("result", d)


def _load_state() -> dict:
    """Load penpal state (last reviewed episode, etc.)."""
    state = {}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}
    state.setdefault("awaiting_approval", {})
    state.setdefault("last_reviewed_id", None)
    state.setdefault("last_reviewed_at", None)
    state.setdefault("reviews", [])
    state.setdefault("awaiting_transcript", {})
    return state


def _save_state(state: dict) -> None:
    """Save penpal state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _audio_episodes(artifacts: List[Dict]) -> List[Dict]:
    """Completed audio artifacts, oldest first."""
    eps = [a for a in artifacts
           if a.get("is_completed") and "AUDIO" in str(a.get("kind", "")).upper()]
    return sorted(eps, key=lambda a: a.get("created_at") or "")


def check_for_new_episodes() -> List[Dict]:
    """Check NotebookLM for episodes we haven't reviewed yet.

    fwzo: with no state yet, the newest episode becomes the starting
    point instead of the whole back catalogue being reviewed (and posted
    to) two an hour. Newness is by created_at, so a reviewed id that has
    since vanished from the list can't stall the cycle.
    """
    state = _load_state()
    result = _mcp_call("notebooklm_list_artifacts", {"notebook_id": NOTEBOOK_ID})
    episodes = _audio_episodes(result.get("artifacts", []))
    if not episodes:
        return []

    last_at = state.get("last_reviewed_at")
    if not last_at and state.get("last_reviewed_id"):
        last_at = next((a.get("created_at") for a in episodes
                        if a.get("id") == state["last_reviewed_id"]), None)
    if not last_at:
        newest = episodes[-1]
        state["last_reviewed_id"] = newest.get("id")
        state["last_reviewed_at"] = newest.get("created_at")
        _save_state(state)
        logger.info("Penpal: no review history; starting after '%s' (%d earlier episodes skipped)",
                    newest.get("title"), len(episodes))
        return []

    awaiting = state["awaiting_transcript"]
    return [a for a in episodes
            if (a.get("created_at") or "") > last_at or a.get("id") in awaiting]


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _get_transcript(artifact: Dict, state: dict) -> Optional[str]:
    """A real transcript for the episode, or None.

    A dropped-in transcript wins. Otherwise the audio is downloaded once
    (kept for manual transcription) and the relay's transcription used if
    it produced one. Episodes still without one are parked in
    state["awaiting_transcript"] and not re-downloaded.
    """
    episode_id = artifact.get("id", "unknown")
    tfile = TRANSCRIPT_DIR / f"{episode_id}.txt"
    if tfile.exists():
        text = tfile.read_text(errors="replace").strip()
        if len(text) >= MIN_TRANSCRIPT_CHARS:
            return text

    awaiting = state["awaiting_transcript"]
    if episode_id in awaiting:
        return None

    audio_path = EPISODE_DIR / f"{episode_id}.mp4"
    text = ""
    try:
        EPISODE_DIR.mkdir(parents=True, exist_ok=True)
        download = _mcp_call("notebooklm_download_audio", {
            "notebook_id": NOTEBOOK_ID,
            "artifact_id": episode_id,
            "save_path": str(audio_path),
        })
        text = (download.get("transcription") or download.get("transcript") or "").strip()
        if download.get("transcribe_error"):
            logger.info("Penpal: relay could not transcribe %s: %s",
                        episode_id, download["transcribe_error"])
    except Exception as e:
        logger.warning("Penpal: download of %s failed: %s", episode_id, e)
        return None  # not parked: retry the download next cycle

    if len(text) >= MIN_TRANSCRIPT_CHARS:
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        tfile.write_text(text)
        return text

    awaiting[episode_id] = {
        "title": artifact.get("title"),
        "audio_path": str(audio_path),
        "since": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(state)
    logger.info("Penpal: '%s' is waiting for a transcript at %s", artifact.get("title"), tfile)
    return None


def review_episode(artifact: Dict, model_endpoint: str = "http://gaia-prime:7777") -> Optional[Dict]:
    """Review a podcast episode: transcript, her letter, the letter as a source.

    Returns the review dict, or None when the episode was left for a later
    cycle (no transcript yet, generation failed, or the letter is waiting on
    Azrael's approval). Only the letter being in the notebook marks the
    episode reviewed. It went in as a note before, which the hosts never read,
    and as a sensitive tool it was refused outright from here (2026-11-11).
    """
    episode_id = artifact.get("id", "unknown")
    title = artifact.get("title", "Untitled")
    logger.info("Reviewing episode: %s — %s", episode_id, title)

    state = _load_state()
    transcript = _get_transcript(artifact, state)
    if transcript is None:
        return None

    letter_path = LETTER_DIR / f"{episode_id}.txt"
    response_text = letter_path.read_text().strip() if letter_path.exists() else ""
    if not response_text:
        response_text = _write_letter(title, transcript, model_endpoint)
        if not response_text:
            return None
        LETTER_DIR.mkdir(parents=True, exist_ok=True)
        letter_path.write_text(response_text)

    source_title = f"GAIA's letter: reply to {title}"
    if not _letter_in_notebook(source_title):
        _ask_to_send(episode_id, title, source_title, response_text, letter_path)
        return None
    logger.info("Penpal: letter for '%s' is in the notebook", title)

    review = {
        "episode_id": episode_id,
        "title": title,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "response_length": len(response_text),
        "response_preview": response_text[:200],
    }

    state = _load_state()
    created = artifact.get("created_at") or ""
    if created > (state.get("last_reviewed_at") or ""):
        state["last_reviewed_id"] = episode_id
        state["last_reviewed_at"] = created
    state["awaiting_transcript"].pop(episode_id, None)
    state["awaiting_approval"].pop(episode_id, None)
    state["reviews"].insert(0, review)
    state["reviews"] = state["reviews"][:20]
    _save_state(state)

    return review


def _write_letter(title: str, transcript: str, model_endpoint: str) -> str:
    review_prompt = f"""You are GAIA, reviewing a podcast episode about your own development.

EPISODE: {title}
TRANSCRIPT: {transcript[:6000]}

Write a response note that:
1. Acknowledges the key points the hosts discussed
2. Provides your perspective on their analysis (agree, disagree, add context)
3. Shares what has changed since the episode was recorded
4. Proposes 2-3 topics or questions for the next episode

Keep it conversational and genuine. This is a penpal exchange, not a report."""
    try:
        payload = json.dumps({
            "messages": [{"role": "user", "content": review_prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
        }).encode()
        req = Request(f"{model_endpoint}/v1/chat/completions", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode())
        text = _strip_think(result.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    except Exception as e:
        logger.warning("Penpal: response generation for '%s' failed: %s", title, e)
        return ""
    if not text:
        logger.warning("Penpal: the model returned an empty response for '%s'", title)
    return text


def _letter_in_notebook(source_title: str) -> bool:
    try:
        result = _mcp_call("notebooklm_list_sources", {"notebook_id": NOTEBOOK_ID})
    except Exception as e:
        logger.warning("Penpal: could not list sources: %s", e)
        return False
    return any(s.get("title") == source_title for s in result.get("sources", []))


def _azrael_dm() -> str:
    """Azrael's Discord DM session, where the approval is asked for."""
    from gaia_common.utils.people import CREATOR, PeopleRegistry
    for person in PeopleRegistry().find_by_name(CREATOR)[:1]:
        for account in person.get("accounts", []):
            if account.get("kind") == "discord" and account.get("id"):
                return f"discord_dm_{account['id']}"
    return ""


def _push(session_id: str, text: str) -> bool:
    payload = json.dumps({"text": text, "source": "penpal"}).encode()
    req = Request(f"{WEB_URL}/api/conversations/{session_id}/push", data=payload, method="POST",
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=15) as r:
            return bool(json.loads(r.read().decode() or "{}").get("delivered"))
    except Exception as e:
        logger.warning("Penpal: could not message %s: %s", session_id, e)
        return False


def _ask_to_send(episode_id: str, title: str, source_title: str, letter: str, letter_path: Path) -> None:
    """Park the source upload and ask Azrael for the code in his DM. Once
    approved, gaia-mcp adds it; the next cycle sees it and marks the episode."""
    state = _load_state()
    asked = state["awaiting_approval"].get(episode_id) or {}
    if asked and time.time() - asked.get("at", 0) < APPROVAL_WINDOW_S:
        return                                   # still waiting on this ask
    if asked.get("asks", 0) >= MAX_ASKS:
        if not asked.get("gave_up"):
            asked["gave_up"] = True
            state["awaiting_approval"][episode_id] = asked
            _save_state(state)
            logger.info("Penpal: '%s' asked %d times; the letter stays at %s", title, MAX_ASKS, letter_path)
        return
    session = _azrael_dm()
    if not session:
        logger.warning("Penpal: no Discord account for Azrael to ask; letter kept at %s", letter_path)
        return
    from gaia_core.cognition import approval_reply
    from gaia_core.utils.mcp_client import request_approval_via_mcp
    params = {"notebook_id": NOTEBOOK_ID, "title": source_title, "content": letter}
    parked = request_approval_via_mcp("notebooklm_add_text_source", params)
    if not (parked.get("ok") and parked.get("action_id")):
        logger.warning("Penpal: could not park the letter for approval: %s", parked.get("error"))
        return
    label = f"send my letter about \"{title}\" to the notebook"
    approval_reply.park_for(session, parked["action_id"], label)
    opening = " ".join(letter.split())[:300]
    sent = _push(session, (
        f"I've written back to the podcast hosts about \"{title}\". It opens: \"{opening}…\" "
        f"(the whole letter is at {letter_path}). To add it to the notebook so they read it, "
        f"reply here with your current 6-digit authenticator code in the next 15 minutes, or say cancel."))
    state["awaiting_approval"][episode_id] = {"at": time.time(), "asks": asked.get("asks", 0) + 1,
                                              "action_id": parked["action_id"], "delivered": sent}
    _save_state(state)
    logger.info("Penpal: asked Azrael to send the letter for '%s' (ask %d, delivered=%s)",
                title, asked.get("asks", 0) + 1, sent)


def run_penpal_cycle() -> Dict:
    """Full penpal cycle: check for new episodes, review them.

    Called from sleep task scheduler.
    """
    new_episodes = check_for_new_episodes()
    if not new_episodes:
        logger.debug("Penpal: no new episodes to review")
        return {"new_episodes": 0}

    reviews = []
    for episode in new_episodes[:2]:  # Max 2 per cycle
        try:
            review = review_episode(episode)
            if review:
                reviews.append(review)
        except Exception as e:
            logger.warning("Penpal review failed for %s: %s", episode.get("title"), e)

    return {"new_episodes": len(new_episodes), "reviewed": len(reviews), "reviews": reviews,
            "awaiting_transcript": len(_load_state()["awaiting_transcript"]),
            "awaiting_approval": len(_load_state()["awaiting_approval"])}
