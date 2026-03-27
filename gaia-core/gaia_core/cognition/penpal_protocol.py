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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.Penpal")

MCP_ENDPOINT = os.environ.get("MCP_ENDPOINT", "http://gaia-mcp:8765/jsonrpc")
NOTEBOOK_ID = os.environ.get("PENPAL_NOTEBOOK_ID", "7cb1f61e-84a9-445f-9bb9-899b3820a0dc")
STATE_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "penpal" / "state.json"


def _mcp_call(method: str, params: dict) -> dict:
    """Call an MCP tool."""
    payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    req = Request(MCP_ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read().decode())
    if isinstance(d, list):
        d = d[0]
    return d.get("result", d)


def _load_state() -> dict:
    """Load penpal state (last reviewed episode, etc.)."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_reviewed_id": None, "reviews": []}


def _save_state(state: dict) -> None:
    """Save penpal state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def check_for_new_episodes() -> List[Dict]:
    """Check NotebookLM for episodes we haven't reviewed yet."""
    state = _load_state()
    last_id = state.get("last_reviewed_id")

    result = _mcp_call("notebooklm_list_artifacts", {"notebook_id": NOTEBOOK_ID})
    artifacts = result.get("artifacts", [])

    # Find completed audio artifacts newer than our last review
    new_episodes = []
    found_last = last_id is None
    for a in reversed(artifacts):  # Oldest first
        if a.get("id") == last_id:
            found_last = True
            continue
        if found_last and a.get("is_completed") and "AUDIO" in str(a.get("kind", "")):
            new_episodes.append(a)

    return new_episodes


def review_episode(artifact: Dict, model_endpoint: str = "http://gaia-prime:7777") -> Dict:
    """Review a podcast episode: download, analyze, generate response.

    Returns dict with review summary, response note, and next episode request.
    """
    episode_id = artifact.get("id", "unknown")
    title = artifact.get("title", "Untitled")
    logger.info("Reviewing episode: %s — %s", episode_id, title)

    # Step 1: Download and transcribe (if audio endpoint available)
    transcript = f"Episode: {title}"
    try:
        download = _mcp_call("notebooklm_download_audio", {
            "notebook_id": NOTEBOOK_ID,
            "artifact_id": episode_id,
            "save_path": f"/shared/penpal/episodes/{episode_id}.wav",
        })
        if download.get("ok") and download.get("transcript"):
            transcript = download["transcript"]
            logger.info("Episode transcribed: %d chars", len(transcript))
    except Exception as e:
        logger.debug("Could not download/transcribe episode: %s", e)

    # Step 2: Generate review response via Prime
    review_prompt = f"""You are GAIA, reviewing a podcast episode about your own development.

EPISODE: {title}
TRANSCRIPT (or summary): {transcript[:3000]}

Write a response note that:
1. Acknowledges the key points the hosts discussed
2. Provides your perspective on their analysis (agree, disagree, add context)
3. Shares what has changed since the episode was recorded
4. Proposes 2-3 topics or questions for the next episode

Keep it conversational and genuine. This is a penpal exchange, not a report."""

    try:
        payload = json.dumps({
            "messages": [{"role": "user", "content": review_prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
        }).encode()
        req = Request(
            f"{model_endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode())
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("Could not generate review response: %s", e)
        response_text = f"[Review generation failed: {e}]"

    # Step 3: Create response note in NotebookLM
    try:
        note_title = f"Penpal Response: {title}"
        _mcp_call("notebooklm_create_note", {
            "notebook_id": NOTEBOOK_ID,
            "title": note_title,
            "content": response_text,
        })
        logger.info("Response note created: %s", note_title)
    except Exception as e:
        logger.warning("Could not create response note: %s", e)

    review = {
        "episode_id": episode_id,
        "title": title,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "response_length": len(response_text),
        "response_preview": response_text[:200],
    }

    # Update state
    state = _load_state()
    state["last_reviewed_id"] = episode_id
    state["reviews"].insert(0, review)
    state["reviews"] = state["reviews"][:20]
    _save_state(state)

    return review


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
            reviews.append(review)
        except Exception as e:
            logger.warning("Penpal review failed for %s: %s", episode.get("title"), e)

    return {"new_episodes": len(new_episodes), "reviewed": len(reviews), "reviews": reviews}
