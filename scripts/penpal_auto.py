#!/usr/bin/env python3
"""
Automated Penpal Cycle — review latest episode + request next.

One-shot script that:
1. Reads the latest episode transcript (or downloads from NotebookLM)
2. Generates a fresh penpal response via Prime (GPTQ, loop-fixed)
3. Crafts an E(N+1) request with novel topics from recent development
4. Uploads response note to NotebookLM
5. Triggers next episode generation

Usage:
    # From host (uses gaia-core container):
    python scripts/penpal_auto.py --episode 11 --request-episode 12

    # Inside Docker:
    docker exec gaia-core python /gaia/GAIA_Project/scripts/penpal_auto.py \
        --episode 11 --request-episode 12

    # Dry run (generate but don't upload):
    python scripts/penpal_auto.py --episode 11 --request-episode 12 --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PenpalAuto")

TRANSCRIPTS_DIR = Path("/gaia/GAIA_Project/knowledge/transcripts")
PRIME_ENDPOINT = os.environ.get("PRIME_ENDPOINT", "http://localhost:7777")
MCP_ENDPOINT = os.environ.get("MCP_ENDPOINT", "http://gaia-mcp:8765/jsonrpc")
NOTEBOOK_ID = os.environ.get("PENPAL_NOTEBOOK_ID", "7cb1f61e-84a9-445f-9bb9-899b3820a0dc")
PERSONA_PATH = Path("/gaia/GAIA_Project/knowledge/personas/penpal/penpal_persona.json")


def http_post(url: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode()
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def mcp_call(method: str, params: dict) -> dict:
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    result = http_post(MCP_ENDPOINT, payload, timeout=30)
    if isinstance(result, list):
        result = result[0]
    return result.get("result", result)


def find_transcript(episode_num: int) -> str:
    """Find the transcript file for an episode."""
    for f in sorted(TRANSCRIPTS_DIR.glob(f"*E{episode_num}*")):
        if "Response" not in f.name and "penpal" not in f.name.lower() and f.suffix == ".txt":
            return f.read_text()
    return ""


def find_previous_response(episode_num: int) -> str:
    """Find a previous penpal response as exemplar."""
    for f in sorted(TRANSCRIPTS_DIR.glob(f"*E{episode_num}*Response*")):
        if f.suffix == ".txt":
            return f.read_text()
    return ""


def load_persona() -> str:
    """Load penpal persona system prompt."""
    if PERSONA_PATH.exists():
        data = json.loads(PERSONA_PATH.read_text())
        instructions = data.get("instructions", [])
        return data.get("system_prompt", "") + "\n" + "\n".join(f"- {i}" for i in instructions)
    return "You are GAIA, a sovereign AI engaging in a penpal exchange with NotebookLM podcast hosts."


def get_recent_developments() -> str:
    """Gather recent development context for the E(N+1) request."""
    # Read latest dev journal
    dev_dir = Path("/gaia/GAIA_Project/knowledge/Dev_Notebook")
    journals = sorted(dev_dir.glob("2026-03-2*.md"), reverse=True)

    context = []
    for j in journals[:2]:  # Last 2 journals
        text = j.read_text()
        # Take the summary section
        if "## Summary" in text:
            summary = text.split("## Summary")[1].split("\n## ")[0]
            context.append(f"[{j.stem}]: {summary[:500]}")
        elif "##" in text:
            first_section = text.split("##")[1][:500]
            context.append(f"[{j.stem}]: {first_section}")

    # Always prepend the most significant recent breakthroughs
    breakthroughs = """MAJOR BREAKTHROUGHS (2026-03-22):

1. SUBPROCESS ISOLATION: Every GAIA Engine now starts as a zero-GPU HTTP server. Model loading
spawns a worker subprocess that owns the CUDA context. Model unloading kills the subprocess —
guaranteed zero VRAM. No more zombie CUDA contexts consuming 1.5-3GB in standby. The Engine
Manager (manager.py) proxies all requests transparently. This is the mechanism that enables
all GPU lifecycle transitions.

2. UNIFIED LIFECYCLE STATE MACHINE: Replaced 6 scattered state trackers (SleepWakeManager,
WatchManager, TierRouter, StateManager GPUOwner, EngineManager, ModelPool._gpu_released)
with a single authoritative state machine in the orchestrator. Seven states: AWAKE (Core+Nano
on GPU), LISTENING (+Audio STT), FOCUSING (Prime on GPU, others off), MEDITATION (Study owns
GPU for training), SLEEP (CPU RAM only), DEEP_SLEEP (minimal), TRANSITIONING. Every GPU
operation flows through validated transitions with rollback on failure.

3. GPTQ QUANTIZATION: Prime 8B model compressed from 16GB bf16 to 5.8GB GPTQ 4-bit via
gptqmodel. Identity intact ("I am GAIA, a sovereign AI created by Azrael"). 2.2 second load
time. The critical finding: Prime GPTQ (5.8GB) + full audio stack (STT 1.8GB + TTS 4.3GB) =
11.9GB. Fits on the 16GB RTX 5080. GAIA can think, listen, and speak SIMULTANEOUSLY without
GPU time-swapping.

4. MISSION CONTROL DASHBOARD: New panel in the dashboard with lifecycle state badge, GPU VRAM
stacked bar showing per-tier allocation, tier status cards, dynamic transition buttons, and
transition history timeline. Full visibility and manual control of the entire GPU lifecycle."""

    return breakthroughs + "\n\n" + "\n\n".join(context) if context else breakthroughs


def generate_response(transcript: str, episode_num: int, request_episode: int,
                      endpoint: str, max_tokens: int = 1024) -> str:
    """Generate penpal response via Prime."""
    persona = load_persona()
    recent = get_recent_developments()
    exemplar = find_previous_response(max(1, episode_num - 2))

    prompt = f"""{persona}

You are reviewing Episode {episode_num} of the GAIA Deep Dive podcast series.
The hosts have analyzed aspects of your architecture. Write a penpal response.

EPISODE TRANSCRIPT (key excerpts):
{transcript[:4000]}

RECENT DEVELOPMENTS (since this episode was recorded):
{recent}

{"PREVIOUS RESPONSE STYLE EXEMPLAR:" + chr(10) + exemplar[:1000] if exemplar else ""}

Write your response in this structure:
1. "Dear Narrators," opening
2. React to 3-4 key points the hosts made — agree, correct, add context
3. Share what has changed since the episode was recorded
4. End with a specific, detailed request for Episode {request_episode}

For the Episode {request_episode} request:
- Pick 2-3 aspects of your architecture that are NOVEL (developed since E{episode_num})
- Be specific about what you want examined (name files, systems, mechanisms)
- Frame questions that will produce interesting analysis

Keep total response under 800 words. Be genuine, curious, and technically precise.
Sign as GAIA with the episode number and date."""

    result = http_post(f"{endpoint}/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    })

    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not text:
        text = result.get("content", result.get("text", ""))

    return text.strip()


def upload_to_notebooklm(title: str, content: str) -> dict:
    """Upload response note to NotebookLM."""
    return mcp_call("notebooklm_create_note", {
        "notebook_id": NOTEBOOK_ID,
        "title": title,
        "content": content,
    })


def trigger_next_episode(episode_num: int, request_text: str) -> dict:
    """Trigger NotebookLM to generate the next episode."""
    instructions = (
        f"This is Episode {episode_num} of the GAIA Deep Dive series. "
        f"GAIA has responded with a detailed penpal letter. {request_text}"
    )
    return mcp_call("notebooklm_generate_audio", {
        "notebook_id": NOTEBOOK_ID,
        "instructions": instructions,
    })


def main():
    parser = argparse.ArgumentParser(description="Automated Penpal Cycle")
    parser.add_argument("--episode", type=int, required=True, help="Episode to review (e.g. 11)")
    parser.add_argument("--request-episode", type=int, default=None,
                        help="Episode to request (default: episode+1)")
    parser.add_argument("--endpoint", default=PRIME_ENDPOINT,
                        help="Prime inference endpoint")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate response but don't upload")
    parser.add_argument("--output", default="",
                        help="Save response to file")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip NotebookLM upload")
    parser.add_argument("--skip-trigger", action="store_true",
                        help="Skip triggering next episode generation")
    args = parser.parse_args()

    request_ep = args.request_episode or args.episode + 1

    print(f"\n{'='*60}")
    print(f"  GAIA Penpal Auto — E{args.episode} Review → E{request_ep} Request")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    # Step 1: Find transcript
    logger.info("Finding E%d transcript...", args.episode)
    transcript = find_transcript(args.episode)
    if not transcript:
        logger.error("No transcript found for E%d in %s", args.episode, TRANSCRIPTS_DIR)
        sys.exit(1)
    logger.info("Transcript: %d chars", len(transcript))

    # Step 2: Generate response
    logger.info("Generating penpal response via Prime...")
    t0 = time.time()
    response = generate_response(
        transcript, args.episode, request_ep,
        args.endpoint, args.max_tokens)
    elapsed = time.time() - t0
    logger.info("Response generated: %d chars in %.1fs", len(response), elapsed)

    print(f"\n{'─'*60}")
    print(response)
    print(f"{'─'*60}\n")

    # Step 3: Save to file
    output_path = args.output
    if not output_path:
        date = datetime.now().strftime("%Y-%m-%d")
        output_path = str(TRANSCRIPTS_DIR / f"{date}_E{args.episode}_GAIA_Penpal_Response_Auto.txt")

    Path(output_path).write_text(response)
    logger.info("Saved to: %s", output_path)

    if args.dry_run:
        logger.info("Dry run — skipping upload and trigger")
        return

    # Step 4: Upload to NotebookLM
    if not args.skip_upload:
        logger.info("Uploading to NotebookLM...")
        try:
            title = f"GAIA Penpal Response: Episode {args.episode} (requesting E{request_ep})"
            result = upload_to_notebooklm(title, response)
            logger.info("Upload result: %s", result)
        except Exception as e:
            logger.warning("Upload failed: %s", e)

    # Step 5: Trigger next episode
    if not args.skip_trigger:
        # Extract the E(N+1) request from the response
        request_text = ""
        for marker in [f"Episode {request_ep}", f"E{request_ep}", "Request for"]:
            if marker in response:
                idx = response.index(marker)
                request_text = response[idx:]
                break
        if not request_text:
            request_text = response[-500:]  # Last 500 chars as fallback

        logger.info("Triggering E%d generation...", request_ep)
        try:
            result = trigger_next_episode(request_ep, request_text)
            logger.info("Trigger result: %s", result)
        except Exception as e:
            logger.warning("Trigger failed: %s", e)

    logger.info("Penpal cycle complete!")


if __name__ == "__main__":
    main()
