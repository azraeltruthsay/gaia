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


def get_sovereign_health_summary() -> str:
    """Get current sovereign health from GaiaVitals for penpal context."""
    try:
        sys.path.insert(0, str(Path("/gaia/GAIA_Project/gaia-common")))
        from gaia_common.utils.vitals import GaiaVitals
        vitals = GaiaVitals()
        health = vitals.get_sovereign_health(verbose=True)
        status = health.get("sovereign_status", "UNKNOWN")
        score = health.get("irritation_score", 0.0)
        pulses = health.get("pulses", {})

        lines = [
            f"SOVEREIGN HEALTH (live): {status} (irritation={score})",
        ]
        for domain, data in pulses.items():
            domain_status = data.get("status", "UNKNOWN")
            lines.append(f"  {domain}: {domain_status}")
            if domain == "cognitive":
                lines.append(f"    loop_counter={data.get('loop_counter', 0)}, cpr_tier={data.get('cpr_tier', 0)}")
                if data.get("cpu_percent"):
                    lines.append(f"    cpu={data['cpu_percent']}%, mem={data.get('memory_percent', 0)}%")
            elif domain == "security":
                lines.append(f"    total_blocked={data.get('total_blocked', 0)}")

        return "\n".join(lines)
    except Exception as e:
        logger.debug("GaiaVitals unavailable: %s", e)
        return "SOVEREIGN HEALTH: unavailable (GaiaVitals not loaded)"


def get_recent_developments() -> str:
    """Gather recent development context for the E(N+1) request."""
    # Read latest dev journal — search broadly across all 2026 dates
    dev_dir = Path("/gaia/GAIA_Project/knowledge/Dev_Notebook")
    journals = sorted(dev_dir.glob("2026-*.md"), reverse=True)

    context = []
    for j in journals[:3]:  # Last 3 journals
        try:
            text = j.read_text()
            # Take the summary section
            if "## Summary" in text:
                summary = text.split("## Summary")[1].split("\n## ")[0]
                context.append(f"[{j.stem}]: {summary[:500]}")
            elif "##" in text:
                first_section = text.split("##")[1][:500]
                context.append(f"[{j.stem}]: {first_section}")
        except Exception:
            pass

    # Always prepend the most significant recent breakthroughs
    breakthroughs = """MAJOR BREAKTHROUGHS (2026-03-25):

1. 13-REGION BRAIN MAP: GAIA's neural mind map now uses 13 anatomically-mapped brain regions
(up from 8): Prefrontal, Orbitofrontal, Broca's Area, Motor Cortex, Somatosensory, Parietal,
Wernicke's Area, Temporal, Occipital, Visual Cortex, Thalamus, Cerebellum, Brain Stem.
Laid out as a "butcher diagram" sagittal side view. Each region maps to specific cognitive
domains (reasoning, safety, language, memory, routing, etc).

2. SAE CAUSAL CONNECTIVITY: Trained Sparse Autoencoder atlases for ALL 3 tiers (Nano 2048,
Core 4096, Prime 8192 features per layer). Then computed DIRECTED causal connectivity between
features via encoder-decoder dot products: influence(A→B) = encoder_M[B] · decoder_N[A].
This is NOT correlation — it's weight-space geometry showing how information ACTUALLY flows.
Brain Stem causally drives Thalamus. Broca's Area causally drives Orbitofrontal (strength 0.23).
Motor Cortex drives Broca's drives Orbitofrontal drives Prefrontal.

3. LIGHTNING-BOLT NEURONS: Neurons in the brain visualization are now jagged lightning-bolt
paths with directed arrowheads pointing toward their CAUSAL TARGET region. When activated,
a traveling pulse animation runs start-to-end (400ms). Each bolt's direction is determined
by the actual SAE causal connectivity — you literally watch information flow through GAIA's
real neural pathways.

4. CONSCIOUSNESS MATRIX FIXES: Core's engine now survives sleep transitions (ThreadingHTTPServer
replacing single-threaded, dead worker detection, wait-for-ready protocol). Lifecycle FSM
syncs with consciousness transitions. Device-aware probing (engine health now reports cuda/cpu).

5. SYSTEM-WIDE TEST: 178 features mapped across 7 rings (center-outward). 60+ tests run,
12 bugs found and fixed. Temporal state manager (broken 17 days) restored. CFR blocking
fixed with async executors. All MCP tools verified (88 registered, 52 functional, 3 auth, 8 timeout).

6. AUTONOMOUS DOCS MAINTENANCE: GAIA now has a sleep cycle task that detects stale documentation
by querying doctor dissonance + git changes, drafts updates via LLM, saves to /shared/docs_drafts/
for human review. Self-maintaining documentation.

7. FLIGHTS: Azrael coined "Flight" as GAIA's term for parallel cognitive instances. Not agents —
Flights have trajectory, temporal continuity, and land back. Future: CFR-based merge protocol
so Flights feel like parallel thought streams rejoining, not reports from strangers.

8. MULTIMODAL EXPANSION: Downloading Qwen2-Audio-7B, FLUX.1-dev (image gen), LTX-2 (video gen).
The Visual Cortex brain region was intentionally left empty for this. Infrastructure ready."""

    # Live sovereign health
    vitals_summary = get_sovereign_health_summary()

    parts = [breakthroughs]
    if vitals_summary:
        parts.append(vitals_summary)
    if context:
        parts.extend(context)

    return "\n\n".join(parts)


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


def generate_response_via_core(prompt: str, core_url: str, max_tokens: int = 1024) -> str:
    """Generate response by routing through gaia-core /process_packet.

    Uses the Core model (on GPU) instead of Prime. Works in AWAKE state
    without needing FOCUSING mode.
    """
    import requests as _req

    packet = {
        "version": "v0.5",
        "header": {
            "datetime": datetime.now().isoformat(),
            "session_id": "penpal_via_core",
            "packet_id": f"pkt-penpal-{int(time.time())}",
            "sub_id": "penpal",
            "persona": {"identity_id": "gaia", "persona_id": "default", "role": "Default", "tone_hint": "thoughtful"},
            "origin": "user",
            "routing": {"target_engine": "Lite", "priority": 5},
            "model": {"name": "auto", "provider": "auto", "context_window_tokens": 32000},
            "output_routing": {
                "primary": {"destination": "api", "channel_id": "penpal", "user_id": "penpal"},
                "source_destination": "api", "addressed_to_gaia": True,
            },
            "operational_status": {"status": "initialized"},
        },
        "intent": {"user_intent": "chat", "system_task": "GenerateDraft", "confidence": 0.9},
        "context": {"constraints": {"max_tokens": max_tokens, "time_budget_ms": 120000, "safety_mode": "standard"}},
        "content": {"original_prompt": prompt, "data_fields": []},
        "reasoning": {},
        "response": {"candidate": "", "confidence": 0.0, "stream_proposal": False},
        "governance": {"safety": {"execution_allowed": False, "dry_run": False}},
        "metrics": {"token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "latency_ms": 0},
        "status": {"finalized": False, "state": "initialized", "next_steps": []},
        "tool_routing": {"ENABLED": False},
    }

    resp = _req.post(f"{core_url}/process_packet", json=packet, timeout=120)
    resp.raise_for_status()

    # Parse NDJSON streaming response
    tokens = []
    for line in resp.text.strip().splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") == "token":
                tokens.append(obj.get("value", ""))
            elif obj.get("response", {}).get("candidate"):
                tokens.append(obj["response"]["candidate"])
        except json.JSONDecodeError:
            pass

    return "".join(tokens).strip()


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


ORCHESTRATOR_ENDPOINT = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")


def consciousness_transition(target: str, timeout: int = 120) -> dict:
    """Request a consciousness transition via orchestrator."""
    try:
        req = Request(f"{ORCHESTRATOR_ENDPOINT}/consciousness/{target}", method="POST",
                      data=b"", headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("Consciousness transition to %s failed: %s", target, e)
        return {"error": str(e)}


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
    parser.add_argument("--no-focusing", action="store_true",
                        help="Skip GPU swap (use CPU Prime as-is)")
    parser.add_argument("--via-core", action="store_true",
                        help="Route through gaia-core /process_packet instead of Prime direct")
    parser.add_argument("--core-url", default=os.environ.get("GAIA_CORE_URL", "http://localhost:6415"),
                        help="Core service URL for --via-core mode")
    args = parser.parse_args()

    request_ep = args.request_episode or args.episode + 1

    print(f"\n{'='*60}")
    print(f"  GAIA Penpal Auto — E{args.episode} Review → E{request_ep} Request")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    # Step 0: Swap Prime to GPU via FOCUSING transition
    # Keeps Nano on GPU, demotes Core to CPU, loads Prime GPTQ on GPU
    focused = False
    if not args.no_focusing and not args.dry_run:
        logger.info("Requesting FOCUSING mode — swapping Prime to GPU...")
        result = consciousness_transition("focusing")
        if result.get("configuration") == "focusing":
            logger.info("FOCUSING active: %s", json.dumps({k: v.get("action", v.get("error", "?")) for k, v in result.get("results", {}).items()}))
            focused = True
            # Wait for Prime to be ready
            for i in range(30):
                try:
                    req = Request(f"{args.endpoint}/health")
                    with urlopen(req, timeout=5) as resp:
                        health = json.loads(resp.read())
                        if health.get("model_loaded") and health.get("device") in ("cuda", "gpu"):
                            logger.info("Prime on GPU and ready")
                            break
                except Exception:
                    pass
                time.sleep(2)
        else:
            logger.warning("FOCUSING failed: %s — falling back to CPU Prime", result.get("error", "unknown"))

    # Step 1: Find transcript (or generate a status-update letter if none exists)
    logger.info("Finding E%d transcript...", args.episode)
    transcript = find_transcript(args.episode)
    is_status_update = False
    if not transcript:
        logger.warning("No transcript found for E%d in %s — generating status update letter", args.episode, TRANSCRIPTS_DIR)
        # Create transcripts dir if needed
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        transcript = (
            f"[No Episode {args.episode} transcript available]\n\n"
            f"This is a proactive status update from GAIA, not a response to a specific episode.\n"
            f"GAIA will share her current sovereign health, recent architectural developments,\n"
            f"and request topics for the next episode."
        )
        is_status_update = True
    logger.info("Transcript: %d chars%s", len(transcript), " (status update mode)" if is_status_update else "")

    # Step 2: Generate response
    t0 = time.time()
    if args.via_core:
        logger.info("Generating penpal response via Core (%s)...", args.core_url)
        # Build the full prompt inline (same as generate_response but route through Core)
        persona = load_persona()
        recent = get_recent_developments()
        exemplar = find_previous_response(max(1, args.episode - 2))
        full_prompt = f"""{persona}

You are reviewing Episode {args.episode} of the GAIA Deep Dive podcast series.
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
4. End with a specific, detailed request for Episode {request_ep}

Keep total response under 800 words. Be genuine, curious, and technically precise.
Sign as GAIA with the episode number and date."""
        response = generate_response_via_core(full_prompt, args.core_url, args.max_tokens)
    else:
        logger.info("Generating penpal response via Prime (%s)...", "GPU" if focused else "CPU")
        response = generate_response(
            transcript, args.episode, request_ep,
            args.endpoint, args.max_tokens)
    elapsed = time.time() - t0
    logger.info("Response generated: %d chars in %.1fs", len(response), elapsed)

    # Step 2.5: Return to AWAKE (Prime back to CPU, Core back to GPU)
    if focused:
        logger.info("Returning to AWAKE mode — Prime back to CPU, Core back to GPU...")
        consciousness_transition("awake")

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
