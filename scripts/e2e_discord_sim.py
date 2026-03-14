#!/usr/bin/env python3
"""
E2E Discord Message Simulator

Simulates the exact packet flow that discord_interface.py uses:
  1. Constructs a CognitionPacket identical to what Discord would create
  2. Runs the security scan (if middleware is importable)
  3. POSTs to gaia-core /process_packet as NDJSON stream
  4. Processes the response exactly as discord_interface.py does
  5. Prints the final "Discord message" that would be sent to the channel

Usage:
    python scripts/e2e_discord_sim.py
    python scripts/e2e_discord_sim.py --message "Hello GAIA" --user "Azrael"
    python scripts/e2e_discord_sim.py --core-url http://gaia-core:6415
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime

import httpx

# ---------------------------------------------------------------------------
# Packet construction — mirrors discord_interface.py _handle_message()
# ---------------------------------------------------------------------------

def build_discord_packet(
    content: str,
    author_name: str = "Rupert",
    user_id: str = "100000000000000001",
    channel_id: str = "200000000000000001",
    message_id: str = "",
    is_dm: bool = True,
) -> dict:
    """Build the exact CognitionPacket dict that discord_interface.py creates."""
    if not message_id:
        message_id = str(uuid.uuid4().int)[:18]

    packet_id = str(uuid.uuid4())
    session_id = f"discord_dm_{user_id}" if is_dm else f"discord_channel_{channel_id}"
    current_time = datetime.now().isoformat()

    packet = {
        "version": "0.2",
        "header": {
            "datetime": current_time,
            "session_id": session_id,
            "packet_id": packet_id,
            "sub_id": "0",
            "persona": {
                "identity_id": "default_user",
                "persona_id": "default_persona",
                "role": "Default",
                "tone_hint": "conversational",
            },
            "origin": "user",
            "routing": {
                "target_engine": "Prime",
                "priority": 5,
            },
            "model": {
                "name": "default_model",
                "provider": "default_provider",
                "context_window_tokens": 8192,
            },
            "output_routing": {
                "primary": {
                    "destination": "discord",
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "reply_to_message_id": message_id,
                    "metadata": {"is_dm": is_dm, "author_name": author_name},
                },
                "source_destination": "discord",
                "addressed_to_gaia": True,
            },
            "operational_status": {"status": "initialized"},
        },
        "intent": {
            "user_intent": "chat",
            "system_task": "GenerateDraft",
            "confidence": 0.0,
        },
        "context": {
            "session_history_ref": {
                "type": "discord_channel",
                "value": session_id,
            },
            "cheatsheets": [],
            "constraints": {
                "max_tokens": 2048,
                "time_budget_ms": 30000,
                "safety_mode": "strict",
            },
        },
        "content": {
            "original_prompt": content,
            "data_fields": [
                {"key": "user_message", "value": content, "type": "text"},
            ],
        },
        "reasoning": {},
        "response": {
            "candidate": "",
            "confidence": 0.0,
            "stream_proposal": False,
        },
        "governance": {
            "safety": {"execution_allowed": False, "dry_run": True},
        },
        "metrics": {
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "latency_ms": 0,
        },
        "status": {
            "finalized": False,
            "state": "initialized",
            "next_steps": [],
        },
        "tool_routing": {},
    }
    return packet, packet_id, session_id


# ---------------------------------------------------------------------------
# Discord message splitting — mirrors discord_interface._split_message()
# ---------------------------------------------------------------------------

def split_message(content: str, max_length: int = 2000) -> list:
    """Split message into chunks respecting Discord's 2000-char limit."""
    if len(content) <= max_length:
        return [content]

    messages = []
    remaining = content
    while remaining:
        if len(remaining) <= max_length:
            messages.append(remaining)
            break
        # Try to split at newline
        split_point = remaining[:max_length].rfind("\n")
        if split_point < max_length // 2:
            split_point = max_length
        messages.append(remaining[:split_point])
        remaining = remaining[split_point:].lstrip("\n")
    return messages


# ---------------------------------------------------------------------------
# Response processing — mirrors discord_interface._handle_message() stream
# ---------------------------------------------------------------------------

def process_stream(response_lines: list) -> dict:
    """Process NDJSON lines exactly as discord_interface.py does.

    Returns:
        {
            "discord_messages": [...],   # What Discord would receive
            "reflex_messages": [...],    # Immediate reflex sends
            "final_packet": {...} | None,
            "events": [...],             # Raw event log
            "error": str | None,
        }
    """
    full_response = ""
    reflex_messages = []
    final_packet = None
    events = []
    error = None

    for line in response_lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            events.append(event)
            event_type = event.get("type")

            if event_type == "token":
                val = event.get("value", "")
                if val:
                    if val.startswith("⚡ **[(Reflex)"):
                        # Reflex — sent immediately to Discord
                        reflex_messages.append(val)
                    else:
                        full_response += val

            elif event_type == "flush":
                # In real Discord, this triggers an immediate send of the buffer
                pass  # We'll send full_response at the end

            elif event_type == "packet":
                final_packet = event.get("value", {})

            elif event_type == "error":
                error = event.get("value", "unknown error")

        except json.JSONDecodeError:
            pass

    # Build the list of Discord messages that would actually be sent
    discord_messages = []

    # Reflex messages go first (they're sent immediately upon receipt)
    for msg in reflex_messages:
        discord_messages.extend(split_message(msg))

    # Then the accumulated response
    if full_response.strip():
        discord_messages.extend(split_message(full_response.strip()))

    return {
        "discord_messages": discord_messages,
        "reflex_messages": reflex_messages,
        "final_packet": final_packet,
        "events": events,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E2E Discord message simulator")
    parser.add_argument("--message", default="Can you please tell me what time it is?")
    parser.add_argument("--user", default="Rupert")
    parser.add_argument("--user-id", default="100000000000000001")
    parser.add_argument("--channel-id", default="200000000000000001")
    parser.add_argument("--is-dm", action="store_true", default=True)
    parser.add_argument("--core-url", default="http://localhost:6415")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ── Step 1: Build packet ──────────────────────────────────────────
    print(f"{'═' * 70}")
    print(f"  E2E DISCORD MESSAGE SIMULATOR")
    print(f"{'═' * 70}")
    print(f"  User:      {args.user}")
    print(f"  User ID:   {args.user_id}")
    print(f"  Channel:   {'DM' if args.is_dm else args.channel_id}")
    print(f"  Message:   {args.message!r}")
    print(f"  Core URL:  {args.core_url}")
    print(f"{'─' * 70}")

    packet, packet_id, session_id = build_discord_packet(
        content=args.message,
        author_name=args.user,
        user_id=args.user_id,
        channel_id=args.channel_id,
        is_dm=args.is_dm,
    )

    print(f"\n📦 Packet constructed:")
    print(f"   packet_id:  {packet_id}")
    print(f"   session_id: {session_id}")

    if args.verbose:
        print(f"\n   Full packet JSON:")
        print(json.dumps(packet, indent=2))

    # ── Step 2: Security scan (best-effort) ───────────────────────────
    try:
        from gaia_web.security.middleware import SecurityScanMiddleware
        middleware = SecurityScanMiddleware()
        from gaia_common.protocols.cognition_packet import CognitionPacket
        pkt_obj = CognitionPacket.from_dict(packet)
        pkt_obj, blocked = middleware.scan_packet(pkt_obj)
        if blocked:
            print(f"\n🛑 SECURITY SCAN BLOCKED THIS MESSAGE")
            print(f"   The message would not reach gaia-core.")
            sys.exit(1)
        print(f"\n🔒 Security scan: PASSED")
    except ImportError:
        print(f"\n🔒 Security scan: SKIPPED (middleware not importable on host)")

    # ── Step 3: POST to gaia-core ─────────────────────────────────────
    print(f"\n📡 Sending to gaia-core {args.core_url}/process_packet ...")
    start_time = time.time()

    response_lines = []
    token_count = 0

    try:
        with httpx.Client(timeout=args.timeout) as client:
            with client.stream(
                "POST",
                f"{args.core_url}/process_packet",
                json=packet,
                timeout=args.timeout,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    print(f"\n❌ Core returned HTTP {response.status_code}")
                    print(f"   Body: {response.text[:500]}")
                    sys.exit(1)

                print(f"   HTTP 200 — streaming NDJSON response...\n")

                # Process stream in real-time (like Discord would)
                if args.verbose:
                    print(f"   {'─' * 50}")
                    print(f"   NDJSON Events:")

                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    response_lines.append(line)
                    try:
                        evt = json.loads(line)
                        if args.verbose:
                            etype = evt.get("type", "?")
                            val = str(evt.get("value", ""))[:80]
                            print(f"   [{etype:>7}] {val}")
                        if evt.get("type") == "token":
                            token_count += 1
                    except json.JSONDecodeError:
                        pass

    except httpx.TimeoutException:
        print(f"\n❌ Request timed out after {args.timeout}s")
        sys.exit(1)
    except httpx.ConnectError as e:
        print(f"\n❌ Cannot connect to {args.core_url}: {e}")
        print(f"   Is gaia-core running? Try: docker compose ps gaia-core")
        sys.exit(1)

    elapsed = time.time() - start_time

    # ── Step 4: Process response as Discord would ─────────────────────
    result = process_stream(response_lines)

    print(f"\n{'═' * 70}")
    print(f"  DISCORD OUTPUT (what {args.user} would see)")
    print(f"{'═' * 70}")

    if result["error"]:
        print(f"\n❌ Error from core: {result['error']}")
    elif not result["discord_messages"]:
        print(f"\n⚠️  No response generated (empty output)")
    else:
        for i, msg in enumerate(result["discord_messages"]):
            if len(result["discord_messages"]) > 1:
                print(f"\n  ── Message {i+1}/{len(result['discord_messages'])} ──")
            print()
            # Indent each line for readability
            for line in msg.split("\n"):
                print(f"  {line}")

    # ── Step 5: Summary ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  SUMMARY")
    print(f"{'─' * 70}")
    print(f"  Elapsed:          {elapsed:.2f}s")
    print(f"  NDJSON events:    {len(result['events'])}")
    print(f"  Token events:     {token_count}")
    print(f"  Reflex messages:  {len(result['reflex_messages'])}")
    print(f"  Discord messages: {len(result['discord_messages'])}")
    total_chars = sum(len(m) for m in result["discord_messages"])
    print(f"  Total chars:      {total_chars}")
    if result["final_packet"]:
        pkt = result["final_packet"]
        model = pkt.get("header", {}).get("model", {}).get("name", "?")
        state = pkt.get("status", {}).get("state", "?")
        tok = pkt.get("metrics", {}).get("token_usage", {})
        print(f"  Model used:       {model}")
        print(f"  Final state:      {state}")
        if tok.get("total_tokens"):
            print(f"  Token usage:      {tok['prompt_tokens']}p + {tok['completion_tokens']}c = {tok['total_tokens']}t")
    else:
        print(f"  Final packet:     (not received)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
