#!/usr/bin/env python3
"""
test_prime_cogpacket.py — Send a test inference to gaia-prime using a
fully-formed CognitionPacket, bypassing gaia-core's pipeline.

Usage:
    # From host (prime must be reachable):
    python scripts/test_prime_cogpacket.py

    # Custom endpoint:
    PRIME_ENDPOINT=http://localhost:7777 python scripts/test_prime_cogpacket.py

    # Custom prompt:
    python scripts/test_prime_cogpacket.py "Explain quantum entanglement in one paragraph."

    # Stream mode:
    python scripts/test_prime_cogpacket.py --stream "Tell me a joke."
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Allow running from repo root or from the scripts/ directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_ROOT = os.path.dirname(_SCRIPT_DIR)           # candidates/gaia-core
_CANDIDATES = os.path.dirname(_CORE_ROOT)            # candidates/
sys.path.insert(0, _CORE_ROOT)
sys.path.insert(0, os.path.join(_CANDIDATES, "gaia-common"))

from gaia_core.cognition.cognition_packet import (
    CognitionPacket,
    Header, Persona, PersonaRole, Routing, TargetEngine, Model,
    Intent, SystemTask,
    Context, SessionHistoryRef, Cheatsheet, Constraints,
    Content,
    Reasoning,
    Response,
    Governance, Safety, Signatures, Audit, Privacy,
    Metrics, TokenUsage,
    Status, PacketState,
    Origin,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PRIME_ENDPOINT = os.getenv("PRIME_ENDPOINT", "http://gaia-prime-candidate:7777")
PRIME_MODEL = os.getenv("PRIME_MODEL", "/models/Claude")
DEFAULT_PROMPT = "You are GAIA, an advanced AI assistant. Respond briefly: What is your status?"


def build_test_packet(prompt: str, temperature: float = 0.7) -> CognitionPacket:
    """Build a minimal-but-complete CognitionPacket for a direct prime test."""

    now = datetime.now(timezone.utc).isoformat()
    session_id = "test-session-" + uuid.uuid4().hex[:8]
    packet_id = "pkt-test-" + uuid.uuid4().hex[:12]

    packet = CognitionPacket(
        version="0.3.0-test",
        schema_id="gaia-cogpacket-v0.3",

        header=Header(
            datetime=now,
            session_id=session_id,
            packet_id=packet_id,
            sub_id="sub-0",
            persona=Persona(
                identity_id="gaia-prime-test",
                persona_id="Default",
                role=PersonaRole.DEFAULT,
                tone_hint="neutral",
            ),
            origin=Origin.SYSTEM,
            routing=Routing(
                target_engine=TargetEngine.PRIME,
                priority=5,
            ),
            model=Model(
                name=PRIME_MODEL,
                provider="vllm_remote",
                context_window_tokens=8192,
                max_output_tokens=1024,
                response_buffer_tokens=256,
                temperature=temperature,
                top_p=0.95,
            ),
        ),

        intent=Intent(
            user_intent=prompt,
            system_task=SystemTask.STREAM,
            confidence=1.0,
            tags=["test", "direct-prime"],
        ),

        context=Context(
            session_history_ref=SessionHistoryRef(
                type="none",
                value="direct-test-no-history",
            ),
            cheatsheets=[],
            constraints=Constraints(
                max_tokens=1024,
                time_budget_ms=30000,
                safety_mode="permissive",
            ),
        ),

        content=Content(
            original_prompt=prompt,
        ),

        reasoning=Reasoning(),

        response=Response(
            candidate="",
            confidence=0.0,
            stream_proposal=True,
        ),

        governance=Governance(
            safety=Safety(
                execution_allowed=True,
                dry_run=False,
            ),
        ),

        metrics=Metrics(
            token_usage=TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            latency_ms=0,
        ),

        status=Status(
            finalized=False,
            state=PacketState.INITIALIZED,
            next_steps=["send_to_prime"],
        ),
    )

    packet.compute_hashes()
    return packet


def packet_to_chat_payload(packet: CognitionPacket) -> dict:
    """
    Extract the OpenAI-compatible chat completion payload from a CognitionPacket.
    This mirrors what gaia-core's prompt_builder + VLLMRemoteModel do internally.
    """
    model = packet.header.model

    system_prompt = (
        "You are GAIA, an advanced sovereign AI system. "
        "You are running on local hardware. "
        "Respond thoughtfully and concisely."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": packet.content.original_prompt},
    ]

    payload = {
        "model": model.name,
        "messages": messages,
        "max_tokens": model.max_output_tokens or 1024,
        "temperature": model.temperature or 0.7,
        "top_p": model.top_p or 0.95,
    }

    if model.stop:
        payload["stop"] = model.stop

    return payload


def send_to_prime(payload: dict, stream: bool = False) -> dict:
    """Send the payload to prime and return the response."""
    import requests

    endpoint = PRIME_ENDPOINT.rstrip("/")
    url = f"{endpoint}/v1/chat/completions"

    if stream:
        payload["stream"] = True

    print(f"\n--- Sending to {url} ---")
    print(f"    Model:       {payload['model']}")
    print(f"    Temperature: {payload['temperature']}")
    print(f"    Max tokens:  {payload['max_tokens']}")
    print(f"    Stream:      {stream}")
    print()

    start = time.time()

    if stream:
        with requests.post(url, json=payload, timeout=120, stream=True) as r:
            r.raise_for_status()
            full_content = []
            sys.stdout.write("GAIA> ")
            sys.stdout.flush()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_content.append(content)
            print()  # newline after stream
            duration = time.time() - start
            return {
                "content": "".join(full_content),
                "duration_s": round(duration, 2),
                "streamed": True,
            }
    else:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        duration = time.time() - start
        resp = r.json()
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        return {
            "content": content,
            "duration_s": round(duration, 2),
            "usage": usage,
            "streamed": False,
        }


def fill_response_into_packet(packet: CognitionPacket, result: dict) -> CognitionPacket:
    """Fill the prime response back into the packet, completing the lifecycle."""
    packet.response.candidate = result["content"]
    packet.response.confidence = 0.85  # placeholder

    packet.metrics.latency_ms = int(result["duration_s"] * 1000)
    if "usage" in result:
        usage = result["usage"]
        packet.metrics.token_usage.prompt_tokens = usage.get("prompt_tokens", 0)
        packet.metrics.token_usage.completion_tokens = usage.get("completion_tokens", 0)
        packet.metrics.token_usage.total_tokens = usage.get("total_tokens", 0)

    packet.status.finalized = True
    packet.status.state = PacketState.COMPLETED
    packet.status.next_steps = []

    return packet


def main():
    parser = argparse.ArgumentParser(
        description="Send a test inference to gaia-prime with a full CognitionPacket"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Prompt to send (default: status check)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Use streaming mode",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--dump-packet", action="store_true",
        help="Dump the full CognitionPacket JSON (before and after)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the packet and payload but don't send to prime",
    )
    args = parser.parse_args()

    # 1. Build the CognitionPacket
    print("=== Building CognitionPacket ===")
    packet = build_test_packet(args.prompt, temperature=args.temperature)
    print(f"    Packet ID:  {packet.header.packet_id}")
    print(f"    Session:    {packet.header.session_id}")
    print(f"    Target:     {packet.header.routing.target_engine.value}")
    print(f"    Prompt:     {packet.content.original_prompt[:80]}...")

    # 2. Extract the OpenAI payload
    payload = packet_to_chat_payload(packet)

    if args.dump_packet:
        print("\n=== CognitionPacket (pre-inference) ===")
        print(packet.to_json())

    if args.dry_run:
        print("\n=== Payload (dry run — not sending) ===")
        print(json.dumps(payload, indent=2))
        return

    # 3. Send to prime
    try:
        result = send_to_prime(payload, stream=args.stream)
    except Exception as e:
        print(f"\n[ERROR] Failed to reach prime: {e}")
        print("  Hint: Is gaia-prime running? Try: docker ps | grep prime")
        print(f"  Endpoint: {PRIME_ENDPOINT}")
        sys.exit(1)

    # 4. Fill response back into packet
    packet = fill_response_into_packet(packet, result)

    # 5. Print results
    print(f"\n=== Response ===")
    if not args.stream:
        print(f"GAIA> {result['content']}")
    print(f"\n    Duration:   {result['duration_s']}s")
    if "usage" in result:
        u = result["usage"]
        print(f"    Tokens:     {u.get('prompt_tokens', '?')}p / {u.get('completion_tokens', '?')}c / {u.get('total_tokens', '?')}t")
    print(f"    State:      {packet.status.state.value}")

    if args.dump_packet:
        print("\n=== CognitionPacket (post-inference) ===")
        print(packet.to_json())

    print("\n[OK] Direct prime test complete.")


if __name__ == "__main__":
    main()
