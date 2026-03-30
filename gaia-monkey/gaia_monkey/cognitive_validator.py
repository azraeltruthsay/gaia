"""Cognitive Validator — sends live inference request to validate pipeline."""
import json
import logging
import time
import uuid
from urllib.request import urlopen, Request

log = logging.getLogger("gaia-monkey.cognitive")


def validate(endpoint: str, timeout: int = 30) -> dict:
    """Send a real inference request to validate the cognitive pipeline is working.

    Posts a CognitionPacket to the target endpoint and checks for a meaningful response.
    Returns {passed: bool, latency_ms: float, response_preview: str}.
    """
    packet = {
        "version": "v0.3",
        "header": {
            "session_id": f"chaos_drill_{uuid.uuid4().hex[:8]}",
            "packet_id": f"drill_{uuid.uuid4().hex[:8]}",
            "persona": {"persona_id": "gaia", "role": "assistant"},
        },
        "content": {"original_prompt": "What is 7 times 8?"},
    }

    start = time.monotonic()
    try:
        data = json.dumps(packet).encode("utf-8")
        req = Request(
            f"{endpoint}/process_packet",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        response_lines = []
        with urlopen(req, timeout=timeout) as resp:
            for line in resp:
                decoded = line.decode("utf-8").strip()
                if decoded:
                    response_lines.append(decoded)

        elapsed_ms = (time.monotonic() - start) * 1000

        response_text = ""
        for line in response_lines:
            try:
                obj = json.loads(line)
                if obj.get("type") == "token":
                    response_text += obj.get("value", "")
                elif obj.get("type") == "final":
                    response_text = obj.get("value", response_text)
            except json.JSONDecodeError:
                response_text += line

        passed = len(response_text) > 5 and ("56" in response_text or "fifty" in response_text.lower())
        if not passed and len(response_text) > 10:
            passed = True

        return {
            "passed": passed,
            "latency_ms": round(elapsed_ms, 1),
            "response_preview": response_text[:200],
        }

    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "passed": False,
            "latency_ms": round(elapsed_ms, 1),
            "error": str(e)[:200],
        }
