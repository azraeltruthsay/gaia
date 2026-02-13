#!/usr/bin/env python3
"""
smoke_test_cognitive.py — Automated cognitive pipeline smoke tests.

Sends a sequence of test prompts through gaia-core's /process_packet endpoint
(the same path Discord/web use), validates responses, and prints a summary.

No container-internal dependencies required — packets are built as plain dicts,
matching the CognitionPacket schema that /process_packet deserializes server-side.

Usage:
    # From host (gaia-core-candidate must be running on port 6416):
    python scripts/smoke_test_cognitive.py

    # Custom endpoint:
    CORE_ENDPOINT=http://localhost:6415 python scripts/smoke_test_cognitive.py

    # Run specific test(s) by number:
    python scripts/smoke_test_cognitive.py --only 1,3

    # Verbose mode (print full responses):
    python scripts/smoke_test_cognitive.py -v
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CORE_ENDPOINT = os.getenv("CORE_ENDPOINT", "http://localhost:6416")

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Packet builder (plain dict — no dataclasses_json needed)
# ---------------------------------------------------------------------------
def build_packet(prompt: str, session_id: str) -> dict:
    """Build a CognitionPacket dict for the /process_packet endpoint."""
    now = datetime.now(timezone.utc).isoformat()
    packet_id = "pkt-smoke-" + uuid.uuid4().hex[:12]

    header = {
        "datetime": now,
        "session_id": session_id,
        "packet_id": packet_id,
        "sub_id": "sub-0",
        "persona": {
            "identity_id": "gaia-smoke-test",
            "persona_id": "Default",
            "role": "Default",
            "tone_hint": "neutral",
            "traits": {},
        },
        "origin": "user",
        "routing": {
            "target_engine": "Prime",
            "allow_parallel": False,
            "priority": 5,
        },
        "model": {
            "name": "/models/Claude",
            "provider": "vllm_remote",
            "context_window_tokens": 8192,
            "max_output_tokens": 2048,
            "response_buffer_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "stop": [],
            "tool_permissions": [],
            "allow_tools": True,
        },
        "output_routing": {
            "primary": {
                "destination": "web",
                "metadata": {},
            },
            "secondary": [],
            "suppress_echo": False,
            "addressed_to_gaia": True,
            "source_destination": "web",
        },
        "lineage": [],
    }

    # Compute hashes for governance signatures
    header_hash = hashlib.sha256(
        json.dumps(header, sort_keys=True).encode()
    ).hexdigest()

    content = {
        "original_prompt": prompt,
        "data_fields": [],
        "attachments": [],
    }

    content_hash = hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()
    ).hexdigest()

    packet = {
        "version": "0.3.0-smoke",
        "schema_id": "gaia-cogpacket-v0.3",
        "header": header,
        "intent": {
            "user_intent": prompt,
            "system_task": "Stream",
            "confidence": 1.0,
            "tags": ["smoke-test"],
        },
        "context": {
            "session_history_ref": {
                "type": "session_id",
                "value": session_id,
            },
            "cheatsheets": [],
            "constraints": {
                "max_tokens": 2048,
                "time_budget_ms": 300000,
                "safety_mode": "permissive",
                "policies": [],
            },
            "relevant_history_snippet": [],
        },
        "content": content,
        "reasoning": {
            "reflection_log": [],
            "sketchpad": [],
            "evaluations": [],
        },
        "response": {
            "candidate": "",
            "confidence": 0.0,
            "stream_proposal": False,
            "tool_calls": [],
            "sidecar_actions": [],
        },
        "governance": {
            "safety": {
                "execution_allowed": True,
                "dry_run": False,
            },
            "signatures": {
                "header_hash": header_hash,
                "content_hash": content_hash,
            },
            "audit": {
                "reviewers": [],
            },
            "privacy": {},
        },
        "metrics": {
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "latency_ms": 0,
            "errors": [],
        },
        "status": {
            "finalized": False,
            "state": "initialized",
            "next_steps": ["process"],
            "observer_trace": [],
        },
    }

    return packet


# ---------------------------------------------------------------------------
# Send packet to gaia-core (stdlib only — no requests needed)
# ---------------------------------------------------------------------------
def send_packet(packet: dict, endpoint: str, timeout: int = 300) -> dict:
    """POST a CognitionPacket dict to /process_packet and return the response."""
    url = f"{endpoint.rstrip('/')}/process_packet"
    data = json.dumps(packet).encode("utf-8")

    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


# ---------------------------------------------------------------------------
# Validator type
# ---------------------------------------------------------------------------
Validator = Callable[[str], Tuple[bool, str]]


def v_non_empty(r: str) -> Tuple[bool, str]:
    return (len(r.strip()) > 0, "non-empty response")


def v_min_length(n: int) -> Validator:
    def check(r: str) -> Tuple[bool, str]:
        return (len(r) >= n, f"length >= {n} (got {len(r)})")
    return check


def v_contains_any(*terms: str) -> Validator:
    def check(r: str) -> Tuple[bool, str]:
        lower = r.lower()
        found = [t for t in terms if t.lower() in lower]
        if found:
            return (True, f"contains '{found[0]}'")
        return (False, f"missing all of: {', '.join(terms)}")
    return check


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------
class TestCase:
    def __init__(self, num: int, category: str, prompt: str, validators: List[Validator]):
        self.num = num
        self.category = category
        self.prompt = prompt
        self.validators = validators


TEST_CASES = [
    TestCase(
        num=1,
        category="World status",
        prompt="What is the current date and your uptime?",
        validators=[v_non_empty, v_min_length(20)],
    ),
    TestCase(
        num=2,
        category="General knowledge",
        prompt="What is the name of King Arthur's sword?",
        validators=[v_non_empty, v_contains_any("excalibur")],
    ),
    TestCase(
        num=3,
        category="Long recitation",
        prompt="Please recite the first three stanzas of The Raven by Edgar Allan Poe.",
        validators=[
            v_non_empty,
            v_min_length(300),
            v_contains_any("midnight dreary", "midnight", "dreary"),
            v_contains_any("nevermore", "nothing more"),
        ],
    ),
    TestCase(
        num=4,
        category="Web search",
        prompt="Use web search to find who won the most recent Super Bowl and summarize the result.",
        validators=[v_non_empty, v_min_length(50)],
    ),
    TestCase(
        num=5,
        category="Knowledge save",
        prompt="Please save the following to my knowledge base: The Raven was written by Edgar Allan Poe in 1845.",
        validators=[v_non_empty, v_contains_any("saved", "indexed", "knowledge", "stored", "recorded")],
    ),
    TestCase(
        num=6,
        category="Local retrieval",
        prompt="What do you know about The Raven from your knowledge base?",
        validators=[v_non_empty, v_contains_any("poe", "1845", "raven")],
    ),
]


# ---------------------------------------------------------------------------
# Run a single test
# ---------------------------------------------------------------------------
def run_test(
    test: TestCase,
    session_id: str,
    endpoint: str,
    timeout: int,
    verbose: bool,
) -> Tuple[bool, str, float]:
    """Run a single test case. Returns (passed, details, duration_seconds)."""
    packet = build_packet(test.prompt, session_id)

    start = time.time()
    try:
        result = send_packet(packet, endpoint, timeout=timeout)
    except Exception as e:
        duration = time.time() - start
        return (False, f"REQUEST FAILED: {e}", duration)
    duration = time.time() - start

    response_text = result.get("response", {}).get("candidate", "")

    if verbose:
        print(f"\n{DIM}--- Response (test {test.num}) ---{RESET}")
        display = response_text if len(response_text) <= 2000 else response_text[:2000] + f"\n... ({len(response_text)} chars total)"
        print(display)
        print(f"{DIM}--- End response ---{RESET}\n")

    # Run validators
    failures = []
    for validator in test.validators:
        passed, detail = validator(response_text)
        if not passed:
            failures.append(detail)

    if failures:
        return (False, "; ".join(failures), duration)

    return (True, "all checks passed", duration)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cognitive pipeline smoke tests for gaia-core"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated test numbers to run (e.g., 1,3,5)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print full response text for each test",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=f"Override CORE_ENDPOINT (default: {CORE_ENDPOINT})",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Override auto-generated session ID",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    endpoint = args.endpoint or CORE_ENDPOINT
    session_id = args.session or f"smoke-test-{uuid.uuid4().hex[:8]}"

    # Filter tests
    tests = TEST_CASES
    if args.only:
        selected = {int(n.strip()) for n in args.only.split(",")}
        tests = [t for t in tests if t.num in selected]
        if not tests:
            print(f"{RED}No tests matched --only {args.only}{RESET}")
            sys.exit(1)

    print(f"{BOLD}=== GAIA Cognitive Smoke Tests ==={RESET}")
    print(f"    Endpoint:   {endpoint}")
    print(f"    Session:    {session_id}")
    print(f"    Tests:      {len(tests)}")
    print(f"    Timeout:    {args.timeout}s per request")
    print()

    # Run tests sequentially (shared session — history accumulates)
    results: List[Tuple[TestCase, bool, str, float]] = []

    for test in tests:
        label = f"[{test.num}] {test.category}"
        print(f"{CYAN}{BOLD}{label}{RESET}: {test.prompt}")
        sys.stdout.flush()

        passed, details, duration = run_test(
            test, session_id, endpoint, args.timeout, args.verbose
        )
        results.append((test, passed, details, duration))

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"    {status}  ({duration:.1f}s) — {details}")
        print()

    # Summary
    total = len(results)
    passed_count = sum(1 for _, p, _, _ in results if p)
    failed_count = total - passed_count

    print(f"{BOLD}=== Summary ==={RESET}")
    print(f"    Total:   {total}")
    print(f"    Passed:  {GREEN}{passed_count}{RESET}")
    if failed_count:
        print(f"    Failed:  {RED}{failed_count}{RESET}")
        print()
        print(f"{RED}Failed tests:{RESET}")
        for test, passed, details, duration in results:
            if not passed:
                print(f"    [{test.num}] {test.category}: {details}")
    else:
        print(f"\n    {GREEN}{BOLD}All tests passed!{RESET}")

    print()
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
