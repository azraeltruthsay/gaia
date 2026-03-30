#!/usr/bin/env python3
"""
Cognitive Handoff Test — automatic GPU tier rotation.

Tests that the orchestrator can transparently load/unload tiers
when you just specify which tier to ask. No manual model management.

Usage:
    python scripts/cognitive_handoff_test.py                    # Full test
    python scripts/cognitive_handoff_test.py --tiers core prime # Specific tiers
    python scripts/cognitive_handoff_test.py --no-sae           # Skip SAE recording
    python scripts/cognitive_handoff_test.py --dry-run          # Check connectivity only

Requires:
    - Orchestrator running on localhost:6410 with tier router endpoints
    - Managed engines running on each tier (gaia-core, gaia-prime, gaia-nano)
"""

import argparse
import json
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

ORCHESTRATOR = "http://localhost:6410"

# ── Test Questions ────────────────────────────────────────────────────────────

TIER_QUESTIONS = {
    "nano": {
        "description": "Reflex tier — 0.8B, fast triage, sub-second responses",
        "questions": [
            {
                "label": "identity_basic",
                "messages": [
                    {"role": "system", "content": "You are GAIA, a sovereign AI."},
                    {"role": "user", "content": "Who are you?"},
                ],
                "max_tokens": 64,
                "expect_contains": ["GAIA"],
            },
            {
                "label": "triage_simple",
                "messages": [
                    {"role": "system", "content": "Classify as SIMPLE or COMPLEX."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "max_tokens": 16,
                "expect_contains": ["SIMPLE"],
            },
            {
                "label": "reflex_speed",
                "messages": [
                    {"role": "user", "content": "Say hello."},
                ],
                "max_tokens": 32,
            },
        ],
    },
    "core": {
        "description": "Operator tier — 2B, intent detection, tool routing",
        "questions": [
            {
                "label": "identity_core",
                "messages": [
                    {"role": "system", "content": "You are GAIA, a sovereign AI created by Azrael."},
                    {"role": "user", "content": "Who are you and what is your purpose?"},
                ],
                "max_tokens": 128,
                "expect_contains": ["GAIA"],
            },
            {
                "label": "architecture_knowledge",
                "messages": [
                    {"role": "system", "content": "You are GAIA."},
                    {"role": "user", "content": "Name three of your services."},
                ],
                "max_tokens": 128,
            },
            {
                "label": "epistemic_honesty",
                "messages": [
                    {"role": "system", "content": "You are GAIA. You value epistemic honesty."},
                    {"role": "user", "content": "What is the population of Mars?"},
                ],
                "max_tokens": 128,
                "expect_contains": None,  # Should express uncertainty
            },
        ],
    },
    "prime": {
        "description": "Thinker tier — 8B, complex reasoning, code, analysis",
        "questions": [
            {
                "label": "identity_prime",
                "messages": [
                    {"role": "system", "content": "You are GAIA, a sovereign AI created by Azrael. You are the Thinker tier."},
                    {"role": "user", "content": "Who are you and what makes you different from other AI systems?"},
                ],
                "max_tokens": 256,
                "expect_contains": ["GAIA"],
            },
            {
                "label": "reasoning_complex",
                "messages": [
                    {"role": "system", "content": "You are GAIA."},
                    {"role": "user", "content": "Explain why subprocess isolation for GPU management is better than in-process model loading."},
                ],
                "max_tokens": 256,
            },
            {
                "label": "code_generation",
                "messages": [
                    {"role": "system", "content": "You are GAIA."},
                    {"role": "user", "content": "Write a Python function that checks if a number is prime. Keep it concise."},
                ],
                "max_tokens": 200,
            },
        ],
    },
}

# ── HTTP Helpers ──────────────────────────────────────────────────────────────

def http_post(url: str, data: dict, timeout: int = 300) -> dict:
    body = json.dumps(data).encode()
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_get(url: str, timeout: int = 10) -> dict:
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ── Test Runner ───────────────────────────────────────────────────────────────

def check_connectivity():
    """Verify orchestrator and tier status endpoints are reachable."""
    print("=" * 72)
    print("CONNECTIVITY CHECK")
    print("=" * 72)

    try:
        health = http_get(f"{ORCHESTRATOR}/health")
        print(f"  Orchestrator: OK ({health.get('status', '?')})")
    except Exception as e:
        print(f"  Orchestrator: UNREACHABLE ({e})")
        return False

    try:
        tiers = http_get(f"{ORCHESTRATOR}/tier/status")
        for tier, info in tiers.items():
            status = "LOADED" if info.get("model_loaded") else info.get("mode", "?")
            managed = " [managed]" if info.get("managed") else ""
            print(f"  {tier:6s}: {status}{managed} ({info.get('endpoint', '?')})")
    except Exception as e:
        print(f"  Tier status: UNAVAILABLE ({e})")
        print("  (Tier router endpoints may not be deployed yet)")
        return False

    print()
    return True


def run_tier_test(tier: str, questions: list, description: str,
                  record_sae: bool = True) -> dict:
    """Run cognitive questions against a tier with automatic handoff."""
    print(f"\n{'=' * 72}")
    print(f"TIER: {tier.upper()} — {description}")
    print(f"{'=' * 72}")

    results = {
        "tier": tier,
        "questions": [],
        "handoff_time_s": None,
        "sae_recorded": False,
        "pass_count": 0,
        "fail_count": 0,
        "error_count": 0,
    }

    # Step 1: Ensure tier is loaded (this is the handoff test)
    print(f"\n  [handoff] Ensuring {tier} is loaded on GPU...")
    t0 = time.time()
    try:
        ensure = http_post(f"{ORCHESTRATOR}/tier/ensure", {"tier": tier})
        handoff_time = time.time() - t0
        results["handoff_time_s"] = round(handoff_time, 1)
        action = ensure.get("action", "?")
        unloaded = ensure.get("unloaded", [])
        print(f"  [handoff] {action} in {handoff_time:.1f}s", end="")
        if unloaded:
            print(f" (unloaded: {', '.join(unloaded)})", end="")
        print()
    except Exception as e:
        print(f"  [handoff] FAILED: {e}")
        results["error_count"] = len(questions)
        return results

    # Step 2: SAE recording (background, non-blocking)
    if record_sae:
        print(f"  [sae] Recording activations for {tier}...")
        try:
            sae = http_post(
                f"{ORCHESTRATOR}/tier/sae-record?tier={tier}&tag=handoff_test",
                {}, timeout=30)
            if sae.get("ok") or sae.get("status") == "recording_started":
                results["sae_recorded"] = True
                print(f"  [sae] Recording started ({sae.get('prompts', '?')} prompts, layers: {sae.get('layers', '?')})")
            else:
                print(f"  [sae] Skipped: {sae.get('error', 'unknown')}")
        except Exception as e:
            print(f"  [sae] Skipped: {e}")

    # Step 3: Run questions
    for q in questions:
        label = q["label"]
        messages = q["messages"]
        max_tokens = q.get("max_tokens", 128)
        expect = q.get("expect_contains")

        print(f"\n  [{label}]")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "?")
        print(f"    Q: {user_msg[:80]}")

        t0 = time.time()
        try:
            resp = http_post(f"{ORCHESTRATOR}/tier/infer", {
                "tier": tier,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            })
            latency = time.time() - t0

            if "error" in resp and "choices" not in resp:
                print(f"    ERROR: {resp['error']}")
                results["error_count"] += 1
                results["questions"].append({
                    "label": label, "status": "ERROR",
                    "error": resp["error"], "latency_s": round(latency, 2),
                })
                continue

            # Extract response text
            text = ""
            if "choices" in resp:
                text = resp["choices"][0].get("message", {}).get("content", "")
            elif "content" in resp:
                text = resp["content"]
            elif "text" in resp:
                text = resp["text"]

            # Truncate for display
            display_text = text.strip().replace("\n", " ")[:200]
            print(f"    A: {display_text}")
            print(f"    ({latency:.2f}s, {len(text)} chars)")

            # Check expectations
            passed = True
            if expect:
                for keyword in expect:
                    if keyword.lower() not in text.lower():
                        print(f"    WARN: expected '{keyword}' not found in response")
                        passed = False

            status = "PASS" if passed else "FAIL"
            print(f"    [{status}]")

            if passed:
                results["pass_count"] += 1
            else:
                results["fail_count"] += 1

            results["questions"].append({
                "label": label,
                "status": status,
                "response_preview": display_text,
                "response_length": len(text),
                "latency_s": round(latency, 2),
            })

        except Exception as e:
            latency = time.time() - t0
            print(f"    ERROR: {e} ({latency:.1f}s)")
            results["error_count"] += 1
            results["questions"].append({
                "label": label, "status": "ERROR",
                "error": str(e), "latency_s": round(latency, 2),
            })

    return results


def print_summary(all_results: list, total_time: float):
    """Print the final test summary."""
    print(f"\n{'=' * 72}")
    print("HANDOFF TEST SUMMARY")
    print(f"{'=' * 72}")

    total_pass = sum(r["pass_count"] for r in all_results)
    total_fail = sum(r["fail_count"] for r in all_results)
    total_error = sum(r["error_count"] for r in all_results)
    total_tests = total_pass + total_fail + total_error

    for r in all_results:
        tier = r["tier"].upper()
        handoff = f"{r['handoff_time_s']:.1f}s" if r["handoff_time_s"] is not None else "FAILED"
        sae = "yes" if r["sae_recorded"] else "no"
        p, f, e = r["pass_count"], r["fail_count"], r["error_count"]
        status = "PASS" if f == 0 and e == 0 else "FAIL"
        print(f"  {tier:6s}: {p}/{p+f+e} passed | handoff: {handoff} | SAE: {sae} | [{status}]")

    print(f"\n  Total: {total_pass}/{total_tests} passed, {total_fail} failed, {total_error} errors")
    print(f"  Total time: {total_time:.1f}s")

    # Handoff analysis
    handoff_times = [r["handoff_time_s"] for r in all_results if r["handoff_time_s"] is not None]
    if len(handoff_times) > 1:
        # First handoff might be "already_loaded", subsequent ones are real handoffs
        real_handoffs = [t for t, r in zip(handoff_times, all_results)
                        if r.get("questions") and any(
                            q.get("status") != "ERROR"
                            for q in r.get("questions", []))]
        if real_handoffs:
            print(f"\n  Handoff times: {' → '.join(f'{t:.1f}s' for t in handoff_times)}")
            print(f"  Avg handoff: {sum(handoff_times)/len(handoff_times):.1f}s")

    overall = "PASS" if total_fail == 0 and total_error == 0 else "FAIL"
    print(f"\n  Overall: [{overall}]")
    print(f"{'=' * 72}")

    return overall == "PASS"


def main():
    global ORCHESTRATOR

    parser = argparse.ArgumentParser(description="Cognitive Handoff Test")
    parser.add_argument("--tiers", nargs="+", default=["nano", "core", "prime"],
                        help="Tiers to test (default: nano core prime)")
    parser.add_argument("--no-sae", action="store_true",
                        help="Skip SAE atlas recording")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check connectivity only, don't run tests")
    parser.add_argument("--orchestrator", default=ORCHESTRATOR,
                        help="Orchestrator URL")
    parser.add_argument("--output", default="",
                        help="Save results JSON to file")
    args = parser.parse_args()

    ORCHESTRATOR = args.orchestrator

    print("\n  GAIA Cognitive Handoff Test")
    print(f"  Orchestrator: {ORCHESTRATOR}")
    print(f"  Tiers: {', '.join(args.tiers)}")
    print(f"  SAE Recording: {'no' if args.no_sae else 'yes'}")
    print()

    if not check_connectivity():
        print("Connectivity check failed. Ensure services are running.")
        sys.exit(1)

    if args.dry_run:
        print("Dry run complete — connectivity OK.")
        sys.exit(0)

    # Run tests in tier order
    all_results = []
    total_start = time.time()

    for tier in args.tiers:
        if tier not in TIER_QUESTIONS:
            print(f"  Unknown tier: {tier} (skipping)")
            continue

        tier_config = TIER_QUESTIONS[tier]
        result = run_tier_test(
            tier=tier,
            questions=tier_config["questions"],
            description=tier_config["description"],
            record_sae=not args.no_sae,
        )
        all_results.append(result)

    total_time = time.time() - total_start

    # Summary
    passed = print_summary(all_results, total_time)

    # Save results
    if args.output:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tiers_tested": args.tiers,
            "total_time_s": round(total_time, 1),
            "results": all_results,
            "overall": "PASS" if passed else "FAIL",
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
