#!/usr/bin/env python3
"""
Sovereign Capability Test runner.

Evaluates a model against questions that were NEVER in training data.
Reports per-category pass rate and keyword-match scores.

Usage:
    python run_sct.py --target core
    python run_sct.py --target prime
    python run_sct.py --compare core,prime
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

log = logging.getLogger("sct")

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = Path(os.environ.get(
    "SCT_RESULTS_DIR",
    "/gaia/GAIA_Project/shared/doctor/sct_history"
))
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://localhost:6415")


def load_questions() -> list:
    """Load held-out test questions."""
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def query_model(prompt: str, target: str, endpoint: str, timeout: int = 90) -> str:
    """Send prompt via full pipeline (/process_packet). Returns stripped response."""
    packet = {
        "version": "v0.4",
        "header": {
            "session_id": f"sct-{uuid.uuid4().hex[:8]}",
            "packet_id": f"sct-{uuid.uuid4().hex[:12]}",
            "persona": {"persona_id": "gaia", "role": "assistant"},
            "model": {"name": target},
        },
        "content": {"original_prompt": prompt},
        "governance": {"security_scan": {"ran": False, "passed": True, "injection_score": 0.0}},
    }

    auth_key = ""
    try:
        with open("/gaia/GAIA_Project/shared/secrets/gaia_service_key") as f:
            auth_key = f.read().strip()
    except Exception:
        pass

    req = Request(
        f"{endpoint}/process_packet",
        data=json.dumps(packet).encode(),
        headers={"Content-Type": "application/json", "X-Service-Auth": auth_key},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            lines = resp.read().decode().strip().split("\n")
            # Collect token values from NDJSON stream
            tokens = []
            for line in lines:
                try:
                    event = json.loads(line)
                    if event.get("type") == "token":
                        tokens.append(event.get("value", ""))
                except Exception:
                    pass
            return "".join(tokens).strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def score_question(question: dict, response: str) -> dict:
    """Score a response against expected keywords.

    Returns {score, detail, matched_keywords, matched_wrong}.
    """
    resp_lower = response.lower()

    # Check for expected keywords (any match = partial credit, multiple = full)
    expected = question.get("expected_keywords", [])
    matched = [kw for kw in expected if kw.lower() in resp_lower]

    # Check for expected-wrong keywords (hedges or common wrong answers)
    wrong = question.get("expected_wrong_keywords", [])
    matched_wrong = [kw for kw in wrong if kw.lower() in resp_lower]

    # Scoring logic
    if matched_wrong:
        score = "fail"
        detail = f"matched wrong keywords: {matched_wrong}"
    elif len(matched) >= 2:
        score = "pass"
        detail = f"matched {len(matched)}/{len(expected)} expected keywords"
    elif len(matched) == 1:
        score = "partial"
        detail = f"matched 1/{len(expected)} expected keywords"
    elif not response or response.startswith("[ERROR"):
        score = "error"
        detail = f"no response: {response[:80]}"
    else:
        score = "fail"
        detail = f"no expected keywords matched (looked for: {expected[:3]}...)"

    return {
        "score": score,
        "detail": detail,
        "matched": matched,
        "matched_wrong": matched_wrong,
    }


def run_suite(target: str, endpoint: str, timeout: int = 90) -> dict:
    """Run full SCT suite and return results."""
    questions = load_questions()
    run_id = f"sct-{target}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    log.info("Running SCT against %s (%d questions)", target, len(questions))

    results = {
        "run_id": run_id,
        "target": target,
        "endpoint": endpoint,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "by_category": {},
    }

    t0 = time.time()
    for i, q in enumerate(questions, 1):
        log.info("[%d/%d] %s (%s)", i, len(questions), q["id"], q["category"])
        response = query_model(q["prompt"], target, endpoint, timeout)
        scored = score_question(q, response)

        result = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q.get("difficulty", "medium"),
            "response_excerpt": response[:500],
            **scored,
        }
        results["results"].append(result)

        # Per-category aggregation
        cat = q["category"]
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"pass": 0, "partial": 0, "fail": 0, "error": 0, "total": 0}
        results["by_category"][cat][scored["score"]] += 1
        results["by_category"][cat]["total"] += 1

        log.info("  → %s: %s", scored["score"].upper(), scored["detail"][:80])

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    results["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Summary
    total = len(results["results"])
    pass_count = sum(1 for r in results["results"] if r["score"] == "pass")
    partial_count = sum(1 for r in results["results"] if r["score"] == "partial")
    fail_count = sum(1 for r in results["results"] if r["score"] == "fail")
    error_count = sum(1 for r in results["results"] if r["score"] == "error")

    results["summary"] = {
        "total": total,
        "pass": pass_count,
        "partial": partial_count,
        "fail": fail_count,
        "error": error_count,
        "score_full_credit": pass_count / total if total else 0,
        "score_partial_credit": (pass_count + 0.5 * partial_count) / total if total else 0,
    }

    return results


def print_report(results: dict):
    """Pretty-print results."""
    print("=" * 60)
    print(f"  SCT Results: {results['target']}")
    print("=" * 60)
    s = results["summary"]
    print(f"Total: {s['total']}  Pass: {s['pass']}  Partial: {s['partial']}  Fail: {s['fail']}  Error: {s['error']}")
    print(f"Full-credit score:    {s['score_full_credit']:.1%}")
    print(f"Partial-credit score: {s['score_partial_credit']:.1%}")
    print(f"Elapsed: {results['elapsed_seconds']}s")
    print()
    print("By Category:")
    for cat, stats in sorted(results["by_category"].items()):
        pct = stats["pass"] / stats["total"] if stats["total"] else 0
        print(f"  {cat:15} {stats['pass']:2}/{stats['total']:2} pass ({pct:.0%}) "
              f"[partial={stats['partial']}, fail={stats['fail']}, error={stats['error']}]")


def save_results(results: dict):
    """Save results to history."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning("Could not create results dir: %s", e)
        return
    path = RESULTS_DIR / f"{results['run_id']}.json"
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved to %s", path)
    except Exception as e:
        log.warning("Could not save results: %s", e)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Sovereign Capability Test runner")
    parser.add_argument("--target", default="core", help="Model target (core, prime, oracle)")
    parser.add_argument("--endpoint", default=CORE_ENDPOINT, help="gaia-core endpoint")
    parser.add_argument("--timeout", type=int, default=90, help="Per-question timeout")
    parser.add_argument("--category", default=None, help="Only run one category")
    parser.add_argument("--compare", default=None, help="Compare two targets (comma-separated)")

    args = parser.parse_args()

    if args.compare:
        targets = args.compare.split(",")
        all_results = {}
        for t in targets:
            all_results[t] = run_suite(t.strip(), args.endpoint, args.timeout)
            save_results(all_results[t])
            print_report(all_results[t])
            print()
        # Comparison summary
        print("=" * 60)
        print("  Comparison")
        print("=" * 60)
        cats = set()
        for r in all_results.values():
            cats.update(r["by_category"].keys())
        print(f"{'Category':15} " + " ".join(f"{t:>12}" for t in targets))
        for cat in sorted(cats):
            row = [f"{cat:15}"]
            for t in targets:
                stats = all_results[t]["by_category"].get(cat, {"pass": 0, "total": 0})
                pct = stats["pass"] / stats["total"] if stats["total"] else 0
                row.append(f"{stats['pass']:>3}/{stats['total']:<3} ({pct:>4.0%})")
            print(" ".join(row))
        return

    results = run_suite(args.target, args.endpoint, args.timeout)
    save_results(results)
    print_report(results)


if __name__ == "__main__":
    main()
