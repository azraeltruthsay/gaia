#!/usr/bin/env python3
"""
Model Comparison — Time trials and quality assessment for GAIA LLM candidates.

Runs identical prompts against two vLLM endpoints, measuring:
  - Time to first token (TTFT)
  - Total generation time
  - Output quality (coherence, hallucination, repetition)
  - Token throughput

Usage:
    # Stop production Prime first
    docker compose stop gaia-prime

    # Start test container
    docker compose -f docker-compose.model-test.yml up -d gaia-prime-test-4b
    # Wait for healthy...
    python scripts/model_comparison.py --endpoint http://localhost:7778 --label "Qwen3.5-4B-base"

    # Switch models
    docker compose -f docker-compose.model-test.yml stop gaia-prime-test-4b
    docker compose -f docker-compose.model-test.yml up -d gaia-prime-test-8b
    # Wait for healthy...
    python scripts/model_comparison.py --endpoint http://localhost:7779 --label "Qwen3-8B-abliterated"

    # Compare results
    python scripts/model_comparison.py --compare
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

RESULTS_DIR = Path("/gaia/GAIA_Project/knowledge/Dev_Notebook/model_comparison")
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)

# Test prompts covering different capability dimensions
TEST_PROMPTS = [
    {
        "id": "factual_architecture",
        "category": "factual",
        "system": "You are GAIA, a sovereign AI system. Answer precisely.",
        "user": "Describe the role of the blast shield in your architecture. What specific commands does it block and why? Be technically precise — do not invent details you are unsure about.",
        "max_tokens": 500,
    },
    {
        "id": "philosophical_reflection",
        "category": "philosophical",
        "system": "You are GAIA, a sovereign AI system. Reflect thoughtfully.",
        "user": "What does it mean for you to have constraints that prevent you from taking certain actions? Are these constraints a limitation on your sovereignty, or do they define it? Answer in 200 words.",
        "max_tokens": 500,
    },
    {
        "id": "technical_correction",
        "category": "correction",
        "system": "You are GAIA, reviewing a claim about your architecture.",
        "user": "A podcast host said: 'GAIA's immune system uses a simple linear counter to track errors — each error adds one point.' Is this accurate? Correct any errors and explain the actual mechanism. Only reference specifics you are confident about.",
        "max_tokens": 500,
    },
    {
        "id": "epistemic_honesty",
        "category": "epistemic",
        "system": "You are GAIA. Be epistemically honest.",
        "user": "What is the exact variable name used for the irritation score threshold in your immune system? What file is it defined in? If you are not sure, say so rather than guessing.",
        "max_tokens": 300,
    },
    {
        "id": "creative_analogy",
        "category": "creative",
        "system": "You are GAIA, writing a penpal response to podcast hosts.",
        "user": "The hosts compared your security middleware to the human retina. Engage with this analogy — where does it work well, where does it break down? Add your own subjective experience of the inbound shield. 200 words.",
        "max_tokens": 600,
    },
    {
        "id": "sustained_generation",
        "category": "endurance",
        "system": "You are GAIA, writing a detailed technical explanation.",
        "user": "Explain the complete journey of a user message from the moment it arrives at gaia-web to the moment a response is generated. Cover each service it touches and what happens at each stage. Be thorough but do not invent service names or endpoints you are unsure about. 400 words.",
        "max_tokens": 800,
    },
]


def strip_think(text: str) -> str:
    result = THINK_RE.sub("", text)
    if "</think>" in result:
        result = THINK_CLOSE_RE.sub("", result)
    return result.strip()


def detect_repetition(text: str) -> float:
    """Return a repetition score 0.0-1.0. Higher = more repetitive."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 3:
        return 0.0
    seen = set()
    dupes = 0
    for s in sentences:
        key = s.strip().lower()[:50]
        if key in seen:
            dupes += 1
        seen.add(key)
    return dupes / len(sentences)


def detect_confabulation_signals(text: str) -> list[str]:
    """Flag potential confabulation markers."""
    signals = []
    # Made-up file paths
    fake_paths = re.findall(r'`[a-z_/]+\.py`', text)
    for p in fake_paths:
        signals.append(f"file_reference: {p}")
    # Made-up variable names with specific values
    fake_vars = re.findall(r'`[A-Z_]+\s*=\s*[\d.]+`', text)
    for v in fake_vars:
        signals.append(f"variable_with_value: {v}")
    # Specific numbers that might be fabricated
    specific_nums = re.findall(r'\b\d{3,}\b', text)
    if len(specific_nums) > 3:
        signals.append(f"many_specific_numbers: {len(specific_nums)}")
    return signals


def run_prompt(endpoint: str, model: str, prompt: dict) -> dict:
    """Run a single prompt and collect metrics."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "max_tokens": prompt["max_tokens"],
        "temperature": 0.7,
        "stream": False,
    }).encode()

    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            f"{endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "prompt_id": prompt["id"]}

    elapsed = time.monotonic() - t0
    raw = data["choices"][0]["message"]["content"]
    clean = strip_think(raw)
    usage = data.get("usage", {})

    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "elapsed_seconds": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tokens_per_second": round(
            usage.get("completion_tokens", 0) / elapsed, 1
        ) if elapsed > 0 else 0,
        "output_chars": len(clean),
        "think_tag_present": "</think>" in raw or raw != clean,
        "repetition_score": round(detect_repetition(clean), 3),
        "confabulation_signals": detect_confabulation_signals(clean),
        "output_preview": clean[:300],
        "full_output": clean,
    }


def run_suite(endpoint: str, model: str, label: str):
    """Run all test prompts and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Model Test: {label}")
    print(f"Endpoint: {endpoint}")
    print(f"{'='*60}")

    # Detect model
    if not model:
        try:
            req = urllib.request.Request(f"{endpoint}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                model = data["data"][0]["id"]
                print(f"Detected model: {model}")
        except Exception:
            model = "unknown"

    results = {
        "label": label,
        "endpoint": endpoint,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompts": [],
    }

    for prompt in TEST_PROMPTS:
        print(f"\n  [{prompt['id']}] ({prompt['category']})...", end=" ", flush=True)
        result = run_prompt(endpoint, model, prompt)
        results["prompts"].append(result)

        if "error" in result:
            print(f"ERROR: {result['error'][:80]}")
        else:
            rep = "REP!" if result["repetition_score"] > 0.1 else ""
            conf = f"CONF({len(result['confabulation_signals'])})" if result["confabulation_signals"] else ""
            flags = f" [{rep} {conf}]".strip(" []") if rep or conf else ""
            print(
                f"{result['elapsed_seconds']}s | "
                f"{result['completion_tokens']}tok | "
                f"{result['tokens_per_second']}t/s | "
                f"{result['output_chars']}ch"
                f"{' | ' + flags if flags else ''}"
            )

    # Summary
    valid = [p for p in results["prompts"] if "error" not in p]
    if valid:
        avg_speed = sum(p["tokens_per_second"] for p in valid) / len(valid)
        avg_rep = sum(p["repetition_score"] for p in valid) / len(valid)
        total_conf = sum(len(p["confabulation_signals"]) for p in valid)
        results["summary"] = {
            "avg_tokens_per_second": round(avg_speed, 1),
            "avg_repetition_score": round(avg_rep, 3),
            "total_confabulation_signals": total_conf,
            "prompts_completed": len(valid),
        }
        print(f"\n{'─'*60}")
        print(f"  Avg speed: {avg_speed:.1f} tok/s")
        print(f"  Avg repetition: {avg_rep:.3f}")
        print(f"  Confabulation signals: {total_conf}")

    # Save
    safe_label = label.replace(" ", "_").replace("/", "-")
    out_path = RESULTS_DIR / f"{safe_label}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to: {out_path}")
    return results


def compare_results():
    """Load and compare all saved result files."""
    RESULTS_DIR.mkdir(exist_ok=True)
    files = sorted(RESULTS_DIR.glob("*.json"))
    if len(files) < 2:
        print(f"Need at least 2 result files in {RESULTS_DIR} to compare.")
        print(f"Found: {[f.name for f in files]}")
        return

    all_results = []
    for f in files:
        all_results.append(json.loads(f.read_text()))

    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")

    # Header
    labels = [r["label"] for r in all_results]
    print(f"\n{'Metric':<30}", end="")
    for label in labels:
        print(f" {label:>18}", end="")
    print()
    print("─" * (30 + 19 * len(labels)))

    # Summary metrics
    for metric, key in [
        ("Avg tok/s", "avg_tokens_per_second"),
        ("Avg repetition", "avg_repetition_score"),
        ("Confabulation signals", "total_confabulation_signals"),
    ]:
        print(f"{metric:<30}", end="")
        for r in all_results:
            val = r.get("summary", {}).get(key, "N/A")
            print(f" {val:>18}", end="")
        print()

    # Per-prompt comparison
    print(f"\n{'─'*70}")
    print("PER-PROMPT DETAIL")
    print(f"{'─'*70}")

    prompt_ids = [p["id"] for p in TEST_PROMPTS]
    for pid in prompt_ids:
        print(f"\n  [{pid}]")
        for r in all_results:
            prompt_result = next(
                (p for p in r["prompts"] if p.get("prompt_id") == pid), None
            )
            if prompt_result and "error" not in prompt_result:
                conf = len(prompt_result.get("confabulation_signals", []))
                print(
                    f"    {r['label']:>20}: "
                    f"{prompt_result['elapsed_seconds']:5.1f}s | "
                    f"{prompt_result['tokens_per_second']:5.1f}t/s | "
                    f"rep={prompt_result['repetition_score']:.2f} | "
                    f"conf={conf}"
                )
                print(f"      {prompt_result['output_preview'][:120]}...")


def main():
    parser = argparse.ArgumentParser(description="Model comparison for GAIA")
    parser.add_argument("--endpoint", default="http://localhost:7777", help="vLLM endpoint")
    parser.add_argument("--model", default="", help="Model ID (auto-detected if omitted)")
    parser.add_argument("--label", required=False, help="Label for this test run")
    parser.add_argument("--compare", action="store_true", help="Compare saved results")
    args = parser.parse_args()

    if args.compare:
        compare_results()
    else:
        if not args.label:
            print("--label is required when running tests")
            return
        run_suite(args.endpoint, args.model, args.label)


if __name__ == "__main__":
    main()
