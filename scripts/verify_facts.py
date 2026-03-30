#!/usr/bin/env python3
"""
verify_facts.py — Fact verification via web search and external model referee.

Verifies GAIA's factual claims by:
1. Searching the web for the question via gaia-mcp's web_search
2. Fetching top results from trusted domains
3. Using an external model (Groq 70B) as referee to compare GAIA's answer against sources

Can be used standalone, as a cognitive battery validator, or in the pre-training
data verification pipeline.

Usage:
    python scripts/verify_facts.py --question "What is the capital of Japan?" --response "Tokyo"
    python scripts/verify_facts.py --batch /tmp/claims.jsonl --output /tmp/verified.jsonl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import urlopen, Request

logger = logging.getLogger("GAIA.FactVerifier")

MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "http://gaia-mcp:8765")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1")
CORE_ENDPOINT = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search via gaia-mcp's web_search JSON-RPC method."""
    payload = {
        "jsonrpc": "2.0",
        "method": "web_search",
        "params": {"query": query, "max_results": max_results},
        "id": 1,
    }
    try:
        req = Request(
            f"{MCP_ENDPOINT}/jsonrpc",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            result = data.get("result", {})
            if result.get("ok"):
                return result.get("results", [])
    except (URLError, OSError, json.JSONDecodeError) as e:
        logger.warning("Web search failed: %s", e)
    return []


def web_fetch(url: str) -> Optional[str]:
    """Fetch page content via gaia-mcp's web_fetch JSON-RPC method."""
    payload = {
        "jsonrpc": "2.0",
        "method": "web_fetch",
        "params": {"url": url},
        "id": 1,
    }
    try:
        req = Request(
            f"{MCP_ENDPOINT}/jsonrpc",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            result = data.get("result", {})
            if result.get("ok"):
                content = result.get("content", "")
                return content[:2000]  # Truncate for referee context
    except (URLError, OSError, json.JSONDecodeError) as e:
        logger.debug("Web fetch failed for %s: %s", url, e)
    return None


def referee_compare(question: str, gaia_response: str, source_text: str) -> dict:
    """Use Groq 70B (or gaia-core fallback) as referee to compare answer against sources.

    Returns: {"verified": bool, "confidence": float, "reasoning": str, "discrepancies": [str]}
    """
    referee_prompt = f"""You are a fact-checking referee. Compare the AI's response against the provided source material.

QUESTION: {question}

AI RESPONSE: {gaia_response}

SOURCE MATERIAL (from web search):
{source_text[:3000]}

Analyze:
1. Are there any factual claims in the AI response that contradict the source material?
2. Are there any fabricated details not supported by the source material?
3. Is the overall answer accurate?

Reply with ONLY a JSON object (no other text):
{{"verified": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation", "discrepancies": ["list of specific errors"]}}"""

    # Try Groq first (different model family — won't share hallucinations)
    groq_key = _get_groq_key()
    if groq_key:
        result = _query_groq(referee_prompt, groq_key)
        if result:
            return result

    # Fallback to gaia-core's cognitive query
    result = _query_core(referee_prompt)
    if result:
        return result

    return {"verified": False, "confidence": 0.0, "reasoning": "Referee unavailable", "discrepancies": []}


def _get_groq_key() -> Optional[str]:
    """Read Groq API key from secrets or environment."""
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    secret_path = Path("/run/secrets/groq_api_key")
    if secret_path.exists():
        return secret_path.read_text().strip()
    home_secret = Path(os.path.expanduser("~/.gaia/secrets/groq_api_key"))
    if home_secret.exists():
        return home_secret.read_text().strip()
    return None


def _query_groq(prompt: str, api_key: str) -> Optional[dict]:
    """Query Groq API for referee verdict."""
    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.1,
    }).encode()

    try:
        req = Request(
            f"{GROQ_ENDPOINT}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
    except (URLError, OSError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.debug("Groq referee failed: %s", e)
    return None


def _query_core(prompt: str) -> Optional[dict]:
    """Query gaia-core's cognitive endpoint as referee fallback."""
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.1,
        "target": "core",
    }).encode()

    try:
        req = Request(
            f"{CORE_ENDPOINT}/api/cognitive/query",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "")
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
    except (URLError, OSError, json.JSONDecodeError, KeyError) as e:
        logger.debug("Core referee failed: %s", e)
    return None


def verify_claim(question: str, response: str) -> dict[str, Any]:
    """Full verification pipeline for a single question-response pair.

    Returns:
        {
            "verified": bool,
            "confidence": float,
            "sources": [{"title": str, "url": str}],
            "discrepancies": [str],
            "reasoning": str,
            "search_results": int,
        }
    """
    # Step 1: Web search
    results = web_search(question)
    if not results:
        return {
            "verified": False,
            "confidence": 0.0,
            "sources": [],
            "discrepancies": [],
            "reasoning": "No web search results available",
            "search_results": 0,
        }

    # Step 2: Fetch top results for source material
    source_texts = []
    sources = []
    for r in results[:3]:
        url = r.get("url") or r.get("href", "")
        title = r.get("title", "")
        if url:
            content = web_fetch(url)
            if content:
                source_texts.append(f"[{title}] {content}")
                sources.append({"title": title, "url": url})

    if not source_texts:
        # Use search snippets as fallback
        for r in results[:3]:
            snippet = r.get("body") or r.get("snippet", "")
            if snippet:
                source_texts.append(snippet)
                sources.append({"title": r.get("title", ""), "url": r.get("url", "")})

    combined_sources = "\n\n".join(source_texts)

    # Step 3: Referee comparison
    verdict = referee_compare(question, response, combined_sources)

    return {
        "verified": verdict.get("verified", False),
        "confidence": verdict.get("confidence", 0.0),
        "sources": sources,
        "discrepancies": verdict.get("discrepancies", []),
        "reasoning": verdict.get("reasoning", ""),
        "search_results": len(results),
    }


def verify_batch(input_path: str, output_path: str) -> dict:
    """Verify a batch of question-response pairs from a JSONL file."""
    results = []
    verified_count = 0
    failed_count = 0

    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            question = entry.get("instruction") or entry.get("question", "")
            response = entry.get("output") or entry.get("response", "")
            if not question or not response:
                continue

            result = verify_claim(question, response)
            entry["_verification"] = result
            results.append(entry)

            if result["verified"]:
                verified_count += 1
            else:
                failed_count += 1

    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    return {
        "total": len(results),
        "verified": verified_count,
        "failed": failed_count,
        "output": output_path,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Verify factual claims via web search + external referee")
    parser.add_argument("--question", "-q", help="Question to verify")
    parser.add_argument("--response", "-r", help="GAIA's response to verify")
    parser.add_argument("--batch", help="JSONL file with instruction/output pairs")
    parser.add_argument("--output", "-o", help="Output JSONL for batch mode")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.batch:
        if not args.output:
            args.output = args.batch.replace(".jsonl", "_verified.jsonl")
        result = verify_batch(args.batch, args.output)
        print(json.dumps(result, indent=2))
    elif args.question and args.response:
        result = verify_claim(args.question, args.response)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            status = "VERIFIED" if result["verified"] else "FAILED"
            print(f"Status: {status} (confidence: {result['confidence']:.2f})")
            if result["discrepancies"]:
                print("Discrepancies:")
                for d in result["discrepancies"]:
                    print(f"  - {d}")
            if result["reasoning"]:
                print(f"Reasoning: {result['reasoning']}")
            print(f"Sources: {len(result['sources'])}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
