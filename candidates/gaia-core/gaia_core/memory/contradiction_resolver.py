"""
Contradiction Resolver — Observer-based adjudication for knowledge graph conflicts.

When the KnowledgeGraph detects a conflict (same subject+predicate, different object),
this module uses the Observer model (Prime on CPU) to determine the correct resolution.

Three-tier architecture:
  Tier 1: Deterministic conflict detection (in knowledge_graph.py)
  Tier 2: Observer LLM adjudication (this module)
  Tier 3: Flag for human review (when Observer confidence is low)

Usage:
    from gaia_core.memory.contradiction_resolver import create_observer_resolver

    resolver = create_observer_resolver(model_pool)
    kg = KnowledgeGraph(contradiction_callback=resolver)
"""

import json
import logging
import os
from typing import Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.ContradictionResolver")

# Observer endpoint — Prime on CPU (always available, doesn't compete for GPU)
_OBSERVER_ENDPOINT = os.environ.get("PRIME_INFERENCE_ENDPOINT", "http://gaia-prime:7777")

_ADJUDICATION_PROMPT = """\
You are a fact-checking system. A knowledge graph has detected a conflict.

EXISTING FACT(S):
{existing}

INCOMING FACT:
{incoming}

Determine the correct resolution:
- UPDATE: The incoming fact replaces the existing one (e.g., model was upgraded)
- REJECT: The incoming fact is wrong, keep the existing one
- COEXIST: Both facts are simultaneously true (e.g., a service runs on BOTH GPU and CPU)

Respond with exactly one word on the first line: UPDATE, REJECT, or COEXIST
On the second line, explain why in one sentence."""


def _format_fact(fact: dict) -> str:
    parts = [f"{fact.get('subject', '?')} → {fact.get('predicate', '?')} → {fact.get('object', '?')}"]
    if fact.get("valid_from"):
        parts.append(f"(since {fact['valid_from']})")
    if fact.get("source"):
        parts.append(f"[source: {fact['source']}]")
    return " ".join(parts)


def _call_observer(prompt: str, endpoint: str = _OBSERVER_ENDPOINT, timeout: int = 15) -> Optional[str]:
    """Call the Observer model for adjudication."""
    try:
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
        }).encode()
        req = Request(
            f"{endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        # Strip think tags
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        return answer
    except Exception as e:
        logger.debug("Observer call failed: %s", e)
        return None


def create_observer_resolver(model_pool=None, endpoint: str = None):
    """Create a contradiction callback that uses the Observer for adjudication.

    Args:
        model_pool: Optional ModelPool for direct model access (unused for now —
                    using HTTP endpoint for container isolation)
        endpoint: Override for the Observer endpoint URL

    Returns:
        A callable(Contradiction) → Contradiction suitable for KnowledgeGraph
    """
    obs_endpoint = endpoint or _OBSERVER_ENDPOINT

    def resolve(contradiction):
        """Tier 2: Observer adjudication."""
        existing_str = "\n".join(_format_fact(f) for f in contradiction.existing)
        incoming_str = _format_fact(contradiction.incoming)

        prompt = _ADJUDICATION_PROMPT.format(
            existing=existing_str,
            incoming=incoming_str,
        )

        answer = _call_observer(prompt, endpoint=obs_endpoint)
        if not answer:
            # Observer unavailable — default to update with low confidence
            contradiction.resolution = "update"
            contradiction.reason = "Observer unavailable — defaulting to update"
            logger.warning("Observer unavailable for contradiction: %s", contradiction)
            return contradiction

        # Parse the response
        lines = answer.strip().split("\n")
        decision = lines[0].strip().upper() if lines else ""
        reason = lines[1].strip() if len(lines) > 1 else ""

        if "UPDATE" in decision:
            contradiction.resolution = "update"
        elif "REJECT" in decision:
            contradiction.resolution = "reject"
        elif "COEXIST" in decision:
            contradiction.resolution = "coexist"
        else:
            # Can't parse Observer response — flag for review
            contradiction.resolution = "pending"
            reason = f"Observer response unparseable: {answer[:100]}"
            logger.warning("Observer contradiction resolution unparseable: %s", answer[:100])

        contradiction.reason = reason
        logger.info(
            "Observer resolved contradiction: %s (reason: %s)",
            contradiction.resolution, reason[:100],
        )
        return contradiction

    return resolve
