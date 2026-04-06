"""Samvega → KV Cache Fold — persistent learning without training.

During sleep cycles, this task:
1. Queries unreviewed samvega artifacts (corrections, confidence mismatches)
2. Formats them as correction context
3. Injects into the model's prefix cache via a forward pass
4. Saves the updated KV state to disk

The result: the model wakes with corrections baked into its attention state.
Not as permanent as weight training, but persistent across restarts and
updatable each sleep cycle. The middle layer between volatile context
and permanent weights.

Called from sleep_task_scheduler.py during the DREAMING phase.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.SamvegaKVFold")

# Where to save the updated KV state
KV_STATE_PATH = "/shared/kvcache/samvega_prefix_state.pt"

# Engine endpoint (Core's embedded engine)
CORE_ENGINE = "http://localhost:8092"


def _engine_call(path: str, data: dict = None, timeout: int = 30) -> dict:
    """Call the local GAIA Engine HTTP endpoint."""
    url = f"{CORE_ENGINE}{path}"
    body = json.dumps(data or {}).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _format_corrections(artifacts: list) -> str:
    """Format samvega artifacts into a correction context block.

    This text gets injected into the prefix cache as a "world_state" or
    "behavioral" segment update, so the model processes it on the next
    forward pass.
    """
    if not artifacts:
        return ""

    lines = [
        "## Recent Corrections & Learnings",
        "The following are corrections from recent conversations.",
        "Internalize these — they override any conflicting prior understanding.",
        "",
    ]

    for i, (path, data) in enumerate(artifacts[:20], 1):  # Cap at 20
        trigger = data.get("trigger", "unknown")
        wrong = data.get("what_went_wrong", "").strip()
        corrected = data.get("corrected_understanding", "").strip()
        root = data.get("root_cause", "").strip()
        weight = data.get("weight", 0)

        if not corrected:
            continue

        lines.append(f"**Correction {i}** (weight={weight:.1f}, trigger={trigger}):")
        if wrong:
            lines.append(f"  Previous (incorrect): {wrong}")
        lines.append(f"  Corrected understanding: {corrected}")
        if root:
            lines.append(f"  Root cause: {root}")
        lines.append("")

    return "\n".join(lines)


def run_samvega_kv_fold(
    max_artifacts: int = 20,
    save_path: str = KV_STATE_PATH,
    min_weight: float = 0.3,
) -> Dict:
    """Main entry point — fold samvega corrections into KV prefix cache.

    Called from sleep_task_scheduler.py.

    Args:
        max_artifacts: Maximum number of artifacts to fold per cycle
        save_path: Where to save the updated KV state
        min_weight: Minimum artifact weight to include

    Returns:
        Dict with results: artifacts_processed, corrections_text_len, kv_saved
    """
    from gaia_core.cognition.samvega import list_unreviewed_artifacts

    t0 = time.monotonic()

    # Step 1: Gather unreviewed artifacts above minimum weight
    all_artifacts = list_unreviewed_artifacts()
    eligible = [(p, d) for p, d in all_artifacts
                if d.get("weight", 0) >= min_weight
                and d.get("corrected_understanding", "").strip()]

    if not eligible:
        logger.info("SamvegaKVFold: no eligible artifacts (min_weight=%.1f)", min_weight)
        return {"artifacts_processed": 0, "skipped": "no eligible artifacts"}

    selected = eligible[:max_artifacts]
    logger.info("SamvegaKVFold: %d artifacts eligible, processing %d",
                len(eligible), len(selected))

    # Step 2: Format corrections as text
    corrections_text = _format_corrections(selected)
    if not corrections_text:
        return {"artifacts_processed": 0, "skipped": "no correction text generated"}

    logger.info("SamvegaKVFold: correction text = %d chars", len(corrections_text))

    # Step 3: Check engine readiness
    try:
        health = _engine_call("/health")
        if not health.get("model_loaded"):
            logger.warning("SamvegaKVFold: engine not loaded, skipping")
            return {"artifacts_processed": 0, "skipped": "engine not loaded"}
    except Exception as e:
        logger.warning("SamvegaKVFold: engine unreachable: %s", e)
        return {"artifacts_processed": 0, "skipped": f"engine unreachable: {e}"}

    # Step 4: Inject corrections into the prefix cache
    # Update the "behavioral" segment with the correction text
    try:
        result = _engine_call("/cache/update", {
            "behavioral": corrections_text,
        })
        logger.info("SamvegaKVFold: cache updated: %s", result)
    except Exception as e:
        logger.error("SamvegaKVFold: cache update failed: %s", e)
        return {"artifacts_processed": 0, "skipped": f"cache update failed: {e}"}

    # Step 5: Force a KV recompute by reading the cache
    # (The update invalidated the cached KV; next get_kv() will recompute)
    # We trigger this by doing a minimal inference that warms the cache
    try:
        _engine_call("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "acknowledge"}],
            "max_tokens": 1,
        }, timeout=60)
        logger.info("SamvegaKVFold: prefix cache warmed with corrections")
    except Exception as e:
        logger.warning("SamvegaKVFold: cache warm failed (non-fatal): %s", e)

    # Step 6: Save the updated KV state to disk
    kv_saved = False
    try:
        result = _engine_call("/cache/save", {"path": save_path})
        kv_saved = result.get("ok", False)
        if kv_saved:
            logger.info("SamvegaKVFold: KV state saved to %s (%d tokens)",
                        save_path, result.get("prefix_tokens", 0))
    except Exception as e:
        logger.warning("SamvegaKVFold: KV save failed: %s", e)

    # Step 7: Mark artifacts as reviewed
    for path, data in selected:
        try:
            data["reviewed"] = True
            data["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            data["review_method"] = "kv_cache_fold"
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    elapsed = time.monotonic() - t0
    result = {
        "artifacts_processed": len(selected),
        "corrections_text_len": len(corrections_text),
        "kv_saved": kv_saved,
        "save_path": save_path,
        "elapsed_s": round(elapsed, 1),
    }
    logger.info("SamvegaKVFold complete: %s", result)
    return result
