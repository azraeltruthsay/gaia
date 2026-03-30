"""
Knowledge Ingestion sleep task — wires OCW fetcher, CFR compression, and curriculum pair generator.

During sleep, GAIA:
  1. Reads KNOWLEDGE_INGESTION config for enabled courses
  2. Fetches OCW course pages via MCP web_fetch (with caching)
  3. Compresses content via CFR (falls back to raw text)
  4. Generates QLoRA training pairs from compressed/raw content
  5. Deduplicates against existing training data
  6. Queues validated pairs as candidate JSON files
  7. Plants a thought seed summarizing the ingestion run

No GPU needed — only HTTP calls and file I/O.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.SleepTask.KnowledgeIngestion")

# ── Directories ───────────────────────────────────────────────────────────────

SHARED_DIR = Path(os.getenv("SHARED_DIR", "/shared"))
RAW_CACHE_DIR = SHARED_DIR / "knowledge" / "raw"
CANDIDATE_DIR = SHARED_DIR / "knowledge" / "candidates"
SEEDS_DIR = Path("/knowledge/seeds")
EXISTING_TRAIN_PATH = "/knowledge/curricula/self-model/train.jsonl"

# ── Staleness threshold for raw page cache (seconds) ─────────────────────────

RAW_CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days


# ── Main entry point ─────────────────────────────────────────────────────────

def run_knowledge_ingestion(
    config,
    model_pool=None,
    check_interrupted: Optional[Callable] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Main entry point for the knowledge ingestion sleep task.

    Parameters
    ----------
    config : Config
        GAIA config singleton.
    model_pool : optional
        Model pool (not used — no GPU needed).
    check_interrupted : optional
        Callable that raises TaskInterruptedError if wake signal pending.

    Returns
    -------
    dict
        Summary of ingestion run: courses processed, pages fetched, pairs generated.
    """
    # ── Phase 1: Source selection ─────────────────────────────────
    ki_config = config.constants.get("KNOWLEDGE_INGESTION", {})
    if not ki_config.get("enabled", True):
        logger.info("KnowledgeIngestion: disabled in config")
        return {"status": "disabled", "courses": [], "total_pairs": 0}

    max_pages = ki_config.get("max_pages_per_cycle", 5)
    raw_dir_override = ki_config.get("raw_cache_dir")
    candidate_dir_override = ki_config.get("candidate_dir")

    raw_dir = Path(raw_dir_override) if raw_dir_override else RAW_CACHE_DIR
    candidate_dir = Path(candidate_dir_override) if candidate_dir_override else CANDIDATE_DIR

    # Collect enabled courses from config
    sources = ki_config.get("sources", {})
    enabled_courses: List[str] = []
    for source_key, source_cfg in sources.items():
        if source_cfg.get("enabled", False):
            enabled_courses.extend(source_cfg.get("courses", []))

    if not enabled_courses:
        logger.info("KnowledgeIngestion: no enabled courses in config")
        return {"status": "no_courses", "courses": [], "total_pairs": 0}

    logger.info("KnowledgeIngestion: %d courses enabled", len(enabled_courses))

    if check_interrupted:
        check_interrupted()

    # ── Process each course ───────────────────────────────────────
    from gaia_core.cognition.sleep_tasks.ocw_fetcher import (
        _mcp_call,
        fetch_course_page,
        get_course_manifest,
        PHASE_A_COURSES,
    )
    from gaia_core.cognition.sleep_tasks.curriculum_pair_generator import (
        deduplicate_against_existing,
        generate_pairs_from_cfr,
        generate_pairs_from_raw,
    )

    results: List[Dict[str, Any]] = []
    total_pairs = 0
    total_pages_fetched = 0

    for course_id in enabled_courses:
        if check_interrupted:
            check_interrupted()

        course_result = _process_course(
            course_id=course_id,
            max_pages=max_pages,
            raw_dir=raw_dir,
            candidate_dir=candidate_dir,
            check_interrupted=check_interrupted,
            _mcp_call=_mcp_call,
            fetch_course_page=fetch_course_page,
            get_course_manifest=get_course_manifest,
            generate_pairs_from_cfr=generate_pairs_from_cfr,
            generate_pairs_from_raw=generate_pairs_from_raw,
            deduplicate_against_existing=deduplicate_against_existing,
            phase_a_courses=PHASE_A_COURSES,
        )
        results.append(course_result)
        total_pairs += course_result.get("pair_count", 0)
        total_pages_fetched += course_result.get("pages_fetched", 0)

    # ── Phase 6: Plant thought seed ───────────────────────────────
    if total_pairs > 0:
        course_titles = [r.get("course_title", r.get("course_id", "?")) for r in results]
        _plant_thought_seed(
            total_pairs=total_pairs,
            pages_fetched=total_pages_fetched,
            course_titles=course_titles,
            results=results,
        )

    summary = {
        "status": "completed",
        "courses": results,
        "total_pairs": total_pairs,
        "total_pages_fetched": total_pages_fetched,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "KnowledgeIngestion: completed — %d pairs from %d pages across %d courses",
        total_pairs, total_pages_fetched, len(results),
    )
    return summary


# ── Per-course processing ─────────────────────────────────────────────────────

def _process_course(
    course_id: str,
    max_pages: int,
    raw_dir: Path,
    candidate_dir: Path,
    check_interrupted: Optional[Callable],
    _mcp_call,
    fetch_course_page,
    get_course_manifest,
    generate_pairs_from_cfr,
    generate_pairs_from_raw,
    deduplicate_against_existing,
    phase_a_courses: list,
) -> Dict[str, Any]:
    """Process a single course through fetch → CFR → pairs → queue."""

    # Look up course title
    course_title = course_id
    for c in phase_a_courses:
        if c["course_id"] == course_id:
            course_title = c.get("title", course_id)
            break

    course_raw_dir = raw_dir / course_id
    course_raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 2: Fetch ────────────────────────────────────────────
    manifest = get_course_manifest(course_id)
    pages_fetched = 0
    raw_texts: Dict[str, str] = {}

    for entry in manifest[:max_pages]:
        if check_interrupted:
            check_interrupted()

        page = entry["page"]
        raw_path = course_raw_dir / f"{page}.txt"

        # Skip if cached and recent
        if _is_cached_recent(raw_path):
            logger.debug("KnowledgeIngestion: using cached %s/%s", course_id, page)
            try:
                raw_texts[page] = raw_path.read_text(encoding="utf-8")
            except OSError:
                pass
            continue

        # Fetch via OCW fetcher
        result = fetch_course_page(course_id, page)
        if not result.get("ok"):
            logger.warning(
                "KnowledgeIngestion: fetch failed for %s/%s: %s",
                course_id, page, result.get("error", "unknown"),
            )
            continue

        content = result.get("content", "")
        if not content:
            continue

        # Save raw content atomically
        _atomic_write_text(raw_path, content)
        raw_texts[page] = content
        pages_fetched += 1
        logger.info("KnowledgeIngestion: fetched %s/%s (%d chars)", course_id, page, len(content))

    if not raw_texts:
        return {
            "course_id": course_id,
            "course_title": course_title,
            "pages_fetched": 0,
            "pair_count": 0,
            "status": "no_content",
        }

    if check_interrupted:
        check_interrupted()

    # ── Phase 3: CFR compression ──────────────────────────────────
    cfr_sections = _try_cfr_compression(raw_texts, course_id, _mcp_call)

    if check_interrupted:
        check_interrupted()

    # ── Phase 4: Pair generation ──────────────────────────────────
    metadata = {
        "category": "computer_security",
        "source_file": f"ocw/{course_id}",
        "_dataset": "OCW",
    }

    all_pairs: List[dict] = []

    if cfr_sections:
        # CFR succeeded — use compressed sections
        pairs = generate_pairs_from_cfr(cfr_sections, metadata)
        all_pairs.extend(pairs)
    else:
        # Fallback: generate from raw text
        for page, text in raw_texts.items():
            page_metadata = {**metadata, "source_file": f"ocw/{course_id}/{page}"}
            pairs = generate_pairs_from_raw(text, page_metadata)
            all_pairs.extend(pairs)

    if not all_pairs:
        return {
            "course_id": course_id,
            "course_title": course_title,
            "pages_fetched": pages_fetched,
            "pair_count": 0,
            "status": "no_pairs_generated",
        }

    # Deduplicate against existing training data
    all_pairs = deduplicate_against_existing(all_pairs, EXISTING_TRAIN_PATH)

    if check_interrupted:
        check_interrupted()

    # ── Phase 5: Candidate queuing ────────────────────────────────
    pair_count = len(all_pairs)
    if pair_count > 0:
        _queue_candidates(
            course_id=course_id,
            course_title=course_title,
            pairs=all_pairs,
            candidate_dir=candidate_dir,
        )

    return {
        "course_id": course_id,
        "course_title": course_title,
        "pages_fetched": pages_fetched,
        "pair_count": pair_count,
        "status": "completed",
    }


# ── Phase 3: CFR compression helper ──────────────────────────────────────────

def _try_cfr_compression(
    raw_texts: Dict[str, str],
    course_id: str,
    mcp_call,
) -> List[dict]:
    """Try to compress raw texts via CFR. Returns sections or empty list on failure."""
    sections: List[dict] = []

    try:
        # Ingest each page into CFR
        for page, text in raw_texts.items():
            mcp_call(
                "cfr_ingest",
                {
                    "text": text,
                    "source": f"ocw/{course_id}/{page}",
                    "tags": ["ocw", course_id],
                },
            )

        # Get synthesis
        synthesis = mcp_call("cfr_synthesize", {})

        # Extract sections from synthesis result
        if isinstance(synthesis, dict):
            raw_sections = synthesis.get("sections", synthesis.get("results", []))
            if isinstance(raw_sections, list):
                sections = raw_sections
            elif synthesis.get("content") or synthesis.get("summary"):
                # Single-section synthesis
                sections = [{
                    "title": f"MIT OCW: {course_id}",
                    "content": synthesis.get("content") or synthesis.get("summary", ""),
                }]

        if sections:
            logger.info(
                "KnowledgeIngestion: CFR produced %d sections for %s",
                len(sections), course_id,
            )
    except Exception as exc:
        logger.info(
            "KnowledgeIngestion: CFR unavailable for %s, falling back to raw text: %s",
            course_id, exc,
        )
        sections = []

    return sections


# ── Phase 5: Candidate queuing ────────────────────────────────────────────────

def _queue_candidates(
    course_id: str,
    course_title: str,
    pairs: List[dict],
    candidate_dir: Path,
) -> None:
    """Write validated pairs as a candidate JSON file."""
    candidate_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    bundle = {
        "course_id": course_id,
        "course_title": course_title,
        "source": "MIT OCW",
        "timestamp": timestamp,
        "pair_count": len(pairs),
        "pairs": pairs,
    }

    filename = f"{course_id}_{ts_slug}.json"
    bundle_path = candidate_dir / filename
    _atomic_write_json(bundle_path, bundle)
    logger.info("KnowledgeIngestion: queued %d pairs to %s", len(pairs), bundle_path)


# ── Phase 6: Thought seed ────────────────────────────────────────────────────

def _plant_thought_seed(
    total_pairs: int,
    pages_fetched: int,
    course_titles: List[str],
    results: List[dict],
) -> None:
    """Plant a thought seed summarizing the knowledge ingestion run."""
    try:
        SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("Cannot create seeds dir, skipping thought seed")
        return

    courses_text = ", ".join(course_titles[:5])
    if len(course_titles) > 5:
        courses_text += f" (+{len(course_titles) - 5} more)"

    seed_text = (
        f"Ingested {total_pairs} curriculum pairs from MIT OCW: {courses_text}. "
        f"Fetched {pages_fetched} pages. "
        f"Candidate pairs saved to {CANDIDATE_DIR}/. "
        f"Review and approve for next QLoRA training cycle."
    )

    fname = f"seed_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S%f')}.json"
    seed_obj = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed_type": "knowledge_ingestion",
        "context": {
            "prompt": "sleep_task:knowledge_ingestion",
            "packet_id": "sleep_knowledge_ingestion",
            "persona": "scholar",
        },
        "seed": seed_text,
        "reviewed": False,
        "action_taken": False,
        "result": None,
        "details": {
            "courses": [
                {"course_id": r.get("course_id"), "pairs": r.get("pair_count", 0)}
                for r in results
            ],
            "total_pairs": total_pairs,
            "pages_fetched": pages_fetched,
        },
    }

    try:
        seed_path = SEEDS_DIR / fname
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(seed_obj, f, indent=2)
        logger.info("KnowledgeIngestion: thought seed planted — %s", fname)
    except Exception as exc:
        logger.debug("Failed to plant thought seed: %s", exc)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _is_cached_recent(path: Path) -> bool:
    """Check if a raw cache file exists and is recent enough to reuse."""
    try:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < RAW_CACHE_MAX_AGE
    except OSError:
        return False


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ── Standalone test mode ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    # Standalone test mode
    try:
        from gaia_core.config import Config
        config = Config()
    except ImportError:
        # Minimal stub for testing outside the container
        class _StubConfig:
            constants = {
                "KNOWLEDGE_INGESTION": {
                    "enabled": True,
                    "max_pages_per_cycle": 5,
                    "candidate_dir": "/shared/knowledge/candidates",
                    "raw_cache_dir": "/shared/knowledge/raw",
                    "sources": {
                        "ocw_mit": {
                            "enabled": True,
                            "courses": ["6-858-computer-systems-security-fall-2014"],
                        }
                    },
                }
            }
        config = _StubConfig()

    result = run_knowledge_ingestion(config)
    print(json.dumps(result, indent=2))
