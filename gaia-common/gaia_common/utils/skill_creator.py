"""Skill draft generator from failure patterns (GAIA_Project-5qy Phase 1).

When skill_failures.find_patterns_above_threshold reports a pattern that
hit the failure threshold, this module turns the accumulated failures
into a draft SKILL.md the human (or a future LLM pass) can complete.

Phase 1 is **deterministic** — no LLM call. The body is a structured
template that surfaces:
  - the trigger pattern
  - the recent failure queries (so the author sees what kind of asks
    keep failing)
  - a TODO marker for the playbook steps

Phase 2 (filed separately) will plug an LLM-driven body_generator into
the same call shape, generating the playbook automatically.

Output goes to /knowledge/skills/auto/<slug>/SKILL.md by default. The
"auto" subdir keeps generated skills separate from hand-written ones,
so it's clear which were synthesized and need review.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("GAIA.SkillCreator")


DEFAULT_AUTO_SKILLS_DIR = Path("/gaia/GAIA_Project/knowledge/skills/auto")


def slugify_pattern(pattern: str) -> str:
    """Turn a pattern key into a safe directory slug.

    e.g. "research:latest python release" → "research_latest_python_release"
    """
    if not pattern:
        return "unnamed"
    s = re.sub(r"[^a-z0-9]+", "_", pattern.lower()).strip("_")
    return s[:64] or "unnamed"


def _format_failure_block(failures: list[dict], *, max_items: int = 10) -> str:
    """Render the 'recent failures' bullet list for the SKILL.md body."""
    if not failures:
        return "(no failure samples recorded)"
    lines: list[str] = []
    for f in failures[:max_items]:
        query = (f.get("query") or "").strip()
        if not query:
            query = "(empty query)"
        if len(query) > 140:
            query = query[:140] + "..."
        intent = f.get("intent") or "unknown"
        ts = f.get("t", "")[:19]
        lines.append(f"- [{ts}] (intent={intent}) {query}")
    if len(failures) > max_items:
        lines.append(f"- ... ({len(failures) - max_items} more)")
    return "\n".join(lines)


def _default_body_generator(
    pattern: str,
    failures: list[dict],
    *,
    description: str,
) -> str:
    """Phase 1 deterministic body template. Has structure + failure
    context + a TODO marker for the playbook. Designed to be readable
    by both humans and a downstream LLM body-completer."""
    return f"""# Auto-generated skill from {len(failures)} failures

**Trigger pattern**: `{pattern}`

**Description**: {description}

**Status**: TODO — playbook body not yet written. Generated automatically
because the failure pattern crossed the threshold; review the failures
below and either flesh out the steps or delete this skill.

## Recent failure samples

{_format_failure_block(failures)}

## Playbook steps

TODO: write the steps that would have answered the queries above.
Until this section is filled in, the skill gateway will treat this as
a knowledge-style stub: surfacing the pattern but not auto-routing to it.

## Notes

- This file lives under `/knowledge/skills/auto/` — auto-generated
  skills are kept separate from hand-written ones.
- Delete this directory if the pattern is not worth a dedicated skill.
- Increment `version` in frontmatter when the playbook is filled in.
"""


def draft_skill_from_failures(
    pattern: str,
    failures: list[dict],
    *,
    output_dir: Optional[Path] = None,
    description: Optional[str] = None,
    body_generator: Optional[Callable[[str, list[dict]], str]] = None,
    overwrite: bool = False,
) -> dict:
    """Generate a SKILL.md draft for a failure pattern.

    Returns a dict:
      {"ok": bool, "path": str|None, "skill_name": str, "skipped": bool,
       "skip_reason": str}

    - ok=False with skip_reason="exists" means a draft already lives at
      that path; pass overwrite=True to replace.
    - ok=False with skip_reason="write_error" means IO failed
      (logged at DEBUG).
    - body_generator: optional callable(pattern, failures) → str. When
      provided, its return value replaces the default deterministic
      body. Phase 2 will pass an LLM-backed function here.
    """
    base_dir = Path(output_dir) if output_dir else DEFAULT_AUTO_SKILLS_DIR
    slug = slugify_pattern(pattern)
    skill_name = f"auto_{slug}"
    skill_dir = base_dir / skill_name
    skill_path = skill_dir / "SKILL.md"

    if skill_path.exists() and not overwrite:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "exists",
            "path": str(skill_path),
            "skill_name": skill_name,
        }

    # Description from intent + the most common-prefix query summary
    if description is None:
        intent_part = pattern.split(":", 1)[0]
        query_part = pattern.split(":", 1)[1] if ":" in pattern else pattern
        description = (
            f"Auto-generated handler for failed '{intent_part}' "
            f"queries about: {query_part}"
        )

    gen = body_generator or (
        lambda p, fs: _default_body_generator(p, fs, description=description)
    )

    try:
        body = gen(pattern, failures)
    except Exception as e:
        logger.debug("Body generator raised: %s — falling back to default", e)
        body = _default_body_generator(pattern, failures, description=description)

    # SKILL.md = YAML frontmatter + markdown body
    frontmatter = (
        "---\n"
        f"name: {skill_name}\n"
        f"description: {description}\n"
        "version: 1\n"
        "execution_mode: PLAYBOOK\n"
        "domain: auto\n"
        "status: draft\n"
        f"generated_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"trigger_pattern: {pattern}\n"
        f"failure_count: {len(failures)}\n"
        "---\n\n"
    )
    content = frontmatter + body

    try:
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(content)
    except OSError as e:
        logger.debug("draft_skill_from_failures write failed: %s", e)
        return {
            "ok": False,
            "skipped": False,
            "skip_reason": "write_error",
            "path": str(skill_path),
            "skill_name": skill_name,
        }
    logger.info(
        "Drafted skill %s from %d failures → %s",
        skill_name, len(failures), skill_path,
    )
    return {
        "ok": True,
        "skipped": False,
        "skip_reason": "",
        "path": str(skill_path),
        "skill_name": skill_name,
    }


def maybe_draft_skills(
    *,
    threshold: int = 3,
    window_hours: float = 7 * 24,
    log_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    body_generator: Optional[Callable[[str, list[dict]], str]] = None,
    clear_after_draft: bool = True,
) -> list[dict]:
    """Scan the failure log; for every pattern above threshold, draft a
    skill (skipping any that already exist). Returns list of result
    dicts from draft_skill_from_failures.

    When clear_after_draft is True (default) and a skill was newly
    written, the pattern's entries are cleared from the failure log so
    the trigger doesn't re-fire on the same accumulation.
    """
    from gaia_common.utils.skill_failures import (
        find_patterns_above_threshold,
        recent_failures_for_pattern,
        clear_pattern,
    )
    results: list[dict] = []
    patterns = find_patterns_above_threshold(
        threshold=threshold,
        window_hours=window_hours,
        log_path=log_path,
    )
    for pat in patterns:
        pattern_key = pat["pattern"]
        failures = recent_failures_for_pattern(
            pattern_key, limit=20,
            window_hours=window_hours, log_path=log_path,
        )
        result = draft_skill_from_failures(
            pattern_key, failures,
            output_dir=output_dir,
            body_generator=body_generator,
        )
        results.append(result)
        if result.get("ok") and clear_after_draft:
            try:
                cleared = clear_pattern(pattern_key, log_path=log_path)
                result["pattern_entries_cleared"] = cleared
            except Exception as e:
                logger.debug("clear_pattern after draft failed: %s", e)
    return results
