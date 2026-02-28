#!/usr/bin/env python3
"""Generate a curated daily digest from GAIA runtime logs.

Reads raw logs (generation_stream, ring buffers, self_reflection, chat_history),
extracts key signals, and renders a compact markdown digest suitable for
NotebookLM ingestion via flatten_soa.sh.

Stdlib-only — no external dependencies.

Usage:
    python scripts/generate_digest.py                         # yesterday's digest
    python scripts/generate_digest.py --date 2026-02-27       # specific date
    python scripts/generate_digest.py --log-dir /path/to/logs  # custom log dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DIGEST_BYTES = 4096
DEFAULT_LOG_DIR = "logs"
DEFAULT_OUTPUT_DIR = "knowledge/digests"

# Pattern to normalise error/warning messages for deduplication:
# strip timestamps, hex IDs, UUIDs, numbers (but keep words), paths
_NORM_RE = re.compile(
    r"""
      [0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}  # UUID
    | [0-9a-f]{6,}          # hex IDs
    | \d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d*[+Z]?\S*        # timestamps
    | \d+\.\d+[sm]?         # decimal durations
    | (?<![a-zA-Z])\d+(?![a-zA-Z])  # bare numbers (not part of words)
    """,
    re.VERBOSE,
)

# Ring-buffer log line pattern:
# 2026-02-25T03:23:30.053348+00:00 [gaia-core] ERROR:GAIA.Component:message
_RING_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})\.\d+\+\d{2}:\d{2}\s+"
    r"\[.*?\]\s+(\w+):([^:]+):(.*)"
)

# Self-reflection log pattern:
# 2026-02-18 23:04:45,696 GAIA.SelfReflection INFO message
_REFL_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2},\d+\s+\S+\s+(\w+)\s+(.*)"
)

# Confidence score pattern in reflection messages
_CONF_RE = re.compile(r"confidence[:\s]+([0-9.]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_msg(msg: str) -> str:
    """Collapse variable parts of a log message for dedup grouping."""
    return _NORM_RE.sub("<X>", msg).strip()


def _representative_msg(msg: str) -> str:
    """Return a short, readable form of a log message (first 120 chars)."""
    msg = msg.strip()
    if len(msg) > 120:
        msg = msg[:117] + "..."
    return msg


def _safe_read_lines(path: Path) -> list[str]:
    """Read a file's lines, returning [] if missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, IOError):
        return []


def _format_num(n: int | float) -> str:
    """Format a number with thousands separators."""
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_generation_stream(log_dir: Path, target_date: str) -> dict:
    """Parse generation_stream.jsonl for the target date.

    Returns dict with keys: gen_count, models (Counter), latencies (by model),
    prompt_tokens, completion_tokens, phases (Counter), roles (Counter).
    """
    path = log_dir / "generation_stream.jsonl"
    result = {
        "gen_count": 0,
        "models": Counter(),
        "latencies": defaultdict(list),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "phases": Counter(),
        "roles": Counter(),
    }

    # Track gen_id → model mapping (gen_end doesn't carry model)
    gen_id_model: dict[str, str] = {}

    for line in _safe_read_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = entry.get("ts", "")
        if not ts.startswith(target_date):
            continue

        event = entry.get("event", "")
        gen_id = entry.get("gen_id", "")

        if event == "gen_start":
            result["gen_count"] += 1
            model = entry.get("model", "unknown")
            result["models"][model] += 1
            gen_id_model[gen_id] = model
            phase = entry.get("phase", "unknown")
            result["phases"][phase] += 1
            role = entry.get("role", "unknown")
            result["roles"][role] += 1

        elif event == "gen_end":
            model = gen_id_model.get(gen_id, entry.get("model", "unknown"))
            elapsed = entry.get("elapsed_ms")
            tokens = entry.get("tokens", 0)
            if elapsed is not None:
                result["latencies"][model].append(elapsed)
            result["completion_tokens"] += tokens

    return result


def parse_ring_buffer(log_dir: Path, level: str, target_date: str) -> list[tuple[str, str, str]]:
    """Parse a ring-buffer log file for a specific level and date.

    Returns list of (component, raw_message, normalised_message) tuples.
    """
    path = log_dir / "gaia-core" / f"{level}.log"
    entries = []

    for line in _safe_read_lines(path):
        m = _RING_RE.match(line)
        if not m:
            continue
        date_str, _time, _lvl, component, message = m.groups()
        if date_str != target_date:
            continue
        norm = _normalise_msg(message)
        entries.append((component.strip(), message.strip(), norm))

    return entries


def parse_self_reflection(log_dir: Path, target_date: str) -> dict:
    """Parse self_reflection.log for the target date.

    Returns dict with: count, confidences (list[float]),
    low_confidence_entries (list[str]).
    """
    path = log_dir / "self_reflection.log"
    result = {
        "count": 0,
        "confidences": [],
        "low_confidence_entries": [],
    }

    # Pattern for final confidence lines: "completed up to N iterations; final confidence approx X"
    _final_conf_re = re.compile(r"final confidence approx\s+([0-9.]+)")
    # Pattern for per-iteration confidence: "confidence (iter N) X"
    _iter_conf_re = re.compile(r"confidence \(iter \d+\)\s+([0-9.]+)")

    for line in _safe_read_lines(path):
        m = _REFL_RE.match(line)
        if not m:
            continue
        date_str, level, message = m.groups()
        if date_str != target_date:
            continue

        # Count cycles by "completed up to" lines (one per reflection cycle)
        final_match = _final_conf_re.search(message)
        if final_match:
            result["count"] += 1
            try:
                conf = float(final_match.group(1))
                result["confidences"].append(conf)
                if conf < 0.5:
                    result["low_confidence_entries"].append(
                        (conf, _representative_msg(message))
                    )
            except ValueError:
                pass
            continue

        # Also capture per-iteration confidence for cycles that didn't
        # reach "completed" (e.g. failures) — but don't double-count
        # We'll handle this below

    # If no "completed" lines found but there are log entries,
    # fall back to counting "LLM call iteration 0" as cycle starts
    if result["count"] == 0:
        for line in _safe_read_lines(path):
            m = _REFL_RE.match(line)
            if not m:
                continue
            date_str, level, message = m.groups()
            if date_str != target_date:
                continue
            if "LLM call iteration 0 took" in message:
                result["count"] += 1

    return result


def parse_chat_history(log_dir: Path, target_date: str) -> dict:
    """Parse structured chat history for the target date.

    Returns dict with: source_counts (Counter by source type).
    """
    chat_dir = log_dir / "chat_history"
    result = {"source_counts": Counter()}

    if not chat_dir.is_dir():
        return result

    # Look for structured_YYYYMMDD.jsonl files
    date_compact = target_date.replace("-", "")
    for fname in sorted(chat_dir.iterdir()):
        if not fname.name.startswith("structured_"):
            continue
        # Match files for this date
        if date_compact not in fname.name:
            continue
        for line in _safe_read_lines(fname):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = entry.get("source", "unknown")
            result["source_counts"][source] += 1

    return result


def parse_info_signals(log_dir: Path, target_date: str) -> dict:
    """Extract system activity signals from info.log.

    Returns dict with: session_count, persona_switches (Counter),
    sleep_tasks (Counter of task names with completed counts).
    """
    entries = parse_ring_buffer(log_dir, "info", target_date)
    result = {
        "session_count": 0,
        "persona_switches": Counter(),
        "sleep_tasks": Counter(),
    }

    for component, msg, _norm in entries:
        msg_lower = msg.lower()

        # Session tracking
        if "active session" in msg_lower or "new session" in msg_lower:
            # Try to extract session count from messages like "25 active sessions"
            nums = re.findall(r"(\d+)\s+active\s+session", msg_lower)
            if nums:
                result["session_count"] = max(result["session_count"], int(nums[0]))

        # Persona switches
        if "persona" in msg_lower and ("switch" in msg_lower or "loaded" in msg_lower):
            # Extract persona name
            persona_match = re.search(r"persona[:\s]+['\"]?(\w+)", msg, re.IGNORECASE)
            if persona_match:
                result["persona_switches"][persona_match.group(1)] += 1

        # Sleep task completions
        if "completed" in msg_lower and component == "GAIA.SleepTaskScheduler":
            task_match = re.search(r"[Cc]ompleted\s+(\w+)", msg)
            if task_match:
                result["sleep_tasks"][task_match.group(1)] += 1

    return result


# ---------------------------------------------------------------------------
# Digest rendering
# ---------------------------------------------------------------------------


def render_digest(
    target_date: str,
    gen_data: dict,
    errors: list[tuple[str, str, str]],
    warnings: list[tuple[str, str, str]],
    reflection: dict,
    chat_data: dict,
    info_data: dict,
) -> str:
    """Render all parsed data into a markdown digest string."""
    sections = []
    sections.append(f"# GAIA Daily Digest — {target_date}\n")

    # --- Conversations ---
    conv_lines = []
    if gen_data["gen_count"] > 0:
        model_parts = ", ".join(
            f"{m}: {c}" for m, c in gen_data["models"].most_common()
        )
        conv_lines.append(
            f"- {_format_num(gen_data['gen_count'])} generations "
            f"across {len(gen_data['models'])} model(s) ({model_parts})"
        )

        # Latencies per model
        lat_parts = []
        for model, times in sorted(gen_data["latencies"].items()):
            if times:
                avg = sum(times) / len(times)
                lat_parts.append(f"{model}: {_format_num(avg)}ms")
        if lat_parts:
            conv_lines.append(f"- Avg latency: {', '.join(lat_parts)}")

        # Tokens
        if gen_data["completion_tokens"] > 0:
            conv_lines.append(
                f"- Completion tokens: {_format_num(gen_data['completion_tokens'])}"
            )

        # Phases
        if gen_data["phases"]:
            phase_parts = ", ".join(
                f"{p} ({c})" for p, c in gen_data["phases"].most_common(5)
            )
            conv_lines.append(f"- Phases: {phase_parts}")

    # Chat sources
    if chat_data["source_counts"]:
        src_parts = ", ".join(
            f"{s} ({c})" for s, c in chat_data["source_counts"].most_common()
        )
        conv_lines.append(f"- Sources: {src_parts}")

    if conv_lines:
        sections.append("## Conversations\n" + "\n".join(conv_lines))
    else:
        sections.append("## Conversations\nNo generation activity recorded.")

    # --- Self-Reflection ---
    refl_lines = []
    if reflection["count"] > 0:
        avg_conf = ""
        if reflection["confidences"]:
            avg_conf = f", avg confidence: {sum(reflection['confidences']) / len(reflection['confidences']):.2f}"
        refl_lines.append(
            f"- {reflection['count']} reflection cycle(s){avg_conf}"
        )
        for conf, msg in reflection["low_confidence_entries"][:3]:
            refl_lines.append(f"- Low confidence ({conf:.1f}): \"{msg}\"")
    if refl_lines:
        sections.append("## Self-Reflection\n" + "\n".join(refl_lines))

    # --- Errors ---
    error_section = _render_grouped_messages(errors, "Errors", max_items=10)
    if error_section:
        sections.append(error_section)

    # --- Warnings ---
    warn_section = _render_grouped_messages(warnings, "Warnings", max_items=5)
    if warn_section:
        sections.append(warn_section)

    # --- System Activity ---
    sys_lines = []
    if info_data["session_count"] > 0:
        sys_lines.append(f"- {info_data['session_count']} active sessions (peak)")

    if info_data["persona_switches"]:
        persona_parts = ", ".join(
            f"{p} ({c})" for p, c in info_data["persona_switches"].most_common()
        )
        sys_lines.append(f"- Persona switches: {persona_parts}")

    if info_data["sleep_tasks"]:
        task_parts = ", ".join(
            f"{_format_num(c)} {t} runs" for t, c in info_data["sleep_tasks"].most_common()
        )
        sys_lines.append(f"- Sleep tasks completed: {task_parts}")

    if sys_lines:
        sections.append("## System Activity\n" + "\n".join(sys_lines))

    digest = "\n\n".join(sections) + "\n"

    # Enforce size cap
    if len(digest.encode("utf-8")) > MAX_DIGEST_BYTES:
        digest = _truncate_digest(digest)

    return digest


def _render_grouped_messages(
    entries: list[tuple[str, str, str]],
    heading: str,
    max_items: int = 10,
) -> str | None:
    """Group messages by normalised form and render as a counted list."""
    if not entries:
        return None

    # Group by normalised message, keep a representative raw message
    groups: dict[str, dict] = {}
    for component, raw, norm in entries:
        key = f"{component}:{norm}"
        if key not in groups:
            groups[key] = {
                "component": component,
                "raw": raw,
                "count": 0,
            }
        groups[key]["count"] += 1

    # Sort by count descending
    sorted_groups = sorted(groups.values(), key=lambda g: g["count"], reverse=True)

    unique_count = len(sorted_groups)
    total_count = sum(g["count"] for g in sorted_groups)
    lines = [f"## {heading} ({unique_count} unique, {total_count} total)"]

    for g in sorted_groups[:max_items]:
        short_msg = _representative_msg(g["raw"])
        lines.append(f"- `{g['component']}: {short_msg}` — {g['count']} occurrence(s)")

    remainder = len(sorted_groups) - max_items
    if remainder > 0:
        lines.append(f"- ... and {remainder} more unique message(s)")

    return "\n".join(lines)


def _truncate_digest(digest: str) -> str:
    """Truncate the digest to fit within MAX_DIGEST_BYTES.

    Strategy: keep the header and first section, then truncate later sections
    progressively until under budget.
    """
    sections = digest.split("\n\n")
    # Always keep header
    result = sections[:1]
    budget = MAX_DIGEST_BYTES - len(result[0].encode("utf-8")) - 50  # buffer

    for section in sections[1:]:
        section_size = len(section.encode("utf-8"))
        if budget >= section_size:
            result.append(section)
            budget -= section_size + 2  # account for \n\n
        else:
            # Truncate this section
            lines = section.split("\n")
            trunc_section = [lines[0]]  # keep heading
            remaining_budget = budget - len(lines[0].encode("utf-8")) - 30
            for line in lines[1:]:
                line_size = len(line.encode("utf-8"))
                if remaining_budget >= line_size:
                    trunc_section.append(line)
                    remaining_budget -= line_size + 1
                else:
                    trunc_section.append("- *(truncated for size)*")
                    break
            result.append("\n".join(trunc_section))
            break

    return "\n\n".join(result) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a curated daily digest from GAIA runtime logs.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (default: yesterday)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        help=f"Path to logs directory (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Resolve target date
    if args.date:
        target_date = args.date
        # Validate format
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{target_date}'. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    if not log_dir.is_dir():
        print(f"Error: Log directory '{log_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating digest for {target_date}...")

    # Parse all sources
    gen_data = parse_generation_stream(log_dir, target_date)
    errors = parse_ring_buffer(log_dir, "error", target_date)
    warnings = parse_ring_buffer(log_dir, "warning", target_date)
    reflection = parse_self_reflection(log_dir, target_date)
    chat_data = parse_chat_history(log_dir, target_date)
    info_data = parse_info_signals(log_dir, target_date)

    # Render
    digest = render_digest(
        target_date, gen_data, errors, warnings, reflection, chat_data, info_data
    )

    # Write output
    output_path = output_dir / f"{target_date}_digest.md"
    output_path.write_text(digest, encoding="utf-8")

    size = len(digest.encode("utf-8"))
    print(f"Wrote {output_path} ({_format_num(size)} bytes)")

    # Summary stats
    print(f"  Generations: {gen_data['gen_count']}")
    print(f"  Errors: {len(errors)} entries ({len(set(n for _, _, n in errors))} unique)")
    print(f"  Warnings: {len(warnings)} entries ({len(set(n for _, _, n in warnings))} unique)")
    print(f"  Reflections: {reflection['count']}")


if __name__ == "__main__":
    main()
