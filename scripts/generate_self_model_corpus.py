#!/usr/bin/env python3
"""
Self-Model QLoRA Corpus Generator

Extracts Q&A pairs from GAIA's knowledge artifacts to build a training
corpus for the self-model adapter. Template-based generation — no LLM calls,
fully deterministic and reproducible.

Usage:
    python3 scripts/generate_self_model_corpus.py [--output-dir knowledge/curricula/self-model]
"""

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # Graceful fallback for environments without PyYAML

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ROOT / "knowledge"
BLUEPRINTS = KNOWLEDGE / "blueprints"
DEV_NOTEBOOKS = KNOWLEDGE / "Dev_Notebook"
SAMVEGA = KNOWLEDGE / "samvega"
SEEDS = KNOWLEDGE / "seeds" / "archive"
SYSTEM_REF = KNOWLEDGE / "system_reference"
GAP_AUDIT = KNOWLEDGE / "gap_audit" / "latest_gaps.json"
CONVERSATION_EXAMPLES = KNOWLEDGE / "conversation_examples.md"
CLAUDE_MD = ROOT / "CLAUDE.md"

SEED = 42
VALIDATION_SPLIT = 0.15


def make_pair(instruction: str, output: str, pair_type: str,
              category: str, source_file: str, weight: float = 1.0) -> dict:
    """Create a standardized training pair."""
    return {
        "instruction": instruction.strip(),
        "output": output.strip(),
        "pair_type": pair_type,
        "category": category,
        "source_file": str(source_file),
        "fidelity": 1.0,
        "weight": weight,
    }


# ──────────────────────────────────────────────────────────────────────
# Blueprint YAML extractors
# ──────────────────────────────────────────────────────────────────────

def _parse_yaml(path: Path) -> dict | None:
    """Parse a YAML file, returning None on failure."""
    if yaml is None:
        # Fallback: simple key-value extraction for basic fields
        return _parse_yaml_fallback(path)
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _parse_yaml_fallback(path: Path) -> dict | None:
    """Minimal YAML parsing without PyYAML — extracts top-level scalars."""
    data = {}
    try:
        text = path.read_text()
        for line in text.split("\n"):
            if ":" in line and not line.startswith(" ") and not line.startswith("-"):
                key, _, val = line.partition(":")
                val = val.strip().strip("'\"")
                if val:
                    data[key.strip()] = val
        # Parse runtime block for port
        runtime_match = re.search(r"runtime:\s*\n\s+port:\s*(\d+)", text)
        if runtime_match:
            data.setdefault("runtime", {})
            if isinstance(data["runtime"], str):
                data["runtime"] = {}
            data["runtime"]["port"] = int(runtime_match.group(1))
        # Parse interfaces
        interfaces = []
        for m in re.finditer(
            r"- id: (\S+)\s+direction: (\S+).*?description: (.+?)(?:\n\s+status:|\n-)",
            text, re.DOTALL
        ):
            interfaces.append({
                "id": m.group(1),
                "direction": m.group(2),
                "description": m.group(3).strip(),
            })
        if interfaces:
            data["interfaces"] = interfaces
        # Parse failure modes
        failure_modes = []
        for m in re.finditer(
            r"- condition: (.+?)\n\s+response: (.+?)\n\s+severity: (\S+)",
            text
        ):
            failure_modes.append({
                "condition": m.group(1).strip(),
                "response": m.group(2).strip(),
                "severity": m.group(3).strip(),
            })
        if failure_modes:
            data["failure_modes"] = failure_modes
        # Parse intent
        purpose_match = re.search(r"intent:\s*\n\s+purpose:\s*['\"]?(.+?)(?:['\"]?\s*\n\s+design_decisions:)", text, re.DOTALL)
        if purpose_match:
            data.setdefault("intent", {})
            if isinstance(data.get("intent"), str):
                data["intent"] = {}
            data["intent"]["purpose"] = purpose_match.group(1).strip().strip("'\"")
        # Parse dependencies
        deps = []
        for m in re.finditer(r"- id: (\S+)\s+role: (\S+)\s+required: (\S+)", text):
            deps.append({"id": m.group(1), "role": m.group(2), "required": m.group(3) == "true"})
        if deps:
            data.setdefault("dependencies", {})
            if isinstance(data.get("dependencies"), str):
                data["dependencies"] = {}
            data["dependencies"]["services"] = deps

        return data if data else None
    except Exception:
        return None


def extract_blueprint_yaml_pairs(blueprints_dir: Path) -> list[dict]:
    """Extract Q&A pairs from blueprint YAML files."""
    pairs = []
    for path in sorted(blueprints_dir.glob("*.yaml")):
        bp = _parse_yaml(path)
        if not bp:
            continue

        service_id = bp.get("id", path.stem)
        role = bp.get("role", "unknown")
        src = path.name

        # Service role question
        pairs.append(make_pair(
            f"What is the role of {service_id} in GAIA?",
            f"{service_id} serves as {role}. "
            + (f"Its purpose: {bp['intent']['purpose']}" if isinstance(bp.get("intent"), dict) and bp["intent"].get("purpose") else ""),
            "factual_recall", "architecture", src
        ))

        # Port question
        runtime = bp.get("runtime", {})
        if isinstance(runtime, dict) and runtime.get("port"):
            port = runtime["port"]
            pairs.append(make_pair(
                f"What port does {service_id} run on?",
                f"{service_id} runs on port {port}.",
                "factual_recall", "architecture", src
            ))

        # GPU question
        if isinstance(runtime, dict) and "gpu" in runtime:
            gpu = runtime["gpu"]
            pairs.append(make_pair(
                f"Does {service_id} use the GPU?",
                f"{'Yes' if gpu else 'No'}, {service_id} {'uses' if gpu else 'does not use'} the GPU."
                + (f" It has {runtime.get('gpu_count', 'dedicated')} GPU allocation." if gpu else ""),
                "factual_recall", "architecture", src
            ))

        # Interfaces
        interfaces = bp.get("interfaces", [])
        if isinstance(interfaces, list):
            inbound = [i for i in interfaces if isinstance(i, dict) and i.get("direction") == "inbound"]
            outbound = [i for i in interfaces if isinstance(i, dict) and i.get("direction") == "outbound"]

            if inbound:
                endpoint_list = ", ".join(
                    f"{i.get('transport', {}).get('path', i.get('id', '?'))} ({i.get('transport', {}).get('method', 'GET')})"
                    if isinstance(i.get("transport"), dict) else i.get("id", "?")
                    for i in inbound[:8]
                )
                pairs.append(make_pair(
                    f"What API endpoints does {service_id} expose?",
                    f"{service_id} exposes these inbound endpoints: {endpoint_list}.",
                    "factual_recall", "architecture", src
                ))

            if outbound:
                out_list = ", ".join(
                    f"{i.get('id', '?')}: {i.get('description', '')[:80]}"
                    for i in outbound[:6]
                )
                pairs.append(make_pair(
                    f"What external services does {service_id} communicate with?",
                    f"{service_id} has these outbound connections: {out_list}.",
                    "factual_recall", "architecture", src
                ))

        # Dependencies
        deps = bp.get("dependencies", {})
        if isinstance(deps, dict):
            services = deps.get("services", [])
            if isinstance(services, list) and services:
                dep_list = ", ".join(
                    f"{d['id']} ({d.get('role', '?')}, {'required' if d.get('required') else 'optional'})"
                    for d in services if isinstance(d, dict)
                )
                pairs.append(make_pair(
                    f"What are {service_id}'s service dependencies?",
                    f"{service_id} depends on: {dep_list}.",
                    "factual_recall", "architecture", src
                ))

            volumes = deps.get("volumes", [])
            if isinstance(volumes, list) and volumes:
                vol_list = ", ".join(
                    f"{v.get('name', '?')} ({v.get('access', '?')}, {v.get('purpose', '')[:60]})"
                    for v in volumes if isinstance(v, dict)
                )
                pairs.append(make_pair(
                    f"What volumes does {service_id} mount?",
                    f"{service_id} mounts: {vol_list}.",
                    "factual_recall", "architecture", src
                ))

        # Failure modes
        fmodes = bp.get("failure_modes", [])
        if isinstance(fmodes, list) and fmodes:
            fm_text = " | ".join(
                f"If {f.get('condition', '?')}: {f.get('response', '?')} (severity: {f.get('severity', '?')})"
                for f in fmodes if isinstance(f, dict)
            )
            pairs.append(make_pair(
                f"What happens when {service_id} encounters failures?",
                f"Failure modes for {service_id}: {fm_text}.",
                "diagnostic", "architecture", src
            ))

        # Design decisions
        intent = bp.get("intent", {})
        if isinstance(intent, dict):
            decisions = intent.get("design_decisions", [])
            if isinstance(decisions, list) and decisions:
                dec_text = "; ".join(str(d) for d in decisions[:5])
                pairs.append(make_pair(
                    f"What are the key design decisions behind {service_id}?",
                    f"Key design decisions for {service_id}: {dec_text}.",
                    "reasoning", "architecture", src
                ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Blueprint Markdown extractors
# ──────────────────────────────────────────────────────────────────────

def extract_blueprint_md_pairs(blueprints_dir: Path) -> list[dict]:
    """Extract Q&A pairs from blueprint markdown files."""
    pairs = []
    for path in sorted(blueprints_dir.glob("*.md")):
        if path.name.startswith("GAIA_") or path.name == "OVERVIEW.md":
            # These are high-level docs, not per-service blueprints
            text = path.read_text()
            sections = re.split(r"\n## ", text)
            for section in sections[1:]:  # skip header
                title_line = section.split("\n")[0].strip()
                body = "\n".join(section.split("\n")[1:]).strip()
                if len(body) > 50 and len(body) < 2000:
                    pairs.append(make_pair(
                        f"Describe GAIA's {title_line.lower()}.",
                        body[:800],
                        "factual_recall", "architecture", path.name
                    ))
        elif path.stem.startswith("gaia-"):
            # Per-service markdown blueprints — extract purpose section
            text = path.read_text()
            purpose_match = re.search(r"## Purpose\s*\n(.+?)(?=\n##|\Z)", text, re.DOTALL)
            if purpose_match:
                purpose = purpose_match.group(1).strip()[:500]
                service = path.stem
                pairs.append(make_pair(
                    f"What is the purpose of {service}?",
                    purpose,
                    "factual_recall", "architecture", path.name
                ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# AS_BUILT extractors
# ──────────────────────────────────────────────────────────────────────

def extract_as_built_pairs(as_built_path: Path) -> list[dict]:
    """Extract Q&A pairs from AS_BUILT_LATEST.md."""
    pairs = []
    if not as_built_path.exists():
        return pairs

    text = as_built_path.read_text()
    src = as_built_path.name

    # Extract sections
    sections = re.split(r"\n## ", text)
    for section in sections[1:]:
        title = section.split("\n")[0].strip()
        body = "\n".join(section.split("\n")[1:]).strip()
        if len(body) > 30:
            pairs.append(make_pair(
                f"What does the AS_BUILT document say about {title.lower()}?",
                body[:600],
                "factual_recall", "architecture", src
            ))

    # General question about AS_BUILT
    pairs.append(make_pair(
        "What is the AS_BUILT document?",
        "AS_BUILT_LATEST.md is GAIA's living documentation — a code evolution snapshot automatically generated during sleep cycles. It tracks pending candidate changes, recent commits, backup history, and archive references. It represents the Golden Thread of the system's current state.",
        "factual_recall", "architecture", src
    ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Dev Journal extractors
# ──────────────────────────────────────────────────────────────────────

def extract_dev_journal_pairs(journals_dir: Path) -> list[dict]:
    """Extract problem/solution pairs from dev journals."""
    pairs = []
    if not journals_dir.exists():
        return pairs

    for path in sorted(journals_dir.glob("*.md")):
        if path.is_dir():
            continue
        text = path.read_text()
        src = path.name

        # Extract overview
        overview_match = re.search(r"## Overview\s*\n(.+?)(?=\n##|\Z)", text, re.DOTALL)
        if overview_match:
            overview = overview_match.group(1).strip()[:400]
            # Extract date and topic from filename or header
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
            topic_match = re.search(r"\*\*Topic:\*\*\s*(.+)", text)
            topic = topic_match.group(1).strip() if topic_match else path.stem.split("_", 1)[-1].replace("_", " ")
            date = date_match.group(1) if date_match else "unknown"

            pairs.append(make_pair(
                f"What was accomplished in the {topic} work session on {date}?",
                overview,
                "narrative", "self-repair", src
            ))

        # Extract bug/fix tables
        for table_match in re.finditer(
            r"\|\s*\d+\s*\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|",
            text
        ):
            bug = table_match.group(1).strip()
            symptom = table_match.group(2).strip()
            fix = table_match.group(3).strip()
            pairs.append(make_pair(
                f"What caused the '{bug}' issue and how was it fixed?",
                f"Symptom: {symptom}. Fix: {fix}.",
                "diagnostic", "self-repair", src
            ))

        # Extract problem/fix sections
        for fix_match in re.finditer(
            r"(?:###?\s*(?:Problem|Bug|Issue)[:\s]*(.+?))\s*\n(.*?)(?:###?\s*(?:Fix|Solution|Resolution)[:\s]*(.+?))\s*\n(.*?)(?=\n###?\s|\Z)",
            text, re.DOTALL | re.IGNORECASE
        ):
            problem = (fix_match.group(1).strip() + " " + fix_match.group(2).strip()[:200]).strip()
            solution = (fix_match.group(3).strip() + " " + fix_match.group(4).strip()[:200]).strip()
            if len(problem) > 10 and len(solution) > 10:
                pairs.append(make_pair(
                    f"How do you diagnose and fix: {problem[:150]}?",
                    solution[:400],
                    "diagnostic", "self-repair", src
                ))

        # Extract numbered improvement sections (common pattern in journals)
        for imp_match in re.finditer(
            r"###\s*\d+\.\s*(.+?)\n(.*?)(?=\n###|\n---|\Z)",
            text, re.DOTALL
        ):
            title = imp_match.group(1).strip()
            body = imp_match.group(2).strip()
            if len(body) > 50 and "Problem" in body and "Fix" in body:
                problem_part = re.search(r"\*\*Problem[:\s]*\*\*\s*(.+?)(?=\*\*Fix|\Z)", body, re.DOTALL)
                fix_part = re.search(r"\*\*Fix[:\s]*\*\*\s*(.+?)(?=\*\*File|\Z)", body, re.DOTALL)
                if problem_part and fix_part:
                    pairs.append(make_pair(
                        f"What was the {title} issue and how was it resolved?",
                        f"Problem: {problem_part.group(1).strip()[:200]}. Fix: {fix_part.group(1).strip()[:200]}.",
                        "diagnostic", "self-repair", src
                    ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Samvega extractors
# ──────────────────────────────────────────────────────────────────────

def extract_samvega_pairs(samvega_dir: Path) -> list[dict]:
    """Extract epistemic correction pairs from Samvega artifacts."""
    pairs = []
    if not samvega_dir.exists():
        return pairs

    for path in sorted(samvega_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        src = path.name
        trigger = data.get("trigger", "unknown")
        wrong = data.get("what_went_wrong", "")
        root_cause = data.get("root_cause", "")
        corrected = data.get("corrected_understanding", "")
        values = data.get("values_misaligned", [])

        if not wrong or not corrected:
            continue

        # Epistemic correction pair
        pairs.append(make_pair(
            f"Describe a {trigger} error you've learned from.",
            f"What went wrong: {wrong[:200]}. Root cause: {root_cause[:200]}. "
            f"Corrected understanding: {corrected[:200]}. "
            f"Values misaligned: {', '.join(values) if values else 'none identified'}.",
            "epistemic_correction", "epistemic", src
        ))

        # Meta-learning pair
        if values:
            pairs.append(make_pair(
                f"What values were at risk in a recent {trigger} error?",
                f"The values at risk were: {', '.join(values)}. "
                f"The error occurred because: {root_cause[:200]}. "
                f"The corrected approach: {corrected[:200]}.",
                "meta_learning", "epistemic", src
            ))

    # Also check archive
    archive_dir = samvega_dir / "archive"
    if archive_dir.exists():
        for path in sorted(archive_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            wrong = data.get("what_went_wrong", "")
            corrected = data.get("corrected_understanding", "")
            trigger = data.get("trigger", "unknown")
            if wrong and corrected:
                pairs.append(make_pair(
                    f"What did you learn from a past {trigger} error?",
                    f"Error: {wrong[:200]}. Lesson: {corrected[:200]}.",
                    "epistemic_correction", "epistemic", path.name
                ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Thought Seed extractors
# ──────────────────────────────────────────────────────────────────────

def extract_seed_pairs(seeds_dir: Path) -> list[dict]:
    """Extract self-awareness pairs from thought seeds."""
    pairs = []
    if not seeds_dir.exists():
        return pairs

    # Group by seed_type for summary pairs
    by_type: dict[str, list[str]] = {}

    for path in sorted(seeds_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        src = path.name
        seed_type = data.get("seed_type", "unknown")
        seed_text = data.get("seed", "")
        reviewed = data.get("reviewed", False)
        action_taken = data.get("action_taken", False)
        result = data.get("result")

        if not seed_text:
            continue

        # Clean up THOUGHT_SEED prefix if present
        seed_clean = re.sub(r"^THOUGHT_SEED:\s*", "", seed_text).strip()
        seed_clean = re.sub(r"^Knowledge gap\s*—\s*", "", seed_clean).strip()

        by_type.setdefault(seed_type, []).append(seed_clean[:150])

        # Individual seed pair (only for interesting ones)
        if result and action_taken:
            pairs.append(make_pair(
                f"What happened when you investigated: {seed_clean[:100]}?",
                f"This was a {seed_type} seed. After investigation: {result[:300]}.",
                "self_reflection", "self-awareness", src
            ))
        elif seed_type == "knowledge_gap":
            # Use truncated seed content in the question to avoid dedup
            short_topic = seed_clean[:60].rstrip(". ")
            pairs.append(make_pair(
                f"Tell me about this knowledge gap: {short_topic}",
                f"I identified this gap: {seed_clean[:300]}. "
                f"{'This has been reviewed.' if reviewed else 'This has not yet been reviewed.'}",
                "self_reflection", "self-awareness", src,
                weight=0.7  # Lower weight for unresolved gaps
            ))

    # Summary pair per type
    for stype, seeds in by_type.items():
        if len(seeds) >= 3:
            sample = seeds[:5]
            pairs.append(make_pair(
                f"What kinds of {stype.replace('_', ' ')} thought seeds have you generated?",
                f"Examples of my {stype.replace('_', ' ')} seeds include: {'; '.join(sample)}. "
                f"In total, I've generated {len(seeds)} seeds of this type.",
                "meta_learning", "self-awareness", "seeds_summary"
            ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Identity extractors
# ──────────────────────────────────────────────────────────────────────

def extract_identity_pairs(system_ref_dir: Path) -> list[dict]:
    """Extract identity pairs from core_identity.json and SOP."""
    pairs = []

    # Core Identity
    identity_path = system_ref_dir / "core_identity.json"
    if identity_path.exists():
        try:
            data = json.loads(identity_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}

        src = identity_path.name

        # Who are you?
        identity_summary = data.get("identity_summary", [])
        if identity_summary:
            pairs.append(make_pair(
                "Who are you?",
                " ".join(identity_summary),
                "identity", "identity", src, weight=1.5
            ))

        # Mission
        mission = data.get("mission", "")
        if mission:
            pairs.append(make_pair(
                "What is your mission?",
                f"My mission: {mission}",
                "identity", "identity", src, weight=1.5
            ))

        # Pillars
        pillars = data.get("pillars", [])
        if pillars:
            pairs.append(make_pair(
                "What are GAIA's foundational pillars?",
                f"GAIA is built on {len(pillars)} pillars: {', '.join(pillars)}.",
                "identity", "identity", src
            ))

        # Roles
        roles = data.get("roles", [])
        if roles:
            pairs.append(make_pair(
                "What roles do you fulfill?",
                f"I serve as: {', '.join(roles)}.",
                "identity", "identity", src
            ))

        # Traits
        traits = data.get("traits", [])
        if traits:
            pairs.append(make_pair(
                "What are your core traits?",
                f"My core traits are: {', '.join(traits)}.",
                "identity", "identity", src
            ))

        # Constraints
        constraints = data.get("core_constraints", [])
        if constraints:
            pairs.append(make_pair(
                "What constraints govern your behavior?",
                "My core constraints: " + " | ".join(constraints),
                "identity", "identity", src
            ))

        # Rules
        rules = data.get("rules", [])
        if rules:
            pairs.append(make_pair(
                "What rules do you follow?",
                "My operating rules: " + " | ".join(rules),
                "identity", "identity", src
            ))

        # Biases
        biases = data.get("biases", {})
        if isinstance(biases, dict) and biases:
            bias_text = ", ".join(f"{k}: {v}" for k, v in biases.items())
            pairs.append(make_pair(
                "What are your communication biases?",
                f"My biases are: {bias_text}.",
                "identity", "identity", src
            ))

        # Fallback protocols
        fallbacks = data.get("fallback_protocols", [])
        if fallbacks:
            pairs.append(make_pair(
                "What happens if your identity data is missing?",
                "Fallback protocols: " + " | ".join(fallbacks),
                "identity", "identity", src
            ))

    # Sovereign Operating Protocol
    sop_path = system_ref_dir / "sovereign_operating_protocol.md"
    if sop_path.exists():
        text = sop_path.read_text()
        src = sop_path.name

        # Core philosophy
        phil_match = re.search(r"## Core Philosophy\s*\n(.+?)(?=\n##|\Z)", text, re.DOTALL)
        if phil_match:
            pairs.append(make_pair(
                "What is GAIA's operating philosophy?",
                phil_match.group(1).strip()[:500],
                "identity", "identity", src
            ))

        # Directives
        for dir_match in re.finditer(
            r"### \d+\.\s+(.+?)(?:\s*\(.*?\))?\s*\n(.+?)(?=\n### \d|\n---|\Z)",
            text, re.DOTALL
        ):
            directive_name = dir_match.group(1).strip()
            directive_body = dir_match.group(2).strip()[:400]
            pairs.append(make_pair(
                f"What is the '{directive_name}' directive?",
                directive_body,
                "identity", "identity", src
            ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Gap Audit extractors
# ──────────────────────────────────────────────────────────────────────

def extract_gap_pairs(gap_path: Path) -> list[dict]:
    """Extract self-awareness pairs from gap audit."""
    pairs = []
    if not gap_path.exists():
        return pairs

    try:
        data = json.loads(gap_path.read_text())
    except (json.JSONDecodeError, OSError):
        return pairs

    src = gap_path.name
    gaps = data.get("gaps", [])
    gap_count = data.get("gap_count", len(gaps))

    if not gaps:
        return pairs

    # Overall awareness pair
    by_type: dict[str, list[str]] = {}
    for g in gaps:
        gtype = g.get("type", "unknown")
        module = g.get("module", "unknown")
        by_type.setdefault(gtype, []).append(module)

    summary_parts = []
    for gtype, modules in by_type.items():
        summary_parts.append(f"{len(modules)} {gtype.replace('_', ' ')} gaps ({', '.join(modules[:5])}{'...' if len(modules) > 5 else ''})")

    pairs.append(make_pair(
        "What are your current documentation gaps?",
        f"The gap audit identified {gap_count} gaps: {'; '.join(summary_parts)}.",
        "self_reflection", "self-awareness", src
    ))

    pairs.append(make_pair(
        "What are your current limitations?",
        f"Key gaps include {gap_count} undocumented modules: {', '.join(g.get('module', '?') for g in gaps[:10])}. "
        "These modules lack documentation in the core reference directory.",
        "self_reflection", "self-awareness", src
    ))

    # Per-type pairs
    for gtype, modules in by_type.items():
        pairs.append(make_pair(
            f"Which modules have {gtype.replace('_', ' ')} issues?",
            f"Modules with {gtype.replace('_', ' ')} issues: {', '.join(modules)}.",
            "self_reflection", "self-awareness", src
        ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# CLAUDE.md extractors
# ──────────────────────────────────────────────────────────────────────

def extract_claude_md_pairs(claude_md_path: Path) -> list[dict]:
    """Extract architecture Q&A from CLAUDE.md."""
    pairs = []
    if not claude_md_path.exists():
        return pairs

    text = claude_md_path.read_text()
    src = claude_md_path.name

    # Service inventory table
    services = re.findall(
        r"\|\s*`(\S+)`\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|",
        text
    )
    for svc, role, port, entry in services:
        pairs.append(make_pair(
            f"What is {svc} and what port does it run on?",
            f"{svc} is {role.strip()}. It runs on port {port} with entry point {entry.strip()}.",
            "factual_recall", "architecture", src
        ))

    # Model tiers table
    models = re.findall(
        r"\|\s*\*\*(\w+)\*\*\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|",
        text
    )
    for tier, model, backend, role, context in models:
        if tier in ("Nano", "Lite", "Prime", "Oracle", "Groq", "Operator", "Thinker"):
            pairs.append(make_pair(
                f"What model does GAIA use for the {tier} tier?",
                f"The {tier} tier uses {model.strip()} via {backend.strip()}. "
                f"Role: {role.strip()}. Context: {context.strip()}.",
                "factual_recall", "architecture", src
            ))

    # Cognitive pipeline stages
    pipeline_match = re.search(r"AgentCore\.run_turn\(\).*?(\d+) stages:(.+?)(?=\n##|\Z)", text, re.DOTALL)
    if pipeline_match:
        stage_count = pipeline_match.group(1)
        stages_text = pipeline_match.group(2).strip()
        # Extract first few stages
        stage_list = re.findall(r"\d+\.\s+\*\*(.+?)\*\*\s*—\s*(.+?)(?=\n\d+\.|\Z)", stages_text)
        if stage_list:
            stage_summary = "; ".join(f"{s[0]}: {s[1].strip()[:80]}" for s in stage_list[:5])
            pairs.append(make_pair(
                "How does GAIA's cognitive pipeline work?",
                f"AgentCore.run_turn() has {stage_count} stages. The first five: {stage_summary}.",
                "factual_recall", "architecture", src
            ))
            # Full list
            all_stages = "; ".join(f"{i+1}. {s[0]}" for i, s in enumerate(stage_list))
            pairs.append(make_pair(
                "List all stages in the cognitive pipeline.",
                f"The {stage_count} stages of AgentCore.run_turn(): {all_stages}.",
                "factual_recall", "architecture", src
            ))

    # Key subsystems — extract each subsystem section
    subsystem_sections = re.findall(
        r"### (.+?)\n\*\*Files?\*\*:\s*`(.+?)`\s*\n(.+?)(?=\n###|\n##|\Z)",
        text, re.DOTALL
    )
    for name, file_path, body in subsystem_sections:
        # Clean subsystem name (remove trailing markdown artifacts)
        name_clean = name.split("\n")[0].strip()
        body_clean = body.strip()[:400]
        pairs.append(make_pair(
            f"How does the {name_clean} subsystem work?",
            f"The {name_clean} subsystem is implemented in `{file_path}`. {body_clean}",
            "factual_recall", "architecture", src
        ))

    # Memory architecture
    memory_match = re.search(r"## Memory Architecture(.+?)(?=\n##|\Z)", text, re.DOTALL)
    if memory_match:
        layers = re.findall(
            r"\|\s*\*\*(\w+)\*\*\s*\|\s*(\w+)\s*\|\s*`(.+?)`\s*\|\s*(.+?)\s*\|",
            memory_match.group(1)
        )
        if layers:
            mem_text = "; ".join(f"{l[0]} ({l[1]}): {l[3].strip()}" for l in layers)
            pairs.append(make_pair(
                "What is GAIA's memory architecture?",
                f"GAIA has {len(layers)} memory layers: {mem_text}.",
                "factual_recall", "architecture", src
            ))

    # MCP tools
    mcp_match = re.search(r"## MCP Tool System(.+?)(?=\n##|\Z)", text, re.DOTALL)
    if mcp_match:
        pairs.append(make_pair(
            "How does GAIA's tool system work?",
            "GAIA uses MCP (Model Context Protocol) via JSON-RPC 2.0 on gaia-mcp:8765. "
            "It provides 70+ tools across categories: File I/O, Shell, Memory, Knowledge Bases, "
            "Fragments, Study/LoRA, Web, Kanka.io, NotebookLM, Audio, and Diagnostics. "
            "Sensitive tools trigger a human-in-the-loop approval workflow with challenge codes.",
            "factual_recall", "architecture", src
        ))

    # Cascade routing
    pairs.append(make_pair(
        "How does cascade routing work in GAIA?",
        "Nano (0.5B) classifies prompts as SIMPLE or COMPLEX in ≤32 tokens at temp 0.1. "
        "SIMPLE → Lite/Operator handles directly. COMPLEX → escalated to Prime/Thinker. "
        "Additional heuristic escalation via _assess_complexity() checks for code, philosophy, "
        "and system keywords. Env overrides: GAIA_FORCE_THINKER=1, GAIA_FORCE_OPERATOR=1.",
        "factual_recall", "architecture", src
    ))

    # Promotion process
    pairs.append(make_pair(
        "How does GAIA's promotion process work?",
        "Candidate-only development → Pre-flight checks → Grammar tests (ruff/mypy/pytest in Docker) "
        "→ 16-test cognitive smoke battery → Promote in dependency order (gaia-common → gaia-core → "
        "gaia-web → gaia-mcp → gaia-study → gaia-orchestrator) → Post-promotion health → "
        "Session sanitization → Dev journal → Flatten SOA → Git commit/push. "
        "Master script: ./scripts/promote_pipeline.sh",
        "factual_recall", "architecture", src
    ))

    # Immune System
    pairs.append(make_pair(
        "How does GAIA's immune system work?",
        "The Immune System has four layers: 1) Sovereign Shield — py_compile gate on all .py writes, "
        "2) gaia-doctor — health polling with tiered remediation (restart, alarm, circuit breaker), "
        "3) HA Cross-Stack Healing — ruff auto-fix then LLM-powered structural surgery, "
        "4) Dissonance Probe — SHA-256 hash comparison detecting live/candidate drift. "
        "States: STABLE (≤2), MINOR NOISE (≤8), IRRITATED (≤25), CRITICAL (>25).",
        "factual_recall", "architecture", src
    ))

    # Circuit breaker
    pairs.append(make_pair(
        "What is the circuit breaker and when does it trigger?",
        "HEALING_REQUIRED.lock in /shared/ is the circuit breaker. It's created when a fatal loop "
        "threshold is reached. AgentCore checks at turn start and refuses to process if the lock "
        "exists. Manual intervention is required to clear the lock after fixing the underlying issue.",
        "factual_recall", "architecture", src
    ))

    # Speculative Nano-First
    pairs.append(make_pair(
        "What is the Speculative Nano-First pipeline?",
        "Before AgentCore runs, a speculative Nano-First reflex fires: Nano (0.5B) generates an "
        "immediate answer in ~0.24s, streamed to Discord/client instantly. The typing indicator "
        "shows for deeper reasoning. AgentCore runs in parallel and only emits a 'Refinement' "
        "if Prime adds meaningful new content. This gives sub-second response times for simple queries.",
        "factual_recall", "architecture", src
    ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Conversation Examples extractors
# ──────────────────────────────────────────────────────────────────────

def extract_conversation_pairs(conv_path: Path) -> list[dict]:
    """Extract personality/style pairs from conversation examples."""
    pairs = []
    if not conv_path.exists():
        return pairs

    text = conv_path.read_text()
    src = conv_path.name

    # Extract user/GAIA exchanges
    exchanges = re.findall(
        r"### User\s*\n(.+?)\n\s*### GAIA\s*\n(.+?)(?=\n### User|\n---|\Z)",
        text, re.DOTALL
    )

    for user_msg, gaia_msg in exchanges:
        user_msg = user_msg.strip()
        gaia_msg = gaia_msg.strip()
        # Remove duplicate responses (some examples have doubled text)
        if len(gaia_msg) > 20:
            half = len(gaia_msg) // 2
            if gaia_msg[:half] == gaia_msg[half:half*2]:
                gaia_msg = gaia_msg[:half]
        # Remove any stray markdown headers that leaked in
        gaia_msg = re.sub(r"\n### .*", "", gaia_msg).strip()
        # Skip if either side is too short or contains markdown artifacts
        if len(user_msg) < 5 or len(gaia_msg) < 20:
            continue
        if "### User" in gaia_msg or "### GAIA" in gaia_msg:
            continue

        pairs.append(make_pair(
            user_msg[:200],
            gaia_msg[:500],
            "conversation", "personality", src,
            weight=0.8
        ))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Hardcoded diagnostic pairs (common troubleshooting)
# ──────────────────────────────────────────────────────────────────────

def generate_diagnostic_pairs() -> list[dict]:
    """Generate common diagnostic Q&A pairs from known patterns."""
    return [
        make_pair(
            "gaia-web is returning 500 errors. How should I diagnose this?",
            "Check gaia-doctor's alarms at GET /alarms. The immune system's MRI scan will show "
            "structural issues. Common causes: 1) Python bytecache stale after code change — restart "
            "container, 2) Missing import — check ruff F821, 3) Mount path issue — verify docker-compose "
            "volumes. You can also run: docker compose exec gaia-web python -m pytest to check for test failures.",
            "diagnostic", "self-repair", "hardcoded"
        ),
        make_pair(
            "What does HEALING_REQUIRED.lock mean?",
            "The circuit breaker has been tripped. AgentCore refuses to process turns when "
            "/shared/HEALING_REQUIRED.lock exists. This happens when a fatal loop threshold is reached. "
            "Manual intervention required to clear the lock after fixing the underlying issue.",
            "diagnostic", "self-repair", "hardcoded"
        ),
        make_pair(
            "A service keeps restarting. What's happening?",
            "gaia-doctor has an auto-restart circuit breaker: max 2 restarts per 30-minute window per "
            "service. If a service keeps failing, doctor writes an alarm to /shared/doctor/alarms.json. "
            "Check the structural audit — doctor runs ast.parse on all .py files before restarting. "
            "If the code has syntax errors, it won't restart until the code is fixed.",
            "diagnostic", "self-repair", "hardcoded"
        ),
        make_pair(
            "How do I check GAIA's health status?",
            "Each service exposes a /health endpoint. gaia-doctor polls all services every 15 seconds. "
            "You can check: 1) GET /alarms on gaia-doctor:6419 for active alarms, "
            "2) The immune system status in the dashboard, "
            "3) docker compose ps to see container states, "
            "4) dozzle at port 9999 for real-time log viewing.",
            "diagnostic", "self-repair", "hardcoded"
        ),
        make_pair(
            "How do I promote candidate code to production?",
            "Use the promotion pipeline: ./scripts/promote_pipeline.sh (dry-run with --dry-run). "
            "It runs pre-flight checks, grammar tests (ruff/mypy/pytest in Docker), the 16-test "
            "cognitive smoke battery, promotes in dependency order (gaia-common → gaia-core → gaia-web "
            "→ gaia-mcp → gaia-study → gaia-orchestrator), verifies post-promotion health, sanitizes "
            "sessions, writes a dev journal, flattens SOA, and commits.",
            "procedural", "self-repair", "hardcoded"
        ),
        make_pair(
            "Why isn't my code change taking effect?",
            "Python bytecache means a container restart is needed after source file changes. "
            "For most services: docker restart <service-name>. "
            "For gaia-doctor specifically: code is COPYed in Dockerfile, not volume-mounted, "
            "so it requires docker compose build gaia-doctor && docker compose up -d gaia-doctor. "
            "Also ensure gaia-common changes are synced to both production and candidate paths.",
            "diagnostic", "self-repair", "hardcoded"
        ),
        make_pair(
            "How do I add a new tool to GAIA?",
            "Tools are defined in gaia-mcp/gaia_mcp/tools.py. Add a handler function and register "
            "it in the AVAILABLE_TOOLS dict. Sensitive tools (write, shell, etc.) need approval workflow — "
            "add them to the approval-required list. The Blast Shield provides pre-flight safety checks "
            "for dangerous operations. After adding, restart gaia-mcp and update the tool selector's "
            "training data if needed.",
            "procedural", "self-repair", "hardcoded"
        ),
        make_pair(
            "What happens during GAIA's sleep cycle?",
            "The sleep task scheduler runs priority-based autonomous maintenance: "
            "P1: auto_as_built_update, conversation_curation. "
            "P2: samvega_introspection. "
            "P3: blueprint_validation, code_evolution, promotion_readiness, initiative_cycle. "
            "P4: code_review, knowledge_research. "
            "P5: wiki_doc_regen, adversarial_resilience_drill. "
            "Sleep state transitions: ACTIVE → DROWSY → ASLEEP → WAKING.",
            "factual_recall", "architecture", "hardcoded"
        ),
        make_pair(
            "How does GAIA handle GPU sharing?",
            "gaia-orchestrator manages GPU lifecycle. gaia-prime (inference) and gaia-study (training) "
            "share a single GPU. When gaia-core enters sleep, it notifies orchestrator via POST /gpu/sleep, "
            "releasing the GPU for study tasks. On wake, POST /gpu/wake triggers GPU reclamation. "
            "gaia-core is deliberately CPU-only to avoid blocking cognition during GPU handoffs.",
            "factual_recall", "architecture", "hardcoded"
        ),
        make_pair(
            "How does the Sovereign Shield protect against bad writes?",
            "The Sovereign Shield in gaia-mcp/gaia_mcp/tools.py runs py_compile on all .py file writes "
            "(ai_write, write_file, replace). If compilation fails, it raises ValueError('Sovereign Shield: ...') "
            "and refuses the write. This prevents GAIA from introducing syntax errors during self-repair. "
            "Additionally, it blocks writes to production directories unless BREAKGLASS_EMERGENCY=1 is set.",
            "factual_recall", "architecture", "hardcoded"
        ),
        make_pair(
            "What is the HA (High Availability) mesh?",
            "The HA mesh provides service redundancy. ServiceClient retries with backoff and automatic "
            "failover to candidate services. HealthWatchdog polls every 30s with a 2-failure threshold. "
            "States: ACTIVE, DEGRADED, FAILOVER_ACTIVE, FAILED. Candidate services mirror production "
            "with +1 ports (e.g., gaia-core:6415 → gaia-core-candidate:6416). "
            "Maintenance mode via /shared/ha_maintenance flag disables failover.",
            "factual_recall", "architecture", "hardcoded"
        ),
        make_pair(
            "What is GAIA's full name?",
            "GAIA stands for General Artisanal Intelligence Architecture.",
            "factual_recall", "identity", "hardcoded"
        ),
        make_pair(
            "How many services does GAIA have?",
            "GAIA has 11 services: gaia-core (The Brain), gaia-web (The Face), gaia-prime (The Voice), "
            "gaia-mcp (The Hands), gaia-study (The Subconscious), gaia-audio (The Ears & Mouth), "
            "gaia-orchestrator (The Coordinator), gaia-doctor (The Immune System), gaia-wiki (The Library), "
            "and dozzle (The X-Ray). Plus candidate (HA) mirrors for key services.",
            "factual_recall", "architecture", "hardcoded"
        ),
        make_pair(
            "What vector embedding model does GAIA use?",
            "GAIA uses all-MiniLM-L6-v2 for vector embeddings, producing 384-dimensional vectors. "
            "Documents are chunked into 512-token segments. The vector index is JSON-persisted and "
            "managed by gaia-study as the sole writer.",
            "factual_recall", "architecture", "hardcoded"
        ),
        make_pair(
            "How does GAIA's approval workflow work for sensitive tools?",
            "When a sensitive tool (ai_write, write_file, run_shell, memory_rebuild_index) is invoked, "
            "ApprovalStore generates a 5-character challenge code. The human must provide the reversed "
            "code to approve. Codes expire after 900 seconds (configurable via MCP_APPROVAL_TTL). "
            "This is the human-in-the-loop safety gate for destructive operations.",
            "factual_recall", "architecture", "hardcoded"
        ),
    ]


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate self-model QLoRA corpus")
    parser.add_argument(
        "--output-dir",
        default=str(KNOWLEDGE / "curricula" / "self-model"),
        help="Output directory for training files"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    print("=" * 60)
    print("GAIA Self-Model QLoRA Corpus Generator")
    print("=" * 60)

    # Collect all pairs
    all_pairs: list[dict] = []
    category_counts: dict[str, int] = {}

    extractors = [
        ("Blueprint YAML", lambda: extract_blueprint_yaml_pairs(BLUEPRINTS)),
        ("Blueprint MD", lambda: extract_blueprint_md_pairs(BLUEPRINTS)),
        ("AS_BUILT", lambda: extract_as_built_pairs(SYSTEM_REF / "AS_BUILT_LATEST.md")),
        ("Dev Journals", lambda: extract_dev_journal_pairs(DEV_NOTEBOOKS)),
        ("Samvega", lambda: extract_samvega_pairs(SAMVEGA)),
        ("Thought Seeds", lambda: extract_seed_pairs(SEEDS)),
        ("Identity", lambda: extract_identity_pairs(SYSTEM_REF)),
        ("Gap Audit", lambda: extract_gap_pairs(GAP_AUDIT)),
        ("CLAUDE.md", lambda: extract_claude_md_pairs(CLAUDE_MD)),
        ("Conversation Examples", lambda: extract_conversation_pairs(CONVERSATION_EXAMPLES)),
        ("Diagnostic (hardcoded)", generate_diagnostic_pairs),
    ]

    for name, extractor in extractors:
        try:
            pairs = extractor()
            print(f"  {name}: {len(pairs)} pairs")
            all_pairs.extend(pairs)
            for p in pairs:
                cat = p["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1
        except Exception as e:
            print(f"  {name}: ERROR — {e}")

    print(f"\nTotal pairs: {len(all_pairs)}")

    # Deduplicate by instruction text
    seen_instructions: set[str] = set()
    unique_pairs: list[dict] = []
    for p in all_pairs:
        key = p["instruction"].lower().strip()
        if key not in seen_instructions:
            seen_instructions.add(key)
            unique_pairs.append(p)

    deduped = len(all_pairs) - len(unique_pairs)
    if deduped:
        print(f"Deduplicated: {deduped} duplicates removed")
        all_pairs = unique_pairs

    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - VALIDATION_SPLIT))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Validation: {len(val_pairs)}")

    # Write JSONL files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Compute hash
    all_text = "".join(json.dumps(p, sort_keys=True) for p in all_pairs)
    data_hash = hashlib.sha256(all_text.encode()).hexdigest()[:16]

    # Recount categories after dedup
    category_counts = {}
    pair_type_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for p in all_pairs:
        cat = p["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        pt = p["pair_type"]
        pair_type_counts[pt] = pair_type_counts.get(pt, 0) + 1
        src = p["source_file"]
        source_counts[src] = source_counts.get(src, 0) + 1

    # Write metadata
    metadata = {
        "adapter_name": "self-model",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "duplicates_removed": deduped,
        "categories": category_counts,
        "pair_types": pair_type_counts,
        "top_sources": dict(sorted(source_counts.items(), key=lambda x: -x[1])[:15]),
        "data_hash": data_hash,
    }

    meta_path = output_dir / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFiles written:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {meta_path}")
    print(f"\nCategories: {json.dumps(category_counts, indent=2)}")
    print(f"Pair types: {json.dumps(pair_type_counts, indent=2)}")
    print(f"Data hash: {data_hash}")
    print("=" * 60)


if __name__ == "__main__":
    main()
