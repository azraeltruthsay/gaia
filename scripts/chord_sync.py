#!/usr/bin/env python3
"""
Chord Sync — Generate GAIA_CHORD_MANIFEST.aaak from system state.

Parses COUNCIL_CHAMBER.md, TODO.md, and training/orchestrator status
to produce a compact AAAK symbolic manifest (~30 tokens) that both
Claude and Gemini can load instead of reading full documents.

Usage:
    python scripts/chord_sync.py
    # Or as a git post-commit hook / session start hook

Schema:
    [STATE:mode] [TRAIN:tier:phase:%] [IDENTITY:tier:status]
    [SHIELD:status:sha] [GATE:next_task] [SENSORY:mode]
"""

import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / "GAIA_CHORD_MANIFEST.aaak"
TODO_PATH = PROJECT_ROOT / "knowledge" / "Dev_Notebook" / "TODO.md"
CHAMBER_PATH = PROJECT_ROOT / "COUNCIL_CHAMBER.md"


def get_lifecycle_state() -> str:
    """Query orchestrator for current lifecycle state."""
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:6410/lifecycle/state", timeout=3)
        data = json.loads(resp.read())
        return data.get("state", "unknown").upper()
    except Exception:
        return "OFFLINE"


def get_training_status() -> str:
    """Query gaia-study for training status."""
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8766/study/adaptive-train/status", timeout=3)
        data = json.loads(resp.read())
        status = data.get("status", "idle")
        if status == "running":
            phase = data.get("current_phase", "?")
            passed = len(data.get("globally_passed", []))
            return f"RUNNING:P{phase}:{passed}skills"
        return status.upper()
    except Exception:
        return "OFFLINE"


def get_last_commit() -> str:
    """Get short hash of last commit."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:7]
    except Exception:
        pass
    return "unknown"


def parse_todo_state() -> dict:
    """Extract key state from TODO.md."""
    state = {"shield": "UNKNOWN", "identity": "UNKNOWN", "gate": "NONE"}
    try:
        text = TODO_PATH.read_text()
        # Check shield status
        if "Blast Shield" in text:
            state["shield"] = "HARDENED"
        # Check for pending items (the next gate)
        pending = re.findall(r'- \[ \] \*\*(.+?)\*\*', text)
        if pending:
            state["gate"] = pending[0][:30].replace(" ", "_")
        # Check identity
        if "gemma" in text.lower() or "26b" in text.lower():
            state["identity"] = "SOVEREIGN_GEMMA"
        elif "identity-baked" in text.lower() or "v2" in text.lower():
            state["identity"] = "BAKED_V2"
    except Exception:
        pass
    return state


def get_tier_matrix() -> str:
    """Query consciousness matrix for tier states (Gemma 4 Chord)."""
    try:
        import urllib.request
        # We query the orchestrator's matrix status
        resp = urllib.request.urlopen("http://localhost:6410/consciousness/matrix", timeout=3)
        data = json.loads(resp.read())
        parts = []
        # Mapping to the Gemma 4 ecosystem
        for tier_id, label in [("nano", "E2B"), ("core", "E4B"), ("prime", "A4B")]:
            if tier_id in data:
                actual = data[tier_id].get("actual", "?")[:4].upper()
                parts.append(f"{label}:{actual}")
        return " ".join(parts)
    except Exception:
        # Static fallback if orchestrator is offline
        return "E2B:OFF E4B:OFF A4B:OFF"


def generate_manifest():
    """Generate the AAAK manifest."""
    state = get_lifecycle_state()
    train = get_training_status()
    todo = parse_todo_state()
    commit = get_last_commit()
    tiers = get_tier_matrix()

    lines = [
        f"[STATE:{state}] [TIERS:{tiers}] [TRAIN:{train}]",
        f"[SHIELD:{todo['shield']}:{commit}] [GATE:{todo['gate']}]",
        f"[IDENTITY:{todo['identity']}] [COMMIT:{commit}]",
    ]

    manifest = "\n".join(lines) + "\n"
    MANIFEST_PATH.write_text(manifest)
    print(f"Manifest written to {MANIFEST_PATH}")
    print(manifest)


if __name__ == "__main__":
    generate_manifest()
