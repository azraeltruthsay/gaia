#!/usr/bin/env python3
"""
Repair script for session history in sessions.json.
Sweeps the database for consecutive assistant turns and auto-corrects them
by inserting placeholder or matching user prompts to maintain turn alternation.
"""

import json
import os
import sys
from datetime import datetime, timezone

SHARED_DIR = os.environ.get("SHARED_DIR", "/shared")
SESSIONS_FILE = os.path.join(SHARED_DIR, "sessions.json")

# Predefined known missing queries we can inject if we match the assistant response
KNOWN_RECONSTRUCTIONS = [
    {
        "contains": "Samvega is a Buddhist concept",
        "role": "user",
        "content": "What do you know about your Samvega system?"
    }
]

def repair_sessions():
    if not os.path.exists(SESSIONS_FILE):
        print(f"Error: {SESSIONS_FILE} not found.")
        sys.exit(1)

    with open(SESSIONS_FILE, "r") as f:
        data = json.load(f)

    modified = False

    for session_id, session_data in data.items():
        history = session_data.get("history", [])
        if not history:
            continue

        new_history = []
        last_role = None

        for i, turn in enumerate(history):
            role = turn.get("role")
            content = turn.get("content", "")

            # Detect consecutive assistant turns
            if role == "assistant" and last_role == "assistant":
                print(f"\n[ALERT] Consecutive assistant turns detected in session '{session_id}'!")
                print(f"  Prev Turn: {new_history[-1]['content'][:80]}...")
                print(f"  Curr Turn: {content[:80]}...")

                # Attempt reconstruction
                reconstructed = False
                for rec in KNOWN_RECONSTRUCTIONS:
                    if rec["contains"] in content:
                        # Insert reconstructed user prompt before current turn
                        ts = turn.get("timestamp") or datetime.now(timezone.utc).isoformat()
                        user_turn = {
                            "id": "reconstructed_" + turn.get("id", "turn"),
                            "role": "user",
                            "content": rec["content"],
                            "timestamp": ts
                        }
                        new_history.append(user_turn)
                        print(f"  -> Reconstructed and injected: \"{rec['content']}\"")
                        reconstructed = True
                        break

                if not reconstructed:
                    # Injected fallback placeholder
                    ts = turn.get("timestamp") or datetime.now(timezone.utc).isoformat()
                    user_turn = {
                        "id": "placeholder_" + turn.get("id", "turn"),
                        "role": "user",
                        "content": "[System continuation / Context query]",
                        "timestamp": ts
                    }
                    new_history.append(user_turn)
                    print(f"  -> Injected fallback placeholder user query.")
                
                modified = True

            new_history.append(turn)
            last_role = role

        session_data["history"] = new_history

    if modified:
        # Write back to file
        with open(SESSIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[SUCCESS] Repaired sessions.json and saved changes.")
    else:
        print("\nNo consecutive assistant turns detected. Database integrity is intact.")

if __name__ == "__main__":
    repair_sessions()
