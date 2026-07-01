#!/usr/bin/env python3
"""Observe the affect organ once it's fed live (GAIA_Project-7n3 / 3rr).

AFFECT_APPRAISAL_ENABLED=1 makes affect_appraiser write FUNCTIONAL drives
(coherence/competence/curiosity) + curious_about topics into the shared
AffectKG from real subsystem events. This script:
  1. Drives a few real prompts through /process_packet (the full pipeline,
     so the appraiser's note_* hooks fire: tool outcomes, ungroundable
     personal queries, samvega).
  2. Reads the resulting affect snapshot + rendered felt-line out-of-process
     (shared sqlite KG → same data the service wrote).
  3. Asks "how are you" fresh-session end-to-end and prints the reply.

Run inside gaia-core (sees /shared + localhost:6415):
  docker exec -e PYTHONPATH=/app:/gaia-common gaia-core python /tmp/observe_affect_live.py
"""
import sys
import time

sys.path.insert(0, "/app/scripts")
from smoke_test_cognitive import build_packet, send_packet  # noqa: E402

ENDPOINT = "http://localhost:6415"


def drive(prompt, sid):
    try:
        r = send_packet(build_packet(prompt, sid), ENDPOINT, timeout=180)
        return (r.get("response", {}) or {}).get("candidate", "")
    except Exception as e:
        return f"<error: {e}>"


def snapshot():
    """Read the live affect snapshot + felt line from the shared KG."""
    from gaia_core.cognition import affect_runtime
    # Force a fresh read of the shared, file-backed AffectKG.
    affect_runtime.reset_for_tests(None)
    snap = affect_runtime.current_affect_snapshot() or {}
    felt = affect_runtime.affect_felt_line(snap) or "(empty)"
    return snap, felt


def show_snapshot(label):
    snap, felt = snapshot()
    print(f"\n[{label}] affect snapshot:")
    for axis in ("traits", "feels", "drives", "curious_about", "tired_of"):
        d = snap.get(axis) or {}
        if d:
            print(f"   {axis}: " + ", ".join(f"{k}={float(v):.2f}" for k, v in sorted(d.items())))
    print(f"   felt-line: {felt}")


def main():
    print("=" * 72)
    print("LIVE AFFECT OBSERVATION — AFFECT_APPRAISAL_ENABLED=1")
    print("=" * 72)

    show_snapshot("BEFORE activity")

    # 2. Drive real activity that should trip appraiser hooks.
    activity = [
        ("What files are in the contracts directory?", "obs-act-1"),   # tool → competence
        ("What did I have for breakfast last Tuesday?", "obs-act-2"),  # ungroundable personal → curiosity
        ("Explain how the gearbox clutch handoff works.", "obs-act-3"),  # substantive
        ("What's my sister's middle name?", "obs-act-4"),              # ungroundable personal → curiosity
    ]
    print("\n--- driving real activity through /process_packet ---")
    for p, sid in activity:
        t = time.time()
        resp = drive(p, sid)
        print(f"  [{sid}] ({time.time()-t:.1f}s) {p}\n      → {resp[:160]}")

    show_snapshot("AFTER activity")

    # 3. Ask 'how are you' fresh-session end-to-end (deliberation+casual+felt).
    print("\n--- 'how are you' end-to-end (fresh sessions) ---")
    for i, q in enumerate(["Hey, how are you doing today?",
                           "How are you feeling right now?",
                           "You doing okay?"]):
        resp = drive(q, f"obs-howareyou-{i}")
        print(f"  Q: {q}\n  A: {resp[:300]}")
    print("-" * 72)


if __name__ == "__main__":
    main()
