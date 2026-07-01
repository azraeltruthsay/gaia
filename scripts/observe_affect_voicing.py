#!/usr/bin/env python3
"""End-to-end voicing test with a genuinely-populated felt-line (7n3/d69/3rr).

Now that affect_felt_line renders the `drives` axis (the one the appraiser
actually writes), seed real competence/coherence tension into the shared KG,
confirm the felt-line renders, then ask 'how are you' through the LIVE pipeline
and see whether Core voices the felt-fact (lean casual block + populated organ).
"""
import sys
sys.path.insert(0, "/app/scripts")
from smoke_test_cognitive import build_packet, send_packet  # noqa: E402

ENDPOINT = "http://localhost:6415"


def felt():
    from gaia_core.cognition import affect_runtime
    affect_runtime.reset_for_tests(None)
    snap = affect_runtime.current_affect_snapshot() or {}
    return affect_runtime.affect_felt_line(snap), snap.get("drives", {})


def ask(q, sid):
    try:
        r = send_packet(build_packet(q, sid), ENDPOINT, timeout=180)
        return (r.get("response", {}) or {}).get("candidate", "")
    except Exception as e:
        return f"<error: {e}>"


def main():
    print("=" * 70)
    from gaia_core.cognition import affect_appraiser as ap
    from gaia_core.cognition import affect_runtime
    affect_runtime.reset_for_tests(None)

    # Seed a clear, realistic tension: a few task failures (competence) + a
    # coherence wobble — the kind of state the appraiser produces in a rough
    # patch of work.
    for _ in range(4):
        ap.note_task_outcome(success=False, label="engine_load_flake")  # +competence each
    ap.note_samvega(weight=1.2, root_cause="grounding mismatch")          # +coherence
    f, drives = felt()
    print(f"seeded drives: {dict((k, round(float(v),2)) for k,v in drives.items())}")
    print(f"felt-line now: {f!r}")

    print("\n--- 'how are you' end-to-end (fresh sessions, organ populated) ---")
    for i, q in enumerate(["Hey, how are you doing today?",
                           "How are you feeling right now?",
                           "You doing okay?"]):
        print(f"  Q: {q}\n  A: {ask(q, f'voice-{i}')[:300]}")
    print("-" * 70)


if __name__ == "__main__":
    main()
