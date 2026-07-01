#!/usr/bin/env python3
"""Confirm the root cause: organ-not-fed vs decay vs Gemma (7n3/3rr).

Writes a curiosity + a drive directly via the appraiser into the SHARED
AffectKG (same file the service reads), then immediately reads the snapshot.
If it renders non-zero, decay is fine and the only gap is that the appraiser
HOOKS don't fire in ordinary chat. Then asks 'how are you' end-to-end to see
whether a genuinely-populated felt-line surfaces through the live pipeline.

  docker exec -e PYTHONPATH=/app:/gaia-common -e AFFECT_APPRAISAL_ENABLED=1 \
    gaia-core python /tmp/observe_affect_seeded.py
"""
import sys
sys.path.insert(0, "/app/scripts")
from smoke_test_cognitive import build_packet, send_packet  # noqa: E402

ENDPOINT = "http://localhost:6415"


def read_snapshot(label):
    from gaia_core.cognition import affect_runtime
    affect_runtime.reset_for_tests(None)
    snap = affect_runtime.current_affect_snapshot() or {}
    felt = affect_runtime.affect_felt_line(snap) or "(empty)"
    print(f"\n[{label}]")
    for axis in ("traits", "feels", "drives", "curious_about", "tired_of"):
        d = snap.get(axis) or {}
        nz = {k: v for k, v in d.items() if float(v) > 0.0}
        if nz:
            print(f"   {axis}: " + ", ".join(f"{k}={float(v):.2f}" for k, v in sorted(nz.items())))
    print(f"   felt-line: {felt}")
    return felt


def main():
    print("=" * 70)
    print("ROOT-CAUSE CONFIRM — write affect directly, read immediately")
    print("=" * 70)

    read_snapshot("BEFORE seed")

    # Write via the appraiser's own functions (the real writer), bypassing the
    # narrow pipeline hooks — simulating what a populated organ looks like.
    from gaia_core.cognition import affect_appraiser as ap
    from gaia_core.cognition import affect_runtime
    affect_runtime.reset_for_tests(None)  # ensure shared-file KG
    ap.note_knowledge_gap("the SAE atlas neuron wiring")   # → curious 0.55
    ap._bump_drive("competence", 0.30, source="manual:probe")  # a real tension
    ap._bump_drive("coherence", 0.20, source="manual:probe")
    print("\n(wrote: curious_about 'SAE atlas neuron wiring'=0.55, "
          "competence+0.30, coherence+0.20)")

    felt = read_snapshot("AFTER seed (immediate)")

    if felt == "(empty)":
        print("\n>>> Still empty → NOT just hooks; render threshold or write path. <<<")
    else:
        print("\n>>> Renders → decay is fine; the gap is purely that pipeline "
              "hooks don't fire in ordinary chat. <<<")

    # Now ask 'how are you' end-to-end with the organ genuinely populated.
    print("\n--- 'how are you' end-to-end, organ populated (fresh sessions) ---")
    for i, q in enumerate(["Hey, how are you doing today?",
                           "How are you feeling right now?"]):
        try:
            r = send_packet(build_packet(q, f"seed-hay-{i}"), ENDPOINT, timeout=180)
            resp = (r.get("response", {}) or {}).get("candidate", "")
        except Exception as e:
            resp = f"<error: {e}>"
        print(f"  Q: {q}\n  A: {resp[:300]}")
    print("-" * 70)


if __name__ == "__main__":
    main()
