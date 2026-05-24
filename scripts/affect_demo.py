"""Lived-session demo for the affect system (GAIA_Project-usv).

Exercises every shipped surface against a real KnowledgeGraph in a
scratch file. Doesn't touch any LLM; just demonstrates the affect
machinery end-to-end:

  Scene 1 — base persona traits laid down in actuality
  Scene 2 — turn-intake detects a coding_debug context, activates it
  Scene 3 — that context's overlay shadows base playfulness, ups caution
  Scene 4 — record a transient feeling (irritation from a failing test)
  Scene 5 — affect_state_lines + affect_inference_params show the impact
  Scene 6 — apply_affect_modulation shapes a baseline (temp=0.7, max=1024)
  Scene 7 — theory-of-mind: GAIA forms a belief about Azrael's state
  Scene 8 — fast-forward: feeling decays half-life; modulation shifts
  Scene 9 — deactivate context, snapshot returns to actuality only

Run inside the gaia-core container (so gaia_common + gaia_core resolve):

  docker compose exec -T gaia-core python /gaia/GAIA_Project/scripts/affect_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


def banner(s: str) -> None:
    print()
    print("─" * 68)
    print(s)
    print("─" * 68)


def fmt_snapshot(snap: dict, label: str) -> None:
    print(f"  [{label}] active_context={snap.get('active_context')!r}")
    for axis in ("traits", "feels", "drives", "curious_about", "tired_of"):
        d = snap.get(axis) or {}
        if d:
            kvs = ", ".join(f"{k}={v:.2f}" for k, v in sorted(d.items()))
            print(f"    {axis}: {kvs}")


def main() -> int:
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    from gaia_common.utils.affect_kg import AffectKG
    from gaia_core.cognition import affect_runtime

    db = Path(tempfile.gettempdir()) / "gaia_affect_demo.sqlite"
    if db.exists():
        db.unlink()
    print(f"scratch kg: {db}")

    kg = KnowledgeGraph(db_path=str(db))
    affect = AffectKG(kg)
    affect_runtime.reset_for_tests(affect)

    # ─── Scene 1 ───────────────────────────────────────────────────
    banner("Scene 1 — Lay down base persona traits in actuality")
    affect.record_trait("curiosity", 0.85)
    affect.record_trait("warmth", 0.7)
    affect.record_trait("caution", 0.4)
    affect.record_trait("logic_priority", 0.55)
    affect.record_trait("playfulness", 0.45)
    fmt_snapshot(affect.flatten_current_affect(), "baseline")

    # ─── Scene 2 ───────────────────────────────────────────────────
    banner("Scene 2 — Turn intake: 'Help me debug this traceback'")
    user_input = "Help me debug this traceback from the deploy script"
    activated = affect_runtime.activate_detected_contexts(
        user_input, session_id="demo_session", ttl_seconds=3600,
    )
    print(f"  detected & activated: {activated}")

    # ─── Scene 3 ───────────────────────────────────────────────────
    banner("Scene 3 — Apply coding_debug trait deltas to the overlay")
    # Coding debug → less playful, more cautious, more logic-driven.
    affect.record_trait("playfulness", 0.10, world="ctx_coding_debug")
    affect.record_trait("caution", 0.80, world="ctx_coding_debug")
    affect.record_trait("logic_priority", 0.90, world="ctx_coding_debug")
    fmt_snapshot(
        affect.flatten_current_affect(active_context="ctx_coding_debug"),
        "with coding_debug active",
    )

    # ─── Scene 4 ───────────────────────────────────────────────────
    banner("Scene 4 — A specific failure: feel irritation, fatigue")
    affect.record_feeling("irritation", 0.65)
    affect.record_feeling("fatigue", 0.3)
    affect.record_curious_about("deploy_script_failure", 0.8)
    fmt_snapshot(
        affect.flatten_current_affect(active_context="ctx_coding_debug"),
        "after failure event",
    )

    # ─── Scene 5 ───────────────────────────────────────────────────
    banner("Scene 5 — Render affect into prompt + inference params")
    snap = affect.flatten_current_affect(active_context="ctx_coding_debug")
    lines = affect_runtime.affect_state_lines(snap)
    print("  affect_state_lines() →")
    for L in lines:
        print(f"    {L}")
    params = affect_runtime.affect_inference_params(snap)
    print("  affect_inference_params() →")
    for k, v in params.items():
        print(f"    {k}: {v}")

    # ─── Scene 6 ───────────────────────────────────────────────────
    banner("Scene 6 — apply_affect_modulation on baseline (0.7, 1024)")
    t, m, dbg = affect_runtime.apply_affect_modulation(
        base_temperature=0.7, base_max_tokens=1024, snapshot=snap,
    )
    print(f"  temperature: 0.7 → {t:.3f}")
    print(f"  max_tokens:  1024 → {m}")
    print(f"  reasons:     {dbg['reasons']}")
    print(f"  style_hint:  {dbg.get('style_hint')}")

    # ─── Scene 7 ───────────────────────────────────────────────────
    banner("Scene 7 — Theory of mind: belief about Azrael")
    affect.record_belief_about("azrael", "current_mood", "focused", 0.8)
    affect.record_belief_about("azrael", "frustration_at", "deploy_script", 0.6)
    print(f"  affect.belief_about('azrael') →")
    for attr, info in affect.belief_about("azrael").items():
        print(f"    {attr}: value={info['value']!r} conf={info['confidence']:.2f}")
    # Modality firewall: self snapshot is unaffected
    self_snap = affect.flatten_current_affect(active_context="ctx_coding_debug")
    print(f"  self.feels (firewall check, no leakage): "
          f"{sorted(self_snap['feels'].keys())}")

    # ─── Scene 8 ───────────────────────────────────────────────────
    banner("Scene 8 — Fast-forward 12h (irritation halflife). Snapshot.")
    future = datetime.now(timezone.utc) + timedelta(hours=12)
    snap_future = affect.flatten_current_affect(
        active_context="ctx_coding_debug", now=future,
    )
    fmt_snapshot(snap_future, "after 12h")
    print("  irritation should be ~half of recorded value (decay placeholder)")

    # ─── Scene 9 ───────────────────────────────────────────────────
    banner("Scene 9 — Deactivate coding_debug, back to baseline traits")
    affect.deactivate_context("coding_debug")
    fmt_snapshot(affect.flatten_current_affect(), "after deactivate")

    print()
    print("Demo complete. Cleaning up scratch KG.")
    db.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
