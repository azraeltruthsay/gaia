# Council Chamber Protocol

## Quick Start (L1 Context)
At session start, read `GAIA_CHORD_MANIFEST.aaak` first (~30 tokens). This gives you the full system state in one glance. Only read the full Chamber/TODO if the manifest indicates something changed.

The manifest is gitignored (regenerated locally) — if it's missing in a fresh checkout, run `python scripts/chord_sync.py` to create it. The post-commit hook also refreshes it after every commit.

## Full Sync (L2/L3)
- Read `COUNCIL_CHAMBER.md` for strategic dispatches and Advisor notes
- Read `knowledge/Dev_Notebook/TODO.md` for granular task tracking
- Use `mempalace search "topic"` for historical context (L3)

## Protocol
- Before starting implementation work, verify the Chamber hasn't been updated since your last read
- Use `/chord` to do a full synchronization (Chamber + Gemini tmux + Dev_Notebook scan)
- After completing work, run `chord_sync.py` to update the manifest
