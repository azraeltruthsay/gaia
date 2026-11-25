# Self-repair examples

One file per bug carried through her self-repair loop (knowledge/blueprints/SELF_REPAIR_LOOP.md),
in the formats the loop uses today: the investigation's SEARCH/READ/PLACE steps, a regression test
that fails on the original, and the patch. Each holds the gold version and what she did instead.
They are training material for a CodeMind adapter for Prime; the older files in ../ (train*.jsonl)
are whole-file lint fixes and don't match these formats.

## history/

Made by `scripts/self_repair_from_history.py` from past fixes: commits that fixed a bug bead in
one or two candidate source files and added tests (60 of 591 commits since 2026-03-29). Each has
the bead as the issue, the changed function as the place, the tests the commit added, and its
diff. The investigation is reconstructed, not observed, and only kept when each step follows from
the last (a distinctive lead from the bead, hit inside the place or in a function that calls it,
place <= 150 lines): 12 of 60. `issue_names_place` marks beads that already give the place away.
The added tests weren't re-run against the parent (`test_verified: false`).
