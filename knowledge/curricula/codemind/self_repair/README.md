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

## ../investigation_practice/ (generated, git-ignored)

`scripts/investigation_practice.py` (run inside gaia-core): one exercise per log call with a literal
message in a candidate function of at most 150 lines. Evidence quotes the line (plus one or two
unrelated log lines, with the symptom naming the one that matters); the gold actions SEARCH its
fixed words, READ the hit and PLACE the function. Each is run through the real
`issue_investigation.investigate()` with a scripted mind, so every reply is exactly what she sees.
2026-11-25: 940 exercises (765 train / 175 eval, held out by source file); 100 first search the
line's data and recover.
