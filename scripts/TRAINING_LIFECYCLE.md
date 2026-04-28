# Training Lifecycle Policy

After every training run, the model that takes over is **not** weight-compatible
with the model that just left. KV cache, session history bias, and the baked
identity prefix from the previous model silently propagate into the new one's
behavior unless explicitly invalidated.

## Mandatory post-training reset

Run after every successful training run for the trained tier:

```bash
python scripts/post_training_reset.py --tier <core|prime|nano>
# or for everything:
python scripts/post_training_reset.py --all
```

What it does (idempotent, safe to re-run):

1. Archives `/shared/sessions.json` → `/shared/sessions.archived/YYYY-MM-DD/`
2. Distills a memory journal from the archive (uses a non-training tier
   like Groq or pre-swap Core to avoid self-referential contamination)
3. Invalidates KV cache:
   - clears `/shared/kvcache/<tier>/handoff_context.json`
   - clears `/shared/kvcache/<tier>/core_checkpoint`
   - POSTs `/cache/invalidate` to each reachable engine
4. Regenerates `identity_prefix.pt` for each tier with the new weights
5. Optionally clears session vector indexes (new thresholds filter old noise)
6. Writes a dev journal entry noting which model is now live

## When to skip

- **Re-runs of the same training data** — KV/sessions are already aligned;
  archive churn is wasteful. Run only after weights actually changed.
- **Adapter-only training** (LoRA without merge) — adapters layer onto the
  same base, so KV state still aligns. Skip unless you're swapping the base.

## Why this matters

Past incidents:
- Core trained with new identity but inherited the old KV cache → assistant
  introduced itself with the previous identity for the first ~10 turns
  before the cache rolled over.
- Session history retrieved high-similarity matches from pre-training
  conversations, anchoring the new model to old context. Looked like the
  training "didn't take" but was actually retrieval contamination.

Both are silent until you notice the regression. Better to always reset.

## Adding to your training script

Append to the success path:

```python
log.info("Next: run scripts/post_training_reset.py --tier <name>")
```

Don't auto-invoke — the operator may have parallel work that depends on
the existing sessions/KV. Make it the explicit next manual step.
