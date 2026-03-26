# Commands Page Audit — 2026-03-26

## Summary

Audited all 8 functional groups on the Commands page of Mission Control.
Tested live API endpoints, traced JS component wiring, and verified proxy routes.

**Found 7 bugs, fixed all 7.**

---

## Endpoint Test Results (All Responding)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /api/system/lifecycle/state` | OK | Returns state, tiers with vram_mb per tier |
| `GET /api/system/lifecycle/transitions` | OK | Returns array of trigger/target objects |
| `GET /api/system/lifecycle/history` | OK | Returns empty array (no transitions yet) |
| `GET /api/hooks/sleep/status` | OK | Returns state, seconds_in_state, auto_sleep_enabled |
| `GET /api/hooks/sleep/config` | OK | Returns auto_sleep_enabled, idle_threshold_minutes |
| `GET /api/hooks/sleep/wake-config` | OK | Returns discord_typing, workstation_activity |
| `GET /api/hooks/gpu/status` | OK | Returns gpu_state, gpu_prime_loaded, gpu_prime_status |
| `GET /api/chaos/config` | OK | Returns mode, drill_types, schedule_interval_hours |
| `GET /api/chaos/serenity` | OK | Returns serene, score, threshold |
| `GET /api/system/cognitive/status` | OK | Returns running, alignment, last_run |
| `GET /api/system/pipeline/status` | OK | Returns stages, alignment_status, failed_stages |
| `GET /api/system/doctor/status` | OK | Returns alarms, remediations, serenity, maintenance |
| `GET /api/system/irritations` | OK | Returns irritation array with time/service/pattern |
| `GET /api/system/dissonance` | OK | Returns vital_divergent + standard_divergent arrays |
| `GET /api/system/surgeon/config` | OK | Returns approval_required |
| `GET /api/system/surgeon/queue` | OK | Returns queue array |
| `GET /api/system/surgeon/history` | OK | Returns history array |
| `GET /api/system/training/progress` | OK | Returns manager + progress_file objects |
| `GET /api/system/maintenance/status` | OK | Returns active boolean |

---

## Bugs Found and Fixed

### Bug 1: Lifecycle VRAM bar always empty
**File:** `gaia-web/static/app.js` (lifecyclePanel.refresh)
**Problem:** JS read `data.vram_used_mb` from lifecycle state response, but the API only returns per-tier `vram_mb` values -- no top-level totals.
**Fix:** Compute `vramUsed` by summing `info.vram_mb` across all tiers. Fall back to API field if present.

### Bug 2: Sleep uptime always shows "--"
**File:** `gaia-web/static/app.js` (hooksPanel.refreshSleep)
**Problem:** JS read `data.uptime_seconds || data.uptime` but the `/sleep/status` response uses `seconds_in_state`. Also `cycle_count` doesn't exist; response has `phase` instead.
**Fix:** Added `data.seconds_in_state` as primary source for uptime. Fall back to `phase` for cycle display.

### Bug 3: GPU owner always shows "none", VRAM always "--"
**File:** `gaia-web/static/app.js` (hooksPanel.refreshGpu)
**Problem:** JS expected `owner`/`gpu_owner`/`vram_used_mb`/`used_mb`/`total_mb` fields, but `/gpu/status` returns `gpu_state`, `gpu_prime_loaded`, `gpu_prime_status`, `prime_reachable` -- completely different schema.
**Fix:** Use `gpu_state` for owner display. When VRAM numbers unavailable, build descriptive string from `gpu_prime_loaded` and `gpu_prime_status`.

### Bug 4: Dissonance count always zero
**File:** `gaia-web/static/app.js` (doctorPanel.pollDoctor, doctorPanel.toggleDissonance)
**Problem:** JS read `data.diverged` array, but `/dissonance` endpoint returns `vital_divergent` and `standard_divergent` as separate arrays. No `diverged` key exists.
**Fix:** Merge `vital_divergent` + `standard_divergent` into combined diverged array. Applied in both pollDoctor and toggleDissonance.

### Bug 5: Irritation timestamps not displayed
**File:** `gaia-web/static/index.html` (irritation template)
**Problem:** Template used `irr.timestamp` but API returns `irr.time`. Also template key used `irr.timestamp` causing dedup issues. Message field is `message` not `line`.
**Fix:** Changed to `(irr.time || irr.timestamp)` for both key and display. Added `irr.message` as fallback for display text.

### Bug 6: Remediation timestamps not displayed
**File:** `gaia-web/static/index.html` (remediation template)
**Problem:** Same as irritations -- `rem.timestamp` should be `rem.time`. Also some remediations have `success` boolean instead of `action`/`type`.
**Fix:** Changed to `(rem.time || rem.timestamp)`. Added `rem.success ? 'restart ok' : 'restart failed'` as fallback label.

### Bug 7: Duplicate lifecycle routes in system.py
**File:** `gaia-web/gaia_web/routes/system.py` (lines 680-703)
**Problem:** Duplicate `@router.get("/lifecycle/state")` and `@router.get("/lifecycle/history")` routes at end of file. The duplicate `/lifecycle/history` returned `{"history": []}` (object wrapper) while the original returns `[]` (bare array). FastAPI would use whichever registered first, but the duplicates caused import warnings and confusion.
**Fix:** Removed duplicate routes, added comment pointing to canonical definitions.

---

## Verified Working (No Changes Needed)

| Group | Status |
|-------|--------|
| **Lifecycle Panel** - tier cards, transition controls, reconcile | OK |
| **Lifecycle Panel** - transition labels, history display | OK |
| **Sleep Control** - wake/sleep/shutdown buttons | OK (proxy routes exist) |
| **Sleep Control** - auto-sleep toggle, threshold display | OK |
| **Prime Wake Triggers** - discord typing, workstation toggles | OK |
| **GPU Management** - release/reclaim buttons | OK (proxy routes exist) |
| **Chaos Monkey** - mode select, drill types, inject button | OK |
| **Chaos Monkey** - serenity badge, config save | OK |
| **Cognitive Battery** - run battery, poll status, fetch results | OK |
| **Cognitive Battery** - section filter, failure display, by-section | OK |
| **Training Pipeline** - poll status, run pipeline, smoke test | OK |
| **Training Pipeline** - dry run, skip nano options | OK |
| **Doctor Panel** - alarms, maintenance toggle | OK |
| **Doctor Panel** - serenity meter reads from doctor/status.serenity | OK |
| **Surgeon** - approval toggle, queue display, approve/reject | OK |
| **Surgeon** - history fetch, expanded repair detail | OK |
| **Training Monitor** - progress polling, loss sparkline, log SSE | OK |

---

## Files Modified

- `gaia-web/static/app.js` — 4 JS fixes (VRAM sum, sleep uptime, GPU fields, dissonance merge)
- `gaia-web/static/index.html` — 2 template fixes (irritation + remediation field names)
- `gaia-web/gaia_web/routes/system.py` — Removed duplicate lifecycle routes
