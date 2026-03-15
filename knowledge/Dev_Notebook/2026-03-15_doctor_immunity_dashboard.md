# Dev Journal — 2026-03-15: Doctor & Immunity Dashboard Panel + Surgeon Approval Queue

## Context

gaia-doctor has a rich immune system (irritation tracking, alarm system, two-tier repair with LLM surgeon, maintenance mode, dissonance detection, serenity state) but none of it was surfaced in the Mission Control dashboard. Additionally, the LLM surgeon's auto-repair of candidate code had no human-in-the-loop gate — repairs were applied immediately.

## What Was Built

### 1. Surgeon Approval Queue (gaia-doctor/doctor.py)

**New module-level state:**
- `_surgeon_approval_required: bool` — toggled via API, persisted to `/shared/doctor/surgeon_config.json`
- `_surgeon_queue: list[dict]` — pending repair proposals
- `_surgeon_history: list[dict]` — completed/rejected (capped at 50)

**6 new endpoints:**
- `GET /surgeon/config` — current approval mode
- `POST /surgeon/config` — toggle `{"approval_required": true/false}`, persists to disk
- `GET /surgeon/queue` — pending proposals
- `POST /surgeon/approve` — apply queued fix (write → validate → restart → history)
- `POST /surgeon/reject` — discard queued fix → history
- `GET /surgeon/history` — recent resolved proposals

**Modified `repair_candidate_file()`:** After LLM returns `fixed_code`, if approval is required, the fix is queued instead of immediately written. Returns `{"status": "pending_approval", "repair_id": "..."}`.

**Approve flow:** Writes fixed code to container → validates (py_compile + lint) → restarts container → moves to history. If validation fails, reports failure without leaving broken code.

### 2. Proxy Routes (gaia-web/gaia_web/routes/system.py)

9 new routes added:
- `GET /doctor/status` — raw doctor status (alarms, remediations, serenity, maintenance)
- `GET /irritations` — full irritation list
- `GET /dissonance` — prod vs candidate drift report
- 6 surgeon routes mapping to doctor endpoints

### 3. Dashboard UI (index.html + app.js + style.css)

**New "Doctor & Immunity" panel** in Commands tab (wide, after Training Pipeline):

Immunity section:
- Irritation count with expandable detail list (service + pattern + time)
- Active alarms with red badge indicator
- Maintenance mode toggle button
- Serenity score meter (color-coded: red < 3, yellow 3-5, green 5+)
- Dissonance count with expandable file list
- Remediation count with expandable detail

Surgeon section (separated by divider):
- Human Approval toggle (ON/OFF)
- Pending repairs list — expandable cards showing:
  - repair_id, service, filename
  - Error message
  - Broken code excerpt (pre block, max 500 chars)
  - Fixed code excerpt (pre block, max 500 chars)
  - Approve (green) / Reject (red) buttons
- Recent repairs history with expandable detail

**`doctorPanel()` Alpine component:** Polls 6 endpoints in parallel via `Promise.allSettled` every 10s. Lazy-loads detail sections on expand. Interactive toggles for maintenance and surgeon approval.

## Deployment

- **gaia-doctor**: `docker compose build gaia-doctor && docker compose up -d gaia-doctor` (COPY-based container)
- **gaia-web**: `docker restart gaia-web` (volume-mounted)
- All gaia-web files synced to `candidates/`

## Verification

- `curl localhost:6419/surgeon/config` → `{"approval_required": false}`
- Both services healthy post-deploy
- Dashboard panel renders in Commands tab

## Files Modified

| File | Change |
|------|--------|
| `gaia-doctor/doctor.py` | Surgeon queue state, 6 endpoints, modified repair flow, config persistence |
| `gaia-web/gaia_web/routes/system.py` | 9 new proxy routes |
| `gaia-web/static/index.html` | Doctor & Immunity panel HTML |
| `gaia-web/static/app.js` | `doctorPanel()` Alpine component |
| `gaia-web/static/style.css` | Doctor/surgeon styles |
| `candidates/gaia-web/**` | Synced copies of all gaia-web changes |
