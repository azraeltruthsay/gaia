# Known Drift Patterns

> Historical divergence patterns observed in GAIA. Use these to calibrate what to look for during review.

## Pattern: Candidate/Live Desynchronization

**What happens**: Changes in `candidates/gaia-{service}/` not propagated to `gaia-{service}/`, or vice versa. Docker mounts one path; Python only sees what's mounted.

**How it manifests**:
- Candidate tests pass but production fails (or reverse)
- Stale bytecache masks code changes until container restart
- HA failover routes to candidate running old code

**Example** (2026-02-22): Candidate image stale by 5 days, missing `gaia` user in `/etc/passwd`. HA failover routed traffic to broken candidate.

**What to check**: Any PR modifying service code should touch both paths, or the promotion pipeline should handle the sync.

## Pattern: Lite Model Pattern Matching

**What happens**: The 3B Lite model pattern-matches syntax from system prompts instead of following instructions. Most dangerous when tool-shaped patterns (`EXECUTE:` directives) are in the prompt.

**How it manifests**:
- Response contains echoed `EXECUTE:` directives for already-executed tools
- Output sanitizer catches duplicate tool calls
- Empty responses after stripping duplicate EXECUTE

**Root cause**: Small models don't comprehend instructions — they reproduce patterns they've seen.

**Fix pattern** (commit 37fddb0):
- Suppress tool syntax examples when `tool_routing.execution_status == EXECUTED`
- Inject assistant prefill to steer toward prose
- Promote Prime for final generation if used during reflection

## Pattern: Blueprint Validation Gaps

**What happens**: Code drifts from blueprint declarations. Structural validator catches missing endpoints but misses semantic divergence.

**Current validator coverage** (`sleep_task_scheduler.py::_run_blueprint_validation()`):
- Checks: enum members, endpoints, constants declared in blueprint exist in code ✓
- Misses: intent drift, failure mode coverage gaps, dependency signature correctness ✗

**What to check**: Don't trust "blueprint validated" as proof of compliance. Manually verify the semantic dimensions (intent alignment, failure mode handling).

## Pattern: False Positive Observer Noise

**What happens** (commit 3336ec9): When fallback candidates are running, they emit spurious "candidate service online" observations. StreamObserver treats these as noteworthy events, cluttering the dev matrix.

**Fix**: Observer now checks candidate status in context; suppresses presence reports for candidates unless explicitly running as primary.

**What to check**: Observer-related changes should consider both live and candidate contexts.

## Pattern: Stale Model Selection

**What happens**: After a reflection cycle uses a Prime-class model, the final generation step releases Prime and falls back to Lite. The response quality drops because the 3B model can't handle the enriched context.

**Fix**: Post-reflection logic checks if reflection model is PRIME-class and promotes it to generation instead of releasing.

**What to check**: Any code touching model selection or the reflection→generation handoff should verify the promotion path.
