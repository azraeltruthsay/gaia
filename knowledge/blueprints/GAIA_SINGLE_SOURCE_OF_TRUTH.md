# GAIA Single Source of Truth — Centralized Configuration Blueprint

**Status:** Concept / Not Scheduled
**Recorded:** 2026-02-28
**Author:** Azrael

---

## Problem Statement

Specific values — port numbers, VRAM percentages, model names, token lengths, timeouts, chunk durations, health check intervals — are currently scattered across:

- Python source files (`gaia_constants.json`, hardcoded literals in `.py` files)
- Docker Compose files (env vars, port mappings, healthcheck intervals)
- Dockerfiles (ARG/ENV, HEALTHCHECK timeouts)
- Shell scripts (port numbers, service names, timeouts)
- Documentation (README.md, blueprint files, dev notebooks)

When a value changes (e.g. gaia-core moves from port 6415 to a different port, or the VRAM threshold changes), every instance must be found and updated manually. The goal is a **single master config** where each specific value lives exactly once, and everything else is a reference or a derived artifact.

---

## Design

### Master Config: `gaia_constants.json` (extended)

`gaia_constants.json` already exists as GAIA's runtime constants file. Extend it to be the **single master record** for all configuration values across the entire project — not just runtime, but also infrastructure, tooling, and documentation.

Organize by domain:

```json
{
  "ports": {
    "gaia_core_live": 6415,
    "gaia_core_candidate": 6416,
    "gaia_web": 6414,
    "gaia_mcp_live": 8765,
    "gaia_mcp_candidate": 8767,
    "gaia_prime_live": 7777,
    "gaia_prime_candidate": 7778,
    "gaia_study_live": 8766,
    "gaia_study_candidate": 8768,
    "gaia_orchestrator_live": 6410,
    "gaia_orchestrator_candidate": 6411,
    "gaia_audio_live": 8080,
    "gaia_audio_candidate": 8081
  },
  "models": {
    "prime_default": "Qwen3-4B-Instruct-2507-heretic",
    "warm_pool_path": "/mnt/gaia_warm_pool",
    "study_embedding_model": "...",
    "bootstrap_groq_model": "llama-3.1-8b-instant"
  },
  "vram": {
    "prime_vram_gb": 10.5,
    "gpu_cleanup_threshold_mb": 3000,
    "distracted_cpu_threshold_pct": 25,
    "distracted_sustained_seconds": 5
  },
  "timeouts": {
    "web_to_core_s": 300,
    "core_to_mcp_s": 20,
    "core_to_prime_s": 120,
    "health_check_interval_s": 30,
    "health_check_timeout_s": 10,
    "prime_start_period_s": 120,
    "gpu_cleanup_timeout_s": 30,
    "handoff_timeout_s": 120,
    "promotion_health_poll_s": 10,
    "promotion_health_max_s": 180
  },
  "tokens": {
    "audio_transcript_max_chars": 30000,
    "..."
  },
  "sleep": {
    "idle_threshold_minutes": 5,
    "heartbeat_interval_minutes": 20,
    "checkpoint_timeout_s": 15,
    "stop_grace_period_s": 25
  },
  "retry": {
    "web_to_core_retries": 3,
    "web_to_core_backoff_base_s": 2,
    "core_to_mcp_retries": 3,
    "core_to_mcp_backoff_base_s": 2
  },
  "audio": {
    "listener_chunk_duration_s": 30,
    "listener_ingest_interval_s": 60
  }
}
```

---

### Derived Artifacts

The master config is the only file humans edit. Everything else is generated from it by a render script.

#### 1. `gaia.env` — Docker Compose variable file

Generated from `gaia_constants.json`. Docker Compose natively substitutes `${VAR_NAME}` from `.env`.

```env
PORT_GAIA_CORE_LIVE=6415
PORT_GAIA_CORE_CANDIDATE=6416
PORT_GAIA_WEB=6414
TIMEOUT_WEB_TO_CORE=300
...
```

Compose files reference these:
```yaml
ports:
  - "${PORT_GAIA_CORE_LIVE}:${PORT_GAIA_CORE_LIVE}"
healthcheck:
  interval: ${TIMEOUT_HEALTH_CHECK_INTERVAL}s
```

#### 2. `scripts/gaia_config.sh` — Shell script variable file

Generated from `gaia_constants.json`. Shell scripts source this at the top.

```bash
# AUTO-GENERATED — edit gaia_constants.json, not this file
export PORT_GAIA_CORE_LIVE=6415
export PORT_GAIA_CORE_CANDIDATE=6416
export TIMEOUT_PROMOTION_HEALTH_MAX=180
...
```

Shell scripts:
```bash
source "$(dirname "$0")/gaia_config.sh"
curl -sf "http://localhost:${PORT_GAIA_CORE_LIVE}/health"
```

#### 3. `README.md` — Rendered from `README.md.j2`

Generated via Jinja2 (see below). The template is the source of truth; `README.md` is a build artifact.

```jinja
gaia-core runs on port **{{ ports.gaia_core_live }}** (candidate: {{ ports.gaia_core_candidate }}).
Prime model: `{{ models.prime_default }}`, using ~{{ vram.prime_vram_gb }}GB VRAM.
```

---

### Render Script: `scripts/render_config.sh`

A single script that reads `gaia_constants.json` and generates all derived artifacts.

```
scripts/render_config.sh
  ├─ Reads: gaia_constants.json
  ├─ Writes: gaia.env
  ├─ Writes: scripts/gaia_config.sh
  └─ Writes: README.md (from docs/README.md.j2)
```

The render script is a thin Python wrapper (uses `json` + `jinja2`, both already in the stack). It should be fast (<1s) and idempotent.

**Run manually** when `gaia_constants.json` changes, or **automatically** as part of the promotion pipeline (Stage 7, alongside flatten_soa).

---

## Jinja2 for Documentation

### Why Jinja2 and not something else

- Already in GAIA's Python dependency tree (FastAPI pulls it in)
- Zero new dependencies to install
- Simple, well-understood template syntax
- Sufficient for README and blueprint templating

### Implementation

Template files live alongside their outputs with a `.j2` extension:
- `docs/README.md.j2` → renders to `README.md`
- Optionally: `docs/OVERVIEW.md.j2` → renders to `knowledge/blueprints/OVERVIEW.md`

The render script passes the full `gaia_constants.json` dict as the Jinja2 context:

```python
from jinja2 import Environment, FileSystemLoader
import json

constants = json.load(open("gaia_constants.json"))
env = Environment(loader=FileSystemLoader("docs/"))
template = env.get_template("README.md.j2")
output = template.render(**constants)
open("README.md", "w").write(output)
```

### Scope

Jinja2 templating is appropriate for documents that contain **specific values that change** — port numbers, model names, VRAM figures. It is NOT appropriate for documents that are primarily prose. Blueprint files (like this one) should stay as plain Markdown unless they happen to contain many specific values.

---

## MkDocs — Separate Future Decision

MkDocs is a free/OSS static site generator (BSD license) with built-in Jinja2 templating via `mkdocs-macros-plugin`. It would allow the entire `knowledge/` directory to become a navigable documentation website.

**It solves a different problem** than config centralization. Don't couple the two efforts.

**Overhead if pursued:**
- New Python dependency: `mkdocs`, `mkdocs-material` (MIT, ~2MB), `mkdocs-macros-plugin`
- New directory structure: `docs/` with `mkdocs.yml` config
- Build step: `mkdocs build` generates static HTML into `site/`
- Hosting: GitHub Pages (free, zero infra) or serve `site/` from gaia-wiki
- No new container required; `mkdocs serve` can be a dev convenience command

**Material for MkDocs** (MIT, free tier) is excellent and would give GAIA a professional docs site at zero cost. The paid "Insiders" tier adds features like social cards and offline search — not needed.

**When to pursue:** If GAIA is distributed to other users (see Bootstrap Install blueprint), a proper docs site becomes genuinely valuable. At that point, the Jinja2 templates from this effort port directly into MkDocs with minimal rework — `mkdocs-macros-plugin` uses the same `{{ variable }}` syntax.

---

## What Does NOT Move Into the Master Config

Not everything should be centralized. Keep the following where they are:

- **Secrets and credentials** — never in `gaia_constants.json`, never in generated files that get committed
- **Environment-specific overrides** — `docker-compose.override.yml` is intentionally local; don't template it
- **Service-internal logic constants** — thresholds used only within one service and never referenced externally; leave them in the service's own code
- **Prose documentation** — blueprint files, dev journals, narrative text stays as plain Markdown

The test: if a value appears in more than one place in the codebase, it belongs in the master config. If it appears in exactly one place, leave it there.

---

## Migration Strategy

1. Audit the codebase for values that appear in multiple places (ports are the most obvious)
2. Extend `gaia_constants.json` with the new domains (ports, timeouts, etc.)
3. Write `render_config.sh`
4. Replace hardcoded values in Compose files with `${VAR}` references
5. Replace hardcoded values in shell scripts with sourced vars
6. Create `README.md.j2` template
7. Add render step to promotion pipeline (Stage 7, before flatten_soa)
8. Add generated files to `.gitignore` OR commit them with a clear "AUTO-GENERATED" header — decide which convention to use

**`.gitignore` vs commit:** The pragmatic answer is to commit the generated files with `AUTO-GENERATED` headers. This means the repo is always self-contained (clone + run works without a render step), and the generated files are visible in diffs so changes are auditable. The downside is that someone can edit them directly and create drift — the `AUTO-GENERATED` header is the only safeguard against that.

---

## Config Format: TOML Migration Plan

TOML is the target format. It supports comments (critical for a config this large), is more human-readable than JSON, and `tomllib` is built into Python 3.11+ (no new dependency). The migration should be gradual to avoid breaking anything.

### Phase 1: Introduce TOML alongside JSON
- Create `gaia_constants.toml` as the new canonical master config
- Update `render_config.sh` to read from TOML
- Update any new code to read from TOML
- Leave `gaia_constants.json` in place and untouched — existing callers keep working

### Phase 2: Migrate callers
- Audit all code that reads `gaia_constants.json` directly
- Update each caller to read from `gaia_constants.toml` instead
- Keep both files in sync manually during the transition (or generate JSON from TOML as a compatibility artifact)

### Phase 3: Retire JSON
- Once no callers remain: rename `gaia_constants.json` → `gaia_constants.json.bak`
- Re-run the full test suite + smoke tests
- If nothing breaks: delete the `.bak`
- If something breaks: it surfaces the missed caller; fix and repeat

### Why not just convert JSON → TOML in one shot?
The gradual approach means at no point is the system broken mid-migration. The `.bak` step in Phase 3 is a reversible canary — it simulates deletion without actually losing the file.

### TOML format example
```toml
# GAIA Master Configuration
# Edit this file. Do NOT edit derived artifacts (gaia.env, scripts/gaia_config.sh, README.md).

[ports]
gaia_core_live = 6415        # Live cognitive loop
gaia_core_candidate = 6416   # Candidate / HA fallback
gaia_web = 6414
gaia_mcp_live = 8765
gaia_mcp_candidate = 8767
gaia_prime_live = 7777       # vLLM inference
gaia_prime_candidate = 7778
gaia_study_live = 8766
gaia_study_candidate = 8768
gaia_orchestrator_live = 6410
gaia_orchestrator_candidate = 6411
gaia_audio_live = 8080
gaia_audio_candidate = 8081

[models]
prime_default = "Qwen3-4B-Instruct-2507-heretic"
warm_pool_path = "/mnt/gaia_warm_pool"

[vram]
prime_vram_gb = 10.5
gpu_cleanup_threshold_mb = 3000
distracted_cpu_threshold_pct = 25
distracted_sustained_seconds = 5

[timeouts]
web_to_core_s = 300
core_to_mcp_s = 20
core_to_prime_s = 120
health_check_interval_s = 30
health_check_timeout_s = 10
prime_start_period_s = 120

[sleep]
idle_threshold_minutes = 5
heartbeat_interval_minutes = 20
checkpoint_timeout_s = 15
stop_grace_period_s = 25
```

---

## Open Questions

- **Schema validation**: Should `gaia_constants.toml` have a schema (e.g. via `pydantic` or a simple hand-written validator) so that invalid edits are caught before render? Low overhead, high value — a typo in a port number should fail loudly.
- **Hot reload**: gaia-core already loads `gaia_constants.json` at runtime. Does it need to watch for changes, or is a restart sufficient? Currently restart — this doesn't change that.

---

*Not scheduled for implementation. Record only.*
