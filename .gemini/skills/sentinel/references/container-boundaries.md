# Container Boundaries

> What each service is and is not permitted to do. Violations of these boundaries are findings.

## gaia-mcp (The Hands — Tool Sandbox)

**Hardening:**
- `security_opt: no-new-privileges:true`
- `cap_drop: ALL` → `cap_add: CHOWN, SETGID, SETUID`
- No host networking

**Permitted:**
- Read: /knowledge, /gaia-common, /sandbox, /models (all within size limits)
- Write: /knowledge, /sandbox (allowlisted paths only)
- Execute: whitelisted shell commands (shell=False, 10s timeout)
- Network: internal Docker network only (gaia-network)

**Not permitted:**
- Write outside /knowledge and /sandbox
- Execute arbitrary commands
- Access /app (own source code directly)
- Privilege escalation
- Host resource access beyond mounts

## gaia-core (The Brain)

**Permitted:**
- Read/write: /app (own source, development mount), /shared (session state)
- Read: /knowledge, /gaia-common, /vector_store (read-only)
- Network: call gaia-prime, gaia-mcp, gaia-study via HTTP
- CPU only — no direct GPU access

**Not permitted:**
- Write to /knowledge/vector_store (sole writer is gaia-study)
- Write to /models (sole writer is gaia-study)
- Direct external network access (all external goes through gaia-mcp tools)

## gaia-web (The Face — Gateway)

**Permitted:**
- Read: /knowledge (read-only), static assets
- Read: Docker socket (read-only, for terminal/introspection)
- Network: accept external connections on port 6414, call gaia-core
- Write: /logs, DM blocklist, voice whitelist

**Not permitted:**
- Write to /knowledge (read-only mount)
- Direct calls to gaia-mcp or gaia-prime (must go through gaia-core)
- Docker socket write operations (mounted :ro)

## gaia-study (The Subconscious)

**Permitted:**
- SOLE WRITER to /vector_store and /models (exclusive access)
- Read: /knowledge, /gaia-common
- GPU access: all GPUs (for QLoRA training)
- Network: accept calls from gaia-core

**Not permitted:**
- Direct external network access
- Modification of other services' state

## gaia-prime (The Voice — Inference)

**Permitted:**
- GPU: 1 GPU (device 0) for vLLM inference
- Read: /models (LoRA adapters, read-only)
- Network: accept inference requests on port 7777
- Sleep/wake lifecycle (GPU handoff with gaia-study)

**Not permitted:**
- Filesystem writes (inference only)
- Direct access to /knowledge or /vector_store
- Network calls to other services (it only receives, never initiates)

## gaia-audio (The Ears & Mouth)

**Permitted:**
- GPU: shared with gaia-prime (for Whisper/Coqui)
- Network: accept requests on port 8080
- Read: /models (Whisper/Coqui models)

**Not permitted:**
- Filesystem writes beyond /tmp
- Direct access to /knowledge
- Initiate calls to other services

## Network Topology

- All services on `gaia-network` bridge (172.28.0.0/16)
- No host network mode on any service
- Only gaia-web exposes ports externally
- gaia-wiki exposes no ports (internal documentation only)
- Dozzle mounts Docker socket read-only (log viewer only)

## HA Fallback Boundaries

- Candidate services run on separate ports (+1 offset)
- Share the same Docker network as live services
- Maintenance mode flag (`/shared/ha_maintenance`) suppresses failover
- Candidates use separate volume instances for sandbox/shared state
