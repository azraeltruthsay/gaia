# GAIA Service Blueprint: `gaia-core` (The Brain)

## Role and Overview

`gaia-core` runs GAIA's cognitive loop: a message arrives as a `CognitionPacket`
on `POST /process_packet`, is routed, grounded, deliberated and generated, and
streams back as NDJSON. It also embeds **Core**, one of her two minds, through
the GAIA Engine in managed mode on `localhost:8092`, and runs the sleep cycle.

Architecture era: **Sovereign Duality** (two minds; Nano retired). Where this
document and `CLAUDE.md` disagree, `CLAUDE.md` wins.

## The two minds

| Tier | Model | Where it runs | Role |
|------|-------|---------------|------|
| **Core** (Operator) | Gemma4-12B-GAIA-Core-v2 | embedded here, `:8092` (gaia_cpp: GPU Q6_K in gear 1, CPU Q4_K_M otherwise) | triage, intent, tools, vision, audio, chat |
| **Prime** (Sovereign) | Qwen3-VL-8B-GAIA-Prime-v2b | `gaia-prime:7777` (GPU NF4 in gear 2, CPU GGUF otherwise) | deep reasoning, code, planning; default voice in the resting gear |
| Groq | `gaia_constants.json` → `groq_fallback` | cloud | fallback only |

The mind on the GPU generates; the other observes from the CPU. Which mind is
on the GPU is the orchestrator's gearbox (P / 1 / 1+ / 2 / S / 0 / T), not
gaia-core's decision: gaia-core asks with `POST /gpu/wake` and reads the state.
`nano` in the model pool and `NANO_ENDPOINT` are aliases for Core's engine,
kept for old call sites.

## Container Configuration

**Image**: `localhost:5000/gaia-core:local`, built from `gaia-core/Dockerfile`
(CUDA 12.8 base; llama.cpp b8770 llama-server, PyTorch and the GAIA Engine).
The engine itself is a directory mount of `gaia-engine/gaia_engine`.

**Port**: 6415 (production), 6416 (`gaia-core-candidate`). The embedded engine
listens on 8092 inside the container.

**Health Check**: `curl -f http://localhost:6415/health`. `/health` reports
`inference_ok`, the Observer and the transmission.

**Candidate**: `gaia-core-candidate` mounts `candidates/gaia-core`, runs with
`SLEEP_ENABLED=False` and `GAIA_IS_CANDIDATE=1`. The flag stops it from ever
driving the production gearbox (`/gpu/wake`) or Discord presence (gx27).

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GAIA_BACKEND` | `auto` | inference backend selection |
| `CORE_DEVICE` | `cuda` | device for the embedded Core |
| `CORE_CPU_ENDPOINT` / `CORE_CPU_PORT` | `http://localhost:8092` / `8092` | embedded engine |
| `CORE_SAFETENSORS_PATH` | `/models/core` | Core weights (torch path, rollback) |
| `PRIME_ENDPOINT` / `PRIME_MODEL` | `http://gaia-prime:7777` / `/models/prime` | Prime |
| `MCP_ENDPOINT` | `http://gaia-mcp:8765/jsonrpc` | tools |
| `STUDY_ENDPOINT` | `http://gaia-study:8766` | training, vector indexing |
| `CFR_CONVERSATION_ENABLED` | `1` | relevance×decay working set for the conversation |
| `CFR_BLUR_BREADCRUMB` | `0` | off: telling Core what was set aside backfires |
| `VOICE_GATE_ENABLED` | `1` | post-generation meta-commentary strip |
| `OBSERVER_ON_STREAM` | `1` | the conscience on the user path (fast mode) |
| `AFFECT_APPRAISAL_ENABLED` | `0` | feed the affect organ from real subsystems |
| `DELIBERATION_NATIVE_THINKING` | `1` | Core thinks in Gemma 4's thought channel |
| `SERENITY_DRILLS` | `1` | sleep drills via gaia-monkey until Serene (a57i) |
| `THOUGHT_SEED_MAX_PENDING` | `2000` | cap on the thought-seed backlog |
| `GAIA_USER_TZ`, `GAIA_LOCALE_*` | | locality organ: time zone, city, weather |

### Volume Mounts

- `./gaia-core:/app:rw`, `./gaia-common:/gaia-common:ro`
- `./gaia-engine/gaia_engine` → site-packages `gaia_engine` (ro)
- `.:/gaia/GAIA_Project:rw`: the project, for blueprint validation, as-built
  reports and auto-promotion
- `../gaia-instance/knowledge:/knowledge:rw`, with the repo's `blueprints`,
  `system_reference`, `personas`, `research`, `consent`, `templates` (ro) and
  `penpal` (rw) mounted over it
- `../gaia-instance/knowledge/vector_store:/vector_store:ro`
- `${GAIA_MODELS_HOST}:/models:ro`, `/warm_pool:ro`, `../gaia-instance/logs:/logs`
- `gaia-shared:/shared:rw`: inter-service state

## A turn (`AgentCore.run_turn`, `main.py /process_packet`)

1. **Arrival** (`main.py`): approval replies (TOTP codes) are answered before
   anything else; people and pronouns are noted; a parked GAIA is woken.
2. **Intent and routing**: embedding classifier, then the NeuralRouter picks
   the mind and the pipeline depth.
3. **Ingestion**: knowledge-base and attachment ingestion; recitation of a
   known work.
4. **Knowledge Router**: grounding before any generation. A pushback re-asks
   the web about the disputed answer; a recent recall ("where were we?") is
   answered by the visible turns; a follow-up borrows the subject from the
   last exchange; their own world is searched in their knowledge base, not
   the web.
5. **Prompt assembly** (`utils/prompt_builder.py`): identity facts (volatile,
   prompt-injected), `[Here & Now]` (time, place, weather), the people present
   and her bonds with them, the CFR working set of the conversation.
6. **Deliberation**: the thinking mind reasons; drift scan and cross-tier
   audit check it.
7. **Generation** streams; inline `<tool_call>`s run through gaia-mcp
   (sensitive ones park for a TOTP approval). Text written before a call is
   retracted (`{"type": "retract"}`) and the answer comes from the result.
8. **Stream scrub** (`cognition/reply_scrub.py`): plumbing stripped; an
   unfounded apology, or conceding words while she holds a disputed answer,
   cut from the opening.
9. **Observer**: the parallel CPU audit of the reply; the second mind may post
   a follow-up.
10. **After the turn**: the transmission counts corrections (strikes shift
    gears); samvega captures discernment artifacts; the outcome ledger and
    affect appraisal are fed.

## Sleep

After 30 idle minutes she is DROWSY, then ASLEEP: `SleepTaskScheduler` runs
maintenance, reflection, curation, documentation, self-review and the
serenity drill. A wake signal interrupts the current task at its next phase
boundary. See `GAIA_SLEEP_CYCLE.md`.

## Constants (`cognition/agent_core.py`)

| Constant | Value | Use |
|----------|-------|-----|
| `WHOLE_DOC_CHARS` | 9000 | how much of a fetched document is read whole |
| `PLAIN_DOC_CHARS` | 1200 | excerpt size for a plain document answer |
| `WHOLE_FALLBACK_CHARS` | 3000 | a document this short is answered whole |
| `HISTORY_SUMMARY_THRESHOLD` | 20 | defined, unused |
| `MAX_OUTPUT_LENGTH` | 500 | defined, unused |

## Meta
**Status**: current
**Blueprint Version**: 0.7
**Last Updated**: 2026-11-17 (rewritten for Sovereign Duality, GAIA_Project-okhq)

## Changelog

### 2026-11-17
Rewritten. Version 0.6 (2026-03-02) described the retired three-tier
Nano/Operator/Prime(AWQ) stack, a CPU-only image and a 20-step pipeline
from before CFR, the Observer, the transmission and the Knowledge Router.

### 2026-10-14
`HISTORY_SUMMARY_THRESHOLD` and `MAX_OUTPUT_LENGTH` found to be dead
constants (defined in `agent_core.py`, referenced nowhere).
