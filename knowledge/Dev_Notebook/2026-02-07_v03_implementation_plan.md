# GAIA v0.3 Implementation Plan

**Date:** 2026-02-07
**Author:** Claude Code (Opus 4.6) via Happy
**Status:** Awaiting approval

---

## Phase 1: Decentralize gaia-prime (Standalone Inference Server)

**Goal:** Add gaia-prime as a standalone vLLM inference server with 0.65 VRAM cap. Zero changes to gaia-core — it continues working as-is until Phase 2.

### Step 1.1: Add gaia-prime to `docker-compose.yml`

Add a new service using the official `vllm/vllm-openai` image:

```yaml
gaia-prime:
  image: vllm/vllm-openai:latest
  container_name: gaia-prime
  hostname: gaia-prime
  restart: unless-stopped
  volumes:
    - ./gaia-models:/models:ro
  environment:
    - VLLM_WORKER_MULTIPROC_METHOD=spawn
  ports:
    - "7777:7777"
  command: >
    --model /models/Claude
    --host 0.0.0.0
    --port 7777
    --gpu-memory-utilization 0.65
    --max-model-len 8192
    --max-num-seqs 4
    --trust-remote-code
    --dtype auto
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  networks:
    - gaia-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:7777/health"]
    interval: 30s
    timeout: 10s
    retries: 5
    start_period: 120s
```

**Key decisions:**
- `--gpu-memory-utilization 0.65` enforces the VRAM cap at the vLLM level
- `--max-num-seqs 4` allows limited batching (Prime handles one user but may batch observer)
- `start_period: 120s` gives model loading time before health checks fail
- Model path: `/models/Claude` (same as current gpu_prime config)
- Port 7777 chosen to avoid conflicts with existing services

### Step 1.2: Add gaia-prime-candidate to `docker-compose.candidate.yml`

Mirror of live definition with candidate naming:

```yaml
gaia-prime-candidate:
  image: vllm/vllm-openai:latest
  container_name: gaia-prime-candidate
  hostname: gaia-prime-candidate
  restart: "no"
  volumes:
    - ./gaia-models:/models:ro
  environment:
    - VLLM_WORKER_MULTIPROC_METHOD=spawn
  ports:
    - "7778:7777"
  command: >
    --model /models/Claude
    --host 0.0.0.0
    --port 7777
    --gpu-memory-utilization 0.65
    --max-model-len 8192
    --max-num-seqs 4
    --trust-remote-code
    --dtype auto
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  networks:
    - gaia-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:7777/health"]
    interval: 30s
    timeout: 10s
    retries: 5
    start_period: 120s
  profiles:
    - full
    - prime
```

### Step 1.3: Update `gaia-orchestrator/gaia_orchestrator/config.py`

Add:
```python
prime_url: str = "http://gaia-prime:7777"
prime_candidate_url: str = "http://gaia-prime-candidate:7777"
gpu_prime_vram_quota: float = 0.65
gpu_audio_vram_quota: float = 0.35
```

### Step 1.4: Update `gaia.sh`

Add `prime` as a recognized service in `show_status()` (port 7777), and add it to the `cmd_swap` case for completeness.

### Step 1.5: Verify

```bash
# Start just gaia-prime alongside existing stack
docker compose -f docker-compose.yml up -d gaia-prime

# Wait for model load (~60-90s), then test
curl http://localhost:7777/v1/models
curl http://localhost:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Claude","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

**Exit criteria:** gaia-prime responds to chat completions while existing gaia-core continues working independently with its embedded vLLM.

---

## Phase 2: Refactor gaia-core as HTTP Client to gaia-prime

**Goal:** gaia-core stops loading vLLM locally. All `gpu_prime` / `prime` inference goes through HTTP to gaia-prime. Lite model stays local for observer. gaia-core loses its GPU reservation.

### Step 2.1: Add `PRIME_ENDPOINT` to config

**File:** `candidates/gaia-core/gaia_core/config.py`

```python
PRIME_ENDPOINT: str = os.getenv("PRIME_ENDPOINT", "http://gaia-prime:7777")
```

### Step 2.2: Create `PrimeHttpModel` adapter

**File:** `candidates/gaia-core/gaia_core/models/prime_http_model.py` (NEW)

A thin adapter that implements the same interface as VLLMChatModel but delegates to HTTP:

```python
class PrimeHttpModel:
    """HTTP adapter for gaia-prime's OpenAI-compatible API."""

    def __init__(self, endpoint: str, model_name: str = "Claude"):
        self.endpoint = endpoint
        self.model_name = model_name
        self._client = httpx.AsyncClient(base_url=endpoint, timeout=300.0)

    async def create_chat_completion(self, messages, **kwargs):
        """Non-streaming completion."""
        response = await self._client.post("/v1/chat/completions", json={
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        })
        return response.json()

    async def create_chat_completion_stream(self, messages, **kwargs):
        """Streaming completion via SSE."""
        async with self._client.stream("POST", "/v1/chat/completions", json={
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": True,
        }) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    yield json.loads(line[6:])
```

### Step 2.3: Modify `_model_pool_impl.py`

**File:** `candidates/gaia-core/gaia_core/models/_model_pool_impl.py`

Changes:
1. When loading `gpu_prime` or `prime`: instantiate `PrimeHttpModel` instead of `VLLMChatModel`
2. Remove vLLM import and initialization code paths for prime models
3. Remove GPU memory management for prime (no longer needed)
4. Keep `lite` model loading (local llama.cpp) for the observer
5. Keep API fallbacks (groq, openai, gemini) unchanged

The key change in `ensure_model_loaded()`:
```python
if model_type == "vllm" and name in ("gpu_prime", "prime"):
    # v0.3: Delegate to gaia-prime via HTTP
    self.models[name] = PrimeHttpModel(
        endpoint=self.config.PRIME_ENDPOINT,
        model_name=cfg.get("model_name", "Claude")
    )
```

### Step 2.4: Update `gaia_constants.json`

**File:** `candidates/gaia-core/gaia_core/gaia_constants.json`

Change gpu_prime config:
```json
"gpu_prime": {
  "enabled": true,
  "type": "http_prime",
  "model_name": "Claude",
  "endpoint": "http://gaia-prime:7777"
}
```

### Step 2.5: Update `docker-compose.candidate.yml`

Remove GPU reservation from gaia-core-candidate:
```yaml
# REMOVE this block from gaia-core-candidate:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Add `PRIME_ENDPOINT` to gaia-core-candidate environment:
```yaml
- PRIME_ENDPOINT=${CANDIDATE_PRIME_ENDPOINT:-http://gaia-prime-candidate:7777}
```

### Step 2.6: Verify

```bash
# Start gaia-prime-candidate + gaia-core-candidate
docker compose -f docker-compose.candidate.yml --env-file ./.env.discord --profile full up -d

# Send Discord message → response should come from gaia-prime via HTTP
# Check gaia-core logs: should see HTTP calls to gaia-prime, NOT local vLLM loading
docker logs gaia-core-candidate 2>&1 | grep -i "prime\|vllm\|http"
```

**Exit criteria:** Discord message gets a response. gaia-core logs show HTTP calls to gaia-prime. No vLLM loading in gaia-core. GPU memory shows gaia-prime at ~0.65 utilization, gaia-core at 0.

---

## Phase 3: Formalize the Bicameral Observer

**Goal:** Lite model in gaia-core watches Prime's streaming output from gaia-prime in real-time. Kill switch can abort a response mid-stream.

### Step 3.1: Modify streaming path in `agent_core.py`

**File:** `candidates/gaia-core/gaia_core/cognition/agent_core.py`

The current `_stream_response()` gets tokens from a local model. Change to:
1. Stream SSE chunks from gaia-prime via `PrimeHttpModel.create_chat_completion_stream()`
2. Feed accumulated buffer to local Lite model every N tokens (configurable, default: 16)
3. Lite evaluates for: constitution violations, loop detection, error patterns
4. On BLOCK verdict → send cancel signal, return fallback response

### Step 3.2: Update StreamObserver

**File:** `candidates/gaia-core/gaia_core/cognition/external_voice.py`

Fix the aggressive false-positive issue (currently blocks on "error" appearing in think tags):
- Only check user-facing output, not `<|start_thinking|>` blocks
- Use Lite model for semantic evaluation instead of string matching

### Step 3.3: Verify

Send a message that triggers extended reasoning. Confirm in logs:
- Tokens stream from gaia-prime
- Observer checkpoints appear every 16 tokens
- No false blocks on normal output

---

## Phase 4: MemoryRequest Relay Protocol

**Goal:** When gaia-core determines a fact should be remembered, it POSTs a structured MemoryRequest to gaia-study instead of writing directly.

### Step 4.1: Define MemoryRequest schema

**File:** `candidates/gaia-common/gaia_common/protocols/memory_request.py` (NEW)

```python
class MemoryRequest(BaseModel):
    fact: str
    source: str  # "user_stated", "inferred", "oracle"
    confidence: float  # 0.0-1.0
    tier: int  # 1=core, 2=episodic, 3=reference
    session_id: str
    timestamp: datetime
```

### Step 4.2: Add endpoint to gaia-study

**File:** `candidates/gaia-study/` — add `/memory/request` POST endpoint

Validates, embeds (CPU), stores in ChromaDB.

### Step 4.3: Add relay in gaia-core

**File:** `candidates/gaia-core/gaia_core/cognition/agent_core.py`

When the cognitive loop produces a "remember" action → POST to `http://gaia-study:8766/memory/request`.

### Step 4.4: Verify

Tell GAIA "Remember that my favorite color is blue." → Confirm fact appears in gaia-study's ChromaDB. Ask about it in a new session → RAG retrieves it.

---

## Phase 5: gaia-audio (The Temporal Sensory Organ)

**Goal:** Half-duplex audio: passive listening via Whisper, active speech via Bark/Coqui.

### Step 5.1: Create service skeleton

**Directory:** `gaia-audio/` (NEW)

- FastAPI service on port 7778
- Audio capture thread (10s WAV chunks, rolling buffer)
- Whisper transcription endpoint (12s window, 2s overlap dedup)
- TTS synthesis endpoint (accepts text + speech directives)
- Model swapping logic (load Whisper OR TTS, never both)

### Step 5.2: Add to docker-compose

```yaml
gaia-audio:
  build:
    context: ./gaia-audio
  container_name: gaia-audio
  hostname: gaia-audio
  volumes:
    - ./gaia-models:/models:ro
  environment:
    - WHISPER_MODEL=large-v3-turbo
    - TTS_MODEL=bark-small
    - AUDIO_DEVICE=default
    - GPU_MEMORY_FRACTION=0.35
  ports:
    - "7778:7778"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  devices:
    - /dev/snd:/dev/snd  # Audio hardware passthrough
  networks:
    - gaia-net
```

### Step 5.3: Integration with gaia-core

gaia-core receives transcriptions as a new input source (alongside Discord, HTTP). Responses tagged with `<voice>` directives get routed to gaia-audio for TTS before reaching the user.

### Step 5.4: Verify

Speak → see transcription in gaia-core logs → get text response → hear synthesized speech.

---

## Phase 6: gaia-siem (The Sentinel's Tower)

**Goal:** Holistic observability via Wazuh.

### Step 6.1: Add Wazuh stack to docker-compose

Single-node Wazuh (manager + dashboard):
```yaml
gaia-siem:
  image: wazuh/wazuh-manager:4.x
  container_name: gaia-siem
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - ./knowledge/system_reference:/fim-watch:ro
  ports:
    - "1514:1514"    # Agent communication
    - "55000:55000"  # API
  networks:
    - gaia-net
```

### Step 6.2: Configure monitoring rules

- Docker socket events (container starts, stops, OOM kills)
- FIM on `/knowledge/system_reference/` (constitution tampering detection)
- `dev_matrix_audit.log` ingestion
- Custom rules for identity violations, loop detection, VRAM exhaustion

### Step 6.3: Alert-to-CognitionPacket bridge

Critical Wazuh alerts → POST to gaia-core as a system CognitionPacket → triggers self-reflection.

### Step 6.4: Verify

Modify a file in `/knowledge/system_reference/` → Wazuh alert fires → gaia-core receives self-reflection packet.

---

## Implementation Sequence

```
Week 1:  Phase 1 (gaia-prime standalone)     — LOW RISK, additive
Week 2:  Phase 2 (core refactor to HTTP)      — HIGH RISK, critical path
Week 2:  Phase 4 (memory relay)               — LOW RISK, independent
Week 3:  Phase 3 (bicameral observer)          — MEDIUM RISK, depends on Phase 2
Week 3+: Phase 5 (gaia-audio)                  — MEDIUM RISK, new service
Week 4:  Phase 6 (gaia-siem)                   — LOW RISK, independent
```

## VRAM Conflict Warning

**gaia-prime and the current gaia-core-candidate cannot run simultaneously on the same GPU.** Current gaia-core-candidate uses 0.85 VRAM utilization for its embedded vLLM. gaia-prime wants 0.65. Together they exceed 16GB.

**Mitigation for Phase 1:** Start gaia-prime with gaia-core stopped, OR set `GAIA_CANDIDATE_BACKEND=lite` to keep gaia-core on CPU while testing gaia-prime in isolation.

**Resolved in Phase 2:** Once gaia-core drops its GPU reservation, the conflict disappears.

---

## Approval

**Approval authority:** Azrael (System Architect)
**System host:** gaia-host (RTX 5080 16GB / Ryzen 9 / 32GB RAM)
**Agent instruction:** Begin with Phase 1 upon approval.
