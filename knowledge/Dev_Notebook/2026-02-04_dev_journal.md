# Dev Journal - 2026-02-04

## Dynamic GPU Handoff: Enabling Full Candidate Capabilities

### The Problem

The candidate testing system had a fundamental limitation: candidates couldn't use the GPU because the live `gaia-core` service was holding it via vLLM. The existing solution required:

1. Manually stopping the live service (`docker compose stop gaia-core`)
2. Starting candidates with `GAIA_CANDIDATE_GPU=1`
3. Restarting the live service when done

This was clunky and meant the live service was completely offline during candidate testing. The goal: **allow candidates to do everything the live service can do, including GPU inference**.

### The Solution: Dynamic GPU Release/Reclaim API

Instead of stopping services, we implemented API endpoints that allow the live service to **release** its GPU resources on demand, then **reclaim** them when candidate testing is complete. The service continues running throughout, falling back to CPU/API models while GPU is released.

### Implementation Details

#### 1. ModelPool Methods (`gaia-core/gaia_core/models/_model_pool_impl.py`)

Added three new methods to the ModelPool class:

```python
def release_gpu(self) -> dict:
    """
    Release GPU resources by shutting down vLLM/GPU-backed models.
    - Identifies GPU-backed models (gpu_prime, VLLM instances)
    - Calls shutdown() on each
    - Clears CUDA cache via torch.cuda.empty_cache()
    - Returns list of released models
    """

def reclaim_gpu(self) -> dict:
    """
    Reclaim GPU resources by reloading vLLM models.
    - Reloads previously released models via _load_model_entry()
    - Re-promotes prime aliases
    - Returns list of loaded models
    """

def get_gpu_status(self) -> dict:
    """
    Get current GPU status including:
    - gpu_released: bool
    - gpu_models_loaded: list of active GPU models
    - gpu_info: memory stats (free_gb, total_gb, used_gb, utilization_pct)
    """
```

Key implementation detail: The `release_gpu()` method stores `_released_model_names` so `reclaim_gpu()` knows what to reload. It also calls `torch.cuda.empty_cache()` and `torch.cuda.synchronize()` to ensure VRAM is actually freed.

#### 2. API Endpoints (`gaia-core/gaia_core/main.py`)

Three new REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/gpu/status` | GET | Returns GPU state, loaded models, memory usage |
| `/gpu/release` | POST | Releases vLLM models, frees VRAM |
| `/gpu/reclaim` | POST | Reloads vLLM models after candidate testing |

Example responses:

```bash
# GET /gpu/status
{
    "gpu_released": false,
    "gpu_models_loaded": ["gpu_prime"],
    "all_models_loaded": ["gpu_prime", "embed", "lite"],
    "gpu_info": {
        "free_gb": 2.45,
        "total_gb": 15.92,
        "used_gb": 13.47,
        "utilization_pct": 84.6
    }
}

# POST /gpu/release
{
    "success": true,
    "released_models": ["gpu_prime", "prime"],
    "message": "Released 2 GPU model(s)"
}
```

#### 3. CLI Integration (`test_candidate.sh`)

Added new commands for GPU management:

```bash
# Check GPU status
./test_candidate.sh --gpu-status

# Manual workflow
./test_candidate.sh --release-gpu      # Step 1: Release GPU from live
./test_candidate.sh all --gpu          # Step 2: Start candidates with GPU
# ... test ...
./test_candidate.sh --stop             # Step 3: Stop candidates
./test_candidate.sh --reclaim-gpu      # Step 4: Restore live GPU

# Automated handoff (does release + start in one command)
./test_candidate.sh all --gpu-handoff
# ... test ...
./test_candidate.sh --reclaim-gpu
```

The `--gpu-handoff` command:
1. Calls `/gpu/release` on live service
2. Waits for CUDA to release resources
3. Starts candidates with `GAIA_CANDIDATE_GPU=1`

The `--reclaim-gpu` command:
1. Stops any running candidate containers
2. Waits briefly for CUDA cleanup
3. Calls `/gpu/reclaim` on live service

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE SERVICE STATE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Normal Operation:                                           │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │  gaia-core   │    │    GPU       │                       │
│  │  (vLLM)      │◄───│  (occupied)  │                       │
│  └──────────────┘    └──────────────┘                       │
│                                                              │
│  After /gpu/release:                                         │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │  gaia-core   │    │    GPU       │                       │
│  │  (CPU only)  │    │   (FREE!)    │◄─── Candidates can    │
│  └──────────────┘    └──────────────┘     now claim this    │
│                                                              │
│  After /gpu/reclaim:                                         │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │  gaia-core   │    │    GPU       │                       │
│  │  (vLLM)      │◄───│  (occupied)  │                       │
│  └──────────────┘    └──────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Testing Verification

```bash
$ ./test_candidate.sh --gpu-status
[INFO] GPU status from live gaia-core:
{
    "gpu_released": false,
    "gpu_models_loaded": [],
    "all_models_loaded": ["embed"],
    "model_status": {},
    "gpu_info": {
        "free_gb": 14.35,
        "total_gb": 15.92,
        "used_gb": 1.57,
        "utilization_pct": 9.9
    }
}
```

### Design Decisions

1. **Why not just stop the container?**
   - Service continuity: live endpoints remain available
   - Faster iteration: no container restart overhead
   - State preservation: session state, connections maintained

2. **Why track released model names?**
   - Allows selective reclaim (only reload what was released)
   - Handles alias relationships (prime → gpu_prime)
   - Future: could support partial GPU release

3. **Why call torch.cuda.empty_cache()?**
   - vLLM shutdown doesn't guarantee VRAM release
   - Python GC may not run immediately
   - Explicit cache clear ensures CUDA memory is freed

### Future Considerations

- **Automatic handoff**: Could detect when candidate needs GPU and auto-release
- **Partial GPU sharing**: vLLM's `gpu_memory_utilization` could allow coexistence
- **Health monitoring**: Warn if reclaim fails, suggest container restart
- **Multi-GPU**: Current impl assumes single GPU; could extend to selective device release

---

## Files Changed

| File | Changes |
|------|---------|
| `gaia-core/gaia_core/models/_model_pool_impl.py` | +`release_gpu()`, `reclaim_gpu()`, `get_gpu_status()` |
| `gaia-core/gaia_core/main.py` | +`/gpu/release`, `/gpu/reclaim`, `/gpu/status` endpoints |
| `test_candidate.sh` | +`--gpu-status`, `--release-gpu`, `--reclaim-gpu`, `--gpu-handoff` commands |

---

## Session Notes

- GPU handoff system implemented and tested
- Candidates can now have full GPU capability without stopping live service
- CLI provides both manual step-by-step and automated handoff workflows

---

## Proposal: Groq as Free API Fallback Oracle

### The Opportunity

[Groq](https://groq.com/) offers **free API access** to high-quality models like `llama-3.3-70b-versatile`. This presents an opportunity to add a zero-cost fallback when:

1. Local GPU inference fails (OOM, CUDA errors, model not loaded)
2. GPU has been released for candidate testing
3. User wants to conserve local GPU for other tasks

### Groq API Overview

```python
from groq import Groq

client = Groq(api_key="gsk_...")

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=1024,
    stream=False,
)
print(completion.choices[0].message.content)
```

**Key Points:**
- OpenAI-compatible API (uses same request/response format)
- Free tier with generous rate limits
- Models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`
- Very fast inference (Groq's custom LPU hardware)

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL FALLBACK CHAIN                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Request for "prime" model inference                         │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────┐                   │
│  │ 1. gpu_prime (vLLM local)            │ ◄── First choice  │
│  │    - Fast, private, no API costs     │                   │
│  └──────────────────┬───────────────────┘                   │
│                     │ if unavailable/fails                   │
│                     ▼                                         │
│  ┌──────────────────────────────────────┐                   │
│  │ 2. groq_fallback (Groq API)          │ ◄── Free fallback │
│  │    - llama-3.3-70b-versatile         │                   │
│  │    - Zero cost, fast, good quality   │                   │
│  └──────────────────┬───────────────────┘                   │
│                     │ if unavailable/fails                   │
│                     ▼                                         │
│  ┌──────────────────────────────────────┐                   │
│  │ 3. oracle_openai (OpenAI API)        │ ◄── Paid fallback │
│  │    - GPT-4, most capable             │                   │
│  │    - Costs money per token           │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### 1. New Model Class: `GroqAPIModel`

Create `gaia-core/gaia_core/models/groq_model.py`:

```python
import logging
import time
from typing import List, Dict, Any

class GroqAPIModel:
    """
    Groq API wrapper for free LLM inference.
    Uses the Groq Python SDK (OpenAI-compatible).
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        import os
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Groq API key missing (set GROQ_API_KEY)")

        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs,
    ):
        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        duration = time.time() - start
        self.logger.debug(f"Groq request duration: {duration:.2f}s")

        if stream:
            return self._stream_response(response)

        content = response.choices[0].message.content
        self._log_usage(response)

        return {"choices": [{"message": {"content": content}}]}

    def _stream_response(self, response_stream):
        content = ""
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content += delta
        return {"choices": [{"message": {"content": content}}]}

    def _log_usage(self, response):
        try:
            usage = response.usage
            self.logger.info(
                f"Groq Token Usage - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, Total: {usage.total_tokens}"
            )
        except Exception:
            pass
```

#### 2. Configuration Changes

Add to `MODEL_CONFIGS` (in `gaia_constants.json` or environment):

```json
{
  "MODEL_CONFIGS": {
    "groq_fallback": {
      "type": "groq",
      "model": "llama-3.3-70b-versatile",
      "enabled": true,
      "fallback_priority": 2
    },
    "oracle_openai": {
      "type": "api",
      "model": "gpt-4-turbo",
      "enabled": true,
      "fallback_priority": 3
    }
  }
}
```

#### 3. Environment Variables

Add to `docker-compose.yml`:

```yaml
environment:
  - GROQ_API_KEY=${GROQ_API_KEY:-}
```

#### 4. ModelPool Integration

Modify `_model_pool_impl.py` to:

1. Import GroqAPIModel
2. Handle `type: "groq"` in `_load_model_entry()`
3. Add fallback chain logic when GPU model fails

```python
# In _load_model_entry()
elif model_type == 'groq':
    logger.info(f"🔹 Loading Groq model {model_name}")
    from .groq_model import GroqAPIModel
    api_key = os.getenv("GROQ_API_KEY")
    model_id = model_config.get("model", "llama-3.3-70b-versatile")
    self.models[model_name] = GroqAPIModel(model_name=model_id, api_key=api_key)
```

#### 5. Automatic Fallback Logic

Add to `acquire_model_for_role()`:

```python
def acquire_model_for_role(self, role: str, lazy_load: bool = True):
    # Try primary model first
    model = self._try_acquire(role)
    if model:
        return model

    # GPU model unavailable - try fallback chain
    if role in ('prime', 'gpu_prime'):
        for fallback in ['groq_fallback', 'oracle_openai']:
            if fallback in self.models:
                logger.warning(f"Using {fallback} as fallback for {role}")
                return self.models[fallback]
            elif self.ensure_model_loaded(fallback):
                return self.models.get(fallback)

    return None
```

### Groq Model Options

| Model | Context | Speed | Use Case |
|-------|---------|-------|----------|
| `llama-3.3-70b-versatile` | 128K | Fast | General purpose, best quality |
| `llama-3.1-8b-instant` | 128K | Very fast | Quick responses, lighter tasks |
| `mixtral-8x7b-32768` | 32K | Fast | Good balance of speed/quality |

### Rate Limits (Free Tier)

- Requests per minute: 30
- Requests per day: 14,400
- Tokens per minute: 6,000
- Tokens per day: 500,000

These limits are generous for a personal assistant use case.

### Benefits

1. **Zero Cost**: No API charges for fallback inference
2. **High Quality**: Llama 3.3 70B is competitive with GPT-4 for many tasks
3. **Fast**: Groq's LPU hardware is extremely fast
4. **Privacy-Aware Fallback**: Still better than pure cloud (data not retained)
5. **Graceful Degradation**: System keeps working even if GPU fails

### Considerations

1. **API Key Required**: User needs to sign up at console.groq.com
2. **Rate Limits**: May hit limits under heavy use
3. **Internet Required**: Unlike local GPU inference
4. **Model Differences**: Llama 3.3 may behave differently than local model

### Implementation Steps

1. [ ] Create `groq_model.py` with GroqAPIModel class
2. [ ] Add `groq>=0.4.0` to `gaia-core/pyproject.toml`
3. [ ] Register Groq model type in `_model_pool_impl.py`
4. [ ] Add fallback chain logic to model acquisition
5. [ ] Add `GROQ_API_KEY` to docker-compose environment
6. [ ] Update config documentation
7. [ ] Test fallback scenarios

---

## Detailed Implementation Specification

### File: `gaia-core/gaia_core/models/groq_model.py`

```python
"""
Groq API model wrapper for GAIA.

Groq provides free, fast inference on open-source models via their custom LPU hardware.
This wrapper provides an OpenAI-compatible interface for use as a fallback when local
GPU inference is unavailable.

Environment:
    GROQ_API_KEY: API key from console.groq.com (required)
    GROQ_MODEL: Model to use (default: llama-3.3-70b-versatile)
    GROQ_TIMEOUT: Request timeout in seconds (default: 60)
"""

import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("GAIA.Groq")

# Lazy import to avoid dependency issues if groq not installed
Groq = None


def _ensure_groq_imported():
    """Lazy import of groq SDK."""
    global Groq
    if Groq is None:
        try:
            from groq import Groq as _Groq
            Groq = _Groq
        except ImportError:
            raise RuntimeError(
                "groq package not installed. Install with: pip install groq"
            )


class GroqAPIModel:
    """
    Groq API wrapper providing create_chat_completion interface.

    Compatible with GAIA's model pool and can serve as a fallback for gpu_prime.

    Attributes:
        model_name: Groq model identifier (e.g., "llama-3.3-70b-versatile")
        api_key: Groq API key
        timeout: Request timeout in seconds
    """

    # Available models and their characteristics
    AVAILABLE_MODELS = {
        "llama-3.3-70b-versatile": {
            "context_window": 128000,
            "description": "Best quality, general purpose",
            "tokens_per_minute": 6000,
        },
        "llama-3.1-70b-versatile": {
            "context_window": 128000,
            "description": "Previous gen, still excellent",
            "tokens_per_minute": 6000,
        },
        "llama-3.1-8b-instant": {
            "context_window": 128000,
            "description": "Fast, good for simple tasks",
            "tokens_per_minute": 20000,
        },
        "mixtral-8x7b-32768": {
            "context_window": 32768,
            "description": "MoE model, good balance",
            "tokens_per_minute": 5000,
        },
        "gemma2-9b-it": {
            "context_window": 8192,
            "description": "Google Gemma, instruction-tuned",
            "tokens_per_minute": 15000,
        },
    }

    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        timeout: int = None,
    ):
        _ensure_groq_imported()

        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.timeout = timeout or int(os.getenv("GROQ_TIMEOUT", "60"))

        if not self.api_key:
            raise RuntimeError(
                "Groq API key not configured. Set GROQ_API_KEY environment variable. "
                "Get a free key at https://console.groq.com"
            )

        if self.model_name not in self.AVAILABLE_MODELS:
            logger.warning(
                f"Model '{self.model_name}' not in known models. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.client = Groq(api_key=self.api_key, timeout=self.timeout)
        self._request_count = 0
        self._total_tokens = 0

        logger.info(f"GroqAPIModel initialized with model={self.model_name}")

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
        """
        Create a chat completion using Groq API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            stream: If True, return a generator yielding chunks
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dict with 'choices' key containing the response, or generator if streaming

        Raises:
            RuntimeError: If API call fails
        """
        self._request_count += 1
        start_time = time.time()

        # Sanitize messages - Groq is strict about format
        clean_messages = self._sanitize_messages(messages)

        # Clamp parameters to valid ranges
        temperature = max(0.0, min(2.0, float(temperature)))
        top_p = max(0.0, min(1.0, float(top_p)))
        max_tokens = max(1, min(32768, int(max_tokens)))

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=clean_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )

            duration = time.time() - start_time

            if stream:
                return self._stream_response(response, duration)

            # Extract content from response
            content = response.choices[0].message.content
            if content is None:
                content = ""

            # Log usage stats
            self._log_usage(response, duration)

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": self.model_name,
                "provider": "groq",
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Groq API error after {duration:.2f}s: {e}")
            raise RuntimeError(f"Groq API call failed: {e}") from e

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Clean messages for Groq API compatibility.

        Groq is strict about message format:
        - Only 'role' and 'content' keys allowed
        - Role must be 'system', 'user', or 'assistant'
        - Content must be string (not None)
        """
        clean = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Normalize role
            if role not in ("system", "user", "assistant"):
                role = "user"

            # Ensure content is string
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)

            # Skip empty messages (except system)
            if not content.strip() and role != "system":
                continue

            clean.append({"role": role, "content": content})

        # Groq requires at least one user message
        if not any(m["role"] == "user" for m in clean):
            clean.append({"role": "user", "content": "(continue)"})

        return clean

    def _stream_response(
        self,
        response_stream,
        start_duration: float
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process streaming response and yield chunks.

        Yields dicts compatible with GAIA's streaming interface.
        """
        content_buffer = []

        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content_buffer.append(delta.content)
                yield {
                    "choices": [{
                        "delta": {"content": delta.content},
                        "finish_reason": None,
                    }]
                }

        # Final chunk with complete content
        full_content = "".join(content_buffer)
        logger.debug(f"Groq stream complete: {len(full_content)} chars")

        yield {
            "choices": [{
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }],
            "provider": "groq",
        }

    def _log_usage(self, response, duration: float):
        """Log token usage and request stats."""
        try:
            usage = response.usage
            self._total_tokens += usage.total_tokens

            logger.info(
                f"Groq [{self.model_name}] - "
                f"Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}, "
                f"Duration: {duration:.2f}s, "
                f"Session total: {self._total_tokens} tokens"
            )
        except Exception as e:
            logger.debug(f"Could not log Groq usage: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics for this session."""
        return {
            "model": self.model_name,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }

    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """Return available models and their characteristics."""
        return cls.AVAILABLE_MODELS.copy()
```

### ModelPool Integration (`_model_pool_impl.py`)

#### 1. Add import at top (with other lazy imports):

```python
GroqAPIModel = None
try:
    from .groq_model import GroqAPIModel as _GroqAPIModel
    GroqAPIModel = _GroqAPIModel
except Exception:
    GroqAPIModel = None
```

#### 2. Add to `_load_model_entry()` (around line 619, after vllm case):

```python
elif model_type == 'groq':
    if GroqAPIModel is None:
        logger.warning("GroqAPIModel unavailable (groq package not installed)")
        return False
    logger.info(f"🔹 Loading Groq model {model_name}")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning(f"GROQ_API_KEY not set; skipping {model_name}")
        return False
    model_id = model_config.get("model", "llama-3.3-70b-versatile")
    self.models[model_name] = GroqAPIModel(model_name=model_id, api_key=api_key)
```

#### 3. Add fallback chain to `acquire_model_for_role()`:

```python
def acquire_model_for_role(self, role: str, lazy_load: bool = True):
    """Resolve role to a model name and acquire it (mark busy)."""
    name = self._resolve_model_name_for_role(role)

    # If no model resolved, try lazy loading first
    if not name and lazy_load:
        if self.ensure_model_loaded(role):
            name = self._resolve_model_name_for_role(role)

    # FALLBACK CHAIN: If primary model unavailable, try fallbacks
    if not name or name not in self.models:
        if role in ('prime', 'gpu_prime', 'cpu_prime'):
            fallback_chain = ['groq_fallback', 'oracle_openai', 'oracle_gemini']
            for fallback in fallback_chain:
                if fallback in self.models:
                    logger.warning(f"🔄 Using {fallback} as fallback for {role}")
                    self.set_status(fallback, "busy")
                    return self.models[fallback]
                # Try lazy loading the fallback
                if self.ensure_model_loaded(fallback):
                    if fallback in self.models:
                        logger.warning(f"🔄 Loaded {fallback} as fallback for {role}")
                        self.set_status(fallback, "busy")
                        return self.models[fallback]
            logger.error(f"No fallback available for {role}")
            return None

    if not name:
        logger.error("ModelPool.acquire_model_for_role: no model resolved for role '%s'", role)
        return None

    model = self.models.get(name)

    # If model not in pool, try lazy loading
    if not model and lazy_load:
        if self.ensure_model_loaded(name):
            model = self.models.get(name)

    if not model:
        logger.error("ModelPool.acquire_model_for_role: resolved model '%s' missing from pool", name)
        return None
    self.set_status(name, "busy")
    return model
```

### Docker Compose Changes

#### `docker-compose.yml` - Add to gaia-core environment:

```yaml
gaia-core:
  environment:
    # ... existing vars ...
    - GROQ_API_KEY=${GROQ_API_KEY:-}
    - GROQ_MODEL=${GROQ_MODEL:-llama-3.3-70b-versatile}
```

#### `.env.example` - Add documentation:

```bash
# Groq API (free fallback for GPU inference)
# Get your free key at https://console.groq.com
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### pyproject.toml Changes

Add to dependencies:

```toml
dependencies = [
    # ... existing deps ...
    "groq>=0.4.0",  # Free API fallback
]
```

Or as optional dependency:

```toml
[project.optional-dependencies]
api = [
    "groq>=0.4.0",
    "openai>=1.0.0",
    "google-generativeai>=0.3.0",
]
```

### Configuration Example

Add to `gaia_constants.json`:

```json
{
  "MODEL_CONFIGS": {
    "gpu_prime": {
      "type": "vllm",
      "path": "/models/your-local-model",
      "enabled": true
    },
    "groq_fallback": {
      "type": "groq",
      "model": "llama-3.3-70b-versatile",
      "enabled": true,
      "description": "Free Groq API fallback for when GPU unavailable"
    },
    "oracle_openai": {
      "type": "api",
      "provider": "openai",
      "model": "gpt-4-turbo",
      "enabled": false
    }
  }
}
```

---

## Error Handling & Edge Cases

### 1. Rate Limiting

Groq free tier has limits. Handle gracefully:

```python
from groq import RateLimitError

try:
    response = self.client.chat.completions.create(...)
except RateLimitError as e:
    logger.warning(f"Groq rate limit hit: {e}")
    # Could trigger next fallback in chain
    raise RuntimeError("Groq rate limited") from e
```

### 2. API Key Missing

If `GROQ_API_KEY` not set:
- Model won't load (returns False from `_load_model_entry`)
- Fallback chain skips it and tries next option
- No crash, graceful degradation

### 3. Network Errors

```python
from groq import APIConnectionError, APITimeoutError

try:
    response = self.client.chat.completions.create(...)
except APIConnectionError:
    logger.error("Cannot reach Groq API - network issue")
    raise
except APITimeoutError:
    logger.error(f"Groq request timed out after {self.timeout}s")
    raise
```

### 4. Model Changes

Groq may deprecate models. The `AVAILABLE_MODELS` dict serves as documentation but doesn't enforce - unknown models log a warning but still attempt the call.

---

## Testing Plan

### Unit Tests

```python
# test_groq_model.py

import pytest
from unittest.mock import Mock, patch

def test_groq_init_without_key():
    """Should raise if no API key."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(RuntimeError, match="API key"):
            from gaia_core.models.groq_model import GroqAPIModel
            GroqAPIModel()

def test_groq_sanitize_messages():
    """Should clean messages for Groq format."""
    model = Mock()
    model._sanitize_messages = GroqAPIModel._sanitize_messages

    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": None},  # Should become ""
        {"role": "function", "content": "test"},  # Should become "user"
    ]

    clean = model._sanitize_messages(model, messages)
    assert all(m["role"] in ("system", "user", "assistant") for m in clean)
    assert all(isinstance(m["content"], str) for m in clean)

def test_fallback_chain_triggers():
    """Should use Groq when GPU model unavailable."""
    # Test that acquire_model_for_role falls back correctly
    pass
```

### Integration Tests

```bash
# Test 1: Direct Groq call
curl -X POST http://localhost:6415/test/groq \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello in 5 words"}'

# Test 2: Fallback triggers on GPU release
./test_candidate.sh --release-gpu
./test_candidate.sh --gpu-status  # Should show gpu_released: true

# Send request - should use Groq (check logs)
curl -X POST http://localhost:6415/process_packet \
  -H "Content-Type: application/json" \
  -d '{"header": {...}, "content": {"original_prompt": "Hello"}}'

# Verify in logs: "Using groq_fallback as fallback for prime"

# Test 3: Reclaim restores GPU
./test_candidate.sh --reclaim-gpu
# Next request should use local GPU again
```

### Load Testing

```bash
# Verify rate limits don't crash system
for i in {1..50}; do
  curl -X POST http://localhost:6415/process_packet -d '...' &
done
wait
# Should see some rate limit warnings but no crashes
```

---

## Monitoring & Observability

### Logging

The GroqAPIModel logs:
- Request count and token usage per request
- Session totals
- Warnings for rate limits, unknown models
- Errors for API failures

### Metrics Endpoint

Could add `/metrics/groq`:

```json
{
  "model": "llama-3.3-70b-versatile",
  "session_requests": 47,
  "session_tokens": 12450,
  "rate_limit_hits": 2,
  "errors": 0
}
```

### Health Check

Extend `/health` to include fallback status:

```json
{
  "status": "healthy",
  "gpu_available": false,
  "fallback_available": true,
  "fallback_model": "groq:llama-3.3-70b-versatile"
}
```

---

## Security Considerations

1. **API Key Storage**: Use environment variables, never commit to repo
2. **Data Privacy**: Groq processes prompts on their servers (unlike local GPU)
3. **Rate Limits**: Prevent abuse by internal rate limiting if needed
4. **Error Messages**: Don't leak API keys in error logs

---

## Ready to Implement

Once you have the Groq API key (`gsk_...`), set it via:

```bash
# Option 1: Environment variable
export GROQ_API_KEY="gsk_your_key_here"

# Option 2: .env file for docker-compose
echo "GROQ_API_KEY=gsk_your_key_here" >> .env

# Option 3: Docker compose override
GROQ_API_KEY=gsk_... docker compose up -d gaia-core
```

Get your free key at: **https://console.groq.com**
