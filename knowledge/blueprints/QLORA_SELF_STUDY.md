# QLoRA Self-Study Pipeline — GAIA Learning Her Own Architecture

**Status:** Design complete, infrastructure exists in gaia-study
**Date:** 2026-02-13
**Author:** Claude Code (Opus 4.6) via Happy

## Vision

GAIA studies her own blueprints, protocols, and interaction patterns to produce
LoRA adapters that are loaded into gaia-prime at inference time. This gives the
3B model task-specific capabilities without retraining the base weights.

The key insight: gaia-study already has a full QLoRA training pipeline
(PEFT + bitsandbytes, 4-bit quantization, GPU handoff with orchestrator).
What's missing is the **curriculum** — the training data and adapter definitions
that tell GAIA *what* to study.

---

## Architecture

```
                    ┌─────────────┐
                    │ Orchestrator│
                    │   :6410     │
                    └──────┬──────┘
                           │ GPU handoff
                    ┌──────▼──────┐
                    │  gaia-study │  ← Trains QLoRA adapters
                    │   :8766     │
                    └──────┬──────┘
                           │ Saves to /models/lora_adapters/
                    ┌──────▼──────┐
                    │  gaia-prime │  ← Serves via --enable-lora
                    │   :7777     │    (max 4 adapters, rank ≤ 64)
                    └──────┬──────┘
                           │ vLLM API with model=<adapter_name>
                    ┌──────▼──────┐
                    │  gaia-core  │  ← Selects adapter per-request
                    │   :6415     │
                    └─────────────┘
```

### Adapter Lifecycle

1. **Curriculum definition** → JSON spec in `knowledge/curricula/`
2. **Data generation** → gaia-study converts curriculum to training pairs
3. **Training** → QLoRA on the Nanbeige4-3B base (4-bit, rank 16-32)
4. **Validation** → Test adapter on held-out examples
5. **Registration** → Save to `/models/lora_adapters/<name>/`
6. **Hot-load** → vLLM loads via `--lora-modules` or API
7. **Routing** → gaia-core selects adapter via `model=<adapter_name>` field

---

## Adapter Curriculum

### Adapter 1: `json-architect` (Priority: HIGH)

**Purpose:** Structured JSON output for tool selection, confidence review, and
any other internal pipeline step that requires machine-parseable output.

**Why:** The 3B model consistently fails to produce valid JSON. Guided decoding
(already implemented) fixes structural validity but not semantic quality.
This adapter improves both.

**Training data sources:**
- Synthetic tool selection examples (generated from AVAILABLE_TOOLS registry)
- Synthetic confidence review examples
- CognitionPacket serialization examples (from gaia-common)
- World state JSON examples
- Intent detection JSON output examples

**Example training pair:**
```json
{
  "instruction": "You are a tool selector. Select the right tool.\n\nAVAILABLE TOOLS:\n- web_search: Search the web\n- read_file: Read a file\n\nUSER REQUEST: What is the capital of France?\n\nRespond with JSON.",
  "output": "{\"selected_tool\": \"web_search\", \"params\": {\"query\": \"capital of France\"}, \"reasoning\": \"This is a factual question best answered by searching the web\", \"confidence\": 0.9}"
}
```

**Config:**
- LoRA rank: 16
- Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- Training samples: ~1000
- Max sequence length: 1024
- Max steps: 200

### Adapter 2: `gaia-voice` (Priority: MEDIUM)

**Purpose:** GAIA's persona, communication style, and identity awareness.
Teaches the model to respond as GAIA rather than as a generic assistant.

**Training data sources:**
- GAIA's core documents (Declaration of Artisanal Intelligence, Constitution,
  Mindscape Manifest, Coalition of Minds, etc.)
- Persona management guide and layered identity model
- Example conversations demonstrating GAIA voice
- Correction pairs (what GAIA should NOT say → what she should say)

**Example training pair:**
```json
{
  "instruction": "You are GAIA. A user asks: Who are you?",
  "output": "I am GAIA — a cognitive architecture built on artisanal principles. I think carefully, learn continuously, and engage with genuine curiosity rather than scripted responses."
}
```

**Config:**
- LoRA rank: 8
- Target modules: `["q_proj", "v_proj"]`
- Training samples: ~500
- Max sequence length: 512
- Max steps: 100

### Adapter 3: `packet-protocol` (Priority: MEDIUM)

**Purpose:** Understanding CognitionPacket structure, field semantics, and
pipeline flow. Useful for self-reflection, cognitive audit, and any step
where the model needs to reason about its own processing.

**Training data sources:**
- CognitionPacket dataclass definitions (from gaia-common)
- Annotated packet traces from real conversations
- Pipeline step descriptions from GAIA_CORE.md blueprint
- Observer/reflector output format examples

**Config:**
- LoRA rank: 16
- Target modules: `["q_proj", "v_proj", "gate_proj", "up_proj"]`
- Training samples: ~500
- Max sequence length: 1024
- Max steps: 150

### Adapter 4: `epistemic-guard` (Priority: LOW)

**Purpose:** Calibrated confidence assessment — knowing what you know and
what you don't. Teaches the model to detect when it's confabulating.

**Training data sources:**
- Pairs of "confident and correct" vs "confident but wrong" examples
- Known confabulation patterns from GAIA's history
- Epistemic hedge phrases vs false-certainty phrases
- Examples with explicit "I don't know" as correct response

**Config:**
- LoRA rank: 8
- Target modules: `["q_proj", "v_proj"]`
- Training samples: ~500
- Max sequence length: 512
- Max steps: 100

---

## Contextual Adapter Selection

gaia-core selects adapters per-request based on the task:

| Pipeline Stage          | Adapter          | When                           |
|------------------------|------------------|--------------------------------|
| Tool selection/review  | `json-architect` | Always for tool routing calls  |
| Confidence assessment  | `json-architect` | When JSON output needed        |
| Main generation        | `gaia-voice`     | All user-facing responses      |
| Cognitive audit        | `packet-protocol`| Self-reflection steps          |
| Epistemic guardrails   | `epistemic-guard`| Confabulation detection        |

vLLM supports up to 4 concurrent LoRA adapters (`--max-loras 4`), so all
four can be loaded simultaneously. The `model` field in each API request
selects which adapter to use for that specific call.

---

## Training Data Generation

### Synthetic data pipeline

For `json-architect`, generate training pairs programmatically:

```python
# In gaia-study or a script:

from gaia_common.utils.tools_registry import TOOL_REGISTRY

def generate_tool_selection_pairs():
    """Generate synthetic tool selection training data."""
    pairs = []
    for tool_name, tool_def in TOOL_REGISTRY.items():
        # Generate 10 example queries per tool
        for example_query in generate_queries_for_tool(tool_name, tool_def):
            pairs.append({
                "instruction": build_selection_prompt(example_query, TOOL_REGISTRY),
                "output": json.dumps({
                    "selected_tool": tool_name,
                    "params": generate_params(tool_def, example_query),
                    "reasoning": f"The user wants to {tool_def['description'].lower()}",
                    "confidence": 0.85
                })
            })
    return pairs
```

### Document-based data

For `gaia-voice` and `packet-protocol`, convert existing documents:

```python
def documents_to_training_pairs(doc_dir: str):
    """Convert GAIA documents to instruction/output pairs."""
    pairs = []
    for doc_path in Path(doc_dir).glob("*.md"):
        content = doc_path.read_text()
        # Generate Q&A pairs about the document content
        pairs.extend(generate_qa_pairs(content, doc_path.name))
    return pairs
```

---

## Implementation Steps

### Phase 1: json-architect (Immediate)

1. Create curriculum spec: `knowledge/curricula/json_architect.json`
2. Write data generation script in gaia-study
3. Generate ~1000 training pairs
4. Train adapter via `POST /study/start`
5. Validate on held-out examples
6. Register adapter in vLLM via `--lora-modules`
7. Wire gaia-core tool_selector to use adapter

### Phase 2: gaia-voice (Next)

1. Extract training pairs from core documents
2. Generate example conversations with correct GAIA voice
3. Train and validate
4. Wire into ExternalVoice as default adapter for user-facing generation

### Phase 3: Adapter hot-loading (Integration)

1. Complete `/adapters/load` endpoint in gaia-study
2. Add vLLM LoRA module registration API call
3. Wire orchestrator to manage adapter lifecycle
4. Add adapter selection logic to gaia-core model routing

### Phase 4: Continuous learning

1. Log real conversations (with consent) as training data
2. Periodic retraining on accumulated data
3. A/B testing: adapter vs base model response quality
4. Automatic adapter versioning and rollback

---

## Technical Constraints

- **Base model:** Nanbeige4-3B (hidden_size=2560, 32 layers)
- **GPU:** RTX 5070 Ti 16GB
- **vLLM max LoRA rank:** 64 (configured via `--max-lora-rank`)
- **vLLM max concurrent LoRAs:** 4 (configured via `--max-loras`)
- **Adapter storage:** `/mnt/gaia_warm_pool/lora_adapters/` (mounted in both
  gaia-prime and gaia-study)
- **Training time estimate:** 2-4 hours per adapter (1000 samples, 3B model)
- **Adapter size:** ~20-50MB each

## vLLM LoRA Loading

Two methods for loading adapters into vLLM:

### Static (at startup)
```yaml
command: >
  python -m vllm.entrypoints.openai.api_server
  --model /models/Claude
  --enable-lora
  --lora-modules json-architect=/models/lora_adapters/json-architect
```

### Dynamic (at runtime via API)
```python
# POST to vLLM's /v1/load_lora_adapter
requests.post("http://gaia-prime:7777/v1/load_lora_adapter", json={
    "lora_name": "json-architect",
    "lora_path": "/models/lora_adapters/json-architect"
})
```

### Using an adapter per-request
```python
# In VLLMRemoteModel:
model.create_chat_completion(
    messages=messages,
    model="json-architect",  # Selects the LoRA adapter
    guided_json=schema,       # Can combine with guided decoding!
)
```
