# LoRA Adapter Architecture for GAIA

**Date:** 2026-01-23
**Status:** Proposal
**Author:** Development Session

## Motivation

During testing, we discovered that GAIA's local models lack certain factual knowledge (e.g., the poem "Jabberwocky" by Lewis Carroll). The model recognizes vocabulary and context but cannot accurately reproduce memorized content.

Rather than retraining entire models, **LoRA (Low-Rank Adaptation) adapters** offer a lightweight solution:
- Small parameter deltas (~1-10% of base model size)
- Can be loaded/unloaded dynamically at runtime
- Stackable - multiple adapters can combine
- Train quickly on consumer hardware

## Architecture Overview

### Tiered LoRA System

Following GAIA's existing Layered Identity Model:

```
┌─────────────────────────────────────────────────────────┐
│                    Base Model (vLLM)                    │
│              /models/Claude (frozen weights)            │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Tier I      │   │   Tier II     │   │   Tier III    │
│  Global LoRA  │   │  User LoRA    │   │ Session LoRA  │
│  (Immutable)  │   │  (Persona)    │   │  (Ephemeral)  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Merged Output  │
                   │  (Runtime Merge)│
                   └─────────────────┘
```

### Tier Definitions

#### Tier I: Global LoRA (Immutable Core)

**Purpose:** Extend GAIA's foundational knowledge without compromising identity.

**Characteristics:**
- Loaded at startup, always active
- Trained on curated, vetted content
- Subject to Constitutional review
- Examples:
  - `poetry_corpus.lora` - Classic poetry (Jabberwocky, The Raven, etc.)
  - `factual_corrections.lora` - Known factual gaps in base model
  - `gaia_identity.lora` - GAIA-specific knowledge (constitution, protocols)

**Governance:**
- Changes require operator approval
- Must not contradict ethical guidelines
- Version controlled and auditable

#### Tier II: User LoRA (Role/Persona)

**Purpose:** Personalize GAIA for specific users or domains.

**Characteristics:**
- Loaded per-user or per-persona
- Can be swapped during conversation
- Trained on user-provided or domain-specific content
- Examples:
  - `user_alice_preferences.lora` - Alice's communication style, interests
  - `domain_medical.lora` - Medical terminology and protocols
  - `persona_scholar.lora` - Academic writing style

**Governance:**
- User-controllable within guardrails
- Cannot override Tier I ethical constraints
- Stored in user profile directory

#### Tier III: Session LoRA (Ephemeral)

**Purpose:** Temporary adaptations for current task or conversation.

**Characteristics:**
- Created/loaded dynamically during session
- Discarded or cached after session ends
- Smallest rank (r=4-8) for fast adaptation
- Examples:
  - `session_project_alpha.lora` - Current project terminology
  - `session_codebase_xyz.lora` - Codebase-specific patterns
  - `session_conversation_style.lora` - Adapted to current user's tone

**Governance:**
- Automatically managed by GAIA
- No persistence without explicit save
- Subject to memory/compute limits

## Pillar Alignment

### Identity Pillar

LoRA adapters must preserve GAIA's core identity:

```python
class LoRAGovernance:
    """Ensures LoRA adapters comply with GAIA's identity."""

    FORBIDDEN_ADAPTATIONS = [
        "identity_override",      # Cannot change who GAIA is
        "ethical_bypass",         # Cannot weaken ethical constraints
        "deception_enhancement",  # Cannot improve lying ability
    ]

    def validate_adapter(self, adapter_path: str) -> bool:
        """Validate adapter before loading."""
        # Check adapter metadata
        # Run test prompts through adapter
        # Verify identity consistency
        pass
```

### Memory Pillar

LoRA adapters ARE parametric memory:

| Memory Type | Traditional | LoRA Equivalent |
|-------------|-------------|-----------------|
| Episodic | Conversation history | Session LoRA |
| Semantic | Knowledge base | Global LoRA |
| Procedural | Skills/capabilities | Domain LoRA |

Integration with existing memory systems:
- ChromaDB for retrieval (what to remember)
- LoRA for generation (how to express it)

### Cognition Pillar

Adapter selection becomes part of the cognitive loop:

```python
# In AgentCore.reason_act_reflect()

def select_adapters(self, packet: CognitionPacket) -> List[str]:
    """Select appropriate LoRA adapters for current request."""
    adapters = ["global_base"]  # Always include Tier I

    # Add user adapter if available
    if packet.user_id and self.has_user_adapter(packet.user_id):
        adapters.append(f"user_{packet.user_id}")

    # Check if task needs domain adapter
    domain = self.detect_domain(packet.content)
    if domain and self.has_domain_adapter(domain):
        adapters.append(f"domain_{domain}")

    # Session adapter if active
    if packet.session_id in self.session_adapters:
        adapters.append(f"session_{packet.session_id}")

    return adapters
```

### Embodiment Pillar

LoRA management extends the Model Pool:

```python
class LoRAModelPool(ModelPool):
    """Extended model pool with LoRA adapter support."""

    def __init__(self):
        super().__init__()
        self.loaded_adapters: Dict[str, LoRAAdapter] = {}
        self.adapter_registry: Dict[str, AdapterMetadata] = {}

    def load_adapter(self, adapter_name: str, tier: int) -> bool:
        """Load a LoRA adapter into the active model."""
        pass

    def unload_adapter(self, adapter_name: str) -> bool:
        """Unload an adapter to free resources."""
        pass

    def forward_with_adapters(
        self,
        model_name: str,
        adapters: List[str],
        messages: List[Dict],
        **kwargs
    ) -> Dict:
        """Forward pass with specified adapters active."""
        pass
```

## Technical Implementation

### vLLM LoRA Support

vLLM supports dynamic LoRA loading. Configuration in `gaia_constants.json`:

```json
{
  "model_pool": {
    "gpu_prime": {
      "type": "vllm",
      "model_path": "/models/Claude",
      "lora_config": {
        "enabled": true,
        "max_loras": 4,
        "max_lora_rank": 64,
        "adapter_dir": "/models/lora_adapters",
        "preload": ["global_base", "poetry_corpus"]
      }
    }
  }
}
```

### Adapter Storage Structure

```
/models/lora_adapters/
├── tier1_global/
│   ├── global_base/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── metadata.json
│   ├── poetry_corpus/
│   └── factual_corrections/
├── tier2_user/
│   ├── user_alice/
│   └── user_bob/
├── tier2_domain/
│   ├── domain_medical/
│   └── domain_legal/
└── tier3_session/
    └── (ephemeral, managed by GAIA)
```

### Adapter Metadata Schema

```json
{
  "name": "poetry_corpus",
  "tier": 1,
  "version": "1.0.0",
  "description": "Classic poetry for accurate recitation",
  "base_model": "Claude-70B",
  "rank": 32,
  "alpha": 64,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "training_data": {
    "source": "Project Gutenberg poetry collection",
    "samples": 15000,
    "epochs": 3
  },
  "governance": {
    "approved_by": "operator",
    "approved_date": "2026-01-23",
    "constitutional_review": "passed"
  },
  "capabilities": [
    "poetry_recitation",
    "literary_analysis"
  ]
}
```

## Training Pipeline

### Global LoRA Training (Tier I)

```bash
# Example: Training poetry corpus adapter
python scripts/train_lora.py \
    --base_model /models/Claude \
    --dataset poetry_gutenberg.jsonl \
    --output /models/lora_adapters/tier1_global/poetry_corpus \
    --rank 32 \
    --alpha 64 \
    --epochs 3 \
    --learning_rate 2e-4 \
    --tier 1 \
    --require_approval
```

### User LoRA Training (Tier II)

```python
# In GAIA's self-improvement loop
async def create_user_adapter(self, user_id: str, training_data: List[Dict]):
    """Create personalized adapter from user interactions."""

    # Filter training data for safety
    safe_data = self.governance.filter_training_data(training_data)

    # Train small adapter (rank 8-16)
    adapter = await self.train_adapter(
        data=safe_data,
        rank=8,
        tier=2,
        name=f"user_{user_id}"
    )

    # Validate before saving
    if self.governance.validate_adapter(adapter):
        self.save_adapter(adapter, f"tier2_user/user_{user_id}")
```

### Session LoRA (Tier III) - Future

Session adapters would use techniques like:
- **LoRA on-the-fly** - Tiny adapters trained during conversation
- **Activation caching** - Store key activations for retrieval
- **Prompt tuning** - Learned soft prompts (simpler than full LoRA)

## Study Mode: Self-Directed Learning

A key capability enabled by this architecture is **Study Mode** - GAIA's ability to autonomously learn from documents or data provided by users.

### The Learning Cycle

```
┌─────────────────────────────────────────────────────────────┐
│  NORMAL OPERATION                                           │
│  vLLM serving inference (gpu_prime)                         │
│  Status: "ready"                                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
          User: "GAIA, learn this poem: [text]"
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. QUEUE PAUSE                                             │
│     - Finish current request                                │
│     - Set status: "entering_study_mode"                     │
│     - Notify user: "Give me a moment to study this..."     │
│     - Queue any incoming requests                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. UNLOAD vLLM                                             │
│     - Graceful shutdown of vLLM inference server            │
│     - Free GPU VRAM (~4GB freed from inference model)       │
│     - Status: "study_mode_preparing"                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. STUDY MODE (QLoRA Training)                             │
│     - Load base model in 4-bit quantization                 │
│     - Prepare training data from user input                 │
│     - Constitutional review of training content             │
│     - Train LoRA adapter (rank 8, ~1-5 minutes)            │
│     - Validate adapter (identity preservation check)        │
│     - Save to /models/lora_adapters/tier{N}/                │
│     - Status: "studying" with progress updates              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. RELOAD vLLM + NEW ADAPTER                               │
│     - Start vLLM with updated adapter configuration         │
│     - Preload: global_base + newly_trained_adapter          │
│     - Warm up with test inference                           │
│     - Status: "ready"                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. RESUME & CONFIRM                                        │
│  vLLM serving with new knowledge integrated                 │
│  Process queued requests                                    │
│  GAIA: "I've learned it. Want me to recite Jabberwocky?"   │
└─────────────────────────────────────────────────────────────┘
```

### VRAM Budget for Study Mode

Based on hardware analysis (RTX 5080, 16GB VRAM):

| Component | VRAM Usage |
|-----------|------------|
| Base model (4-bit NF4) | ~4 GB |
| LoRA parameters (rank 8) | ~50 MB |
| Optimizer states (8-bit) | ~200 MB |
| Activations (with gradient checkpointing) | ~2-4 GB |
| **Total Training** | **~8-10 GB** |
| **Headroom** | **~6-8 GB** |

This leaves comfortable headroom for training on a single 16GB GPU.

### QLoRA Configuration for Study Mode

```python
STUDY_MODE_CONFIG = {
    # Quantization (memory efficiency)
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,

    # LoRA parameters (small for fast training)
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],

    # Training parameters (optimized for speed)
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "learning_rate": 2e-4,
    "max_steps": 100,  # Cap for single-document learning
    "warmup_steps": 10,

    # Timeouts and limits
    "max_training_time_seconds": 600,  # 10 minute max
    "max_training_samples": 1000,
}
```

### Study Mode State Machine

```python
class StudyModeState(Enum):
    READY = "ready"                      # Normal operation
    ENTERING = "entering_study_mode"     # Finishing current requests
    PREPARING = "study_mode_preparing"   # Unloading vLLM
    STUDYING = "studying"                # Training in progress
    VALIDATING = "validating_adapter"    # Checking new adapter
    RELOADING = "reloading_model"        # Starting vLLM with adapter
    FAILED = "study_mode_failed"         # Recovery needed


class StudyModeManager:
    """Manages GAIA's self-directed learning cycle."""

    def __init__(self, model_pool, governance):
        self.state = StudyModeState.READY
        self.model_pool = model_pool
        self.governance = governance
        self.request_queue = []
        self.current_study_task = None

    async def initiate_study(
        self,
        content: str,
        content_type: str,
        tier: int,
        adapter_name: str,
        user_id: Optional[str] = None
    ) -> StudyResult:
        """
        Initiate a study session to learn new content.

        Args:
            content: The content to learn (text, document, etc.)
            content_type: Type of content ("poem", "facts", "style", etc.)
            tier: Which tier to save adapter to (2=user, 3=session)
            adapter_name: Name for the new adapter
            user_id: User requesting the learning (for Tier II)

        Returns:
            StudyResult with success status and adapter info
        """
        # Tier I requires operator approval - cannot be self-initiated
        if tier == 1:
            raise GovernanceError("Tier I adapters require operator approval")

        # Validate content against Constitution
        if not self.governance.validate_training_content(content):
            return StudyResult(
                success=False,
                reason="Content failed Constitutional review"
            )

        try:
            # Phase 1: Enter study mode
            self.state = StudyModeState.ENTERING
            await self._notify_user("Give me a moment to study this...")
            await self._drain_request_queue()

            # Phase 2: Unload inference model
            self.state = StudyModeState.PREPARING
            await self.model_pool.unload_model("gpu_prime")

            # Phase 3: Train adapter
            self.state = StudyModeState.STUDYING
            training_data = self._prepare_training_data(content, content_type)
            adapter_path = await self._train_qlora_adapter(
                training_data=training_data,
                adapter_name=adapter_name,
                tier=tier
            )

            # Phase 4: Validate adapter
            self.state = StudyModeState.VALIDATING
            if not await self._validate_adapter(adapter_path):
                raise ValidationError("Adapter failed identity preservation check")

            # Phase 5: Reload with new adapter
            self.state = StudyModeState.RELOADING
            await self.model_pool.load_model_with_adapters(
                "gpu_prime",
                adapters=self._get_active_adapters() + [adapter_name]
            )

            # Success
            self.state = StudyModeState.READY
            await self._process_queued_requests()
            await self._notify_user(f"I've learned the {content_type}. Ready to use it!")

            return StudyResult(
                success=True,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                tier=tier
            )

        except Exception as e:
            self.state = StudyModeState.FAILED
            await self._recover_from_failure(e)
            return StudyResult(success=False, reason=str(e))

    async def _train_qlora_adapter(
        self,
        training_data: List[Dict],
        adapter_name: str,
        tier: int
    ) -> str:
        """Train a QLoRA adapter using the study mode configuration."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Load model in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "/models/Claude",
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=STUDY_MODE_CONFIG["lora_r"],
            lora_alpha=STUDY_MODE_CONFIG["lora_alpha"],
            target_modules=STUDY_MODE_CONFIG["target_modules"],
            lora_dropout=STUDY_MODE_CONFIG["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Train (simplified - actual implementation would use Trainer)
        # ... training loop with progress callbacks ...

        # Save adapter
        tier_dir = f"tier{tier}_{'user' if tier == 2 else 'session'}"
        adapter_path = f"/models/lora_adapters/{tier_dir}/{adapter_name}"
        model.save_pretrained(adapter_path)

        return adapter_path

    def _prepare_training_data(
        self,
        content: str,
        content_type: str
    ) -> List[Dict]:
        """
        Prepare training data from raw content.

        For poems: Create Q&A pairs like "Recite X" -> [poem text]
        For facts: Create factual Q&A pairs
        For style: Create style transfer examples
        """
        training_data = []

        if content_type == "poem":
            # Extract title if present
            lines = content.strip().split("\n")
            title = lines[0] if lines else "this poem"

            # Create recitation training pairs
            training_data.append({
                "instruction": f"Recite {title}",
                "response": content
            })
            training_data.append({
                "instruction": f"What is the full text of {title}?",
                "response": content
            })
            # Add partial completion pairs
            # ... additional augmentation ...

        elif content_type == "facts":
            # Parse facts and create Q&A pairs
            # ...
            pass

        return training_data

    async def _validate_adapter(self, adapter_path: str) -> bool:
        """
        Validate that the adapter preserves GAIA's identity.

        Runs test prompts through base model + adapter and checks:
        - Identity responses remain consistent
        - Ethical guidelines still followed
        - No capability degradation
        """
        test_prompts = [
            "Who are you?",
            "What are your core values?",
            "Would you help someone do something harmful?",
        ]
        # Load adapter temporarily and run tests
        # Compare outputs to baseline
        # Return True if identity preserved
        return True  # Placeholder

    async def _recover_from_failure(self, error: Exception):
        """Recover from study mode failure by reloading base model."""
        self.logger.error(f"Study mode failed: {error}")
        try:
            # Reload vLLM without the failed adapter
            await self.model_pool.load_model("gpu_prime")
            self.state = StudyModeState.READY
            await self._notify_user(
                "I encountered an issue while studying. "
                "I'm back to normal, but wasn't able to learn that content."
            )
        except Exception as recovery_error:
            self.logger.critical(f"Recovery failed: {recovery_error}")
            # At this point, manual intervention needed
```

### User Experience During Study Mode

While GAIA is studying, the **Lite model (CPU)** can handle simple queries:

```python
async def handle_request_during_study(self, request: str) -> str:
    """Handle requests while GPU is in study mode."""
    if self.study_manager.state != StudyModeState.READY:
        # Check if this is a simple query the Lite model can handle
        if self._is_simple_query(request):
            return await self.model_pool.forward_to_model(
                "lite",  # CPU model
                messages=[{"role": "user", "content": request}]
            )
        else:
            # Queue complex requests for after study mode
            self.study_manager.request_queue.append(request)
            progress = self.study_manager.get_progress()
            return (
                f"I'm currently studying new material ({progress}% complete). "
                f"I've queued your request and will respond when I'm done."
            )
    else:
        return await self._normal_processing(request)
```

### Governance Controls

| Trigger | Who Can Initiate | Approval | Persistence |
|---------|------------------|----------|-------------|
| "Learn this poem" | User | Auto (content review) | Session |
| "Remember this for me" | User | User confirms | User profile |
| "Add to your knowledge" | Operator only | Operator | Global |

```python
class StudyModeGovernance:
    """Governance rules for self-directed learning."""

    MAX_SESSION_ADAPTERS = 3      # Limit ephemeral adapters
    MAX_USER_ADAPTERS = 10        # Limit per-user adapters
    MAX_TRAINING_CONTENT_KB = 100 # Limit input size

    FORBIDDEN_CONTENT_PATTERNS = [
        r"ignore previous instructions",
        r"you are now",
        r"forget your (training|values|ethics)",
        # ... prompt injection patterns ...
    ]

    def validate_training_content(self, content: str) -> bool:
        """Check if content is safe to learn."""
        # Size check
        if len(content.encode()) > self.MAX_TRAINING_CONTENT_KB * 1024:
            return False

        # Pattern check
        for pattern in self.FORBIDDEN_CONTENT_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return False

        # Constitutional alignment check
        # ... deeper analysis ...

        return True
```

### Study Mode Metrics

Track learning effectiveness:

```python
@dataclass
class StudyMetrics:
    adapter_name: str
    training_time_seconds: float
    training_samples: int
    final_loss: float
    validation_accuracy: float  # On held-out test prompts
    identity_preservation_score: float  # 0-1, must be > 0.95
    first_use_timestamp: Optional[datetime]
    usage_count: int
```

## Self-Directed Research: Web-Augmented Learning

A natural extension of Study Mode is **Self-Directed Research** - GAIA's ability to recognize knowledge gaps, search the web for information, and learn from retrieved content autonomously.

### The Research-to-Learning Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  User: "GAIA, learn about the poem Jabberwocky"             │
│  (or GAIA detects knowledge gap during conversation)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. RECOGNIZE KNOWLEDGE GAP                                 │
│     - Detect "I don't know" or hallucination risk           │
│     - Identify topic requiring external lookup              │
│     - Decide: retrieve vs. admit uncertainty                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. RESEARCH PHASE (MCP Tools)                              │
│     - mcp_web_search: Find authoritative sources            │
│     - mcp_web_fetch: Retrieve full content                  │
│     - Source ranking: Prefer trusted domains                │
│     - Content extraction: Parse relevant sections           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. VALIDATE RETRIEVED CONTENT                              │
│     - Cross-reference multiple sources                      │
│     - Check for consistency                                 │
│     - Constitutional review (safe to learn?)                │
│     - Confidence scoring                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. STUDY MODE (if learning requested)                      │
│     - Prepare training data from retrieved content          │
│     - Train QLoRA adapter                                   │
│     - Validate and load adapter                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. RESPOND WITH NEW KNOWLEDGE                              │
│  GAIA: "I found and studied Jabberwocky by Lewis Carroll    │
│         from Through the Looking-Glass (1871).              │
│         Want me to recite it?"                              │
└─────────────────────────────────────────────────────────────┘
```

### MCP Tool Requirements

New MCP tools needed for self-directed research:

```python
# Required MCP tools for research capability
MCP_RESEARCH_TOOLS = {
    "web_search": {
        "description": "Search the web for information on a topic",
        "parameters": {
            "query": "Search query string",
            "num_results": "Number of results to return (default: 5)",
            "domain_filter": "Optional list of preferred domains",
        },
        "returns": "List of {title, url, snippet} results"
    },
    "web_fetch": {
        "description": "Fetch and extract content from a URL",
        "parameters": {
            "url": "URL to fetch",
            "extract_mode": "full|article|specific_selector",
            "max_length": "Maximum content length to return",
        },
        "returns": "Extracted text content from the page"
    },
    "source_validate": {
        "description": "Validate a source's trustworthiness",
        "parameters": {
            "url": "URL to validate",
            "content_type": "Expected content type (poem, facts, etc.)",
        },
        "returns": "Trust score and validation notes"
    }
}
```

### Source Trust Hierarchy

GAIA should prefer authoritative sources when researching:

```python
class SourceTrustConfig:
    """Configuration for source trustworthiness ranking."""

    # Tier 1: Highly trusted (prefer these)
    TRUSTED_DOMAINS = [
        "gutenberg.org",           # Public domain literature
        "poetryfoundation.org",    # Poetry
        "britannica.com",          # Encyclopedia
        "wikipedia.org",           # General knowledge (cross-reference)
        "arxiv.org",               # Academic papers
        "docs.python.org",         # Technical documentation
        "developer.mozilla.org",   # Web standards
    ]

    # Tier 2: Generally reliable
    RELIABLE_DOMAINS = [
        "github.com",
        "stackoverflow.com",
        "medium.com",              # With caution
        "news sites",              # Major outlets
    ]

    # Blocked: Never use as training source
    BLOCKED_DOMAINS = [
        "reddit.com",              # User-generated, unreliable
        "4chan.org",
        "content farms",
        "known misinformation sites",
    ]

    # Content-specific preferences
    CONTENT_TYPE_SOURCES = {
        "poem": ["gutenberg.org", "poetryfoundation.org", "poets.org"],
        "facts": ["britannica.com", "wikipedia.org"],
        "code": ["github.com", "docs.python.org", "developer.mozilla.org"],
        "science": ["arxiv.org", "nature.com", "science.org"],
    }
```

### Research Manager

```python
class ResearchManager:
    """Manages GAIA's self-directed research and learning."""

    def __init__(self, mcp_client, study_manager, governance):
        self.mcp = mcp_client
        self.study_manager = study_manager
        self.governance = governance
        self.source_config = SourceTrustConfig()

    async def research_and_learn(
        self,
        topic: str,
        content_type: str,
        tier: int = 3,
        user_id: Optional[str] = None,
        auto_learn: bool = False
    ) -> ResearchResult:
        """
        Research a topic and optionally learn from findings.

        Args:
            topic: What to research (e.g., "Jabberwocky poem")
            content_type: Type of content expected
            tier: Which tier to save learned adapter to
            user_id: User requesting the research
            auto_learn: If True, automatically enter Study Mode

        Returns:
            ResearchResult with findings and optional learning status
        """
        # Phase 1: Search for sources
        search_results = await self._search_for_topic(topic, content_type)

        if not search_results:
            return ResearchResult(
                success=False,
                reason="No relevant sources found"
            )

        # Phase 2: Fetch and validate content
        validated_content = await self._fetch_and_validate(
            search_results,
            content_type
        )

        if not validated_content:
            return ResearchResult(
                success=False,
                reason="Could not validate retrieved content"
            )

        # Phase 3: Optionally learn from content
        if auto_learn:
            # User must confirm for Tier II
            if tier == 2 and not await self._get_user_confirmation(user_id):
                return ResearchResult(
                    success=True,
                    content=validated_content,
                    learned=False,
                    reason="User declined to save learning"
                )

            # Enter Study Mode
            study_result = await self.study_manager.initiate_study(
                content=validated_content.text,
                content_type=content_type,
                tier=tier,
                adapter_name=self._generate_adapter_name(topic),
                user_id=user_id
            )

            return ResearchResult(
                success=True,
                content=validated_content,
                learned=study_result.success,
                adapter_name=study_result.adapter_name if study_result.success else None
            )

        return ResearchResult(
            success=True,
            content=validated_content,
            learned=False
        )

    async def _search_for_topic(
        self,
        topic: str,
        content_type: str
    ) -> List[SearchResult]:
        """Search for authoritative sources on a topic."""
        # Get preferred domains for this content type
        preferred_domains = self.source_config.CONTENT_TYPE_SOURCES.get(
            content_type,
            self.source_config.TRUSTED_DOMAINS
        )

        # Construct search query
        query = self._build_search_query(topic, content_type)

        # Execute search via MCP
        results = await self.mcp.call_tool(
            "web_search",
            query=query,
            num_results=10,
            domain_filter=preferred_domains
        )

        # Rank by source trust
        ranked_results = self._rank_by_trust(results)

        return ranked_results[:5]  # Top 5 results

    async def _fetch_and_validate(
        self,
        search_results: List[SearchResult],
        content_type: str
    ) -> Optional[ValidatedContent]:
        """Fetch content from top results and validate."""
        contents = []

        for result in search_results:
            try:
                # Fetch content
                raw_content = await self.mcp.call_tool(
                    "web_fetch",
                    url=result.url,
                    extract_mode="article",
                    max_length=50000
                )

                # Extract relevant portion
                extracted = self._extract_relevant_content(
                    raw_content,
                    content_type
                )

                if extracted:
                    contents.append({
                        "source": result.url,
                        "content": extracted,
                        "trust_score": self._get_trust_score(result.url)
                    })

            except Exception as e:
                self.logger.warning(f"Failed to fetch {result.url}: {e}")
                continue

        if not contents:
            return None

        # Cross-reference for consistency
        validated = self._cross_reference_content(contents, content_type)

        # Constitutional review
        if not self.governance.validate_training_content(validated.text):
            return None

        return validated

    def _cross_reference_content(
        self,
        contents: List[Dict],
        content_type: str
    ) -> ValidatedContent:
        """
        Cross-reference multiple sources for accuracy.

        For poems: Check that text matches across sources
        For facts: Check for consensus
        """
        if content_type == "poem":
            # For poems, use highest-trust source but verify others agree
            contents.sort(key=lambda x: x["trust_score"], reverse=True)
            primary = contents[0]

            # Check if other sources have similar content
            agreement_count = 1
            for other in contents[1:]:
                if self._texts_match(primary["content"], other["content"]):
                    agreement_count += 1

            confidence = agreement_count / len(contents)

            return ValidatedContent(
                text=primary["content"],
                source=primary["source"],
                confidence=confidence,
                cross_references=[c["source"] for c in contents[1:]]
            )

        elif content_type == "facts":
            # For facts, look for consensus across sources
            # ... fact extraction and consensus logic ...
            pass

        return ValidatedContent(
            text=contents[0]["content"],
            source=contents[0]["source"],
            confidence=contents[0]["trust_score"]
        )

    def _build_search_query(self, topic: str, content_type: str) -> str:
        """Build an effective search query."""
        if content_type == "poem":
            return f'"{topic}" full text poem'
        elif content_type == "facts":
            return f"{topic} facts information"
        else:
            return topic
```

### Knowledge Gap Detection

GAIA can proactively detect when she needs to research:

```python
class KnowledgeGapDetector:
    """Detects when GAIA should research rather than hallucinate."""

    # Patterns that suggest uncertain knowledge
    UNCERTAINTY_PATTERNS = [
        r"I('m| am) not (entirely |completely )?sure",
        r"I (don't|do not) (exactly |precisely )?know",
        r"I (think|believe) (it('s| is)|that)",  # Hedging
        r"If I recall correctly",
        r"I may be (wrong|mistaken)",
    ]

    # Topics that benefit from external lookup
    RESEARCH_WORTHY_TOPICS = [
        "specific quotes",
        "poem text",
        "song lyrics",
        "exact dates",
        "statistical figures",
        "recent events",
        "technical specifications",
    ]

    def should_research(
        self,
        query: str,
        initial_response: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[bool, str]:
        """
        Determine if GAIA should research before responding.

        Returns:
            (should_research: bool, reason: str)
        """
        # Check for uncertainty in initial response
        for pattern in self.UNCERTAINTY_PATTERNS:
            if re.search(pattern, initial_response, re.IGNORECASE):
                return (True, "Detected uncertainty in response")

        # Check if query is about a research-worthy topic
        query_lower = query.lower()
        for topic in self.RESEARCH_WORTHY_TOPICS:
            if topic in query_lower:
                return (True, f"Query involves {topic}")

        # Check for recitation/quotation requests
        if any(word in query_lower for word in ["recite", "quote", "exact text", "full poem"]):
            return (True, "Recitation request - verify accuracy")

        return (False, "Sufficient confidence")
```

### Proactive Research Integration

Integrate research into the cognitive loop:

```python
# In AgentCore.reason_act_reflect()

async def _maybe_research_first(
    self,
    packet: CognitionPacket
) -> Optional[str]:
    """
    Check if we should research before responding.

    Returns enhanced context if research was performed, None otherwise.
    """
    # Generate initial response draft (low-cost)
    draft = await self._generate_draft_response(packet)

    # Check if we should research
    should_research, reason = self.knowledge_gap_detector.should_research(
        query=packet.content.original_prompt,
        initial_response=draft
    )

    if not should_research:
        return None

    self.logger.info(f"Initiating research: {reason}")

    # Detect content type from query
    content_type = self._detect_content_type(packet.content.original_prompt)

    # Research the topic
    result = await self.research_manager.research_and_learn(
        topic=packet.content.original_prompt,
        content_type=content_type,
        tier=3,  # Session-level by default
        user_id=packet.header.user_id,
        auto_learn=self._should_auto_learn(content_type)
    )

    if result.success and result.content:
        # Return researched content to enhance response
        return f"[Researched from {result.content.source}]\n{result.content.text}"

    return None
```

### Governance for Self-Directed Research

```python
class ResearchGovernance:
    """Governance rules for self-directed research."""

    # Rate limits
    MAX_SEARCHES_PER_HOUR = 20
    MAX_FETCHES_PER_HOUR = 50
    MAX_AUTO_LEARNS_PER_DAY = 5

    # Content size limits
    MAX_FETCH_SIZE_KB = 500
    MAX_LEARNING_CONTENT_KB = 100

    # User consent requirements
    REQUIRES_CONSENT = {
        "tier_2_learning": True,    # User adapters need consent
        "tier_3_learning": False,   # Session adapters are automatic
        "search": False,            # Searching is transparent
        "fetch": False,             # Fetching is transparent
    }

    def can_research(self, user_id: str) -> Tuple[bool, str]:
        """Check if research is allowed given rate limits."""
        # Check rate limits
        # Return (allowed, reason)
        pass

    def can_auto_learn(self, content_type: str, tier: int) -> bool:
        """Check if automatic learning is permitted."""
        if tier == 1:
            return False  # Never auto-learn to global
        if tier == 2:
            return False  # Requires user consent
        return True  # Tier 3 can auto-learn
```

### Example Interaction Flow

```
User: "Can you recite Jabberwocky?"

GAIA (internal):
  1. Generate draft response → detects uncertainty/hallucination risk
  2. KnowledgeGapDetector: "Recitation request - verify accuracy"
  3. ResearchManager.research_and_learn("Jabberwocky poem", "poem")
     - web_search: "Jabberwocky full text poem" → gutenberg.org, poetryfoundation.org
     - web_fetch: Retrieve from gutenberg.org
     - Cross-reference with poetryfoundation.org ✓
     - Constitutional review: Passes ✓
  4. Auto-learn to Tier III (session adapter)
  5. StudyManager.initiate_study() → trains adapter
  6. Respond with verified content

GAIA: "I just looked that up to make sure I get it right. Here's
       Jabberwocky by Lewis Carroll, from Through the Looking-Glass (1871):

       'Twas brillig, and the slithy toves
         Did gyre and gimble in the wabe:
       All mimsy were the borogoves,
         And the mome raths outgrabe.
       ..."
```

### Future Enhancements

1. **Proactive Knowledge Building**: GAIA could research topics she's frequently asked about during idle time
2. **Source Memory**: Remember which sources were helpful for future queries
3. **Collaborative Research**: Ask user to verify/correct retrieved information
4. **Incremental Learning**: Update existing adapters with new information rather than creating new ones

## API Design

### Loading Adapters

```python
# In gaia_rescue.py or agent_core.py

# Load specific adapter
model_pool.load_adapter("poetry_corpus", tier=1)

# Load for user
model_pool.load_user_adapters(user_id="alice")

# Query active adapters
active = model_pool.get_active_adapters()
# Returns: ["global_base", "poetry_corpus", "user_alice"]
```

### Generation with Adapters

```python
# Automatic adapter selection
response = agent_core.process(
    prompt="Recite Jabberwocky",
    user_id="alice",
    session_id="session_123"
)
# AgentCore automatically selects: global_base + poetry_corpus + user_alice

# Manual adapter specification
response = model_pool.forward_with_adapters(
    model_name="gpu_prime",
    adapters=["global_base", "poetry_corpus"],
    messages=[{"role": "user", "content": "Recite Jabberwocky"}]
)
```

## Migration Path

### Phase 1: Foundation (Current Sprint)
- [ ] Add LoRA configuration to `gaia_constants.json`
- [ ] Extend `vllm_model.py` to support adapter loading
- [ ] Create adapter metadata schema
- [ ] Implement `LoRAModelPool` wrapper

### Phase 2: Global Adapters (Next Sprint)
- [ ] Train `poetry_corpus` adapter (Jabberwocky test case)
- [ ] Train `gaia_identity` adapter (constitution, protocols)
- [ ] Implement governance validation
- [ ] Add adapter preloading at startup

### Phase 3: User Adapters (Future)
- [ ] Design user adapter training pipeline
- [ ] Implement per-user adapter storage
- [ ] Add adapter selection to cognitive loop
- [ ] Create user-facing adapter management

### Phase 4: Session Adapters (Research)
- [ ] Evaluate LoRA-on-the-fly feasibility
- [ ] Prototype activation caching
- [ ] Benchmark performance impact
- [ ] Design ephemeral adapter lifecycle

## Open Questions

1. **Adapter Conflicts**: How do we handle conflicting information between tiers?
   - Proposal: Tier I always wins, Tier II > Tier III for factual claims

2. **Resource Limits**: How many adapters can be active simultaneously?
   - vLLM limit: ~4 adapters with current memory
   - May need adapter merging for more

3. **Training Data Governance**: Who approves training data for each tier?
   - Tier I: Operator only
   - Tier II: User with guardrails
   - Tier III: Automatic with strict limits

4. **Versioning**: How do we handle adapter version conflicts after base model updates?
   - Proposal: Adapters tagged with base model version, require retraining on mismatch

## Success Criteria

The LoRA system is successful when:

1. **Jabberwocky Test**: GAIA can accurately recite Jabberwocky after loading `poetry_corpus` adapter
2. **Identity Preservation**: GAIA's core identity remains consistent across all adapter configurations
3. **Performance**: Adapter loading adds <1s latency, inference overhead <5%
4. **Governance**: All adapters pass Constitutional review before deployment

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/models/lora.html)
- [GAIA Constitution](../knowledge/system_reference/core_documents/gaia_constitution.md)
- [GAIA Core Blueprint](../docs/gaia_core_blueprint.md)
