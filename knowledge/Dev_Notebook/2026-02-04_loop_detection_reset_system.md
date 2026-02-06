# GAIA Development Journal

## Date: 2026-02-04

### Subject: Loop Detection and Reset System - Design Proposal

**Summary:**

Design proposal for a system that detects when the model enters a generation loop (repeated tool calls, oscillating states, error cycles) and gracefully resets while preserving context. The key insight is treating loops as recoverable states rather than failures, by saving the last Cognitive Packet, annotating it with loop metadata, and re-injecting it with recovery hints.

**Status:** Proposal - Awaiting implementation in candidate environment

---

## Problem Statement

Models can enter various types of loops during extended sessions:

1. **Tool call repetition** - Calling the same tool with same arguments repeatedly
2. **Ping-pong patterns** - Alternating between tools (Read → Edit → Read → Edit)
3. **Error cycles** - Same error appearing repeatedly despite fix attempts
4. **Whack-a-mole errors** - Fixing error A causes B, fixing B causes A
5. **Output repetition** - Generating nearly identical responses
6. **State oscillation** - Making changes, undoing them, remaking them

Currently these loops:
- Waste tokens and compute
- Frustrate users
- Eventually require manual intervention
- Lose context about what was already attempted

---

## Proposed Solution

A multi-stage system that detects, captures, resets, and re-injects with context:

```
┌─────────────────────────────────────────────────────────────┐
│                    LOOP DETECTION PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Detector │ → │ Capturer │ → │ Resetter │ → │ Injector │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │              │              │              │        │
│       ▼              ▼              ▼              ▼        │
│   Signals      CapturedPacket   CleanState    RecoveryCtx  │
│   + Scores     + LoopMetadata                 + Hints      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovation

The last Cognitive Packet before reset is **preserved and annotated**:

```typescript
interface CapturedPacket {
  packet_id: string;
  content: CognitivePacket;
  loop_metadata: {
    detected_at: string;
    loop_type: LoopCategory;
    pattern: string;
    pattern_hash: string;
    reset_count: number;
    previous_attempts: {
      approach_summary: string;
      failed_at: string;
    }[];
  };
}
```

This allows the model to see:
- What it was trying to do
- Why it looped
- How many times this has happened
- What approaches have already failed

---

## Architecture

### Component Overview

| Component | Responsibility | Location |
|-----------|---------------|----------|
| LoopDetector | Monitors signals, detects loops | `gaia-core/cognitive/` |
| PatternClassifier | Categorizes loop types | `gaia-core/cognitive/` |
| PacketCapturer | Preserves state before reset | `gaia-core/cognitive/` |
| PatternRenderer | Generates human-readable descriptions | `gaia-core/cognitive/` |
| ResetManager | Orchestrates reset flow | `gaia-core/cognitive/` |
| RecoveryInjector | Re-injects context post-reset | `gaia-core/cognitive/` |

### Detection Algorithms

Five parallel detectors vote on loop presence:

```
Incoming Packet
     │
     ├──► Tool Call Repetition Detector ──► score
     ├──► Output Similarity Detector ────► score
     ├──► State Oscillation Detector ────► score
     ├──► Error Cycle Detector ──────────► score
     ├──► Token Pattern Detector ────────► score
     │
     ▼
┌─────────────────────────────────────┐
│  AGGREGATOR                         │
│  • Any single detector > 0.9        │
│  • Multiple detectors > 0.7         │
│  • Weighted combination > 0.6       │
└─────────────────────────────────────┘
```

#### 1. Tool Call Repetition Detector

```typescript
class ToolCallRepetitionDetector {
  private history: ToolCallRecord[] = [];
  private window_size = 10;

  detect(newCall: ToolCallRecord): DetectionResult {
    // Strategy 1: Exact repetition (same tool, same args, 3+ times)
    // Strategy 2: Similar args (same tool, >0.8 similarity, 4+ times)
    // Strategy 3: Ping-pong (A→B→A→B pattern)
    // Strategy 4: Same result (different calls, identical output)
  }
}
```

**Triggers:**
- Exact match: 3+ consecutive identical calls
- Similar match: 4+ calls with >80% arg similarity
- Ping-pong: 2+ complete cycles of alternating tools
- Same result: 3+ calls returning identical output

#### 2. Output Similarity Detector

```typescript
class OutputSimilarityDetector {
  private outputs: string[] = [];

  detect(newOutput: string): DetectionResult {
    const normalized = this.normalize(newOutput);

    // Multi-strategy similarity:
    // - Jaccard on word sets (0.3 weight)
    // - N-gram similarity (0.4 weight)
    // - Structural similarity (0.3 weight)
  }

  private normalize(text: string): string {
    // Remove timestamps, UUIDs, line numbers
    // Normalize whitespace
  }
}
```

**Triggers:**
- >95% similarity to immediate predecessor
- >85% similarity to 2+ recent outputs
- Convergence trend (outputs getting more similar)

#### 3. State Oscillation Detector

Tracks high-level state snapshots:
- Current goal
- Files modified
- Todo statuses
- Error state

**Triggers:**
- State hash repetition (uniqueHashes/totalHashes < 0.3)
- Goal flip-flopping
- Edit-undo patterns on same file
- Todo status cycling (pending→in_progress→pending)

#### 4. Error Cycle Detector

```typescript
interface ErrorRecord {
  error_type: string;
  error_message: string;
  error_hash: string;
  context: string;
  attempted_fix: string;
}
```

**Triggers:**
- Same error 3+ times
- Same fix attempted 2+ times
- Whack-a-mole pattern (A→B→A errors)

#### 5. Token Pattern Detector (Streaming)

Operates on raw token stream during generation:

**Triggers:**
- Exact phrase repeated 3+ times (20-200 char phrases)
- Structural repetition (same "shape" of lines)
- Character-level degeneration ("aaaa" or "the the the")

---

## Loop Classification Taxonomy

```typescript
enum LoopCategory {
  // Tool-related
  TOOL_REPETITION      // Same call repeated
  TOOL_PING_PONG       // Alternating tools
  TOOL_PARAMETER_DRIFT // Similar but drifting params

  // Output-related
  OUTPUT_VERBATIM      // Identical output
  OUTPUT_PARAPHRASE    // Same meaning, different words
  OUTPUT_STRUCTURAL    // Same structure, different content

  // State-related
  STATE_OSCILLATION    // Bouncing between states
  STATE_REGRESSION     // Make→undo→make pattern
  GOAL_DRIFT           // Changing goals repeatedly

  // Error-related
  ERROR_REPETITION     // Same error recurring
  ERROR_WHACK_A_MOLE   // Errors causing each other
  FIX_REPETITION       // Same fix attempted repeatedly

  // Generation-related
  TOKEN_REPETITION     // Token-level loops
  PHRASE_LOOP          // Phrase repetition
  STRUCTURAL_LOOP      // Structure repetition
}
```

---

## Pattern Description System

Generates human-readable descriptions at three verbosity levels:

### Brief (for status line)
```
"Bash(git status) called 5x"
"Fixing TypeError breaks ImportError"
```

### Summary (for user notification)
```
"You called Bash(git status) 5 times with the same arguments,
getting the same result each time. The output isn't changing,
so repeating won't help."
```

### Full (for model re-injection)
```xml
<loop-recovery reset="1" urgency="warning">

## Loop Detected: Tool Repetition

**Pattern**: You called `Bash` 5 times consecutively.
**Arguments**: `git status`
**Result**: Each call returned the same output.

---

## Constraints for This Attempt

🚫 Do not call Bash with `git status` again

## Suggestions

1. The result will not change if you call this again
2. Consider what information you actually need
3. Try a different tool or approach

</loop-recovery>
```

---

## Escalation Ladder

```
reset_count: 1
└─► Re-inject with "try different approach" hint

reset_count: 2
└─► Add stronger constraints
    └─► "Do NOT use [looping tool/pattern]"
    └─► Suggest specific alternatives

reset_count: 3
└─► Force user intervention
    └─► "I've tried 3 approaches and keep looping.
         Here's what I attempted: [summary]
         How would you like me to proceed?"

reset_count: 4+
└─► Options:
    └─► Switch to different model
    └─► Decompose task into smaller pieces
    └─► Mark task as blocked, move on
```

---

## Data Structures

### Core Types

```typescript
interface LoopDetectorConfig {
  similarity_threshold: number;    // 0.0-1.0, default 0.95
  max_repetitions: number;         // Before triggering, default 3
  detection_window: number;        // Packets to examine, default 10
  escalation_thresholds: number[]; // [1, 2, 3] for escalation
}

interface DetectionResult {
  triggered: boolean;
  confidence: number;              // 0.0-1.0
  pattern: string;                 // Human-readable
  type: LoopCategory;
}

interface AggregatedResult {
  isLoop: boolean;
  confidence: number;
  primaryType: LoopCategory;
  pattern: string;
  triggeredBy: string[];           // Which detectors fired
  evidence: PatternEvidence;
}

interface PatternEvidence {
  repeated_elements: string[];
  repetition_count: number;
  window_size: number;
  first_occurrence: number;
  last_occurrence: number;
  examples: EvidenceExample[];
  variations: string[];
}
```

### Captured State

```typescript
interface CapturedPacket {
  packet_id: string;
  content: CognitivePacket;
  loop_metadata: LoopMetadata;
}

interface LoopMetadata {
  detected_at: string;
  loop_type: LoopCategory;
  pattern: string;
  pattern_hash: string;
  reset_count: number;
  previous_attempts: {
    approach_summary: string;
    failed_at: string;
  }[];
}
```

---

## Reset Flow

```
LOOP DETECTED
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CAPTURE                                                  │
│     - Save last Cognitive Packet                            │
│     - Attach loop_metadata                                  │
│     - Increment reset_count                                 │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. RESET                                                    │
│     - Halt current generation                               │
│     - Clear model's immediate context/KV cache              │
│     - Preserve: conversation history, task objectives       │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. RE-INJECT                                                │
│     - Add <loop-recovery> context block                     │
│     - Include pattern description                           │
│     - Add constraints (what NOT to do)                      │
│     - Add recovery suggestions                              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. RESUME                                                   │
│     - Model continues with full context                     │
│     - Aware of previous failed approaches                   │
│     - Constrained from repeating same pattern               │
└─────────────────────────────────────────────────────────────┘
```

---

## User-Facing Notifications

```typescript
interface LoopNotification {
  statusLine: string;          // "⟳ Loop detected: Bash(git status) called 5x"

  toast: {
    title: string;             // "Loop Detected" or "Loop Detected (Reset #2)"
    body: string;              // Summary description
    severity: 'info' | 'warning' | 'error';
  };

  details: string;             // Full description if user clicks "more info"
}
```

---

## Integration Points

### Where It Fits in GAIA

```
┌─────────────────────────────────────────────────────────────┐
│                        gaia-core                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────┐                                      │
│  │ CognitiveManager  │ ◄─── Owns packet creation/routing    │
│  └─────────┬─────────┘                                      │
│            │                                                │
│            ▼                                                │
│  ┌───────────────────┐                                      │
│  │ LoopDetector      │ ◄─── NEW: Monitors for loops         │
│  └─────────┬─────────┘                                      │
│            │                                                │
│            ├── on_packet() ◄─── Called for each packet      │
│            ├── on_tool_result() ◄─── Called after tools     │
│            ├── on_generation_chunk() ◄─── Streaming         │
│            │                                                │
│            ▼                                                │
│  ┌───────────────────┐                                      │
│  │ ResetManager      │ ◄─── NEW: Orchestrates reset         │
│  └───────────────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `candidates/gaia-core/gaia_core/cognitive/loop_detector.py` | Create | Main detection logic |
| `candidates/gaia-core/gaia_core/cognitive/pattern_classifier.py` | Create | Loop classification |
| `candidates/gaia-core/gaia_core/cognitive/pattern_renderer.py` | Create | Description generation |
| `candidates/gaia-core/gaia_core/cognitive/reset_manager.py` | Create | Reset orchestration |
| `candidates/gaia-core/gaia_core/cognitive/types.py` | Modify | Add new types |
| `candidates/gaia-core/gaia_core/cognitive/manager.py` | Modify | Integrate detector |
| `candidates/gaia-core/tests/test_loop_detector.py` | Create | Unit tests |
| `candidates/gaia-core/tests/test_pattern_classifier.py` | Create | Classification tests |

---

## Implementation Plan

### Phase 1: Core Detection (Candidate)

1. Create type definitions in `types.py`
2. Implement `ToolCallRepetitionDetector`
3. Implement `OutputSimilarityDetector`
4. Implement basic `LoopDetectionAggregator`
5. Add unit tests with synthetic loop scenarios
6. Test in candidate environment

### Phase 2: Pattern Description (Candidate)

1. Implement `PatternClassifier`
2. Create template system in `PatternRenderer`
3. Implement all template types
4. Test description generation
5. Verify re-injection format works

### Phase 3: Reset Flow (Candidate)

1. Implement `PacketCapturer`
2. Implement `ResetManager`
3. Implement `RecoveryInjector`
4. Integration test full reset flow
5. Test escalation ladder

### Phase 4: Advanced Detectors (Candidate)

1. Implement `StateOscillationDetector`
2. Implement `ErrorCycleDetector`
3. Implement `TokenPatternDetector` (streaming)
4. Add detector weights tuning
5. Comprehensive integration tests

### Phase 5: Testing & Promotion

1. Synthetic loop injection tests
2. Stress tests with deliberately bad prompts
3. Verify no false positives on normal usage
4. User notification integration
5. Promote to live after validation

---

## Testing Strategy

### Unit Tests

```python
# test_loop_detector.py

def test_tool_repetition_exact_match():
    """3+ identical tool calls should trigger"""
    detector = ToolCallRepetitionDetector()
    call = ToolCallRecord(tool="Bash", args_hash="abc123")

    detector.detect(call)  # 1
    detector.detect(call)  # 2
    result = detector.detect(call)  # 3

    assert result.triggered == True
    assert result.confidence > 0.9

def test_tool_repetition_below_threshold():
    """2 identical calls should NOT trigger"""
    detector = ToolCallRepetitionDetector()
    call = ToolCallRecord(tool="Bash", args_hash="abc123")

    detector.detect(call)  # 1
    result = detector.detect(call)  # 2

    assert result.triggered == False

def test_ping_pong_detection():
    """A→B→A→B pattern should trigger"""
    detector = ToolCallRepetitionDetector()

    detector.detect(ToolCallRecord(tool="Read", args_hash="a"))
    detector.detect(ToolCallRecord(tool="Edit", args_hash="b"))
    detector.detect(ToolCallRecord(tool="Read", args_hash="a"))
    result = detector.detect(ToolCallRecord(tool="Edit", args_hash="b"))

    assert result.triggered == True
    assert "ping_pong" in result.pattern.lower()

def test_output_similarity_verbatim():
    """Identical outputs should trigger"""
    detector = OutputSimilarityDetector()
    output = "Here is my response about the topic."

    detector.detect(output)
    detector.detect(output)
    result = detector.detect(output)

    assert result.triggered == True
    assert result.confidence > 0.95

def test_error_whack_a_mole():
    """A→B→A error pattern should trigger"""
    detector = ErrorCycleDetector()

    detector.detect(ErrorRecord(error_type="TypeError"), False)
    detector.detect(ErrorRecord(error_type="ImportError"), False)
    detector.detect(ErrorRecord(error_type="TypeError"), False)
    result = detector.detect(ErrorRecord(error_type="ImportError"), False)

    assert result.triggered == True
    assert "whack_a_mole" in result.pattern.lower()
```

### Integration Tests

```python
# test_reset_flow.py

async def test_full_reset_flow():
    """End-to-end test of detection → capture → reset → inject"""
    manager = ResetManager(config=LoopDetectorConfig())

    # Simulate 5 identical tool calls
    for i in range(5):
        await manager.on_tool_call(
            tool="Bash",
            args={"command": "git status"}
        )

    # Should have triggered reset
    assert manager.reset_count == 1

    # Should have captured packet
    captured = manager.last_captured_packet
    assert captured is not None
    assert captured.loop_metadata.reset_count == 1

    # Recovery context should contain constraints
    context = manager.get_recovery_context()
    assert "Do not call Bash with" in context
    assert "git status" in context

async def test_escalation_to_user():
    """After 3 resets, should require user intervention"""
    manager = ResetManager(config=LoopDetectorConfig())

    # Force 3 resets
    for reset in range(3):
        for i in range(5):
            await manager.on_tool_call(
                tool="Bash",
                args={"command": "git status"}
            )

    assert manager.reset_count == 3
    assert manager.requires_user_intervention == True
```

### False Positive Tests

```python
def test_no_false_positive_on_legitimate_repetition():
    """Legitimate repeated checks should not trigger"""
    detector = ToolCallRepetitionDetector()

    # Build status checks are legitimate
    for i in range(3):
        detector.detect(ToolCallRecord(
            tool="Bash",
            args_hash=hash(f"npm test"),
            result_hash=hash(f"PASS tests {i}")  # Different results!
        ))

    # Should NOT trigger because results differ
    assert detector.last_result.triggered == False

def test_no_false_positive_on_similar_reads():
    """Reading related files is normal"""
    detector = ToolCallRepetitionDetector()

    files = ["src/a.ts", "src/b.ts", "src/c.ts", "src/d.ts"]
    for f in files:
        detector.detect(ToolCallRecord(
            tool="Read",
            args_hash=hash(f)
        ))

    # Different files, should not trigger
    assert detector.last_result.triggered == False
```

---

## Configuration

```yaml
# loop_detection.yaml

detection:
  enabled: true

  thresholds:
    tool_repetition:
      exact_match: 3
      similar_match: 4
      similarity_threshold: 0.8

    output_similarity:
      verbatim_threshold: 0.95
      paraphrase_threshold: 0.85
      min_occurrences: 2

    error_cycle:
      same_error: 3
      same_fix: 2

    aggregator:
      single_high_confidence: 0.9
      multiple_medium_confidence: 0.7
      weighted_combination: 0.6

  window_size: 10

escalation:
  thresholds: [1, 2, 3]
  user_intervention_at: 3

notifications:
  status_line: true
  toast: true
  toast_duration_ms: 5000
```

---

## Metrics & Observability

Track for tuning and validation:

```typescript
interface LoopDetectionMetrics {
  // Detection stats
  loops_detected: Counter;
  loops_by_type: Counter<LoopCategory>;
  detection_confidence: Histogram;

  // Reset stats
  resets_performed: Counter;
  resets_by_escalation_level: Counter<number>;

  // Recovery stats
  successful_recoveries: Counter;    // Broke out of loop
  failed_recoveries: Counter;        // Looped again after reset
  user_interventions: Counter;

  // False positive tracking
  user_reported_false_positives: Counter;

  // Performance
  detection_latency_ms: Histogram;
}
```

---

## Open Questions - ANSWERED

Based on exploration of the existing gaia-core cognitive module, here are the answers:

---

### 1. KV Cache Reset

**Question:** Do we have access to reset the model's KV cache, or do we need to rely on context manipulation only?

**Answer: Context manipulation only - and this is fine.**

The gaia-core system does not expose direct KV cache control. Each model backend (vLLM, HuggingFace, llama_cpp) manages its own caching internally. However, this is actually acceptable because:

1. **New API calls = fresh context.** Each `model.create_chat_completion()` call is independent. The "reset" happens naturally when we construct a new prompt.

2. **SessionManager provides the lever.** We can manipulate what goes into the next prompt via:
   - `session_manager.reset_session(session_id)` - nuclear option
   - `session_manager.get_history()` + selective filtering - surgical option
   - Injecting `<loop-recovery>` blocks into the prompt context

3. **Sketchpad for working memory.** The existing `sketchpad_write/read/clear` system provides transient state that we can use to track loop metadata without polluting conversation history.

**Implementation approach:**
```python
# Instead of KV cache reset, we:
# 1. Capture current packet state
# 2. Clear or filter session history as needed
# 3. Inject loop-recovery context into next prompt build
# 4. Let the next run_turn() start fresh with modified context
```

---

### 2. Streaming Detection

**Question:** Should token-level detection halt generation mid-stream, or wait for completion?

**Answer: Halt mid-stream for severe loops; otherwise wait.**

The existing `ExternalVoice.stream_response()` already has infrastructure for mid-stream interruption:

```python
# From external_voice.py lines 273-349
# Observer integration with interrupt levels:
# - BLOCK (terminal) - stops generation immediately
# - CAUTION (re-reflect) - flags for post-processing
```

**Implementation approach:**

| Loop Severity | Action | Rationale |
|---------------|--------|-----------|
| Token-level degeneration ("aaaa", "the the the") | **Halt immediately** | Wasting tokens on garbage |
| Phrase repetition (same sentence 3x) | **Halt immediately** | Clear loop, no value continuing |
| Structural repetition | **Wait + flag** | Might recover; let it finish then reset |
| Tool call patterns | **Post-turn detection** | Tool calls are discrete, not streaming |

**Integration point:** Add a `LoopDetectorObserver` that can be registered with `ExternalVoice`:

```python
class LoopDetectorObserver:
    def on_token(self, token: str, buffer: str) -> Optional[InterruptLevel]:
        result = self.token_detector.detect(token)
        if result.triggered and result.confidence > 0.95:
            return InterruptLevel.BLOCK
        return None
```

This hooks into the existing observer rate-limiting (min 15s interval, max 6 per stream).

---

### 3. User Override

**Question:** Should users be able to say "no, keep going" if they believe the detection is wrong?

**Answer: Yes, with friction that increases with reset count.**

**Rationale:**
- Users may have legitimate reasons (testing, exploring edge cases)
- False positives are possible, especially early on
- But repeated overrides suggest misconfiguration, not user intent

**Escalation with override:**

| Reset # | Override Available | Friction |
|---------|-------------------|----------|
| 1 | Yes | Simple "Continue anyway?" prompt |
| 2 | Yes | Explain what was detected, require confirmation |
| 3 | Yes, but warned | "This is the 3rd loop. Override will disable detection for this session." |
| 4+ | Requires explicit flag | Must set `LOOP_DETECTION_OVERRIDE=true` or use rescue shell |

**Implementation:** Add to the user notification system:

```python
class LoopNotification:
    # ... existing fields ...
    allow_override: bool
    override_callback: Optional[Callable]
    override_warning: Optional[str]
```

**Tracking:** Log all overrides for tuning:
```python
metrics.user_overrides.inc(labels={
    "loop_type": classification.category,
    "reset_count": reset_count,
    "session_id": session_id
})
```

---

### 4. Memory Persistence

**Question:** Should loop patterns be persisted across sessions to avoid known problematic patterns?

**Answer: Yes, but carefully scoped.**

The existing memory architecture supports this:

1. **Short-term (in-process dict):** Current session loop state
2. **Working (SessionManager):** Loop metadata attached to session
3. **Long-term (vector store via MCP):** Historical loop patterns

**What to persist:**

| Data | Storage | Purpose |
|------|---------|---------|
| Current session loop count | Working memory | Escalation ladder |
| Pattern hashes that caused loops | Long-term | Predictive avoidance |
| User overrides | Long-term | Tuning false positive thresholds |
| Recovery success/failure | Long-term | Which interventions work |

**What NOT to persist:**
- Raw conversation content (privacy)
- Exact tool arguments (may contain secrets)
- User-specific patterns without consent

**Implementation:** Extend the existing archival system:

```python
class LoopArchiver:
    def archive_loop_event(self, event: LoopEvent):
        # Store to vector DB with semantic embedding
        # Pattern: "tool_repetition:Bash:git_status:reset_1:recovered"
        self.mcp_client.embedding_store(
            content=event.to_searchable_string(),
            metadata=event.to_metadata()
        )

    def check_known_patterns(self, current_state: PacketState) -> List[KnownLoopPattern]:
        # Query for similar past loops
        return self.mcp_client.embedding_query(
            query=current_state.to_pattern_string(),
            top_k=5
        )
```

---

### 5. Model-Specific Tuning

**Question:** Different models may loop differently; do we need model-specific thresholds?

**Answer: Yes, with sensible defaults and per-model overrides.**

From the codebase, GAIA uses multiple models with different characteristics:

| Model | Role | Loop Risk | Notes |
|-------|------|-----------|-------|
| lite (Operator) | Intent detection, quick tasks | Low | Short outputs, constrained |
| prime | Planning, reflection | Medium | Longer reasoning chains |
| gpu_prime | Heavy generation | High | Complex outputs, more prone to loops |
| cpu_prime | Fallback | Medium | Similar to prime |
| oracle | Research | Low | Read-only, less tool use |

**Configuration structure:**

```yaml
loop_detection:
  defaults:
    tool_repetition_threshold: 3
    output_similarity_threshold: 0.95
    window_size: 10

  model_overrides:
    gpu_prime:
      # More aggressive detection - this model loops more
      tool_repetition_threshold: 2
      output_similarity_threshold: 0.90

    lite:
      # More lenient - short outputs naturally repeat
      output_similarity_threshold: 0.98
      window_size: 5

    oracle:
      # Very lenient - research queries legitimately repeat
      tool_repetition_threshold: 5
```

**Runtime selection:**

```python
def get_config_for_model(model_name: str) -> LoopDetectorConfig:
    base = load_defaults()
    overrides = load_model_overrides(model_name)
    return base.merge(overrides)
```

**Adaptive tuning (future):**
Track loop rates per model and auto-adjust thresholds:

```python
if model_loop_rate > expected_rate * 1.5:
    # Model is looping more than expected
    tighten_thresholds(model_name)
elif model_false_positive_rate > acceptable_rate:
    # Too many false positives
    loosen_thresholds(model_name)
```

---

## Additional Discoveries from Codebase Exploration

### Existing Loop Prevention

The tool routing system already has a basic loop prevention mechanism:

```python
# From cognition_packet.py
class ToolRoutingState:
    reinjection_count: int  # Safety limit to prevent loops
    max_reinjections: int = 3
```

**Integration opportunity:** Our loop detector should be aware of this and:
1. Increment detection sensitivity when `reinjection_count > 1`
2. Share state between tool routing and general loop detection
3. Use the same escalation ladder

### Observer System

The existing observer system (`stream_observer.py`) provides hooks we can use:

```python
# Observer can interrupt with levels:
# - InterruptLevel.BLOCK - terminal, stops generation
# - InterruptLevel.CAUTION - flags for review

# Rate limiting already in place:
# - OBSERVER_MIN_INTERVAL = 15 seconds
# - OBSERVER_MAX_PER_STREAM = 6 calls
```

Our `TokenPatternDetector` should integrate as an observer rather than a separate system.

### Session Archival

When sessions hit 20 messages, they're summarized and archived:

```python
# From session_manager.py
# ConversationSummarizer + ConversationArchiver handle this
```

**Opportunity:** Include loop events in the summary:
```
"Session included 2 detected loops (tool_repetition, error_cycle),
both successfully recovered via reset."
```

### Ethical Sentinel Reset

There's already a pattern for resetting safety state:

```python
# From stream_observer.py
ethical_sentinel.reset_loop()  # Clears cached defaults
```

We should follow this pattern for our loop detector state.

---

## Revised Integration Points

Based on codebase exploration, here's the updated integration plan:

```
┌─────────────────────────────────────────────────────────────┐
│                        gaia-core                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────┐                                      │
│  │ AgentCore         │                                      │
│  │ (agent_core.py)   │                                      │
│  └─────────┬─────────┘                                      │
│            │                                                │
│            │ run_turn()                                     │
│            │                                                │
│  ┌─────────▼─────────┐     ┌───────────────────┐           │
│  │ LoopDetector      │◄───►│ ToolRoutingState  │           │
│  │ (NEW)             │     │ (existing)        │           │
│  └─────────┬─────────┘     └───────────────────┘           │
│            │                                                │
│            │ Hooks into:                                    │
│            │                                                │
│  ┌─────────▼─────────┐     ┌───────────────────┐           │
│  │ ExternalVoice     │     │ SessionManager    │           │
│  │ (streaming)       │     │ (history)         │           │
│  │ - TokenObserver   │     │ - reset_session() │           │
│  └───────────────────┘     │ - add_message()   │           │
│                            └───────────────────┘           │
│                                                             │
│  ┌───────────────────┐     ┌───────────────────┐           │
│  │ Sketchpad         │     │ PromptBuilder     │           │
│  │ (working memory)  │     │ (context inject)  │           │
│  │ - loop_metadata   │     │ - recovery hints  │           │
│  └───────────────────┘     └───────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### File Modifications (Revised)

| File | Action | Description |
|------|--------|-------------|
| `gaia_core/cognition/loop_detector.py` | Create | Main detection + aggregation |
| `gaia_core/cognition/loop_patterns.py` | Create | Pattern classification + templates |
| `gaia_core/cognition/loop_recovery.py` | Create | Reset orchestration + injection |
| `gaia_core/cognition/agent_core.py` | Modify | Hook detector into run_turn() |
| `gaia_core/cognition/external_voice.py` | Modify | Add TokenPatternObserver |
| `gaia_core/cognition/cognition_packet.py` | Modify | Add LoopMetadata to packet |
| `gaia_core/cognition/prompt_builder.py` | Modify | Inject recovery context |
| `gaia_core/memory/session_manager.py` | Modify | Add loop event archival |
| `tests/test_loop_*.py` | Create | Comprehensive test suite |

---

## Future Enhancements

1. **Predictive Detection** - Detect early signs of looping before it fully manifests
2. **Adaptive Thresholds** - Learn from false positives/negatives to tune per-user
3. **Loop Prevention** - Pre-emptively warn model about high-risk patterns
4. **Cross-Session Learning** - Remember what caused loops historically
5. **Automatic Task Decomposition** - If task causes loops, auto-split into subtasks

---

## References

- Conversation designing this system: 2026-02-04
- Related: `2026-02-04_candidate_workflow.md` (candidate testing process)
- Cognitive Packet system: `gaia-core/gaia_core/cognition/cognition_packet.py`
- Agent Core (main loop): `gaia-core/gaia_core/cognition/agent_core.py`
- External Voice (streaming): `gaia-core/gaia_core/cognition/external_voice.py`
- Session Manager: `gaia-core/gaia_core/memory/session_manager.py`
- Prompt Builder: `gaia-core/gaia_core/cognition/prompt_builder.py`
- Tool Routing: `gaia-core/gaia_core/cognition/tool_selector.py`
- Stream Observer: `gaia-core/gaia_core/utils/stream_observer.py`

---

## Implementation Notes (Phase 1 Complete)

**Date:** 2026-02-04

Phase 1 implementation completed in `candidates/gaia-core/`. All core components implemented.

### Files Created

| File | Lines | Description |
|------|-------|-------------|
| `gaia_core/cognition/loop_detector.py` | ~750 | All 5 detectors + aggregator |
| `gaia_core/cognition/loop_patterns.py` | ~450 | Classification + template system |
| `gaia_core/cognition/loop_recovery.py` | ~400 | Reset orchestration + injection |

### Files Modified

| File | Changes |
|------|---------|
| `gaia_core/cognition/cognition_packet.py` | Added `LoopState`, `LoopAttempt` dataclasses |
| `gaia_core/cognition/agent_core.py` | Hooks in run_turn() for detection + recording |
| `gaia_core/cognition/external_voice.py` | `LoopDetectorObserver` for streaming |
| `gaia_core/utils/prompt_builder.py` | Recovery context injection |
| `gaia_core/gaia_constants.json` | Loop detection config keys |

### Behavior

1. **Warn-then-block**: First loop detection issues warning, second occurrence blocks
2. **Post-turn detection**: Checks after each turn completes (output, state, errors)
3. **Streaming detection**: Token-level patterns caught during generation
4. **Tool call recording**: Every tool call recorded for repetition detection
5. **Recovery context**: On reset, context injected into next prompt

### Config Keys Added

```json
"LOOP_DETECTION_ENABLED": true,
"LOOP_DETECTION_TOOL_THRESHOLD": 3,
"LOOP_DETECTION_OUTPUT_THRESHOLD": 0.95,
"LOOP_DETECTION_WINDOW_SIZE": 10,
"LOOP_DETECTION_WARN_FIRST": true,
"LOOP_DETECTION_HIGH_CONFIDENCE": 0.9,
"LOOP_DETECTION_MEDIUM_CONFIDENCE": 0.7,
"LOOP_DETECTION_WEIGHTED_THRESHOLD": 0.6
```

### Testing Plan

To test in candidate environment:

```bash
# Restart candidate container
docker restart gaia-core-candidate

# Check health
curl http://localhost:6416/health

# Test via API or rescue shell with deliberate loops:
# 1. Same tool call repeatedly
# 2. Similar outputs
# 3. Token repetition ("the the the")

# Check logs
docker logs gaia-core-candidate --tail 100 | grep -i loop
```

### Known Limitations (for future phases)

1. **No KV cache reset**: Relies on context manipulation only
2. **No adaptive tuning**: Thresholds are static
3. **No cross-session persistence**: Loop patterns not stored long-term yet
4. **No predictive detection**: Only reactive

---

**Next Steps:**

1. ~~Answer open questions~~ ✓ DONE
2. ~~Review revised integration plan~~ ✓ DONE
3. ~~Implement Phase 1 in `candidates/gaia-core/`~~ ✓ DONE
4. Validate in candidate environment
5. Iterate based on testing
6. Promote to live after validation
