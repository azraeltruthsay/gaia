# GAIA Sleep Cycle System - Blueprint

**Date:** 2026-02-14 (original), 2026-02-15 (7-state refactor)
**Author:** Claude Sonnet 4.5 / Azrael
**Reviewed by:** Claude Opus 4.6 (2026-02-14, 2026-02-15)
**Status:** Implemented — Current
**Target System:** gaia-host (RTX 5080 16GB / Ryzen 9 / 32GB RAM)

---

## Review Notes (2026-02-15, Opus 4.6 — 7-State Refactor)

Blueprint updated to reflect implemented 7-state sleep machine:

1. **Renamed states** — AWAKE→ACTIVE, SLEEPING→ASLEEP throughout.
2. **Internal transient phases** — FINISHING_TASK and WAKING moved to `_TransientPhase` internal enum (not public states). They are sub-phases of the ASLEEP state.
3. **Added DREAMING state** — GPU handed to Study for training. Orchestrator triggers via POST /sleep/study-handoff. Canned response sent to users.
4. **Added DISTRACTED state** — CPU or GPU >25% sustained for 5s. Canned response sent. Rechecks every 5 min or on message arrival.
5. **Added OFFLINE state** — Graceful shutdown via POST /sleep/shutdown. Discord goes invisible.
6. **Resource monitoring** — psutil CPU monitoring added alongside NVML GPU. Sustained load tracking triggers DISTRACTED.
7. **Canned responses** — CANNED_DREAMING and CANNED_DISTRACTED constants. gaia-web checks GET /sleep/distracted-check before forwarding messages.
8. **Discord presence** — ACTIVE=online, DROWSY=online, ASLEEP=idle, DREAMING=dnd, DISTRACTED=dnd, OFFLINE=invisible.

---

## Review Notes (2026-02-14, Opus 4.6)

Corrections applied based on codebase audit:

1. **Added DROWSY state** — Cancellable checkpoint-writing phase between AWAKE and SLEEPING. If a message arrives during checkpoint writing, sleep is aborted.
2. **Added CPU Lite parallel wake** — CPU Lite handles first queued message immediately (~2s) while Prime boots in background (~37-60s). Fixes unrealistic <5s wake latency target.
3. **Fixed circular dependency** — Sleep logic moved to gaia-core (not gaia-common). gaia-common provides primitives only. Legacy `processor.py` broken import acknowledged.
4. **Ported archived GIL** — `initiative_engine.py` ports the design from `archive/gaia-assistant-monolith/run_gil.py` rather than reimplementing from scratch.
5. **Fixed checkpoint path** — `/home/claude/state` → `/shared/sleep_state/` (uses existing Docker named volume).
6. **Dropped KV cache serialization** — `kv_cache_manager.py` removed. prime.md IS the KV cache solution. vLLM KV cache can't be serialized across container restarts.
7. **Added thought seed prerequisites** — Observer must be updated to generate `THOUGHT_SEED:` directives. `knowledge/seeds/` directory must be created.
8. **Fixed conversation sanitizer** — Now wraps existing `ConversationCurator` instead of duplicating notability logic.
9. **Fixed config pattern** — Settings in `gaia_constants.json` (existing pattern), not new `.env` file. Removed stdlib `asyncio` from dependencies.
10. **Fixed endpoint pattern** — Uses `APIRouter` with prefix (matches `gpu_endpoints.py` pattern), not direct `@app` decorators.

---

## Executive Summary

### Vision
Transform GAIA's idle time into productive autonomous operation through a biologically-inspired sleep cycle. When no users are actively engaged, GAIA enters a sleep state to perform maintenance, learning, and self-improvement tasks. Messages queue during sleep and trigger graceful wake-up, preserving cognitive context across state transitions.

### Key Features
- **Autonomous Maintenance**: Background tasks execute during idle periods
- **Cognitive Continuity**: Prime model preserves context through checkpoint files
- **Graceful Wake-Up**: Queued messages trigger prioritized state transitions
- **Self-Improvement**: Learning and reflection happen without user prompting
- **Resource Optimization**: GPU/CPU allocated efficiently across sleep/wake cycles

### Timeline
- **Phase 1 (Foundation)**: 2-3 weeks (~40 hours) - Sleep/wake state machine and cognitive checkpointing
- **Phase 2 (Task System)**: 2-3 weeks (~40 hours) - Productive sleep tasks and enhanced initiative
- **Phase 3 (Advanced)**: 3-4 weeks (~50 hours) - QLoRA training, dream mode, optimization
- **Total**: 7-10 weeks (~130 hours)

### Success Metrics
- Zero context loss across sleep/wake transitions
- <5 second perceived wake latency (CPU Lite responds immediately while Prime boots in background)
- <60 second full Prime restoration from tmpfs warm pool
- >80% sleep time utilization for productive tasks
- User satisfaction with autonomous improvements
- No message loss during state transitions

---

## Architecture Overview

### State Machine (6 public states + 2 internal phases)

```
Public States (GaiaState enum):

              ┌──────────┐
              │  ACTIVE  │  Normal operation: process messages, stream responses
              └────┬─────┘  Discord: online, "watching over the studio"
                   │ idle > 5 min AND no active stream
                   ▼
              ┌──────────┐
              │  DROWSY  │  Prime writes prime.md checkpoint (own cognitive state)
              └────┬─────┘  Discord: online, "drifting off..."
                   │        If message arrives here → cancel, return to ACTIVE
                   │ checkpoint written
                   ▼
              ┌──────────┐          ┌───────────┐
              │  ASLEEP  │─────────►│ DREAMING  │  GPU handed to Study
              └────┬─────┘ study    └─────┬─────┘  Discord: dnd, "studying..."
                   │       handoff        │        Canned response only
                   │                      │ study complete
                   │◄─────────────────────┘
                   │
                   │ CPU/GPU >25% for 5s
                   ▼
              ┌────────────┐
              │ DISTRACTED │  System under sustained load
              └────┬───────┘  Discord: dnd, "occupied..."
                   │          Canned response only
                   │ load drops on recheck (every 5 min)
                   │
                   ▼ (returns to ASLEEP)

Internal Phases (inside ASLEEP, not public):

  ASLEEP + wake signal:
    → _FINISHING_TASK (if non-interruptible task running)
    → _WAKING (parallel: CPU Lite handles first msg, Prime boots in background)
    → ACTIVE (context restored from prime.md)

Shutdown (from ANY state):

              ┌──────────┐
              │ OFFLINE  │  Graceful shutdown, Discord: invisible
              └──────────┘  Triggered by POST /sleep/shutdown or app shutdown
```

### Component Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                         gaia-web                            │
│  ┌──────────────┐       ┌──────────────┐                   │
│  │ Discord Bot  │──────►│ Message Queue│                   │
│  └──────────────┘       └──────┬───────┘                   │
│                                 │                           │
│                                 │ enqueue()                 │
│                                 │ send_wake_signal()        │
└─────────────────────────────────┼───────────────────────────┘
                                  │
                                  ▼ HTTP POST /sleep/wake
┌─────────────────────────────────────────────────────────────┐
│                        gaia-core                            │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ SleepCycleLoop   │         │ InitiativeEngine │         │
│  │ (daemon thread)  │         │ (ported from GIL)│         │
│  └────────┬─────────┘         └──────────────────┘         │
│           │                                                 │
│  ┌────────▼──────────┐                                     │
│  │ SleepWakeManager  │  initiate_drowsy()                  │
│  │ (6-state+2-phase) │  receive_wake_signal()              │
│  └────────┬─────────┘  complete_wake()                     │
│           │                                                 │
│  ┌────────▼──────────┐      ┌──────────────────┐          │
│  │ PrimeCheckpoint   │      │ SleepTask        │          │
│  │ (/shared/          │      │ Scheduler        │          │
│  │  sleep_state/     │      └────────┬─────────┘          │
│  │  prime.md)        │               │                     │
│  └───────────────────┘               ▼                     │
│                              ┌─────────────────┐           │
│                              │ - Sanitization  │           │
│                              │ - Curation      │           │
│                              │ - Thought Seeds │           │
│                              │ - GIL Topics    │           │
│                              │ - Vector Reflect│           │
│                              │ - QLoRA (Ph. 3) │           │
│                              └─────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow: Sleep Transition

```
User Message Stops
       ↓
Idle Monitor detects 5min inactivity
       ↓
SleepWakeManager.should_transition_to_drowsy() → true
       ↓
Enter DROWSY state — Discord: online, "drifting off..."
       ↓
   ┌──────────────────────────────────────────┐
   │ 1. Wait for any streaming to finish      │
   │ 2. Send meta-cognitive prompt to Prime:  │
   │    "Write your current state for later"  │
   │ 3. Prime generates cognitive summary     │
   │ 4. Write prime.md checkpoint             │
   │ 5. Rotate: prime.md → prime_previous.md  │
   │ 6. Archive to prime_history/             │
   │                                          │
   │ ⚡ If message arrives during DROWSY:     │
   │    Cancel checkpoint, return to ACTIVE   │
   └──────────────────────────────────────────┘
       ↓ checkpoint complete
Enter ASLEEP state — Discord: idle, "sleeping..."
       ↓
SleepTaskScheduler.get_next_task()
       ↓
Execute: conversation_curation → thought_seed_review →
         initiative_cycle → blueprint_validation →
         (GPU tasks if available)
       ↓
   ┌──────────────────────────────────────────┐
   │ DURING ASLEEP, three things can happen:  │
   │                                          │
   │ A. MESSAGE ARRIVES:                      │
   │    → gaia-web checks /distracted-check   │
   │    → sends POST /sleep/wake to gaia-core │
   │                                          │
   │ B. STUDY HANDOFF (orchestrator):         │
   │    → POST /sleep/study-handoff           │
   │    → enters DREAMING (dnd, canned resp)  │
   │                                          │
   │ C. SUSTAINED LOAD (>25% for 5s):         │
   │    → enters DISTRACTED (dnd, canned resp)│
   │    → rechecks every 5 min                │
   └──────────────────────────────────────────┘
       ↓ wake signal received
SleepWakeManager receives wake signal
       ↓
   ┌──────────────────────────────────────────┐
   │ 1. Check current task interruptibility   │
   │ 2. If interruptible: _WAKING phase      │
   │ 3. If not: _FINISHING_TASK phase, wait   │
   └──────────────────────────────────────────┘
       ↓
_WAKING phase — Discord: "waking up..."
       ↓
   ┌──────────────────────────────────────────┐
   │ PARALLEL WAKE STRATEGY:                  │
   │                                          │
   │ Track A (immediate, ~2s):                │
   │  1. Load prime.md checkpoint             │
   │  2. Format as REVIEW context (NOT prompt)│
   │  3. Route first message to CPU Lite      │
   │  4. CPU Lite responds with REVIEW context│
   │                                          │
   │ Track B (background, ~37-60s):           │
   │  1. Start gaia-prime container           │
   │  2. Wait for /health endpoint            │
   │  3. Prefix caching rebuilds system prompt│
   │  4. Mark Prime as available              │
   └──────────────────────────────────────────┘
       ↓ Prime ready
Enter ACTIVE state — Discord: online, "watching over the studio"
       ↓
Process remaining queued messages via Prime with restored context
```

---

## Phase 1: Foundation (REQUIRED)

**Goal:** Implement core sleep/wake state machine with cognitive continuity
**Duration:** 2-3 weeks
**Deliverable:** GAIA can sleep when idle, wake when needed, preserve context

### 1.1 Sleep State Machine

**File:** `candidates/gaia-core/gaia_core/cognition/sleep_wake_manager.py` (NEW)

> **Architecture note:** This module lives in gaia-core, NOT gaia-common.
> gaia-common provides low-level primitives (IdleMonitor, TaskQueue).
> gaia-core owns the sleep/wake orchestration since it depends on gaia-common,
> not the other way around. This avoids a circular dependency.

```python
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger("GAIA.SleepWake")

class GaiaState(Enum):
    OFFLINE = "offline"         # System shut down, Discord invisible
    ACTIVE = "active"           # Normal operation (was AWAKE)
    DROWSY = "drowsy"           # Checkpoint in progress — cancellable
    ASLEEP = "asleep"           # Executing sleep tasks (was SLEEPING)
    DREAMING = "dreaming"       # GPU handed to Study for training
    DISTRACTED = "distracted"   # System under sustained CPU/GPU load

class _TransientPhase(Enum):
    """Internal waking sub-phases — not visible in the public state enum."""
    NONE = "none"
    FINISHING_TASK = "finishing_task"
    WAKING = "waking"

# Canned responses for states that don't forward to the model
CANNED_DREAMING = (
    "I'm studying right now and can't chat — "
    "I'll be back once my training session wraps up!"
)
CANNED_DISTRACTED = (
    "I'm a little occupied at the moment — "
    "give me a few minutes and I'll get back to you!"
)

class SleepWakeManager:
    """
    Manages GAIA's sleep/wake state transitions with cognitive continuity.

    6 public states + 2 internal transient phases:
        OFFLINE → ACTIVE → DROWSY → ASLEEP → DREAMING / DISTRACTED
        Internal phases: _FINISHING_TASK, _WAKING (sub-states of ASLEEP)

    Key design decisions:
    - DROWSY is cancellable: if a message arrives during checkpoint writing,
      we abort and return to ACTIVE immediately.
    - WAKING uses parallel strategy: CPU Lite handles the first queued message
      while Prime boots in the background (~37-60s from tmpfs).
    - prime.md checkpoint is the KV cache replacement.
    - DREAMING = GPU handed to Study (training); canned response only.
    - DISTRACTED = CPU or GPU under sustained load (>25% for 5s); canned response.
    """

    def __init__(self, config):
        self.config = config
        self.state = GaiaState.ACTIVE
        self._phase = _TransientPhase.NONE
        self.current_task = None
        self.wake_signal_pending = False
        self.prime_available = False
        self.checkpoint_manager = PrimeCheckpointManager(config)
        self.last_state_change = datetime.now(timezone.utc)
        self.dreaming_handoff_id = None

        logger.info("SleepWakeManager initialized")

    def get_state(self) -> GaiaState:
        return self.state

    def should_transition_to_drowsy(self, idle_minutes: float) -> bool:
        """
        Decides if GAIA should begin the sleep transition.

        Rules:
        - Must be ACTIVE
        - Idle > configured threshold (default 5 minutes)
        - No active streaming response
        """
        if self.state != GaiaState.ACTIVE:
            return False

        threshold = getattr(self.config, 'SLEEP_IDLE_THRESHOLD_MINUTES', 5)
        return idle_minutes >= threshold

    def initiate_drowsy(self, current_packet=None) -> bool:
        """
        Transition from ACTIVE to DROWSY.

        In DROWSY state, Prime writes its cognitive checkpoint.
        This is cancellable — if a message arrives, we abort and return to ACTIVE.

        Returns: True if entered DROWSY, False if failed
        """
        if self.state != GaiaState.ACTIVE:
            logger.warning(f"Cannot enter DROWSY from state: {self.state}")
            return False

        self.state = GaiaState.DROWSY
        self.last_state_change = datetime.utcnow()
        logger.info("Entering DROWSY state — writing checkpoint...")

        try:
            # Generate and write checkpoint (calls Prime)
            checkpoint_path = self.checkpoint_manager.create_checkpoint(current_packet)
            self.checkpoint_manager.rotate_checkpoints()

            # Check if we were interrupted during checkpoint
            if self.wake_signal_pending:
                logger.info("Message arrived during DROWSY — cancelling sleep")
                self.state = GaiaState.ACTIVE
                self.wake_signal_pending = False
                self.last_state_change = datetime.utcnow()
                return False

            # Checkpoint complete — enter SLEEPING
            self.state = GaiaState.ASLEEP
            self.last_state_change = datetime.utcnow()
            logger.info(f"Checkpoint written: {checkpoint_path} — entering ASLEEP")
            return True

        except Exception as e:
            logger.error(f"Checkpoint failed: {e}", exc_info=True)
            self.state = GaiaState.ACTIVE  # Fail safe: stay active
            self.last_state_change = datetime.utcnow()
            return False

    def receive_wake_signal(self):
        """
        Called by gaia-web (via POST /sleep/wake) when a message is queued.
        """
        self.wake_signal_pending = True

        if self.state == GaiaState.DROWSY:
            # Cancel checkpoint and return to ACTIVE (handled in initiate_drowsy)
            logger.info("Wake signal during DROWSY — will cancel checkpoint")

        elif self.state == GaiaState.ASLEEP:
            logger.info("Wake signal during ASLEEP")
            if self.current_task and not self.current_task.get("interruptible", True):
                logger.info(f"Non-interruptible task running: {self.current_task.get('task_id')}")
                self.state = _TransientPhase.FINISHING_TASK
            else:
                self.transition_to_waking()

        elif self.state == GaiaState.ACTIVE:
            logger.debug("Wake signal received but already awake")
            self.wake_signal_pending = False

    def transition_to_waking(self):
        """Move to WAKING state. Begins parallel wake strategy."""
        if self.state not in (GaiaState.ASLEEP, _TransientPhase.FINISHING_TASK):
            logger.warning(f"Cannot wake from state: {self.state}")
            return

        self.state = _TransientPhase.WAKING
        self.last_state_change = datetime.utcnow()
        logger.info("Entering WAKING state — starting parallel wake")

    def complete_wake(self) -> Dict[str, Any]:
        """
        Complete wake-up and return restored context.

        Called after CPU Lite has already handled the first message.
        Prime is now available in the background.

        Returns: {
            "checkpoint_loaded": bool,
            "context": str (REVIEW-formatted checkpoint),
            "timestamp": str
        }
        """
        if self.state != _TransientPhase.WAKING:
            logger.warning(f"Cannot complete wake from state: {self.state}")
            return {"checkpoint_loaded": False}

        try:
            checkpoint = self.checkpoint_manager.load_latest()
            review_context = self._format_checkpoint_as_review(checkpoint)

            self.state = GaiaState.ACTIVE
            self.wake_signal_pending = False
            self.prime_available = True
            self.last_state_change = datetime.utcnow()

            logger.info("Wake complete, context restored")
            return {
                "checkpoint_loaded": bool(checkpoint),
                "context": review_context,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Wake completion failed: {e}", exc_info=True)
            self.state = GaiaState.ACTIVE
            self.wake_signal_pending = False
            return {"checkpoint_loaded": False, "error": str(e)}

    def _format_checkpoint_as_review(self, checkpoint: str) -> str:
        """
        Format checkpoint as REVIEW material, NOT as a prompt.

        Critical: This is injected at Tier 1 in prompt_builder alongside
        session summaries. The model must understand this is context
        restoration, not a user message requiring a response.
        """
        if not checkpoint:
            return ""

        return f"""[SLEEP RESTORATION CONTEXT — Internal Review Only]
These are your notes from your last active session before sleep.
Use them to restore your working context. Do not respond to them directly.

{checkpoint}

Context restoration timestamp: {datetime.utcnow().isoformat()}"""

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "wake_signal_pending": self.wake_signal_pending,
            "prime_available": self.prime_available,
            "current_task": self.current_task.get("task_id") if self.current_task else None,
            "last_state_change": self.last_state_change.isoformat(),
            "seconds_in_state": (datetime.utcnow() - self.last_state_change).total_seconds()
        }
```

**File:** `candidates/gaia-core/gaia_core/cognition/prime_checkpoint.py` (NEW)

```python
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("GAIA.Checkpoint")

class PrimeCheckpointManager:
    """
    Manages Prime model's cognitive state checkpointing.

    Checkpoints preserve working memory across GPU sleep/wake cycles.
    This is the natural-language replacement for KV cache persistence:
    we can't serialize vLLM's KV cache across container restarts, but
    we CAN have Prime write down what it was thinking about.

    Storage: Uses the existing SHARED_DIR volume mount (/shared) which
    persists across container restarts via Docker named volume.
    """

    def __init__(self, config):
        self.config = config
        # Use the existing shared volume mount — survives container restarts
        shared_dir = getattr(config, 'SHARED_DIR', os.getenv('SHARED_DIR', '/shared'))
        self.checkpoint_dir = Path(shared_dir) / "sleep_state"
        self.checkpoint_file = self.checkpoint_dir / "prime.md"
        self.backup_file = self.checkpoint_dir / "prime_previous.md"
        self.history_dir = self.checkpoint_dir / "prime_history"

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def create_checkpoint(self, packet=None) -> Path:
        """
        Generate and write cognitive state checkpoint.
        
        Uses Prime itself to summarize current state, then writes to prime.md.
        
        Args:
            packet: Optional CognitionPacket with current context
        
        Returns: Path to written checkpoint
        """
        logger.info("💾 Creating cognitive checkpoint...")
        
        try:
            # Generate state summary
            state_summary = self._generate_state_summary(packet)
            
            # Write to checkpoint file
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                f.write(state_summary)
            
            logger.info(f"✅ Checkpoint written: {self.checkpoint_file}")
            return self.checkpoint_file
            
        except Exception as e:
            logger.error(f"❌ Checkpoint creation failed: {e}", exc_info=True)
            raise
    
    def _generate_state_summary(self, packet) -> str:
        """
        Use Prime to generate a summary of current cognitive state.
        
        This is a meta-cognitive task: Prime reflects on its own state.
        """
        # Build meta-cognitive prompt
        meta_prompt = self._build_state_summary_prompt(packet)
        
        # Call Prime with very low temperature for deterministic summary
        # TODO: Integrate with model pool to call Prime
        # For now, use template
        
        state_summary = self._template_checkpoint(packet)
        return state_summary
    
    def _build_state_summary_prompt(self, packet) -> str:
        """
        Construct prompt for Prime to summarize its own state.
        
        This prompt asks Prime to write notes for itself to read later.
        """
        if not packet:
            return """You are about to enter a sleep state. Before doing so, write a brief summary of your current cognitive state that you can review when you wake up.

Include:
- What you were discussing
- Key entities or concepts mentioned
- Your current reasoning state
- Tone and relationship context
- Any pending actions or follow-ups

Format as clear notes to yourself. Be concise but complete."""
        
        return f"""You are about to enter a sleep state. Before doing so, write a brief summary of your current cognitive state that you can review when you wake up.

Current context:
- Session ID: {packet.header.session_id if packet.header else 'unknown'}
- Last user message: {packet.content.original_prompt if packet.content else 'none'}
- Your persona: {packet.persona if hasattr(packet, 'persona') else 'default'}

Include in your summary:
1. What you were discussing (topic, context)
2. Key entities or concepts mentioned
3. Your current reasoning state (hypotheses, confidence)
4. Tone and relationship context (how you're relating to the user)
5. Any pending actions or follow-ups

Write this as clear notes to yourself. Be concise but complete.
Format as markdown with clear sections."""
    
    def _template_checkpoint(self, packet) -> str:
        """
        Template for checkpoint when Prime call isn't available.
        """
        timestamp = datetime.utcnow().isoformat()
        
        session_id = "unknown"
        last_prompt = "none"
        persona = "default"
        
        if packet:
            if hasattr(packet, 'header') and packet.header:
                session_id = packet.header.session_id or "unknown"
            if hasattr(packet, 'content') and packet.content:
                last_prompt = packet.content.original_prompt or "none"
            if hasattr(packet, 'persona'):
                persona = packet.persona or "default"
        
        return f"""# Prime Cognitive State Checkpoint
**Last Updated:** {timestamp}
**Session ID:** {session_id}
**State:** SLEEP_INITIATED

## Active Context Summary
{self._extract_context_summary(packet)}

## Conversation State
**Last user message:** {last_prompt}
**Persona:** {persona}
**Response status:** Complete

## Key Entities Referenced
{self._extract_entities(packet)}

## Reasoning State
**Current task:** Context preservation across sleep/wake cycle
**Confidence:** High (checkpoint system operational)

## Tone & Relationship Context
Technical collaboration with system architect
Detail-oriented explanations expected
Iterative design process

## Next Expected Actions
If woken: Process queued messages with this context available
If continuing sleep: Proceed with scheduled sleep tasks

## Notes
This checkpoint was generated automatically during sleep transition.
Review this content to restore working memory context.
"""
    
    def _extract_context_summary(self, packet) -> str:
        """Extract brief summary from packet context."""
        if not packet:
            return "No active context"
        
        if hasattr(packet, 'content') and packet.content:
            prompt = packet.content.original_prompt or ""
            if prompt:
                return f"Last interaction: {prompt[:200]}"
        
        return "Context restoration in progress"
    
    def _extract_entities(self, packet) -> str:
        """Extract key entities from packet."""
        if not packet:
            return "- (no entities tracked)"
        
        # TODO: Integrate with entity extraction if available
        return "- System architecture discussion\n- Sleep cycle implementation\n- Cognitive continuity"
    
    def load_latest(self) -> str:
        """
        Load the most recent checkpoint.
        
        Returns: Checkpoint content as string
        """
        if not self.checkpoint_file.exists():
            logger.warning("No checkpoint file found")
            return ""
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"📖 Checkpoint loaded: {len(content)} chars")
            return content
            
        except Exception as e:
            logger.error(f"❌ Checkpoint load failed: {e}", exc_info=True)
            return ""
    
    def rotate_checkpoints(self):
        """
        Backup current checkpoint and archive to history.
        
        prime.md → prime_previous.md
        prime.md → prime_history/YYYY-MM-DD_HH-MM-SS.md
        """
        if not self.checkpoint_file.exists():
            return
        
        try:
            # Copy to backup
            if self.checkpoint_file.exists():
                import shutil
                shutil.copy2(self.checkpoint_file, self.backup_file)
            
            # Archive to history with timestamp
            timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            archive_path = self.history_dir / f"{timestamp}-sleep.md"
            
            import shutil
            shutil.copy2(self.checkpoint_file, archive_path)
            
            logger.info(f"📦 Checkpoint archived: {archive_path}")
            
        except Exception as e:
            logger.error(f"❌ Checkpoint rotation failed: {e}", exc_info=True)
    
    def get_checkpoint_history(self, limit: int = 10) -> list:
        """
        Returns list of recent checkpoint files.
        
        Args:
            limit: Maximum number of checkpoints to return
        
        Returns: List of (timestamp, path) tuples
        """
        if not self.history_dir.exists():
            return []
        
        checkpoints = []
        for path in sorted(self.history_dir.glob("*.md"), reverse=True)[:limit]:
            checkpoints.append((path.stem, path))
        
        return checkpoints
```

### 1.2 Message Queue System

**File:** `candidates/gaia-web/gaia_web/queue/message_queue.py` (NEW)

```python
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging
import asyncio

logger = logging.getLogger("GAIA.MessageQueue")

@dataclass
class QueuedMessage:
    """A message waiting to be processed."""
    message_id: str
    content: str
    source: str  # "discord", "web", "cli"
    session_id: str
    priority: int = 0  # Higher = more urgent
    queued_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MessageQueue:
    """
    Thread-safe message queue for sleep/wake cycle.
    
    Messages arrive during sleep and trigger wake-up.
    """
    
    def __init__(self):
        self.queue: List[QueuedMessage] = []
        self.lock = asyncio.Lock()
        self.wake_signal_sent = False
        self.core_url = "http://gaia-core-candidate:6416"  # Will be configurable
        
        logger.info("📬 MessageQueue initialized")
    
    async def enqueue(self, message: QueuedMessage) -> bool:
        """
        Add message to queue and send wake signal if needed.
        
        Returns: True if message was queued, False if rejected
        """
        async with self.lock:
            # Add to queue
            self.queue.append(message)
            logger.info(f"📥 Message queued: {message.message_id} from {message.source}")
            
            # Send wake signal to core (only once per wake cycle)
            if not self.wake_signal_sent:
                await self._send_wake_signal()
                self.wake_signal_sent = True
            
            return True
    
    async def dequeue(self) -> Optional[QueuedMessage]:
        """
        Remove and return highest priority message.
        
        Returns: QueuedMessage or None if queue empty
        """
        async with self.lock:
            if not self.queue:
                return None
            
            # Sort by priority (descending), then by queued_at (ascending)
            self.queue.sort(key=lambda m: (-m.priority, m.queued_at))
            
            message = self.queue.pop(0)
            logger.info(f"📤 Message dequeued: {message.message_id}")
            
            # If queue is now empty, reset wake signal flag
            if not self.queue:
                self.wake_signal_sent = False
            
            return message
    
    async def peek(self) -> Optional[QueuedMessage]:
        """View next message without removing."""
        async with self.lock:
            if not self.queue:
                return None
            self.queue.sort(key=lambda m: (-m.priority, m.queued_at))
            return self.queue[0]
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Returns queue statistics."""
        async with self.lock:
            return {
                "count": len(self.queue),
                "wake_signal_sent": self.wake_signal_sent,
                "oldest_message_age_seconds": (
                    (datetime.utcnow() - self.queue[0].queued_at).total_seconds()
                    if self.queue else 0
                )
            }
    
    async def _send_wake_signal(self):
        """
        Send HTTP POST to gaia-core to wake from sleep.
        """
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.core_url}/core/wake",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    logger.info("✅ Wake signal sent to core")
                else:
                    logger.warning(f"⚠️ Wake signal returned {response.status_code}")
                    
        except Exception as e:
            logger.error(f"❌ Wake signal failed: {e}", exc_info=True)
```

**File:** `candidates/gaia-web/gaia_web/api/routes.py` (MODIFY - add queue endpoints)

```python
# Add to existing routes

@app.post("/queue/enqueue")
async def enqueue_message(message_data: dict):
    """
    Enqueue a message for processing.
    Called by Discord bot, web interface, etc.
    """
    from gaia_web.queue.message_queue import QueuedMessage
    
    message = QueuedMessage(
        message_id=message_data.get("message_id"),
        content=message_data.get("content"),
        source=message_data.get("source", "unknown"),
        session_id=message_data.get("session_id"),
        priority=message_data.get("priority", 0),
        metadata=message_data.get("metadata", {})
    )
    
    success = await app.state.message_queue.enqueue(message)
    
    return {
        "queued": success,
        "message_id": message.message_id
    }

@app.get("/queue/status")
async def get_queue_status():
    """Get queue statistics."""
    return await app.state.message_queue.get_queue_status()
```

### 1.3 Sleep Endpoints (APIRouter)

**File:** `candidates/gaia-core/gaia_core/api/sleep_endpoints.py` (NEW)

> Follows the existing pattern from `gpu_endpoints.py`: separate router file,
> registered in `main.py` via `app.include_router(sleep_router)`.

```python
from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/sleep", tags=["sleep"])

@router.post("/wake")
async def receive_wake_signal(request: Request):
    """
    Receive wake signal from gaia-web.
    Called when first message is queued during sleep.
    """
    manager = request.app.state.sleep_wake_manager
    if not manager:
        return JSONResponse(status_code=500, content={"error": "SleepWakeManager not initialized"})

    manager.receive_wake_signal()

    return {
        "received": True,
        "state": manager.get_state().value,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/status")
async def get_sleep_status(request: Request):
    """Get current sleep/wake state and task info."""
    manager = request.app.state.sleep_wake_manager
    if not manager:
        return JSONResponse(status_code=500, content={"error": "SleepWakeManager not initialized"})

    return manager.get_status()
```

**File:** `candidates/gaia-core/gaia_core/main.py` (MODIFY — register router)

```python
# Add alongside existing gpu_router registration:
from gaia_core.api.sleep_endpoints import router as sleep_router
app.include_router(sleep_router)
```

#### Additional Endpoints (7-State Refactor)

**`POST /sleep/study-handoff`** — Orchestrator notifies core that GPU ownership is changing for Study training.

```python
@router.post("/study-handoff")
async def study_handoff(request: Request):
    body = await request.json()
    direction = body.get("direction")    # "prime_to_study" or "study_to_prime"
    handoff_id = body.get("handoff_id")
    manager = request.app.state.sleep_wake_manager
    if direction == "prime_to_study":
        manager.enter_dreaming(handoff_id)
    elif direction == "study_to_prime":
        manager.exit_dreaming(handoff_id)
    return {"state": manager.get_state().value}
```

**`GET /sleep/distracted-check`** — gaia-web calls this before forwarding every message. If GAIA is DREAMING or DISTRACTED, a canned response is returned instead of invoking the LLM.

```python
@router.get("/distracted-check")
async def distracted_check(request: Request):
    manager = request.app.state.sleep_wake_manager
    state = manager.get_state()
    canned = manager.get_canned_response()
    return {"state": state.value, "canned_response": canned}
```

**`POST /sleep/shutdown`** — Triggers graceful OFFLINE transition. The sleep cycle loop sets state to OFFLINE, updates Discord to invisible, then shuts down.

```python
@router.post("/shutdown")
async def shutdown(request: Request):
    loop = request.app.state.sleep_cycle_loop
    loop.initiate_shutdown()
    return {"state": "offline", "message": "Shutdown initiated"}
```

### 1.4 Sleep Cycle Loop (in gaia-core, NOT gaia-common)

> **Critical architecture decision:** The existing `processor.py` in gaia-common
> has a broken import (`from app.cognition.initiative_handler import gil_check_and_generate`)
> and a circular dependency if we add gaia-core imports. Instead, the sleep cycle
> loop lives in gaia-core as a new module that USES gaia-common primitives (IdleMonitor)
> but doesn't require gaia-common to import gaia-core.
>
> The legacy `processor.py` in gaia-common should be left as-is (or cleaned up separately)
> since it's not actively used in the v0.3 architecture.

**File:** `candidates/gaia-core/gaia_core/cognition/sleep_cycle_loop.py` (NEW)

```python
"""
Sleep cycle loop — runs as a daemon thread in gaia-core.

Uses gaia-common primitives (IdleMonitor) for idle detection,
but owns all sleep/wake orchestration logic.
"""

import asyncio
import threading
import time
import logging
from gaia_common.utils.background.idle_monitor import IdleMonitor
from gaia_core.cognition.sleep_wake_manager import SleepWakeManager, GaiaState

logger = logging.getLogger("GAIA.SleepCycle")

class SleepCycleLoop:
    """
    Background thread that monitors idle state and drives sleep/wake transitions.
    Replaces the legacy BackgroundProcessor for v0.3 architecture.
    """

    def __init__(self, config, discord_connector=None):
        self.config = config
        self.idle_monitor = IdleMonitor()
        self.sleep_wake_manager = SleepWakeManager(config)
        self.discord_connector = discord_connector
        self.thread = None
        self.running = False

    def start(self):
        if not self.thread:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("Sleep cycle loop started")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _run(self):
        while self.running:
            try:
                idle_minutes = self.idle_monitor.get_idle_minutes()
                state = self.sleep_wake_manager.get_state()

                if state == GaiaState.ACTIVE:
                    if self.sleep_wake_manager.should_transition_to_drowsy(idle_minutes):
                        logger.info(f"Idle for {idle_minutes:.1f}min — entering DROWSY")
                        self._update_presence("Drifting off...")
                        success = self.sleep_wake_manager.initiate_drowsy()
                        if success:
                            self._update_presence("Sleeping")
                        else:
                            self._update_presence(None)  # Reset to idle

                elif state == GaiaState.ASLEEP:
                    # Phase 2 will add task execution here
                    # For now, just poll for wake signal
                    pass

                elif state == _TransientPhase.WAKING:
                    restored = self.sleep_wake_manager.complete_wake()
                    if restored.get("checkpoint_loaded"):
                        logger.info("Context restored from checkpoint")
                    self._update_presence(None)  # Reset to dynamic idle

                time.sleep(10)

            except Exception as e:
                logger.error(f"Sleep cycle error: {e}", exc_info=True)
                time.sleep(15)

    def _update_presence(self, status_text):
        """Update Discord presence. None resets to dynamic idle."""
        if self.discord_connector:
            if status_text is None:
                self.discord_connector.set_idle()
            else:
                self.discord_connector.update_presence(status_text)
```

**File:** `candidates/gaia-core/gaia_core/main.py` (MODIFY — start sleep loop in lifespan)

```python
# Add to the existing lifespan context manager:
from gaia_core.cognition.sleep_cycle_loop import SleepCycleLoop

async def lifespan(app):
    # ... existing initialization ...

    # Start sleep cycle loop
    sleep_loop = SleepCycleLoop(
        config=app.state.config,
        discord_connector=app.state.discord_connector
    )
    app.state.sleep_wake_manager = sleep_loop.sleep_wake_manager
    sleep_loop.start()

    yield

    sleep_loop.stop()
```

**File:** `candidates/gaia-common/gaia_common/utils/background/processor.py` (MODIFY — fix broken import)

```python
# Remove the broken import on line 14:
#   from app.cognition.initiative_handler import gil_check_and_generate
#
# Replace with a no-op or remove the GIL call entirely.
# The GIL is being revived in gaia-core as part of Phase 2 (see section 2.2).
# processor.py in gaia-common is legacy and not used in v0.3 architecture.
```

### 1.5 Discord Status Updates

> **No new code needed.** The existing `DiscordConnector` in gaia-common already
> provides `update_presence(activity_name)` and `set_idle()` with thread-safe,
> rate-limited (12s interval) Discord presence updates. The sleep cycle loop
> calls these directly via the injected `discord_connector` reference.

#### Discord Presence Table

| GaiaState | Discord Status | Activity Text | Notes |
|-----------|---------------|---------------|-------|
| ACTIVE | `online` | `"watching over the studio"` | Dynamic via `set_idle()` |
| DROWSY | `online` | `"drifting off..."` | Checkpoint writing in progress |
| ASLEEP | `idle` | `"sleeping..."` | Running sleep tasks |
| DREAMING | `dnd` | `"studying..."` | GPU handed to Study for training |
| DISTRACTED | `dnd` | `"occupied..."` | Sustained CPU/GPU load detected |
| OFFLINE | `invisible` | *(none)* | Graceful shutdown, `activity=None` |

> **Implementation:** `gaia-web/gaia_web/main.py` maps status strings. The
> `status_map` includes `"invisible": discord.Status.invisible`. When status
> is invisible, activity is set to `None` (no activity text shown).

### Phase 1 Testing Strategy

#### Unit Tests

**File:** `candidates/gaia-core/tests/test_sleep_wake_manager.py` (NEW)

```python
import pytest
from gaia_core.cognition.sleep_wake_manager import SleepWakeManager, GaiaState
from unittest.mock import MagicMock

def test_initial_state_is_awake():
    config = MagicMock()
    manager = SleepWakeManager(config)
    assert manager.get_state() == GaiaState.ACTIVE

def test_should_not_drowsy_when_not_idle():
    config = MagicMock()
    manager = SleepWakeManager(config)
    assert not manager.should_transition_to_drowsy(idle_minutes=2.0)

def test_should_drowsy_when_idle():
    config = MagicMock()
    manager = SleepWakeManager(config)
    assert manager.should_transition_to_drowsy(idle_minutes=6.0)

def test_drowsy_transitions_to_sleeping():
    config = MagicMock()
    manager = SleepWakeManager(config)
    manager.initiate_drowsy()
    assert manager.get_state() == GaiaState.ASLEEP

def test_wake_during_drowsy_cancels_sleep():
    config = MagicMock()
    manager = SleepWakeManager(config)
    manager.state = GaiaState.DROWSY
    manager.receive_wake_signal()
    assert manager.wake_signal_pending
    # The actual cancellation happens in initiate_drowsy() when it checks the flag

def test_wake_signal_during_sleep():
    config = MagicMock()
    manager = SleepWakeManager(config)
    manager.state = GaiaState.ASLEEP
    manager.current_task = None  # No task running
    manager.receive_wake_signal()
    assert manager.get_state() == _TransientPhase.WAKING

def test_wake_signal_during_non_interruptible_task():
    config = MagicMock()
    manager = SleepWakeManager(config)
    manager.state = GaiaState.ASLEEP
    manager.current_task = {"task_id": "qlora_training", "interruptible": False}
    manager.receive_wake_signal()
    assert manager.get_state() == _TransientPhase.FINISHING_TASK

def test_complete_wake_returns_to_awake():
    config = MagicMock()
    manager = SleepWakeManager(config)
    manager.state = _TransientPhase.WAKING
    result = manager.complete_wake()
    assert manager.get_state() == GaiaState.ACTIVE
    assert "checkpoint_loaded" in result
```

#### Integration Tests

**Test Scenario 1: Full Sleep → Wake → Process Message**

```python
# File: candidates/gaia-core/tests/integration/test_sleep_cycle.py

@pytest.mark.asyncio
async def test_full_sleep_wake_cycle():
    """
    Test: ACTIVE → DROWSY → ASLEEP → _WAKING → ACTIVE
    """
    config = load_test_config()
    manager = SleepWakeManager(config)

    # Go idle → DROWSY → SLEEPING
    assert manager.should_transition_to_drowsy(idle_minutes=6.0)
    assert manager.initiate_drowsy()  # Writes checkpoint, transitions to ASLEEP
    assert manager.get_state() == GaiaState.ASLEEP

    # Message arrives → WAKING
    manager.receive_wake_signal()
    assert manager.get_state() == _TransientPhase.WAKING

    # Complete wake → ACTIVE
    result = manager.complete_wake()
    assert result["checkpoint_loaded"]
    assert manager.get_state() == GaiaState.ACTIVE

@pytest.mark.asyncio
async def test_drowsy_cancellation():
    """
    Test: ACTIVE → DROWSY → (message arrives) → ACTIVE (sleep cancelled)
    """
    config = load_test_config()
    manager = SleepWakeManager(config)

    # Simulate message arriving while checkpoint is being written
    manager.wake_signal_pending = True  # Set before initiate_drowsy
    result = manager.initiate_drowsy()
    assert not result  # Should return False (cancelled)
    assert manager.get_state() == GaiaState.ACTIVE
```

**Test Scenario 2: Checkpoint Persistence**

```python
@pytest.mark.asyncio
async def test_checkpoint_persists_across_restart():
    """
    Test that checkpoints survive process restart (via shared volume).
    """
    config = load_test_config()

    # Create checkpoint
    manager1 = PrimeCheckpointManager(config)
    checkpoint_path = manager1.create_checkpoint(test_packet)
    assert checkpoint_path.exists()

    # Simulate restart — new instance reads same volume
    manager2 = PrimeCheckpointManager(config)
    loaded = manager2.load_latest()
    assert len(loaded) > 0
    assert "Cognitive State Checkpoint" in loaded
```

### Phase 1 Success Criteria

- [ ] State machine transitions correctly (ACTIVE → DROWSY → ASLEEP → _WAKING → ACTIVE)
- [ ] DROWSY state cancels cleanly if message arrives during checkpoint writing
- [ ] Idle detection triggers DROWSY after 5 minutes
- [ ] Prime generates cognitive checkpoint (prime.md) during DROWSY
- [ ] Checkpoint persists across container restarts (shared volume)
- [ ] Checkpoint loaded and formatted as REVIEW context on wake
- [ ] Wake signal interrupts interruptible tasks, waits for non-interruptible
- [ ] Message queue holds messages during sleep
- [ ] CPU Lite handles first queued message while Prime boots (parallel wake)
- [ ] Discord status reflects state (Drifting off / Sleeping / Waking up / dynamic idle)
- [ ] No message loss during transitions
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Broken import in gaia-common/processor.py fixed

---

## Phase 2: Task System (DEPENDS ON PHASE 1)

**Goal:** Make sleep productive with autonomous tasks  
**Duration:** 2-3 weeks  
**Deliverable:** GAIA performs maintenance, learning, and reflection during sleep  

### 2.1 Sleep Task Scheduler

**File:** `candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py` (NEW)

```python
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger("GAIA.SleepTasks")

@dataclass
class SleepTask:
    """A task to execute during sleep."""
    task_id: str
    task_type: str  # "thought_seed_hydration", "conversation_sanitation", etc.
    priority: int  # 1 = highest, 5 = lowest
    interruptible: bool  # Can be interrupted by wake signal
    estimated_duration_seconds: int
    function: Callable
    params: Dict[str, Any]
    last_run: Optional[datetime] = None
    run_count: int = 0

class SleepTaskScheduler:
    """
    Manages execution of tasks during sleep state.
    
    Tasks are prioritized and executed until wake signal or completion.
    """
    
    def __init__(self, config):
        self.config = config
        self.tasks: List[SleepTask] = []
        self.current_task: Optional[SleepTask] = None
        self._register_default_tasks()
        
        logger.info("📋 SleepTaskScheduler initialized")
    
    def _register_default_tasks(self):
        """Register built-in sleep tasks."""
        
        # Priority 1: Quick, interruptible maintenance
        self.register_task(SleepTask(
            task_id="thought_seed_hydration",
            task_type="maintenance",
            priority=1,
            interruptible=True,
            estimated_duration_seconds=120,
            function=self._task_thought_seed_hydration,
            params={}
        ))
        
        self.register_task(SleepTask(
            task_id="conversation_sanitation",
            task_type="maintenance",
            priority=1,
            interruptible=True,
            estimated_duration_seconds=180,
            function=self._task_conversation_sanitation,
            params={}
        ))
        
        self.register_task(SleepTask(
            task_id="vector_reflection",
            task_type="learning",
            priority=2,
            interruptible=True,
            estimated_duration_seconds=240,
            function=self._task_vector_reflection,
            params={}
        ))
        
        # Priority 3: Longer tasks (Phase 3)
        # QLoRA training, blueprint processing, etc.
    
    def register_task(self, task: SleepTask):
        """Add a task to the scheduler."""
        self.tasks.append(task)
        logger.info(f"✅ Task registered: {task.task_id} (priority {task.priority})")
    
    def get_next_task(self) -> Optional[SleepTask]:
        """
        Get highest priority task that's ready to run.
        
        Selection criteria:
        1. Priority (lower number = higher priority)
        2. Least recently run
        3. Never run takes precedence
        """
        if not self.tasks:
            return None
        
        # Sort by priority, then by last_run (None first)
        available = [t for t in self.tasks]
        available.sort(key=lambda t: (
            t.priority,
            t.last_run if t.last_run else datetime.min
        ))
        
        if available:
            task = available[0]
            self.current_task = task
            return task
        
        return None
    
    async def execute_task(self, task: SleepTask) -> bool:
        """
        Execute a sleep task.
        
        Returns: True if successful, False if failed
        """
        logger.info(f"🔄 Executing task: {task.task_id}")
        start_time = datetime.utcnow()
        
        try:
            # Call task function
            await task.function(**task.params)
            
            # Update task metadata
            task.last_run = datetime.utcnow()
            task.run_count += 1
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"✅ Task complete: {task.task_id} ({duration:.1f}s)")
            
            self.current_task = None
            return True
            
        except Exception as e:
            logger.error(f"❌ Task failed: {task.task_id}: {e}", exc_info=True)
            self.current_task = None
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Return scheduler status."""
        return {
            "total_tasks": len(self.tasks),
            "current_task": self.current_task.task_id if self.current_task else None,
            "tasks": [
                {
                    "id": t.task_id,
                    "priority": t.priority,
                    "last_run": t.last_run.isoformat() if t.last_run else None,
                    "run_count": t.run_count
                }
                for t in self.tasks
            ]
        }
    
    # ========== Task Implementations ==========
    
    async def _task_thought_seed_hydration(self):
        """
        Review and process unreviewed thought seeds.

        Calls existing thought_seed.review_and_process_seeds() which already
        supports auto_act mode (lines 108-156 of thought_seed.py).

        PREREQUISITE: Seeds must actually be generated during normal operation.
        The THOUGHT_SEED: directive is already parsed by the output router, but
        the Observer's system prompt must include instructions to emit these
        directives when it notices something worth reflecting on later.
        See Phase 2 prerequisite task below.
        """
        logger.info("Starting thought seed hydration...")

        from gaia_core.cognition.thought_seed import review_and_process_seeds

        try:
            # Use CPU Lite model for review (GPU may be unavailable during sleep)
            llm = self.config.model_pool.get_model_for_role("lite")

            review_and_process_seeds(
                config=self.config,
                llm=llm,
                auto_act=True  # Execute seed actions during sleep
            )

            logger.info("Thought seed hydration complete")

        except Exception as e:
            logger.error(f"Thought seed hydration failed: {e}", exc_info=True)
            raise
    
    async def _task_conversation_sanitation(self):
        """
        Check recent Discord conversations for accuracy.
        Edit messages if errors found.
        """
        logger.info("🧹 Starting conversation sanitation...")
        
        # TODO: Implement conversation checking
        # 1. Get recent messages from Discord
        # 2. Extract factual claims
        # 3. Verify against knowledge base
        # 4. Edit messages if corrections needed
        # 5. Log notable conversations for NotebookLM
        
        logger.info("⚠️ Conversation sanitation not yet implemented")
    
    async def _task_vector_reflection(self):
        """
        Review vector store against local documentation.
        Update stale embeddings.
        """
        logger.info("🔍 Starting vector reflection...")
        
        # TODO: Implement vector store reflection
        # 1. Compare vector store entries with source files
        # 2. Identify stale or outdated embeddings
        # 3. Re-embed updated documents
        # 4. Remove orphaned entries
        
        logger.info("⚠️ Vector reflection not yet implemented")
```

### 2.2 Revived Initiative Loop (GIL)

> **Historical context:** GAIA had a fully functional Initiative Loop called GIL
> (GAIA Initiative Loop) in the monolith architecture. The complete implementation
> exists in `archive/gaia-assistant-monolith/run_gil.py`. During the Jan 2026
> modular refactoring, most infrastructure was migrated but the core
> `initiative_handler` integration was lost, leaving a broken import in
> `gaia-common/utils/background/processor.py` line 14.
>
> Rather than reimplementing from scratch, we port the archived GIL design into
> the sleep task system. The key pattern from `run_gil.py`:
> 1. Check idle state
> 2. Prioritize topics from topic_cache.json
> 3. Generate self-prompt for highest-priority topic
> 4. Feed through AgentCore.run_turn() for full cognitive processing

**File:** `candidates/gaia-core/gaia_core/cognition/initiative_engine.py` (NEW — ports archived run_gil.py)

```python
"""
GAIA Initiative Loop (GIL) — revived from archive/gaia-assistant-monolith/run_gil.py.

During sleep, this engine processes high-priority topics from the topic cache
by generating self-prompts and feeding them through the cognitive pipeline.
"""

import logging
from typing import Optional, Dict, Any
from gaia_core.cognition.topic_manager import prioritize_topics

logger = logging.getLogger("GAIA.InitiativeEngine")

GIL_SESSION_ID = "gaia_initiative_loop_session"
TOPIC_CACHE_PATH = "/knowledge/system_reference/topic_cache.json"

class InitiativeEngine:
    """
    Autonomous topic processing during sleep.
    Ported from archived run_gil.py with v0.3 architecture adaptations.
    """

    def __init__(self, config, agent_core=None):
        self.config = config
        self.agent_core = agent_core  # Injected from main.py

    async def execute_initiative_turn(self) -> Optional[Dict[str, Any]]:
        """
        Single autonomous thought cycle.
        Returns dict with results or None if no topics available.
        """
        top_topics = prioritize_topics(TOPIC_CACHE_PATH, top_n=1)
        if not top_topics:
            logger.info("No active topics — GAIA is at peace")
            return None

        topic = top_topics[0]
        topic_id = topic.get("topic_id")
        topic_desc = topic.get("topic")
        logger.info(f"Selected topic: [{topic_id}] — {topic_desc}")

        self_prompt = f"""[Autonomous Reflection Cycle]
My current highest-priority, unresolved topic is: '{topic_desc}'.
The topic's metadata is: {topic}.

My task is to analyze this topic and decide on the next step.
- If I have enough information to resolve it, I will use the resolve_topic primitive.
- If I need to do more work or break it down, I will use the update_topic primitive.
- If the task requires writing code or a document, I will use the ai.write primitive.

Based on this, what is my next logical action?"""

        if self.agent_core:
            # Feed through full cognitive pipeline
            result = await self.agent_core.process_initiative(
                prompt=self_prompt,
                session_id=GIL_SESSION_ID
            )
            logger.info(f"GIL turn complete for topic [{topic_id}]")
            return result

        logger.warning("AgentCore not available for GIL turn")
        return None
```

**Integrate into SleepCycleLoop** (modify `sleep_cycle_loop.py` from Phase 1):

```python
# In the SLEEPING state handler, add task execution:

elif state == GaiaState.ASLEEP:
    task = self.task_scheduler.get_next_task()
    if task:
        self.sleep_wake_manager.current_task = {
            "task_id": task.task_id,
            "interruptible": task.interruptible
        }
        asyncio.run(self.task_scheduler.execute_task(task))
        self.sleep_wake_manager.current_task = None

        # Re-check wake signal after each task
        if self.sleep_wake_manager.wake_signal_pending:
            self.sleep_wake_manager.transition_to_waking()
```

### 2.3 Conversation Sanitation (extends existing ConversationCurator)

> **Existing code:** `conversation_curator.py` already handles notable conversation
> detection and appending to `knowledge/conversation_examples.md` using heuristics.
> The sanitizer WRAPS the curator rather than duplicating its notability checks.
> The new piece is claim verification and Discord message editing.

**File:** `candidates/gaia-core/gaia_core/cognition/conversation_sanitizer.py` (NEW)

```python
"""
Sleep-mode conversation sanitizer.

Wraps the existing ConversationCurator for notability detection,
and adds claim verification + Discord message editing.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
from gaia_core.cognition.conversation_curator import ConversationCurator

logger = logging.getLogger("GAIA.ConversationSanitizer")

class ConversationSanitizer:
    """
    Reviews recent Discord conversations for accuracy during sleep.

    Two responsibilities:
    1. Accuracy checking: Extract factual claims from GAIA's messages,
       verify against knowledge base, edit if corrections needed.
    2. Curation: Delegates to existing ConversationCurator for notability
       detection and NotebookLM appending (no duplication).
    """

    def __init__(self, config, discord_connector=None, llm=None):
        self.config = config
        self.discord_connector = discord_connector
        self.llm = llm
        self.curator = ConversationCurator()  # Reuse existing curator

    async def sanitize_recent_conversations(self, hours: int = 24):
        """Check recent Discord messages for accuracy."""
        logger.info(f"Sanitizing conversations from last {hours}h...")

        if not self.discord_connector:
            logger.warning("Discord connector not available, skipping")
            return {"corrections": 0, "curated": 0}

        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            conversations = await self._fetch_recent_conversations(cutoff)
            logger.info(f"Found {len(conversations)} recent conversations")

            corrections_made = 0
            curated_count = 0

            for convo in conversations:
                # Step 1: Accuracy check on GAIA's messages
                gaia_msgs = [m for m in convo["messages"] if m.get("role") == "assistant"]
                for msg in gaia_msgs:
                    claims = await self._extract_claims(msg["content"])
                    if claims:
                        result = await self._verify_claims(claims)
                        if result.get("needs_correction"):
                            await self._edit_discord_message(
                                msg["discord_message_id"],
                                result["corrected_content"]
                            )
                            corrections_made += 1

                # Step 2: Delegate notability check to existing curator
                if self.curator.curate(convo["session_id"], convo["messages"]):
                    curated_count += 1

            logger.info(f"Sanitation complete: {corrections_made} corrections, "
                       f"{curated_count} notable conversations curated")
            return {"corrections": corrections_made, "curated": curated_count}

        except Exception as e:
            logger.error(f"Sanitation failed: {e}", exc_info=True)
            return {"corrections": 0, "curated": 0, "error": str(e)}

    async def _fetch_recent_conversations(self, since: datetime) -> List[Dict]:
        """Fetch grouped conversations from Discord since given time."""
        # TODO: Use discord_connector to fetch channel history
        # Group messages into conversations by session_id
        return []

    async def _extract_claims(self, content: str) -> List[str]:
        """Use CPU Lite to identify factual claims in a message."""
        if not self.llm:
            return []
        # TODO: LLM call to extract verifiable claims
        return []

    async def _verify_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Verify claims against vector store and documentation."""
        # TODO: RAG lookup + verification
        return {"needs_correction": False}

    async def _edit_discord_message(self, message_id: str, new_content: str):
        """Edit a Discord message via the bot."""
        # TODO: Use discord_connector to edit message
        # Note: Bot can only edit its own messages
        pass
```

### Phase 2 Testing Strategy

#### Unit Tests

```python
# File: candidates/gaia-core/tests/test_sleep_task_scheduler.py

def test_task_registration():
    scheduler = SleepTaskScheduler(mock_config)
    initial_count = len(scheduler.tasks)
    
    scheduler.register_task(test_task)
    assert len(scheduler.tasks) == initial_count + 1

def test_get_next_task_prioritizes_correctly():
    scheduler = SleepTaskScheduler(mock_config)
    
    # Should get priority 1 task first
    next_task = scheduler.get_next_task()
    assert next_task.priority == 1
```

#### Integration Tests

```python
# File: candidates/gaia-core/tests/integration/test_sleep_tasks.py

@pytest.mark.asyncio
async def test_thought_seed_hydration_during_sleep():
    """Test that thought seeds are processed during sleep."""
    # Create unreviewed seeds
    create_test_seed()
    
    # Enter sleep (ACTIVE → DROWSY → ASLEEP)
    manager.initiate_drowsy()
    
    # Execute task
    task = scheduler.get_next_task()
    assert task.task_id == "thought_seed_hydration"
    
    success = await scheduler.execute_task(task)
    assert success
    
    # Verify seeds were reviewed
    seeds = list_unreviewed_seeds()
    assert len(seeds) == 0
```

### 2.4 Thought Seed Pipeline Prerequisites

> **Problem:** The thought seed hydration task reviews seeds, but seeds aren't
> currently being generated during normal operation. The `THOUGHT_SEED:` directive
> is parsed by the output router, but nothing prompts the model to emit it.
>
> **Existing infrastructure:**
> - `thought_seed.py` — save, review, refine, link seeds (complete)
> - Output router — parses `THOUGHT_SEED:` directive (complete)
> - `knowledge/seeds/` directory — **does not exist yet** (must create + volume mount)
>
> **Required changes:**
> 1. Create `knowledge/seeds/` directory (add to Dockerfile or entrypoint)
> 2. Add `THOUGHT_SEED:` directive instructions to the Observer's system prompt
>    so it generates seeds when it notices interesting patterns, knowledge gaps,
>    or topics worth reflecting on outside the current session
> 3. Ensure the `knowledge/seeds/` path is volume-mounted in docker-compose
>    (already covered by the `/knowledge:rw` mount)

### Phase 2 Success Criteria

- [ ] Task scheduler executes tasks in priority order
- [ ] Thought seed hydration runs during sleep (seeds reviewed via CPU Lite)
- [ ] `knowledge/seeds/` directory exists and is writable
- [ ] Observer generates THOUGHT_SEED directives during normal conversation
- [ ] Conversation sanitizer checks recent messages and delegates curation to existing curator
- [ ] Initiative engine (GIL) processes topics from topic_cache.json
- [ ] Vector reflection updates stale embeddings
- [ ] Interruptible tasks stop on wake signal
- [ ] Non-interruptible tasks complete before wake
- [ ] Task metrics tracked (run count, duration, success rate)
- [ ] Unit tests pass
- [ ] Integration tests pass

---

## Phase 3: Advanced Features (DEPENDS ON PHASE 1 & 2)

**Goal:** Add learning, optimization, and experimental features  
**Duration:** 3-4 weeks  
**Deliverable:** GAIA autonomously learns and optimizes itself  

### 3.1 QLoRA Training During Sleep

**File:** `candidates/gaia-core/gaia_core/cognition/sleep_learning.py` (NEW)

```python
"""
QLoRA training tasks for sleep cycle.

When GAIA learns notable patterns or receives user feedback,
sleep time is used to fine-tune adapter weights.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger("GAIA.SleepLearning")

class SleepLearningManager:
    """
    Manages autonomous learning during sleep.
    """
    
    def __init__(self, config):
        self.config = config
        self.training_queue = []
        
        logger.info("🎓 SleepLearningManager initialized")
    
    async def train_session_adapter(self):
        """
        Train a session-level LoRA adapter on recent interactions.
        
        This is a Tier 3 adapter (ephemeral, session-scoped).
        """
        logger.info("📚 Starting session adapter training...")
        
        # Check if there's training data queued
        if not self.training_queue:
            logger.info("No training data queued, skipping")
            return
        
        # Prepare training data
        training_data = self._prepare_training_data()
        
        # Train adapter (estimated 15-20 minutes)
        # This is a MUST-COMPLETE task (non-interruptible)
        adapter_name = await self._train_qlora_adapter(training_data)
        
        logger.info(f"✅ Session adapter trained: {adapter_name}")
    
    def _prepare_training_data(self) -> Dict[str, Any]:
        """
        Compile training examples from recent interactions.
        """
        # TODO: Extract high-quality examples from session history
        return {}
    
    async def _train_qlora_adapter(self, data: Dict) -> str:
        """
        Execute QLoRA training.
        
        Returns: Adapter name
        """
        # TODO: Integrate with gaia-study training pipeline
        return "session_adapter_20260214"
```

### 3.2 KV Cache Strategy: prime.md IS the Solution

> **Design decision:** vLLM KV cache cannot be serialized/restored across container
> restarts. The KV cache is tied to the CUDA context; the tensor layouts can't be
> deserialized into a new process. We evaluated this and determined:
>
> - vLLM native sleep mode (`POST /sleep?level=1`) only frees KV cache (~1.7GB),
>   not model weights, and doesn't work on Blackwell (sm_120) with `--enforce-eager`
> - `torch.save` of KV state requires custom vLLM surgery and restore time would
>   negate any benefit vs. re-computing from prefix cache
>
> **The prime.md checkpoint IS the KV cache replacement.** It compresses the model's
> working context into natural language. On wake:
> 1. `--enable-prefix-caching` rebuilds the system prompt prefix quickly (~first request)
> 2. prime.md content is injected at Tier 1 in prompt_builder, restoring conversational context
> 3. The CPU Lite model (llama.cpp) retains its KV cache in system RAM across sleep cycles
>    since it's always running — no special handling needed
>
> **No kv_cache_manager.py file is needed.** This section is documentation only.

### 3.3 Dream Mode (Experimental)

**File:** `candidates/gaia-core/gaia_core/cognition/dream_mode.py` (NEW)

```python
"""
Dream mode: Self-directed reflection and insight generation.

During deep sleep, GAIA reflects on recent experiences and generates
insights, connections, and thought seeds without external prompting.
"""

import logging

logger = logging.getLogger("GAIA.DreamMode")

class DreamMode:
    """
    Experimental self-reflection during sleep.
    """
    
    def __init__(self, config):
        self.config = config
        self.llm = None  # Injected
        
        logger.info("💭 DreamMode initialized (experimental)")
    
    async def dream_cycle(self):
        """
        Generate insights through self-directed reflection.
        
        Process:
        1. Sample recent conversations
        2. Generate reflection prompts
        3. Use Prime to reflect
        4. Save insights as thought seeds
        5. Identify learning opportunities
        """
        logger.info("💭 Starting dream cycle...")
        
        # Get recent session history
        recent_sessions = self._get_recent_sessions(days=7)
        
        # Generate reflection prompts
        prompts = self._generate_reflection_prompts(recent_sessions)
        
        # Reflect on each prompt
        insights = []
        for prompt in prompts:
            insight = await self._reflect(prompt)
            insights.append(insight)
        
        # Save as thought seeds
        for insight in insights:
            self._save_as_thought_seed(insight)
        
        logger.info(f"✅ Dream cycle complete: {len(insights)} insights generated")
    
    def _get_recent_sessions(self, days: int):
        """Get session summaries from recent days."""
        # TODO: Query session manager
        return []
    
    def _generate_reflection_prompts(self, sessions):
        """Generate prompts for self-reflection."""
        prompts = [
            "What patterns have emerged in recent conversations?",
            "What knowledge gaps have I encountered?",
            "What could I improve about my responses?",
            "What surprising connections can I make between topics?",
        ]
        return prompts
    
    async def _reflect(self, prompt: str) -> str:
        """Use Prime to reflect on prompt."""
        # TODO: Call LLM with reflection prompt
        return ""
    
    def _save_as_thought_seed(self, insight: str):
        """Save insight as thought seed for later action."""
        from gaia_core.cognition.thought_seed import save_thought_seed
        # TODO: Create packet and save seed
        pass
```

### Phase 3 Testing Strategy

Testing focuses on non-disruption and quality:

```python
# File: candidates/gaia-core/tests/test_sleep_learning.py

@pytest.mark.slow
@pytest.mark.asyncio
async def test_qlora_training_completes():
    """Test that QLoRA training finishes without crashing."""
    manager = SleepLearningManager(config)

    # Queue training data
    manager.training_queue.append(test_data)

    # Execute (this takes 15-20 minutes in real scenario)
    await manager.train_session_adapter()

    # Verify adapter was created
    assert adapter_exists("session_adapter_test")

@pytest.mark.experimental
async def test_dream_mode_generates_insights():
    """Test dream mode self-reflection."""
    dream = DreamMode(config)

    await dream.dream_cycle()

    # Should have generated thought seeds
    seeds = list_unreviewed_seeds()
    assert len(seeds) > 0
```

### Phase 3 Success Criteria

- [ ] QLoRA training executes without crashes during sleep (GPU handoff works)
- [ ] Trained adapters improve response quality on held-out validation set
- [ ] Dream mode generates thought seeds that pass review
- [ ] All advanced features degrade gracefully if unavailable
- [ ] System remains stable with all features enabled
- [ ] Wake signal preempts GPU-heavy tasks with checkpoint/resume
- [ ] User feedback confirms autonomous improvements are valuable

---

## Integration Points

### With Existing Systems

**SessionManager**
- Sleep triggers archival when appropriate
- Wake restores session history alongside checkpoint

**VectorIndexer**
- Vector reflection task updates stale embeddings
- Thought seeds can trigger new embeddings

**ThoughtSeed System**
- Hydration runs during sleep with auto-action enabled
- Dream mode generates new seeds

**InitiativeLoop**
- Enhanced to orchestrate sleep tasks
- Becomes the "brain" of autonomous operation

**Discord Bot**
- Status updates reflect sleep/wake state
- Message queueing during sleep

**gaia-study**
- QLoRA training during deep sleep
- Coordinates GPU handoff via orchestrator

### Configuration

> Settings follow the existing pattern: defaults in `gaia_constants.json`,
> overridable via environment variables, accessed through the `Config` dataclass.
> No separate `.env` file needed.

**File:** `gaia-common/gaia_common/constants/gaia_constants.json` (MODIFY — add sleep section)

```json
{
  "SLEEP_CYCLE": {
    "enabled": true,
    "idle_threshold_minutes": 5,
    "enable_qlora_training": false,
    "enable_dream_mode": false,
    "task_timeout_seconds": 600
  }
}
```

**File:** `candidates/gaia-core/gaia_core/config.py` (MODIFY — add fields to Config dataclass)

```python
# Add to existing Config dataclass:
SLEEP_ENABLED: bool = True
SLEEP_IDLE_THRESHOLD_MINUTES: int = 5
SLEEP_ENABLE_QLORA: bool = False
SLEEP_ENABLE_DREAM: bool = False
SLEEP_TASK_TIMEOUT: int = 600
```

---

## Rollout Strategy

### Development Environment

1. **Phase 1** on candidate containers first
2. Test with Discord integration
3. Verify checkpoint persistence
4. Monitor for 1 week

### Staging Environment

1. Promote to live containers
2. Enable for single test user
3. Monitor sleep/wake cycles
4. Collect metrics

### Production

1. Enable for all users
2. Monitor system health
3. Tune idle threshold based on usage
4. Gradually enable Phase 2/3 features

### Feature Flags

> Stored in `gaia_constants.json` under `SLEEP_CYCLE` section, accessed via Config dataclass.

```json
{
  "SLEEP_CYCLE": {
    "enabled": true,
    "idle_threshold_minutes": 5,
    "enable_qlora_training": false,
    "enable_dream_mode": false,
    "task_timeout_seconds": 600
  }
}
```

---

## Monitoring & Observability

### Metrics to Track

```python
# File: candidates/gaia-core/gaia_core/cognition/sleep_metrics.py

from dataclasses import dataclass
from datetime import datetime

@dataclass
class SleepCycleMetrics:
    """Metrics for sleep cycle performance."""
    
    # State transition metrics
    sleep_count: int = 0
    wake_count: int = 0
    average_sleep_duration_seconds: float = 0.0
    average_wake_latency_seconds: float = 0.0
    
    # Checkpoint metrics
    checkpoints_created: int = 0
    checkpoints_loaded: int = 0
    checkpoint_failures: int = 0
    average_checkpoint_size_kb: float = 0.0
    
    # Task metrics
    tasks_executed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_interrupted: int = 0
    sleep_time_utilization_percent: float = 0.0
    
    # Quality metrics
    context_loss_incidents: int = 0
    message_loss_incidents: int = 0
    user_satisfaction_score: float = 0.0
    
    # Last updated
    last_updated: datetime = None
```

### Logging Strategy

```python
# Enhanced logging for sleep cycle events

logger.info("🌙 SLEEP_TRANSITION", extra={
    "event": "sleep_initiated",
    "idle_minutes": 6.5,
    "checkpoint_size_kb": 12.3,
    "timestamp": datetime.utcnow().isoformat()
})

logger.info("☀️ WAKE_TRANSITION", extra={
    "event": "wake_completed",
    "wake_latency_seconds": 3.2,
    "checkpoint_loaded": True,
    "messages_queued": 2,
    "timestamp": datetime.utcnow().isoformat()
})

logger.info("🔄 TASK_EXECUTION", extra={
    "event": "task_completed",
    "task_id": "thought_seed_hydration",
    "duration_seconds": 142,
    "success": True,
    "timestamp": datetime.utcnow().isoformat()
})
```

### Dashboard

Create monitoring dashboard showing:
- Current state (ACTIVE/ASLEEP/DREAMING/DISTRACTED/OFFLINE)
- Sleep cycle timeline
- Task execution history
- Checkpoint health
- Message queue status
- Performance metrics

---

## File Modification Checklist

### Phase 1

**New Files:**
- [ ] `candidates/gaia-core/gaia_core/cognition/sleep_wake_manager.py`
- [ ] `candidates/gaia-core/gaia_core/cognition/prime_checkpoint.py`
- [ ] `candidates/gaia-core/gaia_core/cognition/sleep_cycle_loop.py`
- [ ] `candidates/gaia-core/gaia_core/api/sleep_endpoints.py`
- [ ] `candidates/gaia-web/gaia_web/queue/message_queue.py`
- [ ] `candidates/gaia-core/tests/test_sleep_wake_manager.py`
- [ ] `candidates/gaia-core/tests/test_prime_checkpoint.py`
- [ ] `candidates/gaia-core/tests/integration/test_sleep_cycle.py`

**Modified Files:**
- [ ] `candidates/gaia-core/gaia_core/main.py` (register sleep router + start sleep loop in lifespan)
- [ ] `candidates/gaia-core/gaia_core/config.py` (add SLEEP_* fields to Config dataclass)
- [ ] `candidates/gaia-core/gaia_core/utils/prompt_builder.py` (inject sleep restoration context at Tier 1)
- [ ] `candidates/gaia-web/gaia_web/api/routes.py` (add queue endpoints)
- [ ] `candidates/gaia-common/gaia_common/utils/background/processor.py` (fix broken import on line 14)
- [ ] `gaia-common/gaia_common/constants/gaia_constants.json` (add SLEEP_CYCLE section)

**No changes needed:**
- Discord bot status — uses existing `discord_connector.update_presence()` / `set_idle()`
- No new `.env` file — config via gaia_constants.json

### Phase 2

**New Files:**
- [ ] `candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py`
- [ ] `candidates/gaia-core/gaia_core/cognition/conversation_sanitizer.py`
- [ ] `candidates/gaia-core/gaia_core/cognition/initiative_engine.py` (ported from archived run_gil.py)
- [ ] `candidates/gaia-core/tests/test_sleep_task_scheduler.py`
- [ ] `candidates/gaia-core/tests/integration/test_sleep_tasks.py`
- [ ] `knowledge/seeds/` (directory — create for thought seed storage)
- [ ] `knowledge/system_reference/topic_cache.json` (seed with empty topic list)

**Modified Files:**
- [ ] `candidates/gaia-core/gaia_core/cognition/sleep_cycle_loop.py` (add task execution in ASLEEP state)
- [ ] Observer system prompt (add THOUGHT_SEED: directive instructions)

### Phase 3

**New Files:**
- [ ] `candidates/gaia-core/gaia_core/cognition/sleep_learning.py`
- [ ] `candidates/gaia-core/gaia_core/cognition/dream_mode.py`
- [ ] `candidates/gaia-core/gaia_core/cognition/sleep_metrics.py`
- [ ] `candidates/gaia-core/tests/test_sleep_learning.py`

**Removed from plan (not needed):**
- ~~`kv_cache_manager.py`~~ — prime.md is the KV cache solution (see Section 3.2)

---

## Dependencies

### Python Packages

No new dependencies required:
- `asyncio` — stdlib, already available
- `httpx` — already in gaia-common dependencies (used for health checks)
- `threading` — stdlib
- All other imports are internal gaia packages

### System Requirements

- Disk space for checkpoints: ~100MB (shared volume, already provisioned)
- Docker named volume `gaia-candidate-shared` / `gaia-shared` (already exists)
- Discord bot permissions for message editing (bot can only edit its own messages)
- Network connectivity between gaia-web and gaia-core (already in Docker network)

---

## Risk Assessment & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Context loss during sleep | High | Low | Robust checkpoint system (prime.md), extensive testing |
| Message loss during transition | High | Medium | Persistent queue in gaia-web, transaction-like semantics |
| Task execution crashes | Medium | Medium | Graceful error handling, task isolation, try/except per task |
| Wake latency too high | Medium | Low | CPU Lite responds immediately; Prime boots in parallel (~37s) |
| Prime unavailable on wake | Medium | Low | CPU Lite fallback; prefix caching speeds Prime restoration |
| Sleep never triggers (always active) | Low | Low | Configurable idle threshold, manual override endpoint |
| QLoRA training crashes GPU | High | Low | Monitor VRAM, timeout mechanisms, separate gaia-study container |
| Circular dependency (gaia-common ↔ core) | High | Eliminated | Sleep logic lives in gaia-core; gaia-common only provides primitives |

---

## Approval & Next Steps

**Approval Required From:** Azrael (System Architect)

**Upon Approval:**
1. Begin Phase 1 implementation in candidates (standard candidate-only development)
2. Fix broken import in `gaia-common/processor.py` first (quick win, unblocks startup)
3. Implement files in order listed in checklist
4. Write unit tests alongside implementation
5. Integration testing after all Phase 1 files complete
6. Promote via standard pipeline (`promote_pipeline.sh`)
7. Document learnings in dev journal
8. Request Phase 2 approval after Phase 1 stable for 1 week

**Estimated Start Date:** Upon approval
**Phase 1 Target Completion:** 2-3 weeks from start
**Full System Target:** 10 weeks from start

---

## 7-State Refactor: New States & Protocols

### DREAMING State & Study Handoff Protocol

**Purpose:** Allow the GPU to be temporarily handed to the Study model for QLoRA training while GAIA is asleep. During DREAMING, GAIA cannot process messages via the LLM — a canned response is returned instead.

**Trigger:** The orchestrator's `handoff_manager.py` calls `POST /sleep/study-handoff` with `{"direction": "prime_to_study", "handoff_id": "<uuid>"}` after GPU reservation succeeds.

**State flow:**
```
ASLEEP → enter_dreaming(handoff_id) → DREAMING
DREAMING → exit_dreaming(handoff_id) → ASLEEP
```

**Behavior during DREAMING:**
- `get_canned_response()` returns `CANNED_DREAMING` ("I'm deep in study right now — absorbing new patterns. I'll be back soon!")
- Discord presence: `dnd`, activity `"studying..."`
- Sleep tasks are paused (GPU unavailable)
- If a wake signal arrives, it is queued but GAIA does not wake until DREAMING ends
- `handoff_id` must match on exit to prevent mismatched handoff

**Files:** `sleep_wake_manager.py` (`enter_dreaming()`, `exit_dreaming()`), `sleep_endpoints.py` (`/study-handoff`), `handoff_manager.py` (`_notify_study_handoff()`)

### DISTRACTED State & Resource Monitoring

**Purpose:** Detect when external processes (games, compilation, rendering) are consuming significant GPU/CPU and avoid generating slow or degraded responses.

**Detection:** `ResourceMonitor` (singleton, `resource_monitor.py`) polls every 5 seconds:
- **GPU:** `pynvml` — `nvmlDeviceGetUtilizationRates()` for GPU utilization %
- **CPU:** `psutil` — `psutil.cpu_percent(interval=None)` for CPU utilization %
- **Threshold:** If either metric exceeds **25%** sustained for **5 consecutive seconds**, `_distracted` flag is set

**State flow:**
```
ANY_STATE → (sustained load detected by monitor) → DISTRACTED
DISTRACTED → (check_and_clear_distracted: 3 samples over 3s all below 25%) → previous state
```

**Behavior during DISTRACTED:**
- `get_canned_response()` returns `CANNED_DISTRACTED` ("I'm a bit occupied right now — the system is under heavy load. I'll respond properly once things settle down!")
- Discord presence: `dnd`, activity `"occupied..."`
- gaia-web calls `GET /sleep/distracted-check` before every message forward — if canned_response is non-null, sends that instead of forwarding
- Re-check happens: (a) every 5 minutes via sleep cycle loop, or (b) on each incoming message via the distracted-check endpoint

**Clearing:** `check_and_clear_distracted()` takes 3 samples over 3 seconds. If all are below the 25% threshold, the distracted flag is cleared and the previous state resumes.

**Files:** `resource_monitor.py` (detection), `sleep_wake_manager.py` (state transitions), `discord_interface.py` (message gate), `sleep_endpoints.py` (`/distracted-check`)

### OFFLINE State & Graceful Shutdown

**Purpose:** Allow a clean shutdown sequence that properly transitions Discord presence and persists state before the process exits.

**Trigger:** `POST /sleep/shutdown` or `sleep_cycle_loop.initiate_shutdown()` called from the FastAPI shutdown handler.

**State flow:**
```
ANY_STATE → initiate_shutdown() → OFFLINE
OFFLINE → (process exits)
```

**Behavior during OFFLINE:**
- Discord presence: `invisible` (completely hidden), no activity text
- Sleep cycle loop breaks its main loop
- No further messages are processed
- State is persisted to checkpoint if currently ASLEEP

**Files:** `sleep_wake_manager.py` (`initiate_shutdown()`), `sleep_cycle_loop.py` (`initiate_shutdown()`), `sleep_endpoints.py` (`/shutdown`), `main.py` (FastAPI shutdown event)

### Canned Response System

**Purpose:** When GAIA cannot process messages through the LLM (DREAMING or DISTRACTED), pre-written responses are sent so users know GAIA is alive but temporarily unavailable.

**Constants** (in `sleep_wake_manager.py`):
```python
CANNED_DREAMING = (
    "I'm deep in study right now — absorbing new patterns. "
    "I'll be back soon!"
)
CANNED_DISTRACTED = (
    "I'm a bit occupied right now — the system is under heavy load. "
    "I'll respond properly once things settle down!"
)
```

**Flow:**
1. User sends message to Discord
2. `discord_interface.py` calls `GET /sleep/distracted-check` on gaia-core
3. gaia-core returns `{"state": "dreaming", "canned_response": "I'm deep in study..."}`
4. gaia-web sends the canned response directly to the Discord channel
5. Message is NOT forwarded to the LLM

**Method:** `SleepWakeManager.get_canned_response()` returns the appropriate string if the current state is DREAMING or DISTRACTED, otherwise returns `None`.

---

## Appendices

### A. State Transition Diagram (Detailed)

```
┌─────────────────────────────────────────────────────────────┐
│               GAIA Sleep Cycle State Machine                │
│                                                             │
│  6 states + 2 phases: ACTIVE → DROWSY → ASLEEP → DREAMING / DISTRACTED / OFFLINE  │
└─────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │                    ACTIVE                             │
    │  • Process messages immediately via Prime or Lite    │
    │  • Stream responses                                  │
    │  • Update session history                            │
    │  • IdleMonitor tracks last activity                  │
    │  • Discord: online, "watching over the studio"       │
    └────────────┬─────────────────────────────────────────┘
                 │
                 │ idle > 5min AND no active stream
                 ▼
    ┌──────────────────────────────────────────────────────┐
    │                    DROWSY                             │
    │  • Discord: "Drifting off..."                        │
    │  1. Wait for any streaming to complete               │
    │  2. Send meta-cognitive prompt to Prime:             │
    │     "Write your current state for later review"      │
    │  3. Prime generates cognitive summary                │
    │  4. Write to /shared/sleep_state/prime.md            │
    │  5. Rotate: prime.md → prime_previous.md             │
    │  6. Archive to prime_history/TIMESTAMP.md            │
    │                                                      │
    │  ⚡ CANCELLABLE: If message arrives during DROWSY,   │
    │     abort checkpoint and return to ACTIVE immediately  │
    └────────────┬─────────────────────────────────────────┘
                 │
                 │ checkpoint complete
                 ▼
    ┌──────────────────────────────────────────────────────┐
    │                   SLEEPING                            │
    │  • Discord: "Sleeping"                               │
    │  • Execute tasks in priority order:                  │
    │    P1: Session sanitization (interruptible)          │
    │    P1: Conversation curation (interruptible)         │
    │    P1: Thought seed hydration (interruptible)        │
    │    P2: Vector reflection (interruptible)             │
    │    P2: GIL topic processing (interruptible)          │
    │    P3: Blueprint verification (interruptible)        │
    │    P4: QLoRA training (NON-interruptible, Phase 3)   │
    │    P5: Dream mode (interruptible, Phase 3)           │
    │  • Poll for wake signal between tasks                │
    └────────────┬─────────────────────────────────────────┘
                 │
                 │ message queued → wake signal received
                 ▼
    ┌──────────────────────────────────────────────────────┐
    │            CHECK CURRENT TASK                        │
    │  If no task or interruptible: → WAKING               │
    │  If non-interruptible: → FINISHING_TASK              │
    └────────────┬─────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
┌──────────┐          ┌──────────────┐
│  WAKING  │          │ FINISHING_   │
│          │          │ TASK         │
│          │          │              │
│          │          │ (e.g. QLoRA  │
│          │          │  checkpoint) │
└────┬─────┘          └─────┬────────┘
     │                      │
     │                      │ task complete
     │                      ▼
     │                ┌──────────┐
     │                │  WAKING  │
     │                └────┬─────┘
     │                     │
     └─────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────────────┐
    │               WAKING (Parallel Strategy)             │
    │  • Discord: "Waking up..."                           │
    │                                                      │
    │  Track A — Immediate (~2s):                          │
    │    1. Load prime.md checkpoint                        │
    │    2. Format as REVIEW context (not prompt)          │
    │    3. Route first queued message to CPU Lite          │
    │    4. CPU Lite responds with checkpoint as context   │
    │                                                      │
    │  Track B — Background (~37-60s):                     │
    │    1. Start gaia-prime container from tmpfs           │
    │    2. Wait for /health endpoint                      │
    │    3. Prefix caching rebuilds system prompt           │
    │    4. Mark Prime as available                        │
    └────────────┬─────────────────────────────────────────┘
                 │
                 │ Prime ready + first message handled
                 ▼
    ┌──────────────────────────────────────────────────────┐
    │                    ACTIVE                             │
    │  • Process remaining queued messages via Prime       │
    │  • prime.md REVIEW context injected at Tier 1        │
    │  • Resume normal operation                           │
    │  • Discord: online, "watching over the studio"       │
    └──────────────────────────────────────────────────────┘
```

### B. Checkpoint File Format

```markdown
# Prime Cognitive State Checkpoint
**Last Updated:** 2026-02-14T03:15:42.123456Z
**Session ID:** discord_channel_12345
**State:** SLEEP_INITIATED
**Persona:** default

## Active Context Summary
We were discussing the GAIA sleep cycle implementation with Azrael.
Key discussion points included:
- State machine architecture (ACTIVE/ASLEEP/DREAMING/DISTRACTED/OFFLINE)
- Prime checkpoint strategy (prime.md file)
- KV cache persistence challenge
- Integration with existing thought seed system

## Conversation State
**Last user message:** "Can this be written as a single major plan?"
**My last response:** [Comprehensive analysis of implementation approach]
**Response status:** Complete
**Pending actions:** None

## Key Entities Referenced
- Azrael (user, system architect, project lead)
- gaia-prime (vLLM inference server, 16GB VRAM)
- gaia-core (cognitive engine, processes packets)
- gaia-web (REST API, Discord bot interface)
- tmpfs (/tmp/model-weights/, 16GB RAM-backed storage)
- Thought seed system (autonomous learning)

## Reasoning State
**Current hypothesis:** Single comprehensive plan with 3 phases is optimal
**Confidence:** High (0.90)
**Why:** Aligns with existing project patterns (orchestrator, v0.3)
**Supporting evidence:**
- Strong dependencies between phases
- Shared codebase modifications
- Unified testing strategy
- Precedent in existing plans

**Alternative considered:** Split into 3 separate documents
**Rejected because:** Would create synchronization issues

## Tone & Relationship Context
**Mode:** Technical collaboration with system architect
**User preferences:**
- Detailed explanations with code examples
- Practical, implementable solutions
- Architecture diagrams (ASCII format)
- Reference to existing patterns
**Conversation style:** Iterative design, building on previous sessions
**User expertise level:** Advanced (system architect, knows codebase)

## Important Facts Cached
**System configuration:**
- RTX 5080 GPU with 16GB VRAM
- 32GB system RAM total (16GB allocated to tmpfs)
- Ryzen 9 processor (high-performance CPU)
- Arch Linux host OS
- Docker-based containerization

**Project context:**
- Running in candidate containers during development
- Live/candidate separation pattern
- v0.3 architecture planning in progress
- Existing initiative loop foundation
- Thought seed system operational

**Technical constraints:**
- GPU memory must be carefully managed
- KV cache lives in VRAM (lost on sleep without serialization)
- vLLM may not support KV cache serialization
- CPU Lite model always loaded (persistent KV cache)

## Next Expected Actions
**If woken immediately:**
- Process queued messages with this context available
- Likely: Continue discussing implementation plan
- Maintain technical detail level established in conversation

**If continuing sleep:**
- Proceed with thought seed hydration task
- Then conversation sanitation
- Then vector reflection

**On extended sleep (>1 hour):**
- Consider QLoRA training session (Phase 3)
- Dream mode reflection (experimental)

## Meta-Notes
This checkpoint was generated automatically during sleep transition.
Upon wake, review this content to restore working memory context.
This is NOT a prompt - it is your own notes to yourself.

Checkpoint version: 1.0
Generated by: SleepWakeManager.initiate_drowsy()
Storage: /shared/sleep_state/prime.md
Estimated restoration time: <2 seconds (injected at Tier 1 in prompt_builder)
```

### C. Message Queue Schema

```python
@dataclass
class QueuedMessage:
    """Schema for queued messages during sleep."""
    
    # Identification
    message_id: str          # Unique ID (UUID)
    session_id: str          # Session this belongs to
    
    # Content
    content: str             # The actual message
    source: str              # "discord", "web", "cli"
    metadata: Dict[str, Any]  # Source-specific data
    
    # Priority
    priority: int            # 0 = normal, 1 = high, 2 = urgent
    
    # Timestamps
    queued_at: datetime      # When message entered queue
    processed_at: Optional[datetime] = None
    
    # Context
    user_id: str             # Who sent it
    channel_id: str          # Where it came from
    requires_prime: bool     # Needs GPU model or CPU ok?
    
    # State
    attempts: int = 0        # Processing attempts
    max_attempts: int = 3    # Give up after this many
    error: Optional[str] = None
```

---

**END OF IMPLEMENTATION PLAN**
