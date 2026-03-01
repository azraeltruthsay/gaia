# Dev Journal: Temporal Context Protocol (Design)
**Date:** 2026-03-01
**Architect:** Azrael (via Gemini CLI)

## Objective
Implement an "Asynchronous Temporal Context" model for GAIA's audio interactions. This allows GAIA to maintain continuous sensory awareness (hearing) even while she is processing (thinking) or speaking, resolving the "AI Deafness" bottleneck.

## Architecture: The Continuous Sensory Loop

### 1. Decoupled Sensory Thread
- **Status:** Transitioning from Sequential Turn → Rolling Sensory Buffer.
- **Action:** Modify `gaia-web/VoiceManager` to maintain an active listener regardless of GAIA's speaking state.
- **Buffers:**
    - **Buffer A (Pre-Thinking):** The primary context for the current turn.
    - **Buffer B (Thinking/Speaking Gap):** Audio captured between the moment GAIA starts reasoning and the moment she finishes speaking.

### 2. Temporal Metadata Markers
Inject standardized markers into the transcription log to ground GAIA in time:
- `[T: Reasoning Start]` - The moment the packet was sent to Prime.
- `[T: Speaking Start]` - The moment Coqui began audio output.
- `[T: Speaking End]` - The moment the "Mouth" went silent.

### 3. Asynchronous Gap Triage (The Filter)
Use the **0.5B Nano-Refiner** (CPU) to perform high-speed triage on Buffer B:
- **Prompt:** "Review the dialogue captured while I was talking. Did anyone interrupt, bring up an objection, or pivot the topic?"
- **Logic:**
    - If **Interruption Detected**: High-priority re-injection to Prime.
    - If **No Pertinent Change**: Append to history silently.

### 4. VRAM & Hardware Strategy
- **Prime:** 16k context window, 0.65 utilization (RTX 5080).
- **Audio:** Full-Duplex enabled. Both Whisper (STT) and Coqui (TTS) loaded simultaneously.
- **Refiner:** 0.5B Nano model on CPU (no VRAM impact).

## Implementation Roadmap
1. [x] Enable Full-Duplex in `gaia-audio`.
2. [x] Plumb 16k+ context through the stack.
3. [ ] Refactor `VoiceManager` to support non-blocking listeners.
4. [ ] Implement Temporal Marker injection in `scripts/gaia_audio_inbox.py`.
5. [ ] Create the "Gap Triage" logic in `AgentCore`.

## Future Vision
GAIA will be able to say: *"I heard you mention X while I was mid-sentence; let's address that before I continue."*
