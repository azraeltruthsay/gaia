# Lite Model EXECUTE Echo Fix + Model Promotion + Self-Observation — 2026-02-24

## Summary

Diagnosed and fixed a multi-layered failure where the Lite model (Qwen 3B) re-emitted `EXECUTE:` directives for already-completed tool calls, causing GAIA's safety gate to deny the response and discard Prime's analysis. The fix progressed from a reactive patch to proactive input shaping, then to a deeper architectural improvement: post-reflection model promotion and rule-based self-observation.

## Problem

User asked GAIA to promote gaia-audio. The tool routing pipeline correctly executed `assess_promotion`, and Prime produced excellent analysis during the reflection phase. But Lite — selected as the final generation model because Prime was "sleeping" — just output `EXECUTE: assess_promotion {"service": "gaia-audio"}` instead of synthesizing the results. The safety gate denied this (not in `SAFE_SIDECAR_TOOLS`), and the user received a generic denial message.

**Root causes identified:**
1. `_parse_llm_output_into_packet()` blindly converted all `EXECUTE:` directives to sidecar actions with no awareness of already-executed tools
2. The system prompt included `EXECUTE:` syntax examples that Lite pattern-matched and echoed
3. No assistant prefill to steer Lite toward prose generation
4. Model selection was a one-shot decision — even when Prime got loaded for reflection, Lite still handled final generation
5. When Lite was solo, the Observer was disabled entirely because it required an LLM

## What Changed

### Layer 1: Reactive Fix (output_router.py)
- **Duplicate EXECUTE detection**: When parsing `EXECUTE:` directives, check if `packet.tool_routing.execution_status == EXECUTED` and the action matches the already-executed tool. Skip it instead of creating a sidecar action.
- **Reflection fallback**: When the candidate response is empty after stripping a duplicate EXECUTE (no sidecar actions either), recover the response from the `refined_plan` entry in the reflection log.

### Layer 2: Input Shaping (prompt_builder.py)
- **TOOL CALLING CONVENTION suppression**: When `tool_routing.execution_status == EXECUTED`, the entire TOOL CALLING CONVENTION block (with `EXECUTE:` format, examples, and instructions) is omitted from the system prompt. Removes the syntactic pattern Lite was echoing.
- **Output scaffolding**: Inject an assistant prefill message (`"Based on the results,"`) when tool results are present and the tool was already executed. Steers generation toward prose synthesis.

### Layer 3: ChatML Assistant Prefill (hf_prompting.py)
- Modified `_build_chatml()` to detect when the last message has role "assistant" and treat it as a generation prefix — content placed after `<|im_start|>assistant\n` without a closing `<|im_end|>`. This is the standard "assistant prefill" pattern for LLM inference, and it ensures the model continues from the provided text rather than treating it as a completed prior turn.

### Layer 4: Post-Reflection Model Promotion (agent_core.py)
- After reflection completes, if a Prime-class model was borrowed for reflection and Lite is the current responder, **promote Prime to final generation** instead of releasing it back to idle. This prevents the "Prime thinks, Lite speaks" relay that was corrupting output.
- The `finally` block now checks `reflection_model_name in PRIME_MODELS and selected_model_name not in PRIME_MODELS`. If so, it releases Lite to idle, swaps `selected_model` and `selected_model_name` to the reflection model, and keeps it busy.
- Downstream effects: Council notes injection activates correctly, ExternalVoice uses Prime, Observer picks Lite (now idle) as its model, and post-response escalation doesn't fire needlessly.

### Layer 5: Rule-Based Self-Observation (stream_observer.py, agent_core.py)
- `StreamObserver` now accepts `llm=None` (removed the `ValueError`). Rule-based checks — `fast_check()`, `_validate_code_paths()`, `_verify_citations_against_rag()`, and identity keyword heuristics — all work without an LLM.
- Added a safety guard before the LLM call path: if `self.llm is None` and `use_llm=True`, return OK with a warning instead of crashing.
- Observer creation in `agent_core.py` no longer requires `observer_model is not None`. When Lite operates solo, it still gets rule-based observation.

## Files Modified

| File | Change |
|------|--------|
| `candidates/gaia-core/.../output_router.py` | Duplicate EXECUTE detection, reflection fallback, ToolExecutionStatus import |
| `candidates/gaia-core/.../prompt_builder.py` | TOOL CALLING CONVENTION suppression, assistant prefill injection, ToolExecutionStatus import |
| `candidates/gaia-common/.../hf_prompting.py` | ChatML assistant prefill support in `_build_chatml()` |
| `candidates/gaia-core/.../agent_core.py` | Post-reflection model promotion, rule-based observer creation |
| `candidates/gaia-core/.../stream_observer.py` | Allow `llm=None`, LLM-unavailable safety guard |

All mirrored to production paths. Both `gaia-core` and `gaia-core-candidate` rebuilt/restarted and verified healthy.

## Testing

- 4-scenario unit tests for duplicate EXECUTE detection (duplicate skipped, different tool allowed, no-routing normal path, reflection fallback recovery)
- 3-scenario ChatML prefill tests (prefill renders correctly, normal path unaffected, empty content degrades gracefully)
- 4-scenario prompt_builder integration tests (EXECUTE syntax suppressed, tool results present, assistant prefix injected, instruction preserved)
- 4-scenario StreamObserver tests (llm=None creation, rule-based observe(), fast_check, LLM-unavailable guard)
- Full test suites: gaia-core 56/57 passed (1 pre-existing), gaia-common 315/315 passed

## Architectural Insight

The core lesson: a 3B model doesn't *read* instructions — it *pattern-matches* against its training distribution. When it sees tool-shaped context, it produces tool-shaped output regardless of meta-instructions. The fix philosophy at every layer was **shape the corridor, don't trust the comprehension**: remove the patterns it could echo, pre-fill where it should start, and when a smarter model is already loaded, just let it speak directly.

## Commit

`37fddb0` — `fix: Lite model EXECUTE echo — input shaping, scaffolding, model promotion, rule-based self-observation`
