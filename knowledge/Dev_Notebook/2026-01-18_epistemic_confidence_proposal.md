# GAIA Development Journal

## Date: 2026-01-18

### Subject: Epistemic Confidence Assessment - "Do I Actually Know This?"

**Context:**

Testing the sketchpad-based fragmentation with "Recite The Raven" revealed a deeper issue: GAIA attempts the task with full confidence despite not having the content memorized accurately. The result is hallucinated, repetitive garbage.

A human in this situation would:
1. Feel uncertainty while attempting the recitation
2. Recognize they're probably wrong
3. Stop due to social embarrassment / epistemic humility
4. Offer an alternative: "I don't have it memorized, but I can summarize it..."

GAIA has no such "embarrassment signal" - she plows ahead confidently producing nonsense.

**Proposal: Pre-Task Confidence Check**

Before attempting certain task types (especially `recitation`), GAIA should perform a self-assessment:

```python
def assess_task_confidence(self, intent: str, user_input: str, model_name: str = "lite") -> Dict[str, Any]:
    """
    Ask the model to assess its own confidence in completing a task BEFORE attempting it.

    Returns:
        confidence_score: 0.0-1.0
        can_attempt: bool - should we even try?
        reasoning: str - why this confidence level
        alternative_offer: str - what to offer if confidence is low
    """

    assessment_prompt = f"""You are about to attempt a task. Before starting, honestly assess your capabilities.

TASK TYPE: {intent}
USER REQUEST: {user_input}

Please honestly evaluate:
1. Do you have this content accurately memorized (if recitation)?
2. What's your confidence you can complete this accurately (0.0-1.0)?
3. If confidence is below 0.7, what alternative could you offer?

Respond in this format:
CONFIDENCE: [0.0-1.0]
CAN_ATTEMPT: [yes/no]
REASONING: [why this confidence level - be honest about limitations]
ALTERNATIVE: [what to offer if you shouldn't attempt this directly]

Be brutally honest. It's better to admit uncertainty than produce inaccurate content."""
```

**Integration Points:**

1. **In `_run_with_fragmentation`**: Before generating the first fragment, call `assess_task_confidence`
2. **Confidence thresholds**:
   - `>= 0.8`: Proceed with generation
   - `0.5 - 0.8`: Warn user, offer alternative, proceed if they confirm
   - `< 0.5`: Do NOT attempt. Offer alternative (summarize, retrieve from file, etc.)

3. **Alternative actions**:
   - "I don't have The Raven memorized accurately. Would you like me to retrieve it from a file, or provide a summary?"
   - "My confidence in verbatim recitation is low. I can offer the general themes and famous lines instead."

**The Key Question:**

Can the Heretic Claude model (or any local model) actually perform accurate self-assessment? This is the crux:

- If the model says "Confidence: 0.9" and then produces garbage, the self-assessment is useless
- If the model can honestly say "Confidence: 0.3 - I don't have this memorized accurately", that's genuine epistemic awareness

**Test Protocol:**

1. Implement `assess_task_confidence`
2. Run it on "Recite The Raven" and observe the confidence score
3. If confidence is low, we have a working epistemic check
4. If confidence is high but output is garbage, the model lacks self-awareness

**Why This Matters:**

This is a foundational capability for trustworthy AI:
- Knowing what you don't know
- Honest uncertainty quantification
- Offering alternatives when uncertain

If we can get this working for recitation, it generalizes to:
- Code generation ("I'm not confident this is correct, should I add tests?")
- Factual claims ("My training data may be outdated on this topic")
- Complex reasoning ("This involves multiple steps where I could make errors")

**Next Steps:**

1. ~~Implement `assess_task_confidence` as a quick prototype~~ DONE
2. Test with recitation intent
3. Evaluate whether the model's self-assessment correlates with actual performance
4. ~~If it works, integrate into the recitation/fragmentation flow with gating logic~~ DONE

---

## Implementation Complete: 2026-01-18

Added `assess_task_confidence()` method to `agent_core.py` (lines 1921-2027).

**Flow:**
1. Recitation intent detected
2. `assess_task_confidence()` called with lite model
3. Model asked to honestly evaluate:
   - Do I have this memorized accurately?
   - What's my confidence (0.0-1.0)?
   - What alternative can I offer?
4. If `confidence_score < 0.5` or `can_attempt == False`:
   - Return honest response explaining limitations
   - Offer alternative (summary, partial recitation, etc.)
5. If confidence adequate, proceed with fragmented generation

**Key prompt language:**
> "Be brutally honest. It is far better to admit 'I don't have this memorized accurately' than to produce incorrect content. Humans respect honesty about limitations."

**Test command:**
```bash
docker exec -it gaia-assistant python3 gaia_rescue.py
# Then: "Recite The Raven by Edgar Allan Poe"
```

Watch for:
- `Task confidence assessment: score=X.X, can_attempt=Y`
- `Confidence reasoning: ...`
- Either proceeds with generation OR returns honest decline with alternative

---

## GCP Integration: 2026-01-18

### Problem Discovered

Testing with "Recite the GAIA Constitution" revealed that the confidence assessment was running as a "naked" model call - no identity, no world state, no tool awareness. The model didn't know it WAS GAIA or that the Constitution existed in its knowledge base.

### Architectural Fix

Refactored `assess_task_confidence()` to use the full GCP pipeline:

1. **Creates a proper CognitionPacket** using `_create_initial_packet()`
   - Includes identity injection (Tier I/II/III)
   - Includes world state snapshot (directory structure, files)
   - Includes MCP tool discovery (file read, list_tree, etc.)

2. **Uses `build_from_packet()`** with task instruction key
   - Added `"confidence_assessment"` task instruction to `gaia_constants.json`
   - Prompt builder injects all context automatically

3. **Result**: Model now has full awareness when self-assessing:
   - Knows it is GAIA
   - Knows its knowledge base structure
   - Knows it has file read tools available
   - Can reason: "I don't have this memorized, but I can read it from knowledge/system_reference/core_documents/"

### Files Changed

- `app/cognition/agent_core.py`: Refactored `assess_task_confidence()` to use GCP
- `app/gaia_constants.json`: Added `confidence_assessment` task instruction

### Key Principle

**All cognition flows through the GCP.** Every model interaction must include:
- Identity context
- World state
- Tool awareness
- Proper packet structure

This prevents "naked" model calls that lack GAIA's self-awareness.
