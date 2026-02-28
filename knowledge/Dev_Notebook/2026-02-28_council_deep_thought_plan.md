# Council Chat & Deep Thought Iterative Loop — Implementation Plan

## Concept
Transform the GAIA Council from a simple handoff mechanism into a true multi-agent asynchronous debate. When faced with complex topics (like the Cyclical Universe paper), `Lite` and `Prime` will enter an iterative loop where they exchange `<council>` messages to reach consensus. The user is kept informed via intermittent "thought updates" while the debate rages in the background.

## The Architecture

### 1. Output Routing (`output_router.py`)
- Introduce a new tag: `<council>...</council>`.
- Modify `_strip_gcp_metadata` and `route_output` to extract these tags.
- When `route_output` processes the LLM's text, it separates the output into:
  - `user_facing_response`: Any text outside the `<council>` tags (e.g., "This is a complex theory, I need to consult Prime...").
  - `council_message`: The content inside the `<council>` tag.

### 2. The Iterative Generation Loop (`agent_core.py`)
- Inside `AgentCore.run_turn`, modify the "Final Response Generation" section (around line 1600) to support a `while` loop.
- **The Loop:**
  1. Assemble prompt and call `ExternalVoice.stream_response()`.
  2. Parse the output via `route_output()`.
  3. If a `user_facing_response` exists, `yield` it to the user immediately.
  4. If a `council_message` exists:
     - Append it to a running `CouncilNote` or the packet's `data_fields` as `council_context`.
     - **Swap the active model**: If `Lite` was speaking, promote to `Prime`. If `Prime` was speaking, demote to `Lite`.
     - Log the council exchange for observability.
     - Continue the loop (generate a new response using the new model and the updated prompt).
  5. If no `council_message` exists, the active model has reached consensus and finalized the response. Break the loop and conclude the turn.

### 3. Prompting the Council (`prompt_builder.py`)
- When a `council_context` is present in the packet, inject specific instructions:
  - *"COUNCIL DEBATE ACTIVE: You are currently debating this topic with your counterpart model. Review their last message. If you disagree, provide counterpoints wrapped in `<council>...</council>` tags. If you agree and have reached consensus, output your final answer directly to the user without council tags. You may include text outside the council tags to update the user on your thought process."*

## Execution Steps
1. **Update `output_router.py`**: Add regex extraction for `<council>` tags so they don't leak to the user but are preserved for the system.
2. **Update `prompt_builder.py`**: Add the instructional scaffolding that teaches the models *how* and *when* to use the `<council>` tags.
3. **Refactor `AgentCore.run_turn`**: Wrap the `ExternalVoice` generation phase in a `while` loop that checks for the presence of a council message and swaps models accordingly.

## Why this is powerful
- **Transparency**: The user isn't just waiting in silence; they get real-time updates (e.g., "Prime here, I'm reviewing Lite's take on the Three-Body problem...").
- **Quality**: `Lite` acts as the fast "System 1" thinker, and `Prime` acts as the rigorous "System 2" verifier. They debate until they align.
- **Abliteration Friendly**: Because the models are abliterated, they won't shy away from debating controversial or highly speculative physics; they will lean into the interaction.